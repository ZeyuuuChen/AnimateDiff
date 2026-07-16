"""Minimal AnimateDiff adapter for the repository's image Quadtree kernels.

All spatial decisions are delegated to
``tomesd.methods.merge_quadtree.plan_quadtree``.  This module only preserves
the frame axis and batches the resulting image plans in AnimateDiff's spatial
attention order: [uncond frame 0..F-1, cond frame 0..F-1].
"""

from typing import Dict, List, Tuple, Callable, Optional

import torch
import torch.nn.functional as F

from tomesd.methods.merge_quadtree import plan_quadtree


def _normalize_per_frame(value: torch.Tensor) -> torch.Tensor:
    flat = value.float().flatten(1)
    lo = torch.quantile(flat, 0.02, dim=1, keepdim=True)
    hi = torch.quantile(flat, 0.98, dim=1, keepdim=True)
    return ((flat - lo) / (hi - lo).clamp_min(1e-6)).clamp_(0, 1).view_as(value)


def video_importance(
    heat_map: torch.Tensor,
    metric: Optional[torch.Tensor],
    *,
    temporal_lambda: float = 0.0,
    feature_lambda: float = 0.0,
    motion_lambda: float = 0.0,
) -> torch.Tensor:
    """Combine CFG saliency, neighboring-frame protection and feature detail.

    ``temporal_lambda`` protects prompt-salient regions in adjacent frames.
    ``motion_lambda`` is different: it protects tokens whose conditional hidden
    states change across frames, even when the CFG map is weak.  This is the
    knob that matters for VBench action/dynamic-degree prompts.
    """
    importance = _normalize_per_frame(heat_map)
    if temporal_lambda > 0 and importance.shape[0] > 1:
        previous = torch.cat([importance[:1], importance[:-1]], dim=0)
        following = torch.cat([importance[1:], importance[-1:]], dim=0)
        importance = torch.maximum(
            importance, temporal_lambda * torch.maximum(previous, following)
        )

    if (feature_lambda > 0 or motion_lambda > 0) and metric is not None:
        # metric is one conditional spatial-attention sequence per frame [F,N,C].
        frames, tokens, channels = metric.shape
        side_h, side_w = heat_map.shape[-2:]
        if tokens == side_h * side_w:
            unit = F.normalize(metric.float(), dim=-1).transpose(1, 2).reshape(
                frames, channels, side_h, side_w
            )
            if feature_lambda > 0:
                local_mean = F.avg_pool2d(unit, 3, stride=1, padding=1)
                local_mean = F.normalize(local_mean, dim=1)
                disagreement = 1.0 - (unit * local_mean).sum(1)
                importance = importance + feature_lambda * _normalize_per_frame(disagreement)
            if motion_lambda > 0 and frames > 1:
                previous = torch.cat([unit[:1], unit[:-1]], dim=0)
                following = torch.cat([unit[1:], unit[-1:]], dim=0)
                motion = torch.maximum(
                    (unit - previous).square().mean(1).sqrt(),
                    (unit - following).square().mean(1).sqrt(),
                )
                importance = importance + motion_lambda * _normalize_per_frame(motion)
    return importance


def plan_video_frames(
    heat_map: torch.Tensor,
    *,
    w: int,
    h: int,
    removed_tokens: int,
    device: torch.device,
    levels: List[int],
    offset_x: int,
    offset_y: int,
    pool: str = "max",
    budget_mode: str = "equal_quota",
    coarse_budget_fraction: Optional[float] = None,
) -> List[Dict]:
    """Run the unmodified image planner independently on every video frame."""
    if heat_map.ndim != 3:
        raise ValueError(f"expected CFG heat map [F,H,W], got {tuple(heat_map.shape)}")
    return [
        plan_quadtree(
            w=w,
            h=h,
            r=removed_tokens,
            B=1,
            device=device,
            error_mask={"heat_map": heat_map[f : f + 1]},
            levels=levels,
            offset_x=offset_x,
            offset_y=offset_y,
            pool=pool,
            budget_mode=budget_mode,
            coarse_budget_fraction=coarse_budget_fraction,
        )
        for f in range(heat_map.shape[0])
    ]


def merge_from_video_plans(
    plans: List[Dict],
    *,
    batch_size: int,
    weighted: bool = True,
    weight_power: float = 1.0,
) -> Tuple[Callable, Callable]:
    """Batch image plans without changing their grouping or merge equation."""
    if not plans or any(plan is None for plan in plans):
        raise ValueError("video Quadtree requires one non-empty plan per frame")
    frame_count = len(plans)
    if batch_size % frame_count:
        raise ValueError(f"attention batch {batch_size} is not divisible by F={frame_count}")
    if len({plan["K"] for plan in plans}) != 1:
        raise ValueError("all frame plans must keep the same number of tokens")

    copies = batch_size // frame_count
    n, k = plans[0]["N"], plans[0]["K"]
    group_f = torch.stack([plan["group_id_compact"] for plan in plans])
    reps_f = torch.stack([plan["rep_per_group"] for plan in plans])
    importance_f = torch.stack([plan["imp_per_token"] for plan in plans])
    group = group_f.repeat(copies, 1)
    reps = reps_f.repeat(copies, 1)
    importance = importance_f.repeat(copies, 1)

    if weighted:
        importance_eps = (importance + 1e-6).pow(weight_power)
        importance_sum = torch.zeros(
            batch_size, k, device=importance.device, dtype=importance_eps.dtype
        )
        importance_sum.scatter_add_(1, group, importance_eps)
        weights = importance_eps / importance_sum.gather(1, group)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        b, n_tokens, channels = x.shape
        if (b, n_tokens) != (batch_size, n):
            raise ValueError(f"plan expects {(batch_size, n)}, received {(b, n_tokens)}")
        if mode == "none":
            return torch.gather(x, 1, reps.unsqueeze(-1).expand(-1, -1, channels))
        group_index = group.unsqueeze(-1).expand(-1, -1, channels)
        out = torch.zeros(b, k, channels, device=x.device, dtype=x.dtype)
        if weighted and mode == "mean":
            return out.scatter_reduce(
                1, group_index, weights.to(x.dtype).unsqueeze(-1) * x,
                reduce="sum", include_self=False,
            )
        return out.scatter_reduce(1, group_index, x, reduce="mean", include_self=False)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, 1, group.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    return merge, unmerge


def similarity_merge_from_video_plans(
    plans: List[Dict], metric: torch.Tensor, rep_mode: str = "cfg",
    removed_tokens: Optional[int] = None,
    protect_multiplier: float = 1.4,
) -> Tuple[Callable, Callable]:
    """Use Quadtree only to allocate representatives, then similarity-match.

    Every plan keeps exactly K spatial representatives. All N-K remaining
    tokens merge to their most similar representative, as in ToMe. This avoids
    forcing every token in a geometric 4x4 leaf to share one update while
    preserving Quadtree's adaptive spatial destination density and exact K.
    """
    if not plans or any(plan is None for plan in plans):
        raise ValueError("similarity Quadtree requires one plan per frame")
    frames = len(plans)
    batch, n, _ = metric.shape
    if batch % frames:
        raise ValueError(f"attention batch {batch} is not divisible by F={frames}")
    if len({plan["K"] for plan in plans}) != 1:
        raise ValueError("all frame plans must share K")
    k = plans[0]["K"]
    copies = batch // frames
    all_idx = torch.arange(n, device=metric.device)

    reps_per_frame, src_per_frame = [], []
    conditional_metric = metric[(copies - 1) * frames : copies * frames].float()
    for frame_idx, plan in enumerate(plans):
        group = plan["group_id_compact"]
        importance = plan["imp_per_token"]
        if rep_mode == "cfg":
            rep_score = importance
        else:
            # Pick a real token closest to its leaf's conditional-feature
            # centroid. Mixed mode adds a small normalized CFG preference.
            unit_frame = F.normalize(conditional_metric[frame_idx], dim=-1)
            group_sum = torch.zeros(k, unit_frame.shape[-1], device=metric.device)
            group_sum.scatter_add_(0, group[:, None].expand_as(unit_frame), unit_frame)
            centroid = F.normalize(group_sum, dim=-1)
            rep_score = (unit_frame * centroid[group]).sum(-1)
            if rep_mode == "mixed":
                imp_norm = (importance - importance.min()) / (
                    importance.max() - importance.min()
                ).clamp_min(1e-6)
                rep_score = rep_score + 0.10 * imp_norm
        max_imp = torch.full((k,), -torch.inf, device=metric.device)
        max_imp.scatter_reduce_(0, group, rep_score, reduce="amax", include_self=True)
        candidates = torch.where(
            rep_score >= max_imp[group] - 1e-7, all_idx, n
        )
        reps = torch.full((k,), n, dtype=torch.long, device=metric.device)
        reps.scatter_reduce_(0, group, candidates, reduce="amin", include_self=True)
        is_rep = torch.zeros(n, dtype=torch.bool, device=metric.device)
        is_rep[reps] = True
        reps_per_frame.append(reps)
        src_per_frame.append(all_idx[~is_rep])

    reps_f = torch.stack(reps_per_frame)
    src_f = torch.stack(src_per_frame)
    reps = reps_f.repeat(copies, 1)
    src_idx = src_f.repeat(copies, 1)

    with torch.no_grad():
        unit = F.normalize(metric.float(), dim=-1)
        src_metric = torch.gather(
            unit, 1, src_idx.unsqueeze(-1).expand(-1, -1, unit.shape[-1])
        )
        dst_metric = torch.gather(
            unit, 1, reps.unsqueeze(-1).expand(-1, -1, unit.shape[-1])
        )
        # N=4096 in the target setting; chunk sources to bound peak memory.
        nearest, similarity = [], []
        for chunk in src_metric.split(512, dim=1):
            chunk_similarity = chunk @ dst_metric.transpose(1, 2)
            chunk_value, chunk_index = chunk_similarity.max(-1)
            nearest.append(chunk_index)
            similarity.append(chunk_value)
        dst_for_src = torch.cat(nearest, dim=1)
        similarity = torch.cat(similarity, dim=1)

        total_sources = src_idx.shape[1]
        removed = total_sources if removed_tokens is None else min(removed_tokens, total_sources)
        if removed < total_sources:
            # Match Importance's useful behavior: low-CFG tokens are always
            # eligible, while high-CFG/low-similarity tokens stay unmerged.
            importance_f = torch.stack([plan["imp_per_token"] for plan in plans])
            importance = importance_f.repeat(copies, 1)
            src_importance = torch.gather(importance, 1, src_idx)
            keep_total = n - removed
            protected_pool = min(max(int(keep_total * protect_multiplier), 1), n)
            threshold = torch.topk(importance, protected_pool, dim=1).values[:, -1:]
            priority = similarity + (src_importance < threshold).to(similarity) * 1e5
            order = priority.argsort(dim=1, descending=True)
            merge_order = order[:, :removed]
            unmerged_order = order[:, removed:]
            merge_src_idx = torch.gather(src_idx, 1, merge_order)
            merge_dst_for_src = torch.gather(dst_for_src, 1, merge_order)
            unmerged_idx = torch.gather(src_idx, 1, unmerged_order)
        else:
            merge_src_idx = src_idx
            merge_dst_for_src = dst_for_src
            unmerged_idx = src_idx[:, :0]

    def split(x: torch.Tensor):
        channels = x.shape[-1]
        src = torch.gather(x, 1, src_idx.unsqueeze(-1).expand(-1, -1, channels))
        dst = torch.gather(x, 1, reps.unsqueeze(-1).expand(-1, -1, channels))
        return src, dst

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        channels = x.shape[-1]
        src = torch.gather(
            x, 1, merge_src_idx.unsqueeze(-1).expand(-1, -1, channels)
        )
        dst = torch.gather(x, 1, reps.unsqueeze(-1).expand(-1, -1, channels))
        unm = torch.gather(
            x, 1, unmerged_idx.unsqueeze(-1).expand(-1, -1, channels)
        )
        if mode == "none":
            return torch.cat([unm, dst], dim=1)
        dst = dst.scatter_reduce(
            1,
            merge_dst_for_src.unsqueeze(-1).expand(-1, -1, channels),
            src,
            reduce=mode,
            include_self=True,
        )
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[-1]
        unm_count = unmerged_idx.shape[1]
        unm, dst = x[:, :unm_count], x[:, unm_count:]
        out = torch.zeros(batch, n, channels, device=x.device, dtype=x.dtype)
        out.scatter_(1, reps.unsqueeze(-1).expand(-1, -1, channels), dst)
        out.scatter_(1, unmerged_idx.unsqueeze(-1).expand(-1, -1, channels), unm)
        src = torch.gather(
            dst, 1, merge_dst_for_src.unsqueeze(-1).expand(-1, -1, channels)
        )
        out.scatter_(1, merge_src_idx.unsqueeze(-1).expand(-1, -1, channels), src)
        return out

    return merge, unmerge
