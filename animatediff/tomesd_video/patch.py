import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Type

import torch
from einops import rearrange


# Reuse the outer quadtree-token-merge ToMe / importance / quadtree kernels.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tomesd import merge  # noqa: E402
from tomesd.utils import init_generator, isinstance_str  # noqa: E402
from .quadtree_adapter import (
    merge_from_video_plans,
    plan_video_frames,
    similarity_merge_from_video_plans,
    video_importance,
)


_DEBUG_PLAN_INDEX = 0


def _save_quadtree_debug(plan, w, h):
    """Save the actual multi-resolution leaf map used at runtime.

    This intentionally matches tools/visualize_multires_merging.py:
    green=1x1, yellow=2x2 and red=4x4.  It visualizes spatial resolution,
    not the source/destination roles of a merge implementation.
    """
    global _DEBUG_PLAN_INDEX
    debug_dir = os.environ.get("QT_DEBUG_DIR")
    if not debug_dir:
        return

    import numpy as np
    from PIL import Image

    os.makedirs(debug_dir, exist_ok=True)
    group = plan["group_id_compact"].detach().cpu().reshape(h, w).numpy()
    importance = plan["imp_per_token"].detach().float().cpu().reshape(h, w).numpy()
    sizes = np.bincount(group.reshape(-1))
    token_group_size = sizes[group]
    grouped = token_group_size > 1
    reps = plan["rep_per_group"].detach().cpu().numpy()
    representative = np.zeros(h * w, dtype=bool)
    representative[reps] = sizes > 1
    representative = representative.reshape(h, w)
    removed_source = grouped & ~representative
    singleton = ~grouped

    lo, hi = np.percentile(importance, [1, 99])
    heat = np.clip((importance - lo) / max(hi - lo, 1e-8), 0, 1)
    colors = {
        1: np.array([68, 170, 92], dtype=np.uint8),
        4: np.array([245, 185, 63], dtype=np.uint8),
        16: np.array([208, 65, 70], dtype=np.uint8),
    }
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    for group_size, color in colors.items():
        rgb[token_group_size == group_size] = color

    stem = f"plan_{_DEBUG_PLAN_INDEX:04d}_{h}x{w}"
    Image.fromarray(rgb).resize(
        (w * 8, h * 8), resample=Image.Resampling.NEAREST
    ).save(os.path.join(debug_dir, stem + ".png"))
    np.savez_compressed(
        os.path.join(debug_dir, stem + ".npz"),
        group=group,
        importance=importance,
        grouped=grouped,
        representative=representative,
        removed_source=removed_source,
        singleton=singleton,
        group_sizes=sizes,
        token_group_size=token_group_size,
        kept_groups=plan["K"],
        merged_tokens=plan["N"] - plan["K"],
    )
    _DEBUG_PLAN_INDEX += 1


def _save_matching_debug(merge_fn, error_mask, w, h, method):
    """Save actual ToMe/Importance kept and removed spatial indices."""
    global _DEBUG_PLAN_INDEX
    debug_dir = os.environ.get("QT_DEBUG_DIR")
    debug = getattr(merge_fn, "_debug_plan", None)
    if not debug_dir or debug is None:
        return

    import numpy as np
    from PIL import Image

    os.makedirs(debug_dir, exist_ok=True)
    removed_idx = debug["removed_idx"][0].detach().cpu().numpy()
    destination_idx = debug["destination_idx"][0].detach().cpu().numpy()
    unmerged_idx = debug["unmerged_idx"][0].detach().cpu().numpy()
    removed = np.zeros(h * w, dtype=bool)
    destination = np.zeros(h * w, dtype=bool)
    unmerged = np.zeros(h * w, dtype=bool)
    removed[removed_idx] = True
    destination[destination_idx] = True
    unmerged[unmerged_idx] = True

    if error_mask is not None:
        heat = error_mask["heat_map"][0].detach().float().cpu()[None, None]
        heat = torch.nn.functional.interpolate(heat, size=(h, w), mode="area")[0, 0].numpy()
        lo, hi = np.percentile(heat, [1, 99])
        heat = np.clip((heat - lo) / max(hi - lo, 1e-8), 0, 1).reshape(-1)
    else:
        heat = np.full(h * w, 0.65, dtype=np.float32)

    rgb = np.zeros((h * w, 3), dtype=np.float32)
    rgb[removed, 2] = 0.45 + 0.55 * heat[removed]
    rgb[destination, 1] = 0.45 + 0.55 * heat[destination]
    rgb[unmerged, 0] = 0.35 + 0.65 * heat[unmerged]
    rgb = rgb.reshape(h, w, 3)
    stem = f"{method}_plan_{_DEBUG_PLAN_INDEX:04d}_{h}x{w}"
    Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (w * 8, h * 8), Image.Resampling.NEAREST
    ).save(os.path.join(debug_dir, stem + ".png"))
    np.savez_compressed(
        os.path.join(debug_dir, stem + ".npz"),
        removed=removed.reshape(h, w),
        destination=destination.reshape(h, w),
        unmerged=unmerged.reshape(h, w),
        importance=heat.reshape(h, w),
    )
    _DEBUG_PLAN_INDEX += 1


def compute_merge(
    x: torch.Tensor, tome_info: Dict[str, Any], stage: str = "other"
) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])

        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]

        if args.get("use_quadtree", False) and args["err_mask"] is not None:
            qt_levels = args.get("quadtree_levels", [1, 2, 4])
            if args.get("quadtree_decoder_fine", False) and stage == "decoder":
                qt_levels = [level for level in qt_levels if level <= 2]
            if args.get("quadtree_res_adaptive", False) and downsample == 1:
                qt_levels = [lvl for lvl in qt_levels if lvl <= 2]

            heat_map = args["err_mask"]["heat_map"]
            frame_count = heat_map.shape[0]
            cfg_copies = x.shape[0] // frame_count
            # Spatial attention is [unconditional frames, conditional frames].
            # Use conditional features to prevent merging low-CFG but textured
            # regions, while CFG remains the primary semantic signal.
            conditional_metric = x[(cfg_copies - 1) * frame_count : cfg_copies * frame_count]
            plan_heat_map = video_importance(
                heat_map,
                conditional_metric,
                temporal_lambda=args.get("quadtree_temporal_lambda", 0.0),
                feature_lambda=args.get("quadtree_feature_lambda", 0.0),
            )
            # Encoder and decoder must never share a spatial plan: the decoder
            # restores final detail and may use a finer leaf set.
            similarity_dst_ratio = args.get("quadtree_similarity_dst_ratio", None)
            plan_removed = (
                x.shape[1] - int(x.shape[1] * similarity_dst_ratio)
                if similarity_dst_ratio is not None
                else r
            )
            cache_key = (
                stage, x.shape[1], r, plan_removed, tuple(qt_levels), heat_map.shape[0]
            )
            # Feature-aware scoring depends on the current transformer's hidden
            # states, so a shape/stage cache would silently reuse the first
            # block's feature plan for later blocks. Pure CFG plans are shared.
            cache_feature_plans = args.get("quadtree_feature_lambda", 0.0) <= 0.0
            qt_cache = args.setdefault("_quadtree_plan_cache", {})
            if cache_feature_plans and cache_key in qt_cache:
                plans = qt_cache[cache_key]
            else:
                plans = plan_video_frames(
                    plan_heat_map,
                    w=w,
                    h=h,
                    removed_tokens=plan_removed,
                    device=x.device,
                    levels=qt_levels,
                    offset_x=args.get("quadtree_offset_x", 0),
                    offset_y=args.get("quadtree_offset_y", 0),
                    pool=args.get("quadtree_pool", "max"),
                    budget_mode=args.get("quadtree_budget_mode", "equal_quota"),
                    coarse_budget_fraction=args.get(
                        "quadtree_coarse_budget_fraction", None
                    ),
                )
                if cache_feature_plans:
                    qt_cache[cache_key] = plans
            if cache_key not in args.setdefault("_debugged_quadtree_plans", set()):
                _save_quadtree_debug(plans[min(7, len(plans) - 1)], w, h)
                args["_debugged_quadtree_plans"].add(cache_key)
            if args.get("quadtree_similarity_merge", False):
                m, u = similarity_merge_from_video_plans(
                    plans, x,
                    rep_mode=args.get("quadtree_similarity_rep", "cfg"),
                    removed_tokens=r,
                )
            else:
                m, u = merge_from_video_plans(
                    plans,
                    batch_size=x.shape[0],
                    weighted=args.get("quadtree_weighted", True),
                    weight_power=args.get("quadtree_weight_power", 1.0),
                )
        elif args["err_mask"] is not None:
            m, u = merge.matching_by_heat_map(
                x, w, h, r, generator=args["generator"], error_mask=args["err_mask"]
            )
        else:
            m, u = merge.bipartite_soft_matching_random2d(
                x,
                w,
                h,
                args["sx"],
                args["sy"],
                r,
                no_rand=not use_rand,
                generator=args["generator"],
            )

        if not args.get("_debugged_matching_plan", False):
            method = "importance" if args["err_mask"] is not None else "tome"
            _save_matching_debug(m, args["err_mask"], w, h, method)
            args["_debugged_matching_plan"] = True
    else:
        m = u = merge.do_nothing

    m_a, u_a = (m, u) if args["merge_attn"] else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"] else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m


def make_animatediff_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class ToMeBlock(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            attention_mask=None,
            video_length=None,
        ):
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(
                hidden_states, self._tome_info, self._tome_stage
            )

            norm_hidden_states = (
                self.norm1(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm1(hidden_states)
            )
            norm_hidden_states = m_a(
                norm_hidden_states, mode=self._tome_info["args"]["m_mode"]
            )

            if self.unet_use_cross_frame_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    attention_mask=attention_mask,
                    video_length=video_length,
                )
            else:
                attn_output = self.attn1(
                    norm_hidden_states, attention_mask=attention_mask
                )
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )
                norm_hidden_states = m_c(
                    norm_hidden_states, mode=self._tome_info["args"]["m_mode"]
                )
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                hidden_states = u_c(attn_output) + hidden_states

            norm_hidden_states = self.norm3(hidden_states)
            norm_hidden_states = m_m(
                norm_hidden_states, mode=self._tome_info["args"]["m_mode"]
            )
            hidden_states = u_m(self.ff(norm_hidden_states)) + hidden_states

            if self.unet_use_temporal_attention:
                d = hidden_states.shape[1]
                hidden_states = rearrange(
                    hidden_states, "(b f) d c -> (b d) f c", f=video_length
                )
                norm_hidden_states = (
                    self.norm_temp(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm_temp(hidden_states)
                )
                hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                hidden_states = rearrange(
                    hidden_states, "(b d) f c -> (b f) d c", d=d
                )

            return hidden_states

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[-2], args[0].shape[-1])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def apply_patch(
    model: torch.nn.Module,
    ratio: float = 0.5,
    max_downsample: int = 1,
    sx: int = 2,
    sy: int = 2,
    use_rand: bool = True,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
    err_mask=None,
    m_mode: str = "mean",
    use_quadtree: bool = False,
    quadtree_levels: list = None,
    quadtree_res_adaptive: bool = False,
    quadtree_pool: str = "max",
    quadtree_weighted: bool = True,
    quadtree_shift: bool = True,
    quadtree_budget_mode: str = "equal_quota",
    quadtree_weight_power: float = 1.0,
    quadtree_temporal_lambda: float = 0.0,
    quadtree_feature_lambda: float = 0.0,
    quadtree_coarse_budget_fraction: float = None,
    quadtree_decoder_fine: bool = False,
    quadtree_similarity_merge: bool = False,
    quadtree_similarity_rep: str = "cfg",
    quadtree_similarity_dst_ratio: float = None,
):
    remove_patch(model)

    diffusion_model = model.unet if hasattr(model, "unet") else model
    _qt_levels = quadtree_levels if quadtree_levels is not None else [1, 2, 4]
    _max_bs = max(_qt_levels) if use_quadtree else 1
    _qt_off_x = (
        int(torch.randint(0, _max_bs, (1,)).item())
        if use_quadtree and quadtree_shift
        else 0
    )
    _qt_off_y = (
        int(torch.randint(0, _max_bs, (1,)).item())
        if use_quadtree and quadtree_shift
        else 0
    )

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx,
            "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "err_mask": err_mask,
            "m_mode": m_mode,
            "use_quadtree": use_quadtree,
            "quadtree_levels": _qt_levels,
            "quadtree_res_adaptive": quadtree_res_adaptive,
            "quadtree_offset_x": _qt_off_x,
            "quadtree_offset_y": _qt_off_y,
            "quadtree_pool": quadtree_pool,
            "quadtree_weighted": quadtree_weighted,
            "quadtree_budget_mode": quadtree_budget_mode,
            "quadtree_weight_power": quadtree_weight_power,
            "quadtree_temporal_lambda": quadtree_temporal_lambda,
            "quadtree_feature_lambda": quadtree_feature_lambda,
            "quadtree_coarse_budget_fraction": quadtree_coarse_budget_fraction,
            "quadtree_decoder_fine": quadtree_decoder_fine,
            "quadtree_similarity_merge": quadtree_similarity_merge,
            "quadtree_similarity_rep": quadtree_similarity_rep,
            "quadtree_similarity_dst_ratio": quadtree_similarity_dst_ratio,
        },
    }
    hook_tome_model(diffusion_model)

    for name, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.__class__ = make_animatediff_tome_block(module.__class__)
            module._tome_info = diffusion_model._tome_info
            if "down_blocks" in name:
                module._tome_stage = "encoder"
            elif "up_blocks" in name:
                module._tome_stage = "decoder"
            else:
                module._tome_stage = "middle"

    return model


def remove_patch(model: torch.nn.Module):
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent

    return model
