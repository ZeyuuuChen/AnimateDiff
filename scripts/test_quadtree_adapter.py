#!/usr/bin/env python3
"""Numerical parity checks between image Quadtree and its video adapter."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tomesd.methods.merge_quadtree import merge_from_plan
import animatediff.tomesd_video.patch as video_patch
from animatediff.tomesd_video.quadtree_adapter import (
    merge_from_video_plans,
    plan_video_frames,
    similarity_merge_from_video_plans,
)


def main():
    torch.manual_seed(7)
    frames, h, w, channels = 3, 16, 16, 8
    heat = torch.rand(frames, h, w)
    plans = plan_video_frames(
        heat, w=w, h=h, removed_tokens=int(h * w * 0.7),
        device=torch.device("cpu"), levels=[1, 2, 4],
        offset_x=0, offset_y=0, pool="max", budget_mode="equal_quota",
    )

    # F=1 must be identical to the image implementation for every operation.
    x1 = torch.randn(1, h * w, channels)
    image_merge, image_unmerge = merge_from_plan(plans[0], weighted=True)
    video_merge, video_unmerge = merge_from_video_plans(
        plans[:1], batch_size=1, weighted=True
    )
    for mode in ("mean", "none"):
        a, b = image_merge(x1, mode=mode), video_merge(x1, mode=mode)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(image_unmerge(a), video_unmerge(b), rtol=0, atol=0)

    # With CFG, each frame plan must repeat in [frames][frames] branch order.
    x = torch.randn(2 * frames, h * w, channels)
    merge, unmerge = merge_from_video_plans(plans, batch_size=2 * frames, weighted=True)
    merged = merge(x)
    restored = unmerge(merged)
    for branch in range(2):
        for frame in range(frames):
            idx = branch * frames + frame
            m, u = merge_from_plan(plans[frame], weighted=True)
            torch.testing.assert_close(merged[idx : idx + 1], m(x[idx : idx + 1]), rtol=0, atol=0)
            torch.testing.assert_close(restored[idx : idx + 1], u(m(x[idx : idx + 1])), rtol=0, atol=0)

    for rep_mode in ("cfg", "medoid", "mixed"):
        similarity_merge, similarity_unmerge = similarity_merge_from_video_plans(
            plans, x, rep_mode=rep_mode
        )
        similarity_tokens = similarity_merge(x)
        assert similarity_tokens.shape == (2 * frames, plans[0]["K"], channels)
        similarity_full = similarity_unmerge(similarity_tokens)
        assert similarity_full.shape == x.shape
        assert torch.isfinite(similarity_full).all()
    over_merge, over_unmerge = similarity_merge_from_video_plans(
        plans, x, rep_mode="mixed", removed_tokens=h * w
    )
    over_tokens = over_merge(x)
    assert over_tokens.shape == (2 * frames, plans[0]["K"], channels)
    assert torch.isfinite(over_unmerge(over_tokens)).all()
    destination_plans = plan_video_frames(
        heat, w=w, h=h, removed_tokens=192,
        device=torch.device("cpu"), levels=[1, 2, 4],
        offset_x=0, offset_y=0, pool="max", budget_mode="equal_quota",
    )
    hybrid_merge, hybrid_unmerge = similarity_merge_from_video_plans(
        destination_plans, x, rep_mode="mixed", removed_tokens=179
    )
    hybrid_tokens = hybrid_merge(x)
    assert hybrid_tokens.shape == (2 * frames, h * w - 179, channels)
    assert torch.isfinite(hybrid_unmerge(hybrid_tokens)).all()

    original_planner = video_patch.plan_video_frames
    calls = {"count": 0}

    def counted_planner(*args, **kwargs):
        calls["count"] += 1
        return original_planner(*args, **kwargs)

    def make_tome_info(feature_lambda):
        return {
            "size": (h, w),
            "args": {
                "ratio": 0.7,
                "max_downsample": 1,
                "sx": 2,
                "sy": 2,
                "use_rand": True,
                "generator": None,
                "merge_attn": True,
                "merge_crossattn": False,
                "merge_mlp": False,
                "err_mask": {"heat_map": heat},
                "m_mode": "mean",
                "use_quadtree": True,
                "quadtree_levels": [1, 2, 4],
                "quadtree_pool": "max",
                "quadtree_budget_mode": "equal_quota",
                "quadtree_offset_x": 0,
                "quadtree_offset_y": 0,
                "quadtree_weighted": True,
                "quadtree_temporal_lambda": 0.0,
                "quadtree_feature_lambda": feature_lambda,
                "quadtree_coarse_budget_fraction": None,
                "quadtree_decoder_fine": False,
                "quadtree_similarity_merge": False,
                "quadtree_similarity_rep": "cfg",
                "quadtree_similarity_dst_ratio": None,
            },
        }

    try:
        video_patch.plan_video_frames = counted_planner
        static_info = make_tome_info(0.0)
        video_patch.compute_merge(x, static_info, stage="encoder")
        video_patch.compute_merge(x + 1, static_info, stage="encoder")
        assert calls["count"] == 1

        calls["count"] = 0
        feature_info = make_tome_info(0.5)
        video_patch.compute_merge(x, feature_info, stage="encoder")
        video_patch.compute_merge(x + 1, feature_info, stage="encoder")
        assert calls["count"] == 2
    finally:
        video_patch.plan_video_frames = original_planner

    print("PASS: video adapter is numerically identical to image Quadtree per frame")
    print("PASS: Quadtree-guided similarity merge preserves exact K and shape")
    print("PASS: feature-aware Quadtree plans are not reused across blocks")


if __name__ == "__main__":
    main()
