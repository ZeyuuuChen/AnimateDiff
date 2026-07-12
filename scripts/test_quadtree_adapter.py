#!/usr/bin/env python3
"""Numerical parity checks between image Quadtree and its video adapter."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tomesd.methods.merge_quadtree import merge_from_plan
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
    print("PASS: video adapter is numerically identical to image Quadtree per frame")
    print("PASS: Quadtree-guided similarity merge preserves exact K and shape")


if __name__ == "__main__":
    main()
