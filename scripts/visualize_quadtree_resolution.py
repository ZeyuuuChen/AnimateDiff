#!/usr/bin/env python3
"""Render the actual AnimateDiff Quadtree plan like the image-code figure."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    1: np.array([68, 170, 92], dtype=np.uint8),
    4: np.array([245, 185, 63], dtype=np.uint8),
    16: np.array([208, 65, 70], dtype=np.uint8),
}


def read_frame(path, frame_index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_index} from {path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")


def normalize(x):
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / max(float(hi - lo), 1e-8), 0, 1)


def heat_image(importance, size):
    stops = np.array([[39,72,137], [55,143,184], [170,220,172],
                      [254,224,139], [215,48,39]], dtype=np.float32)
    x = normalize(importance) * (len(stops) - 1)
    i = np.floor(x).astype(np.int32)
    j = np.clip(i + 1, 0, len(stops) - 1)
    rgb = stops[i] * (1 - (x - i)[..., None]) + stops[j] * (x - i)[..., None]
    return Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)


def size_image(token_sizes, size):
    rgb = np.full((*token_sizes.shape, 3), 120, dtype=np.uint8)
    for n, color in COLORS.items():
        rgb[token_sizes == n] = color
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


def overlay(frame, group, token_sizes, size):
    base = frame.resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")
    color = np.asarray(size_image(token_sizes, size).convert("RGBA")).copy()
    color[..., 3] = 86
    out = Image.alpha_composite(base, Image.fromarray(color))
    draw = ImageDraw.Draw(out)
    h, w = group.shape
    cell = size / w
    # Draw exact group boundaries, including shifted grids and singleton leaves.
    for y in range(h):
        for x in range(w):
            if x == 0 or group[y, x] != group[y, x - 1]:
                draw.line((x * cell, y * cell, x * cell, (y + 1) * cell), fill=(20,20,20,150), width=1)
            if y == 0 or group[y, x] != group[y - 1, x]:
                draw.line((x * cell, y * cell, (x + 1) * cell, y * cell), fill=(20,20,20,150), width=1)
    return out.convert("RGB")


def labelled(image, title, font):
    out = Image.new("RGB", (image.width, image.height + 42), "white")
    out.paste(image, (0, 42))
    ImageDraw.Draw(out).text((8, 8), title, fill=(25,25,25), font=font)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frame", type=int, default=7)
    p.add_argument("--size", type=int, default=512)
    args = p.parse_args()

    data = np.load(args.plan)
    group = data["group"]
    importance = data["importance"]
    if "token_group_size" in data:
        token_sizes = data["token_group_size"]
    else:
        counts = np.bincount(group.reshape(-1))
        token_sizes = counts[group]

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    panels = [
        labelled(heat_image(importance, args.size), "CFG importance", font),
        labelled(overlay(read_frame(args.video, args.frame), group, token_sizes, args.size),
                 "Multi-resolution overlay", font),
        labelled(size_image(token_sizes, args.size), "Resolution map", font),
    ]
    gap = 14
    footer = 48
    canvas = Image.new("RGB", (3 * args.size + 2 * gap, panels[0].height + footer), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * (args.size + gap), 0))
    counts = np.bincount(group.reshape(-1))
    stats = (f"groups={len(counts)}/{group.size}, merged={group.size-len(counts)}, "
             f"1x1={int(np.sum(counts == 1))}, 2x2={int(np.sum(counts == 4))}, "
             f"4x4={int(np.sum(counts == 16))}")
    ImageDraw.Draw(canvas).text((args.size + gap, panels[0].height + 12), stats,
                                fill=(35,35,35), font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
