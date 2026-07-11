#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /home/zeyu/miniconda3/etc/profile.d/conda.sh
conda activate sana

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

CONFIG="${CONFIG:-configs/prompts/single_quadtree_512.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs/vbench_table_anim}"
W="${W:-512}"
H="${H:-512}"
L="${L:-16}"
FPS="${FPS:-8}"
RATIOS="${RATIOS:-0.40 0.60 0.70 0.75}"
METHODS="${METHODS:-tome importance quadtree}"

mkdir -p "$OUT_ROOT"

for method in $METHODS; do
  for ratio in $RATIOS; do
    ratio_tag="${ratio/./p}"
    savedir="$OUT_ROOT/${method}/r${ratio_tag}"
    echo "==> method=${method} ratio=${ratio} savedir=${savedir}"
    python -u scripts/animate.py \
      --config "$CONFIG" \
      --W "$W" --H "$H" --L "$L" \
      --without-xformers \
      --merge-method "$method" \
      --compress-ratio "$ratio" \
      --savedir "$savedir" \
      --save-mp4 \
      --fps "$FPS"
  done
done

echo "Done. MP4 videos are under: $OUT_ROOT/<method>/r<ratio>/videos/"
