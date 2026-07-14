#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_EXE="${CONDA_EXE:-/mnt/disk2/zeyu/miniconda3/bin/conda}"
if [[ ! -x "$CONDA_EXE" ]]; then
  echo "Conda executable not found: $CONDA_EXE" >&2
  exit 1
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1

CONFIG="${CONFIG:-configs/prompts/single_quadtree_512.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs/vbench_table_anim}"
W="${W:-512}"
H="${H:-512}"
L="${L:-16}"
FPS="${FPS:-8}"
RATIOS="${RATIOS:-0.40 0.60 0.70 0.75}"
METHODS="${METHODS:-tome importance quadtree_best}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/mnt/disk2/zeyu/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14}"

mkdir -p "$OUT_ROOT"

for method in $METHODS; do
  for ratio in $RATIOS; do
    ratio_tag="${ratio/./p}"
    savedir="$OUT_ROOT/${method}/r${ratio_tag}"
    echo "==> method=${method} ratio=${ratio} savedir=${savedir}"
    "$CONDA_EXE" run --no-capture-output -n sana python -u scripts/animate.py \
      --pretrained-model-path "$PRETRAINED_MODEL_PATH" \
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
