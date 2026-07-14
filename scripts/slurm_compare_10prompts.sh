#!/usr/bin/env bash
#SBATCH --job-name=ad_cmp10
#SBATCH --output=animatediff/outputs/compare_10prompts/logs/%x-%j.out
#SBATCH --error=animatediff/outputs/compare_10prompts/logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT="${ROOT:-/mnt/disk2/zeyu/Project/quadtree-token-merge/animatediff}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/compare_10prompts}"
CONFIG="${CONFIG:-configs/prompts/vbench_compare_10.yaml}"
CONDA_EXE="${CONDA_EXE:-/mnt/disk2/zeyu/miniconda3/bin/conda}"
MODEL="${PRETRAINED_MODEL_PATH:-/mnt/disk2/zeyu/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14}"
read -r -a METHODS <<< "${METHODS:-tome importance quadtree_best}"
read -r -a RATIOS <<< "${RATIOS:-0.40 0.60 0.70}"
VBENCH_ROOT="${VBENCH_ROOT:-/mnt/disk2/zeyu/Project/VBench}"
VBENCH_OUT="$OUT_ROOT/vbench"
MANIFEST="${MANIFEST:-$ROOT/configs/prompts/vbench_compare_10_manifest.json}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-10}"

cd "$ROOT"
mkdir -p "$OUT_ROOT/logs"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false XFORMERS_IGNORE_FLASH_VERSION_CHECK=1

for method in "${METHODS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    tag="${ratio/./p}"
    savedir="$OUT_ROOT/$method/r$tag"
    echo "generation: method=$method ratio=$ratio output=$savedir"
    if [[ "$(find "$savedir/videos" -maxdepth 1 -name '*.mp4' -type f 2>/dev/null | wc -l)" -ne "$EXPECTED_VIDEOS" ]]; then
      "$CONDA_EXE" run --no-capture-output -n sana python -u scripts/animate.py \
        --pretrained-model-path "$MODEL" --config "$CONFIG" \
        --W 512 --H 512 --L 16 --without-xformers \
        --merge-method "$method" --compress-ratio "$ratio" \
        --savedir "$savedir" --save-mp4 --fps 8
    fi
    test "$(find "$savedir/videos" -maxdepth 1 -name '*.mp4' -type f | wc -l)" -eq "$EXPECTED_VIDEOS"
  done
done

# Evaluate in the same allocation, so cluster job-count limits cannot leave the
# benchmark generated but unevaluated.
mkdir -p "$VBENCH_OUT/logs"
export PYTHONPATH="$VBENCH_ROOT:${PYTHONPATH:-}"
for method in "${METHODS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    tag="${ratio/./p}"
    videos="$OUT_ROOT/$method/r$tag/videos"
    result="$VBENCH_OUT/$method/r$tag"
    mkdir -p "$result"
    if ! compgen -G "$result/*_eval_results.json" >/dev/null; then
      echo "VBench: method=$method ratio=$ratio"
      "$CONDA_EXE" run --no-capture-output -n vbench python -u "$VBENCH_ROOT/evaluate.py" \
        --output_path "$result" \
        --full_json_dir "$VBENCH_ROOT/vbench/VBench_full_info.json" \
        --videos_path "$videos" --mode custom_input --prompt_file "$MANIFEST" \
        --dimension subject_consistency background_consistency motion_smoothness \
          dynamic_degree aesthetic_quality imaging_quality overall_consistency
    fi
  done
done

echo "generation and VBench evaluation complete: $OUT_ROOT"
