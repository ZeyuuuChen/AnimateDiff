#!/usr/bin/env bash
#SBATCH --job-name=vb_cmp10
#SBATCH --output=animatediff/outputs/compare_10prompts/vbench/logs/%x-%j.out
#SBATCH --error=animatediff/outputs/compare_10prompts/vbench/logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT="/mnt/disk2/zeyu/Project/quadtree-token-merge"
AD="$ROOT/animatediff"
VB="/mnt/disk2/zeyu/Project/VBench"
CONDA="/mnt/disk2/zeyu/miniconda3/bin/conda"
GEN="$AD/outputs/compare_10prompts"
OUT="$GEN/vbench"
MANIFEST="$AD/configs/prompts/vbench_compare_10_manifest.json"
METHODS=(tome importance quadtree)
RATIOS=(0p40 0p60 0p70)

mkdir -p "$OUT/logs"
export PYTHONPATH="$VB:${PYTHONPATH:-}" HF_HUB_DISABLE_XET=1

for method in "${METHODS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    videos="$GEN/$method/r$ratio/videos"
    result="$OUT/$method/r$ratio"
    test "$(find "$videos" -maxdepth 1 -name '*.mp4' -type f | wc -l)" -eq 10
    mkdir -p "$result"
    echo "VBench: method=$method ratio=$ratio"
    "$CONDA" run --no-capture-output -n vbench python -u "$VB/evaluate.py" \
      --output_path "$result" --full_json_dir "$VB/vbench/VBench_full_info.json" \
      --videos_path "$videos" --mode custom_input --prompt_file "$MANIFEST" \
      --dimension subject_consistency background_consistency motion_smoothness \
        dynamic_degree aesthetic_quality imaging_quality overall_consistency
  done
done
