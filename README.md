# AnimateDiff

This repository is the official implementation of [AnimateDiff](https://arxiv.org/abs/2307.04725) [ICLR2024 Spotlight].
It is a plug-and-play module turning most community text-to-image models into animation generators, without the need of additional training.

**[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)** 
</br>
[Yuwei Guo](https://guoyww.github.io/),
[Ceyuan Yang✝](https://ceyuan.me/),
[Anyi Rao](https://anyirao.com/),
[Zhengyang Liang](https://maxleung99.github.io/),
[Yaohui Wang](https://wyhsirius.github.io/),
[Yu Qiao](https://scholar.google.com.hk/citations?user=gFtI-8QAAAAJ),
[Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/),
[Dahua Lin](http://dahua.site),
[Bo Dai](https://daibo.info)
(✝Corresponding Author)  
[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2307.04725)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://animatediff.github.io/)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/Masbfca/AnimateDiff)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/guoyww/AnimateDiff)

***Note:*** The `main` branch is for [Stable Diffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5); for [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), please refer `sdxl-beta` branch.


## Quick Demos
More results can be found in the [Gallery](__assets__/docs/gallery.md).
Some of them are contributed by the community.

<table class="center">
    <tr>
    <td><img src="__assets__/animations/model_01/01.gif"></td>
    <td><img src="__assets__/animations/model_01/02.gif"></td>
    <td><img src="__assets__/animations/model_01/03.gif"></td>
    <td><img src="__assets__/animations/model_01/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/30240/toonyou">ToonYou</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_03/01.gif"></td>
    <td><img src="__assets__/animations/model_03/02.gif"></td>
    <td><img src="__assets__/animations/model_03/03.gif"></td>
    <td><img src="__assets__/animations/model_03/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/4201/realistic-vision-v20">Realistic Vision V2.0</a></p>


## Quick Start
***Note:*** AnimateDiff is also offically supported by Diffusers.
Visit [AnimateDiff Diffusers Tutorial](https://huggingface.co/docs/diffusers/api/pipelines/animatediff) for more details.
*Following instructions is for working with this repository*.

***Note:*** For all scripts, checkpoint downloading will be *automatically* handled, so the script running may take longer time when first executed.

### 1. Setup repository and environment

```
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff

pip install -r requirements.txt
```

### 2. Launch the sampling script!
The generated samples can be found in `samples/` folder.

#### 2.1 Generate animations with comunity models
```
python -m scripts.animate --config configs/prompts/1_animate/1_1_animate_RealisticVision.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_2_animate_FilmVelvia.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_3_animate_ToonYou.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_4_animate_MajicMix.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_5_animate_RcnzCartoon.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_6_animate_Lyriel.yaml
python -m scripts.animate --config configs/prompts/1_animate/1_7_animate_Tusun.yaml
```

#### 2.2 Generate animation with MotionLoRA control
```
python -m scripts.animate --config configs/prompts/2_motionlora/2_motionlora_RealisticVision.yaml
```

#### 2.3 More control with SparseCtrl RGB and sketch
```
python -m scripts.animate --config configs/prompts/3_sparsectrl/3_1_sparsectrl_i2v.yaml
python -m scripts.animate --config configs/prompts/3_sparsectrl/3_2_sparsectrl_rgb_RealisticVision.yaml
python -m scripts.animate --config configs/prompts/3_sparsectrl/3_3_sparsectrl_sketch_RealisticVision.yaml
```

#### 2.4 Gradio app
We created a Gradio demo to make AnimateDiff easier to use. 
By default, the demo will run at `localhost:7860`.
```
python -u app.py
```
<img src="__assets__/figs/gradio.jpg" style="width: 75%">

## Quadtree / VBench Video Evaluation

### Importance-paper Figure 8 reproduction

The supplementary settings are 30 sampling steps and CFG 7.5 for every method.
The three published prompts are stored in
`configs/prompts/importance_fig8.yaml`. Submit the matched baseline, ToMe,
Importance, and Quadtree run with `scripts/slurm_importance_fig8.sh`.
The T2I backbone follows AnimateDiff's official v3 T2V RealisticVision example:
`realisticVisionV60B1_v51VAE.safetensors` with `v3_sd15_mm.ckpt`. The earlier
bare-SD1.5 run under `outputs/importance_fig8` is invalid as a Figure 8
reproduction and must not be used for comparisons.

### Quadtree merge-map diagnosis

Set `QT_DEBUG_DIR` to export the actual runtime plan as PNG plus NPZ. In the
PNG, blue positions are removed sources, green positions mark the retained
group output, red positions are untouched singletons, white lines are group
boundaries, and intensity follows CFG importance. The Tower
`r=0.70`, levels `[1,2]` maps are under `outputs/quadtree_debug_l12/maps`.

The maps expose a hard capacity limit: a 2x2 merge removes three of four
tokens, so removing 70% of all tokens requires selecting about 956 of the 1024
available 2x2 blocks. Consequently 93.36% of spatial positions are members of
merged groups and only 6.64% remain singleton, at every active denoising step.
At this ratio, a 2x2-only method cannot merge only background; foreground
merging is mathematically unavoidable. A foreground-preserving high-ratio
variant needs partial within-block merging or coarse background blocks rather
than whole-block 2x2 collapse.

The corrected role map also exposed a shifted-grid bug: non-wrapping offsets
crop the top/left border, forcing up to 127 arbitrary boundary tokens to remain
singleton on a 64x64 grid. At `r=0.70` this can consume nearly half of the 272
singleton positions. The fair video preset disables shifting until periodic
(wrap-around) shifted blocks are implemented.

This fork adds spatial token merging for AnimateDiff so video diffusion can be
evaluated in the same style as the importance-token-merge paper. The comparison
methods are:

- `tome`: original random bipartite ToMe-style spatial token merging.
- `importance`: CFG-importance-guided token merging from the importance paper.
- `quadtree`: fair-control preset: fixed ratio, pruning/merging from step 4,
  levels `[1,2,4]`, cost-global block allocation, and the same first/last-block
  scope as the baselines. Flat background blocks carry coarse 4x4 merges while
  expensive foreground regions retain finer tokens.
- `qt_opt`: quality-oriented video preset: adaptive ratio, merge from step 14,
  and levels `[1,2]`. Report it separately because its compute schedule is not
  identical to the fair-control baselines.
- `qt_early`: the aggressive image-style quadtree preset, kept for ablation only.

Token merging is applied to spatial transformer tokens in AnimateDiff. Temporal
attention and motion modules are left unmerged.

### Environment Notes

In the local `sana` environment, full 512x512 / 16-frame generation runs with:

```
--without-xformers
```

The installed xFormers path may raise a CUDA kernel configuration error for the
temporal attention module. Disabling xFormers is slower but stable; on a 16 GB
GPU, one 512x512 / 16-frame / 25-step video uses about 14.5 GB VRAM.

If Hugging Face downloads hang through Xet, use:

```
export HF_HUB_DISABLE_XET=1
```

### Smoke Tests

Use the small smoke config to verify that the pipeline, CFG importance map, and
quadtree patch path are working:

```
cd /home/zeyu/Projects/quadtree-token-merge/animatediff
source /home/zeyu/miniconda3/etc/profile.d/conda.sh
conda activate sana

HF_HUB_DISABLE_XET=1 python -u scripts/animate.py \
  --config configs/prompts/smoke_quadtree.yaml \
  --W 64 --H 64 --L 2 \
  --without-xformers \
  --prune-from-i 0 \
  --merge-from-i 1 \
  --compress-ratio 0.5 \
  --free-err-mask \
  --use-quadtree \
  --quadtree-levels 1 2 \
  --adaptive-ratio
```

For a single full-resolution sample:

```
HF_HUB_DISABLE_XET=1 python -u scripts/animate.py \
  --config configs/prompts/single_quadtree_512.yaml \
  --W 512 --H 512 --L 16 \
  --without-xformers \
  --merge-method quadtree \
  --compress-ratio 0.3 \
  --save-mp4
```

The saved `config.yaml` records the resolved `token_merge` settings, including
method, ratio, quadtree levels, adaptive schedule, and merge start step.

### Generate Videos for a VBench Table

On the HCTLRDS cluster, submit generation through Slurm (do not run CUDA jobs
on the login node):

```
cd /mnt/disk2/zeyu/Project/quadtree-token-merge/animatediff
mkdir -p outputs/vbench_table_anim/logs
sbatch -p a100_only scripts/slurm_vbench_generate.sh
```

This submits a 12-task array (3 methods x 4 ratios), capped at two concurrent
GPUs. Each task validates that its MP4 exists and is non-empty. Monitor with
`squeue -u $USER`; logs are in `outputs/vbench_table_anim/logs/`.

The helper script generates videos for the table-style comparison:

```
CONFIG=configs/prompts/single_quadtree_512.yaml \
OUT_ROOT=outputs/vbench_table_anim \
scripts/run_vbench_table_generation.sh
```

Default settings:

- methods: `tome importance quadtree`
- ratios: `0.40 0.60 0.70 0.75`
- resolution: `512x512`
- frames: `16`
- steps: `25`
- output format: GIF plus per-sample MP4

The output layout is:

```
outputs/vbench_table_anim/
  tome/r0p40/videos/00000.mp4
  importance/r0p40/videos/00000.mp4
  quadtree/r0p40/videos/00000.mp4
  ...
```

These `videos/*.mp4` folders are the inputs to VBench. VBench is not installed
by default in `sana`; install/run it separately, then collect scores into:

```
outputs/vbench_table_anim/scores.csv
```

with rows like:

```
ratio,method,semantic,quality,total
0.40,tome,75.40,81.69,80.44
0.40,importance,...
0.40,quadtree,...
```

Generate a markdown table:

```
python scripts/make_vbench_table_template.py \
  --scores outputs/vbench_table_anim/scores.csv
```

The target table format is:

| r | Semantic ToMe | Semantic Importance | Semantic Ours | Quality ToMe | Quality Importance | Quality Ours | Total ToMe | Total Importance | Total Ours |
|---|---|---|---|---|---|---|---|---|---|
| 0.40 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0.60 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0.70 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0.75 | -- | -- | -- | -- | -- | -- | -- | -- | -- |

### Current run (2026-07-11)

- AnimateDiff v3 checkpoint: `models/Motion_Module/v3_sd15_mm.ckpt` (present).
- Paper-compatible generation geometry: 16 frames at 512 x 512, 25 DDIM steps.
- Spatial merging scope: `max_downsample=1`, i.e. the highest-resolution first
  and last UNet blocks; temporal attention remains unmerged.
- The current one-prompt config is a full-resolution pipeline/quality smoke
  test, not a publishable VBench estimate. Standard VBench evaluation requires
  the official prompt suite and multiple samples per prompt.
- Full-resolution preflight passed as Slurm job `17369_0`: ToMe at `r=0.40`
  completed in 68 seconds and produced a 16-frame MP4. The remaining matrix was
  submitted in QOS-safe batches; see `squeue -u chenzeyu` for live status.
- Quadtree quality follow-up: adding 4x4 blocks (`[1,2,4]`, job `17375_0`)
  did not improve the high-ratio samples and caused extra local breakage at
  `r=0.60`, so it was rejected. Moving the conservative `[1,2]` preset from
  merge step 10 to step 14 improved foreground edge/detail retention at
  `r=0.70` (job `17376_0`); this is now the video default.
- Fair-control diagnosis (jobs `17378_0`--`17380_0`): step-4 fixed-ratio
  quadtree merging remains poor even after removing early representative-only
  pruning, temporal-max importance pooling, and restoring image-style
  `[1,2,4]` levels. The dominant failure is early high-noise local block
  averaging, not missing temporal trajectory protection. Temporal-max was
  therefore rejected. Use `quadtree` for controlled comparisons and `qt_opt`
  only as a separately labelled quality/schedule variant.
- A 768x768 test at `r=0.70` (job `17382_0`) worsened framing and subject
  duplication, so the paper-facing setup remains 512x512. Cost-global
  allocation at 512px (job `17383_0`) retained foreground structure better
  than saturating nearly every 2x2 block and is now the fair preset default.
- Single-video VBench screening at `r=0.70` (job `17386`) found that
  cost-global Quadtree improves motion smoothness over ToMe (0.888 vs 0.833),
  but trails in subject consistency (0.704 vs 0.810) and imaging quality
  (0.493 vs 0.700). Sharpening importance weights to power 4 reduced imaging
  quality further (0.419), so that experiment was rejected.
- The late-start `qt_opt` screen improved background consistency to 0.905 and
  motion smoothness to 0.872, but still trailed ToMe in subject consistency
  (0.749 vs 0.810) and imaging quality (0.496 vs 0.700). It is therefore not
  presented as the winning configuration based on this one-video screen.


## Technical Explanation
<details close>
<summary>Technical Explanation</summary>

### AnimateDiff

**AnimateDiff aims to learn transferable motion priors that can be applied to other variants of Stable Diffusion family.**
To this end, we design the following training pipeline consisting of three stages.

<img src="__assets__/figs/adapter_explain.png" style="width:100%">

- In **1. Alleviate Negative Effects** stage, we train the **domain adapter**, e.g., `v3_sd15_adapter.ckpt`, to fit defective visual aritfacts (e.g., watermarks) in the training dataset.
This can also benefit the distangled learning of motion and spatial appearance.
By default, the adapter can be removed at inference. It can also be integrated into the model and its effects can be adjusted by a lora scaler.

- In **2. Learn Motion Priors** stage, we train the **motion module**, e.g., `v3_sd15_mm.ckpt`, to learn the real-world motion patterns from videos.

- In **3. (optional) Adapt to New Patterns** stage, we train **MotionLoRA**, e.g., `v2_lora_ZoomIn.ckpt`, to efficiently adapt motion module for specific motion patterns (camera zooming, rolling, etc.).

### SparseCtrl

**SparseCtrl aims to add more control to text-to-video models by adopting some sparse inputs (e.g., few RGB images or sketch inputs).**
Its technicall details can be found in the following paper:

**[SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models](https://arxiv.org/abs/2311.16933)**  
[Yuwei Guo](https://guoyww.github.io/),
[Ceyuan Yang✝](https://ceyuan.me/),
[Anyi Rao](https://anyirao.com/),
[Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/),
[Dahua Lin](http://dahua.site),
[Bo Dai](https://daibo.info)
(✝Corresponding Author)  
[![arXiv](https://img.shields.io/badge/arXiv-2311.16933-b31b1b.svg)](https://arxiv.org/abs/2311.16933)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://guoyww.github.io/projects/SparseCtrl/)

</details>


## Model Versions
<details close>
<summary>Model Versions</summary>

### AnimateDiff v3 and SparseCtrl (2023.12)

In this version, we use **Domain Adapter LoRA** for image model finetuning, which provides more flexiblity at inference.
We also implement two (RGB image/scribble) [SparseCtrl](https://arxiv.org/abs/2311.16933) encoders, which can take abitary number of condition maps to control the animation contents.

<details close>
<summary>AnimateDiff v3 Model Zoo</summary>

| Name | HuggingFace | Type | Storage | Description |
| - | - | - | - | - |
| `v3_adapter_sd_v15.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_adapter.ckpt) | Domain Adapter | 97.4 MB | |
| `v3_sd15_mm.ckpt.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt) | Motion Module | 1.56 GB | |
| `v3_sd15_sparsectrl_scribble.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_sparsectrl_scribble.ckpt) | SparseCtrl Encoder | 1.86 GB | scribble condition |
| `v3_sd15_sparsectrl_rgb.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_sparsectrl_rgb.ckpt) | SparseCtrl Encoder | 1.85 GB | RGB image condition |
</details>

#### Limitations
1. Small fickering is noticable;
2. To stay compatible with comunity models, there is no specific optimizations for general T2V, leading to limited visual quality under this setting;
3. **(Style Alignment) For usage such as image animation/interpolation, it's recommanded to use images generated by the same community model.**

#### Demos
<table class="center">
    <tr style="line-height: 0">
    <td width=25% style="border: none; text-align: center">Input (by RealisticVision)</td>
    <td width=25% style="border: none; text-align: center">Animation</td>
    <td width=25% style="border: none; text-align: center">Input</td>
    <td width=25% style="border: none; text-align: center">Animation</td>
    </tr>
    <tr>
    <td width=25% style="border: none"><img src="__assets__/demos/image/RealisticVision_firework.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__assets__/animations/v3/animation_fireworks.gif" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__assets__/demos/image/RealisticVision_sunset.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__assets__/animations/v3/animation_sunset.gif" style="width:100%"></td>
    </tr>
</table>

<table class="center">
    <tr style="line-height: 0">
    <td width=25% style="border: none; text-align: center">Input Scribble</td>
    <td width=25% style="border: none; text-align: center">Output</td>
    <td width=25% style="border: none; text-align: center">Input Scribbles</td>
    <td width=25% style="border: none; text-align: center">Output</td>
    </tr>
    <tr>
      <td width=25% style="border: none"><img src="__assets__/demos/scribble/scribble_1.png" style="width:100%"></td>
      <td width=25% style="border: none"><img src="__assets__/animations/v3/sketch_boy.gif" style="width:100%"></td>
      <td width=25% style="border: none"><img src="__assets__/demos/scribble/scribble_2_readme.png" style="width:100%"></td>
      <td width=25% style="border: none"><img src="__assets__/animations/v3/sketch_city.gif" style="width:100%"></td>
    </tr>
</table>


### AnimateDiff SDXL-Beta (2023.11)

Release the Motion Module (beta version) on SDXL, available at [Google Drive](https://drive.google.com/file/d/1EK_D9hDOPfJdK4z8YDB8JYvPracNx2SX/view?usp=share_link
) / [HuggingFace](https://huggingface.co/guoyww/animatediff/blob/main/mm_sdxl_v10_beta.ckpt
) / [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules). High resolution videos (i.e., 1024x1024x16 frames with various aspect ratios) could be produced **with/without** personalized models. Inference usually requires ~13GB VRAM and tuned hyperparameters (e.g., sampling steps), depending on the chosen personalized models.  
Checkout to the branch [sdxl](https://github.com/guoyww/AnimateDiff/tree/sdxl) for more details of the inference.

<details close>
<summary>AnimateDiff SDXL-Beta Model Zoo</summary>

| Name | HuggingFace | Type | Storage Space |
| - | - | - | - |
| `mm_sdxl_v10_beta.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sdxl_v10_beta.ckpt) | Motion Module | 950 MB |
</details>

#### Demos
<table class="center">
    <tr style="line-height: 0">
    <td width=52% style="border: none; text-align: center">Original SDXL</td>
    <td width=30% style="border: none; text-align: center">Community SDXL</td>
    <td width=18% style="border: none; text-align: center">Community SDXL</td>
    </tr>
    <tr>
    <td width=52% style="border: none"><img src="__assets__/animations/motion_xl/01.gif" style="width:100%"></td>
    <td width=30% style="border: none"><img src="__assets__/animations/motion_xl/02.gif" style="width:100%"></td>
    <td width=18% style="border: none"><img src="__assets__/animations/motion_xl/03.gif" style="width:100%"></td>
    </tr>
</table>


### AnimateDiff v2 (2023.09)

In this version, the motion module `mm_sd_v15_v2.ckpt` ([Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI?usp=sharing) / [HuggingFace](https://huggingface.co/guoyww/animatediff) / [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules)) is trained upon larger resolution and batch size.
We found that the scale-up training significantly helps improve the motion quality and diversity.  
We also support **MotionLoRA** of eight basic camera movements.
MotionLoRA checkpoints take up only **77 MB storage per model**, and are available at [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI?usp=sharing) / [HuggingFace](https://huggingface.co/guoyww/animatediff) / [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules).

<details close>
<summary>AnimateDiff v2 Model Zoo</summary>

| Name | HuggingFace | Type | Parameter | Storage |
| - | - | - | - | - |
| `mm_sd_v15_v2.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt) | Motion Module | 453 M | 1.7 GB |
| `v2_lora_ZoomIn.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_ZoomIn.ckpt) | MotionLoRA | 19 M | 74 MB |
| `v2_lora_ZoomOut.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_ZoomOut.ckpt) | MotionLoRA | 19 M | 74 MB |
| `v2_lora_PanLeft.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_PanLeft.ckpt) | MotionLoRA | 19 M | 74 MB |
| `v2_lora_PanRight.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_PanRight.ckpt) | MotionLoRA | 19 M | 74 MB |
| `v2_lora_TiltUp.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_TiltUp.ckpt) | MotionLoRA | 19 M | 74 MB |
| `v2_lora_TiltDown.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_TiltDown.ckpt) | MotionLoRA | 19 M | 74 MB |
| `v2_lora_RollingClockwise.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_RollingClockwise.ckpt) | MotionLoRA | 19 M | 74 MB |
| `v2_lora_RollingAnticlockwise.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_RollingAnticlockwise.ckpt) | MotionLoRA | 19 M | 74 MB |
</details>


#### Demos (MotionLoRA)
<table class="center">
  <tr style="line-height: 0">
    <td colspan="2" style="border: none; text-align: center">Zoom In</td>
    <td colspan="2" style="border: none; text-align: center">Zoom Out</td>
    <td colspan="2" style="border: none; text-align: center">Zoom Pan Left</td>
    <td colspan="2" style="border: none; text-align: center">Zoom Pan Right</td>
  </tr>
  <tr>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/01.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/02.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/02.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/01.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/03.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/04.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/04.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/03.gif"></td>
  </tr>
  <tr style="line-height: 0">
    <td colspan="2" style="border: none; text-align: center">Tilt Up</td>
    <td colspan="2" style="border: none; text-align: center">Tilt Down</td>
    <td colspan="2" style="border: none; text-align: center">Rolling Anti-Clockwise</td>
    <td colspan="2" style="border: none; text-align: center">Rolling Clockwise</td>
  </tr>
  <tr>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/05.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/05.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/06.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/06.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/07.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/07.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/08.gif"></td>
    <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/08.gif"></td>
  </tr>
</table>


#### Demos (Improved Motions)
Here's a comparison between `mm_sd_v15.ckpt` (left) and improved `mm_sd_v15_v2.ckpt` (right).

<table class="center">
  <tr>
    <td><img src="__assets__/animations/compare/old_0.gif"></td>
    <td><img src="__assets__/animations/compare/new_0.gif"></td>
    <td><img src="__assets__/animations/compare/old_1.gif"></td>
    <td><img src="__assets__/animations/compare/new_1.gif"></td>
    <td><img src="__assets__/animations/compare/old_2.gif"></td>
    <td><img src="__assets__/animations/compare/new_2.gif"></td>
    <td><img src="__assets__/animations/compare/old_3.gif"></td>
    <td><img src="__assets__/animations/compare/new_3.gif"></td>
  </tr>
</table>


### AnimateDiff v1 (2023.07)

The first version of AnimateDiff!

<details close>
<summary>AnimateDiff v1 Model Zoo</summary>

| Name | HuggingFace | Parameter | Storage Space |
| - | - | - | - |
| mm_sd_v14.ckpt | [Link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v14.ckpt) | 417 M | 1.6 GB |
| mm_sd_v15.ckpt | [Link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15.ckpt) | 417 M | 1.6 GB |
</details>

</details>


## Training
Please check [Steps for Training](__assets__/docs/animatediff.md) for details.


## Related Resources

AnimateDiff for Stable Diffusion WebUI: [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) (by [@continue-revolution](https://github.com/continue-revolution))  
AnimateDiff for ComfyUI: [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) (by [@Kosinkadink](https://github.com/Kosinkadink))  
Google Colab: [Colab](https://colab.research.google.com/github/camenduru/AnimateDiff-colab/blob/main/AnimateDiff_colab.ipynb) (by [@camenduru](https://github.com/camenduru))


## Disclaimer
This project is released for academic use.
We disclaim responsibility for user-generated content.
Also, please be advised that our only official website are https://github.com/guoyww/AnimateDiff and https://animatediff.github.io, and all the other websites are NOT associated with us at AnimateDiff. 


## Contact Us
Yuwei Guo: [guoyw@ie.cuhk.edu.hk](mailto:guoyw@ie.cuhk.edu.hk)  
Ceyuan Yang: [limbo0066@gmail.com](mailto:limbo0066@gmail.com)  
Bo Dai: [doubledaibo@gmail.com](mailto:doubledaibo@gmail.com)


## BibTeX
```
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Liang, Zhengyang and Wang, Yaohui and Qiao, Yu and Agrawala, Maneesh and Lin, Dahua and Dai, Bo},
  journal={International Conference on Learning Representations},
  year={2024}
}

@article{guo2023sparsectrl,
  title={SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Agrawala, Maneesh and Lin, Dahua and Dai, Bo},
  journal={arXiv preprint arXiv:2311.16933},
  year={2023}
}
```


## Acknowledgements
Codebase built upon [Tune-a-Video](https://github.com/showlab/Tune-A-Video).
### Quadtree video CFG-map correction (2026-07-12)

The first video port did **not** match the image-generation implementation: it
reshaped the CFG residual maps from `[B,F,H,W]` to `[B*F,H,W]`, after which
`plan_quadtree` averaged dimension 0.  Consequently all 16 frames shared one
temporally averaged merge map; moving foreground responses were diluted and the
Tower example mainly protected the roof.

The video implementation now follows `pipe_sd.py` frame by frame.  It computes
`abs(noise_pred_text-noise_pred_uncond).mean(channel)` and constructs one
Quadtree plan for each frame, then applies the corresponding plan to both the
unconditional and conditional CFG branches.  No temporal averaging is used.
The fair Quadtree preset uses only 1x1/2x2 leaves and disables shifted crops.

Validation job `17411` completed successfully (AnimateDiff v3 motion module,
RealisticVision V6 SD1.5 base, 16 frames, 512x512, 30 steps, CFG 7.5, `r=0.7`).
The corrected three-method frame-8 visualization is saved at
`outputs/mergeviz_tower_framecfg_comparison.png`; the corrected Tower video is
`outputs/quadtree_framecfg_l12/quadtree/r0p70/videos/00000.mp4`.

For 2x2-only merging, one selected block removes three of four tokens.  Thus at
`r=0.7`, approximately `0.7/0.75 = 93.3%` of all 2x2 blocks must be selected.
The dense regular background grid in the visualization is therefore expected;
the meaningful diagnostic is whether the remaining singleton blocks follow the
per-frame high-CFG foreground/detail regions.

#### Superseding implementation: exact repository multi-resolution planner

The earlier AnimateDiff debugging preset (`2x2-only`, `cost_global`, no shift)
is not the method visualized by `tools/visualize_multires_merging.py` and must
not be used as the main Quadtree result.  The production `quadtree` preset now
matches the text-to-image code path in `tomesd/patch.py` and
`tomesd/methods/merge_quadtree.py`: levels `[1,2,4]`, max-pooled CFG importance,
`equal_quota`, importance-weighted aggregation, and a random grid shift on each
denoising step.  AnimateDiff invokes that same planner once per video frame so
the image implementation's batch mean cannot average away temporal motion.

Runtime visualization now reports leaf resolution rather than merge-operation
roles: green is an unmerged 1x1 leaf, yellow is a 2x2 group, and red is a 4x4
group.  Validation job `17416` completed successfully.  For Tower frame 8 at
`r=0.7`, the actual 64x64 plan contains 1231 groups (2865 removed tokens), 656
1x1 leaves, 480 2x2 groups, and 95 4x4 groups.  The matching three-panel figure
is `outputs/quadtree_repoimpl_tower/tower_frame8_resolution.png` and can be
regenerated with `scripts/visualize_quadtree_resolution.py`.

The video-specific code is isolated in
`animatediff/tomesd_video/quadtree_adapter.py`.  It contains no block-selection
algorithm: `plan_video_frames` calls the image `plan_quadtree` unchanged for
each `[H,W]` CFG frame, and `merge_from_video_plans` only batches those plans in
AnimateDiff's `[unconditional frames, conditional frames]` order.  Run
`scripts/test_quadtree_adapter.py` to verify exact numerical parity.  The test
checks image versus video `mean` merge, representative-token merge, unmerge,
and both CFG branches with zero tolerance (`rtol=0`, `atol=0`).

#### Video-quality optimized Quadtree (`qt_video_opt`)

`qt_video_opt` preserves the control method's target token count but improves
candidate ranking with three video-specific safeguards: percentile-normalized
CFG maps take a three-frame temporal maximum (`lambda=0.7`), local conditional
attention-feature disagreement is added to the score (`lambda=0.35`), and only
25% of the removal budget may be spent on coarse 4x4 leaves. The remaining
budget is filled with 2x2 leaves, so quality is not purchased by retaining more
tokens. At 64x64 and `r=0.7`, both control and optimized plans keep 1231 groups
and remove 2865 tokens; the 4x4 count changes from 95 to 47. The optimized
10-prompt, three-ratio generation and VBench job is `17427`, dependent on the
control job `17425`.

#### Encoder/decoder plan split ablation (`qt_stage_split`)

High-resolution encoder and decoder transformer blocks now carry explicit stage
labels and use separate plan-cache keys. `qt_stage_split` keeps the control
ratio and target K in every block, but uses `[1,2,4]` leaves in the encoder and
only `[1,2]` leaves in the final detail-restoring decoder. This isolates the
effect of decoder-side 4x4 merging without gaining compute. Its 10-prompt,
three-ratio generation plus VBench job is `17439`, dependent on control job
`17425`; output is `outputs/compare_10prompts_qt_stage_split`.

Completed 10-prompt results show that reducing coarse groups was harmful at
high ratios: at `r=0.7`, the continuous-quality proxy changed from 84.51
(control) to 83.68 (`qt_video_opt`) and 83.05 (`qt_stage_split`). With fixed K,
replacing 4x4 groups by 2x2 groups covers more spatial positions and reduces
the number of untouched 1x1 leaves, which hurts aesthetic/imaging quality.

The follow-up `qt_rank_opt` therefore preserves the control leaf budget exactly
(at `r=0.7`: 656 1x1 leaves, 480 2x2 groups, 95 4x4 groups) and changes only
candidate ranking using temporal CFG protection and local feature disagreement.
Its 10-prompt, three-ratio generation plus VBench run is job `17447`, with
outputs under `outputs/compare_10prompts_qt_rank_opt`.

For rapid iteration, full job `17447` was stopped after one video and replaced
by a three-prompt `r=0.7` gate using Tower, jellyfish, and Victorian streetlamp.
The reusable inputs are `configs/prompts/vbench_quick3.yaml` and
`configs/prompts/vbench_quick3_manifest.json`; the generation script accepts
`EXPECTED_VIDEOS=3`. Ranking-only quick job `17448` generates three videos and
runs VBench in the same allocation under `outputs/quick3_qt_rank_opt_r070`.
Only candidates that improve the continuous quality dimensions at this gate
should advance to the 10-prompt, three-ratio benchmark.

With the experimental constraint relaxed to equal model/data/K rather than
identical merge mechanics, `qt_similarity` replaces rigid block averaging.
Quadtree still allocates exactly K spatial representatives (the highest-CFG
token in each leaf), but every removed token is assigned to its most similar
representative by current-block cosine similarity, as in ToMe. Thus attention
token count/FLOPs remain comparable while adaptive destination density is kept.
Shape/K and finite-output tests pass. Quick3 `r=0.7` job `17453` runs after the
low-weight ablation job `17452`, under `outputs/quick3_qt_similarity_r070`.
