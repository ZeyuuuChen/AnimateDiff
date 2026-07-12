import argparse
import datetime
import inspect
import os
import sys
from omegaconf import OmegaConf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms as transforms

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_video, save_videos_grid
from animatediff.utils.util import load_weights, auto_download
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from PIL import Image
import numpy as np


def apply_merge_preset(args):
    if args.merge_method is None:
        return args

    ratio = args.compress_ratio if args.compress_ratio > 0 else 0.5

    if args.merge_method == "baseline":
        args.prune_from_i = -1
        args.merge_from_i = -1
        args.compress_ratio = 0.0
        args.free_err_mask = False
        args.use_quadtree = False
        args.adaptive_ratio = False
    elif args.merge_method == "tome":
        args.prune_from_i = -1
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = False
        args.use_quadtree = False
        args.adaptive_ratio = False
    elif args.merge_method == "importance":
        args.prune_from_i = 0
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = False
        args.adaptive_ratio = False
    elif args.merge_method == "qt_early":
        args.prune_from_i = 0
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.adaptive_ratio = True
    elif args.merge_method == "quadtree":
        # Match the repository's text-to-image Quadtree implementation:
        # multi-resolution 1x1/2x2/4x4 leaves, max-pooled CFG importance,
        # equal per-level budget, and a randomly shifted grid each step.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = False
        args.adaptive_ratio = False
    elif args.merge_method == "qt_video_opt":
        # Same target K as the control Quadtree, with video-aware ranking:
        # protect saliency in adjacent frames, include local feature detail,
        # and spend less of the removal budget on coarse 4x4 leaves.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = False
        args.quadtree_temporal_lambda = 0.7
        args.quadtree_feature_lambda = 0.35
        args.quadtree_coarse_budget_fraction = 0.25
        args.adaptive_ratio = False
    elif args.merge_method == "qt_stage_split":
        # Isolated fair ablation: the encoder retains multiresolution leaves,
        # while the detail-restoring decoder forbids 4x4 groups. Ratio/K stay
        # unchanged in every block.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = False
        args.quadtree_decoder_fine = True
        args.adaptive_ratio = False
    elif args.merge_method == "qt_rank_opt":
        # Ranking-only ablation: preserve the control leaf-size budget and K,
        # but rank safe blocks using temporally protected CFG saliency plus
        # local attention-feature disagreement.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = False
        args.quadtree_temporal_lambda = 0.7
        args.quadtree_feature_lambda = 0.35
        args.quadtree_coarse_budget_fraction = None
        args.quadtree_decoder_fine = False
        args.adaptive_ratio = False
    elif args.merge_method in ("qt_feature_only", "qt_temporal_only"):
        # Fast ranking ablations retain the exact control leaf budget and K.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = False
        args.quadtree_temporal_lambda = 0.7 if args.merge_method == "qt_temporal_only" else 0.0
        args.quadtree_feature_lambda = 0.35 if args.merge_method == "qt_feature_only" else 0.0
        args.quadtree_coarse_budget_fraction = None
        args.quadtree_decoder_fine = False
        args.adaptive_ratio = False
    elif args.merge_method in ("qt_feature_low", "qt_temporal_low", "qt_uniform"):
        # Second quick-screen round: conservative ranking weights and a merge
        # aggregation ablation, all with the exact control grouping budget/K.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = args.merge_method == "qt_uniform"
        args.quadtree_temporal_lambda = 0.25 if args.merge_method == "qt_temporal_low" else 0.0
        args.quadtree_feature_lambda = 0.05 if args.merge_method == "qt_feature_low" else 0.0
        args.quadtree_coarse_budget_fraction = None
        args.quadtree_decoder_fine = False
        args.adaptive_ratio = False
    elif args.merge_method in ("qt_similarity", "qt_similarity_medoid", "qt_similarity_mixed"):
        # Quadtree allocates spatial representatives; feature similarity, not
        # rigid block membership, determines every source-to-destination edge.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = True
        args.quadtree_temporal_lambda = 0.0
        args.quadtree_feature_lambda = 0.0
        args.quadtree_coarse_budget_fraction = None
        args.quadtree_decoder_fine = False
        args.quadtree_similarity_merge = True
        args.quadtree_similarity_rep = {
            "qt_similarity": "cfg",
            "qt_similarity_medoid": "medoid",
            "qt_similarity_mixed": "mixed",
        }[args.merge_method]
        args.adaptive_ratio = False
    elif args.merge_method in ("qt_similarity_spread", "qt_similarity_focus"):
        # Same K with different spatial destination density. Spread uses more
        # 2x2 leaves and medoid representatives for visual quality; focus uses
        # more 4x4 background leaves, leaving more 1x1 high-CFG representatives.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = True
        args.quadtree_temporal_lambda = 0.0
        args.quadtree_feature_lambda = 0.0
        args.quadtree_coarse_budget_fraction = (
            0.35 if args.merge_method == "qt_similarity_spread" else 0.65
        )
        args.quadtree_decoder_fine = False
        args.quadtree_similarity_merge = True
        args.quadtree_similarity_rep = (
            "medoid" if args.merge_method == "qt_similarity_spread" else "mixed"
        )
        args.adaptive_ratio = False
    elif args.merge_method in ("qt_hybrid_medoid", "qt_hybrid_mixed"):
        # Quadtree allocates 25% adaptive destinations; CFG+similarity chooses
        # exactly r sources, leaving high-value/low-similarity tokens unmerged.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = True
        args.quadtree_temporal_lambda = 0.0
        args.quadtree_feature_lambda = 0.0
        args.quadtree_coarse_budget_fraction = None
        args.quadtree_decoder_fine = False
        args.quadtree_similarity_merge = True
        args.quadtree_similarity_rep = (
            "medoid" if args.merge_method == "qt_hybrid_medoid" else "mixed"
        )
        args.quadtree_similarity_dst_ratio = 0.25
        args.adaptive_ratio = False
    elif args.merge_method in ("qt_hybrid_d20", "qt_hybrid_d30"):
        # Destination-ratio search at fixed final K: d20 leaves more protected
        # unmatched tokens; d30 provides denser similarity representatives.
        args.prune_from_i = 4
        args.merge_from_i = 4
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2, 4]
        args.quadtree_pool = "max"
        args.quadtree_budget_mode = "equal_quota"
        args.quadtree_no_shift = False
        args.quadtree_unweighted = True
        args.quadtree_temporal_lambda = 0.0
        args.quadtree_feature_lambda = 0.0
        args.quadtree_coarse_budget_fraction = None
        args.quadtree_decoder_fine = False
        args.quadtree_similarity_merge = True
        args.quadtree_similarity_rep = (
            "mixed" if args.merge_method == "qt_hybrid_d20" else "medoid"
        )
        args.quadtree_similarity_dst_ratio = (
            0.20 if args.merge_method == "qt_hybrid_d20" else 0.30
        )
        args.adaptive_ratio = False
    elif args.merge_method == "qt_opt":
        args.prune_from_i = 14
        args.merge_from_i = 14
        args.compress_ratio = ratio
        args.free_err_mask = True
        args.use_quadtree = True
        args.quadtree_levels = args.quadtree_levels or [1, 2]
        args.adaptive_ratio = True
    else:
        raise ValueError(f"Unknown merge method: {args.merge_method}")

    return args


@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = args.savedir or f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir, exist_ok=True)

    config  = OmegaConf.load(args.config)
    samples = []

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()

        # load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            auto_download(model_config.controlnet_path, is_dreambooth_lora=False)
            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if "pos_encoder.pe" not in name}
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to("cuda")

        prompts      = model_config.prompt
        n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        
        config[model_idx].token_merge = {
            "merge_method": args.merge_method,
            "prune_from_i": model_config.get("prune_from_i", args.prune_from_i),
            "merge_from_i": model_config.get("merge_from_i", args.merge_from_i),
            "compress_ratio": model_config.get("compress_ratio", args.compress_ratio),
            "free_err_mask": model_config.get("free_err_mask", args.free_err_mask),
            "use_quadtree": model_config.get("use_quadtree", args.use_quadtree),
            "quadtree_levels": model_config.get("quadtree_levels", args.quadtree_levels),
            "quadtree_res_adaptive": model_config.get("quadtree_res_adaptive", args.quadtree_res_adaptive),
            "quadtree_pool": model_config.get("quadtree_pool", args.quadtree_pool),
            "quadtree_weighted": model_config.get("quadtree_weighted", not args.quadtree_unweighted),
            "quadtree_shift": model_config.get("quadtree_shift", not args.quadtree_no_shift),
            "quadtree_budget_mode": model_config.get("quadtree_budget_mode", args.quadtree_budget_mode),
            "quadtree_temporal_lambda": model_config.get("quadtree_temporal_lambda", args.quadtree_temporal_lambda),
            "quadtree_feature_lambda": model_config.get("quadtree_feature_lambda", args.quadtree_feature_lambda),
            "quadtree_coarse_budget_fraction": model_config.get("quadtree_coarse_budget_fraction", args.quadtree_coarse_budget_fraction),
            "quadtree_decoder_fine": model_config.get("quadtree_decoder_fine", args.quadtree_decoder_fine),
            "quadtree_similarity_merge": model_config.get("quadtree_similarity_merge", args.quadtree_similarity_merge),
            "quadtree_similarity_rep": model_config.get("quadtree_similarity_rep", args.quadtree_similarity_rep),
            "quadtree_similarity_dst_ratio": model_config.get("quadtree_similarity_dst_ratio", args.quadtree_similarity_dst_ratio),
            "adaptive_ratio": model_config.get("adaptive_ratio", args.adaptive_ratio),
            "adaptive_alpha": model_config.get("adaptive_alpha", args.adaptive_alpha),
            "max_downsample": model_config.get("max_downsample", args.max_downsample),
            "merge_mlp": model_config.get("merge_mlp", args.merge_mlp),
        }

        config[model_idx].random_seed = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            
            # manually set random seed for reproduction
            if random_seed != -1: torch.manual_seed(random_seed)
            else: torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())
            
            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(
                prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale      = model_config.guidance_scale,
                width               = model_config.W,
                height              = model_config.H,
                video_length        = model_config.L,

                controlnet_images = controlnet_images,
                controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),

                prune_from_i = model_config.get("prune_from_i", args.prune_from_i),
                merge_from_i = model_config.get("merge_from_i", args.merge_from_i),
                compress_ratio = model_config.get("compress_ratio", args.compress_ratio),
                free_err_mask = model_config.get("free_err_mask", args.free_err_mask),
                use_quadtree = model_config.get("use_quadtree", args.use_quadtree),
                quadtree_levels = model_config.get("quadtree_levels", args.quadtree_levels),
                quadtree_res_adaptive = model_config.get("quadtree_res_adaptive", args.quadtree_res_adaptive),
                quadtree_pool = model_config.get("quadtree_pool", args.quadtree_pool),
                quadtree_weighted = model_config.get("quadtree_weighted", not args.quadtree_unweighted),
                quadtree_shift = model_config.get("quadtree_shift", not args.quadtree_no_shift),
                quadtree_budget_mode = model_config.get("quadtree_budget_mode", args.quadtree_budget_mode),
                quadtree_temporal_lambda = model_config.get("quadtree_temporal_lambda", args.quadtree_temporal_lambda),
                quadtree_feature_lambda = model_config.get("quadtree_feature_lambda", args.quadtree_feature_lambda),
                quadtree_coarse_budget_fraction = model_config.get("quadtree_coarse_budget_fraction", args.quadtree_coarse_budget_fraction),
                quadtree_decoder_fine = model_config.get("quadtree_decoder_fine", args.quadtree_decoder_fine),
                quadtree_similarity_merge = model_config.get("quadtree_similarity_merge", args.quadtree_similarity_merge),
                quadtree_similarity_rep = model_config.get("quadtree_similarity_rep", args.quadtree_similarity_rep),
                quadtree_similarity_dst_ratio = model_config.get("quadtree_similarity_dst_ratio", args.quadtree_similarity_dst_ratio),
                adaptive_ratio = model_config.get("adaptive_ratio", args.adaptive_ratio),
                adaptive_alpha = model_config.get("adaptive_alpha", args.adaptive_alpha),
                max_downsample = model_config.get("max_downsample", args.max_downsample),
                merge_mlp = model_config.get("merge_mlp", args.merge_mlp),
            ).videos
            samples.append(sample)

            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
            if args.save_mp4:
                save_video(sample[0], f"{savedir}/videos/{sample_idx:05d}.mp4", fps=args.fps)
            print(f"save to {savedir}/sample/{prompt}.gif")
            
            sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)
    if args.save_mp4:
        for idx, sample in enumerate(samples):
            save_video(sample, f"{savedir}/videos_gridless/{idx:05d}.mp4", fps=args.fps)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    parser.add_argument("--without-xformers", action="store_true")
    parser.add_argument("--savedir", type=str, default=None)
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--fps", type=int, default=8)

    parser.add_argument("--merge-method", type=str, default=None,
                        choices=["baseline", "tome", "importance", "quadtree", "qt_video_opt", "qt_stage_split", "qt_rank_opt", "qt_feature_only", "qt_temporal_only", "qt_feature_low", "qt_temporal_low", "qt_uniform", "qt_similarity", "qt_similarity_medoid", "qt_similarity_mixed", "qt_similarity_spread", "qt_similarity_focus", "qt_hybrid_medoid", "qt_hybrid_mixed", "qt_hybrid_d20", "qt_hybrid_d30", "qt_early", "qt_opt"])
    parser.add_argument("--prune-from-i", type=int, default=-1)
    parser.add_argument("--merge-from-i", type=int, default=-1)
    parser.add_argument("--compress-ratio", type=float, default=0.0)
    parser.add_argument("--free-err-mask", action="store_true")
    parser.add_argument("--use-quadtree", action="store_true")
    parser.add_argument("--quadtree-levels", type=int, nargs="+", default=None)
    parser.add_argument("--quadtree-res-adaptive", action="store_true")
    parser.add_argument("--quadtree-pool", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--quadtree-unweighted", action="store_true")
    parser.add_argument("--quadtree-no-shift", action="store_true")
    parser.add_argument("--quadtree-budget-mode", type=str, default="equal_quota", choices=["equal_quota", "cost_global"])
    parser.add_argument("--quadtree-temporal-lambda", type=float, default=0.0)
    parser.add_argument("--quadtree-feature-lambda", type=float, default=0.0)
    parser.add_argument("--quadtree-coarse-budget-fraction", type=float, default=None)
    parser.add_argument("--quadtree-decoder-fine", action="store_true")
    parser.add_argument("--quadtree-similarity-merge", action="store_true")
    parser.add_argument("--quadtree-similarity-rep", type=str, default="cfg", choices=["cfg", "medoid", "mixed"])
    parser.add_argument("--quadtree-similarity-dst-ratio", type=float, default=None)
    parser.add_argument("--adaptive-ratio", action="store_true")
    parser.add_argument("--adaptive-alpha", type=float, default=1.0)
    parser.add_argument("--max-downsample", type=int, default=1)
    parser.add_argument("--merge-mlp", action="store_true")

    args = parser.parse_args()
    args = apply_merge_preset(args)
    main(args)
