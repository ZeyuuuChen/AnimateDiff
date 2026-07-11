import math
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
from tomesd.methods import merge_quadtree as merge_qt  # noqa: E402
from tomesd.utils import init_generator, isinstance_str  # noqa: E402


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
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
            if args.get("quadtree_res_adaptive", False) and downsample == 1:
                qt_levels = [lvl for lvl in qt_levels if lvl <= 2]

            cache_key = (x.shape[1], r, tuple(qt_levels))
            qt_cache = args.setdefault("_quadtree_plan_cache", {})
            if cache_key not in qt_cache:
                qt_cache[cache_key] = merge_qt.plan_quadtree(
                    w,
                    h,
                    r,
                    B=x.shape[0],
                    device=x.device,
                    error_mask=args["err_mask"],
                    levels=qt_levels,
                    offset_x=args.get("quadtree_offset_x", 0),
                    offset_y=args.get("quadtree_offset_y", 0),
                    pool=args.get("quadtree_pool", "max"),
                    budget_mode=args.get("quadtree_budget_mode", "equal_quota"),
                )
            plan = qt_cache[cache_key]
            m, u = (
                (merge.do_nothing, merge.do_nothing)
                if plan is None
                else merge_qt.merge_from_plan(
                    plan, weighted=args.get("quadtree_weighted", True)
                )
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
                hidden_states, self._tome_info
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
        },
    }
    hook_tome_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.__class__ = make_animatediff_tome_block(module.__class__)
            module._tome_info = diffusion_model._tome_info

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
