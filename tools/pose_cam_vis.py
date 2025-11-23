"""Grad-CAM visualization for Pose-Swin branches.

This script aligns its interface with :mod:`tools.pose_infer_vis` so users can
point it at an image directory, weight file and config, then export
side-by-side Grad-CAM overlays for the global/local branches.
"""
import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg as global_cfg
from model import make_model


@dataclass
class CamResult:
    cam: torch.Tensor
    attention: Optional[torch.Tensor]


class WindowAttentionHook:
    """Register hooks on a ``ShiftWindowMSA`` module for Grad-CAM."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.fmap: Optional[torch.Tensor] = None
        self.grad: Optional[torch.Tensor] = None
        self.hw: Optional[Tuple[int, int]] = None
        self.attn: Optional[torch.Tensor] = None
        self.hooks = [
            module.register_forward_hook(self._forward_hook),
            module.register_full_backward_hook(self._backward_hook),
        ]

    def close(self):
        for h in self.hooks:
            h.remove()

    def _forward_hook(self, module, inputs, output):
        self.hw = tuple(int(v) for v in inputs[1]) if len(inputs) > 1 else None
        self.fmap = output
        attn = getattr(getattr(module, "w_msa", None), "last_attn", None)
        if attn is not None:
            self.attn = attn.detach()

    def _backward_hook(self, module, grad_inputs, grad_outputs):
        if grad_outputs:
            self.grad = grad_outputs[0]

    def build_cam(self, input_hw: Tuple[int, int]) -> CamResult:
        assert self.fmap is not None and self.grad is not None and self.hw is not None

        grad = self.grad
        fmap = self.fmap
        weights = grad.mean(dim=1)  # (B,C)
        cam = (fmap * weights[:, None, :]).sum(dim=2)  # (B,N)
        cam = F.relu(cam)
        B, N = cam.shape
        H, W = self.hw
        cam = cam.view(B, 1, H, W)
        cam = F.interpolate(cam, size=input_hw, mode="bilinear", align_corners=False)
        cam = (cam - cam.amin(dim=(2, 3), keepdim=True)) / (
            cam.amax(dim=(2, 3), keepdim=True) - cam.amin(dim=(2, 3), keepdim=True) + 1e-6
        )

        attn_map = self._build_attention_map(input_hw)
        return CamResult(cam=cam, attention=attn_map)

    def _build_attention_map(self, input_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        if self.attn is None or self.hw is None:
            return None

        B = self.fmap.shape[0]
        H, W = self.hw
        win = self.module.window_size
        pad_r = (win - W % win) % win
        pad_b = (win - H % win) % win
        H_pad, W_pad = H + pad_b, W + pad_r
        nW = (H_pad // win) * (W_pad // win)
        attn = self.attn.view(B, nW, self.module.w_msa.num_heads, win * win, win * win)
        attn = attn.mean(dim=2).mean(dim=-2).view(B * nW, win, win, 1)
        windows = attn
        merged = self.module.window_reverse(windows, H_pad, W_pad)
        if self.module.shift_size > 0:
            merged = torch.roll(merged, shifts=(self.module.shift_size, self.module.shift_size), dims=(1, 2))
        merged = merged[:, :H, :W, :].permute(0, 3, 1, 2).contiguous()
        merged = F.interpolate(merged, size=input_hw, mode="bilinear", align_corners=False)
        merged = (merged - merged.amin(dim=(2, 3), keepdim=True)) / (
            merged.amax(dim=(2, 3), keepdim=True) - merged.amin(dim=(2, 3), keepdim=True) + 1e-6
        )
        return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for Pose-Swin branches")
    parser.add_argument("image_dir", type=str, help="Directory containing input images")
    parser.add_argument("weight", type=str, help="Path to the model weight (.pth)")
    parser.add_argument("--config_file", default="", type=str, help="Path to config file for the model")
    parser.add_argument("--output_dir", default="./pose_cam", type=str, help="Directory to save visualizations")
    parser.add_argument("--branches", nargs="+", choices=["global", "local"], default=["global", "local"],
                        help="Branches to visualize")
    parser.add_argument("--stage_index", type=int, default=-1, help="Stage index to hook (default: last)")
    parser.add_argument("--block_index", type=int, default=-1, help="Block index within the stage to hook (default: last)")
    parser.add_argument("--class_idx", type=int, default=None, help="Target class for CAM; defaults to feature norm")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def build_cfg(cfg_path: str, opts: Optional[Sequence[str]], weight_path: str):
    cfg = global_cfg.clone()
    if cfg_path:
        cfg.merge_from_file(cfg_path)
    if opts:
        cfg.merge_from_list(list(opts))
    cfg.defrost()
    cfg.MODEL.POSE.ENABLE = True
    cfg.TEST.WEIGHT = weight_path
    cfg.freeze()
    return cfg


def build_transform(cfg) -> T.Compose:
    size = cfg.INPUT.SIZE_TEST
    normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        normalize,
    ])


def load_model(cfg, weight_path: str, device: torch.device):
    model = make_model(cfg, num_class=1, camera_num=1, view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    if hasattr(model, "load_param"):
        model.load_param(weight_path)
    else:
        state = torch.load(weight_path, map_location="cpu")
        state = state.get("state_dict", state)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.to(device)
    model.eval()
    return model


def list_images(image_dir: str) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in pathlib.Path(image_dir).rglob("*") if p.suffix.lower() in exts]
    paths.sort()
    return paths


def _overlay_heatmap(image: Image.Image, heatmap: torch.Tensor, alpha: float = 0.4) -> Image.Image:
    np_img = np.array(image).astype(np.float32) / 255.0
    heat = heatmap.detach().cpu().numpy()
    heat = np.clip(heat, 0.0, 1.0)
    color = cm.get_cmap("jet")(heat)[..., :3]
    blended = (1 - alpha) * np_img + alpha * color
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def _fetch_backbone(model: torch.nn.Module):
    if hasattr(model, "base"):
        return model.base
    if hasattr(model, "backbone"):
        return model.backbone
    return model


def _get_stage_list(backbone: torch.nn.Module, branch: str):
    if branch == "local" and hasattr(backbone, "local_stages") and backbone.local_stages:
        return backbone.local_stages
    if hasattr(backbone, "swin"):
        return backbone.swin.stages
    return backbone.stages


def _pick_target_module(backbone: torch.nn.Module, branch: str, stage_index: int, block_index: int):
    stages = _get_stage_list(backbone, branch)
    stage = stages[stage_index]
    block = stage.blocks[block_index]
    return block.attn


def _select_scalar(output: torch.Tensor, classifier: Optional[torch.nn.Module], cls_idx: Optional[int]):
    if classifier is not None and cls_idx is not None:
        logits = classifier(output)
        return logits[:, cls_idx].sum()
    return output.norm(p=2, dim=1).sum()


def _collect_pose_map(backbone: torch.nn.Module, input_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    hm = getattr(backbone, "last_hm", None)
    if hm is None:
        return None
    hm = hm.sum(dim=1, keepdim=True)
    hm = F.interpolate(hm, size=input_hw, mode="bilinear", align_corners=False)
    hm = (hm - hm.amin(dim=(2, 3), keepdim=True)) / (
        hm.amax(dim=(2, 3), keepdim=True) - hm.amin(dim=(2, 3), keepdim=True) + 1e-6
    )
    return hm


def _visualize_single(
    model: torch.nn.Module,
    image: Image.Image,
    tensor: torch.Tensor,
    branch: str,
    stage_index: int,
    block_index: int,
    cls_idx: Optional[int],
    device: torch.device,
) -> List[Image.Image]:
    backbone = _fetch_backbone(model)
    module = _pick_target_module(backbone, branch, stage_index, block_index)
    hook = WindowAttentionHook(module)
    tensor = tensor.unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        model_out = model(tensor)
        outputs = model_out[0] if isinstance(model_out, tuple) else model_out
        if isinstance(outputs, dict):
            if branch in outputs:
                feat_vec = outputs[branch]
            elif "global" in outputs:
                feat_vec = outputs["global"]
            else:
                feat_vec = next(iter(outputs.values()))

            classifier = getattr(model, f"classifier_{branch}", None) or getattr(model, "classifier_global", None)
        else:
            feat_vec = outputs
            classifier = getattr(model, "classifier", None)
        scalar = _select_scalar(feat_vec, classifier, cls_idx)
        scalar.backward()

    cam_pack = hook.build_cam(image.size[::-1])
    hook.close()
    cam_img = _overlay_heatmap(image, cam_pack.cam[0, 0])
    results = [cam_img]
    if cam_pack.attention is not None:
        attn_img = _overlay_heatmap(image, cam_pack.attention[0, 0])
        results.append(attn_img)

    pose_map = _collect_pose_map(backbone, image.size[::-1])
    if pose_map is not None:
        results.append(_overlay_heatmap(image, pose_map[0, 0], alpha=0.5))

    return results


def render_branches(
    model: torch.nn.Module,
    image_path: pathlib.Path,
    transform: T.Compose,
    branches: Sequence[str],
    stage_index: int,
    block_index: int,
    cls_idx: Optional[int],
    device: torch.device,
    out_dir: pathlib.Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)

    outputs: List[Image.Image] = [image]
    for branch in branches:
        outputs.extend(
            _visualize_single(model, image, tensor, branch, stage_index, block_index, cls_idx, device)
        )

    widths, heights = zip(*(img.size for img in outputs))
    total_w = sum(widths)
    max_h = max(heights)
    canvas = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))
    offset = 0
    for img in outputs:
        canvas.paste(img, (offset, 0))
        offset += img.size[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / f"{image_path.stem}_cam.png")



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_cfg(args.config_file, args.opts, args.weight)
    transform = build_transform(cfg)

    model = load_model(cfg, args.weight, device)
    img_paths = list_images(args.image_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    for path in img_paths:
        render_branches(
            model,
            path,
            transform,
            args.branches,
            args.stage_index,
            args.block_index,
            args.class_idx,
            device,
            pathlib.Path(args.output_dir),
        )
        print(f"Saved CAM visualization for {path.name}")


if __name__ == "__main__":
    main()
