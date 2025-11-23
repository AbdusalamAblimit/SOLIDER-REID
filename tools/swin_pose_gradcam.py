"""Grad-CAM visualizer for Pose-Swin models.

This script loads a Pose-Swin checkpoint, runs a single-image forward pass,
computes Grad-CAM for every Swin stage on both the global and local branches,
and exports per-stage overlays.

Usage example:
    python tools/swin_pose_gradcam.py \
        --config_file pose/config_vispredict.py \
        --weight output/pose_swin.pth \
        --image_dir test_images/pose \
        --output_dir ./pose_cam

The script mirrors the CLI style used by :mod:`tools.pose_infer_vis` so configs
and command-line overrides can be reused.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg  # noqa: E402
from model import make_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM for Pose-Swin")
    parser.add_argument("image_dir", type=str, help="Directory containing input images")
    parser.add_argument("weight", type=str, help="Path to the trained weight (.pth)")
    parser.add_argument(
        "--config_file", default="", type=str, help="Path to the model config file"
    )
    parser.add_argument(
        "--output_dir", default="./pose_grad_cam", type=str, help="Directory to save CAM overlays"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Computation device (cuda or cpu)"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


# -------------------- configuration & loading --------------------

def _build_cfg(cfg_file: str, opts: Optional[Sequence[str]], weight_path: str) -> None:
    if cfg_file:
        cfg.merge_from_file(cfg_file)
    if opts:
        cfg.merge_from_list(list(opts))
    cfg.defrost()
    cfg.MODEL.POSE.ENABLE = True
    cfg.TEST.WEIGHT = weight_path
    cfg.freeze()


def _build_transform() -> T.Compose:
    size = cfg.INPUT.SIZE_TRAIN
    normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    return T.Compose([T.Resize(size), T.ToTensor(), normalize])


def _load_model(weight_path: str, device: torch.device):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    model = make_model(
        cfg,
        num_class=1,
        camera_num=1,
        view_num=1,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    if hasattr(model, "load_param"):
        model.load_param(weight_path)
    else:
        state = torch.load(weight_path, map_location="cpu")
        state = state.get("state_dict", state)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


# -------------------- helpers --------------------

def _list_images(root: str) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in pathlib.Path(root).rglob("*") if p.suffix.lower() in exts]
    paths.sort()
    return paths


def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    cam_min = cam.amin(dim=(2, 3), keepdim=True)
    cam_max = cam.amax(dim=(2, 3), keepdim=True)
    return (cam - cam_min) / (cam_max - cam_min + 1e-6)


def _build_cam_from_map(
    fmap: torch.Tensor, grad: torch.Tensor, target_hw: Tuple[int, int]
) -> torch.Tensor:
    """Standard Grad-CAM given feature map and gradient."""
    assert fmap.shape == grad.shape
    weights = grad.mean(dim=(2, 3))  # (B, C)
    cam = (fmap * weights[:, :, None, None]).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
    return _normalize_cam(cam)


def _tensor_to_overlay(cam: torch.Tensor, image_bgr: np.ndarray) -> np.ndarray:
    cam_np = cam.squeeze(0).squeeze(0).detach().cpu().numpy()
    heatmap = cv2.applyColorMap((cam_np * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)
    return overlay


# -------------------- main pipeline --------------------

def _retain_grads(tensors: Iterable[torch.Tensor]) -> None:
    for t in tensors:
        if t.requires_grad:
            t.retain_grad()


def run_grad_cam(model, device: torch.device, image_path: pathlib.Path, transform: T.Compose, out_dir: pathlib.Path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    orig_hw = (image_bgr.shape[0], image_bgr.shape[1])

    model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(True):
        outputs = model(image_tensor)
        global_maps: List[torch.Tensor] = outputs.get("global_maps", [])
        local_maps: List[torch.Tensor] = outputs.get("local_maps", [])
        global_feat: torch.Tensor = outputs["global_feat"]
        local_feat: torch.Tensor = outputs["local_feat"]

        _retain_grads(global_maps)
        _retain_grads(local_maps)

        # Use the L2 norm of both heads as the optimization target to backpropagate.
        score = global_feat.norm(p=2, dim=1).sum() + local_feat.norm(p=2, dim=1).sum()
        score.backward()

    branches = [("global", global_maps), ("local", local_maps)]
    stem = image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for branch_name, maps in branches:
        for idx, fmap in enumerate(maps):
            if fmap.grad is None:
                continue
            cam = _build_cam_from_map(fmap.detach(), fmap.grad.detach(), orig_hw)
            overlay = _tensor_to_overlay(cam, image_bgr)
            save_name = f"{stem}_{branch_name}_stage{idx}.png"
            cv2.imwrite(str(out_dir / save_name), overlay)


# -------------------- entry --------------------

def main():
    args = parse_args()
    _build_cfg(args.config_file, args.opts, args.weight)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = _build_transform()
    model = _load_model(args.weight, device)

    images = _list_images(args.image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    out_dir = pathlib.Path(args.output_dir)
    for img_path in tqdm(images, desc="Grad-CAM"):
        run_grad_cam(model, device, img_path, transform, out_dir)


if __name__ == "__main__":
    main()
