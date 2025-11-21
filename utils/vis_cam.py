import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import cfg as global_cfg
from model import make_model


@dataclass
class CamResult:
    cam: torch.Tensor
    attention: Optional[torch.Tensor]


class WindowAttentionHook:
    """Register hooks on a ``ShiftWindowMSA`` module for Grad-CAM.

    The hook caches the forward activations, corresponding gradients and the
    spatial shape so that Grad-CAM maps can be reconstructed in the original
    image space. Attention weights from ``WindowMSA`` are also recorded for
    visualization.
    """

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


def _default_transform(resize_hw: Tuple[int, int]):
    return transforms.Compose(
        [
            transforms.Resize(resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=global_cfg.INPUT.PIXEL_MEAN, std=global_cfg.INPUT.PIXEL_STD),
        ]
    )


def _load_image(path: str, resize_hw: Tuple[int, int]):
    image = Image.open(path).convert("RGB")
    tensor = _default_transform(resize_hw)(image)
    return image, tensor


def _overlay_heatmap(image: Image.Image, heatmap: torch.Tensor) -> Image.Image:
    np_img = np.array(image).astype(np.float32) / 255.0
    heat = heatmap.detach().cpu().numpy()
    heat = np.clip(heat, 0.0, 1.0)
    color = cm.get_cmap("jet")(heat)[..., :3]
    blended = 0.6 * np_img + 0.4 * color
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def _pick_target_module(backbone: torch.nn.Module, branch: str, target_stage: int):
    if hasattr(backbone, "swin") and branch == "local" and getattr(backbone, "local_stages", None):
        stage_list = backbone.local_stages
    elif hasattr(backbone, "swin"):
        stage_list = backbone.swin.stages
    else:
        stage_list = backbone.stages

    if target_stage < 0:
        target_stage = len(stage_list) + target_stage
    target_stage = max(0, min(target_stage, len(stage_list) - 1))
    stage = stage_list[target_stage]
    return stage.blocks[-1].attn


def _select_scalar(output: torch.Tensor, classifier: Optional[torch.nn.Module], cls_idx: Optional[int]):
    if classifier is not None and cls_idx is not None:
        logits = classifier(output)
        return logits[:, cls_idx].sum()
    return output.norm(p=2, dim=1).sum()


def _fetch_backbone(model: torch.nn.Module):
    if hasattr(model, "base"):
        return model.base
    if hasattr(model, "backbone"):
        return model.backbone
    return model


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


def _visualize_single(model: torch.nn.Module, image: Image.Image, tensor: torch.Tensor, branch: str,
                      target_stage: int, cls_idx: Optional[int], device: torch.device) -> List[Image.Image]:
    backbone = _fetch_backbone(model)
    module = _pick_target_module(backbone, branch, target_stage)
    hook = WindowAttentionHook(module)
    tensor = tensor.unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    model.eval()
    with torch.enable_grad():
        outputs, featmaps = model(tensor)
        if isinstance(outputs, dict):
            if branch not in outputs:
                feat_vec = outputs.get("global")
                classifier = getattr(model, "classifier_global", None)
            else:
                feat_vec = outputs[branch]
                classifier = getattr(model, f"classifier_{branch}", None)
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
        results.append(_overlay_heatmap(image, pose_map[0, 0]))

    return results


def load_cfg(cfg_path: str, opts: Optional[Iterable[str]] = None):
    cfg = global_cfg.clone()
    if cfg_path:
        cfg.merge_from_file(cfg_path)
    if opts:
        cfg.merge_from_list(list(opts))
    cfg.freeze()
    return cfg


def build_model_from_cfg(cfg, num_classes: int, device: torch.device):
    model = make_model(cfg, num_class=num_classes, camera_num=0, view_num=0, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT:
        model.load_param(cfg.TEST.WEIGHT)
    model.to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for Swin/Pose-Swin")
    parser.add_argument("image", help="path to an input image")
    parser.add_argument("--baseline-config", required=True, help="config file for baseline model")
    parser.add_argument("--pose-config", default=None, help="config file for pose model")
    parser.add_argument("--num-classes", type=int, default=751, help="number of ID classes for classifier heads")
    parser.add_argument("--target-stage", type=int, default=-1, help="stage index to attach Grad-CAM (default: last)")
    parser.add_argument("--branch", choices=["global", "local", "concat"], default="global", help="feature branch")
    parser.add_argument("--class-idx", type=int, default=None, help="target class for CAM; defaults to feature norm")
    parser.add_argument("--output", default="cam_vis.png", help="output filename")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, help="additional baseline config options")
    parser.add_argument("--pose-opts", nargs=argparse.REMAINDER, help="additional pose config options")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_cfg = load_cfg(args.baseline_config, args.opts)
    pose_cfg = load_cfg(args.pose_config, args.pose_opts) if args.pose_config else None

    image, tensor = _load_image(args.image, tuple(baseline_cfg.INPUT.SIZE_TEST))

    baseline_model = build_model_from_cfg(baseline_cfg, args.num_classes, device)
    outputs = []
    outputs.extend(_visualize_single(baseline_model, image, tensor, args.branch, args.target_stage, args.class_idx, device))

    if pose_cfg is not None:
        pose_model = build_model_from_cfg(pose_cfg, args.num_classes, device)
        outputs.extend(_visualize_single(pose_model, image, tensor, args.branch, args.target_stage, args.class_idx, device))

    widths, heights = zip(*(img.size for img in outputs))
    total_w = sum(widths)
    max_h = max(heights)
    canvas = Image.new("RGB", (total_w, max_h))
    offset = 0
    for img in outputs:
        canvas.paste(img, (offset, 0))
        offset += img.size[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    canvas.save(args.output)
    print(f"Saved CAM visualization to {args.output}")


if __name__ == "__main__":
    main()
