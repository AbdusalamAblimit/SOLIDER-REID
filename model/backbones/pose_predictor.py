"""Standalone ViTPose predictor with visibility prediction.

This module wraps an MMPose ViTPose-Huge model with VisPredictHead to provide:
  - Heatmaps: [B, 17, h, w] per-keypoint spatial heatmaps
  - Visibility: [B, 17] per-keypoint visibility scores (BCELoss-supervised)

The visibility vector semantics: visibility=0 means the keypoint is OCCLUDED
(not "undetectable"), even if the pose model can infer position from context.

Usage:
    predictor = MMPoseTopDownPredictor(cfg_path, ckpt_path, device)
    heatmaps, visibility = predictor(images)  # images: SOLIDER-normalized
"""

import logging
import torch
import torch.nn as nn
from typing import Tuple

logger = logging.getLogger("transreid.pose")

_HAS_MMPOSE = False
try:
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmpose.registry import MODELS as MMP_MODELS
    from mmengine.registry import init_default_scope, DefaultScope
    _HAS_MMPOSE = True
except Exception:
    _HAS_MMPOSE = False


# COCO 17 keypoint -> 5 body part groupings
COCO_PART_GROUPS = [
    [0, 1, 2, 3, 4],      # head (nose, eyes, ears)
    [5, 6, 11, 12],        # torso (shoulders, hips)
    [7, 8, 9, 10],         # arms (elbows, wrists)
    [13, 14],              # thighs (knees)
    [15, 16],              # calves (ankles)
]


class MMPoseTopDownPredictor(nn.Module):
    """Return heatmaps (B,K,h,w) and visibility (B,K) from ViTPose."""

    def __init__(self, cfg_path: str, ckpt_path: str, device: torch.device):
        super().__init__()
        assert _HAS_MMPOSE, "MMPose is required for pose prediction"

        # Ensure registry scope
        try:
            cur = DefaultScope.get_current_instance()
            if (cur is None) or (cur.scope != 'mmpose'):
                init_default_scope('mmpose')
        except Exception:
            try:
                init_default_scope('mmpose')
            except Exception:
                pass

        cfg = Config.fromfile(cfg_path)
        cfg.model.setdefault('data_preprocessor', None)
        cfg.model['data_preprocessor'] = None

        self.model = MMP_MODELS.build(cfg.model).to(device).eval()

        # Shape-strict partial loading
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict):
            for k in ['state_dict', 'model', 'state_dict_ema', 'ema', 'module']:
                if k in ckpt and isinstance(ckpt[k], dict):
                    ckpt = ckpt[k]
                    break

        def _strip(k):
            for p in ('model.', 'module.'):
                if k.startswith(p):
                    k = k[len(p):]
            if k.startswith('keypoint_head.'):
                k = 'head.' + k[len('keypoint_head.'):]
            return k

        ckpt = {_strip(k): v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        msd = self.model.state_dict()
        loadable = {k: v for k, v in ckpt.items() if (k in msd and msd[k].shape == v.shape)}
        missing = [k for k in msd.keys() if k not in loadable]
        unexpected = [k for k in ckpt.keys() if k not in msd]
        self.model.load_state_dict(loadable, strict=False)
        logger.info(f"[pose_ckpt] loaded={len(loadable)} missing={len(missing)} unexpected={len(unexpected)}")
        assert len(loadable) > 0, "Pose ckpt didn't match any weights."

        # MMPose expects 0..255, then subtract mean / divide std
        self.register_buffer('pose_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('pose_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def _forward_impl(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run pose model on SOLIDER-normalized images."""
        # De-normalize from SOLIDER format to 0..255
        try:
            from config import cfg as _cfg
            mean = images.new_tensor(_cfg.INPUT.PIXEL_MEAN).view(1, 3, 1, 1)
            std = images.new_tensor(_cfg.INPUT.PIXEL_STD).view(1, 3, 1, 1)
        except Exception:
            mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        img = images * std + mean       # 0..1
        img = img * 255.0               # 0..255
        img = (img - self.pose_mean.to(img.device)) / self.pose_std.to(img.device)

        if hasattr(self.model, 'backbone') and hasattr(self.model, 'head'):
            feat = self.model.backbone(img)
            out = self.model.head(feat)
        elif hasattr(self.model, 'extract_feat') and hasattr(self.model, 'head'):
            feat = self.model.extract_feat(img)
            out = self.model.head(feat)
        else:
            out = self.model(img, mode='tensor')

        if isinstance(out, (list, tuple)):
            heatmap, visibility = out[0], (out[1] if len(out) > 1 else None)
        else:
            heatmap, visibility = out, None

        if visibility is None:
            B, K, h, w = heatmap.shape
            visibility = heatmap.view(B, K, -1).amax(dim=-1)
            visibility = (visibility - visibility.min()) / (visibility.max() - visibility.min() + 1e-6)

        return heatmap, visibility

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            return self._forward_impl(images)
        with torch.no_grad():
            return self._forward_impl(images)
