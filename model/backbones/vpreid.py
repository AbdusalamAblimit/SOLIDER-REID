"""VPReID: Visibility-aware Pose-guided ReID backbone.

Architecture:
    Input image → Swin-Tiny (unchanged) → global_feat + stage3 spatial features
                → ViTPose (frozen)       → heatmaps + visibility
                → PosePartHead           → part_feats + part_vis + fg_feat

Key design:
- Backbone is completely untouched (no dual branch, no extra stages)
- PosePartHead uses pose heatmaps as soft attention on spatial features
- No learnable parameters in PosePartHead — gradients flow directly to backbone
- ViTPose is frozen during forward, only used for keypoint/visibility extraction
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .swin_transformer import SwinTransformer

logger = logging.getLogger("transreid.vpreid")

# Try to import pose predictor
_HAS_POSE = False
try:
    from .pose_predictor import MMPoseTopDownPredictor
    _HAS_POSE = True
except Exception:
    pass


# COCO 17 keypoint -> 5 body part groups
COCO_PART_GROUPS = [
    [0, 1, 2, 3, 4],      # head (nose, eyes, ears)
    [5, 6, 11, 12],        # torso (shoulders, hips)
    [7, 8, 9, 10],         # arms (elbows, wrists)
    [13, 14],              # thighs (knees)
    [15, 16],              # calves (ankles)
]


class PosePartHead(nn.Module):
    """Extracts part features from spatial features using pose heatmap guidance.

    No learnable parameters — pure geometric operation using pose information.
    Gradients flow from per-part losses through the attention-weighted pooling
    back to the backbone, teaching it to produce part-discriminative features.
    """

    def __init__(self, n_parts: int = 5, temp: float = 0.1):
        super().__init__()
        self.n_parts = n_parts
        self.temp = temp
        self.part_groups = COCO_PART_GROUPS[:n_parts]

    def forward(
        self,
        feat_map: torch.Tensor,
        heatmaps: torch.Tensor,
        visibility: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_map: [B, C, H, W] stage3 features (e.g. [B, 768, 12, 4])
            heatmaps: [B, 17, h, w] from ViTPose
            visibility: [B, 17] keypoint visibility scores (0=occluded, 1=visible)

        Returns:
            part_feats: [B, K, C] per-part features
            part_vis: [B, K] per-part visibility scores
            fg_feat: [B, C] visibility-weighted foreground feature
        """
        B, C, H, W = feat_map.shape

        # Resize heatmaps to match feature map spatial dims
        hm = F.interpolate(
            heatmaps.float(), (H, W), mode='bilinear', align_corners=False
        )  # [B, 17, H, W]

        # Soft-mask by continuous visibility scores
        vis = visibility.float().unsqueeze(-1).unsqueeze(-1)  # [B, 17, 1, 1]
        hm = hm * vis

        # Build part-level attention masks
        part_masks = []
        part_vis_list = []
        for group in self.part_groups:
            # Max heatmap activation across keypoints in group
            pmask = hm[:, group].max(dim=1)[0]  # [B, H, W]
            part_masks.append(pmask)
            # Part visibility = max visibility of constituent keypoints
            pvis = visibility[:, group].float().max(dim=1)[0]  # [B]
            part_vis_list.append(pvis)

        part_masks = torch.stack(part_masks, dim=1)  # [B, K, H, W]
        part_vis = torch.stack(part_vis_list, dim=1)  # [B, K]

        # Normalize masks via temperature-scaled softmax → attention weights
        flat = part_masks.view(B, self.n_parts, -1)  # [B, K, H*W]
        # Clamp for numerical stability in float16
        flat_clamped = (flat / self.temp).clamp(-20.0, 20.0)
        attn = F.softmax(flat_clamped.float(), dim=-1)  # force float32 for softmax
        attn = attn.to(feat_map.dtype)
        attn = attn.view(B, self.n_parts, H, W)

        # Weighted pooling for each part: [B, K, C]
        feat_exp = feat_map.unsqueeze(1)     # [B, 1, C, H, W]
        attn_exp = attn.unsqueeze(2)         # [B, K, 1, H, W]
        part_feats = (feat_exp * attn_exp).sum(dim=[3, 4])  # [B, K, C]

        # Foreground: visibility-weighted part aggregation
        vis_w = F.softmax(part_vis.float() + 1e-8, dim=1)  # [B, K]
        vis_w = vis_w.to(part_feats.dtype).unsqueeze(2)     # [B, K, 1]
        fg_feat = (part_feats * vis_w).sum(dim=1)           # [B, C]

        return part_feats, part_vis, fg_feat


class VPReIDSwin(nn.Module):
    """VPReID backbone: clean Swin-Tiny + frozen ViTPose + PosePartHead.

    Attributes:
        is_vpreid: True — used by make_model.py to detect VPReID path
        n_parts: number of body part groups
        global_feat_dim: dimension of global features (768 for Swin-Tiny)
        part_feat_dim: dimension of part features (same as global)
    """

    def __init__(
        self,
        base_swin: SwinTransformer,
        pose_cfg: str = '',
        pose_ckpt: str = '',
        n_parts: int = 5,
        part_temp: float = 0.1,
        vis_threshold: float = 0.5,
    ):
        super().__init__()
        self.base = base_swin
        self.n_parts = n_parts
        self.vis_threshold = vis_threshold

        # Flags for make_model.py detection
        self.is_vpreid = True
        self.n_body_parts = n_parts
        self.global_feat_dim = base_swin.num_features[-1]  # 768
        self.part_feat_dim = self.global_feat_dim
        # Expose num_features for build_transformer compatibility
        self.num_features = base_swin.num_features

        # Part head (no learnable params)
        self.part_head = PosePartHead(n_parts=n_parts, temp=part_temp)

        # Pose predictor (frozen)
        self.pose_predictor = None
        if pose_cfg and pose_ckpt and _HAS_POSE:
            import os
            if os.path.exists(pose_cfg) and os.path.exists(pose_ckpt):
                self.pose_predictor = MMPoseTopDownPredictor(
                    pose_cfg, pose_ckpt, torch.device('cuda')
                )
                # Freeze all pose parameters
                for p in self.pose_predictor.parameters():
                    p.requires_grad = False
                logger.info("VPReID: ViTPose predictor loaded and frozen")
            else:
                logger.warning(
                    f"VPReID: Pose files not found (cfg={pose_cfg}, ckpt={pose_ckpt}). "
                    "Using dummy heatmaps."
                )
        else:
            if not _HAS_POSE:
                logger.warning("VPReID: MMPose not available. Using dummy heatmaps.")
            else:
                logger.warning("VPReID: Pose config/ckpt not specified. Using dummy heatmaps.")

    def _get_pose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get heatmaps and visibility from pose predictor or generate dummy."""
        B = x.shape[0]
        if self.pose_predictor is not None:
            with torch.no_grad():
                heatmaps, visibility = self.pose_predictor(x)
            return heatmaps.detach(), visibility.detach()
        else:
            # Dummy: uniform heatmaps, full visibility
            H_hm, W_hm = 64, 48
            heatmaps = torch.ones(B, 17, H_hm, W_hm, device=x.device, dtype=x.dtype) * 0.5
            visibility = torch.ones(B, 17, device=x.device, dtype=x.dtype)
            return heatmaps, visibility

    def init_weights(self, pretrained=None):
        """Delegate to base Swin for weight initialization."""
        if pretrained and hasattr(self.base, 'init_weights'):
            self.base.init_weights(pretrained)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                global_feat: [B, D] from avgpool of stage3
                part_feats: [B, K, D] per-part features
                part_vis: [B, K] per-part visibility scores
                foreground_feat: [B, D] visibility-weighted aggregation
        """
        # Swin backbone features
        global_feat, outs = self.base(x)
        feat_map = outs[-1]  # [B, C, H, W] = [B, 768, 12, 4]

        # Pose prediction (frozen, detached)
        heatmaps, visibility = self._get_pose(x)

        # Part feature extraction
        part_feats, part_vis, fg_feat = self.part_head(feat_map, heatmaps, visibility)

        return {
            'global_feat': global_feat,
            'part_feats': part_feats,       # [B, K, D]
            'part_vis': part_vis,            # [B, K]
            'foreground_feat': fg_feat,      # [B, D]
        }


def _build_vpreid(variant_fn, img_size, drop_path_rate, drop_rate, attn_drop_rate,
                  pretrained, convert_weights, semantic_weight, with_cp=False,
                  pose_cfg='', pose_ckpt='', n_parts=5, part_temp=0.1,
                  vis_threshold=0.5, **kwargs):
    """Factory function to build VPReIDSwin."""
    base = variant_fn(
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        pretrained=pretrained,
        convert_weights=convert_weights,
        semantic_weight=semantic_weight,
        with_cp=with_cp,
    )
    return VPReIDSwin(
        base_swin=base,
        pose_cfg=pose_cfg,
        pose_ckpt=pose_ckpt,
        n_parts=n_parts,
        part_temp=part_temp,
        vis_threshold=vis_threshold,
    )


def vpreid_tiny_patch4_window7_224(img_size=224, drop_rate=0.0, attn_drop_rate=0.0,
                                    drop_path_rate=0., **kwargs):
    from .swin_transformer import swin_tiny_patch4_window7_224 as _tiny
    return _build_vpreid(
        _tiny, img_size, drop_path_rate, drop_rate, attn_drop_rate, **kwargs
    )


def vpreid_small_patch4_window7_224(img_size=224, drop_rate=0.0, attn_drop_rate=0.0,
                                     drop_path_rate=0., **kwargs):
    from .swin_transformer import swin_small_patch4_window7_224 as _small
    return _build_vpreid(
        _small, img_size, drop_path_rate, drop_rate, attn_drop_rate, **kwargs
    )


def vpreid_base_patch4_window7_224(img_size=224, drop_rate=0.0, attn_drop_rate=0.0,
                                    drop_path_rate=0., **kwargs):
    from .swin_transformer import swin_base_patch4_window7_224 as _base
    return _build_vpreid(
        _base, img_size, drop_path_rate, drop_rate, attn_drop_rate, **kwargs
    )
