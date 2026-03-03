"""PAMS: Part-Aware Multi-Scale ReID backbone.

Single-branch Swin + Multi-Scale Fusion + Learned Part Classifier + BPA supervision.
Pose predictor is only needed during training to supervise the part classifier.
At inference, the learned classifier predicts part attention without pose.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .swin_transformer import SwinTransformer

logger = logging.getLogger("transreid.pams")

# Reuse MMPose predictor from existing code
_HAS_MMPOSE = False
try:
    from .pose_swin_transformer import MMPoseTopDownPredictor
    _HAS_MMPOSE = True
except Exception:
    pass


# COCO keypoint -> body part groupings
COCO_PART_GROUPS = [
    [0, 1, 2, 3, 4],      # head (nose, eyes, ears)
    [5, 6, 11, 12],        # torso (shoulders, hips)
    [7, 8, 9, 10],         # arms (elbows, wrists)
    [13, 14],              # thighs (knees)
    [15, 16],              # calves (ankles)
]


class MultiScaleFusion(nn.Module):
    """Fuse multi-scale Swin stage outputs into a single spatial feature map."""

    def __init__(self, in_channels_list: List[int], out_channels: int, target_hw: Tuple[int, int]):
        super().__init__()
        self.target_hw = tuple(target_hw)
        total_channels = sum(in_channels_list)
        self.reduce = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, stage_outputs: List[Tuple[torch.Tensor, Tuple[int, int]]]) -> torch.Tensor:
        """
        Args:
            stage_outputs: list of (tokens [B, N, C], hw (H, W)) per stage
        Returns:
            fused: [B, out_channels, target_H, target_W]
        """
        resized = []
        for tokens, hw in stage_outputs:
            B, N, C = tokens.shape
            feat_2d = tokens.transpose(1, 2).view(B, C, hw[0], hw[1])
            feat_2d = F.interpolate(feat_2d, self.target_hw, mode='bilinear', align_corners=False)
            resized.append(feat_2d)
        fused = torch.cat(resized, dim=1)
        return self.reduce(fused)


class PartClassifier(nn.Module):
    """1x1 Conv classifier that assigns each spatial location to K parts + background."""

    def __init__(self, in_channels: int, n_parts: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.classifier = nn.Conv2d(in_channels, n_parts + 1, 1)  # K+1: parts + bg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, K+1, H, W]."""
        return self.classifier(self.bn(x))


def build_bpa_target(heatmaps: torch.Tensor, visibility: torch.Tensor,
                     target_hw: Tuple[int, int], vis_threshold: float = 0.5) -> torch.Tensor:
    """Build BPA pixel-level classification targets from pose heatmaps.

    Args:
        heatmaps: [B, 17, h, w] raw pose heatmaps
        visibility: [B, 17] keypoint visibility scores
        target_hw: (H, W) target spatial size
        vis_threshold: threshold for considering a keypoint visible

    Returns:
        targets: [B, H, W] integer class labels 0..K (0=background)
    """
    hm = F.interpolate(heatmaps, target_hw, mode='bilinear', align_corners=False)

    # Mask out invisible keypoints
    vis_mask = (visibility > vis_threshold).float()  # [B, 17]
    hm = hm * vis_mask.unsqueeze(-1).unsqueeze(-1)

    # Aggregate keypoints into body part groups
    part_maps = []
    for group in COCO_PART_GROUPS:
        part_hm = hm[:, group].max(dim=1)[0]  # [B, H, W]
        part_maps.append(part_hm)
    part_maps = torch.stack(part_maps, dim=1)  # [B, K, H, W]

    # Background = where no part is dominant
    bg = 1.0 - part_maps.max(dim=1)[0]  # [B, H, W]
    bg = bg.clamp(min=0.0)

    # Concat: [bg, part1, ..., partK]
    all_maps = torch.cat([bg.unsqueeze(1), part_maps], dim=1)  # [B, K+1, H, W]
    return all_maps.argmax(dim=1)  # [B, H, W]


def extract_part_features(spatial_feat: torch.Tensor, part_probs: torch.Tensor):
    """Extract foreground, per-part features and visibility from MSF spatial features.

    Args:
        spatial_feat: [B, C, H, W] (MSF output)
        part_probs:   [B, K+1, H, W] after softmax (channel 0 = background)

    Returns:
        fg_feat:     [B, C]
        part_feats:  [B, K, C]
        part_vis:    [B, K]
    """
    B, C, H, W = spatial_feat.shape

    # Part masks (exclude background channel 0)
    part_masks = part_probs[:, 1:]  # [B, K, H, W]

    # Foreground: max over parts at each location
    fg_mask = part_masks.max(dim=1)[0]  # [B, H, W]
    fg_sum = fg_mask.flatten(1).sum(1, keepdim=True).unsqueeze(1).clamp(min=1.0)  # [B, 1, 1]
    fg_feat = (spatial_feat * fg_mask.unsqueeze(1)).flatten(2).sum(2) / fg_sum.squeeze(2)  # [B, C]

    # Per-part weighted average pooling
    part_masks_exp = part_masks.unsqueeze(2)  # [B, K, 1, H, W]
    spatial_exp = spatial_feat.unsqueeze(1)    # [B, 1, C, H, W]
    weighted = (part_masks_exp * spatial_exp).flatten(3)  # [B, K, C, H*W]
    part_sums = weighted.sum(3)  # [B, K, C]
    # clamp(min=1.0): prevents amplification when a part has near-zero attention.
    # Worst case: small weighted sum / 1.0 = small feature (not huge garbage).
    mask_sums = part_masks.flatten(2).sum(2).unsqueeze(2).clamp(min=1.0)  # [B, K, 1]
    part_feats = part_sums / mask_sums  # [B, K, C]

    # Part visibility: max attention value per part (in [0, 1])
    part_vis = part_masks.flatten(2).max(dim=2)[0]  # [B, K]

    return fg_feat, part_feats, part_vis


class PartAwareMultiScale(nn.Module):
    """PAMS backbone: Swin + MSF + PartClassifier.

    Attributes:
        num_features: list of per-stage channel dimensions (from Swin)
        n_body_parts: number of body parts (K, excluding background)
        is_pams: True, used by make_model.py to detect PAMS mode
    """

    def __init__(
        self,
        *,
        pretrain_img_size=224,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_cfg=None,
        pretrained=None,
        convert_weights=False,
        semantic_weight=0.0,
        n_parts: int = 5,
        msf_target_hw: Tuple[int, int] = (24, 8),
        msf_out_dim: int = 768,
        vis_threshold: float = 0.5,
        pose_predictor: Optional[nn.Module] = None,
        with_cp: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.is_pams = True
        self.n_body_parts = n_parts
        self.vis_threshold = vis_threshold
        self.msf_target_hw = tuple(msf_target_hw)

        # Swin backbone
        self.swin = SwinTransformer(
            pretrain_img_size=pretrain_img_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_cfg=init_cfg,
            semantic_weight=semantic_weight,
            with_cp=with_cp,
            **kwargs,
        )
        self._pretrained = pretrained
        self._convert_weights = convert_weights

        self.num_features = self.swin.num_features  # e.g. [96, 192, 384, 768]
        # Global feat dim = stage 3 output; part feat dim = MSF output
        self.global_feat_dim = self.num_features[-1]  # 768 for tiny
        self.part_feat_dim = msf_out_dim              # 768 default

        # Multi-Scale Fusion
        self.msf = MultiScaleFusion(
            in_channels_list=self.num_features,
            out_channels=msf_out_dim,
            target_hw=msf_target_hw,
        )

        # Part Classifier
        self.part_classifier = PartClassifier(msf_out_dim, n_parts)

        # Pose predictor (frozen, train only)
        self.pose_predictor = pose_predictor

        logger.info(f"Swin stages: {len(self.swin.stages)}, num_features={self.num_features}")
        logger.info(f"global_feat_dim={self.global_feat_dim}, part_feat_dim={self.part_feat_dim}")
        logger.info(f"MSF target_hw={self.msf_target_hw}, n_parts={n_parts}")
        logger.info(f"Pose predictor: {'enabled' if pose_predictor is not None else 'disabled'}")

    def init_weights(self, pretrained=None):
        """Load pretrained weights into Swin backbone."""
        path = pretrained if pretrained is not None else self._pretrained
        if not path:
            self.swin.init_weights(pretrained=None)
            return

        def _pick_state_dict(obj):
            if isinstance(obj, dict):
                for k in ['student', 'state_dict', 'model', 'teacher', 'ema', 'module']:
                    if k in obj and isinstance(obj[k], dict):
                        return obj[k]
                return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            return obj

        def _rename(k: str) -> str:
            if k.startswith('module.'):
                k = k[len('module.'):]
            if k.startswith('backbone.'):
                k = k[len('backbone.'):]
            if k.startswith('layers.'):
                k = 'stages.' + k[len('layers.'):]
            return k

        try:
            obj = torch.load(path, map_location='cpu')
            sd_raw = _pick_state_dict(obj)
            sd = {}
            for k, v in sd_raw.items():
                if not isinstance(v, torch.Tensor):
                    continue
                rk = _rename(k)
                if rk.startswith('head.') or rk.startswith('cls_head.') or rk.startswith('neck.'):
                    continue
                sd[rk] = v

            msd = self.swin.state_dict()
            loadable = {k: v for k, v in sd.items() if (k in msd and msd[k].shape == v.shape)}
            missing = [k for k in msd.keys() if k not in loadable]
            unexpected = [k for k in sd.keys() if k not in msd]
            self.swin.load_state_dict(loadable, strict=False)
            logger.info(f"[swin_ckpt] loaded={len(loadable)} miss={len(missing)} unexp={len(unexpected)} from {path}")
        except Exception as e:
            logger.warning(f"[swin_ckpt] remap failed: {e}; fallback convert_weights={self._convert_weights}")
            if self._convert_weights:
                self.swin.init_weights(path)
            else:
                self.swin.init_weights(pretrained=None)

    def _original_semantic(self, x: torch.Tensor):
        """Generate SOLIDER semantic conditioning weights."""
        if hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1, device=x.device) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], dim=-1)
            return w
        return None

    def forward(self, x, semantic_weight=None) -> Dict[str, torch.Tensor]:
        B = x.shape[0]

        # --- Pose prediction (train only) ---
        heatmaps = None
        visibility = None
        if self.training and self.pose_predictor is not None:
            with torch.no_grad():
                heatmaps, visibility = self.pose_predictor(x)
            heatmaps = heatmaps.detach()
            visibility = visibility.detach() if visibility is not None else None

        # --- Swin backbone: collect all stage outputs ---
        if semantic_weight is None:
            semantic_weight = self._original_semantic(x)

        x_tokens, hw_shape = self.swin.patch_embed(x)
        if getattr(self.swin, 'use_abs_pos_embed', False):
            x_tokens = x_tokens + self.swin.absolute_pos_embed
        x_tokens = self.swin.drop_after_pos(x_tokens)

        stage_outputs = []  # list of (tokens [B, N, C], (H, W))
        last_out_map = None  # stage 3 feature map for global feature
        for i, stage in enumerate(self.swin.stages):
            x_tokens, hw_shape, out, out_hw_shape = stage(x_tokens, hw_shape)

            # SOLIDER semantic conditioning
            if semantic_weight is not None and hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x_tokens = x_tokens * self.swin.softplus(sw) + sb

            # Collect normalized output for this stage
            if i in self.swin.out_indices:
                norm_layer = getattr(self.swin, f'norm{i}', None)
                out_normed = norm_layer(out) if norm_layer is not None else out
                stage_outputs.append((out_normed, out_hw_shape))
                # Keep last stage output for global feature (preserves pretrained quality)
                B_out, N_out, C_out = out_normed.shape
                last_out_map = out_normed.transpose(1, 2).view(B_out, C_out, *out_hw_shape)

        # --- Global feature: directly from stage 3 avg pool (pretrained quality) ---
        global_feat = torch.flatten(self.swin.avgpool(last_out_map), 1)  # [B, D_swin]

        # --- Multi-Scale Fusion (for part classification only) ---
        spatial_feat = self.msf(stage_outputs)  # [B, D_msf, H, W]

        # --- Part Classification ---
        part_logits = self.part_classifier(spatial_feat)  # [B, K+1, H, W]
        part_probs = F.softmax(part_logits, dim=1)        # [B, K+1, H, W]

        # --- Part Feature Extraction (fg + parts from MSF, global from stage 3) ---
        fg_feat, part_feats, part_vis = extract_part_features(spatial_feat, part_probs)

        result = {
            'global_feat': global_feat,       # [B, D_swin] from stage 3
            'foreground_feat': fg_feat,        # [B, D_msf] from MSF
            'part_feats': part_feats,          # [B, K, D_msf]
            'part_vis': part_vis,              # [B, K]
        }

        # --- BPA supervision targets (train only) ---
        if self.training and heatmaps is not None and visibility is not None:
            bpa_targets = build_bpa_target(heatmaps, visibility, self.msf_target_hw, self.vis_threshold)
            result['bpa_logits'] = part_logits    # [B, K+1, H, W]
            result['bpa_targets'] = bpa_targets   # [B, H, W]

        return result


# -------------------- Factory constructors --------------------

def _build_pams(img_size=224, drop_rate=0.0, attn_drop_rate=0.0,
                drop_path_rate=0.0, pretrained=None, convert_weights=False,
                semantic_weight=0.0, with_cp=False, **kwargs):
    """Shared builder; model-size kwargs (embed_dims, depths, num_heads) come from caller."""
    # Get PAMS config
    pams_cfg = None
    pose_cfg = None
    try:
        from config import cfg as _cfg
        pams_cfg = _cfg.MODEL.PAMS
        pose_cfg = _cfg.MODEL.POSE
    except Exception:
        pass

    n_parts = pams_cfg.N_PARTS if pams_cfg else 5
    msf_target_hw = tuple(pams_cfg.MSF_TARGET_HW) if pams_cfg else (24, 8)
    msf_out_dim = pams_cfg.MSF_OUT_DIM if pams_cfg else 768
    vis_threshold = pams_cfg.VIS_THRESHOLD if pams_cfg else 0.5

    # Pose predictor (optional, train-only)
    pose_predictor = None
    if pose_cfg is not None and pose_cfg.get('ENABLE', False) and _HAS_MMPOSE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pose_predictor = MMPoseTopDownPredictor(pose_cfg['CFG'], pose_cfg['CKPT'], device)

    return PartAwareMultiScale(
        pretrain_img_size=img_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained) if (pretrained and convert_weights) else None,
        pretrained=pretrained,
        convert_weights=convert_weights,
        semantic_weight=semantic_weight,
        n_parts=n_parts,
        msf_target_hw=msf_target_hw,
        msf_out_dim=msf_out_dim,
        vis_threshold=vis_threshold,
        pose_predictor=pose_predictor,
        with_cp=with_cp,
        **kwargs,
    )


def pams_base_patch4_window7_224(**kwargs):
    kwargs.setdefault('embed_dims', 128)
    kwargs.setdefault('depths', (2, 2, 18, 2))
    kwargs.setdefault('num_heads', (4, 8, 16, 32))
    kwargs.setdefault('window_size', 7)
    return _build_pams(**kwargs)


def pams_small_patch4_window7_224(**kwargs):
    kwargs.setdefault('embed_dims', 96)
    kwargs.setdefault('depths', (2, 2, 18, 2))
    kwargs.setdefault('num_heads', (3, 6, 12, 24))
    kwargs.setdefault('window_size', 7)
    return _build_pams(**kwargs)


def pams_tiny_patch4_window7_224(**kwargs):
    kwargs.setdefault('embed_dims', 96)
    kwargs.setdefault('depths', (2, 2, 6, 2))
    kwargs.setdefault('num_heads', (3, 6, 12, 24))
    kwargs.setdefault('window_size', 7)
    return _build_pams(**kwargs)
