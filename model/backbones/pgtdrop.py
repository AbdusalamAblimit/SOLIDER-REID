"""Pose-Guided Token Dropping (PGTDrop) backbone.

Single-branch Swin Transformer that uses pose heatmaps to zero-mask occluded
tokens after a configurable stage, forcing later stages to learn from visible
regions only.  Visibility-weighted pooling produces the final feature vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .swin_transformer import SwinTransformer
from .pose_swin_transformer import MMPoseTopDownPredictor


class PoseGuidedTokenDrop(nn.Module):
    """Swin backbone with pose-guided token dropping."""

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
        # pose
        pose_predictor: Optional[nn.Module] = None,
        pose_detach: bool = True,
        n_keypoints: int = 17,
        use_visibility: bool = True,
        heatmap_norm: str = 'none',
        # pgtdrop
        drop_stage: int = 1,
        keep_ratio: float = 0.7,
        random_drop: float = 0.0,
        vis_pool: bool = True,
        reapply: bool = True,
        vis_threshold: float = 0.5,
        vis_hard: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.swin = SwinTransformer(
            pretrain_img_size=pretrain_img_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_cfg=init_cfg,
            semantic_weight=semantic_weight,
            **kwargs,
        )
        self._pretrained = pretrained
        self._convert_weights = convert_weights

        # Pose
        self.pose_predictor = pose_predictor
        self.pose_detach = pose_detach
        self.n_keypoints = n_keypoints
        self.use_visibility = use_visibility
        self.heatmap_norm = heatmap_norm

        # PGTDrop params
        self.drop_stage = drop_stage
        self.keep_ratio = keep_ratio
        self.random_drop = random_drop
        self.vis_pool = vis_pool
        self.reapply = reapply
        self.vis_threshold = vis_threshold
        self.vis_hard = vis_hard

        # Expose for make_model.py compatibility
        self.num_features = self.swin.num_features
        self.avgpool = self.swin.avgpool

        # Cache for pose outputs
        self._hm_fullres: Optional[torch.Tensor] = None
        self._vis: Optional[torch.Tensor] = None

    # -------------------- weight loading --------------------
    def init_weights(self, pretrained=None):
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
            print(f"[PGTDrop][swin_ckpt] loaded={len(loadable)} miss={len(missing)} unexp={len(unexpected)} from {path}")
        except Exception as e:
            print(f"[PGTDrop][swin_ckpt] remap failed: {e}; fallback convert_weights={self._convert_weights}")
            if self._convert_weights:
                self.swin.init_weights(path)
            else:
                self.swin.init_weights(pretrained=None)

    # -------------------- pose helpers --------------------
    def _get_pose(self, images: torch.Tensor):
        if self.pose_predictor is None:
            self._hm_fullres, self._vis = None, None
            return None, None
        if self.pose_detach:
            with torch.no_grad():
                hm, vis = self.pose_predictor(images)
            hm = hm.detach()
            vis = vis.detach() if vis is not None else None
        else:
            hm, vis = self.pose_predictor(images)
        self._hm_fullres = hm
        self._vis = vis
        return hm, vis

    # -------------------- token dropping core --------------------
    def _compute_token_vis(self, heatmaps, visibility, hw_shape):
        """Pose heatmaps -> per-token visibility score in [0, 1]."""
        if heatmaps is None:
            return None
        H, W = hw_shape
        hm = F.interpolate(heatmaps, size=(H, W), mode='bilinear', align_corners=False)
        if self.vis_hard:
            if visibility is not None and self.use_visibility:
                vis_binary = (visibility > self.vis_threshold).float()  # [B, 17]
                hm = hm * vis_binary.unsqueeze(-1).unsqueeze(-1)
        else:
            if visibility is not None and self.use_visibility:
                hm = hm * visibility.unsqueeze(-1).unsqueeze(-1).clamp_min(0.0)
        vis_map = hm.sum(dim=1)       # [B, H, W]
        vis_flat = vis_map.flatten(1)  # [B, N]
        vmin = vis_flat.amin(1, keepdim=True)
        vmax = vis_flat.amax(1, keepdim=True)
        token_vis = (vis_flat - vmin) / (vmax - vmin + 1e-6)  # [B, N] in [0, 1]
        return token_vis

    def _compute_mask(self, token_vis):
        """Keep top keep_ratio tokens, zero others."""
        if token_vis is None:
            return None
        B, N = token_vis.shape
        k = max(1, int(N * self.keep_ratio))
        # Threshold = k-th largest visibility value per sample
        threshold = token_vis.topk(k, dim=1).values[:, -1:]  # [B, 1]
        mask = (token_vis >= threshold).float()               # [B, N]
        # Safety: if pose failed (uniform vis), keep all tokens
        vis_range = token_vis.amax(1) - token_vis.amin(1)     # [B]
        no_signal = (vis_range < 0.05)                         # [B]
        mask[no_signal] = 1.0
        # Training augmentation: randomly drop extra visible tokens
        if self.training and self.random_drop > 0:
            rand_mask = (torch.rand(B, N, device=mask.device) > self.random_drop).float()
            mask = mask * rand_mask
        return mask  # [B, N]

    def _downsample_vis(self, token_vis, old_hw, new_hw):
        """Max-pool visibility after PatchMerging (2x2 -> 1)."""
        B = token_vis.shape[0]
        vis_2d = token_vis.view(B, 1, *old_hw)
        vis_2d = F.adaptive_max_pool2d(vis_2d, new_hw)
        return vis_2d.flatten(1)  # [B, N_new]

    def _vis_pool(self, tokens, token_vis):
        """Pool visible tokens weighted by visibility."""
        weights = token_vis.unsqueeze(-1)                        # [B, N, 1]
        feat = (tokens * weights).sum(1) / (weights.sum(1) + 1e-6)  # [B, C]
        return feat

    # -------------------- SOLIDER semantic conditioning --------------------
    def _original_semantic(self, x, stage_idx, semantic_weight):
        """Apply SOLIDER semantic conditioning (same as SwinTransformer.forward)."""
        if hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0 and semantic_weight is not None:
            sw = self.swin.semantic_embed_w[stage_idx](semantic_weight).unsqueeze(1)
            sb = self.swin.semantic_embed_b[stage_idx](semantic_weight).unsqueeze(1)
            x = x * self.swin.softplus(sw) + sb
        return x

    # -------------------- forward --------------------
    def forward(self, x, semantic_weight=None):
        # Get pose predictions
        heatmaps, visibility = self._get_pose(x)

        # Auto-generate semantic_weight
        if semantic_weight is None and hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin.semantic_weight
            w = torch.cat([w, 1 - w], dim=-1)
            semantic_weight = w.to(x.device)

        # Patch embed
        x_tokens, hw_shape = self.swin.patch_embed(x)
        if getattr(self.swin, 'use_abs_pos_embed', False):
            x_tokens = x_tokens + self.swin.absolute_pos_embed
        x_tokens = self.swin.drop_after_pos(x_tokens)

        token_vis = None
        pre_hw = None
        last_out = None
        last_out_hw = None

        for i, stage in enumerate(self.swin.stages):
            x_tokens, hw_shape, out, out_hw = stage(x_tokens, hw_shape)

            # SOLIDER semantic conditioning
            x_tokens = self._original_semantic(x_tokens, i, semantic_weight)

            if i == self.drop_stage and heatmaps is not None:
                # Compute token visibility and mask at drop_stage
                token_vis = self._compute_token_vis(heatmaps, visibility, hw_shape)
                mask = self._compute_mask(token_vis)
                if mask is not None:
                    x_tokens = x_tokens * mask.unsqueeze(-1)
                pre_hw = hw_shape
            elif i > self.drop_stage and token_vis is not None and self.reapply:
                # Re-apply mask after subsequent stages (downsample vis first)
                token_vis = self._downsample_vis(token_vis, pre_hw, hw_shape)
                mask = self._compute_mask(token_vis)
                if mask is not None:
                    x_tokens = x_tokens * mask.unsqueeze(-1)
                pre_hw = hw_shape

            last_out = out
            last_out_hw = out_hw

        # Final pooling
        if self.vis_pool and token_vis is not None and last_out is not None:
            # Compute vis at out_hw resolution for the last stage's output
            if last_out_hw != pre_hw:
                final_vis = self._downsample_vis(token_vis, pre_hw, last_out_hw)
            else:
                final_vis = token_vis
            final_mask = self._compute_mask(final_vis)
            if final_mask is not None:
                final_vis = final_vis * final_mask  # zero occluded vis scores
            global_feat = self._vis_pool(last_out, final_vis)
        else:
            # Standard avgpool fallback
            n_stages = len(self.swin.stages)
            last_idx = n_stages - 1
            if last_idx in self.swin.out_indices:
                norm_layer = getattr(self.swin, f'norm{last_idx}', None)
                if norm_layer is not None:
                    last_out = norm_layer(last_out)
            B, N, C = last_out.shape
            H, W = last_out_hw
            out_map = last_out.transpose(1, 2).contiguous().view(B, C, H, W)
            global_feat = torch.flatten(self.avgpool(out_map), 1)

        return {'global_feat': global_feat, 'local_feat': None}


# -------------------- Factory functions --------------------
def _build_pgtdrop(img_size=224, drop_rate=0.0, attn_drop_rate=0.0,
                   drop_path_rate=0.0, pretrained=None, convert_weights=False,
                   semantic_weight=0.0, **kwargs):
    """Common builder; callers set embed_dims / depths / num_heads."""
    # Read configs
    pose_cfg = None
    pgtdrop_cfg = None
    try:
        from config import cfg as _cfg
        pose_cfg = _cfg.MODEL.POSE
        pgtdrop_cfg = _cfg.MODEL.PGTDROP
    except Exception:
        pass

    # Build pose predictor
    pose_predictor = None
    if pose_cfg is not None and pose_cfg.get('ENABLE', False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pose_predictor = MMPoseTopDownPredictor(pose_cfg['CFG'], pose_cfg['CKPT'], device)

    # Extract PGTDrop params
    drop_stage = pgtdrop_cfg.get('DROP_STAGE', 1) if pgtdrop_cfg else 1
    keep_ratio = pgtdrop_cfg.get('KEEP_RATIO', 0.7) if pgtdrop_cfg else 0.7
    random_drop_rate = pgtdrop_cfg.get('RANDOM_DROP', 0.0) if pgtdrop_cfg else 0.0
    vis_pool = pgtdrop_cfg.get('VIS_POOL', True) if pgtdrop_cfg else True
    reapply = pgtdrop_cfg.get('REAPPLY', True) if pgtdrop_cfg else True
    vis_threshold = pgtdrop_cfg.get('VIS_THRESHOLD', 0.5) if pgtdrop_cfg else 0.5
    vis_hard = pgtdrop_cfg.get('VIS_HARD', True) if pgtdrop_cfg else True

    return PoseGuidedTokenDrop(
        pretrain_img_size=img_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained) if (pretrained and convert_weights) else None,
        pretrained=pretrained,
        convert_weights=convert_weights,
        semantic_weight=semantic_weight,
        pose_predictor=pose_predictor,
        pose_detach=pose_cfg.get('DETACH', True) if pose_cfg else True,
        n_keypoints=pose_cfg.get('N_KPTS', 17) if pose_cfg else 17,
        use_visibility=pose_cfg.get('USE_VIS', True) if pose_cfg else True,
        heatmap_norm=pose_cfg.get('HM_NORM', 'none') if pose_cfg else 'none',
        drop_stage=drop_stage,
        keep_ratio=keep_ratio,
        random_drop=random_drop_rate,
        vis_pool=vis_pool,
        reapply=reapply,
        vis_threshold=vis_threshold,
        vis_hard=vis_hard,
        **kwargs,
    )


def pgtdrop_base_patch4_window7_224(*args, **kwargs):
    kwargs.setdefault('embed_dims', 128)
    kwargs.setdefault('depths', (2, 2, 18, 2))
    kwargs.setdefault('num_heads', (4, 8, 16, 32))
    kwargs.setdefault('window_size', 7)
    return _build_pgtdrop(*args, **kwargs)


def pgtdrop_small_patch4_window7_224(*args, **kwargs):
    kwargs.setdefault('embed_dims', 96)
    kwargs.setdefault('depths', (2, 2, 18, 2))
    kwargs.setdefault('num_heads', (3, 6, 12, 24))
    kwargs.setdefault('window_size', 7)
    return _build_pgtdrop(*args, **kwargs)


def pgtdrop_tiny_patch4_window7_224(*args, **kwargs):
    kwargs.setdefault('embed_dims', 96)
    kwargs.setdefault('depths', (2, 2, 6, 2))
    kwargs.setdefault('num_heads', (3, 6, 12, 24))
    kwargs.setdefault('window_size', 7)
    return _build_pgtdrop(*args, **kwargs)
