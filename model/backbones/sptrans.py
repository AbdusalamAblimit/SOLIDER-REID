"""SPTrans: Semantic-Pose Joint Conditioning + Part-Aware Routing.

Dual-branch Swin Transformer that replaces the fixed pose gating of
PoseSwinCompose with two novel mechanisms:
  1. Semantic-Pose Joint Conditioning -- spatially adaptive semantic weight
     driven by per-token pose confidence.
  2. Part-Aware Routing -- pose-heatmap-guided per-part feature pooling
     instead of global average pooling on the local branch.
"""

import copy
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .pose_swin_transformer import MMPoseTopDownPredictor

# COCO 17 keypoints -> 5 body-part groups
COCO_PART_GROUPS = {
    'head':       [0, 1, 2, 3, 4],        # nose, eyes, ears
    'upper_body': [5, 6, 11, 12],          # shoulders, hips
    'left_arm':   [7, 9],                  # left elbow, wrist
    'right_arm':  [8, 10],                 # right elbow, wrist
    'legs':       [13, 14, 15, 16],        # knees, ankles
}


class PoseRoutedMoE(nn.Module):
    """Lightweight MoE: shared bottleneck + per-part linear projections.

    Inserted after each local-branch stage.  Zero-initialised expert_up so
    the residual connection makes this an identity at init.
    """

    def __init__(self, dim: int, n_parts: int = 5, bottleneck: int = 64,
                 gate_init: float = 0.1):
        super().__init__()
        self.n_parts = n_parts
        self.dim = dim
        self.shared_down = nn.Linear(dim, bottleneck)
        self.expert_up = nn.Linear(bottleneck, dim * n_parts)
        self.gate_scale = nn.Parameter(torch.tensor(gate_init))
        # Zero-init expert_up so initial output is zero → identity residual
        nn.init.zeros_(self.expert_up.weight)
        nn.init.zeros_(self.expert_up.bias)

    def forward(self, x_tokens: torch.Tensor,
                part_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_tokens:     [B, N, C]
            part_weights: [B, N, K]  (soft part assignment per token)
        Returns:
            x_tokens + gate_scale * routed  [B, N, C]
        """
        B, N, C = x_tokens.shape
        K = self.n_parts
        h = F.gelu(self.shared_down(x_tokens))          # [B, N, bottleneck]
        expert_out = self.expert_up(h)                   # [B, N, K*C]
        expert_out = expert_out.view(B, N, K, C)         # [B, N, K, C]
        # Weighted sum over experts using part assignment
        routed = (expert_out * part_weights.unsqueeze(-1)).sum(dim=2)  # [B, N, C]
        return x_tokens + self.gate_scale * routed


class PartExpertHead(nn.Module):
    """Per-part mask pooling + per-part expert FFN → visibility-weighted average.

    Replaces the flat concat part pooling of SPTrans v1.  Output is [B, D]
    (same dim as global branch) instead of [B, K*D].
    """

    def __init__(self, dim: int, n_parts: int = 5, bottleneck: int = 128):
        super().__init__()
        self.n_parts = n_parts
        self.part_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, bottleneck),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, dim),
            )
            for _ in range(n_parts)
        ])

    def forward(self, feat_map: torch.Tensor, part_masks: torch.Tensor,
                part_vis: torch.Tensor):
        """
        Args:
            feat_map:   [B, C, H, W]
            part_masks: [B, K, H, W]  (softmax-normalised)
            part_vis:   [B, K]
        Returns:
            local_feat: [B, C=D]
            part_feats: [B, K, D]  (for per-part loss)
        """
        B, C, H, W = feat_map.shape
        K = self.n_parts

        # 1. Mask-weighted pooling → [B, K, C]
        feat_expand = feat_map.unsqueeze(1)                  # [B, 1, C, H, W]
        masks_expand = part_masks.unsqueeze(2)               # [B, K, 1, H, W]
        weighted = feat_expand * masks_expand                # [B, K, C, H, W]
        part_feats = weighted.sum(dim=(-2, -1))              # [B, K, C]
        mask_sum = part_masks.sum(dim=(-2, -1)).unsqueeze(2).clamp_min(1e-6)
        part_feats = part_feats / mask_sum                   # [B, K, C]

        # 2. Per-part expert FFN (with residual)
        refined = []
        for k, expert in enumerate(self.part_experts):
            pk = part_feats[:, k]                            # [B, C]
            refined.append(pk + expert(pk))
        part_feats = torch.stack(refined, dim=1)             # [B, K, C]

        # 3. Visibility-weighted average → [B, C=768]
        weights = F.softmax(part_vis, dim=1).unsqueeze(-1)   # [B, K, 1]
        local_feat = (part_feats * weights).sum(dim=1)       # [B, D]

        return local_feat, part_feats


def _compute_part_masks(hm: torch.Tensor, part_groups: List[List[int]],
                        part_temp: float) -> torch.Tensor:
    """Compute softmax-normalised part masks from heatmaps.

    Args:
        hm:           [B, 17, H, W] (visibility-weighted heatmaps)
        part_groups:  list of K lists of keypoint indices
        part_temp:    softmax temperature
    Returns:
        part_masks:   [B, K, H, W]
    """
    masks = []
    for kpts in part_groups:
        masks.append(hm[:, kpts, :, :].max(dim=1)[0])       # [B, H, W]
    part_masks = torch.stack(masks, dim=1)                    # [B, K, H, W]
    part_masks = F.softmax(part_masks / part_temp, dim=1)
    return part_masks


def _part_masks_to_token_weights(part_masks: torch.Tensor) -> torch.Tensor:
    """Convert [B, K, H, W] spatial masks to [B, N, K] token weights.

    Transposes and flattens spatial dims so each token gets a K-dim
    soft part assignment vector.
    """
    B, K, H, W = part_masks.shape
    # [B, K, H*W] -> [B, H*W, K]
    return part_masks.view(B, K, H * W).permute(0, 2, 1)


class SPTransCompose(nn.Module):
    """Dual-branch Swin Transformer with Semantic-Pose Joint Conditioning
    and Part-Aware Routing."""

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
        n_keypoints: int = 17,
        use_visibility: bool = True,
        pose_detach: bool = True,
        heatmap_norm: str = 'none',
        branch_stage: int = 1,
        # SPTrans-specific
        adaptive_sem: bool = True,
        part_routing: bool = True,
        n_parts: int = 5,
        part_temp: float = 0.1,
        # v2: MoE-style Part Expert Routing
        part_expert: bool = True,
        mid_routing: bool = True,
        moe_bottleneck: int = 64,
        expert_bottleneck: int = 128,
        # v2: single branch mode
        single_branch: bool = False,
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

        n_stage = len(self.swin.stages)
        self.branch_stage = max(0, min(branch_stage, n_stage))
        self.n_keypoints = n_keypoints
        self.use_visibility = use_visibility
        self.pose_predictor = pose_predictor
        self.pose_detach = pose_detach
        self.heatmap_norm = heatmap_norm
        self.adaptive_sem = adaptive_sem
        self.part_routing = part_routing
        self.n_parts = n_parts
        self.part_temp = part_temp
        self.part_expert_enabled = part_expert
        self.mid_routing = mid_routing
        self.has_part_expert = part_expert  # exposed for make_model.py
        self.single_branch = single_branch

        self.num_features = self.swin.num_features
        self.avgpool = self.swin.avgpool

        # Part group indices (list of lists)
        part_names = list(COCO_PART_GROUPS.keys())[:n_parts]
        self.part_groups: List[List[int]] = [COCO_PART_GROUPS[n] for n in part_names]

        # Deep-copy split stages for local branch (skip if single branch)
        self.local_stages = nn.ModuleList()
        self.local_norms = nn.ModuleDict()
        if not self.single_branch:
            for stage in self.swin.stages[self.branch_stage:]:
                self.local_stages.append(copy.deepcopy(stage))
            for idx in range(self.branch_stage, n_stage):
                norm_layer = getattr(self.swin, f'norm{idx}', None)
                if norm_layer is not None:
                    self.local_norms[str(idx)] = copy.deepcopy(norm_layer)

        # Semantic-Pose Joint Conditioning: per-stage MLP adaptor
        if self.adaptive_sem and not self.single_branch:
            self.sem_pose_adaptor = nn.ModuleList()
            for _ in range(n_stage):
                mlp = nn.Sequential(
                    nn.Linear(1, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 1),
                    nn.Sigmoid(),
                )
                nn.init.constant_(mlp[2].bias, 0.0)
                self.sem_pose_adaptor.append(mlp)

        # v2: Mid-level PoseRoutedMoE — one per local stage
        if self.mid_routing and not self.single_branch:
            self.local_moe = nn.ModuleList()
            for offset in range(len(self.local_stages)):
                idx = self.branch_stage + offset
                dim = self.num_features[min(idx + 1, len(self.num_features) - 1)]
                self.local_moe.append(
                    PoseRoutedMoE(dim, n_parts=n_parts, bottleneck=moe_bottleneck)
                )

        # v2: PartExpertHead — replaces flat concat
        if self.part_expert_enabled and not self.single_branch:
            self.part_expert_head = PartExpertHead(
                dim=self.num_features[-1], n_parts=n_parts,
                bottleneck=expert_bottleneck,
            )

        # Expose local_feat_dim for make_model.py
        if self.single_branch:
            self.local_feat_dim = 0  # no local branch
        elif self.part_expert_enabled:
            self.local_feat_dim = self.num_features[-1]
        elif self.part_routing:
            self.local_feat_dim = n_parts * self.num_features[-1]
        else:
            self.local_feat_dim = self.num_features[-1]

        # Internal buffers
        self._hm_fullres: Optional[torch.Tensor] = None
        self._vis: Optional[torch.Tensor] = None
        self.pose_enabled = (self.pose_predictor is not None)

    # ------------------------------------------------------------------
    # Weight init (reuse PoseSwinCompose pattern)
    # ------------------------------------------------------------------
    def _sync_local_branch(self):
        if not self.local_stages:
            return
        for offset, global_stage in enumerate(self.swin.stages[self.branch_stage:]):
            self.local_stages[offset].load_state_dict(global_stage.state_dict())
        for idx in range(self.branch_stage, len(self.swin.stages)):
            norm_global = getattr(self.swin, f'norm{idx}', None)
            norm_key = str(idx)
            norm_local = self.local_norms[norm_key] if norm_key in self.local_norms else None
            if norm_global is not None and norm_local is not None:
                norm_local.load_state_dict(norm_global.state_dict())

    def init_weights(self, pretrained=None):
        path = pretrained if pretrained is not None else self._pretrained
        if not path:
            self.swin.init_weights(pretrained=None)
            self._sync_local_branch()
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
            self.swin.load_state_dict(loadable, strict=False)
            print(f"[SPTrans][swin_ckpt] loaded={len(loadable)} miss={len(msd)-len(loadable)} from {path}")
        except Exception as e:
            print(f"[SPTrans][swin_ckpt] remap failed: {e}; fallback")
            if self._convert_weights:
                self.swin.init_weights(path)
            else:
                self.swin.init_weights(pretrained=None)

        self._sync_local_branch()

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------
    def _get_pose(self, images: torch.Tensor):
        if self.pose_predictor is None:
            B = images.shape[0]
            device = images.device
            self._hm_fullres = torch.zeros(B, self.n_keypoints, 8, 6, device=device)
            self._vis = torch.zeros(B, self.n_keypoints, device=device)
            return self._hm_fullres, self._vis

        if self.pose_detach:
            with torch.no_grad():
                hm, vis = self.pose_predictor(images)
            hm = hm.detach()
            vis = vis.detach() if vis is not None else None
        else:
            hm, vis = self.pose_predictor(images)

        if vis is None:
            B, K, h, w = hm.shape
            vis = hm.view(B, K, -1).amax(dim=-1)
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-6)

        self._hm_fullres = hm
        self._vis = vis
        return hm, vis

    def _resized_heatmaps(self, hw_shape):
        """Resize cached heatmaps to target spatial shape.

        When use_visibility=True, each keypoint's heatmap is scaled by its
        visibility score so that occluded keypoints have near-zero activation.
        This is critical for both adaptive semantic (direction 2) and
        part routing (direction 3) to distinguish visible from occluded regions.
        """
        if self._hm_fullres is None:
            return None
        hm = F.interpolate(self._hm_fullres, size=hw_shape, mode='bilinear', align_corners=False)
        if self.heatmap_norm == 'sigmoid':
            hm = hm.sigmoid()
        elif self.heatmap_norm == 'softmax':
            B, K, h, w = hm.shape
            hm = F.softmax(hm.view(B, K, -1), dim=-1).view(B, K, h, w)
        # Visibility weighting: occluded keypoints -> near-zero heatmap
        if self.use_visibility and self._vis is not None:
            hm = hm * self._vis.unsqueeze(-1).unsqueeze(-1).clamp_min(0.0)
        return hm

    # ------------------------------------------------------------------
    # Semantic-Pose Joint Conditioning (Direction 2)
    # ------------------------------------------------------------------
    def _adaptive_semantic(self, x_tokens, hw_shape, stage_idx, semantic_weight):
        """Apply spatially adaptive semantic embedding using pose confidence."""
        if not (hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0
                and semantic_weight is not None):
            return x_tokens

        # Pose confidence map: sum over all keypoints at each spatial location
        hm = self._resized_heatmaps(hw_shape)  # [B, 17, H, W]
        if hm is not None:
            pose_conf = hm.sum(dim=1)           # [B, H, W]
            pose_conf = pose_conf.flatten(1)    # [B, N]
            # Normalize to [0, 1] per-sample
            pc_min = pose_conf.amin(dim=1, keepdim=True)
            pc_max = pose_conf.amax(dim=1, keepdim=True)
            pose_conf = (pose_conf - pc_min) / (pc_max - pc_min + 1e-6)
            # MLP: pose_conf -> factor in (0, 1)
            # High pose_conf -> small factor -> less semantic -> trust vision
            # Low pose_conf -> large factor -> more semantic -> trust prior
            factor = self.sem_pose_adaptor[stage_idx](pose_conf.unsqueeze(-1))  # [B, N, 1]
        else:
            factor = 0.5  # fallback: neutral

        # Original semantic embedding
        sw = self.swin.semantic_embed_w[stage_idx](semantic_weight).unsqueeze(1)  # [B, 1, C]
        sb = self.swin.semantic_embed_b[stage_idx](semantic_weight).unsqueeze(1)  # [B, 1, C]

        # Modulate: factor scales the semantic embedding strength
        sw = sw * factor  # [B, N, C]
        sb = sb * factor  # [B, N, C]

        x_tokens = x_tokens * self.swin.softplus(sw) + sb
        return x_tokens

    def _original_semantic(self, x_tokens, stage_idx, semantic_weight):
        """Original SOLIDER semantic conditioning (global, non-adaptive)."""
        if not (hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0
                and semantic_weight is not None):
            return x_tokens
        sw = self.swin.semantic_embed_w[stage_idx](semantic_weight).unsqueeze(1)
        sb = self.swin.semantic_embed_b[stage_idx](semantic_weight).unsqueeze(1)
        x_tokens = x_tokens * self.swin.softplus(sw) + sb
        return x_tokens

    # ------------------------------------------------------------------
    # Part-Aware Routing (Direction 3)
    # ------------------------------------------------------------------
    def _part_pool(self, feat_map, visibility):
        """Pose-heatmap-guided per-part feature pooling.

        Uses _resized_heatmaps() which already applies visibility weighting
        when use_visibility=True, so occluded keypoints have near-zero
        activation in the heatmaps used for mask generation.

        Args:
            feat_map: [B, C, H, W] - local branch final feature map
            visibility: [B, 17]

        Returns:
            part_feats: [B, K*C]
            part_vis:   [B, K] or None
        """
        B, C, H, W = feat_map.shape
        K = self.n_parts

        # Use _resized_heatmaps to get visibility-weighted heatmaps
        hm = self._resized_heatmaps((H, W))  # [B, 17, H, W], vis-weighted if enabled

        # Keypoints -> part masks
        part_masks = []
        part_vis_list = []
        for part_kpts in self.part_groups:
            part_hm = hm[:, part_kpts, :, :]           # [B, len(part), H, W]
            mask = part_hm.max(dim=1)[0]                 # [B, H, W]
            part_masks.append(mask)
            if visibility is not None and self.use_visibility:
                pv = visibility[:, part_kpts].mean(dim=1)  # [B]
                part_vis_list.append(pv)

        part_masks = torch.stack(part_masks, dim=1)      # [B, K, H, W]
        # Softmax over parts -> each position mainly belongs to one part
        part_masks = F.softmax(part_masks / self.part_temp, dim=1)

        # Mask-weighted average pooling
        feat_expand = feat_map.unsqueeze(1)               # [B, 1, C, H, W]
        masks_expand = part_masks.unsqueeze(2)             # [B, K, 1, H, W]
        weighted = feat_expand * masks_expand              # [B, K, C, H, W]
        part_feats = weighted.sum(dim=(-2, -1))            # [B, K, C]
        mask_sum = part_masks.sum(dim=(-2, -1)).unsqueeze(2).clamp_min(1e-6)  # [B, K, 1]
        part_feats = part_feats / mask_sum                 # [B, K, C]

        part_feats_flat = part_feats.reshape(B, K * C)     # [B, K*C]

        part_vis = torch.stack(part_vis_list, dim=1) if part_vis_list else None  # [B, K]

        return part_feats_flat, part_vis

    # ------------------------------------------------------------------
    # Visibility-weighted pooling (single-branch mode)
    # ------------------------------------------------------------------
    def _vis_weighted_pool(self, feat_map):
        """Pool using pose visibility as spatial weight.

        Occluded regions get low weight so the feature focuses on visible parts.
        Zero extra parameters.
        """
        B, C, H, W = feat_map.shape
        hm = self._resized_heatmaps((H, W))  # [B, 17, H, W]
        if hm is None:
            return torch.flatten(self.avgpool(feat_map), 1)

        vis_map = hm.sum(dim=1)  # [B, H, W]
        # Normalize per sample to [0, 1]
        vmin = vis_map.flatten(1).amin(1, keepdim=True).unsqueeze(-1)
        vmax = vis_map.flatten(1).amax(1, keepdim=True).unsqueeze(-1)
        vis_map = (vis_map - vmin) / (vmax - vmin + 1e-6)
        # Floor so fully occluded images don't collapse to zero
        vis_map = vis_map + 0.1
        # Normalize to sum=1 for weighted average
        vis_map = vis_map / vis_map.sum(dim=(-2, -1), keepdim=True)
        # Weighted pool
        feat = (feat_map * vis_map.unsqueeze(1)).sum(dim=(-2, -1))  # [B, C]
        return feat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, semantic_weight=None):
        # 1. Get pose heatmaps
        heatmaps, visibility = self._get_pose(x)

        # 2. Auto-generate semantic_weight
        if semantic_weight is None and hasattr(self.swin, 'semantic_weight') and self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1, device=x.device) * self.swin.semantic_weight
            semantic_weight = torch.cat([w, 1 - w], dim=-1)

        # 3. Patch embed
        x_tokens, hw_shape = self.swin.patch_embed(x)
        if getattr(self.swin, 'use_abs_pos_embed', False):
            x_tokens = x_tokens + self.swin.absolute_pos_embed
        x_tokens = self.swin.drop_after_pos(x_tokens)

        outs_global = []

        # ============================================================
        # Single-branch mode: one Swin + vis-weighted pooling
        # ============================================================
        if self.single_branch:
            for i, stage in enumerate(self.swin.stages):
                x_tokens, hw_shape, out, out_hw_shape = stage(x_tokens, hw_shape)
                # Standard SOLIDER semantic conditioning
                x_tokens = self._original_semantic(x_tokens, i, semantic_weight)
                if i in self.swin.out_indices:
                    norm_layer = getattr(self.swin, f'norm{i}', None)
                    out_c = out if norm_layer is None else norm_layer(out)
                    B, N, C = out_c.shape
                    H, W = out_hw_shape
                    outs_global.append(out_c.transpose(1, 2).contiguous().view(B, C, H, W))

            global_feat = self._vis_weighted_pool(outs_global[-1])  # [B, D]
            return {
                'global_feat': global_feat,
                'local_feat': None,
                'concat_feat': None,
                'part_feats': None,
                'part_vis': None,
                'global_maps': outs_global,
                'local_maps': [],
            }

        # ============================================================
        # Dual-branch mode (v1/v2)
        # ============================================================
        outs_local = []

        # 4. Shared stages
        for i, stage in enumerate(self.swin.stages):
            if i >= self.branch_stage:
                break
            x_tokens, hw_shape, out, out_hw_shape = stage(x_tokens, hw_shape)

            norm_layer = getattr(self.swin, f'norm{i}', None)
            out_collect = out if norm_layer is None else norm_layer(out)
            B, N, C = out_collect.shape
            H, W = out_hw_shape
            out_map = out_collect.transpose(1, 2).contiguous().view(B, C, H, W)
            outs_global.append(out_map)
            outs_local.append(out_map)

            # Semantic conditioning
            if self.adaptive_sem:
                x_tokens = self._adaptive_semantic(x_tokens, hw_shape, i, semantic_weight)
            else:
                x_tokens = self._original_semantic(x_tokens, i, semantic_weight)

        # 5. Split into global / local
        x_global = x_tokens.clone()
        x_local = x_tokens.clone()
        hw_global = hw_shape
        hw_local = hw_shape

        # 6. Split stages
        for offset, stage_g in enumerate(self.swin.stages[self.branch_stage:]):
            idx = self.branch_stage + offset
            stage_l = self.local_stages[offset]

            x_global, hw_global, out_g, out_hw_g = stage_g(x_global, hw_global)
            x_local, hw_local, out_l, out_hw_l = stage_l(x_local, hw_local)

            # v2: Mid-level PoseRoutedMoE on local branch
            if self.mid_routing:
                hm_local = self._resized_heatmaps(hw_local)
                if hm_local is not None:
                    p_masks = _compute_part_masks(hm_local, self.part_groups, self.part_temp)
                    token_weights = _part_masks_to_token_weights(p_masks)  # [B, N, K]
                    x_local = self.local_moe[offset](x_local, token_weights)

            # Semantic conditioning on both branches
            if self.adaptive_sem:
                x_global = self._adaptive_semantic(x_global, hw_global, idx, semantic_weight)
                x_local = self._adaptive_semantic(x_local, hw_local, idx, semantic_weight)
            else:
                x_global = self._original_semantic(x_global, idx, semantic_weight)
                x_local = self._original_semantic(x_local, idx, semantic_weight)

            if idx in self.swin.out_indices:
                norm_g = getattr(self.swin, f'norm{idx}', None)
                out_g_collect = out_g if norm_g is None else norm_g(out_g)
                B_g, N_g, C_g = out_g_collect.shape
                Hg, Wg = out_hw_g
                outs_global.append(out_g_collect.transpose(1, 2).contiguous().view(B_g, C_g, Hg, Wg))

                norm_key = str(idx)
                norm_l = self.local_norms[norm_key] if norm_key in self.local_norms else None
                out_l_collect = out_l if norm_l is None else norm_l(out_l)
                B_l, N_l, C_l = out_l_collect.shape
                Hl, Wl = out_hw_l
                outs_local.append(out_l_collect.transpose(1, 2).contiguous().view(B_l, C_l, Hl, Wl))

        # 7. Global branch: standard avg pool
        global_feat = torch.flatten(self.avgpool(outs_global[-1]), 1)  # [B, D]

        # 8. Local branch output
        part_feats = None  # [B, K, D] — only set when part_expert is active
        if self.part_expert_enabled:
            # v2: PartExpertHead → [B, D=768]
            local_map = outs_local[-1]  # [B, C, H, W]
            B_l, C_l, H_l, W_l = local_map.shape
            hm = self._resized_heatmaps((H_l, W_l))
            part_masks = _compute_part_masks(hm, self.part_groups, self.part_temp)
            part_vis_list = []
            if visibility is not None and self.use_visibility:
                for part_kpts in self.part_groups:
                    part_vis_list.append(visibility[:, part_kpts].mean(dim=1))
                part_vis = torch.stack(part_vis_list, dim=1)  # [B, K]
            else:
                part_vis = torch.ones(local_map.shape[0], self.n_parts,
                                      device=local_map.device)
            local_feat, part_feats = self.part_expert_head(local_map, part_masks, part_vis)
        elif self.part_routing:
            local_feat, part_vis = self._part_pool(outs_local[-1], visibility)
        else:
            local_feat = torch.flatten(self.avgpool(outs_local[-1]), 1)
            part_vis = None

        concat_feat = torch.cat([global_feat, local_feat], dim=1)

        return {
            'global_feat': global_feat,
            'local_feat': local_feat,
            'concat_feat': concat_feat,
            'part_feats': part_feats,
            'part_vis': part_vis,
            'global_maps': outs_global,
            'local_maps': outs_local,
        }


# -------------------- Factory constructors --------------------
def _build_sptrans(img_size=224, drop_rate=0.0, attn_drop_rate=0.0,
                   drop_path_rate=0.0, pretrained=None, convert_weights=False,
                   semantic_weight=0.0, pose_cfg=None, sptrans_cfg=None,
                   **kwargs):
    """Shared builder for SPTrans variants."""
    if pose_cfg is None:
        try:
            from config import cfg as _cfg
            pose_cfg = _cfg.MODEL.POSE
        except Exception:
            pose_cfg = None

    if sptrans_cfg is None:
        try:
            from config import cfg as _cfg
            sptrans_cfg = _cfg.MODEL.SPTRANS
        except Exception:
            sptrans_cfg = None

    pose_predictor = None
    if pose_cfg is not None and pose_cfg.get('ENABLE', False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pose_predictor = MMPoseTopDownPredictor(pose_cfg['CFG'], pose_cfg['CKPT'], device)

    return SPTransCompose(
        pretrain_img_size=img_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained) if (pretrained and convert_weights) else None,
        pretrained=pretrained,
        convert_weights=convert_weights,
        semantic_weight=semantic_weight,
        # pose
        pose_predictor=pose_predictor,
        n_keypoints=pose_cfg.get('N_KPTS', 17) if pose_cfg else 17,
        use_visibility=pose_cfg.get('USE_VIS', True) if pose_cfg else True,
        pose_detach=pose_cfg.get('DETACH', True) if pose_cfg else True,
        heatmap_norm=pose_cfg.get('HM_NORM', 'none') if pose_cfg else 'none',
        branch_stage=pose_cfg.get('BRANCH_STAGE', 1) if pose_cfg else 1,
        # SPTrans
        adaptive_sem=sptrans_cfg.get('ADAPTIVE_SEM', True) if sptrans_cfg else True,
        part_routing=sptrans_cfg.get('PART_ROUTING', True) if sptrans_cfg else True,
        n_parts=sptrans_cfg.get('N_PARTS', 5) if sptrans_cfg else 5,
        part_temp=sptrans_cfg.get('PART_TEMP', 0.1) if sptrans_cfg else 0.1,
        # v2
        part_expert=sptrans_cfg.get('PART_EXPERT', True) if sptrans_cfg else True,
        mid_routing=sptrans_cfg.get('MID_ROUTING', True) if sptrans_cfg else True,
        moe_bottleneck=sptrans_cfg.get('MOE_BOTTLENECK', 64) if sptrans_cfg else 64,
        expert_bottleneck=sptrans_cfg.get('EXPERT_BOTTLENECK', 128) if sptrans_cfg else 128,
        single_branch=sptrans_cfg.get('SINGLE_BRANCH', False) if sptrans_cfg else False,
        **kwargs,
    )


def sptrans_base_patch4_window7_224(*args, **kwargs):
    kwargs.setdefault('embed_dims', 128)
    kwargs.setdefault('depths', (2, 2, 18, 2))
    kwargs.setdefault('num_heads', (4, 8, 16, 32))
    kwargs.setdefault('window_size', 7)
    return _build_sptrans(*args, **kwargs)


def sptrans_small_patch4_window7_224(*args, **kwargs):
    kwargs.setdefault('embed_dims', 96)
    kwargs.setdefault('depths', (2, 2, 18, 2))
    kwargs.setdefault('num_heads', (3, 6, 12, 24))
    kwargs.setdefault('window_size', 7)
    return _build_sptrans(*args, **kwargs)


def sptrans_tiny_patch4_window7_224(*args, **kwargs):
    kwargs.setdefault('embed_dims', 96)
    kwargs.setdefault('depths', (2, 2, 6, 2))
    kwargs.setdefault('num_heads', (3, 6, 12, 24))
    kwargs.setdefault('window_size', 7)
    return _build_sptrans(*args, **kwargs)
