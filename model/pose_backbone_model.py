"""Pose-guided ReID model with backbone-internal pose injection.

Instead of post-hoc part pooling, injects pose information directly into
the backbone's feature extraction process via Pose Spatial Gates (PSG)
applied between Stage 3 blocks.

This changes HOW features are formed, not just how they're pooled.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .make_model import build_transformer
from .modules.pose_spatial_gate import PoseSpatialGate
from .modules.pose_attention_bias import PoseAttentionBias
from .modules.pose_channel_gate import PoseChannelGate
from .modules.pose_cross_attention import PoseCrossAttention
from .modules.pose_utils import merge_person_heatmaps


class PoseBackboneModel(build_transformer):
    """ReID model with pose injection inside backbone.

    Architecture:
    - Swin backbone Stages 0-2: unchanged
    - Stage 3: PSG applied after each SwinBlock
    - Global feature (GAP -> BN -> classifier)
    - No separate part branch

    Test feature = global feature (pose-aware).
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super().__init__(num_classes, camera_num, view_num, cfg, factory, semantic_weight)

        # Check injection mode: PSG (post-block gate), PAB (attention bias), PXA (cross-attention), or combo
        self.use_attn_bias = getattr(cfg.MODEL, 'POSE_ATTN_BIAS', False)
        self.use_combo = getattr(cfg.MODEL, 'POSE_PSG_PAB_COMBO', False)
        self.use_cross_attn = getattr(cfg.MODEL, 'POSE_CROSS_ATTN', False)

        # Determine which stages get pose injection
        psg_stages = list(getattr(cfg.MODEL, 'POSE_PSG_STAGES', [-1]))
        num_backbone_stages = len(self.base.stages)
        # Resolve negative indices
        self.psg_stage_indices = set()
        for s in psg_stages:
            idx = s if s >= 0 else num_backbone_stages + s
            self.psg_stage_indices.add(idx)

        hidden_dim = getattr(cfg.MODEL, 'POSE_PFM_HIDDEN', 64)
        spatial_conv = getattr(cfg.MODEL, 'POSE_PSG_SPATIAL', False)

        if self.use_combo:
            # Combo mode: both PAB and PSG
            self.pab_modules_dict = nn.ModuleDict()
            self.psg_modules_dict = nn.ModuleDict()
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                num_heads = stage.blocks[0].attn.w_msa.num_heads
                feat_ch = self.base.num_features[stage_idx]
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    self.pab_modules_dict[key] = PoseAttentionBias(
                        pose_channels=17,
                        num_heads=num_heads,
                        hidden_dim=32,  # PAB hidden dim
                    )
                    self.psg_modules_dict[key] = PoseSpatialGate(
                        pose_channels=17,
                        feat_channels=feat_ch,
                        hidden_dim=hidden_dim,
                        spatial_conv=spatial_conv,
                    )
        elif self.use_cross_attn:
            # PXA mode: cross-attention between features and pose tokens
            self.pxa_modules_dict = nn.ModuleDict()
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                feat_ch = self.base.num_features[stage_idx]
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    self.pxa_modules_dict[key] = PoseCrossAttention(
                        pose_channels=17,
                        feat_channels=feat_ch,
                        hidden_dim=hidden_dim,
                    )
        elif self.use_attn_bias:
            # PAB-only mode: create PoseAttentionBias modules per stage
            self.pab_modules_dict = nn.ModuleDict()
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                num_heads = stage.blocks[0].attn.w_msa.num_heads
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    self.pab_modules_dict[key] = PoseAttentionBias(
                        pose_channels=17,
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                    )
        else:
            # PSG-only mode: create PoseSpatialGate modules per stage
            self.psg_modules_dict = nn.ModuleDict()
            for stage_idx in sorted(self.psg_stage_indices):
                stage = self.base.stages[stage_idx]
                feat_ch = self.base.num_features[stage_idx]
                for block_idx in range(len(stage.blocks)):
                    key = f's{stage_idx}_b{block_idx}'
                    self.psg_modules_dict[key] = PoseSpatialGate(
                        pose_channels=17,
                        feat_channels=feat_ch,
                        hidden_dim=hidden_dim,
                        spatial_conv=spatial_conv,
                    )

            # Backward compatibility: also keep psg_modules list for Stage 3
            # (used by PosePSGPartModel which accesses self.psg_modules)
            last_stage_idx = num_backbone_stages - 1
            if last_stage_idx in self.psg_stage_indices:
                self.psg_modules = nn.ModuleList([
                    self.psg_modules_dict[f's{last_stage_idx}_b{j}']
                    for j in range(len(self.base.stages[last_stage_idx].blocks))
                ])

        # Pose-Conditioned Channel Gate (PCG) — after GAP, before BN
        self.use_channel_gate = getattr(cfg.MODEL, 'POSE_CHANNEL_GATE', False)
        if self.use_channel_gate:
            pcg_hidden = getattr(cfg.MODEL, 'POSE_PCG_HIDDEN', 64)
            feat_dim = self.in_planes  # 768 for Swin-Tiny
            self.channel_gate = PoseChannelGate(
                feat_dim=feat_dim,
                pose_channels=17,
                hidden_dim=pcg_hidden,
            )

        # Store backbone's semantic weight for manual forward
        self._semantic_weight_val = semantic_weight

    def _run_backbone_with_psg(self, x, scene_heatmaps):
        """Run backbone forward with PSG injection in Stage 3.

        Manually iterates backbone stages, inserting PSG after each
        Stage 3 block.
        """
        # Patch embedding
        x, hw_shape = self.base.patch_embed(x)
        if self.base.use_abs_pos_embed:
            x = x + self.base.absolute_pos_embed
        x = self.base.drop_after_pos(x)

        # Build semantic weight tensor
        sw_val = self._semantic_weight_val
        if self.base.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1, device=x.device) * sw_val
            w = torch.cat([w, 1 - w], dim=-1)
            sem_weight = w
        else:
            sem_weight = None

        outs = []
        num_stages = len(self.base.stages)

        for i, stage in enumerate(self.base.stages):
            if i in self.psg_stage_indices:
                # Stage with PSG: manually run blocks with gate injection
                x, hw_shape, out, out_hw_shape = self._run_stage_with_psg(
                    stage, x, hw_shape, scene_heatmaps, stage_idx=i)
            else:
                # Normal stage: run without modification
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)

            # Apply semantic weight (from SOLIDER pretraining)
            if sem_weight is not None:
                sw = self.base.semantic_embed_w[i](sem_weight).unsqueeze(1)
                sb = self.base.semantic_embed_b[i](sem_weight).unsqueeze(1)
                x = x * self.base.softplus(sw) + sb

            if i in self.base.out_indices:
                norm_layer = getattr(self.base, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.base.num_features[i]).permute(0, 3, 1,
                                                                   2).contiguous()
                outs.append(out)

        # Global average pool
        global_feat = self.base.avgpool(outs[-1])
        global_feat = torch.flatten(global_feat, 1)

        return global_feat, outs

    def _run_stage_with_psg(self, stage, x, hw_shape, scene_heatmaps, stage_idx=None):
        """Run a stage's blocks with pose injection (PSG, PAB, PXA, or combo)."""
        for block_idx, block in enumerate(stage.blocks):
            key = f's{stage_idx}_b{block_idx}'

            if self.use_combo and scene_heatmaps is not None:
                # Combo mode: PAB inside attention + PSG after block
                pose_bias_map = self.pab_modules_dict[key](
                    scene_heatmaps, hw_shape)
                x = block(x, hw_shape, pose_bias_map=pose_bias_map)
                x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)
            elif self.use_cross_attn and scene_heatmaps is not None:
                # PXA mode: run block then apply cross-attention
                x = block(x, hw_shape)
                x = self.pxa_modules_dict[key](x, hw_shape, scene_heatmaps)
            elif self.use_attn_bias and scene_heatmaps is not None:
                # PAB-only mode: compute pose attention bias and pass to block
                pose_bias_map = self.pab_modules_dict[key](
                    scene_heatmaps, hw_shape)
                x = block(x, hw_shape, pose_bias_map=pose_bias_map)
            else:
                x = block(x, hw_shape)
                # PSG-only mode: apply gate after block
                if not self.use_attn_bias and scene_heatmaps is not None:
                    x = self.psg_modules_dict[key](x, hw_shape, scene_heatmaps)

        # Handle downsample (Stage 3 has no downsample in Swin)
        if stage.downsample:
            x_down, down_hw_shape = stage.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

    def forward(self, x, label=None, cam_label=None, view_label=None,
                pose_dict=None):
        # Prepare pose
        scene_heatmaps = None
        if pose_dict is not None:
            scene_heatmaps, _ = self._prepare_pose(pose_dict)

        # Run backbone with PSG injection
        global_feat, featmaps = self._run_backbone_with_psg(x, scene_heatmaps)

        # Apply Pose-Conditioned Channel Gate (PCG) after GAP, before BN
        if self.use_channel_gate and scene_heatmaps is not None:
            global_feat = self.channel_gate(global_feat, scene_heatmaps)

        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        feat = self.bottleneck(global_feat)

        if self.training:
            feat_cls = self.dropout(feat)

            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            return cls_score, global_feat, featmaps
        else:
            if self.neck_feat == 'after':
                return feat, featmaps
            else:
                return global_feat, featmaps

    @staticmethod
    def _prepare_pose(pose_dict):
        """Merge multi-person heatmaps into scene-level tensors."""
        heatmaps = pose_dict['heatmaps']
        scores = pose_dict['scores']
        person_mask = pose_dict['person_mask']

        scene_heatmaps = merge_person_heatmaps(heatmaps, person_mask)

        score_mask = person_mask.unsqueeze(-1)
        masked_scores = scores * score_mask
        scene_scores = masked_scores.max(dim=1)[0]

        return scene_heatmaps, scene_scores
