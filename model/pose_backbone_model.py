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

        # Create PSG modules for stage 3 blocks
        stage3 = self.base.stages[-1]
        num_blocks = len(stage3.blocks)
        self.psg_modules = nn.ModuleList([
            PoseSpatialGate(
                pose_channels=17,
                feat_channels=self.in_planes,  # 768 for Swin-Tiny
                hidden_dim=getattr(cfg.MODEL, 'POSE_PFM_HIDDEN', 64),
            )
            for _ in range(num_blocks)
        ])

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
            if i < num_stages - 1:
                # Stages 0 to N-2: run normally
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            else:
                # Last stage (Stage 3): manually run blocks with PSG
                x, hw_shape, out, out_hw_shape = self._run_stage_with_psg(
                    stage, x, hw_shape, scene_heatmaps)

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

    def _run_stage_with_psg(self, stage, x, hw_shape, scene_heatmaps):
        """Run a stage's blocks with PSG insertion after each block."""
        for block_idx, block in enumerate(stage.blocks):
            x = block(x, hw_shape)

            # Apply PSG after each block
            if scene_heatmaps is not None:
                x = self.psg_modules[block_idx](x, hw_shape, scene_heatmaps)

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
