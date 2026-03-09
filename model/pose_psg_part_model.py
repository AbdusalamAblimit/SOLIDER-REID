"""Pose-guided ReID model combining PSG backbone injection with part pooling.

Combines:
1. Pose Spatial Gates (PSG) inside Stage 3 for pose-aware feature extraction
2. Pose-guided part pooling on the PSG-enhanced feature map
"""
import torch
import torch.nn as nn
from .pose_backbone_model import PoseBackboneModel
from .modules.pose_part_pooling import PosePartPooling


class PosePSGPartModel(PoseBackboneModel):
    """ReID model with PSG backbone + part pooling.

    Test: configurable (part_only, concat_scaled, etc.)
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super().__init__(num_classes, camera_num, view_num, cfg, factory, semantic_weight)

        # Add part pooling on top of PSG-enhanced features
        self.pose_part = PosePartPooling(
            in_channels=self.in_planes,
            num_classes=num_classes,
            threshold=cfg.MODEL.POSE_THRESHOLD,
            heatmap_norm=cfg.MODEL.POSE_HEATMAP_NORM,
            temperature=cfg.MODEL.POSE_TEMPERATURE,
        )
        self.pose_part_weight = cfg.MODEL.POSE_PART_WEIGHT
        self.pose_test_feat = cfg.MODEL.POSE_TEST_FEAT

    def forward(self, x, label=None, cam_label=None, view_label=None,
                pose_dict=None):
        # Prepare pose
        scene_heatmaps = None
        scene_scores = None
        if pose_dict is not None:
            scene_heatmaps, scene_scores = self._prepare_pose(pose_dict)

        # Run backbone with PSG injection
        global_feat, featmaps = self._run_backbone_with_psg(x, scene_heatmaps)
        last_featmap = featmaps[-1]  # PSG-enhanced feature map

        if self.training:
            if self.reduce_feat_dim:
                global_feat = self.fcneck(global_feat)

            feat = self.bottleneck(global_feat)
            feat_cls = self.dropout(feat)

            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            # Part pooling on PSG-enhanced features
            if pose_dict is not None:
                part_cls_scores, part_feats, part_valid = self.pose_part(
                    last_featmap, scene_heatmaps, scene_scores)

                all_cls = [cls_score] + part_cls_scores
                all_feats = [global_feat] + part_feats
                return all_cls, all_feats, part_valid
            else:
                return cls_score, global_feat, featmaps
        else:
            if self.reduce_feat_dim:
                global_feat = self.fcneck(global_feat)

            feat = self.bottleneck(global_feat)

            if self.neck_feat == 'after':
                test_feat = feat
            else:
                test_feat = global_feat

            if pose_dict is not None:
                _, part_feats, part_valid = self.pose_part(
                    last_featmap, scene_heatmaps, scene_scores)

                if self.pose_test_feat == 'part_only':
                    test_feat = torch.cat(part_feats, dim=1)
                elif self.pose_test_feat == 'equal_concat':
                    test_feat = torch.cat(
                        [test_feat] + part_feats, dim=1)
                else:  # concat_scaled (default)
                    scale = 1.0 / len(part_feats)
                    test_feat = torch.cat(
                        [test_feat] + [f * scale for f in part_feats], dim=1)

            return test_feat, featmaps
