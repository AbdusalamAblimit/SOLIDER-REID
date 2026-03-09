"""Pose-guided ReID model.

Extends build_transformer with pose-guided part pooling using
real ViTPose-Huge heatmaps, and optional Pose Feature Modulation (PFM).
"""
import torch
import torch.nn as nn
from .make_model import build_transformer
from .modules.pose_part_pooling import PosePartPooling
from .modules.pose_feature_modulation import PoseFeatureModulation
from .modules.pose_utils import merge_person_heatmaps


class PoseReIDModel(build_transformer):
    """ReID model with pose-guided part features.

    Architecture:
    - Swin backbone -> feature map
    - (optional) PFM: pose-conditioned feature modulation
    - Global feature (GAP -> BN -> classifier)
    - Pose part pooling -> per-part features (soft attention -> BN -> classifier)
    - Test: configurable feature mode

    pose_dict format (from PoseImageDataset collate):
        heatmaps:     (B, MAX_PERSONS, 17, hH, hW)  real model heatmaps
        keypoints:    (B, MAX_PERSONS, 17, 2)        pixel coordinates
        scores:       (B, MAX_PERSONS, 17)            confidence scores
        person_mask:  (B, MAX_PERSONS)                valid person mask
        num_persons:  (B,)                            actual person count
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super().__init__(num_classes, camera_num, view_num, cfg, factory, semantic_weight)

        self.pose_part = PosePartPooling(
            in_channels=self.in_planes,
            num_classes=num_classes,
            threshold=cfg.MODEL.POSE_THRESHOLD,
            heatmap_norm=cfg.MODEL.POSE_HEATMAP_NORM,
            temperature=cfg.MODEL.POSE_TEMPERATURE,
        )
        self.pose_part_weight = cfg.MODEL.POSE_PART_WEIGHT
        self.pose_test_feat = cfg.MODEL.POSE_TEST_FEAT

        # Optional Pose Feature Modulation
        self.pfm_enabled = getattr(cfg.MODEL, 'POSE_PFM_ENABLED', False)
        if self.pfm_enabled:
            self.pfm = PoseFeatureModulation(
                pose_channels=17,
                feat_channels=self.in_planes,
                hidden_dim=getattr(cfg.MODEL, 'POSE_PFM_HIDDEN', 64),
            )

    def forward(self, x, label=None, cam_label=None, view_label=None,
                pose_dict=None):
        # Backbone forward: global_feat (B, C), featmaps list
        global_feat, featmaps = self.base(x)

        if self.training:
            # Get feature map for part pooling and PFM
            if pose_dict is not None:
                scene_heatmaps, scene_scores = self._prepare_pose(pose_dict)
                last_featmap = featmaps[-1]  # (B, 768, 12, 4) for Swin-Tiny

                # Apply PFM if enabled (modulate feat map before pooling)
                if self.pfm_enabled:
                    last_featmap = self.pfm(last_featmap, scene_heatmaps)
                    # Re-compute global feat from modulated feature map
                    global_feat = last_featmap.mean(dim=(2, 3))  # GAP

            if self.reduce_feat_dim:
                global_feat = self.fcneck(global_feat)

            feat = self.bottleneck(global_feat)
            feat_cls = self.dropout(feat)

            # Global classification
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            # Pose part pooling
            if pose_dict is not None:
                part_cls_scores, part_feats, part_valid = self.pose_part(
                    last_featmap, scene_heatmaps, scene_scores)

                all_cls = [cls_score] + part_cls_scores
                all_feats = [global_feat] + part_feats
                return all_cls, all_feats, part_valid
            else:
                return cls_score, global_feat, featmaps
        else:
            if pose_dict is not None:
                scene_heatmaps, scene_scores = self._prepare_pose(pose_dict)
                last_featmap = featmaps[-1]

                # Apply PFM if enabled
                if self.pfm_enabled:
                    last_featmap = self.pfm(last_featmap, scene_heatmaps)
                    # Re-compute global feat from modulated feature map
                    global_feat = last_featmap.mean(dim=(2, 3))

            if self.reduce_feat_dim:
                global_feat = self.fcneck(global_feat)

            feat = self.bottleneck(global_feat)

            if self.neck_feat == 'after':
                test_feat = feat
            else:
                test_feat = global_feat

            # At test time, also extract part features
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

    @staticmethod
    def _prepare_pose(pose_dict):
        """Merge multi-person heatmaps and scores into scene-level tensors.

        Args:
            pose_dict: batched dict from PoseImageDataset collate

        Returns:
            scene_heatmaps: (B, 17, hH, hW) merged heatmaps
            scene_scores: (B, 17) merged scores (max across persons)
        """
        heatmaps = pose_dict['heatmaps']       # (B, MAX, 17, H, W)
        scores = pose_dict['scores']            # (B, MAX, 17)
        person_mask = pose_dict['person_mask']  # (B, MAX)

        # Merge heatmaps: max over valid persons
        scene_heatmaps = merge_person_heatmaps(heatmaps, person_mask)

        # Merge scores: max over valid persons
        score_mask = person_mask.unsqueeze(-1)  # (B, MAX, 1)
        masked_scores = scores * score_mask
        scene_scores = masked_scores.max(dim=1)[0]  # (B, 17)

        return scene_heatmaps, scene_scores
