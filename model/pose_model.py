"""Pose-guided ReID model.

Extends build_transformer with pose-guided part pooling.
"""
import torch
import torch.nn as nn
from .make_model import build_transformer, weights_init_kaiming, weights_init_classifier
from .modules.pose_part_pooling import PosePartPooling


class PoseReIDModel(build_transformer):
    """ReID model with pose-guided part features.

    Architecture:
    - Swin backbone → global feature (GAP → BN → classifier)
    - Pose part pooling → per-part features (soft attention → BN → classifier)
    - Test: concatenate global + part features
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        super().__init__(num_classes, camera_num, view_num, cfg, factory, semantic_weight)

        # Pose part pooling module
        self.pose_part = PosePartPooling(
            in_channels=self.in_planes,
            num_classes=num_classes,
            sigma=cfg.MODEL.POSE_SIGMA,
            threshold=cfg.MODEL.POSE_THRESHOLD,
        )
        self.pose_part_weight = cfg.MODEL.POSE_PART_WEIGHT

    def forward(self, x, label=None, cam_label=None, view_label=None,
                keypoints=None, kp_scores=None):
        # Backbone forward
        global_feat, featmaps = self.base(x)

        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        if self.training:
            # Global classification
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            # Pose part pooling (only when keypoints are provided)
            if keypoints is not None and kp_scores is not None:
                # Use last stage feature map: featmaps[-1] is (B, C, H, W)
                last_featmap = featmaps[-1]
                part_cls_scores, part_feats, part_valid = self.pose_part(
                    last_featmap, keypoints, kp_scores)

                # Return: [global_cls, part1_cls, ...], [global_feat, part1_feat, ...], part_valid
                all_cls = [cls_score] + part_cls_scores
                all_feats = [global_feat] + part_feats
                return all_cls, all_feats, part_valid
            else:
                return cls_score, global_feat, featmaps
        else:
            if self.neck_feat == 'after':
                test_feat = feat
            else:
                test_feat = global_feat

            # At test time, also extract part features if keypoints available
            if keypoints is not None and kp_scores is not None:
                last_featmap = featmaps[-1]
                _, part_feats, part_valid = self.pose_part(
                    last_featmap, keypoints, kp_scores)

                # Concatenate global + valid part features
                # Scale down part features to balance with global
                scale = 1.0 / len(part_feats)
                test_feat = torch.cat(
                    [test_feat] + [f * scale for f in part_feats], dim=1)

            return test_feat, featmaps
