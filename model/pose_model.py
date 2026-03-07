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
                pose_dict=None):
        """
        Args:
            x: (B, 3, H, W) input images
            label: (B,) person IDs
            cam_label: (B,) camera IDs
            view_label: (B,) view IDs
            pose_dict: dict with pose data (from PoseImageDataset collate):
                - 'primary_keypoints': (B, 17, 2)
                - 'primary_scores': (B, 17)
                - 'primary_heatmap': (B, 17, H, W)
                - 'all_keypoints': (B, MAX_PERSONS, 17, 2)
                - 'all_scores': (B, MAX_PERSONS, 17)
                - 'all_heatmaps': (B, MAX_PERSONS, 17, H, W)
                - 'all_bboxes': (B, MAX_PERSONS, 4)
                - 'person_mask': (B, MAX_PERSONS)
                - 'num_persons': (B,)
        """
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

            # Pose part pooling (uses primary person data)
            if pose_dict is not None:
                keypoints = pose_dict['primary_keypoints']
                kp_scores = pose_dict['primary_scores']
                last_featmap = featmaps[-1]
                part_cls_scores, part_feats, part_valid = self.pose_part(
                    last_featmap, keypoints, kp_scores)

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

            # At test time, also extract part features
            if pose_dict is not None:
                keypoints = pose_dict['primary_keypoints']
                kp_scores = pose_dict['primary_scores']
                last_featmap = featmaps[-1]
                _, part_feats, part_valid = self.pose_part(
                    last_featmap, keypoints, kp_scores)

                scale = 1.0 / len(part_feats)
                test_feat = torch.cat(
                    [test_feat] + [f * scale for f in part_feats], dim=1)

            return test_feat, featmaps
