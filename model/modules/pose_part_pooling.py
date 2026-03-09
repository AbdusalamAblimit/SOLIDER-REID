"""Pose-guided part pooling module for ReID.

Uses real ViTPose-Huge heatmaps (merged across all persons) to perform
soft spatial pooling of backbone features into body-part representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pose_utils import heatmaps_to_parts, NUM_PARTS, PART_NAMES


class PosePartPooling(nn.Module):
    """Pose-guided soft part pooling using real model heatmaps.

    Given backbone feature map and scene-level heatmaps (17 keypoints),
    groups heatmaps into body parts and uses them as spatial attention
    for soft pooling.

    Each part gets:
    - Soft attention pooling from feature map
    - Independent BN + classifier for ID loss
    - Part feature for triplet loss
    """

    def __init__(self, in_channels, num_classes, threshold=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_parts = NUM_PARTS
        self.threshold = threshold

        # Per-part BN + classifier
        self.part_bns = nn.ModuleList()
        self.part_classifiers = nn.ModuleList()
        for _ in range(self.num_parts):
            bn = nn.BatchNorm1d(in_channels)
            bn.bias.requires_grad_(False)
            bn.apply(self._init_kaiming)
            self.part_bns.append(bn)

            cls = nn.Linear(in_channels, num_classes, bias=False)
            cls.apply(self._init_classifier)
            self.part_classifiers.append(cls)

    @staticmethod
    def _init_kaiming(m):
        if isinstance(m, nn.BatchNorm1d):
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _init_classifier(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)

    def forward(self, feat_map, scene_heatmaps, scene_scores=None):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone feature map
            scene_heatmaps: (B, 17, hH, hW) scene-level heatmaps
                (merged across all persons)
            scene_scores: (B, 17) optional merged keypoint scores

        Returns:
            part_cls_scores: list of (B, num_classes) per-part cls scores
            part_feats: list of (B, C) per-part features (before BN)
            part_valid: (B, NUM_PARTS) validity mask
        """
        B, C, fH, fW = feat_map.shape

        # Raw heatmaps are logits (range ~[-5, +20]); apply sigmoid to normalize
        # to [0, 1] for stable attention weights under AMP float16
        scene_heatmaps = torch.sigmoid(scene_heatmaps)

        # Resize heatmaps to feature map spatial dims
        if scene_heatmaps.shape[2:] != (fH, fW):
            scene_heatmaps = F.interpolate(
                scene_heatmaps, size=(fH, fW),
                mode='bilinear', align_corners=False)

        # Group 17 keypoints into 5 body parts
        part_heatmaps, part_valid = heatmaps_to_parts(
            scene_heatmaps, scene_scores, self.threshold)
        # part_heatmaps: (B, NUM_PARTS, fH, fW)
        # part_valid: (B, NUM_PARTS)

        part_cls_scores = []
        part_feats = []

        gap_feat = feat_map.mean(dim=(2, 3))  # (B, C) fallback

        for i in range(self.num_parts):
            # Soft attention pooling
            attn = part_heatmaps[:, i:i+1]  # (B, 1, fH, fW)
            attn_sum = attn.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            attn_norm = attn / attn_sum  # normalized attention

            # Weighted pooling
            part_feat = (feat_map * attn_norm).sum(dim=(2, 3))  # (B, C)

            # For invalid parts, fall back to GAP
            valid_mask = part_valid[:, i:i+1]  # (B, 1)
            part_feat = part_feat * valid_mask + gap_feat * (1 - valid_mask)

            part_feats.append(part_feat)

            # BN + classifier
            part_feat_bn = self.part_bns[i](part_feat)
            cls_score = self.part_classifiers[i](part_feat_bn)
            part_cls_scores.append(cls_score)

        return part_cls_scores, part_feats, part_valid
