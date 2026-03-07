"""Pose-guided part pooling module for ReID.

Uses ViTPose-Huge keypoints to generate Gaussian heatmaps,
then performs soft spatial pooling to extract part features.
"""
import torch
import torch.nn as nn
from .pose_utils import generate_part_heatmaps, NUM_PARTS, PART_NAMES


class PosePartPooling(nn.Module):
    """Pose-guided soft part pooling.

    Given backbone feature map and pose keypoints, generates per-part
    heatmaps and uses them as spatial attention for soft pooling.

    Each part gets:
    - Soft attention pooling from feature map
    - Independent BN + classifier for ID loss
    - Part feature for triplet loss
    """

    def __init__(self, in_channels, num_classes, sigma=2.0, threshold=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_parts = NUM_PARTS
        self.sigma = sigma
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

    def forward(self, feat_map, keypoints, scores):
        """
        Args:
            feat_map: (B, C, H, W) backbone feature map
            keypoints: (B, 17, 2) normalized keypoints [0, 1]
            scores: (B, 17) keypoint confidence scores

        Returns:
            part_cls_scores: list of (B, num_classes) per-part classification scores
            part_feats: list of (B, C) per-part features (before BN)
            part_valid: (B, NUM_PARTS) validity mask
        """
        B, C, H, W = feat_map.shape

        # Generate part heatmaps: (B, NUM_PARTS, H, W)
        part_heatmaps, part_valid = generate_part_heatmaps(
            keypoints, scores, H, W, self.sigma, self.threshold)

        part_cls_scores = []
        part_feats = []

        for i in range(self.num_parts):
            # Soft attention pooling
            attn = part_heatmaps[:, i:i+1]  # (B, 1, H, W)
            attn_sum = attn.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            attn_norm = attn / attn_sum  # normalized attention

            # Weighted pooling
            part_feat = (feat_map * attn_norm).sum(dim=(2, 3))  # (B, C)

            # For invalid parts (no visible keypoints), fall back to GAP
            gap_feat = feat_map.mean(dim=(2, 3))  # (B, C)
            valid_mask = part_valid[:, i:i+1]  # (B, 1)
            part_feat = part_feat * valid_mask + gap_feat * (1 - valid_mask)

            part_feats.append(part_feat)

            # BN + classifier
            part_feat_bn = self.part_bns[i](part_feat)
            cls_score = self.part_classifiers[i](part_feat_bn)
            part_cls_scores.append(cls_score)

        return part_cls_scores, part_feats, part_valid
