"""Dual-Branch Part Feature Extractor using pretrained ResNet-50.

A completely independent branch from the Swin-Tiny backbone that focuses on
extracting pose-guided part features using a pretrained CNN. The two branches
have NO shared parameters, so ResNet gradients cannot affect Swin's PCFC alpha.

Architecture:
  Input Image → ResNet-50 (LUPerson pretrained, last_stride=1)
    → Feature Map [B, 2048, 24, 8]
    → Pose-guided Gaussian Pooling → [B, 5, 2048]
    → Per-part BNNeck + Classifiers
"""

import torch
import torch.nn as nn
from ..backbones.resnet import ResNet, Bottleneck
from .pose_part import PosePartPooling, PosePartHead


class PartBranch(nn.Module):
    """Independent ResNet-50 branch for pose-guided part feature extraction.

    Args:
        num_classes: Number of identity classes
        n_parts: Number of body parts (default 5)
        part_sigma: Gaussian kernel sigma for part pooling
        img_size: (H, W) of input image
        pretrained_path: Path to pretrained ResNet-50 weights (e.g., LUPerson)
    """

    def __init__(self, num_classes, n_parts=5, part_sigma=2.0,
                 img_size=(384, 128), pretrained_path=None):
        super().__init__()

        # ResNet-50 with last_stride=1 for higher spatial resolution
        # Output: 24×8 feature map (for 384×128 input)
        self.backbone = ResNet(last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3])
        self.feat_dim = 2048  # ResNet-50 layer4 output channels

        # Load pretrained weights
        if pretrained_path:
            self.backbone.load_param(pretrained_path)
            print(f'===========PartBranch: loaded pretrained from {pretrained_path}===========')

        # Pose-guided Gaussian pooling
        self.part_pool = PosePartPooling(
            n_parts=n_parts,
            sigma=part_sigma,
            img_size=img_size,
        )

        # Part heads (BNNeck + classifiers)
        self.part_head = PosePartHead(
            in_channels=self.feat_dim,
            num_classes=num_classes,
            n_parts=n_parts,
        )

        self.n_parts = n_parts
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f'===========PartBranch: ResNet-50, {n_parts} parts, {n_params:.1f}M params===========')

    def forward(self, x, keypoints, visibility):
        """
        Args:
            x: [B, 3, H, W] input image (same as Swin branch)
            keypoints: [B, 17, 2] keypoint coords in image space
            visibility: [B, 17] per-keypoint visibility scores

        Returns:
            part_feats: [B, n_parts, 2048] part features (before BNNeck)
            part_vis: [B, n_parts] part visibility scores
            part_logits: list of [B, num_classes] per-part classification logits (training only)
            part_feats_bn: [B, n_parts, 2048] BNNeck-normalized part features
        """
        # Extract feature map from ResNet-50
        feat_map = self.backbone(x)  # [B, 2048, 24, 8]

        # Pose-guided Gaussian pooling
        part_feats, part_vis = self.part_pool(feat_map, keypoints, visibility)
        # part_feats: [B, n_parts, 2048], part_vis: [B, n_parts]

        # Part heads
        part_logits, part_feats_bn = self.part_head(part_feats, part_vis)
        # part_logits: list of [B, num_classes], part_feats_bn: [B, n_parts, feat_dim]

        return part_feats, part_vis, part_logits, part_feats_bn
