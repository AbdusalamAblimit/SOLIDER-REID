"""Pose-guided part feature extraction using offline keypoints.

Given a backbone feature map and keypoint coordinates, extracts part-level
features via Gaussian attention pooling. Visibility scores are used to weight
part losses during training and part features during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# COCO 17 keypoint -> 5 body part groupings
COCO_PART_GROUPS = [
    [0, 1, 2, 3, 4],      # head (nose, eyes, ears)
    [5, 6, 11, 12],        # torso (shoulders, hips)
    [7, 8, 9, 10],         # arms (elbows, wrists)
    [13, 14],              # thighs (knees)
    [15, 16],              # calves (ankles)
]

PART_NAMES = ['head', 'torso', 'arms', 'thighs', 'calves']


class PosePartPooling(nn.Module):
    """Extract part features from feature map using keypoint-guided attention.

    For each body part (group of keypoints), creates a Gaussian attention map
    centered at the keypoint locations and pools features using this map.

    Args:
        n_parts: Number of body parts (default 5)
        sigma: Gaussian kernel sigma in feature map space (default 2.0)
        img_size: (H, W) of input image (default (384, 128))
    """

    def __init__(self, n_parts=5, sigma=2.0, img_size=(384, 128)):
        super().__init__()
        self.n_parts = n_parts
        self.sigma = sigma
        self.img_h, self.img_w = img_size
        self.part_groups = COCO_PART_GROUPS[:n_parts]

    def forward(self, feat_map, keypoints, visibility):
        """
        Args:
            feat_map: [B, C, H, W] backbone feature map or list of multi-scale maps
            keypoints: [B, 17, 2] keypoint coords in image space (x, y)
            visibility: [B, 17] per-keypoint visibility scores [0, 1]

        Returns:
            part_feats: [B, n_parts, C] part features
            part_vis: [B, n_parts] part visibility (mean of keypoint vis in group)
        """
        # Handle multi-scale feature maps (use last stage)
        if isinstance(feat_map, (list, tuple)):
            feat_map = feat_map[-1]

        B, C, H, W = feat_map.shape
        device = feat_map.device

        # Scale keypoints from image space to feature map space
        scale_x = W / self.img_w  # 8 / 128 = 0.0625
        scale_y = H / self.img_h  # 24 / 384 = 0.0625

        kp_x = keypoints[:, :, 0].float() * scale_x  # [B, 17]
        kp_y = keypoints[:, :, 1].float() * scale_y  # [B, 17]

        # Create coordinate grids
        grid_y = torch.arange(H, device=device, dtype=torch.float32)  # [H]
        grid_x = torch.arange(W, device=device, dtype=torch.float32)  # [W]
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')  # [H, W]

        part_feats = []
        part_vis = []

        for group in self.part_groups:
            # Gaussian attention for this part
            # [B, len(group)] -> attention map [B, H, W]
            attn = torch.zeros(B, H, W, device=device)
            group_vis = torch.zeros(B, device=device)

            for kp_idx in group:
                cx = kp_x[:, kp_idx]  # [B]
                cy = kp_y[:, kp_idx]  # [B]
                vis = visibility[:, kp_idx]  # [B]

                # Gaussian: exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
                dx = grid_x.unsqueeze(0) - cx.view(B, 1, 1)  # [B, H, W]
                dy = grid_y.unsqueeze(0) - cy.view(B, 1, 1)  # [B, H, W]
                gauss = torch.exp(-(dx**2 + dy**2) / (2 * self.sigma**2))

                # Weight by visibility
                attn = attn + gauss * vis.view(B, 1, 1)
                group_vis = group_vis + vis

            # Normalize attention
            group_vis = group_vis / len(group)  # mean visibility
            attn_sum = attn.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            attn = attn / attn_sum  # [B, H, W], sums to 1

            # Weighted pooling
            feat = (feat_map * attn.unsqueeze(1)).sum(dim=(2, 3))  # [B, C]
            part_feats.append(feat)
            part_vis.append(group_vis)

        part_feats = torch.stack(part_feats, dim=1)  # [B, n_parts, C]
        part_vis = torch.stack(part_vis, dim=1)  # [B, n_parts]

        return part_feats, part_vis


class PosePartHead(nn.Module):
    """Part-level BNNeck + classifiers for pose-guided part features.

    Args:
        in_channels: Feature dimension from backbone
        num_classes: Number of identity classes
        n_parts: Number of body parts
    """

    def __init__(self, in_channels, num_classes, n_parts=5):
        super().__init__()
        self.n_parts = n_parts

        # Per-part BNNeck
        self.part_bn = nn.ModuleList([
            nn.BatchNorm1d(in_channels) for _ in range(n_parts)
        ])
        # Per-part classifier
        self.part_cls = nn.ModuleList([
            nn.Linear(in_channels, num_classes, bias=False) for _ in range(n_parts)
        ])

        # Initialize
        for bn in self.part_bn:
            bn.bias.requires_grad_(False)
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
        for fc in self.part_cls:
            nn.init.normal_(fc.weight, std=0.001)

    def forward(self, part_feats, part_vis=None):
        """
        Args:
            part_feats: [B, n_parts, C]
            part_vis: [B, n_parts] (used at test time for weighting)

        Returns:
            Training: (list of per-part logits [B, num_classes], part_feats_bn [B, n_parts, C])
            Inference: visibility-weighted concatenated feature [B, n_parts*C]
        """
        B, K, C = part_feats.shape

        part_logits = []
        part_feats_bn = []

        for k in range(self.n_parts):
            feat_k = part_feats[:, k]  # [B, C]
            bn_k = self.part_bn[k](feat_k)  # [B, C]
            part_feats_bn.append(bn_k)

            if self.training:
                logits_k = self.part_cls[k](bn_k)  # [B, num_classes]
                part_logits.append(logits_k)

        part_feats_bn = torch.stack(part_feats_bn, dim=1)  # [B, K, C]

        if self.training:
            return part_logits, part_feats_bn
        else:
            # Visibility-weighted concatenation for test
            if part_vis is not None:
                # Soft weighting: don't zero out, just reduce weight
                weight = part_vis.unsqueeze(-1)  # [B, K, 1]
                weighted = part_feats_bn * weight  # [B, K, C]
            else:
                weighted = part_feats_bn
            return weighted.view(B, -1)  # [B, K*C]
