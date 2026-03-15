"""
Pose Additive Adapter (PAA)

Adds pose-derived features to backbone features via additive injection.
Complements PSG's multiplicative gating with additive information.

PSG: x = x * (1 + gate)  → adjusts feature magnitude spatially
PAA: x = x + adapter     → adds pose-specific feature content

Zero-initialized for safe identity start.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseAdditiveAdapter(nn.Module):
    """Lightweight pose-conditioned additive adapter.

    Args:
        pose_channels: Number of heatmap channels (17)
        feat_channels: Number of feature channels (768)
        bottleneck_dim: Bottleneck dimension for efficiency
    """

    def __init__(self, pose_channels=17, feat_channels=768, bottleneck_dim=32,
                 routed=False):
        super().__init__()
        self.feat_channels = feat_channels
        self.routed = routed

        # Pose encoder with bottleneck: 17 → bottleneck → feat_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(pose_channels, bottleneck_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, feat_channels, kernel_size=1, bias=True),
        )

        # Zero-init: adapter starts at zero, so output = x + 0 = x
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, x, hw_shape, scene_heatmaps):
        """
        Args:
            x: (B, H*W, C) feature tokens (already processed by PSG)
            hw_shape: (H, W) spatial shape
            scene_heatmaps: (B, 17, hH, hW) pose heatmaps

        Returns:
            x + adapter: (B, H*W, C) features with additive pose injection
        """
        B, N, C = x.shape
        H, W = hw_shape

        # Resize heatmaps to feature size
        if scene_heatmaps.shape[2:] != (H, W):
            hm = F.interpolate(scene_heatmaps, size=(H, W),
                               mode='bilinear', align_corners=False)
        else:
            hm = scene_heatmaps

        hm = torch.sigmoid(hm)

        # Encode to additive features: (B, C, H, W)
        adapter_out = self.encoder(hm)

        # Reshape to match token layout
        adapter_out = adapter_out.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Reliability routing: only add adapter to low-confidence (occluded) regions
        if self.routed:
            # body_confidence = max of sigmoid heatmap channels → high for visible parts
            body_conf = hm.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
            # occlusion_mask: 1 for occluded, 0 for visible
            occlusion_mask = (1.0 - body_conf).permute(0, 2, 3, 1).reshape(B, H * W, 1)
            adapter_out = adapter_out * occlusion_mask

        return x + adapter_out
