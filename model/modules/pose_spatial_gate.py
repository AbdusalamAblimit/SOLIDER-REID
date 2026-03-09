"""Pose Spatial Gate (PSG) module for backbone-internal pose injection.

Applies a learnable spatial gate derived from pose heatmaps to backbone
features. Zero-initialized for identity at start, so pretrained features
are preserved initially.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseSpatialGate(nn.Module):
    """Lightweight pose-conditioned spatial gate.

    Given backbone feature tokens and scene-level heatmaps, produces a
    per-position, per-channel gate that modulates features.

    Uses residual gating: out = x * (1 + gate), where gate is zero-initialized.

    Args:
        pose_channels: Number of heatmap channels (17 for COCO keypoints)
        feat_channels: Number of feature channels to gate
        hidden_dim: Hidden dimension in the gate network
    """

    def __init__(self, pose_channels=17, feat_channels=768, hidden_dim=64):
        super().__init__()
        self.feat_channels = feat_channels

        # Pose encoder: (17, H, W) -> (C, H, W) gate values
        self.encoder = nn.Sequential(
            nn.Conv2d(pose_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, feat_channels, kernel_size=1, bias=True),
        )
        # Zero-init final layer so initial gate = 0, output = x * (1+0) = x
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, x, hw_shape, scene_heatmaps):
        """Apply pose spatial gate to feature tokens.

        Args:
            x: (B, H*W, C) feature tokens from Swin block
            hw_shape: (H, W) spatial shape
            scene_heatmaps: (B, 17, hH, hW) scene-level pose heatmaps

        Returns:
            x_gated: (B, H*W, C) gated features
        """
        B, N, C = x.shape
        H, W = hw_shape

        # Resize heatmaps to feature spatial size
        if scene_heatmaps.shape[2:] != (H, W):
            hm = F.interpolate(scene_heatmaps, size=(H, W),
                               mode='bilinear', align_corners=False)
        else:
            hm = scene_heatmaps

        # Apply sigmoid to raw heatmap logits
        hm = torch.sigmoid(hm)

        # Encode to gate values: (B, C, H, W)
        gate = self.encoder(hm)

        # Reshape to (B, H*W, C) to match token layout
        gate = gate.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Residual gating
        return x * (1.0 + gate)
