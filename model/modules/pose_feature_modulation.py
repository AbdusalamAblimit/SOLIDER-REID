"""Pose Feature Modulation (PFM) module.

Uses pose heatmaps to modulate backbone feature maps before pooling,
making features inherently pose-aware at each spatial location.

Unlike part pooling which uses pose for spatial selection (WHERE to look),
PFM uses pose for feature modulation (WHAT to enhance/suppress).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseFeatureModulation(nn.Module):
    """Lightweight pose-conditioned feature modulation.

    Converts pose heatmaps into channel-spatial modulation weights
    and applies residual modulation to backbone features.

    Args:
        pose_channels: Number of input heatmap channels (17 for COCO).
        feat_channels: Number of backbone feature channels (768 for Swin-Tiny).
        hidden_dim: Hidden dimension in the encoder.
    """

    def __init__(self, pose_channels=17, feat_channels=768, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(pose_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, feat_channels, kernel_size=1, bias=True),
        )
        # Initialize to near-zero so initial modulation is identity
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, feat_map, scene_heatmaps):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone feature map
            scene_heatmaps: (B, 17, hH, hW) scene-level heatmaps (raw logits)

        Returns:
            modulated_feat: (B, C, fH, fW) pose-conditioned feature map
        """
        fH, fW = feat_map.shape[2:]

        # Resize heatmaps to feature map spatial dims
        if scene_heatmaps.shape[2:] != (fH, fW):
            hm = F.interpolate(
                scene_heatmaps, size=(fH, fW),
                mode='bilinear', align_corners=False)
        else:
            hm = scene_heatmaps

        # Normalize heatmaps to stable range before encoding
        hm = torch.sigmoid(hm)

        # Generate modulation weights
        mod = self.encoder(hm)  # (B, C, fH, fW)

        # Residual modulation: feat * (1 + mod)
        # Since encoder output is initialized to ~0, this starts as identity
        return feat_map * (1.0 + mod)
