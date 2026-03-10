"""Pose Attention Bias (PAB) module for attention-level pose injection.

Encodes pose heatmaps into per-position, per-head attention bias values
that are added to Swin's self-attention scores before softmax. Uses
additive decomposition: bias(i,j) = pose_val[i] + pose_val[j].

Zero-initialized for identity start (bias=0 → no effect initially).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseAttentionBias(nn.Module):
    """Compute attention bias from pose heatmaps.

    Produces per-position, per-head importance values. These are later
    decomposed additively in the attention computation:
      bias(i,j) = importance[i] + importance[j]

    Args:
        pose_channels: Number of heatmap channels (17 for COCO keypoints)
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in the encoder
    """

    def __init__(self, pose_channels=17, num_heads=24, hidden_dim=32):
        super().__init__()
        self.num_heads = num_heads

        # Encode pose heatmap to per-head importance
        self.encoder = nn.Sequential(
            nn.Conv2d(pose_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_heads, kernel_size=1, bias=True),
        )
        # Zero-init so initial bias = 0, attention unchanged
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, scene_heatmaps, hw_shape):
        """Compute pose importance map for attention bias.

        Args:
            scene_heatmaps: (B, 17, hH, hW) scene-level pose heatmaps
            hw_shape: (H, W) spatial shape of feature map

        Returns:
            pose_bias_map: (B, num_heads, H, W) per-position importance
        """
        H, W = hw_shape

        # Resize heatmaps to feature spatial size
        if scene_heatmaps.shape[2:] != (H, W):
            hm = F.interpolate(scene_heatmaps, size=(H, W),
                               mode='bilinear', align_corners=False)
        else:
            hm = scene_heatmaps

        # Apply sigmoid to raw heatmap logits
        hm = torch.sigmoid(hm)

        # Encode to per-head importance: (B, num_heads, H, W)
        pose_bias_map = self.encoder(hm)

        return pose_bias_map
