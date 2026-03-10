"""Pose Cross-Attention (PXA) module for global pose context injection.

Unlike PSG which gates features position-locally, PXA lets each feature
position attend to ALL pose positions via cross-attention, enabling
global body structure context.

Zero-initialized output projection ensures identity at start.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseCrossAttention(nn.Module):
    """Cross-attention between image features (Q) and pose tokens (K, V).

    Each feature position queries all pose positions, collecting global
    body structure information. This is fundamentally different from PSG's
    position-local gating.

    Args:
        pose_channels: Number of heatmap channels (17 for COCO)
        feat_channels: Feature dimension (768 for Swin-Tiny Stage 3)
        hidden_dim: Dimension for Q/K/V projections
    """

    def __init__(self, pose_channels=17, feat_channels=768, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5

        # Pose encoder: heatmaps -> pose tokens
        self.pose_proj = nn.Sequential(
            nn.Conv2d(pose_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Query projection: features -> queries
        self.q_proj = nn.Linear(feat_channels, hidden_dim, bias=True)

        # Output projection: attention output -> feature update (zero-init)
        self.out_proj = nn.Linear(hidden_dim, feat_channels, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, hw_shape, scene_heatmaps):
        """Apply pose cross-attention to feature tokens.

        Args:
            x: (B, H*W, C) feature tokens from Swin block
            hw_shape: (H, W) spatial shape
            scene_heatmaps: (B, 17, hH, hW) scene-level pose heatmaps

        Returns:
            x_updated: (B, H*W, C) features with global pose context
        """
        B, N, C = x.shape
        H, W = hw_shape

        # Resize heatmaps to feature spatial size
        if scene_heatmaps.shape[2:] != (H, W):
            hm = F.interpolate(scene_heatmaps, size=(H, W),
                               mode='bilinear', align_corners=False)
        else:
            hm = scene_heatmaps

        # Apply sigmoid to raw logits
        hm = torch.sigmoid(hm)

        # Encode pose tokens: (B, d, H, W) -> (B, N, d)
        pose_tokens = self.pose_proj(hm)
        pose_tokens = pose_tokens.permute(0, 2, 3, 1).reshape(B, N, self.hidden_dim)

        # Q from features, K=V from pose
        Q = self.q_proj(x)  # (B, N, d)
        K = pose_tokens      # (B, N, d)
        V = pose_tokens      # (B, N, d)

        # Cross-attention: each feature position attends to all pose positions
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, N, N)
        attn = F.softmax(attn, dim=-1)

        # Aggregate pose information
        update = torch.bmm(attn, V)  # (B, N, d)

        # Project back to feature dimension (zero-init for identity start)
        update = self.out_proj(update)  # (B, N, C)

        return x + update
