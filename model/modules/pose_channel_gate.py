"""Pose-Conditioned Channel Gate (PCG) module.

Applies a learnable channel gate derived from pose heatmaps to the global
feature vector after GAP. Orthogonal to PSG's spatial gating — PCG operates
on channel dimensions while PSG operates on spatial dimensions.

Zero-initialized for identity at start, preserving pretrained features.
"""
import torch
import torch.nn as nn


class PoseChannelGate(nn.Module):
    """Pose-conditioned channel gating on global features.

    Compresses scene-level heatmaps into a pose descriptor via GAP,
    then generates per-channel gates through a lightweight MLP.

    Uses residual gating: out = feat * (1 + gate), where gate is zero-initialized.

    Args:
        feat_dim: Dimension of the global feature vector (768 for Swin-Tiny)
        pose_channels: Number of heatmap channels (17 for COCO keypoints)
        hidden_dim: Hidden dimension in the gate MLP
    """

    def __init__(self, feat_dim=768, pose_channels=17, hidden_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(pose_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )
        # Zero-init last layer so initial gate = 0, output = feat * (1+0) = feat
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, global_feat, scene_heatmaps):
        """Apply pose channel gate to global feature.

        Args:
            global_feat: (B, C) global feature after GAP
            scene_heatmaps: (B, 17, hH, hW) scene-level pose heatmaps

        Returns:
            gated_feat: (B, C) channel-gated feature
        """
        # Compress heatmaps to pose descriptor: (B, 17, H, W) -> (B, 17)
        hm = torch.sigmoid(scene_heatmaps)
        pose_desc = hm.mean(dim=(-2, -1))  # GAP over spatial dims

        # Generate channel gate: (B, 17) -> (B, C)
        gate = self.gate(pose_desc)

        # Residual gating
        return global_feat * (1.0 + gate)
