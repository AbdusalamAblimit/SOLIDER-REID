"""
Pose-FiLM: Feature-wise Linear Modulation conditioned on pose.

Applies per-channel affine modulation (scale + shift) at every backbone
layer, conditioned on pose heatmap statistics. This makes every layer
of the backbone pose-aware, not just Stage 3 (like PSG).

FiLM was originally proposed for visual question answering (AAAI 2018).
We apply it to person ReID with pose as the conditioning signal.

Key differences from PSG:
- PSG: spatial multiplicative gating, Stage 3 only
- FiLM: channel-wise affine modulation (scale + shift), every layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseFiLMGenerator(nn.Module):
    """Generates FiLM parameters (gamma, beta) from pose heatmaps.

    Takes the spatial average of pose heatmaps as a compact pose
    representation, then produces per-channel scale and shift.

    Args:
        pose_channels: Number of heatmap channels (17 for COCO)
        feat_channels: Number of feature channels to modulate
        hidden_dim: Generator hidden dimension
    """

    def __init__(self, pose_channels=17, feat_channels=96, hidden_dim=32):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(pose_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_channels * 2),  # gamma + beta
        )
        # Initialize to identity: gamma=1, beta=0
        nn.init.zeros_(self.generator[-1].weight)
        # Bias: first half (gamma) = 0 (so 1+gamma=1), second half (beta) = 0
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, heatmaps):
        """Generate FiLM parameters from heatmaps.

        Args:
            heatmaps: (B, 17, H, W) pose heatmaps

        Returns:
            gamma: (B, C) per-channel scale (centered at 0, so actual scale = 1+gamma)
            beta: (B, C) per-channel shift
        """
        # Global average of each heatmap channel → compact pose descriptor
        pose_feat = heatmaps.mean(dim=(2, 3))  # (B, 17)

        # Generate gamma and beta
        params = self.generator(pose_feat)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)  # each (B, C)

        return gamma, beta


class PoseFiLMLayer(nn.Module):
    """Applies FiLM modulation to features.

    output = (1 + gamma) * features + beta

    The (1 + gamma) formulation ensures identity initialization
    (when gamma=0, beta=0: output = features).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, hw_shape, gamma, beta):
        """Apply FiLM modulation.

        Args:
            x: (B, L, C) token sequence
            hw_shape: (H, W) spatial dimensions
            gamma: (B, C) per-channel scale offset
            beta: (B, C) per-channel shift

        Returns:
            modulated: (B, L, C) modulated tokens
        """
        # Reshape gamma/beta for broadcasting: (B, 1, C)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # FiLM: (1 + gamma) * x + beta
        return (1.0 + gamma) * x + beta
