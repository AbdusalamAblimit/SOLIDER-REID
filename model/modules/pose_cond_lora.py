"""
Pose-Conditioned LoRA (PCL)

Low-rank pose-conditioned feature adaptation, replacing PAA's
feature-independent addition with feature-dependent modulation.

PAA:  x = x + adapter(heatmap)          — pose content is static per position
PCL:  x = x + lora(x, heatmap)          — pose adaptation depends on features

The adaptation is computed as:
    x_down = W_down(x)                   (B, N, r) — down-project features
    hm_feat = conv_encode(sigmoid(hm))   (B, N, r) — encode heatmap to tokens
    z = x_down * hm_feat                 (B, N, r) — element-wise modulation
    lora_out = W_up(z)                   (B, N, C) — up-project, zero-init

Zero-initialized W_up ensures identity start (same as PAA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseCondLoRA(nn.Module):
    """Pose-conditioned low-rank adaptation module.

    Args:
        pose_channels: Number of heatmap channels (17)
        feat_channels: Number of feature channels (768)
        rank: Low-rank dimension (default 16)
    """

    def __init__(self, pose_channels=17, feat_channels=768, rank=16):
        super().__init__()
        self.feat_channels = feat_channels
        self.rank = rank

        # Heatmap encoder: 17 → rank (spatial, applied on H×W heatmap)
        self.hm_encoder = nn.Sequential(
            nn.Conv2d(pose_channels, rank, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Feature down-projection: C → rank
        self.W_down = nn.Linear(feat_channels, rank, bias=False)

        # Feature up-projection: rank → C (zero-initialized)
        self.W_up = nn.Linear(rank, feat_channels, bias=False)
        nn.init.zeros_(self.W_up.weight)

    def forward(self, x, hw_shape, scene_heatmaps):
        """
        Args:
            x: (B, H*W, C) feature tokens
            hw_shape: (H, W) spatial shape
            scene_heatmaps: (B, 17, hH, hW) pose heatmaps

        Returns:
            x + lora_out: (B, H*W, C)
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

        # Encode heatmap: (B, 17, H, W) → (B, rank, H, W) → (B, N, rank)
        hm_feat = self.hm_encoder(hm)
        hm_feat = hm_feat.permute(0, 2, 3, 1).reshape(B, H * W, self.rank)

        # Down-project features: (B, N, C) → (B, N, rank)
        x_down = self.W_down(x)

        # Element-wise modulation: features × heatmap encoding
        z = x_down * hm_feat

        # Up-project: (B, N, rank) → (B, N, C)
        lora_out = self.W_up(z)

        return x + lora_out
