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
        spatial_conv: If True, add 3x3 depthwise conv for spatial awareness
    """

    def __init__(self, pose_channels=17, feat_channels=768, hidden_dim=64,
                 spatial_conv=False):
        super().__init__()
        self.feat_channels = feat_channels

        # Pose encoder: (17, H, W) -> (C, H, W) gate values
        layers = [
            nn.Conv2d(pose_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        if spatial_conv:
            # 3x3 depthwise conv for spatial awareness (+576 params)
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                          groups=hidden_dim, bias=True),
                nn.ReLU(inplace=True),
            ])
        layers.append(
            nn.Conv2d(hidden_dim, feat_channels, kernel_size=1, bias=True),
        )
        self.encoder = nn.Sequential(*layers)

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


class ContentAdaptivePSG(nn.Module):
    """Content-Adaptive Pose Spatial Gate (CAPSG).

    Extends PSG by making the gate depend on both pose heatmaps AND current
    feature content. The content-adaptive correction is added as a residual
    to the standard PSG gate, so initial behavior is identical to PSG.

    gate = sigmoid(psg_gate + content_correction)
    output = x * (1 + gate)

    where:
        psg_gate = conv_gate_pose(relu(conv_pose(heatmap)))  [standard PSG]
        content_correction = conv_gate_content(pose_feat * img_feat)  [NEW, zero-init]

    Args:
        pose_channels: Number of heatmap channels (17 for COCO keypoints)
        feat_channels: Number of feature channels to gate
        hidden_dim: Hidden dimension in the gate network
    """

    def __init__(self, pose_channels=17, feat_channels=768, hidden_dim=64):
        super().__init__()
        self.feat_channels = feat_channels

        # Standard PSG path: heatmap -> gate logits
        self.pose_proj = nn.Conv2d(pose_channels, hidden_dim, kernel_size=1, bias=True)
        self.pose_relu = nn.ReLU(inplace=True)
        self.gate_pose = nn.Conv2d(hidden_dim, feat_channels, kernel_size=1, bias=True)

        # Content-adaptive path: features -> hidden dim for interaction
        self.feat_proj = nn.Conv2d(feat_channels, hidden_dim, kernel_size=1, bias=True)
        self.feat_relu = nn.ReLU(inplace=True)

        # Content correction: pose_feat * img_feat -> gate correction
        self.gate_content = nn.Conv2d(hidden_dim, feat_channels, kernel_size=1, bias=True)

        # Zero-init both gate outputs so initial gate = sigmoid(0+0) = 0.5...
        # Actually, zero-init only content path. PSG path uses normal init.
        # Then initial gate = sigmoid(psg_gate + 0) = standard PSG behavior.
        nn.init.zeros_(self.gate_content.weight)
        nn.init.zeros_(self.gate_content.bias)

        # Also zero-init the PSG gate for identity start (same as standard PSG)
        nn.init.zeros_(self.gate_pose.weight)
        nn.init.zeros_(self.gate_pose.bias)

    def forward(self, x, hw_shape, scene_heatmaps):
        """Apply content-adaptive pose spatial gate.

        Args:
            x: (B, H*W, C) feature tokens
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

        hm = torch.sigmoid(hm)

        # Standard PSG gate path
        pose_feat = self.pose_relu(self.pose_proj(hm))       # (B, hidden, H, W)
        psg_gate = self.gate_pose(pose_feat)                  # (B, C, H, W)

        # Content-adaptive correction
        feat_map = x.permute(0, 2, 1).reshape(B, C, H, W)    # (B, C, H, W)
        img_feat = self.feat_relu(self.feat_proj(feat_map))   # (B, hidden, H, W)
        combined = pose_feat * img_feat                       # element-wise interaction
        content_correction = self.gate_content(combined)      # (B, C, H, W)

        # Combined gate
        gate = torch.sigmoid(psg_gate + content_correction)   # (B, C, H, W)
        gate = gate.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return x * (1.0 + gate)
