"""
Pose Additive Adapter (PAA)

Adds pose-derived features to backbone features via additive injection.
Complements PSG's multiplicative gating with additive information.

PSG: x = x * (1 + gate)  → adjusts feature magnitude spatially
PAA: x = x + adapter     → adds pose-specific feature content

Zero-initialized for safe identity start.

Variants:
- PoseAdditiveAdapter: generic Conv2d encoder (default)
- PosePartStructuredAdapter: body-part-aware grouped encoder (exp072)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# COCO 17-keypoint body part groups
_PART_GROUPS = [
    [0, 1, 2, 3, 4],    # head: nose, eyes, ears
    [5, 6],              # shoulders
    [7, 8, 9, 10],       # arms: elbows, wrists
    [11, 12],            # hips
    [13, 14, 15, 16],    # legs: knees, ankles
]


class PoseAdditiveAdapter(nn.Module):
    """Lightweight pose-conditioned additive adapter.

    Args:
        pose_channels: Number of heatmap channels (17)
        feat_channels: Number of feature channels (768)
        bottleneck_dim: Bottleneck dimension for efficiency
    """

    def __init__(self, pose_channels=17, feat_channels=768, bottleneck_dim=32,
                 routed=False, adaptive_gate=False):
        super().__init__()
        self.feat_channels = feat_channels
        self.routed = routed
        self.adaptive_gate = adaptive_gate

        # Pose encoder with bottleneck: 17 → bottleneck → feat_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(pose_channels, bottleneck_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, feat_channels, kernel_size=1, bias=True),
        )

        # Zero-init: adapter starts at zero, so output = x + 0 = x
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

        # Adaptive gate: learns to suppress PAA in single-person (low-occlusion) images
        if adaptive_gate:
            self.gate_mlp = nn.Linear(pose_channels, 1)
            # Init to zero → sigmoid(0) = 0.5 → starts at half strength
            nn.init.zeros_(self.gate_mlp.weight)
            nn.init.zeros_(self.gate_mlp.bias)

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

        # Adaptive gate: scalar gate per sample based on heatmap statistics
        if self.adaptive_gate:
            # GAP of sigmoid heatmap → (B, 17) → MLP → sigmoid → (B, 1)
            hm_pool = hm.mean(dim=(2, 3))  # (B, 17)
            gate = torch.sigmoid(self.gate_mlp(hm_pool))  # (B, 1)
            adapter_out = adapter_out * gate.unsqueeze(1)  # (B, 1, 1) broadcast

        return x + adapter_out


class TargetDistractorDiffAdapter(nn.Module):
    """Target-Distractor Differential Adapter (TDDA).

    Takes H_diff = H_target - H_distractor as input and injects
    differential pose information. Uses tanh instead of sigmoid
    to preserve positive/negative semantics of the difference signal.

    Args:
        pose_channels: Number of heatmap channels (17)
        feat_channels: Number of feature channels (768)
        bottleneck_dim: Bottleneck dimension for efficiency
    """

    def __init__(self, pose_channels=17, feat_channels=768, bottleneck_dim=32):
        super().__init__()
        self.feat_channels = feat_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(pose_channels, bottleneck_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, feat_channels, kernel_size=1, bias=True),
        )

        # Zero-init: adapter starts at zero
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, x, hw_shape, diff_heatmaps):
        """
        Args:
            x: (B, H*W, C) feature tokens
            hw_shape: (H, W) spatial shape
            diff_heatmaps: (B, 17, hH, hW) H_target - H_distractor

        Returns:
            x + adapter: (B, H*W, C) features with differential pose injection
        """
        B, N, C = x.shape
        H, W = hw_shape

        if diff_heatmaps.shape[2:] != (H, W):
            hm = F.interpolate(diff_heatmaps, size=(H, W),
                               mode='bilinear', align_corners=False)
        else:
            hm = diff_heatmaps

        # Use tanh to preserve sign (positive = target, negative = distractor)
        hm = torch.tanh(hm)

        adapter_out = self.encoder(hm)
        adapter_out = adapter_out.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return x + adapter_out


class PosePartStructuredAdapter(nn.Module):
    """Part-structured pose additive adapter.

    Instead of a generic Conv2d mixing all 17 channels, uses independent
    encoders per body part group, then merges via a shared projection.

    Body parts: head(5), shoulders(2), arms(4), hips(2), legs(4) = 5 groups
    """

    def __init__(self, feat_channels=768, hidden_per_part=8):
        super().__init__()
        self.feat_channels = feat_channels
        self.part_groups = _PART_GROUPS
        num_parts = len(self.part_groups)

        # Independent encoder per body part
        self.part_encoders = nn.ModuleList()
        for group in self.part_groups:
            self.part_encoders.append(nn.Sequential(
                nn.Conv2d(len(group), hidden_per_part, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
            ))

        total_hidden = hidden_per_part * num_parts  # 8 * 5 = 40

        # Shared projection: merged part features → feat_channels
        self.proj = nn.Conv2d(total_hidden, feat_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, hw_shape, scene_heatmaps):
        B, N, C = x.shape
        H, W = hw_shape

        if scene_heatmaps.shape[2:] != (H, W):
            hm = F.interpolate(scene_heatmaps, size=(H, W),
                               mode='bilinear', align_corners=False)
        else:
            hm = scene_heatmaps

        hm = torch.sigmoid(hm)

        # Encode each body part independently
        part_feats = []
        for i, group in enumerate(self.part_groups):
            part_hm = hm[:, group]  # (B, n_kp, H, W)
            part_feats.append(self.part_encoders[i](part_hm))

        # Concat all part features: (B, total_hidden, H, W)
        merged = torch.cat(part_feats, dim=1)

        # Project to feature space: (B, C, H, W)
        adapter_out = self.proj(merged)
        adapter_out = adapter_out.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return x + adapter_out
