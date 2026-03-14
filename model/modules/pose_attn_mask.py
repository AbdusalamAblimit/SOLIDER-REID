"""
Pose-Guided Attention Masking (PGAM)

Generates hard attention masks from pose heatmaps to prevent
non-body tokens from attending to body tokens in Swin self-attention.

Unlike PSG (which soft-gates feature values), PGAM blocks information
flow through attention, preventing occluder contamination at its source.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseAttnMask(nn.Module):
    """
    Converts pose heatmaps to per-head attention bias maps for Swin attention.

    Non-body spatial positions get a large negative bias (-mask_value),
    effectively blocking them from participating in attention.

    The bias is applied via the additive decomposition in ShiftWindowMSA:
        bias(i,j) = val[i] + val[j]
    So setting val[non_body] = -mask_value/2 gives:
        body↔body: 0, body↔non_body: -mask_value/2, non_body↔non_body: -mask_value
    """

    def __init__(self, num_heads, threshold=0.3, mask_value=100.0):
        """
        Args:
            num_heads: number of attention heads in Swin Stage 3
            threshold: heatmap response threshold for body/non-body classification
            mask_value: negative bias value for masked positions
        """
        super().__init__()
        self.num_heads = num_heads
        self.threshold = threshold
        self.mask_value = mask_value

    def forward(self, scene_heatmaps, hw_shape):
        """
        Args:
            scene_heatmaps: (B, 17, H_hm, W_hm) pose heatmaps
            hw_shape: (H_feat, W_feat) feature map spatial dimensions

        Returns:
            pose_bias_map: (B, num_heads, H_feat, W_feat) attention bias
                0 for body positions, -mask_value/2 for non-body positions
        """
        H_feat, W_feat = hw_shape
        B = scene_heatmaps.shape[0]

        # Max across 17 keypoint channels → body confidence (B, 1, H_hm, W_hm)
        body_conf, _ = scene_heatmaps.max(dim=1, keepdim=True)

        # Normalize to [0, 1] via sigmoid (raw heatmaps are logits with range ~[-5, +15])
        body_conf = torch.sigmoid(body_conf)

        # Resize to feature map size
        body_conf = F.interpolate(
            body_conf, size=(H_feat, W_feat), mode='bilinear', align_corners=False
        )  # (B, 1, H_feat, W_feat)

        # Binary body mask: 1 = body, 0 = non-body
        body_mask = (body_conf > self.threshold).float()  # (B, 1, H_feat, W_feat)

        # Convert to attention bias:
        # body positions get 0 (no change to attention)
        # non-body positions get -mask_value/2 (halved because additive decomposition doubles it)
        attn_bias = (1.0 - body_mask) * (-self.mask_value / 2.0)  # (B, 1, H_feat, W_feat)

        # Expand to all heads
        attn_bias = attn_bias.expand(B, self.num_heads, H_feat, W_feat)

        return attn_bias
