"""
Pose-Guided Token Merging (PGTM)

Merges spatial patch tokens into semantic body-part tokens using pose heatmap,
applies self-attention on body parts, then scatters back to spatial positions.

This fundamentally changes the token-level computation from
spatial-patch attention to semantic body-part attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# COCO 17 keypoints -> 5 body parts (same as PAA)
_PART_KP_INDICES = [
    [0, 1, 2, 3, 4],    # head: nose, eyes, ears
    [5, 6],              # shoulders
    [7, 8, 9, 10],       # arms: elbows, wrists
    [11, 12],            # hips
    [13, 14, 15, 16],    # legs: knees, ankles
]
NUM_PARTS = len(_PART_KP_INDICES)


class PoseTokenMerge(nn.Module):
    """Merge spatial tokens into body-part tokens, attend, then expand back.

    Args:
        feat_dim: Feature dimension (768)
        num_parts: Number of body parts (5)
        num_heads: Attention heads for part-level attention
    """

    def __init__(self, feat_dim=768, num_parts=NUM_PARTS, num_heads=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_parts = num_parts

        # Part-level self-attention
        self.part_attn = nn.MultiheadAttention(
            feat_dim, num_heads, batch_first=True, dropout=0.1)
        self.part_norm = nn.LayerNorm(feat_dim)
        self.part_ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Dropout(0.1),
        )
        self.ffn_norm = nn.LayerNorm(feat_dim)

        # Residual gate (zero-init: starts as identity)
        self.gate = nn.Parameter(torch.zeros(1))

    def _compute_part_weights(self, scene_heatmaps, hw_shape):
        """Compute per-part spatial weight maps from pose heatmap.

        Args:
            scene_heatmaps: (B, 17, hH, hW) raw heatmaps
            hw_shape: (H, W) feature map spatial dims

        Returns:
            part_weights: (B, num_parts, H*W) softmax-normalized weights
        """
        H, W = hw_shape
        B = scene_heatmaps.shape[0]

        # Resize heatmap to feature map size
        if scene_heatmaps.shape[2:] != (H, W):
            hm = F.interpolate(scene_heatmaps, size=(H, W),
                               mode='bilinear', align_corners=False)
        else:
            hm = scene_heatmaps

        hm = torch.sigmoid(hm)  # (B, 17, H, W)

        # Merge keypoints per body part
        part_maps = []
        for kp_indices in _PART_KP_INDICES:
            # Max across keypoints in this part
            part_hm = hm[:, kp_indices].max(dim=1)[0]  # (B, H, W)
            part_maps.append(part_hm)

        # (B, num_parts, H, W)
        part_maps = torch.stack(part_maps, dim=1)

        # Flatten spatial: (B, num_parts, H*W)
        part_maps = part_maps.view(B, self.num_parts, H * W)

        # Softmax along spatial dim (each part gets a distribution over tokens)
        part_weights = F.softmax(part_maps * 5.0, dim=2)  # temperature=5 for sharper focus

        return part_weights

    def forward(self, x, hw_shape, scene_heatmaps):
        """
        Args:
            x: (B, H*W, C) feature tokens
            hw_shape: (H, W)
            scene_heatmaps: (B, 17, hH, hW) raw pose heatmaps

        Returns:
            x_out: (B, H*W, C) features with part-level attention applied
        """
        B, N, C = x.shape

        # 1. Compute part weights from pose heatmap
        part_weights = self._compute_part_weights(scene_heatmaps, hw_shape)
        # (B, num_parts, N)

        # 2. Merge: weighted sum of tokens per body part
        # (B, num_parts, N) @ (B, N, C) → (B, num_parts, C)
        part_tokens = torch.bmm(part_weights, x)

        # 3. Part-level self-attention
        part_out = self.part_attn(part_tokens, part_tokens, part_tokens)[0]
        part_tokens = self.part_norm(part_tokens + part_out)

        part_out = self.part_ffn(part_tokens)
        part_tokens = self.ffn_norm(part_tokens + part_out)

        # 4. Expand: scatter part tokens back to spatial positions
        # Each spatial token = weighted sum of part tokens it belongs to
        # (B, N, num_parts) @ (B, num_parts, C) → (B, N, C)
        expand_weights = part_weights.transpose(1, 2)  # (B, N, num_parts)
        # Normalize: each token sums to 1 across parts
        expand_weights = expand_weights / expand_weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        x_expanded = torch.bmm(expand_weights, part_tokens)  # (B, N, C)

        # 5. Gated residual
        x_out = x + self.gate * x_expanded

        return x_out
