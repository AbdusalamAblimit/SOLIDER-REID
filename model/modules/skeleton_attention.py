"""Skeleton-Aware Self-Attention (SASA) — zero-parameter attention bias.

Uses skeleton geodesic distance to bias window self-attention:
  - Tokens mapped to the same/nearby body parts attend more strongly
  - Tokens mapped to distant body parts attend less
  - ZERO learnable parameters (pure inductive bias from skeleton graph)

Key difference from KP-RPE (exp052, neutral):
  - KP-RPE: MLP(euclidean_distance_diff) — requires learning, spatial distance
  - SASA: -alpha * geodesic_distance — zero params, skeleton topology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# COCO skeleton edges connecting 17 keypoints
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),     # head
    (5, 7), (7, 9),                       # left arm
    (6, 8), (8, 10),                      # right arm
    (5, 6),                               # shoulders
    (5, 11), (6, 12),                     # torso
    (11, 12),                             # hips
    (11, 13), (13, 15),                   # left leg
    (12, 14), (14, 16),                   # right leg
    (0, 5), (0, 6),                       # nose to shoulders
]


def _compute_geodesic_matrix(num_kp=17, edges=None):
    """All-pairs shortest path on the COCO skeleton graph (Floyd-Warshall)."""
    if edges is None:
        edges = COCO_SKELETON
    INF = num_kp
    dist = torch.full((num_kp, num_kp), float(INF), dtype=torch.float32)
    for i in range(num_kp):
        dist[i, i] = 0
    for u, v in edges:
        dist[u, v] = 1
        dist[v, u] = 1
    for k in range(num_kp):
        for i in range(num_kp):
            for j in range(num_kp):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist


class SkeletonAttentionBias(nn.Module):
    """Zero-parameter skeleton-aware attention bias.

    Registered as a module purely to hold the geodesic matrix buffer.
    The actual bias computation is done via compute_sasa_bias().

    Args:
        alpha: Bias strength. Default: 0.1
        num_keypoints: Number of keypoints (17 for COCO)
    """

    def __init__(self, alpha=0.1, num_keypoints=17):
        super().__init__()
        self.alpha = alpha
        geo = _compute_geodesic_matrix(num_keypoints)
        geo = geo / geo.max()  # normalize to [0, 1]
        self.register_buffer('geodesic_matrix', geo)  # (17, 17)

    def compute_bias(self, token_kp_assign, num_heads):
        """Compute pairwise attention bias from token keypoint assignments.

        Args:
            token_kp_assign: (B*nW, ws*ws) — keypoint index per token
            num_heads: number of attention heads

        Returns:
            bias: (B*nW, num_heads, ws*ws, ws*ws)
        """
        BnW, N = token_kp_assign.shape

        # Look up geodesic distance for each token pair
        assign_i = token_kp_assign.unsqueeze(2).expand(-1, -1, N)  # (BnW, N, N)
        assign_j = token_kp_assign.unsqueeze(1).expand(-1, N, -1)  # (BnW, N, N)
        geo_dist = self.geodesic_matrix[assign_i, assign_j]  # (BnW, N, N)

        # Bias: negative scaled geodesic distance
        bias = -self.alpha * geo_dist  # (BnW, N, N)

        # Expand to heads (shared across all heads)
        return bias.unsqueeze(1).expand(-1, num_heads, -1, -1).contiguous()


def compute_token_kp_assignments(heatmaps, hw_shape):
    """Assign each feature map token to its dominant keypoint.

    Args:
        heatmaps: (B, 17, H_hm, W_hm) scene-level pose heatmaps
        hw_shape: (H, W) feature map spatial dimensions

    Returns:
        token_assign: (B, H, W) with values in [0, 16]
    """
    H, W = hw_shape
    hm = F.interpolate(heatmaps, size=(H, W), mode='bilinear',
                       align_corners=False)  # (B, 17, H, W)
    return hm.argmax(dim=1)  # (B, H, W)
