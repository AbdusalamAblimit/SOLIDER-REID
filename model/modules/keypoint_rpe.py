"""Keypoint Relative Position Encoding (KP-RPE) for body-structure-aware attention.

Encodes the structural relationship between token pairs based on their
distances to body keypoints. Unlike PAB (additive decomposition: bias=val[i]+val[j]),
KP-RPE computes true pairwise bias: bias(i,j) = MLP(dist_i - dist_j) where
dist_i is token i's distance vector to all 17 COCO keypoints.

This extends Swin's spatial RPE to body-structure space:
  - Swin RPE: bias based on (xi-xj, yi-yj) grid displacement
  - KP-RPE: bias based on (d_i_kp0 - d_j_kp0, ..., d_i_kp16 - d_j_kp16)

Zero-initialized output layer ensures identity start (no effect initially).
"""
import torch
import torch.nn as nn


class KeypointRPE(nn.Module):
    """Compute pairwise attention bias from keypoint-relative distances.

    For each token pair (i, j) in a window, computes:
      r_ij = dist_i - dist_j  (17-dim: per-keypoint distance difference)
      bias(i,j) = MLP(r_ij)   (num_heads-dim: per-head bias)

    Args:
        num_keypoints: Number of body keypoints (17 for COCO)
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension in the MLP
        score_threshold: Keypoints below this confidence get zero distance
    """

    def __init__(self, num_keypoints=17, num_heads=24, hidden_dim=32,
                 score_threshold=0.3):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_heads = num_heads
        self.score_threshold = score_threshold

        self.mlp = nn.Sequential(
            nn.Linear(num_keypoints, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_heads),
        )
        # Zero-init output so initial bias = 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, token_kp_dists):
        """Compute pairwise attention bias from token-keypoint distances.

        Args:
            token_kp_dists: (B*nW, ws*ws, num_keypoints) — each token's
                normalized distance to each keypoint within a window.

        Returns:
            bias: (B*nW, num_heads, ws*ws, ws*ws) — pairwise attention bias.
        """
        # Pairwise distance difference: r_ij = d_i - d_j
        # (B*nW, ws*ws, 1, K) - (B*nW, 1, ws*ws, K) = (B*nW, ws*ws, ws*ws, K)
        d_diff = token_kp_dists.unsqueeze(2) - token_kp_dists.unsqueeze(1)

        # MLP: (B*nW, ws*ws, ws*ws, K) -> (B*nW, ws*ws, ws*ws, num_heads)
        bias = self.mlp(d_diff)

        # -> (B*nW, num_heads, ws*ws, ws*ws)
        return bias.permute(0, 3, 1, 2).contiguous()


def compute_token_kp_distances(hw_shape, keypoints, scores,
                               score_threshold=0.3,
                               stride=32):
    """Compute normalized distances from each feature map token to each keypoint.

    Args:
        hw_shape: (H, W) feature map spatial dimensions (e.g., 12, 4)
        keypoints: (B, 17, 2) person 0's keypoint coordinates in pixel space
        scores: (B, 17) keypoint confidence scores
        score_threshold: Zero out distances for unreliable keypoints
        stride: Total stride from input image to feature map (patch_size * downsample)

    Returns:
        token_dists: (B, H*W, 17) normalized distances
    """
    H, W = hw_shape
    B = keypoints.shape[0]
    device = keypoints.device

    # Create token position grid in pixel-like coordinates
    # Token at (h, w) in feature map corresponds to pixel (h*stride + stride/2, w*stride + stride/2)
    ys = torch.arange(H, device=device, dtype=torch.float32) * stride + stride / 2
    xs = torch.arange(W, device=device, dtype=torch.float32) * stride + stride / 2
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    # (H*W, 2)
    token_positions = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    # keypoints: (B, 17, 2) in pixel coordinates
    # token_positions: (H*W, 2)
    # Compute L2 distance: (B, H*W, 17)
    # (1, H*W, 1, 2) - (B, 1, 17, 2) -> (B, H*W, 17, 2)
    diff = token_positions.unsqueeze(0).unsqueeze(2) - keypoints.unsqueeze(1)
    dists = torch.norm(diff, dim=-1)  # (B, H*W, 17)

    # Normalize by image diagonal (approximate: stride * sqrt(H^2 + W^2))
    diag = stride * (H ** 2 + W ** 2) ** 0.5
    dists = dists / (diag + 1e-6)

    # Zero out unreliable keypoints
    reliable_mask = (scores > score_threshold).unsqueeze(1)  # (B, 1, 17)
    dists = dists * reliable_mask.float()

    return dists
