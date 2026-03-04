"""
Pose-Guided Part Pooling (PosePartPool)

Lightweight module that uses pre-extracted keypoint coordinates and visibility
to pool body-part-specific features from the Swin-Tiny feature map.

COCO 17 keypoints are grouped into 5 body parts:
  0: head      (nose, left_eye, right_eye, left_ear, right_ear)
  1: torso     (left_shoulder, right_shoulder, left_hip, right_hip)
  2: upper_arm (left_elbow, right_elbow)
  3: lower_arm (left_wrist, right_wrist)
  4: legs      (left_knee, right_knee, left_ankle, right_ankle)

For each part, we create a soft spatial mask from keypoint locations,
apply it to the feature map, and pool to get a part feature.
Parts with visibility=0 (occluded) are down-weighted in the final feature.

Memory overhead: ~0.1GB (just a few linear projections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# COCO keypoint indices grouped by body part
PART_GROUPS = {
    0: [0, 1, 2, 3, 4],       # head: nose, eyes, ears
    1: [5, 6, 11, 12],        # torso: shoulders, hips
    2: [7, 8],                 # upper arms: elbows
    3: [9, 10],                # lower arms: wrists
    4: [13, 14, 15, 16],      # legs: knees, ankles
}
N_PARTS = 5


class PosePartPool(nn.Module):
    """Pool part features from spatial feature map using keypoint locations.

    Args:
        feat_dim: Dimension of the input feature map channels (768 for Swin-Tiny).
        feat_h: Feature map height (24 for 384x128 input with stride 16).
        feat_w: Feature map width (8 for 384x128 input with stride 16).
        n_parts: Number of body parts (default 5).
        sigma: Gaussian sigma for keypoint heatmap generation (in feature map coords).
        min_vis: Minimum visibility to consider a part as present.
    """

    def __init__(self, feat_dim=768, feat_h=24, feat_w=8, n_parts=5,
                 sigma=2.0, min_vis=0.3):
        super().__init__()
        self.feat_dim = feat_dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.n_parts = n_parts
        self.sigma = sigma
        self.min_vis = min_vis

        # Register coordinate grids as buffers (not parameters)
        yy = torch.arange(feat_h, dtype=torch.float32).view(-1, 1).expand(feat_h, feat_w)
        xx = torch.arange(feat_w, dtype=torch.float32).view(1, -1).expand(feat_h, feat_w)
        self.register_buffer('grid_y', yy)  # (H, W)
        self.register_buffer('grid_x', xx)  # (H, W)

    def _make_part_masks(self, kpts, vis):
        """Generate soft part masks from keypoint locations.

        Args:
            kpts: (B, 17, 3) - keypoints with [norm_x, norm_y, confidence]
            vis: (B, 17) - visibility scores from VisPredictHead

        Returns:
            masks: (B, N_PARTS, H, W) - soft spatial masks for each part
            part_vis: (B, N_PARTS) - average visibility per part
        """
        B = kpts.shape[0]
        H, W = self.feat_h, self.feat_w
        device = kpts.device

        masks = torch.zeros(B, self.n_parts, H, W, device=device)
        part_vis = torch.zeros(B, self.n_parts, device=device)

        for part_id, kpt_indices in PART_GROUPS.items():
            # Get keypoints for this part
            part_kpts = kpts[:, kpt_indices]  # (B, K, 3)
            part_v = vis[:, kpt_indices]      # (B, K)

            # Average visibility for this part
            part_vis[:, part_id] = part_v.mean(dim=1)

            # Convert normalized coords to feature map coords
            cx = part_kpts[:, :, 0] * W  # (B, K)
            cy = part_kpts[:, :, 1] * H  # (B, K)
            conf = part_kpts[:, :, 2]     # (B, K)

            # Generate Gaussian heatmap for each keypoint, weighted by conf * vis
            weight = conf * part_v  # (B, K)

            for k in range(len(kpt_indices)):
                # (B, 1, 1) broadcast with (H, W) grid
                dx = self.grid_x.unsqueeze(0) - cx[:, k:k+1].unsqueeze(-1)  # (B, H, W)
                dy = self.grid_y.unsqueeze(0) - cy[:, k:k+1].unsqueeze(-1)  # (B, H, W)
                dist_sq = dx ** 2 + dy ** 2
                gauss = torch.exp(-dist_sq / (2 * self.sigma ** 2))  # (B, H, W)
                masks[:, part_id] += gauss * weight[:, k:k+1].unsqueeze(-1)

            # Normalize mask to sum to 1 (for proper pooling)
            mask_sum = masks[:, part_id].sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            masks[:, part_id] = masks[:, part_id] / mask_sum

        return masks, part_vis

    def forward(self, feat_map, kpts, vis):
        """Pool part features from feature map.

        Args:
            feat_map: (B, H*W, D) or (B, D, H, W) - spatial feature map from Swin-Tiny
            kpts: (B, 17, 3) - normalized keypoints [x, y, conf]
            vis: (B, 17) - visibility scores

        Returns:
            part_feats: (B, N_PARTS, D) - part-specific features
            part_vis: (B, N_PARTS) - part visibility scores
        """
        B = feat_map.shape[0]
        D = self.feat_dim
        H, W = self.feat_h, self.feat_w

        # Reshape feature map to (B, D, H, W) if needed
        if feat_map.dim() == 3:
            # (B, H*W, D) -> (B, D, H, W)
            feat_map = feat_map.transpose(1, 2).reshape(B, D, H, W)
        elif feat_map.dim() == 2:
            raise ValueError("PosePartPool expects spatial features, not pooled features")

        # Generate part masks
        masks, part_vis = self._make_part_masks(kpts, vis)  # (B, N_PARTS, H, W), (B, N_PARTS)

        # Pool features for each part: weighted spatial average
        # (B, 1, D, H, W) * (B, N_PARTS, 1, H, W) -> sum over H,W -> (B, N_PARTS, D)
        feat_expanded = feat_map.unsqueeze(1)  # (B, 1, D, H, W)
        masks_expanded = masks.unsqueeze(2)     # (B, N_PARTS, 1, H, W)
        part_feats = (feat_expanded * masks_expanded).sum(dim=(-2, -1))  # (B, N_PARTS, D)

        return part_feats, part_vis
