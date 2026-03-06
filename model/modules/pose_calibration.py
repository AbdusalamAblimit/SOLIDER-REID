"""Pose-Conditioned Feature Calibration (PCFC).

Generates a visibility-guided spatial attention map from pose keypoints,
then re-weights the backbone feature map before global average pooling.
This makes the global feature inherently occlusion-aware.

The attention map is created by:
1. Mapping keypoints to feature map coordinates
2. Creating Gaussian blobs at visible keypoint locations
3. Taking the max (union) of all Gaussian blobs → visible region mask
4. Adding a learnable residual: final_attn = 1 + alpha * (vis_mask - mean(vis_mask))
   This centers the attention so it doesn't change the scale, only redistributes focus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# COCO 17 keypoint -> 5 body part groupings
COCO_PART_GROUPS = [
    [0, 1, 2, 3, 4],      # head
    [5, 6, 11, 12],        # torso
    [7, 8, 9, 10],         # arms
    [13, 14],              # thighs
    [15, 16],              # calves
]


class PoseVisibilityAttention(nn.Module):
    """Generate visibility-guided spatial attention from pose keypoints.

    Creates a soft attention map highlighting visible body regions on the
    feature map. Applied before GAP to make global features occlusion-aware.

    Args:
        img_size: (H, W) of input image
        sigma: Gaussian kernel sigma in feature map space
        alpha_init: initial strength of the attention modulation
    """

    def __init__(self, img_size=(384, 128), sigma=3.0, alpha_init=0.5):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.sigma = sigma
        # Learnable strength parameter
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, feat_map, keypoints, visibility):
        """
        Args:
            feat_map: [B, C, H, W] backbone feature map (or list, uses last)
            keypoints: [B, 17, 2] keypoint coords in image space (x, y)
            visibility: [B, 17] per-keypoint visibility scores [0, 1]

        Returns:
            calibrated_feat_map: [B, C, H, W] re-weighted feature map
            attn_map: [B, 1, H, W] the attention map (for visualization)
        """
        if isinstance(feat_map, (list, tuple)):
            feat_map = feat_map[-1]

        B, C, H, W = feat_map.shape
        device = feat_map.device

        # Scale keypoints from image space to feature map space
        scale_x = W / self.img_w   # e.g., 4 / 128 = 0.03125
        scale_y = H / self.img_h   # e.g., 12 / 384 = 0.03125

        kp_x = keypoints[:, :, 0].float() * scale_x  # [B, 17]
        kp_y = keypoints[:, :, 1].float() * scale_y  # [B, 17]

        # Create coordinate grids
        gy = torch.arange(H, device=device, dtype=torch.float32)
        gx = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # [H, W]

        # Expand for batch and keypoint dimensions
        grid_x = grid_x[None, None, :, :]  # [1, 1, H, W]
        grid_y = grid_y[None, None, :, :]  # [1, 1, H, W]

        kp_x = kp_x[:, :, None, None]  # [B, 17, 1, 1]
        kp_y = kp_y[:, :, None, None]  # [B, 17, 1, 1]
        vis = visibility[:, :, None, None]  # [B, 17, 1, 1]

        # Gaussian blob for each keypoint: [B, 17, H, W]
        dist_sq = (grid_x - kp_x) ** 2 + (grid_y - kp_y) ** 2
        gauss = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        # Weight by visibility and take max across keypoints → visible region mask
        weighted_gauss = gauss * vis  # [B, 17, H, W]
        vis_mask, _ = weighted_gauss.max(dim=1)  # [B, H, W] — union of visible regions

        # Normalize to [0, 1]
        vis_max = vis_mask.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        vis_mask = vis_mask / vis_max

        # Attention: redistribute focus without changing scale
        # attn = 1 + alpha * (vis_mask - mean(vis_mask))
        vis_mean = vis_mask.mean(dim=(1, 2), keepdim=True)
        attn = 1.0 + self.alpha * (vis_mask - vis_mean)

        # Apply attention
        attn = attn.unsqueeze(1)  # [B, 1, H, W]
        calibrated = feat_map * attn

        return calibrated, attn


class PoseFeatureCalibration(nn.Module):
    """Full PCFC module: visibility attention + optional part loss.

    Combines:
    1. PoseVisibilityAttention: re-weights feature map for occlusion-aware GAP
    2. Optional part features extraction (from PosePartPooling) for auxiliary loss

    Args:
        img_size: (H, W) of input image
        sigma: Gaussian kernel sigma
        alpha_init: initial attention strength
        use_part_loss: whether to also extract part features for auxiliary loss
        n_parts: number of body parts for part loss
        part_sigma: sigma for part feature Gaussian pooling
    """

    def __init__(self, img_size=(384, 128), sigma=3.0, alpha_init=0.5,
                 use_part_loss=True, n_parts=5, part_sigma=2.0):
        super().__init__()
        self.vis_attn = PoseVisibilityAttention(
            img_size=img_size, sigma=sigma, alpha_init=alpha_init
        )
        self.use_part_loss = use_part_loss
        if use_part_loss:
            from .pose_part import PosePartPooling
            self.part_pool = PosePartPooling(
                n_parts=n_parts, sigma=part_sigma, img_size=img_size
            )

    def forward(self, feat_map, keypoints, visibility):
        """
        Returns:
            calibrated_feat_map: [B, C, H, W]
            attn_map: [B, 1, H, W]
            part_feats: [B, n_parts, C] (if use_part_loss) or None
            part_vis: [B, n_parts] (if use_part_loss) or None
        """
        calibrated, attn_map = self.vis_attn(feat_map, keypoints, visibility)

        part_feats = None
        part_vis = None
        if self.use_part_loss:
            # Extract part features from the ORIGINAL feature map (not calibrated)
            # Reason: part features should capture specific body regions,
            # while calibration is for the global pooling
            part_feats, part_vis = self.part_pool(feat_map, keypoints, visibility)

        return calibrated, attn_map, part_feats, part_vis
