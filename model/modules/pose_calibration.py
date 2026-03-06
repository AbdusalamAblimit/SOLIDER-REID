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
    3. Optional Occlusion Simulation Training (OST): randomly mask body parts
       during training to force occlusion-robust feature learning

    Args:
        img_size: (H, W) of input image
        sigma: Gaussian kernel sigma
        alpha_init: initial attention strength
        use_part_loss: whether to also extract part features for auxiliary loss
        n_parts: number of body parts for part loss
        part_sigma: sigma for part feature Gaussian pooling
        ost_prob: probability of applying occlusion simulation per sample
        ost_min_parts: minimum number of parts to occlude
        ost_max_parts: maximum number of parts to occlude
    """

    def __init__(self, img_size=(384, 128), sigma=3.0, alpha_init=0.5,
                 use_part_loss=True, n_parts=5, part_sigma=2.0,
                 ost_prob=0.0, ost_min_parts=1, ost_max_parts=3,
                 ms_part_stage=-1, ms_in_channels=384, ms_out_channels=768):
        super().__init__()
        self.vis_attn = PoseVisibilityAttention(
            img_size=img_size, sigma=sigma, alpha_init=alpha_init
        )
        self.use_part_loss = use_part_loss
        self.ost_prob = ost_prob
        self.ost_min_parts = ost_min_parts
        self.ost_max_parts = ost_max_parts
        self.n_parts = n_parts
        self.ms_part_stage = ms_part_stage
        if use_part_loss:
            from .pose_part import PosePartPooling
            self.part_pool = PosePartPooling(
                n_parts=n_parts, sigma=part_sigma, img_size=img_size
            )
            # Multi-scale: project lower-stage features to match global dim
            if ms_part_stage >= 0 and ms_in_channels != ms_out_channels:
                self.ms_proj = nn.Sequential(
                    nn.Conv2d(ms_in_channels, ms_out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(ms_out_channels),
                )

    def _simulate_occlusion(self, visibility):
        """Randomly zero out visibility for some body parts during training.

        For each sample with prob=ost_prob, select 1-3 body parts and set
        all their keypoints' visibility to 0. This forces the model to
        learn discriminative features from partial observations.
        """
        B = visibility.shape[0]
        sim_vis = visibility.clone()

        # Vectorized: generate random mask for which samples get OST
        apply_mask = torch.rand(B, device=visibility.device) < self.ost_prob

        for b in range(B):
            if not apply_mask[b]:
                continue
            # Select random number of parts to occlude
            n_occlude = torch.randint(
                self.ost_min_parts, self.ost_max_parts + 1, (1,)
            ).item()
            # Select which parts to occlude
            part_indices = torch.randperm(self.n_parts)[:n_occlude]
            for part_idx in part_indices:
                for kp_idx in COCO_PART_GROUPS[part_idx.item()]:
                    sim_vis[b, kp_idx] = 0.0

        return sim_vis

    def forward(self, feat_map, keypoints, visibility, ms_feat_map=None):
        """
        Args:
            feat_map: [B, C, H, W] Stage 3 feature map for global vis-weighted GAP
            keypoints: [B, 17, 2] keypoint coords in image space
            visibility: [B, 17] per-keypoint visibility scores
            ms_feat_map: [B, C', H', W'] optional Stage 2 feature map for part features

        Returns:
            calibrated_feat_map: [B, C, H, W]
            attn_map: [B, 1, H, W]
            part_feats: [B, n_parts, C] (if use_part_loss) or None
            part_vis: [B, n_parts] (if use_part_loss) or None
        """
        # Apply Occlusion Simulation Training during training
        if self.training and self.ost_prob > 0:
            visibility = self._simulate_occlusion(visibility)

        calibrated, attn_map = self.vis_attn(feat_map, keypoints, visibility)

        part_feats = None
        part_vis = None
        if self.use_part_loss:
            # Choose feature map for part extraction
            if ms_feat_map is not None and hasattr(self, 'ms_proj'):
                # Multi-scale: use higher-res features, projected to global dim
                part_fm = self.ms_proj(ms_feat_map)
            else:
                part_fm = feat_map
            part_feats, part_vis = self.part_pool(part_fm, keypoints, visibility)

        return calibrated, attn_map, part_feats, part_vis
