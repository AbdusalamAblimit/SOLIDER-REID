"""Pose-Aware Masking Consistency (PAMC) module.

Provides:
1. PoseBodyMasker: creates masked views by occluding body part regions
   guided by pose heatmap spatial responses.
2. PAMCProjector: SimSiam-style projector MLP for consistency learning.
3. pamc_consistency_loss: asymmetric negative cosine similarity (memory-efficient).
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# COCO 17-keypoint body part groups (0-indexed)
BODY_PARTS = {
    'head':      [0, 1, 2, 3, 4],          # nose, eyes, ears
    'left_arm':  [5, 7, 9],                # left shoulder, elbow, wrist
    'right_arm': [6, 8, 10],               # right shoulder, elbow, wrist
    'torso':     [5, 6, 11, 12],           # shoulders + hips
    'left_leg':  [11, 13, 15],             # left hip, knee, ankle
    'right_leg': [12, 14, 16],             # right hip, knee, ankle
}
BODY_PART_NAMES = list(BODY_PARTS.keys())


class PoseBodyMasker(nn.Module):
    """Creates masked image views by occluding pose-guided body part regions.

    For each image in the batch:
    1. Takes scene heatmaps (17, H, W) and finds per-part spatial regions
    2. Randomly selects 1 body part group to mask
    3. Creates a binary mask covering that part's spatial extent
    4. Applies mask to the image tensor (fill with mean pixel value)
    """

    def __init__(self, pixel_mean=(0.485, 0.456, 0.406),
                 pixel_std=(0.229, 0.224, 0.225),
                 mask_expand=0.5,
                 num_parts_to_mask=1):
        """
        Args:
            pixel_mean/pixel_std: image normalization params (for computing fill value)
            mask_expand: fractional expansion of body part bounding box (0.5 = 50% each side)
            num_parts_to_mask: number of body parts to mask (1 or 2)
        """
        super().__init__()
        self.mask_expand = mask_expand
        self.num_parts_to_mask = num_parts_to_mask
        # Since images are already normalized with (x - mean) / std,
        # pixel_mean in normalized space = 0
        self.fill_value = 0.0

    @torch.no_grad()
    def forward(self, img, scene_heatmaps):
        """Create masked views of images using pose-guided body part masking.

        Args:
            img: (B, 3, H, W) normalized image tensor
            scene_heatmaps: (B, 17, hm_H, hm_W) merged scene-level heatmaps

        Returns:
            img_masked: (B, 3, H, W) masked image tensor
            mask_info: list of dicts with masking details per sample
        """
        B, C, H, W = img.shape
        img_masked = img.clone()
        mask_info = []

        for b in range(B):
            hm = scene_heatmaps[b]  # (17, hm_H, hm_W)

            # Find which body parts have sufficient heatmap response
            available_parts = []
            part_regions = {}

            for part_name, kp_indices in BODY_PARTS.items():
                # Get max heatmap response for this part's keypoints
                part_hm = hm[kp_indices]  # (n_kp, hm_H, hm_W)
                part_response = part_hm.max(dim=0)[0]  # (hm_H, hm_W)

                # Check if this part has any significant response
                max_response = part_response.max().item()
                if max_response < 0.1:
                    continue  # Part not visible, skip

                # Find bounding box of response > 30% of peak
                threshold = max_response * 0.3
                active = (part_response > threshold)
                if not active.any():
                    continue

                ys, xs = torch.where(active)
                # Map from heatmap coords to image coords
                hm_H, hm_W = hm.shape[1], hm.shape[2]
                y1 = (ys.min().item() / hm_H) * H
                y2 = ((ys.max().item() + 1) / hm_H) * H
                x1 = (xs.min().item() / hm_W) * W
                x2 = ((xs.max().item() + 1) / hm_W) * W

                # Expand bounding box
                bh = y2 - y1
                bw = x2 - x1
                y1 = max(0, y1 - bh * self.mask_expand)
                y2 = min(H, y2 + bh * self.mask_expand)
                x1 = max(0, x1 - bw * self.mask_expand)
                x2 = min(W, x2 + bw * self.mask_expand)

                available_parts.append(part_name)
                part_regions[part_name] = (int(y1), int(x1), int(y2), int(x2))

            if not available_parts:
                # No visible parts — apply random rectangle mask as fallback
                rh = int(H * random.uniform(0.2, 0.5))
                rw = int(W * random.uniform(0.3, 0.7))
                ry = random.randint(0, max(1, H - rh))
                rx = random.randint(0, max(1, W - rw))
                img_masked[b, :, ry:ry+rh, rx:rx+rw] = self.fill_value
                mask_info.append({'parts': ['random'], 'regions': [(ry, rx, ry+rh, rx+rw)]})
                continue

            # Randomly select 1 or 2 body parts to mask
            n_mask = min(self.num_parts_to_mask, len(available_parts))
            selected_parts = random.sample(available_parts, n_mask)

            parts_masked = []
            regions_masked = []
            for selected_part in selected_parts:
                y1, x1, y2, x2 = part_regions[selected_part]
                img_masked[b, :, y1:y2, x1:x2] = self.fill_value
                parts_masked.append(selected_part)
                regions_masked.append((y1, x1, y2, x2))
            mask_info.append({'parts': parts_masked, 'regions': regions_masked})

        return img_masked, mask_info


class PAMCProjector(nn.Module):
    """SimSiam-style projector MLP for consistency learning.

    Architecture: Linear → BN → ReLU → Linear
    """

    def __init__(self, feat_dim=768, proj_dim=2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, feat_dim),
        )

    def forward(self, x):
        return self.projector(x)


def pamc_consistency_loss(z1, z2_detached, projector):
    """Compute asymmetric consistency loss (memory-efficient variant).

    Only z1 has gradients; z2 is detached (from no_grad masked forward).
    The projector transforms z1 to predict z2's representation.

    Args:
        z1: (B, D) features from original view (has grad)
        z2_detached: (B, D) features from masked view (no grad)
        projector: PAMCProjector module

    Returns:
        loss: scalar consistency loss (negative cosine similarity)
    """
    p1 = projector(z1)                # predicted from original
    # z2 is already detached (from torch.no_grad block)
    loss = -F.cosine_similarity(p1, z2_detached, dim=1).mean()
    return loss
