"""Keypoint Prompt Embedding (KPE) for Swin Transformer.

Inspired by KPR (ECCV 2024), this module converts pose keypoints into spatial
part tokens that are added to the patch embeddings before Swin processing.

Unlike PCFC (post-hoc visibility attention on final features) or PVFM (middle-layer
modulation), KPE injects pose information at the INPUT level. This is additive
(not multiplicative), so it doesn't destroy pre-trained feature patterns.

Approach:
1. Group 17 COCO keypoints into K body parts
2. For each spatial position in the patch grid, compute soft part assignments
   based on Gaussian proximity to keypoints
3. Map soft assignments through learnable part embeddings
4. Add to image patch features (like a part-aware positional encoding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# COCO 17 keypoint -> body part groupings
# Using 5 parts (head, torso, arms, upper-legs, lower-legs) + background
BODY_PART_GROUPS = [
    [0, 1, 2, 3, 4],      # head (nose, eyes, ears)
    [5, 6, 11, 12],        # torso (shoulders, hips)
    [7, 8, 9, 10],         # arms (elbows, wrists)
    [13, 14],              # upper legs (knees)
    [15, 16],              # lower legs (ankles)
]

NUM_PARTS = len(BODY_PART_GROUPS)  # 5


class KeypointPromptEmbedding(nn.Module):
    """Convert pose keypoints to spatial part tokens for Swin backbone injection.

    For each patch position, computes a soft part assignment based on Gaussian
    proximity to visible keypoints, then maps through learnable part embeddings.

    Args:
        embed_dim: Swin patch embedding dimension (96 for Swin-Tiny)
        img_size: (H, W) of input image
        patch_size: patch size used by Swin (typically 4)
        sigma: Gaussian sigma for keypoint influence in patch grid space
        n_parts: number of body parts
    """

    def __init__(self, embed_dim=96, img_size=(384, 128), patch_size=4,
                 sigma=3.0, n_parts=NUM_PARTS):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_h, self.img_w = img_size
        self.patch_size = patch_size
        self.sigma = sigma
        self.n_parts = n_parts

        # Patch grid dimensions
        self.grid_h = self.img_h // patch_size  # 96 for 384/4
        self.grid_w = self.img_w // patch_size   # 32 for 128/4

        # Learnable part embeddings: K parts + 1 background
        # Initialized small to not disrupt pre-trained features at start
        self.part_embeddings = nn.Parameter(
            torch.zeros(n_parts + 1, embed_dim) * 0.02
        )
        nn.init.normal_(self.part_embeddings, std=0.02)
        # Zero-init background token so it has no effect
        nn.init.zeros_(self.part_embeddings[0])

        # Learnable scale factor (starts small to not disrupt pre-trained features)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, keypoints, visibility):
        """Generate spatial part tokens.

        Args:
            keypoints: [B, 17, 2] in image space (x, y)
            visibility: [B, 17] per-keypoint visibility [0, 1]

        Returns:
            part_tokens: [B, grid_h*grid_w, embed_dim] to add to patch features
        """
        B = keypoints.shape[0]
        device = keypoints.device

        # Scale keypoints from image space to patch grid space
        scale_x = self.grid_w / self.img_w
        scale_y = self.grid_h / self.img_h
        kp_x = keypoints[:, :, 0].float() * scale_x  # [B, 17]
        kp_y = keypoints[:, :, 1].float() * scale_y  # [B, 17]

        # Create coordinate grid for patch positions
        gy = torch.arange(self.grid_h, device=device, dtype=torch.float32)
        gx = torch.arange(self.grid_w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # [H, W]

        # Compute per-part activation maps
        # For each part, compute Gaussian from all keypoints in that group
        # Shape: [B, n_parts, grid_h, grid_w]
        part_maps = torch.zeros(B, self.n_parts, self.grid_h, self.grid_w, device=device)

        grid_x_flat = grid_x.reshape(1, 1, -1)  # [1, 1, H*W]
        grid_y_flat = grid_y.reshape(1, 1, -1)  # [1, 1, H*W]

        for p, kp_indices in enumerate(BODY_PART_GROUPS):
            # Get keypoints for this part group
            kp_x_group = kp_x[:, kp_indices]  # [B, len(group)]
            kp_y_group = kp_y[:, kp_indices]  # [B, len(group)]
            vis_group = visibility[:, kp_indices]  # [B, len(group)]

            # Gaussian distance from each grid position to each keypoint
            # [B, len(group), H*W]
            dx = grid_x_flat - kp_x_group.unsqueeze(-1)
            dy = grid_y_flat - kp_y_group.unsqueeze(-1)
            dist_sq = dx ** 2 + dy ** 2
            gauss = torch.exp(-dist_sq / (2 * self.sigma ** 2))

            # Weight by visibility and take max across keypoints in group
            gauss = gauss * vis_group.unsqueeze(-1)  # [B, len(group), H*W]
            part_activation, _ = gauss.max(dim=1)  # [B, H*W]
            part_maps[:, p] = part_activation.reshape(B, self.grid_h, self.grid_w)

        # Flatten spatial dims: [B, n_parts, H*W]
        part_maps_flat = part_maps.reshape(B, self.n_parts, -1)

        # Add background channel: 1 - max(part_maps)
        fg_max, _ = part_maps_flat.max(dim=1, keepdim=True)  # [B, 1, H*W]
        bg_map = (1.0 - fg_max).clamp(min=0)  # [B, 1, H*W]

        # Concatenate: [B, n_parts+1, H*W] with background first
        all_maps = torch.cat([bg_map, part_maps_flat], dim=1)  # [B, K+1, H*W]

        # Softmax to get soft part assignment probabilities
        # Temperature-scaled softmax for sharper assignments
        all_maps = F.softmax(all_maps * 5.0, dim=1)  # [B, K+1, H*W]

        # Map through part embeddings: weighted sum
        # all_maps: [B, K+1, H*W], part_embeddings: [K+1, D]
        # result: [B, H*W, D]
        part_tokens = torch.einsum('bkn,kd->bnd', all_maps, self.part_embeddings)

        # Scale to control injection strength
        part_tokens = self.scale * part_tokens

        return part_tokens
