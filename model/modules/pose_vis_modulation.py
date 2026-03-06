"""Pose-Guided Visibility Feature Modulation (PVFM).

Multi-stage, per-token visibility modulation for Swin Transformer.
Injects occlusion awareness at every stage by modulating token features
based on their proximity to visible keypoints.

Unlike PCFC which only operates on the final feature map before GAP,
PVFM modulates features throughout the backbone, allowing the attention
mechanism to progressively learn occlusion-aware representations.

The modulation is: x = x * (1 + beta * (vis_map - mean(vis_map)))
This preserves the mean feature magnitude (for compatibility with
pretrained weights) while redistributing emphasis to visible regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisibilityMapGenerator(nn.Module):
    """Generate visibility maps at arbitrary spatial resolutions.

    Given keypoints and visibility scores, creates a soft visibility
    map at the target resolution using Gaussian blobs.
    """

    def __init__(self, img_size=(384, 128), sigma=3.0):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.sigma = sigma

    def forward(self, keypoints, visibility, hw_shape):
        """
        Args:
            keypoints: [B, 17, 2] in image space (x, y)
            visibility: [B, 17] visibility scores [0, 1]
            hw_shape: (H, W) target spatial resolution

        Returns:
            vis_map: [B, H*W] normalized visibility map for tokens
        """
        H, W = hw_shape
        B = keypoints.shape[0]
        device = keypoints.device

        # Scale keypoints to target resolution
        scale_x = W / self.img_w
        scale_y = H / self.img_h

        kp_x = keypoints[:, :, 0].float() * scale_x  # [B, 17]
        kp_y = keypoints[:, :, 1].float() * scale_y  # [B, 17]

        # Create coordinate grids
        gy = torch.arange(H, device=device, dtype=torch.float32)
        gx = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # [H, W]

        grid_x = grid_x[None, None, :, :]  # [1, 1, H, W]
        grid_y = grid_y[None, None, :, :]

        kp_x = kp_x[:, :, None, None]  # [B, 17, 1, 1]
        kp_y = kp_y[:, :, None, None]
        vis = visibility[:, :, None, None]  # [B, 17, 1, 1]

        # Gaussian blobs weighted by visibility
        # Sigma scales with resolution to maintain consistent coverage
        sigma_scaled = self.sigma * (H / 12.0)  # normalize relative to stage3 (12x4)
        sigma_scaled = max(sigma_scaled, 0.5)  # minimum sigma

        dist_sq = (grid_x - kp_x) ** 2 + (grid_y - kp_y) ** 2
        gauss = torch.exp(-dist_sq / (2 * sigma_scaled ** 2))

        # Union of visible keypoint blobs
        weighted_gauss = gauss * vis  # [B, 17, H, W]
        vis_map, _ = weighted_gauss.max(dim=1)  # [B, H, W]

        # Normalize to [0, 1]
        vis_max = vis_map.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        vis_map = vis_map / vis_max

        # Flatten to token sequence
        vis_map = vis_map.view(B, H * W)  # [B, H*W]

        return vis_map


class StageVisModulation(nn.Module):
    """Per-stage visibility modulation with learnable strength.

    Applies: x = x * (1 + beta * (vis_map - mean(vis_map)))

    The centering (subtracting mean) ensures:
    1. Mean feature magnitude is preserved (compatible with pretrained weights)
    2. Visible regions get upweighted, occluded regions get downweighted
    3. When all regions have equal visibility, modulation is identity
    """

    def __init__(self, beta_init=0.3):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x, vis_map):
        """
        Args:
            x: [B, L, C] token features
            vis_map: [B, L] visibility scores per token

        Returns:
            x_mod: [B, L, C] modulated features
        """
        # Center the visibility map
        vis_centered = vis_map - vis_map.mean(dim=1, keepdim=True)  # [B, L]

        # Modulate: preserve mean, redistribute emphasis
        modulation = 1.0 + self.beta * vis_centered.unsqueeze(-1)  # [B, L, 1]

        return x * modulation


class PoseVisFeatureModulation(nn.Module):
    """Multi-stage visibility modulation for Swin Transformer.

    Designed to be called after each Swin stage (in parallel with
    SOLIDER's semantic weight modulation).

    Args:
        n_stages: number of Swin stages (default 4)
        img_size: input image size (H, W)
        sigma: Gaussian sigma for visibility map generation
        beta_init: initial modulation strength per stage
        active_stages: which stages to apply modulation (tuple of ints)
            By default, apply to stages 2 and 3 (the deeper stages)
            to avoid disturbing early low-level feature learning
    """

    def __init__(self, n_stages=4, img_size=(384, 128), sigma=3.0,
                 beta_init=0.3, active_stages=(2, 3)):
        super().__init__()
        self.n_stages = n_stages
        self.active_stages = set(active_stages)

        self.vis_gen = VisibilityMapGenerator(img_size=img_size, sigma=sigma)

        self.stage_mods = nn.ModuleDict()
        for s in active_stages:
            self.stage_mods[str(s)] = StageVisModulation(beta_init=beta_init)

    def forward(self, x, stage_idx, hw_shape, keypoints, visibility):
        """Apply visibility modulation for a specific stage.

        Args:
            x: [B, L, C] token features (after stage, before semantic modulation)
            stage_idx: which stage (0-3)
            hw_shape: (H, W) spatial shape of tokens
            keypoints: [B, 17, 2] in image space
            visibility: [B, 17] visibility scores

        Returns:
            x_mod: [B, L, C] modulated features
        """
        if stage_idx not in self.active_stages:
            return x

        # Generate visibility map at this stage's resolution
        vis_map = self.vis_gen(keypoints, visibility, hw_shape)

        # Apply modulation
        mod = self.stage_mods[str(stage_idx)]
        return mod(x, vis_map)
