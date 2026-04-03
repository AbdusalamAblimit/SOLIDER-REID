"""Feature-Space Diffusion Completion (FSDC) for Occluded ReID.

Reconstructs occluded spatial token features using a lightweight
transformer denoiser conditioned on pose visibility.

Training: randomly mask body-part regions → denoise → L2 reconstruction loss
Testing: use pose confidence to identify occluded regions → denoise → complete features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureDenoiser(nn.Module):
    """Lightweight transformer denoiser for spatial token completion.

    Takes partially masked spatial tokens + pose visibility info,
    outputs completed token features.

    Args:
        feat_dim: token feature dimension (768)
        num_tokens: number of spatial tokens (48 for 12x4)
        num_layers: transformer layers (2)
        num_heads: attention heads (8)
        mask_ratio: fraction of body-part tokens to mask during training (0.3)
        noise_std: Gaussian noise std added to masked tokens (0.1)
    """

    def __init__(self, feat_dim=768, num_tokens=48, num_layers=2,
                 num_heads=8, mask_ratio=0.3, noise_std=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional encoding for spatial tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mask indicator embedding (2-class: visible/masked)
        self.mask_embed = nn.Embedding(2, feat_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=num_heads,
            dim_feedforward=feat_dim * 2,  # lightweight FFN
            dropout=0.1, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection (identity-start via zero init)
        self.output_proj = nn.Linear(feat_dim, feat_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Layer norm
        self.norm = nn.LayerNorm(feat_dim)

        print(f'[FSDC] Feature Denoiser: {num_layers} layers, {num_heads} heads, '
              f'mask_ratio={mask_ratio}, noise_std={noise_std}')

    def _generate_body_part_mask(self, heatmaps, fH, fW):
        """Generate spatial mask based on pose body-part regions.

        Randomly selects 1-3 body parts and masks the corresponding
        spatial tokens based on heatmap activation.

        Args:
            heatmaps: (B, 17, H, W) scene heatmaps
            fH, fW: feature map height and width

        Returns:
            mask: (B, fH*fW) binary mask, True = keep, False = mask out
        """
        B = heatmaps.shape[0]
        device = heatmaps.device

        # 6 body part groups (COCO keypoints)
        part_groups = [
            [0, 1, 2, 3, 4],      # head
            [5, 6, 11, 12],        # torso
            [5, 7, 9],             # left arm
            [6, 8, 10],            # right arm
            [11, 13, 15],          # left leg
            [12, 14, 16],          # right leg
        ]

        # Resize heatmaps to feature map size
        hm = F.interpolate(heatmaps, size=(fH, fW), mode='bilinear', align_corners=False)

        # Randomly select 1-3 parts to mask per sample
        mask = torch.ones(B, fH * fW, dtype=torch.bool, device=device)

        for b in range(B):
            num_parts_to_mask = torch.randint(1, 4, (1,)).item()  # 1-3 parts
            parts_to_mask = torch.randperm(6)[:num_parts_to_mask]

            for part_idx in parts_to_mask:
                kp_indices = part_groups[part_idx]
                # Aggregate heatmaps for this body part
                part_hm = hm[b, kp_indices].max(dim=0)[0]  # (fH, fW)
                # Threshold: mask tokens where body part has significant activation
                threshold = part_hm.mean() + 0.5 * part_hm.std()
                part_mask = part_hm > threshold  # True where body part is
                mask[b] = mask[b] & ~part_mask.flatten()

        return mask  # True = keep, False = masked

    def _generate_random_mask(self, B, device):
        """Fallback: random token masking (not pose-guided)."""
        num_masked = int(self.num_tokens * self.mask_ratio)
        mask = torch.ones(B, self.num_tokens, dtype=torch.bool, device=device)
        for b in range(B):
            indices = torch.randperm(self.num_tokens, device=device)[:num_masked]
            mask[b, indices] = False
        return mask

    def forward(self, spatial_tokens, heatmaps=None, fH=12, fW=4):
        """
        Args:
            spatial_tokens: (B, N, C) backbone spatial features (detached)
            heatmaps: (B, 17, H, W) scene heatmaps, or None
            fH, fW: feature map spatial dimensions

        Returns:
            completed_tokens: (B, N, C) with masked tokens reconstructed
            recon_loss: scalar reconstruction loss (only during training)
            stats: dict with diagnostics
        """
        B, N, C = spatial_tokens.shape
        device = spatial_tokens.device

        if self.training:
            # Generate body-part mask
            if heatmaps is not None:
                mask = self._generate_body_part_mask(heatmaps, fH, fW)
                # Fallback: if all tokens are visible (mask all True), use random mask
                for b in range(B):
                    if mask[b].all():
                        num_masked = max(1, int(N * self.mask_ratio))
                        indices = torch.randperm(N, device=device)[:num_masked]
                        mask[b, indices] = False
            else:
                mask = self._generate_random_mask(B, device)

            # Save original for reconstruction target
            target = spatial_tokens.detach().clone()

            # Replace masked tokens with [MASK] + noise
            masked_tokens = spatial_tokens.clone()
            mask_expanded = mask.unsqueeze(-1).expand_as(masked_tokens)
            noise = torch.randn_like(masked_tokens) * self.noise_std
            masked_tokens = torch.where(
                mask_expanded,
                masked_tokens,  # keep visible tokens
                self.mask_token.expand(B, N, -1) + noise  # replace masked
            )

            # Add positional encoding + mask indicator
            mask_indicator = mask.long()  # 1=visible, 0=masked
            input_tokens = masked_tokens + self.pos_embed + self.mask_embed(mask_indicator)

            # Decode: use visible tokens as memory, predict all tokens
            decoded = self.decoder(input_tokens, input_tokens)
            decoded = self.norm(decoded)

            # Residual output projection (zero-init → identity start)
            # Use masked_tokens as base so denoiser learns: corrupted → original
            output = masked_tokens + self.output_proj(decoded)

            # Reconstruction loss only on masked positions
            masked_positions = ~mask  # True where masked
            if masked_positions.any():
                recon_loss = F.mse_loss(
                    output[masked_positions],
                    target[masked_positions]
                )
            else:
                recon_loss = torch.tensor(0.0, device=device)

            num_masked = (~mask).float().sum().item() / B
            stats = {
                'recon_loss': recon_loss.item(),
                'num_masked': num_masked,
                'mask_ratio': num_masked / N,
            }

            return output, recon_loss, stats

        else:
            # Test time: use pose confidence to identify occluded regions
            if heatmaps is not None:
                # Low heatmap activation = likely occluded
                hm = F.interpolate(heatmaps, size=(fH, fW), mode='bilinear', align_corners=False)
                hm_max = hm.max(dim=1)[0]  # (B, fH, fW) — max across keypoints
                hm_flat = hm_max.flatten(1)  # (B, N)
                # Tokens with low max heatmap value are likely background/occluded
                threshold = hm_flat.mean(dim=1, keepdim=True)
                mask = hm_flat > threshold * 0.5  # True = likely person, False = likely occluded
            else:
                # No heatmap: don't mask anything
                mask = torch.ones(B, N, dtype=torch.bool, device=device)

            if (~mask).any():
                # Some tokens are occluded → denoise
                masked_tokens = spatial_tokens.clone()
                mask_expanded = mask.unsqueeze(-1).expand_as(masked_tokens)
                masked_tokens = torch.where(
                    mask_expanded,
                    masked_tokens,
                    self.mask_token.expand(B, N, -1)
                )
                mask_indicator = mask.long()
                input_tokens = masked_tokens + self.pos_embed + self.mask_embed(mask_indicator)
                decoded = self.decoder(input_tokens, input_tokens)
                decoded = self.norm(decoded)
                output = masked_tokens + self.output_proj(decoded)
            else:
                output = spatial_tokens

            return output, None, {}
