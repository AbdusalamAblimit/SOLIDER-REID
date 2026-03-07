"""Pose-Guided Part Transformer Decoder (PPTD).

Replaces Gaussian attention pooling with learned cross-attention for extracting
part features from backbone feature maps. Key improvements over Gaussian pooling:
1. Learned attention: queries adapt to capture part-specific discriminative features
2. Pose-guided bias: attention is biased toward body region using keypoint positions
3. Part self-attention: visible parts can share info with occluded parts (completion)
4. Higher capacity: multi-head attention captures richer part representations

Architecture:
    5 learnable part queries → Cross-Attention(Q=queries, KV=feature_map_tokens)
    → Self-Attention(parts attend to each other) → Part features [B, 5, C]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# COCO 17 keypoint -> 5 body part groupings
COCO_PART_GROUPS = [
    [0, 1, 2, 3, 4],      # head
    [5, 6, 11, 12],        # torso
    [7, 8, 9, 10],         # arms
    [13, 14],              # thighs
    [15, 16],              # calves
]


class PartDecoderLayer(nn.Module):
    """Single decoder layer: cross-attention + self-attention + FFN."""

    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        # Cross-attention: part queries attend to spatial features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention: parts attend to each other
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, part_queries, spatial_tokens, attn_bias=None):
        """
        Args:
            part_queries: [B, K, C] part query tokens
            spatial_tokens: [B, N, C] flattened feature map tokens (N=H*W)
            attn_bias: [B, K, N] pose-guided attention bias (optional, additive)

        Returns:
            part_queries: [B, K, C] updated part queries
        """
        # Cross-attention: queries attend to spatial features
        # Prepare attn_mask: expand [B, K, N] -> [B*nhead, K, N]
        attn_mask = None
        if attn_bias is not None:
            B = attn_bias.shape[0]
            attn_mask = attn_bias.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            attn_mask = attn_mask.reshape(B * self.nhead, attn_bias.shape[1], attn_bias.shape[2])

        q = self.norm1(part_queries)
        cross_out, _ = self.cross_attn(q, spatial_tokens, spatial_tokens,
                                        attn_mask=attn_mask)
        part_queries = part_queries + cross_out

        # Self-attention: parts attend to each other
        q = self.norm2(part_queries)
        self_out, _ = self.self_attn(q, q, q)
        part_queries = part_queries + self_out

        # FFN
        part_queries = part_queries + self.ffn(self.norm3(part_queries))

        return part_queries


class PosePartDecoder(nn.Module):
    """Pose-Guided Part Transformer Decoder.

    Uses learnable part queries + cross-attention to extract high-quality
    part features from the backbone feature map, with pose-guided attention bias.

    Args:
        d_model: Feature dimension (must match backbone output)
        n_parts: Number of body parts
        n_layers: Number of decoder layers
        nhead: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        sigma: Gaussian sigma for pose attention bias
        img_size: (H, W) of input image
        use_pose_bias: Whether to apply pose-guided attention bias
        grad_scale: Gradient scaling for decoder parameters (0.1 = 10% gradient to backbone)
    """

    def __init__(self, d_model=768, n_parts=5, n_layers=2, nhead=8,
                 dim_feedforward=2048, dropout=0.1, sigma=3.0,
                 img_size=(384, 128), use_pose_bias=True, grad_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.n_parts = n_parts
        self.sigma = sigma
        self.img_h, self.img_w = img_size
        self.part_groups = COCO_PART_GROUPS[:n_parts]
        self.use_pose_bias = use_pose_bias
        self.grad_scale = grad_scale

        # Learnable part queries (initialized from standard normal, scaled)
        self.part_queries = nn.Parameter(
            torch.randn(1, n_parts, d_model) * 0.02
        )

        # LayerNorm for spatial tokens (K/V normalization)
        self.memory_norm = nn.LayerNorm(d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            PartDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Pose bias strength (learnable)
        if use_pose_bias:
            self.bias_scale = nn.Parameter(torch.tensor(1.0))

        # Build and register positional encoding for 12x4 feature map
        pe = self._build_pos_embed(12, 4)
        self.register_buffer('pos_embed', pe)
        self._cached_hw = (12, 4)

    def _build_pos_embed(self, H, W):
        """Build 2D sinusoidal positional encoding."""
        d = self.d_model
        pe = torch.zeros(H, W, d)
        d_half = d // 2

        # Y encoding
        pos_y = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        div_y = torch.exp(torch.arange(0, d_half, 2, dtype=torch.float32) * (-math.log(10000.0) / d_half))
        pe[:, :, 0:d_half:2] = torch.sin(pos_y * div_y).unsqueeze(1).expand(-1, W, -1)
        pe[:, :, 1:d_half:2] = torch.cos(pos_y * div_y).unsqueeze(1).expand(-1, W, -1)

        # X encoding
        pos_x = torch.arange(W, dtype=torch.float32).unsqueeze(1)
        div_x = torch.exp(torch.arange(0, d_half, 2, dtype=torch.float32) * (-math.log(10000.0) / d_half))
        pe[:, :, d_half::2] = torch.sin(pos_x * div_x).unsqueeze(0).expand(H, -1, -1)
        pe[:, :, d_half+1::2] = torch.cos(pos_x * div_x).unsqueeze(0).expand(H, -1, -1)

        return pe.view(H * W, d).unsqueeze(0)  # [1, N, C]

    def _get_pos_embed(self, H, W, device):
        """Get positional encoding, rebuilding if size changed."""
        if self._cached_hw == (H, W):
            return self.pos_embed
        # Rebuild for different size (shouldn't happen in practice)
        pe = self._build_pos_embed(H, W).to(device)
        self.pos_embed = pe
        self._cached_hw = (H, W)
        return self.pos_embed

    def _compute_pose_bias(self, keypoints, visibility, H, W, device):
        """Compute pose-guided attention bias for cross-attention.

        Creates a soft spatial prior: each part query attends more strongly
        to the spatial locations near its keypoints.

        Returns:
            bias: [B, K, N] where N = H*W
        """
        B = keypoints.shape[0]
        K = self.n_parts

        # Scale keypoints to feature map space
        scale_x = W / self.img_w
        scale_y = H / self.img_h
        kp_x = keypoints[:, :, 0].float() * scale_x  # [B, 17]
        kp_y = keypoints[:, :, 1].float() * scale_y  # [B, 17]

        # Coordinate grid
        gy = torch.arange(H, device=device, dtype=torch.float32)
        gx = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid_x = grid_x.reshape(-1)  # [N]
        grid_y = grid_y.reshape(-1)  # [N]
        N = H * W

        bias = torch.zeros(B, K, N, device=device)

        for k, group in enumerate(self.part_groups):
            for kp_idx in group:
                cx = kp_x[:, kp_idx].unsqueeze(1)  # [B, 1]
                cy = kp_y[:, kp_idx].unsqueeze(1)  # [B, 1]
                vis = visibility[:, kp_idx].unsqueeze(1)  # [B, 1]

                dx = grid_x.unsqueeze(0) - cx  # [B, N]
                dy = grid_y.unsqueeze(0) - cy  # [B, N]
                gauss = torch.exp(-(dx**2 + dy**2) / (2 * self.sigma**2))  # [B, N]

                bias[:, k] += gauss * vis  # [B, N]

            # Normalize per-part bias
            bias_max = bias[:, k].amax(dim=1, keepdim=True).clamp(min=1e-6)
            bias[:, k] = bias[:, k] / bias_max

        return bias * self.bias_scale  # [B, K, N]

    def forward(self, feat_map, keypoints, visibility):
        """
        Args:
            feat_map: [B, C, H, W] backbone feature map
            keypoints: [B, 17, 2] keypoint coords in image space
            visibility: [B, 17] per-keypoint visibility scores

        Returns:
            part_feats: [B, K, C] decoded part features
            part_vis: [B, K] part visibility (mean of keypoints in group)
        """
        if isinstance(feat_map, (list, tuple)):
            feat_map = feat_map[-1]

        B, C, H, W = feat_map.shape
        device = feat_map.device

        # Flatten feature map to tokens
        spatial_tokens = feat_map.flatten(2).transpose(1, 2)  # [B, N, C]

        # Add positional encoding and normalize
        pos = self._get_pos_embed(H, W, device)
        spatial_tokens = self.memory_norm(spatial_tokens + pos)

        # Expand part queries for batch
        queries = self.part_queries.expand(B, -1, -1)  # [B, K, C]

        # Compute pose-guided attention bias
        pose_bias = None
        if self.use_pose_bias:
            pose_bias = self._compute_pose_bias(keypoints, visibility, H, W, device)

        # Apply decoder layers
        for layer in self.layers:
            queries = layer(queries, spatial_tokens, attn_bias=pose_bias)

        # Final norm
        part_feats = self.norm(queries)  # [B, K, C]

        # Compute part visibility
        part_vis = torch.zeros(B, self.n_parts, device=device)
        for k, group in enumerate(self.part_groups):
            group_vis = torch.stack([visibility[:, idx] for idx in group], dim=1)
            part_vis[:, k] = group_vis.mean(dim=1)

        return part_feats, part_vis
