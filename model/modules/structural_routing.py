"""Structural Token Decomposition with Pose-guided Routing (STD-PR).

Converts spatial feature tokens into structural body-part tokens via
pose-guided cross-attention. Replaces GCN's point-sampling + graph
propagation with attention-based part feature aggregation.

Each structural token learns to "gather" information from its
corresponding body region in the spatial feature map, guided by
pose heatmap attention bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Default 6 body-part groups (COCO-17 keypoint indices)
BODY_PART_GROUPS = {
    'head': [0, 1, 2, 3, 4],
    'torso': [5, 6, 11, 12],
    'left_arm': [5, 7, 9],
    'right_arm': [6, 8, 10],
    'left_leg': [11, 13, 15],
    'right_leg': [12, 14, 16],
}


class StructuralRoutingLayer(nn.Module):
    """Convert spatial tokens to structural body-part tokens.

    Uses pose-guided cross-attention: K learnable part queries attend
    to spatial feature tokens, with attention biased by pose heatmaps.
    """

    def __init__(self, feat_dim, num_parts=6, num_heads=8, num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.num_parts = num_parts
        self.feat_dim = feat_dim

        # Learnable part query embeddings
        self.part_queries = nn.Parameter(torch.randn(num_parts, feat_dim) * 0.02)

        # Pose heatmap → per-part attention bias
        # Maps 17 keypoint channels to num_parts channels
        self.pose_to_part_bias = nn.Sequential(
            nn.Conv2d(17, num_parts, kernel_size=1, bias=True),
        )
        # Initialize near zero so initial attention is unbiased
        nn.init.zeros_(self.pose_to_part_bias[0].weight)
        nn.init.zeros_(self.pose_to_part_bias[0].bias)

        # Cross-attention layers (part queries attend to spatial tokens)
        self.cross_attn_layers = nn.ModuleList()
        self.cross_norms_q = nn.ModuleList()
        self.cross_norms_kv = nn.ModuleList()
        self.cross_ffns = nn.ModuleList()
        self.cross_ffn_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(feat_dim, num_heads, dropout=dropout,
                                     batch_first=True))
            self.cross_norms_q.append(nn.LayerNorm(feat_dim))
            self.cross_norms_kv.append(nn.LayerNorm(feat_dim))
            self.cross_ffns.append(nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feat_dim * 4, feat_dim),
                nn.Dropout(dropout),
            ))
            self.cross_ffn_norms.append(nn.LayerNorm(feat_dim))

        # Part classifier (for training)
        self.part_bn = nn.BatchNorm1d(feat_dim)
        self.part_bn.bias.requires_grad_(False)

    def forward(self, spatial_tokens, hw_shape, scene_heatmaps=None):
        """
        Args:
            spatial_tokens: (B, N, C) spatial feature tokens from backbone
            hw_shape: (H, W) spatial grid dimensions
            scene_heatmaps: (B, 17, hm_H, hm_W) or None

        Returns:
            structural_tokens: (B, K, C) body-part tokens
            stats: dict with diagnostic info
        """
        B, N, C = spatial_tokens.shape
        H, W = hw_shape
        K = self.num_parts

        # Initialize queries
        queries = self.part_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, C)

        # Compute pose-guided attention bias
        attn_bias = None
        if scene_heatmaps is not None:
            hm = F.interpolate(scene_heatmaps, size=(H, W),
                               mode='bilinear', align_corners=False)
            # (B, 17, H, W) → (B, K, H, W) → (B, K, N)
            part_bias = self.pose_to_part_bias(hm)  # (B, K, H, W)
            attn_bias = part_bias.view(B, K, N)  # (B, K, N)
            # Expand for multi-head: (B*num_heads, K, N)
            num_heads = self.cross_attn_layers[0].num_heads
            attn_bias = attn_bias.unsqueeze(1).expand(-1, num_heads, -1, -1)
            attn_bias = attn_bias.reshape(B * num_heads, K, N)

        # Cross-attention layers
        for i, (attn, norm_q, norm_kv, ffn, ffn_norm) in enumerate(zip(
                self.cross_attn_layers, self.cross_norms_q,
                self.cross_norms_kv, self.cross_ffns, self.cross_ffn_norms)):
            # Cross-attention: queries attend to spatial tokens
            q = norm_q(queries)
            kv = norm_kv(spatial_tokens)
            attn_out = attn(q, kv, kv, attn_mask=attn_bias)[0]
            queries = queries + attn_out

            # FFN
            queries = queries + ffn(ffn_norm(queries))

        structural_tokens = queries  # (B, K, C)

        # Compute stats
        with torch.no_grad():
            token_norms = structural_tokens.norm(dim=-1).mean().item()

        stats = {
            'token_norm': token_norms,
            'num_parts': K,
        }

        return structural_tokens, stats

    def get_part_features(self, structural_tokens):
        """Pool structural tokens into a single part feature vector.

        Args:
            structural_tokens: (B, K, C)
        Returns:
            part_feat: (B, C) averaged part feature
        """
        return structural_tokens.mean(dim=1)  # (B, C)
