"""
Pose-Query Transformer Decoder (PQTD)

Multi-layer transformer decoder that uses pose-guided learnable queries
to extract part-specific features from backbone feature maps.

Architecture:
  - 5 learnable part queries (head, shoulders, arms, hips, legs)
  - Pose position encoding from heatmap statistics
  - N decoder layers with self-attn + cross-attn + FFN
  - Output: concat of 5 part features -> linear -> branch feature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# COCO 17 keypoints -> 5 body parts
_PART_KP_INDICES = [
    [0, 1, 2, 3, 4],    # head: nose, eyes, ears
    [5, 6],              # shoulders
    [7, 8, 9, 10],       # arms: elbows, wrists
    [11, 12],            # hips
    [13, 14, 15, 16],    # legs: knees, ankles
]


class PoseQueryDecoderLayer(nn.Module):
    """Single decoder layer: self-attn -> cross-attn -> FFN."""

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention: queries attend to backbone features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, memory):
        """
        Args:
            queries: (B, N_queries, d_model) part queries
            memory: (B, H*W, d_model) backbone features (projected)
        Returns:
            queries: (B, N_queries, d_model) updated queries
        """
        # Self-attention
        q2 = self.self_attn(queries, queries, queries)[0]
        queries = self.norm1(queries + self.dropout1(q2))

        # Cross-attention: queries attend to backbone
        q2 = self.cross_attn(queries, memory, memory)[0]
        queries = self.norm2(queries + self.dropout2(q2))

        # FFN
        q2 = self.ffn(queries)
        queries = self.norm3(queries + q2)

        return queries


class PoseQueryDecoder(nn.Module):
    """Pose-Query Transformer Decoder for part feature extraction.

    Replaces GCN branch with a more powerful decoder-based approach.

    Args:
        feat_dim: Backbone feature dimension (768 for Swin-Tiny)
        d_model: Decoder internal dimension
        nhead: Number of attention heads
        num_layers: Number of decoder layers
        num_parts: Number of body part queries
        num_classes: Number of identity classes
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
    """

    def __init__(self, feat_dim=768, d_model=256, nhead=8, num_layers=3,
                 num_parts=5, num_classes=702, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.num_parts = num_parts

        # Project backbone features to decoder dimension
        self.input_proj = nn.Linear(feat_dim, d_model)

        # Learnable part queries
        self.part_queries = nn.Parameter(torch.randn(num_parts, d_model) * 0.02)

        # Pose position encoding: project per-part heatmap stats to query space
        # Each part has a different number of keypoints, but we use a shared MLP
        # Input: per-part spatial statistics (mean_x, mean_y, max_val, coverage) = 4 per part
        self.pose_pe_proj = nn.Linear(4, d_model)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            PoseQueryDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output projection: concat parts -> branch feature
        self.output_proj = nn.Sequential(
            nn.Linear(num_parts * d_model, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # Branch classifier and BN (same structure as GCN branch)
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        self._init_weights()

    def _init_weights(self):
        """Initialize decoder weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.classifier:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _compute_pose_pe(self, scene_heatmaps, hw_shape):
        """Compute per-part pose position encoding from heatmaps.

        Args:
            scene_heatmaps: (B, 17, hH, hW) raw heatmaps
            hw_shape: (H, W) feature map spatial dimensions

        Returns:
            pose_pe: (B, num_parts, d_model) position encoding
        """
        B = scene_heatmaps.shape[0]

        # Resize to feature map size
        if scene_heatmaps.shape[2:] != tuple(hw_shape):
            hm = F.interpolate(scene_heatmaps, size=hw_shape,
                               mode='bilinear', align_corners=False)
        else:
            hm = scene_heatmaps

        hm = torch.sigmoid(hm)  # (B, 17, H, W)
        H, W = hw_shape

        # Compute spatial coordinates grid
        gy = torch.linspace(0, 1, H, device=hm.device)
        gx = torch.linspace(0, 1, W, device=hm.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H, W)

        part_stats = []
        for part_kps in _PART_KP_INDICES:
            # Merge keypoints for this part
            part_hm = hm[:, part_kps].max(dim=1)[0]  # (B, H, W)

            # Spatial statistics
            part_sum = part_hm.sum(dim=(1, 2)).clamp(min=1e-6)  # (B,)

            # Weighted mean position
            mean_y = (part_hm * grid_y.unsqueeze(0)).sum(dim=(1, 2)) / part_sum  # (B,)
            mean_x = (part_hm * grid_x.unsqueeze(0)).sum(dim=(1, 2)) / part_sum  # (B,)

            # Max activation
            max_val = part_hm.amax(dim=(1, 2))  # (B,)

            # Coverage (fraction of spatial area activated > 0.5)
            coverage = (part_hm > 0.5).float().mean(dim=(1, 2))  # (B,)

            # Stack: (B, 4)
            stats = torch.stack([mean_y, mean_x, max_val, coverage], dim=1)
            part_stats.append(stats)

        # (B, num_parts, 4)
        part_stats = torch.stack(part_stats, dim=1)

        # Project to query space: (B, num_parts, d_model)
        pose_pe = self.pose_pe_proj(part_stats)

        return pose_pe

    def forward(self, feat_map, pose_dict, return_cls=False, label=None):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone feature map (detached)
            pose_dict: dict with heatmaps, keypoints, scores, etc.
            return_cls: if True, return classifier logits
            label: identity labels (unused, kept for interface compat)

        Returns:
            cls_scores: list of classifier logits (if return_cls)
            feats: list of [branch_feature]
            aux_data: dict with kp_weights etc.
        """
        B, C, fH, fW = feat_map.shape
        hw_shape = (fH, fW)

        # Project backbone features: (B, C, H, W) -> (B, H*W, d_model)
        memory = feat_map.permute(0, 2, 3, 1).reshape(B, fH * fW, C)
        memory = self.input_proj(memory)

        # Initialize queries: learnable + pose PE
        queries = self.part_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 5, d_model)

        # Add pose position encoding
        scene_hm = pose_dict['heatmaps']
        person_mask = pose_dict['person_mask']
        # Use scene-level merged heatmap
        from .pose_utils import merge_person_heatmaps
        scene_heatmaps = merge_person_heatmaps(scene_hm, person_mask)

        pose_pe = self._compute_pose_pe(scene_heatmaps, hw_shape)  # (B, 5, d_model)
        queries = queries + pose_pe

        # Run decoder layers
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        # Output: concat all part queries -> linear -> branch feat
        # (B, 5, d_model) -> (B, 5*d_model) -> (B, feat_dim)
        branch_feat = queries.reshape(B, self.num_parts * self.d_model)
        branch_feat = self.output_proj(branch_feat)

        # BN
        branch_bn = self.bottleneck(branch_feat)

        if return_cls:
            cls_score = self.classifier(branch_bn)
            return [cls_score], [branch_feat], {}
        else:
            return None, [branch_feat], {}
