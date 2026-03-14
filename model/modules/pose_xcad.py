"""Pose Cross-Attention Decoder (XCAD) for keypoint feature extraction.

Instead of bilinear sampling + GCN graph propagation, uses cross-attention
where keypoint-conditioned queries attend to the full backbone feature map.

Each keypoint query can attend to ALL spatial positions, enabling:
1. Broader receptive field than single-point bilinear sampling
2. Multi-head attention for multi-view per-keypoint feature extraction
3. Content-adaptive spatial selection (handle occlusion dynamically)

Zero-initialized output with bilinear-sampled residual ensures identity start.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionEnhancer(nn.Module):
    """Cross-attention: keypoint queries attend to backbone feature map.

    Replaces GCN graph propagation with global cross-attention.
    Residual connection: output = bilinear_sampled + cross_attn_update.

    Args:
        feat_dim: backbone feature dimension (768)
        attn_dim: internal attention dimension (256)
        num_heads: number of attention heads (8)
        num_keypoints: number of keypoints (17)
        input_size: (H, W) of input images for coordinate normalization
    """

    def __init__(self, feat_dim=768, attn_dim=256, num_heads=8,
                 num_keypoints=17, input_size=(384, 128)):
        super().__init__()
        self.feat_dim = feat_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.num_keypoints = num_keypoints
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.input_h, self.input_w = input_size

        # Learnable keypoint query tokens
        self.kp_queries = nn.Parameter(
            torch.randn(num_keypoints, attn_dim) * 0.02)

        # Position encoding: keypoint pixel coords -> attn_dim
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, attn_dim),
        )

        # Project backbone features to K, V
        self.kv_proj = nn.Linear(feat_dim, attn_dim * 2)

        # Output projection: attn_dim -> feat_dim (zero-init for residual)
        self.out_proj = nn.Linear(attn_dim, feat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # LayerNorm on output
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, feat_map, kp_feats_base, keypoints, scores):
        """Cross-attention from keypoint queries to feature map.

        Args:
            feat_map: (B, C, fH, fW) backbone feature map
            kp_feats_base: (B, 17, C) bilinear-sampled keypoint features
            keypoints: (B, 17, 2) pixel coordinates (person 0)
            scores: (B, 17) confidence scores

        Returns:
            kp_feats_enhanced: (B, 17, C) enhanced keypoint features
        """
        B, C, fH, fW = feat_map.shape

        # Flatten feature map: (B, H*W, C)
        feat_flat = feat_map.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        N_tokens = feat_flat.shape[1]

        # Project to K, V: (B, H*W, attn_dim*2)
        kv = self.kv_proj(feat_flat)
        K, V = kv.chunk(2, dim=-1)  # each (B, H*W, attn_dim)

        # Build queries: learnable + position encoding from keypoints
        # Normalize keypoint coords to [-1, 1]
        kp_norm = torch.stack([
            keypoints[:, :, 0] / self.input_w * 2 - 1,
            keypoints[:, :, 1] / self.input_h * 2 - 1,
        ], dim=-1)  # (B, 17, 2)

        pos_enc = self.pos_encoder(kp_norm)  # (B, 17, attn_dim)
        Q = self.kp_queries.unsqueeze(0).expand(B, -1, -1) + pos_enc
        # (B, 17, attn_dim)

        # Mask low-confidence keypoints: zero out their queries
        # This prevents unreliable keypoints from collecting features
        score_mask = (scores > 0.3).float().unsqueeze(-1)  # (B, 17, 1)
        Q = Q * score_mask

        # Multi-head attention
        # Reshape: (B, N, attn_dim) -> (B, num_heads, N, head_dim)
        Q = Q.view(B, self.num_keypoints, self.num_heads,
                    self.head_dim).transpose(1, 2)
        K = K.view(B, N_tokens, self.num_heads,
                    self.head_dim).transpose(1, 2)
        V = V.view(B, N_tokens, self.num_heads,
                    self.head_dim).transpose(1, 2)

        # Attention: (B, heads, 17, H*W)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Aggregate: (B, heads, 17, head_dim)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(
            B, self.num_keypoints, self.attn_dim)

        # Project back to feat_dim (zero-init → starts as zero)
        update = self.out_proj(out)  # (B, 17, C)

        # Residual: bilinear-sampled base + cross-attn update
        kp_feats = self.norm(kp_feats_base + update)

        return kp_feats


class PoseCrossAttnHead(nn.Module):
    """Complete head: bilinear sample → cross-attention → confidence-weighted avg.

    Drop-in replacement for SkeletonGCNHead. Same interface.

    Args:
        feat_dim: backbone feature dimension (768)
        attn_dim: cross-attention internal dimension (256)
        num_heads: number of attention heads (8)
        num_classes: number of identity classes
        input_size: (H, W) of input images
    """

    def __init__(self, feat_dim, attn_dim, num_heads, num_classes,
                 input_size=(384, 128)):
        super().__init__()
        self.feat_dim = feat_dim
        self.input_h, self.input_w = input_size
        self.num_joints = 17

        # Cross-attention enhancer (replaces GCN)
        self.cross_attn = CrossAttentionEnhancer(
            feat_dim=feat_dim,
            attn_dim=attn_dim,
            num_heads=num_heads,
            num_keypoints=17,
            input_size=input_size,
        )

        # BN + Classifier for part feature
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

        # Initialize BN
        self.bn.apply(self._init_bn)

    @staticmethod
    def _init_bn(m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def _sample_keypoint_features(self, feat_map, keypoints, scores):
        """Bilinear sample features at keypoint locations.

        Args:
            feat_map: (B, C, fH, fW) feature map
            keypoints: (B, 17, 2) pixel coordinates (person 0)
            scores: (B, 17) confidence scores

        Returns:
            kp_feats: (B, 17, C) sampled features
        """
        B, C, fH, fW = feat_map.shape

        # Normalize to [-1, 1] for grid_sample
        grid_x = keypoints[:, :, 0] / self.input_w * 2 - 1  # (B, 17)
        grid_y = keypoints[:, :, 1] / self.input_h * 2 - 1  # (B, 17)
        grid_x = grid_x.clamp(-1, 1)
        grid_y = grid_y.clamp(-1, 1)

        # Build grid: (B, 17, 1, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)

        # Sample: (B, C, 17, 1) -> (B, 17, C)
        sampled = F.grid_sample(
            feat_map, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )
        kp_feats = sampled.squeeze(-1).permute(0, 2, 1)

        return kp_feats

    def forward(self, feat_map, pose_dict, return_cls=True, label=None):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone feature map
            pose_dict: dict with keypoints, scores, person_mask
            return_cls: whether to return classification scores

        Returns:
            Same interface as SkeletonGCNHead
        """
        # Extract person 0 keypoints and scores
        keypoints = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2)
        kp_scores = pose_dict['scores'][:, 0, :]  # (B, 17)

        # 1. Bilinear sample at keypoint locations (base features)
        kp_feats_base = self._sample_keypoint_features(
            feat_map, keypoints, kp_scores)

        # 2. Cross-attention enhancement (replaces GCN)
        kp_feats_enhanced = self.cross_attn(
            feat_map, kp_feats_base, keypoints, kp_scores)

        # 3. Confidence-weighted average
        weights = kp_scores.clamp(min=1e-6).unsqueeze(-1)  # (B, 17, 1)
        part_feat = (kp_feats_enhanced * weights).sum(dim=1) / \
                    weights.sum(dim=1).clamp(min=1e-6)  # (B, C)

        aux_data = {
            'kp_feats': kp_feats_enhanced,  # (B, 17, C)
            'kp_weights': kp_scores,         # (B, 17)
        }

        if return_cls:
            feat_bn = self.bn(part_feat)
            aux_data['feat_bn'] = feat_bn
            cls_score = self.classifier(feat_bn)
            return [cls_score], [part_feat], aux_data
        else:
            return None, [part_feat], aux_data
