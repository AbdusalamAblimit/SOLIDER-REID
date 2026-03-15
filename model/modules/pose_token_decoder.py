"""
Pose-Token Distillation Decoder (PTD)

K learnable part tokens attend to backbone features via cross-attention.
During training, attention maps are supervised by pose heatmaps.
During inference, tokens self-localize without pose input.

"Train with Pose, Infer without Pose"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Body part grouping for heatmap supervision (COCO 17 keypoints)
PART_GROUPS = {
    0: [0, 1, 2, 3, 4],       # head: nose, eyes, ears
    1: [5, 6, 11, 12],        # torso: shoulders + hips
    2: [5, 7, 9],             # left arm: shoulder, elbow, wrist
    3: [6, 8, 10],            # right arm
    4: [11, 13, 15, 12, 14, 16],  # legs: hips, knees, ankles
}


class PoseTokenDecoder(nn.Module):
    """Decoder that uses learnable part tokens to extract part features.

    Args:
        feat_dim: backbone feature dimension (768 for Swin-Tiny)
        num_parts: number of part tokens (K)
        attn_dim: cross-attention hidden dimension
        num_heads: number of attention heads
        num_layers: number of decoder layers
        num_classes: number of identity classes
    """

    def __init__(self, feat_dim=768, num_parts=5, attn_dim=256,
                 num_heads=8, num_layers=2, num_classes=702,
                 heatmap_loss_weight=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_parts = num_parts
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.heatmap_loss_weight = heatmap_loss_weight

        # Learnable part tokens
        self.part_tokens = nn.Parameter(
            torch.randn(num_parts, attn_dim) * 0.02)

        # Feature projection to attn_dim (shared input for decoder layers' K/V)
        self.feat_proj = nn.Linear(feat_dim, attn_dim)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(attn_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output projection back to feat_dim
        self.out_proj = nn.Linear(attn_dim, feat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # BN + Classifier for part features
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

        self.bn.apply(self._init_bn)

    @staticmethod
    def _init_bn(m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def _get_heatmap_target(self, scene_heatmaps, fH, fW):
        """Generate per-part attention targets from heatmaps.

        Args:
            scene_heatmaps: (B, 17, H_hm, W_hm) raw heatmaps
            fH, fW: feature map spatial dimensions

        Returns:
            part_targets: (B, K, fH*fW) target attention distribution per part
        """
        B = scene_heatmaps.shape[0]

        # Resize heatmaps to feature map size
        hm = F.interpolate(scene_heatmaps, size=(fH, fW),
                           mode='bilinear', align_corners=False)
        hm = torch.sigmoid(hm)  # normalize to [0, 1]

        # Group channels by body part
        targets = []
        for k in range(self.num_parts):
            channels = PART_GROUPS[k]
            part_hm = hm[:, channels, :, :].max(dim=1)[0]  # (B, fH, fW)
            part_hm = part_hm.reshape(B, fH * fW)
            # Normalize to sum=1 (attention distribution)
            part_hm = part_hm / (part_hm.sum(dim=-1, keepdim=True) + 1e-6)
            targets.append(part_hm)

        return torch.stack(targets, dim=1)  # (B, K, fH*fW)

    def forward(self, feat_map, scene_heatmaps=None, return_cls=True):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone feature map
            scene_heatmaps: (B, 17, H_hm, W_hm) pose heatmaps (only for training)
            return_cls: whether to return classification scores

        Returns:
            cls_scores: [cls_score] list
            feats: [part_feat] list
            aux_data: dict with attention maps and heatmap loss
        """
        B, C, fH, fW = feat_map.shape

        # Flatten spatial dims: (B, fH*fW, C)
        feat_flat = feat_map.flatten(2).permute(0, 2, 1)

        # Project features to attn_dim
        kv_input = self.feat_proj(feat_flat)  # (B, fH*fW, attn_dim)

        # Expand part tokens: (B, K, attn_dim)
        Q = self.part_tokens.unsqueeze(0).expand(B, -1, -1)

        # Run decoder layers, collect attention weights
        all_attn_weights = []
        for layer in self.decoder_layers:
            Q, attn_weights = layer(Q, kv_input)
            all_attn_weights.append(attn_weights)  # (B, K, fH*fW)

        # Project back to feat_dim
        part_feats = self.out_proj(Q)  # (B, K, feat_dim)

        # Pool part features: mean across K parts
        pooled_feat = part_feats.mean(dim=1)  # (B, feat_dim)

        # Compute heatmap distillation loss (training only)
        # Use KL divergence for distribution alignment (more appropriate than MSE)
        heatmap_loss = None
        if self.training and scene_heatmaps is not None:
            targets = self._get_heatmap_target(scene_heatmaps, fH, fW)
            # Use last layer's attention weights
            attn = all_attn_weights[-1]  # (B, K, fH*fW)
            # KL divergence: target * log(target / pred)
            # Use F.kl_div with log_input=True: expects log(pred)
            log_attn = (attn + 1e-8).log()
            heatmap_loss = F.kl_div(log_attn, targets, reduction='batchmean') * self.heatmap_loss_weight

        aux_data = {
            'kp_feats': part_feats,  # (B, K, C) for compatibility
            'kp_weights': torch.ones(B, self.num_parts, device=feat_map.device),
        }
        if heatmap_loss is not None:
            aux_data['ptd_heatmap_loss'] = heatmap_loss

        if return_cls:
            feat_bn = self.bn(pooled_feat)
            cls_score = self.classifier(feat_bn)
            return [cls_score], [pooled_feat], aux_data
        else:
            return None, [pooled_feat], aux_data


class DecoderLayer(nn.Module):
    """Single cross-attention decoder layer with proper per-layer Q/K/V projections."""

    def __init__(self, attn_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Per-layer Q/K/V linear projections (standard multi-head attention)
        self.q_proj = nn.Linear(attn_dim, attn_dim)
        self.k_proj = nn.Linear(attn_dim, attn_dim)
        self.v_proj = nn.Linear(attn_dim, attn_dim)
        self.out_proj_attn = nn.Linear(attn_dim, attn_dim)

        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm_kv = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.GELU(),
            nn.Linear(attn_dim * 4, attn_dim),
        )

    def forward(self, Q, kv_input):
        """
        Args:
            Q: (B, K, D) part token queries
            kv_input: (B, N, D) feature map for K/V

        Returns:
            Q: updated queries (B, K, D)
            attn_weights: (B, K, N) attention weights
        """
        B, K_tokens, D = Q.shape
        N = kv_input.shape[1]

        # Layer norm
        Q_norm = self.norm1(Q)
        kv_norm = self.norm_kv(kv_input)

        # Per-layer Q/K/V projections
        q = self.q_proj(Q_norm).view(B, K_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Average attention across heads for supervision
        attn_weights = attn.mean(dim=1)  # (B, K, N)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, K_tokens, D)
        out = self.out_proj_attn(out)

        # Residual + FFN
        Q = Q + out
        Q = Q + self.ffn(self.norm2(Q))

        return Q, attn_weights
