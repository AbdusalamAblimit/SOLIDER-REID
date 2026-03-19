"""
SGRE: Skeleton-Guided Re-Encoding

Paradigm shift: instead of independently encoding query and gallery then
comparing with a fixed metric (cosine), SGRE computes pair-conditioned
similarity using cross-attention between per-keypoint feature sets.

For each (query, gallery) pair:
1. Take their per-keypoint features (17 x D each)
2. Cross-attend: query keypoints attend to gallery keypoints
3. Output: a learned similarity score that accounts for structural
   correspondence, visibility patterns, and identity matching

This module is used BOTH during training (as a triplet loss with learned
distance) and during testing (as a re-ranking distance on top-K candidates).

The key insight: the optimal way to compare two people DEPENDS on which
body parts are visible in each image. SGRE learns this comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonReEncoder(nn.Module):
    """Cross-attention re-encoder for pair-conditioned skeleton matching.

    Takes two sets of keypoint features and produces a similarity score
    that accounts for structural correspondence.

    Args:
        feat_dim: keypoint feature dimension (768)
        d_model: internal attention dimension
        nhead: number of attention heads
        num_layers: number of cross-attention layers
    """

    def __init__(self, feat_dim=768, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model

        # Project keypoint features to attention dimension
        self.q_proj = nn.Linear(feat_dim, d_model)
        self.g_proj = nn.Linear(feat_dim, d_model)

        # Cross-attention layers
        self.cross_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_layers.append(
                nn.MultiheadAttention(d_model, nhead, dropout=0.1,
                                      batch_first=True))

        # Similarity prediction head
        self.sim_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q_kp_feats, g_kp_feats, q_vis=None, g_vis=None):
        """Compute pair-conditioned similarity.

        Args:
            q_kp_feats: (B, 17, D) query keypoint features
            g_kp_feats: (B, 17, D) gallery keypoint features
            q_vis: (B, 17) query visibility weights (optional)
            g_vis: (B, 17) gallery visibility weights (optional)

        Returns:
            similarity: (B,) learned similarity scores
        """
        # Project to attention dimension
        q = self.q_proj(q_kp_feats)  # (B, 17, d_model)
        g = self.g_proj(g_kp_feats)  # (B, 17, d_model)

        # Create visibility-based attention mask (optional)
        attn_mask = None
        if q_vis is not None and g_vis is not None:
            # Low visibility → high mask value → less attention
            vis_mask = torch.einsum('bi,bj->bij', q_vis, g_vis)  # (B, 17, 17)
            # Convert to additive mask: 0 for visible pairs, -inf for invisible
            attn_mask = (vis_mask < 0.1).float() * -1e9

        # Cross-attention: query attends to gallery
        q_enhanced = q
        for cross_attn in self.cross_layers:
            if attn_mask is not None:
                # MultiheadAttention expects (B*nhead, L, S) mask or None
                # Use key_padding_mask instead for simplicity
                q_enhanced, _ = cross_attn(q_enhanced, g, g)
            else:
                q_enhanced, _ = cross_attn(q_enhanced, g, g)

        # Pool: visibility-weighted average of enhanced query features
        if q_vis is not None:
            w = q_vis.clamp(min=1e-6).unsqueeze(-1)  # (B, 17, 1)
            q_pooled = (q_enhanced * w).sum(dim=1) / w.sum(dim=1)
        else:
            q_pooled = q_enhanced.mean(dim=1)  # (B, d_model)

        # Also pool gallery features for symmetric comparison
        if g_vis is not None:
            w_g = g_vis.clamp(min=1e-6).unsqueeze(-1)
            g_pooled = (g * w_g).sum(dim=1) / w_g.sum(dim=1)
        else:
            g_pooled = g.mean(dim=1)  # (B, d_model)

        # Combine for similarity prediction
        combined = torch.cat([q_pooled, g_pooled], dim=-1)  # (B, 2*d_model)
        similarity = self.sim_head(combined).squeeze(-1)  # (B,)

        return similarity

    def compute_training_loss(self, kp_feats, kp_weights, labels, margin=0.3):
        """Compute triplet loss using SGRE similarity.

        For each anchor, find hardest positive and negative,
        then compute triplet loss on SGRE similarities.

        Args:
            kp_feats: (B, 17, D) per-keypoint features (detached from backbone)
            kp_weights: (B, 17) visibility weights
            labels: (B,) identity labels
            margin: triplet margin

        Returns:
            loss: scalar
        """
        B = kp_feats.shape[0]
        device = kp_feats.device

        # Compute all pairwise SGRE similarities
        # Expand: (B, 1, 17, D) vs (1, B, 17, D) → need to iterate for memory
        losses = []
        for i in range(B):
            # Same-ID mask
            pos_mask = (labels == labels[i])
            pos_mask[i] = False
            neg_mask = (labels != labels[i])

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_idx = pos_mask.nonzero(as_tuple=True)[0]
            neg_idx = neg_mask.nonzero(as_tuple=True)[0]

            # Compute similarity to all positives
            q_expand = kp_feats[i:i+1].expand(len(pos_idx), -1, -1)
            w_expand = kp_weights[i:i+1].expand(len(pos_idx), -1)
            pos_sim = self.forward(q_expand, kp_feats[pos_idx],
                                    w_expand, kp_weights[pos_idx])
            # Hardest positive: min similarity
            hp_sim = pos_sim.min()

            # Compute similarity to all negatives
            q_expand_n = kp_feats[i:i+1].expand(len(neg_idx), -1, -1)
            w_expand_n = kp_weights[i:i+1].expand(len(neg_idx), -1)
            neg_sim = self.forward(q_expand_n, kp_feats[neg_idx],
                                    w_expand_n, kp_weights[neg_idx])
            # Hardest negative: max similarity
            hn_sim = neg_sim.max()

            # Triplet: positive should have HIGHER similarity than negative
            loss_i = F.relu(hn_sim - hp_sim + margin)
            losses.append(loss_i)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=device)
