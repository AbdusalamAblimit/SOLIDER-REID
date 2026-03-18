"""
Sinkhorn Optimal Transport Distance for Person Re-Identification.

Computes differentiable optimal transport distance between two sets of
keypoint features, weighted by pose confidence (visibility).

Key innovation: matching is ADAPTIVE per (query, gallery) pair,
naturally handling different occlusion patterns through transport mass.

References:
- Cuturi (2013): Sinkhorn Distances
- "On Partial Optimal Transport" (AAAI 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornDistance(nn.Module):
    """Differentiable Sinkhorn Optimal Transport distance.

    Given two sets of weighted features (e.g., keypoint features with
    confidence scores), computes the optimal transport cost between them.

    Args:
        eps: Sinkhorn regularization (smaller = sharper transport, but
             less stable). Default 0.1.
        max_iter: Number of Sinkhorn iterations. Default 20.
    """

    def __init__(self, eps=0.1, max_iter=20):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, feat_a, feat_b, w_a, w_b):
        """Compute Sinkhorn OT distance between two feature sets.

        Args:
            feat_a: (B, K, D) keypoint features for set A
            feat_b: (B, K, D) keypoint features for set B
            w_a: (B, K) weights for set A (will be normalized to sum=1)
            w_b: (B, K) weights for set B (will be normalized to sum=1)

        Returns:
            dist: (B,) Sinkhorn distances
        """
        B, K, D = feat_a.shape

        # Normalize features for cosine cost
        fa = F.normalize(feat_a, dim=-1)  # (B, K, D)
        fb = F.normalize(feat_b, dim=-1)  # (B, K, D)

        # Cost matrix: 1 - cosine similarity
        cost = 1.0 - torch.bmm(fa, fb.transpose(1, 2))  # (B, K, K)

        # Normalize weights to probability distributions
        mu = w_a.clamp(min=1e-8)
        mu = mu / mu.sum(dim=1, keepdim=True)  # (B, K)
        nu = w_b.clamp(min=1e-8)
        nu = nu / nu.sum(dim=1, keepdim=True)  # (B, K)

        # Log-domain Sinkhorn for numerical stability
        log_mu = torch.log(mu)  # (B, K)
        log_nu = torch.log(nu)  # (B, K)

        # Kernel matrix in log domain
        M = -cost / self.eps  # (B, K, K)

        # Sinkhorn iterations (log-domain)
        u = torch.zeros_like(log_mu)  # (B, K)
        v = torch.zeros_like(log_nu)  # (B, K)

        for _ in range(self.max_iter):
            # u update: log(mu) - log(sum_j exp(M_ij + v_j))
            u = log_mu - torch.logsumexp(M + v.unsqueeze(1), dim=2)
            # v update: log(nu) - log(sum_i exp(M_ij + u_i))
            v = log_nu - torch.logsumexp(M + u.unsqueeze(2), dim=1)

        # Optimal transport plan
        log_T = M + u.unsqueeze(2) + v.unsqueeze(1)  # (B, K, K)
        T = torch.exp(log_T)

        # Transport cost (Sinkhorn distance)
        dist = (T * cost).sum(dim=(1, 2))  # (B,)

        return dist


class OTTripletLoss(nn.Module):
    """Triplet loss using Sinkhorn OT distance.

    Uses hard mining on pooled features for efficiency, then computes
    triplet loss using OT distance on per-keypoint features.

    Args:
        margin: Triplet margin. Default 0.3.
        eps: Sinkhorn regularization. Default 0.1.
        max_iter: Sinkhorn iterations. Default 20.
    """

    def __init__(self, margin=0.3, eps=0.1, max_iter=20):
        super().__init__()
        self.margin = margin
        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=max_iter)

    def forward(self, kp_feats, kp_weights, labels):
        """Compute OT-based triplet loss.

        Args:
            kp_feats: (B, K, D) per-keypoint features
            kp_weights: (B, K) per-keypoint weights (confidence)
            labels: (B,) identity labels

        Returns:
            loss: scalar triplet loss
        """
        B = kp_feats.shape[0]
        device = kp_feats.device

        # Compute pairwise OT distances (B×B matrix)
        # For efficiency, only compute necessary pairs
        losses = []
        for i in range(B):
            # Find hardest positive (same label, max distance)
            pos_mask = (labels == labels[i])
            pos_mask[i] = False
            if not pos_mask.any():
                continue

            # Find hardest negative (different label, min distance)
            neg_mask = (labels != labels[i])
            if not neg_mask.any():
                continue

            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]

            # Compute OT distance to all positives
            anchor_feats = kp_feats[i:i+1].expand(len(pos_indices), -1, -1)
            anchor_w = kp_weights[i:i+1].expand(len(pos_indices), -1)
            pos_dists = self.sinkhorn(
                anchor_feats, kp_feats[pos_indices],
                anchor_w, kp_weights[pos_indices])

            # Hardest positive
            hp_dist = pos_dists.max()

            # Compute OT distance to all negatives
            anchor_feats_n = kp_feats[i:i+1].expand(len(neg_indices), -1, -1)
            anchor_w_n = kp_weights[i:i+1].expand(len(neg_indices), -1)
            neg_dists = self.sinkhorn(
                anchor_feats_n, kp_feats[neg_indices],
                anchor_w_n, kp_weights[neg_indices])

            # Hardest negative
            hn_dist = neg_dists.min()

            # Triplet loss
            loss_i = F.relu(hp_dist - hn_dist + self.margin)
            losses.append(loss_i)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=device)
