"""Supervised Contrastive Loss for Per-Token Features.

Pulls same-ID features together and pushes different-ID features apart
on the unit hypersphere. Unlike CE which learns classification boundaries,
SupCon directly optimizes the feature metric space — better for retrieval.

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    For each anchor, pulls all same-ID samples close and pushes all
    different-ID samples away, using temperature-scaled cosine similarity.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) L2-normalized feature vectors
            labels: (B,) identity labels

        Returns:
            loss: scalar
        """
        device = features.device
        B = features.shape[0]

        # L2 normalize (should already be normalized, but ensure)
        features = F.normalize(features, p=2, dim=1)

        # Cosine similarity matrix: (B, B)
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask: same identity (excluding self)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = label_eq & self_mask  # same ID, not self
        neg_mask = ~label_eq  # different ID

        # Check that each sample has at least one positive
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return features.sum() * 0.0  # no valid pairs

        # For numerical stability: subtract max from logits
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Denominator: sum over all non-self samples (positives + negatives)
        all_mask = self_mask.float()
        exp_sim = torch.exp(sim) * all_mask
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Numerator: log(exp(sim)) = sim for each positive pair
        # SupCon loss = -1/|P(i)| * sum_{p in P(i)} [sim(i,p) - log(sum_a exp(sim(i,a)))]
        log_prob = sim - log_denom  # (B, B)

        # Average over positive pairs for each anchor
        pos_mask_float = pos_mask.float()
        num_pos = pos_mask_float.sum(dim=1).clamp(min=1)  # (B,)
        mean_log_prob_pos = (log_prob * pos_mask_float).sum(dim=1) / num_pos

        # Loss: negative mean log prob, averaged over valid anchors
        loss = -mean_log_prob_pos[has_pos].mean()

        return loss
