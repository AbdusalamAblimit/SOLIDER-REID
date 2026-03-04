"""Part-level losses for VPReID.

PartAveragedTripletLoss: visibility-weighted triplet loss across body parts.
PushLoss: encourages diversity between part features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartAveragedTripletLoss(nn.Module):
    """Triplet loss averaged across visible body parts.

    For each part k, compute triplet loss using only samples where part k
    is visible. The final loss is the average across parts weighted by
    the fraction of visible samples.

    Uses soft margin (log(1 + exp(d_pos - d_neg))) when margin is None,
    otherwise hard margin max(0, d_pos - d_neg + margin).
    Features are L2-normalized before distance computation to prevent explosion.
    """

    def __init__(self, margin=None, normalize=True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, part_feats, labels, part_vis=None):
        """
        Args:
            part_feats: [B, K, D] per-part features (BN-normalized)
            labels: [B] identity labels
            part_vis: [B, K] visibility scores (optional)

        Returns:
            loss: scalar
        """
        B, K, D = part_feats.shape
        device = part_feats.device

        part_losses = []
        part_weights = []

        for k in range(K):
            feat_k = part_feats[:, k]  # [B, D]

            if self.normalize:
                feat_k = F.normalize(feat_k, p=2, dim=1)

            # Optional visibility masking
            if part_vis is not None:
                vis_k = part_vis[:, k]  # [B]
                vis_mask = vis_k > 0.5  # binary threshold
                n_vis = vis_mask.sum().item()
                if n_vis < 2:
                    continue
                feat_k = feat_k[vis_mask]
                labels_k = labels[vis_mask]
                weight = n_vis / B
            else:
                labels_k = labels
                weight = 1.0

            # Pairwise distances
            dist = torch.cdist(feat_k.unsqueeze(0), feat_k.unsqueeze(0)).squeeze(0)  # [N, N]

            # Mine hardest positive and negative for each sample
            N = feat_k.shape[0]
            is_pos = labels_k.unsqueeze(0) == labels_k.unsqueeze(1)
            is_neg = ~is_pos

            # Mask diagonal
            is_pos.fill_diagonal_(False)

            # Check if we have valid pairs
            if not is_pos.any() or not is_neg.any():
                continue

            # Hardest positive distance
            dist_ap = dist.clone()
            dist_ap[~is_pos] = 0
            dist_ap, _ = dist_ap.max(dim=1)  # [N]

            # Hardest negative distance
            dist_an = dist.clone()
            dist_an[~is_neg] = float('inf')
            dist_an, _ = dist_an.min(dim=1)  # [N]

            # Filter valid samples (have both pos and neg)
            valid = (dist_ap > 0) & (dist_an < float('inf'))
            if valid.sum() < 1:
                continue

            dist_ap = dist_ap[valid]
            dist_an = dist_an[valid]

            if self.margin is not None:
                y = torch.ones_like(dist_an)
                loss_k = self.ranking_loss(dist_an, dist_ap, y)
            else:
                y = torch.ones_like(dist_an)
                loss_k = self.ranking_loss(dist_ap - dist_an, y)

            part_losses.append(loss_k)
            part_weights.append(weight)

        if not part_losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Weighted average across parts
        total_weight = sum(part_weights)
        loss = sum(l * w for l, w in zip(part_losses, part_weights)) / total_weight
        return loss


class PushLoss(nn.Module):
    """Diversity loss that pushes part features apart.

    Penalizes high cosine similarity between different part features
    of the same sample, encouraging each part to capture different information.
    """

    def __init__(self):
        super().__init__()

    def forward(self, part_feats):
        """
        Args:
            part_feats: [B, K, D] per-part features

        Returns:
            loss: scalar (lower = more diverse parts)
        """
        B, K, D = part_feats.shape

        # L2 normalize
        feats_norm = F.normalize(part_feats, p=2, dim=2)  # [B, K, D]

        # Cosine similarity matrix between parts for each sample
        sim = torch.bmm(feats_norm, feats_norm.transpose(1, 2))  # [B, K, K]

        # Zero out diagonal (self-similarity = 1)
        eye = torch.eye(K, device=sim.device).unsqueeze(0)
        sim = sim * (1 - eye)

        # Mean of off-diagonal similarities
        n_off_diag = K * (K - 1)
        loss = sim.sum() / (B * n_off_diag)

        return loss.clamp(min=0.0)
