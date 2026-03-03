"""Part-aware losses for PAMS: Part-Averaged Triplet and Push Diversity."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _euclidean_dist(x, y):
    """Pairwise euclidean distance. x: [m, d], y: [n, d] -> [m, n]."""
    # Force float32: under AMP float16, the xx + yy - 2*xy subtraction
    # loses precision, potentially producing negative values before sqrt.
    x = x.float()
    y = y.float()
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


class PartAveragedTripletLoss(nn.Module):
    """Per-part L2 distances masked by mutual visibility, averaged, then batch-hard mined.

    Args:
        margin: Triplet margin. None = soft margin (SoftMarginLoss).
        normalize: L2-normalize part features before distance computation.
                   Crucial for stability: raw features have norm ~200+, producing
                   distances on scale ~300 that cause explosive loss spikes during
                   batch-hard mining. After normalization, distances are in [0, 2].
    """

    def __init__(self, margin=None, normalize=True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, part_feats, labels, part_vis):
        """
        Args:
            part_feats: [B, K, D] part feature vectors
            labels:     [B] identity labels
            part_vis:   [B, K] part visibility scores (0..1)
        Returns:
            loss: scalar
        """
        B, K, D = part_feats.shape

        # L2 normalize: bounds distances to [0, 2], eliminates scale sensitivity
        if self.normalize:
            part_feats = F.normalize(part_feats, p=2, dim=2)

        # Per-part pairwise distance: [K, B, B]
        per_part_dist = []
        for k in range(K):
            per_part_dist.append(_euclidean_dist(part_feats[:, k], part_feats[:, k]))
        per_part_dist = torch.stack(per_part_dist, dim=0)  # [K, B, B]

        # Mutual visibility mask: [K, B, B]
        # vis_k[i] * vis_k[j] > 0 means both i and j have part k visible
        # part_vis is mask_sum / N (fraction of spatial locations), so threshold
        # should be consistent with extract_part_features vis_threshold
        vis_binary = (part_vis > 0.1).float()  # [B, K]
        vis_mutual = []
        for k in range(K):
            vk = vis_binary[:, k]  # [B]
            vis_mutual.append(vk.unsqueeze(1) * vk.unsqueeze(0))  # [B, B]
        vis_mutual = torch.stack(vis_mutual, dim=0)  # [K, B, B]

        # Masked average distance over valid parts: [B, B]
        valid_count = vis_mutual.sum(dim=0).clamp(min=1e-6)  # [B, B]
        dist_mat = (per_part_dist * vis_mutual).sum(dim=0) / valid_count  # [B, B]

        # Batch-hard mining
        N = B
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        dist_ap = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1)[0]
        dist_an = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1)[0]

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss


class PushLoss(nn.Module):
    """Cosine diversity loss between part prototypes to prevent collapse.

    Minimizes pairwise cosine similarity between different part feature centroids.
    """

    def forward(self, part_feats):
        """
        Args:
            part_feats: [B, K, D] part feature vectors
        Returns:
            loss: scalar
        """
        # Part centroids: [K, D] — force float32 for normalize precision
        centroids = part_feats.float().mean(dim=0)  # [K, D]

        # Skip parts with zero centroids (invisible parts were zeroed out)
        norms = centroids.norm(p=2, dim=1)  # [K]
        valid = norms > 1e-6
        if valid.sum() < 2:
            return centroids.sum() * 0.0  # no valid pairs, return zero with grad

        centroids = F.normalize(centroids[valid], p=2, dim=1)

        # Pairwise cosine similarity
        sim = torch.mm(centroids, centroids.t())  # [K', K']

        K = centroids.shape[0]
        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
        # Mean of off-diagonal similarities (want to minimize)
        loss = sim[mask].mean()

        return loss
