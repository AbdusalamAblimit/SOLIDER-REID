"""
Pose-Aware Matching Network (PAMN)

Learns to match two sets of keypoint features by considering
pose structure, visibility, and per-keypoint discriminability.

Trained with contrastive pairs from same/different IDs.
Used at test time for re-scoring top-K retrieval results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseMatchingNetwork(nn.Module):
    """Learned keypoint-pair matching for pose-aware ReID.

    Takes two sets of keypoint features + visibility scores,
    produces a scalar matching score.

    Args:
        num_keypoints: Number of keypoints (17)
        feat_dim: Per-keypoint feature dimension (768)
        hidden_dim: MLP hidden dimension
    """

    def __init__(self, num_keypoints=17, feat_dim=768, hidden_dim=128):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Per-keypoint feature projection (reduce dim for efficiency)
        self.proj = nn.Linear(feat_dim, 64)

        # Input: [per_kp_sim(17), per_kp_vis_mask(17), per_kp_proj_diff(17*64)]
        # But 17*64=1088 is too large. Use projected cosine sim instead.
        # Input features per pair:
        #   - per_kp_cosine_sim: (17,)
        #   - per_kp_vis_mask: (17,)
        #   - per_kp_L2_dist: (17,)
        #   - global_sim: (1,)  — overall similarity
        input_dim = num_keypoints * 3 + 1

        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_pair_features(self, kp_q, kp_g, vis_q, vis_g):
        """Compute matching features for a query-gallery pair.

        Args:
            kp_q: (17, D) query keypoint features
            kp_g: (17, D) gallery keypoint features
            vis_q: (17,) query visibility scores
            vis_g: (17,) gallery visibility scores

        Returns:
            pair_feat: (52,) matching features
        """
        # Per-keypoint cosine similarity
        kp_q_norm = F.normalize(kp_q, dim=-1)
        kp_g_norm = F.normalize(kp_g, dim=-1)
        cos_sim = (kp_q_norm * kp_g_norm).sum(dim=-1)  # (17,)

        # Per-keypoint L2 distance (projected)
        pq = self.proj(kp_q)  # (17, 64)
        pg = self.proj(kp_g)  # (17, 64)
        l2_dist = (pq - pg).pow(2).sum(dim=-1)  # (17,)
        l2_dist = l2_dist / 64.0  # normalize

        # Common visibility mask
        vis_mask = vis_q * vis_g  # (17,)

        # Global similarity (average of masked cos_sim)
        masked_sum = (cos_sim * vis_mask).sum()
        mask_count = vis_mask.sum().clamp(min=1.0)
        global_sim = (masked_sum / mask_count).unsqueeze(0)  # (1,)

        # Concatenate all features
        pair_feat = torch.cat([cos_sim, vis_mask, l2_dist, global_sim], dim=0)  # (52,)

        return pair_feat

    def forward(self, kp_q, kp_g, vis_q, vis_g):
        """Compute matching score for a batch of pairs.

        Args:
            kp_q: (B, 17, D) or (17, D)
            kp_g: (B, 17, D) or (17, D)
            vis_q: (B, 17) or (17,)
            vis_g: (B, 17) or (17,)

        Returns:
            score: (B,) or scalar matching score
        """
        if kp_q.dim() == 2:
            # Single pair
            pair_feat = self.compute_pair_features(kp_q, kp_g, vis_q, vis_g)
            return self.scorer(pair_feat).squeeze(-1)
        else:
            # Batch of pairs
            B = kp_q.shape[0]
            scores = []
            for i in range(B):
                pf = self.compute_pair_features(
                    kp_q[i], kp_g[i], vis_q[i], vis_g[i])
                scores.append(self.scorer(pf))
            return torch.cat(scores, dim=0).squeeze(-1)  # (B,)

    def compute_training_loss(self, kp_feats, kp_weights, labels, margin=0.5):
        """Compute contrastive loss on batch pairs.

        For each anchor, find hardest positive and hardest negative
        based on PAMN scores.

        Args:
            kp_feats: (B, 17, D) keypoint features from GCN
            kp_weights: (B, 17) visibility/confidence weights
            labels: (B,) identity labels
            margin: contrastive margin

        Returns:
            loss: scalar contrastive loss
        """
        B = kp_feats.shape[0]
        device = kp_feats.device

        # Compute all pairwise PAMN scores
        # For efficiency, precompute pair features
        all_pair_feats = []
        for i in range(B):
            row = []
            for j in range(B):
                pf = self.compute_pair_features(
                    kp_feats[i], kp_feats[j],
                    kp_weights[i], kp_weights[j])
                row.append(pf)
            all_pair_feats.append(torch.stack(row))

        pair_feats_matrix = torch.stack(all_pair_feats)  # (B, B, 52)
        # Score all pairs
        scores_flat = self.scorer(pair_feats_matrix.view(-1, pair_feats_matrix.shape[-1]))
        scores = scores_flat.view(B, B)  # (B, B) matching scores

        # Create masks
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = label_eq & ~eye  # same ID, not self
        neg_mask = ~label_eq       # different ID

        # Contrastive loss: pos pairs should have high score, neg should have low
        loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for i in range(B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            # Hardest positive (lowest score among positives)
            pos_scores = scores[i, pos_idx]
            hard_pos_score = pos_scores.min()

            # Hardest negative (highest score among negatives)
            neg_scores = scores[i, neg_idx]
            hard_neg_score = neg_scores.max()

            # Triplet-style loss: want pos_score > neg_score + margin
            pair_loss = F.relu(margin - hard_pos_score + hard_neg_score)
            loss = loss + pair_loss
            n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss
