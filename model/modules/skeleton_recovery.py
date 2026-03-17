"""
Learned Skeleton Recovery Module (LSRM)

Cross-attention based recovery of occluded keypoint features
using visible keypoint features from other images.

Training: uses same-ID pairs in batch for supervision
Testing: integrated with SGCFR for gallery-based recovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonRecoveryModule(nn.Module):
    """Learns to recover occluded keypoint features from visible ones.

    Uses cross-attention: occluded keypoints attend to all visible keypoints
    (both from self and from candidate images) to produce recovered features.

    Args:
        feat_dim: Per-keypoint feature dimension (768)
        num_keypoints: Number of keypoints (17)
        hidden_dim: Attention hidden dimension
        num_heads: Number of attention heads
    """

    def __init__(self, feat_dim=768, num_keypoints=17, hidden_dim=256, num_heads=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim

        # Cross-attention: queries (occluded) attend to keys/values (visible)
        # MHA handles Q/K/V projection internally
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
            kdim=feat_dim, vdim=feat_dim)
        # Only project queries from feat_dim to hidden_dim
        self.q_proj = nn.Linear(feat_dim, hidden_dim)

        # Project back to feature space
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # Residual gate (zero-init for safe start)
        self.gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.out_proj[0].weight)
        nn.init.zeros_(self.out_proj[0].bias)

    def recover(self, query_kp, cand_kp, query_vis, cand_vis, threshold=0.3):
        """Recover query's occluded keypoints using candidate's visible ones.

        Args:
            query_kp: (17, D) query keypoint features
            cand_kp: (K, 17, D) candidate keypoint features (K candidates)
            query_vis: (17,) query visibility
            cand_vis: (K, 17) candidate visibility

        Returns:
            recovered: (17, D) query features with occluded kp recovered
        """
        D = query_kp.shape[-1]
        recovered = query_kp.clone()

        # Find occluded keypoints in query
        occ_mask = query_vis < threshold  # (17,) bool
        if not occ_mask.any():
            return recovered  # Nothing to recover

        # Collect all visible keypoint features from candidates
        # Flatten: (K, 17, D) → visible subset → (N_vis, D)
        vis_mask = cand_vis >= threshold  # (K, 17) bool
        if not vis_mask.any():
            return recovered  # No visible candidates

        vis_feats = cand_kp[vis_mask]  # (N_vis, D)

        # Query: occluded keypoints
        occ_feats = query_kp[occ_mask]  # (N_occ, D)
        N_occ = occ_feats.shape[0]

        # Cross-attention: occluded attend to visible
        # Q projected by q_proj; K,V projected by MHA's internal weights (kdim=feat_dim)
        Q = self.q_proj(occ_feats).unsqueeze(0)  # (1, N_occ, hidden)
        K_raw = vis_feats.unsqueeze(0)             # (1, N_vis, feat_dim)
        V_raw = vis_feats.unsqueeze(0)             # (1, N_vis, feat_dim)

        attn_out, _ = self.attn(Q, K_raw, V_raw)  # (1, N_occ, hidden)
        attn_out = attn_out.squeeze(0)     # (N_occ, hidden)

        # Project back and apply residual gate
        recovered_feats = self.out_proj(attn_out)  # (N_occ, D)
        # Gated residual: start from original, gradually add recovery
        final_feats = occ_feats + self.gate * recovered_feats

        recovered[occ_mask] = final_feats
        return recovered

    def compute_training_loss(self, kp_feats, kp_weights, labels, threshold=0.3):
        """Compute recovery loss using same-ID pairs in batch.

        For each pair (A, B) of same ID:
        - Use A's features to recover B's occluded keypoints
        - MSE loss between recovered and original B features

        Args:
            kp_feats: (B, 17, D) detached keypoint features
            kp_weights: (B, 17) visibility
            labels: (B,) identity labels

        Returns:
            loss: scalar recovery loss
        """
        B = kp_feats.shape[0]
        device = kp_feats.device
        total_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for i in range(B):
            # Find same-ID partners
            same_id = (labels == labels[i]).nonzero(as_tuple=True)[0]
            same_id = same_id[same_id != i]
            if len(same_id) == 0:
                continue

            # Use partners as candidates to recover i's occluded kp
            cand_kp = kp_feats[same_id]      # (K, 17, D)
            cand_vis = kp_weights[same_id]    # (K, 17)

            # Find which of i's keypoints are occluded
            occ_mask = kp_weights[i] < threshold
            if not occ_mask.any():
                continue

            # Recover
            recovered = self.recover(
                kp_feats[i], cand_kp, kp_weights[i], cand_vis, threshold)

            # Target: the original features (from the detached kp_feats)
            # For occluded kp, we want recovered ≈ what they SHOULD be
            # Best proxy: average of same-ID partners' features at those kp
            target_feats = kp_feats[i].clone()
            for kp in range(17):
                if not occ_mask[kp]:
                    continue
                # Target = average of partners' visible features at this kp
                vis_cand = cand_vis[:, kp] >= threshold
                if vis_cand.any():
                    target_feats[kp] = cand_kp[vis_cand, kp].mean(dim=0)

            # MSE only on recovered (occluded) keypoints
            loss = F.mse_loss(recovered[occ_mask], target_feats[occ_mask].detach())
            total_loss = total_loss + loss
            n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss
