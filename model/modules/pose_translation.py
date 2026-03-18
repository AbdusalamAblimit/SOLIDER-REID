"""
Pose Translation Module (PTM)

Translates a person's feature representation from one pose to another.
Enables comparing query and gallery as if they were in the same pose.

Core idea: PTM(feat_A, pose_A, pose_B) ≈ feat_B when A and B are same person.

Training: supervised by same-ID pairs with different poses in each batch.
Testing: translate query to gallery's pose before distance computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseTranslationModule(nn.Module):
    """Translates features from source pose to target pose.

    Given a person's feature and their pose, produces the feature
    that person would have if they appeared in a different pose.

    Args:
        feat_dim: Feature dimension (768)
        pose_dim: Pose encoding dimension
        hidden_dim: Translation hidden dimension
    """

    def __init__(self, feat_dim=768, pose_dim=128, hidden_dim=512):
        super().__init__()

        # Encode pose pair (src_pose, tgt_pose) into condition vector
        # Input: per-keypoint heatmap statistics (17 values each → 34 total)
        self.pose_encoder = nn.Sequential(
            nn.Linear(34, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, pose_dim),
            nn.ReLU(inplace=True),
        )

        # Feature translator: residual MLP conditioned on pose pair
        self.translator = nn.Sequential(
            nn.Linear(feat_dim + pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

        # Zero-init output: starts as identity (translated = original)
        nn.init.zeros_(self.translator[-1].weight)
        nn.init.zeros_(self.translator[-1].bias)

    def encode_pose(self, kp_weights):
        """Encode pose from keypoint confidence scores.

        Args:
            kp_weights: (B, 17) keypoint confidence scores

        Returns:
            pose_vec: (B, 17) — just the raw scores as pose descriptor
        """
        return kp_weights

    def forward(self, src_feat, src_pose, tgt_pose):
        """Translate feature from source pose to target pose.

        Args:
            src_feat: (B, D) source feature (e.g., query's global feature)
            src_pose: (B, 17) source pose descriptor
            tgt_pose: (B, 17) target pose descriptor

        Returns:
            translated_feat: (B, D) feature as if person appeared in tgt_pose
        """
        # Encode pose pair
        pose_pair = torch.cat([src_pose, tgt_pose], dim=-1)  # (B, 34)
        pose_cond = self.pose_encoder(pose_pair)  # (B, pose_dim)

        # Translate: residual adaptation conditioned on pose pair
        translator_input = torch.cat([src_feat, pose_cond], dim=-1)  # (B, D+pose_dim)
        delta = self.translator(translator_input)  # (B, D)

        return src_feat + delta

    def compute_training_loss(self, global_feats, kp_weights, labels):
        """Compute pose translation loss using same-ID pairs.

        For each same-ID pair (A, B):
          PTM(feat_A, pose_A, pose_B) should ≈ feat_B
          PTM(feat_B, pose_B, pose_A) should ≈ feat_A

        Args:
            global_feats: (B, D) global features (detached from backbone)
            kp_weights: (B, 17) keypoint weights as pose descriptor
            labels: (B,) identity labels

        Returns:
            loss: scalar MSE translation loss
        """
        B = global_feats.shape[0]
        device = global_feats.device
        total_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        feats = global_feats.detach()  # Don't backprop into backbone
        poses = kp_weights.detach()

        for i in range(B):
            same_id = (labels == labels[i]).nonzero(as_tuple=True)[0]
            same_id = same_id[same_id != i]
            if len(same_id) == 0:
                continue

            # Pick one random partner
            j = same_id[torch.randint(len(same_id), (1,)).item()]

            # Forward: translate i to j's pose
            translated_i = self.forward(
                feats[i:i+1], poses[i:i+1], poses[j:j+1])
            loss_ij = F.mse_loss(translated_i, feats[j:j+1])

            # Backward: translate j to i's pose
            translated_j = self.forward(
                feats[j:j+1], poses[j:j+1], poses[i:i+1])
            loss_ji = F.mse_loss(translated_j, feats[i:i+1])

            total_loss = total_loss + loss_ij + loss_ji
            n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / (2 * n_pairs)

        return total_loss
