"""
Pose Translation Module (PTM)

Translates a person's feature representation from one pose to another.
Enables comparing query and gallery as if they were in the same pose.

Core idea: PTM(feat_A, pose_A, pose_B) ≈ feat_B when A and B are same person.

Training: supervised by same-ID pairs with different poses in each batch.
Testing: translate query features to gallery poses for pairwise comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseTranslationModule(nn.Module):
    """Translates features from source pose to target pose.

    Uses actual keypoint COORDINATES (not just confidence) as pose descriptor,
    so it learns genuine pose translation rather than just occlusion pattern mapping.

    Args:
        feat_dim: Feature dimension (768)
        pose_dim: Pose condition dimension
        hidden_dim: Translation hidden dimension
    """

    def __init__(self, feat_dim=768, pose_dim=128, hidden_dim=512):
        super().__init__()
        self.feat_dim = feat_dim

        # Pose descriptor: keypoint coordinates (17×2) + confidence (17) = 51 per person
        # Pose pair: src(51) + tgt(51) = 102
        pose_input_dim = 102

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, pose_dim),
            nn.ReLU(inplace=True),
        )

        self.translator = nn.Sequential(
            nn.Linear(feat_dim + pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

        # Zero-init: starts as identity
        nn.init.zeros_(self.translator[-1].weight)
        nn.init.zeros_(self.translator[-1].bias)

    @staticmethod
    def make_pose_descriptor(keypoints, scores):
        """Create pose descriptor from keypoint coordinates and scores.

        Args:
            keypoints: (B, 17, 2) normalized keypoint coordinates
            scores: (B, 17) confidence scores

        Returns:
            pose_desc: (B, 51) = flattened coordinates + scores
        """
        B = keypoints.shape[0]
        # Normalize coordinates to [0, 1] range (already in pixel coords, divide by image size)
        # For simplicity, just flatten and concatenate with scores
        kp_flat = keypoints.reshape(B, -1)  # (B, 34)
        return torch.cat([kp_flat, scores], dim=-1)  # (B, 51)

    def forward(self, src_feat, src_pose_desc, tgt_pose_desc):
        """Translate feature from source pose to target pose.

        Args:
            src_feat: (B, D) source feature
            src_pose_desc: (B, 51) source pose descriptor
            tgt_pose_desc: (B, 51) target pose descriptor

        Returns:
            translated_feat: (B, D)
        """
        pose_pair = torch.cat([src_pose_desc, tgt_pose_desc], dim=-1)  # (B, 102)
        pose_cond = self.pose_encoder(pose_pair)  # (B, pose_dim)
        translator_input = torch.cat([src_feat, pose_cond], dim=-1)
        delta = self.translator(translator_input)
        return src_feat + delta

    def compute_training_loss(self, global_feats, keypoints, scores, labels):
        """Compute pose translation loss using same-ID pairs.

        Args:
            global_feats: (B, D) global features
            keypoints: (B, 17, 2) keypoint coordinates (person 0)
            scores: (B, 17) confidence scores (person 0)
            labels: (B,) identity labels

        Returns:
            loss: scalar MSE translation loss
        """
        B = global_feats.shape[0]
        device = global_feats.device

        feats = global_feats.detach()
        pose_descs = self.make_pose_descriptor(keypoints.detach(), scores.detach())

        losses = []
        for i in range(B):
            same_id = (labels == labels[i]).nonzero(as_tuple=True)[0]
            same_id = same_id[same_id != i]
            if len(same_id) == 0:
                continue

            j = same_id[torch.randint(len(same_id), (1,)).item()]

            # Translate i→j and j→i
            trans_ij = self.forward(feats[i:i+1], pose_descs[i:i+1], pose_descs[j:j+1])
            trans_ji = self.forward(feats[j:j+1], pose_descs[j:j+1], pose_descs[i:i+1])

            losses.append(F.mse_loss(trans_ij, feats[j:j+1]))
            losses.append(F.mse_loss(trans_ji, feats[i:i+1]))

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=device)
