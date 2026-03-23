"""Pose-Normalized Identity Space (PNIS).

Factors out pose-induced variation from features by learning a
pose-to-offset mapping and subtracting it from the raw feature.

identity_feat = raw_feat - alpha * PoseEncoder(skeleton)
"""

import torch
import torch.nn as nn


class PoseNormalizer(nn.Module):
    """Learns to predict and subtract pose-dependent feature offset."""

    def __init__(self, feat_dim=768, pose_dim=51, hidden_dim=256):
        super().__init__()
        # Pose → offset mapping
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        # Learnable scale: starts near 0 (gentle ramp-up)
        # sigmoid(-3) = 0.047, so initial subtraction is ~5% of offset
        self.alpha = nn.Parameter(torch.tensor([-3.0]))
        # Small random init on last layer (not zero — zero-init causes zero gradient)
        # Alpha starts at 0.047 which keeps subtraction gentle
        nn.init.normal_(self.pose_encoder[-1].weight, std=0.01)
        nn.init.zeros_(self.pose_encoder[-1].bias)

    def forward(self, raw_feat, keypoints, scores):
        """
        Args:
            raw_feat: (B, C) raw pooled feature
            keypoints: (B, 17, 2) keypoint coordinates (normalized to [0,1])
            scores: (B, 17) keypoint confidence scores
        Returns:
            identity_feat: (B, C) pose-normalized feature
            stats: dict with diagnostic info
        """
        B = raw_feat.shape[0]
        # Compose pose descriptor: (x1,y1,s1, x2,y2,s2, ...) = 51-d
        pose_desc = torch.cat([
            keypoints.view(B, -1),  # (B, 34)
            scores,                  # (B, 17)
        ], dim=1)  # (B, 51)

        # Predict pose offset
        pose_offset = self.pose_encoder(pose_desc)  # (B, C)

        # Subtract with learnable scale
        alpha = torch.sigmoid(self.alpha)  # (0, 1)
        identity_feat = raw_feat - alpha * pose_offset

        with torch.no_grad():
            offset_norm = pose_offset.norm(dim=1).mean().item()
            feat_norm = raw_feat.norm(dim=1).mean().item()
            alpha_val = alpha.item()

        stats = {
            'offset_norm': offset_norm,
            'feat_norm': feat_norm,
            'alpha': alpha_val,
            'ratio': offset_norm / max(feat_norm, 1e-8),
        }
        return identity_feat, stats
