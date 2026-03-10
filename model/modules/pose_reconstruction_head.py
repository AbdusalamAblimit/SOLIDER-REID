"""Pose Reconstruction Head (PRA) — auxiliary task for structural regularization.

Forces backbone features to encode body structure information by predicting
pose heatmaps from features. Complementary to PSG (which injects pose INTO
features): PRA ensures features RETAIN structural information.

Only used during training. Can be removed at test time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseReconstructionHead(nn.Module):
    """Lightweight head that predicts pose heatmaps from backbone features.

    Args:
        feat_channels: Input feature channels (768 for Swin-Tiny Stage 3)
        pose_channels: Number of heatmap channels to predict (17 for COCO)
        hidden_channels: Hidden layer channels
        loss_weight: Weight for the MSE reconstruction loss
    """

    def __init__(self, feat_channels=768, pose_channels=17,
                 hidden_channels=128, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

        self.head = nn.Sequential(
            nn.Conv2d(feat_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, pose_channels, kernel_size=1, bias=True),
        )

    def forward(self, feature_map, gt_heatmaps):
        """Predict heatmaps and compute reconstruction loss.

        Args:
            feature_map: (B, C, H, W) backbone output feature map
            gt_heatmaps: (B, 17, hH, hW) ground truth scene-level heatmaps

        Returns:
            recon_loss: scalar MSE loss weighted by loss_weight
        """
        B, C, H, W = feature_map.shape

        # Predict heatmaps from features
        pred = self.head(feature_map)  # (B, 17, H, W)

        # Resize GT heatmaps to match feature map spatial size
        if gt_heatmaps.shape[2:] != (H, W):
            gt = F.interpolate(gt_heatmaps, size=(H, W),
                               mode='bilinear', align_corners=False)
        else:
            gt = gt_heatmaps

        # Apply sigmoid to GT (raw logits → probabilities)
        gt = torch.sigmoid(gt)

        # MSE loss
        loss = F.mse_loss(pred, gt)

        return loss * self.loss_weight
