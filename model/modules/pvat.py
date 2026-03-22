"""Pose-Visibility Adversarial Training (PVAT).

Forces the backbone feature to NOT encode visibility information
by adversarial gradient reversal.
"""

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Reverses gradients during backward pass."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class VisibilityPredictor(nn.Module):
    """Predicts keypoint visibility from the global feature.

    During training, gradient reversal makes the backbone produce features
    that the predictor CANNOT use to predict visibility.
    """

    def __init__(self, feat_dim=768, num_keypoints=17):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_keypoints)
        # Initialize near zero so initial predictions are ~0.5 (neutral)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feat, alpha=1.0):
        """
        Args:
            feat: (B, D) global feature (before BN, after GAP)
            alpha: gradient reversal strength (0=no reversal, 1=full reversal)
        Returns:
            vis_pred: (B, 17) predicted visibility logits
        """
        reversed_feat = GradientReversal.apply(feat, alpha)
        return self.fc(reversed_feat)
