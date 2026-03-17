"""
Momentum Memory for Contrastive Learning in ReID

Maintains a per-class feature memory bank updated with momentum.
Provides contrastive loss that leverages the full class vocabulary.

Based on: HybridMemory (DPEFormer), MoCo (He et al.), XBM (Wang et al.)

Usage:
  memory = MomentumMemory(feat_dim=768, num_classes=702, momentum=0.1, temp=0.05)
  loss = memory(features, labels)  # called every training step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentumMemory(nn.Module):
    """Per-class momentum-updated feature memory bank.

    Args:
        feat_dim: Feature dimension (768)
        num_classes: Number of identity classes
        momentum: EMA momentum for memory update (0 = no update, 1 = replace)
        temp: Temperature for contrastive loss
    """

    def __init__(self, feat_dim=768, num_classes=702, momentum=0.1, temp=0.05):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.temp = temp

        # Memory bank: (num_classes, feat_dim) — NOT a learnable parameter
        self.register_buffer('memory', torch.randn(num_classes, feat_dim))
        # Normalize initial memory
        self.memory = F.normalize(self.memory, dim=1)

        # Track which classes have been seen (for initialization)
        self.register_buffer('initialized', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features, labels):
        """Update memory bank with momentum.

        Args:
            features: (B, D) L2-normalized feature vectors
            labels: (B,) integer class labels
        """
        features = F.normalize(features.detach(), dim=1)

        for cls in labels.unique():
            mask = labels == cls
            cls_feat = features[mask].mean(dim=0)  # average features for this class
            cls_feat = F.normalize(cls_feat, dim=0)

            idx = cls.item()
            if not self.initialized[idx]:
                # First time seeing this class: initialize directly
                self.memory[idx] = cls_feat
                self.initialized[idx] = True
            else:
                # Momentum update
                self.memory[idx] = (1 - self.momentum) * self.memory[idx] + \
                                   self.momentum * cls_feat
                self.memory[idx] = F.normalize(self.memory[idx], dim=0)

    def forward(self, features, labels):
        """Compute contrastive loss against memory bank.

        Args:
            features: (B, D) feature vectors (will be normalized)
            labels: (B,) integer class labels

        Returns:
            loss: scalar contrastive loss
        """
        features = F.normalize(features, dim=1)

        # Update memory with current batch (no grad)
        self.update(features, labels)

        # Compute similarity to all classes in memory: (B, num_classes)
        sim = torch.mm(features, self.memory.t()) / self.temp

        # Cross-entropy with class labels
        loss = F.cross_entropy(sim, labels)

        return loss
