"""Part Prototype Bank — Per-Identity Per-Part Feature Memory.

Maintains an EMA prototype for each (identity, body_part) pair.
During training, visible keypoint features update the bank.
The bank provides identity-specific part prototypes for:
  1. Part-Prototype Consistency Loss (training)
  2. Prototype-Assisted Feature Completion (test-time, future)

This is fundamentally different from class-level prototypes (MSPL)
or batch-level recovery (TTSFR/SKC): PACI maintains per-identity
per-part memories across the ENTIRE training set.
"""

import torch
import torch.nn as nn


class PartPrototypeBank(nn.Module):
    """Per-identity per-part momentum prototype bank.

    Args:
        num_classes: Number of identity classes in training set
        num_parts: Number of body parts (17 for COCO keypoints)
        feat_dim: Feature dimension per part
        momentum: EMA momentum (0.9 = 90% old + 10% new)
        vis_threshold: Minimum keypoint score to consider visible
    """

    def __init__(self, num_classes, num_parts=17, feat_dim=768,
                 momentum=0.9, vis_threshold=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.vis_threshold = vis_threshold

        # Prototype bank: (num_classes, num_parts, feat_dim)
        # Initialized to zero; will be populated during first epoch
        self.register_buffer('bank',
                             torch.zeros(num_classes, num_parts, feat_dim))
        # Count how many times each (class, part) has been updated
        self.register_buffer('update_count',
                             torch.zeros(num_classes, num_parts, dtype=torch.long))

    @torch.no_grad()
    def update(self, kp_feats, kp_scores, labels):
        """Update prototypes with current batch's keypoint features.

        Args:
            kp_feats: (B, 17, C) per-keypoint features (detached)
            kp_scores: (B, 17) keypoint confidence scores
            labels: (B,) identity labels
        """
        B = kp_feats.shape[0]
        visible = kp_scores > self.vis_threshold  # (B, 17)

        for i in range(B):
            label = labels[i].item()
            if label >= self.num_classes:
                continue

            for k in range(self.num_parts):
                if not visible[i, k]:
                    continue

                feat = kp_feats[i, k]  # (C,)
                count = self.update_count[label, k].item()

                if count == 0:
                    # First update: initialize directly
                    self.bank[label, k] = feat
                else:
                    # EMA update
                    self.bank[label, k] = (self.momentum * self.bank[label, k] +
                                           (1 - self.momentum) * feat)

                self.update_count[label, k] += 1

    def get_prototypes(self, labels):
        """Get prototypes for given identity labels.

        Args:
            labels: (B,) identity labels

        Returns:
            prototypes: (B, 17, C) prototypes for each sample
            valid: (B, 17) bool — which parts have been initialized
        """
        B = labels.shape[0]
        prototypes = self.bank[labels]  # (B, 17, C)
        valid = self.update_count[labels] > 0  # (B, 17)
        return prototypes, valid

    def get_negative_prototypes(self, labels, num_neg=4):
        """Get prototypes from random different identities (for contrastive).

        Args:
            labels: (B,) identity labels
            num_neg: number of negative identities per sample

        Returns:
            neg_protos: (B, num_neg, 17, C)
        """
        B = labels.shape[0]
        device = labels.device

        # Find classes with initialized prototypes
        has_data = (self.update_count.sum(dim=1) > 0)  # (num_classes,)
        valid_classes = has_data.nonzero(as_tuple=True)[0]

        if len(valid_classes) < 2:
            return None

        neg_protos = torch.zeros(B, num_neg, self.num_parts, self.feat_dim,
                                 device=device)
        for i in range(B):
            # Sample negative classes (different from current label)
            candidates = valid_classes[valid_classes != labels[i]]
            if len(candidates) == 0:
                continue
            idx = torch.randint(0, len(candidates), (num_neg,), device=device)
            neg_labels = candidates[idx]
            neg_protos[i] = self.bank[neg_labels]  # (num_neg, 17, C)

        return neg_protos

    def stats(self):
        """Return bank statistics for logging."""
        total_slots = self.num_classes * self.num_parts
        initialized = (self.update_count > 0).sum().item()
        coverage = initialized / max(total_slots, 1)
        avg_count = self.update_count[self.update_count > 0].float().mean().item() \
            if initialized > 0 else 0
        return {
            'coverage': coverage,
            'avg_count': avg_count,
            'initialized': initialized,
            'total': total_slots,
        }
