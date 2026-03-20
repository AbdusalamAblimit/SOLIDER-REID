"""Support-complete keypoint prototype bank for training-time distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupportCompleteBank(nn.Module):
    """EMA memory bank of per-identity, per-keypoint prototypes.

    The bank is updated only from high-visibility keypoints and used as a
    teacher for low-visibility keypoints of the same identity.
    """

    def __init__(self, num_classes, feat_dim=768, num_keypoints=17,
                 low_thr=0.3, update_thr=0.5, momentum=0.9, min_count=1):
        super().__init__()
        self.low_thr = low_thr
        self.update_thr = update_thr
        self.momentum = momentum
        self.min_count = min_count

        self.register_buffer('prototype_bank', torch.zeros(num_classes, num_keypoints, feat_dim))
        self.register_buffer('confidence_bank', torch.zeros(num_classes, num_keypoints))
        self.register_buffer('count_bank', torch.zeros(num_classes, num_keypoints, dtype=torch.long))

    def compute_loss(self, kp_feats, kp_weights, labels):
        """Cosine distillation from the support-complete prototype teacher."""
        kp_feats = F.normalize(kp_feats, dim=2)
        proto = self.prototype_bank[labels].detach()
        proto_conf = self.confidence_bank[labels].detach()
        proto_count = self.count_bank[labels]

        low_mask = kp_weights <= self.low_thr
        support_mask = (proto_count >= self.min_count) & (proto_conf > 0)
        mask = low_mask & support_mask
        low_ratio = float(low_mask.float().mean().item())
        active_ratio = float(mask.float().mean().item())
        low_total = int(low_mask.sum().item())
        elig_ratio = float(mask.sum().item() / max(low_total, 1))

        if not mask.any():
            stats = {
                'low_ratio': low_ratio,
                'active_ratio': active_ratio,
                'elig_ratio': elig_ratio,
                'proto_conf': 0.0,
                'proto_count': 0.0,
                'cosine': 0.0,
            }
            return kp_feats.new_zeros(()), 0, stats

        cosine = (kp_feats * proto).sum(dim=2).clamp(min=-1.0, max=1.0)
        point_loss = 1.0 - cosine
        weights = proto_conf * mask.float()
        loss = (point_loss * weights).sum() / weights.sum().clamp(min=1e-12)
        stats = {
            'low_ratio': low_ratio,
            'active_ratio': active_ratio,
            'elig_ratio': elig_ratio,
            'proto_conf': float(proto_conf[mask].mean().item()),
            'proto_count': float(proto_count[mask].float().mean().item()),
            'cosine': float(cosine[mask].mean().item()),
        }
        return loss, int(mask.sum().item()), stats

    @torch.no_grad()
    def update(self, kp_feats, kp_weights, labels):
        """EMA update using only high-visibility keypoints."""
        kp_feats = F.normalize(kp_feats.detach(), dim=2)
        kp_weights = kp_weights.detach()
        labels = labels.detach().long()

        updated = 0
        for b_idx in range(kp_feats.shape[0]):
            cls = int(labels[b_idx].item())
            vis_mask = kp_weights[b_idx] >= self.update_thr
            if not vis_mask.any():
                continue
            kp_ids = vis_mask.nonzero(as_tuple=True)[0]
            new_feat = kp_feats[b_idx, kp_ids]
            new_conf = kp_weights[b_idx, kp_ids]

            old_count = self.count_bank[cls, kp_ids]
            first_mask = old_count == 0
            if first_mask.any():
                first_ids = kp_ids[first_mask]
                self.prototype_bank[cls, first_ids] = new_feat[first_mask]
                self.confidence_bank[cls, first_ids] = new_conf[first_mask]
                self.count_bank[cls, first_ids] += 1
                updated += int(first_ids.numel())

            if (~first_mask).any():
                ema_ids = kp_ids[~first_mask]
                old_proto = self.prototype_bank[cls, ema_ids]
                old_conf = self.confidence_bank[cls, ema_ids]
                mixed = self.momentum * old_proto + (1.0 - self.momentum) * new_feat[~first_mask]
                self.prototype_bank[cls, ema_ids] = F.normalize(mixed, dim=1)
                self.confidence_bank[cls, ema_ids] = self.momentum * old_conf + (1.0 - self.momentum) * new_conf[~first_mask]
                self.count_bank[cls, ema_ids] += 1
                updated += int(ema_ids.numel())

        return updated

    def get_support(self, kp_feats, kp_weights, labels):
        """Query support-complete prototype features for low-visibility keypoints.

        Returns scaled prototypes plus a valid mask. Scaling follows SCFR:
        prototype norms are matched to the sample's visible-keypoint norm so
        the injected support lives in a similar feature range.
        """
        with torch.no_grad():
            labels = labels.long()
            low_mask = kp_weights <= self.low_thr
            proto_count = self.count_bank[labels]
            proto_conf = self.confidence_bank[labels]
            support_mask = (proto_count >= self.min_count) & (proto_conf > 0)
            query_mask = low_mask & support_mask

            proto = self.prototype_bank[labels]  # (B, 17, C)
            orig_norm = kp_feats.norm(dim=2, keepdim=True).clamp(min=1e-6)
            vis_mask = ~low_mask
            vis_count = vis_mask.float().sum(dim=1).clamp(min=1)
            vis_norm = (orig_norm.squeeze(-1) * vis_mask.float()).sum(dim=1) / vis_count
            valid_samples = vis_norm > 0
            final_mask = query_mask & valid_samples.unsqueeze(1)
            scaled_proto = F.normalize(proto, dim=2) * vis_norm.unsqueeze(1).unsqueeze(2)

            if final_mask.any():
                mean_conf = float(proto_conf[final_mask].mean().item())
                mean_count = float(proto_count[final_mask].float().mean().item())
            else:
                mean_conf = 0.0
                mean_count = 0.0

        stats = {
            'n_support': int(final_mask.sum().item()),
            'support_ratio': float(final_mask.float().mean().item()),
            'low_ratio': float(low_mask.float().mean().item()),
            'proto_conf': mean_conf,
            'proto_count': mean_count,
        }
        return {
            'proto': scaled_proto.detach(),
            'mask': final_mask,
            'low_mask': low_mask,
            'proto_conf': proto_conf.detach(),
            'proto_count': proto_count.detach(),
            'stats': stats,
        }

    def replace(self, kp_feats, kp_weights, labels):
        """Replace low-visibility keypoint features with prototype features.

        Unlike compute_loss (which adds a gradient signal), this directly
        substitutes the features. The skeleton_head input is already detached
        from the backbone (feat_map_detached), so no backbone gradients are
        affected. Bank lookups use torch.no_grad internally.

        Note on bank update path: the bank is updated from post-GCN features
        with update_thr >= 0.7, so only high-visibility keypoints (score >= 0.7)
        are written. SCFR replaces keypoints with score <= 0.3. The GCN
        propagation from replaced neighbors to visible keypoints is an
        intentional smoothing effect, not contamination.

        Args:
            kp_feats: (B, 17, C) keypoint features
            kp_weights: (B, 17) confidence / visibility weights
            labels: (B,) identity labels

        Returns:
            replaced_feats: (B, 17, C) with low-vis kps replaced
            replace_mask: (B, 17) bool, True where replacement happened
            stats: dict with replacement statistics
        """
        support = self.get_support(kp_feats, kp_weights, labels)
        replaced = kp_feats.clone()
        replace_mask = support['mask']
        query_mask = support['low_mask'] & (support['proto_count'] >= self.min_count) & (support['proto_conf'] > 0)
        n_replaced = int(replace_mask.sum().item())

        if n_replaced > 0:
            replaced[replace_mask] = support['proto'][replace_mask]

        stats = {
            'n_replaced': n_replaced,
            # Keep historical SCFR log semantics: ratio before valid-sample guard.
            'replace_ratio': float(query_mask.float().mean().item()),
            'applied_ratio': support['stats']['support_ratio'],
            'low_ratio': support['stats']['low_ratio'],
            'proto_conf': support['stats']['proto_conf'],
            'proto_count': support['stats']['proto_count'],
        }
        return replaced, replace_mask, stats
