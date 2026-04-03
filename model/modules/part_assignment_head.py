"""Pose-Prompted Part-Assignment Head (PPA).

Replaces detached GCN keypoint sampling with a learnable softmax
part-assignment head. End-to-end trainable — gradients flow to backbone.

Key insight: KPR (ECCV 2024) proves softmax part assignment works to 75.1% mAP.
We supervise with pose heatmaps instead of parsing labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 5 body parts + background
PART_NAMES = ['head', 'torso', 'arms', 'legs', 'background']
NUM_PARTS = 5  # K=5, plus background = K+1=6

# Mapping from COCO 17 keypoints to 5 body parts
KP_TO_PART = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0,    # head: nose, eyes, ears
    5: 1, 6: 1, 11: 1, 12: 1,          # torso: shoulders, hips
    7: 2, 8: 2, 9: 2, 10: 2,           # arms: elbows, wrists
    13: 3, 14: 3, 15: 3, 16: 3,        # legs: knees, ankles
}


class PartAssignmentHead(nn.Module):
    """Learnable part-assignment head with pose heatmap supervision.

    For each spatial token, predicts which body part it belongs to.
    Pools tokens per-part with soft weights to produce K part embeddings.
    All gradients flow end-to-end through the backbone.

    Args:
        feat_dim: token feature dimension (768)
        num_classes: number of identity classes
        num_parts: number of body parts (5)
        assign_weight: weight of assignment loss
    """

    def __init__(self, feat_dim=768, num_classes=702, num_parts=NUM_PARTS,
                 assign_weight=0.5):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_parts = num_parts
        self.num_labels = num_parts + 1  # +1 for background
        self.assign_weight = assign_weight

        # Part assignment projection: token → part logits
        self.part_proj = nn.Linear(feat_dim, self.num_labels)
        # Zero-init: start with uniform assignment (no bias toward any part)
        nn.init.zeros_(self.part_proj.weight)
        nn.init.zeros_(self.part_proj.bias)

        # Per-part BN + classifiers for ID loss
        self.part_bns = nn.ModuleList([
            nn.BatchNorm1d(feat_dim) for _ in range(num_parts)])
        for bn in self.part_bns:
            bn.bias.requires_grad_(False)
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)

        self.part_classifiers = nn.ModuleList([
            nn.Linear(feat_dim, num_classes, bias=False) for _ in range(num_parts)])

        # Pooled part feature BN + classifier (single combined part feature)
        self.pooled_bn = nn.BatchNorm1d(feat_dim)
        self.pooled_bn.bias.requires_grad_(False)
        nn.init.constant_(self.pooled_bn.weight, 1.0)
        nn.init.constant_(self.pooled_bn.bias, 0.0)
        self.pooled_classifier = nn.Linear(feat_dim, num_classes, bias=False)

        print(f'[PPA] Part Assignment Head: {num_parts} parts + background, '
              f'assign_weight={assign_weight}')

    def _heatmaps_to_part_labels(self, heatmaps, fH, fW):
        """Convert pose heatmaps to per-token part labels.

        Args:
            heatmaps: (B, 17, H, W) scene heatmaps
            fH, fW: feature map spatial size (12, 4)

        Returns:
            labels: (B, fH*fW) long tensor, values in [0, num_parts]
                    0-3 = body parts, 4 = background
        """
        B = heatmaps.shape[0]
        device = heatmaps.device

        # Resize heatmaps to feature map size
        hm = F.interpolate(heatmaps, size=(fH, fW), mode='bilinear',
                           align_corners=False)  # (B, 17, fH, fW)

        # Aggregate keypoint heatmaps per body part
        part_hm = torch.zeros(B, self.num_parts, fH, fW, device=device)
        for kp_idx, part_idx in KP_TO_PART.items():
            part_hm[:, part_idx] = torch.max(part_hm[:, part_idx],
                                              hm[:, kp_idx])

        # For each spatial location, assign to the part with highest activation
        # If all activations are low → background
        max_val, max_part = part_hm.max(dim=1)  # (B, fH, fW)

        # Background threshold: if max activation < mean, assign to background
        threshold = max_val.flatten(1).mean(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
        is_bg = max_val < threshold * 0.3

        labels = max_part.clone()  # (B, fH, fW)
        labels[is_bg] = self.num_parts  # background label

        return labels.flatten(1)  # (B, fH*fW)

    def forward(self, feat_map, scene_heatmaps, return_cls=True):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone feature map (NOT detached!)
            scene_heatmaps: (B, 17, H, W) pose heatmaps
            return_cls: if True, return classification scores

        Returns:
            cls_scores: list of (B, num_classes) — [pooled, part1..partK]
            feats: list of (B, C) — [pooled, part1..partK]
            aux_data: dict with assignment loss, stats, kp_feats/kp_weights for MaxSim
        """
        B, C, fH, fW = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)  # (B, N, C) where N=fH*fW

        # Part assignment logits
        logits = self.part_proj(tokens)  # (B, N, K+1)
        probs = F.softmax(logits, dim=-1)  # (B, N, K+1)

        # Assignment supervision loss
        assign_loss = torch.tensor(0.0, device=feat_map.device)
        if self.training and scene_heatmaps is not None:
            gt_labels = self._heatmaps_to_part_labels(scene_heatmaps, fH, fW)
            assign_loss = F.cross_entropy(
                logits.reshape(-1, self.num_labels),
                gt_labels.reshape(-1)
            )

        # Per-part weighted pooling
        part_feats = []
        part_cls_scores = []
        for k in range(self.num_parts):
            weights = probs[:, :, k].unsqueeze(-1)  # (B, N, 1)
            w_sum = weights.sum(dim=1).clamp(min=1e-6)  # (B, 1)
            part_feat = (tokens * weights).sum(dim=1) / w_sum  # (B, C)
            part_feats.append(part_feat)

            if return_cls:
                part_bn = self.part_bns[k](part_feat)
                part_cls = self.part_classifiers[k](part_bn)
                part_cls_scores.append(part_cls)

        # Pooled part feature: visibility-weighted average of all parts
        visibility = probs[:, :, :self.num_parts].max(dim=1)[0]  # (B, K)
        vis_weights = visibility / visibility.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled_feat = sum(vis_weights[:, k:k+1] * part_feats[k]
                         for k in range(self.num_parts))

        # Aux data for MaxSim and monitoring
        aux_data = {
            'assign_loss': assign_loss,
            'assign_probs': probs.detach(),
            'visibility': visibility.detach(),
            'kp_feats': torch.stack(part_feats, dim=1),  # (B, K, C) for MaxSim
            'kp_weights': visibility.detach(),  # (B, K) for MaxSim weighting
        }

        # Compute assignment stats for logging
        with torch.no_grad():
            assigned = probs.argmax(dim=-1)  # (B, N)
            bg_ratio = (assigned == self.num_parts).float().mean().item()
            aux_data['bg_ratio'] = bg_ratio
            aux_data['assign_entropy'] = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()

        if return_cls:
            pooled_bn = self.pooled_bn(pooled_feat)
            pooled_cls = self.pooled_classifier(pooled_bn)
            return ([pooled_cls] + part_cls_scores,
                    [pooled_feat] + part_feats,
                    aux_data)
        else:
            return (None,
                    [pooled_feat] + part_feats,
                    aux_data)
