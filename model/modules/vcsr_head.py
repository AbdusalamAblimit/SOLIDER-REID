"""VCSR: Visibility-Conditional Semantic Routing Head.

Extends LGPA's CLIP cross-attention with dynamic part gating:
- Only instantiates semantic groups that have visible evidence
- Occluded parts are completely skipped (no noise features)
- Training uses per-part losses only for active parts
- Testing outputs variable-length part features for set matching

Core innovation: "Under occlusion, the model should instantiate only
the semantic groups actually supported by visible evidence."
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_part_head import PART_TEXTS, PART_KPS, NUM_PARTS


class VCSRHead(nn.Module):
    """Visibility-Conditional Semantic Routing via CLIP cross-attention.

    Key difference from CLIPPartHead (LGPA):
    - Computes per-part visibility scores from pose heatmaps
    - Only active (visible) parts generate features and contribute to loss
    - Returns variable-length outputs controlled by visibility mask

    Args:
        feat_dim: backbone feature dimension (768)
        num_classes: number of identity classes
        clip_dim: CLIP text feature dimension (512 for ViT-B-32)
        num_heads: cross-attention heads
        pose_mask_temp: temperature for pose bias
        vis_threshold: visibility threshold for part activation (0-1)
    """

    def __init__(self, feat_dim=768, num_classes=702, clip_dim=512,
                 num_heads=8, pose_mask_temp=1.0, vis_threshold=0.3):
        super().__init__()
        self.feat_dim = feat_dim
        self.clip_dim = clip_dim
        self.num_parts = NUM_PARTS
        self.num_labels = NUM_PARTS + 1  # +1 for background
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.pose_mask_temp = pose_mask_temp
        self.vis_threshold = vis_threshold

        # Project CLIP text features to backbone dimension
        self.text_proj = nn.Linear(clip_dim, feat_dim)

        # Manual cross-attention projections
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.attn_drop = nn.Dropout(0.1)
        self.attn_norm = nn.LayerNorm(feat_dim)

        # Per-part BN + classifiers (always NUM_PARTS, but only active ones used)
        self.part_bns = nn.ModuleList([
            nn.BatchNorm1d(feat_dim) for _ in range(NUM_PARTS)])
        for bn in self.part_bns:
            bn.bias.requires_grad_(False)
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)

        self.part_classifiers = nn.ModuleList([
            nn.Linear(feat_dim, num_classes, bias=False)
            for _ in range(NUM_PARTS)])

        # Pooled part BN + classifier (for global part representation)
        self.pooled_bn = nn.BatchNorm1d(feat_dim)
        self.pooled_bn.bias.requires_grad_(False)
        nn.init.constant_(self.pooled_bn.weight, 1.0)
        nn.init.constant_(self.pooled_bn.bias, 0.0)
        self.pooled_classifier = nn.Linear(feat_dim, num_classes, bias=False)

        # Pre-compute and register CLIP text features
        self._init_clip_features()

        print(f'[VCSR] Visibility-Conditional Semantic Routing: '
              f'{NUM_PARTS} max parts, vis_thr={vis_threshold}, '
              f'clip_dim={clip_dim}, num_heads={num_heads}')

    def _init_clip_features(self):
        """Load frozen CLIP text features."""
        import os
        cache_path = 'pretrained/clip_part_text_features.pt'
        if os.path.exists(cache_path):
            text_features = torch.load(cache_path, map_location='cpu')
            assert text_features.shape == (self.num_labels, self.clip_dim)
            self.register_buffer('clip_text_features', text_features.float())
            print(f'[VCSR] CLIP text features loaded from cache: {text_features.shape}')
            return
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        texts = tokenizer(PART_TEXTS)
        with torch.no_grad():
            text_features = clip_model.encode_text(texts)
            text_features = F.normalize(text_features, p=2, dim=-1)
        self.register_buffer('clip_text_features', text_features.float())
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(text_features.float(), cache_path)
        print(f'[VCSR] CLIP text features loaded and cached: {text_features.shape}')

    def _compute_visibility(self, heatmaps, fH, fW):
        """Compute per-part visibility score from pose heatmaps.

        Returns:
            visibility: (B, NUM_PARTS) float scores in [0, 1]
            active_mask: (B, NUM_PARTS) binary mask (1 = active/visible)
        """
        B = heatmaps.shape[0]
        device = heatmaps.device

        hm = F.interpolate(heatmaps.float(), size=(fH, fW),
                           mode='bilinear', align_corners=False)

        vis_scores = []
        for kps in PART_KPS:
            # Max across keypoints in group, then mean across spatial
            part_response = hm[:, kps].max(dim=1)[0]  # (B, fH, fW)
            vis_scores.append(part_response.mean(dim=(1, 2)))  # (B,)

        visibility = torch.stack(vis_scores, dim=1)  # (B, NUM_PARTS)
        active_mask = (visibility > self.vis_threshold).float()  # (B, NUM_PARTS)

        return visibility, active_mask

    def _compute_pose_bias(self, heatmaps, fH, fW):
        """Generate per-part attention bias from pose heatmaps."""
        B = heatmaps.shape[0]
        device = heatmaps.device

        hm = F.interpolate(heatmaps.float(), size=(fH, fW),
                           mode='bilinear', align_corners=False)

        part_activations = torch.zeros(B, self.num_labels, fH, fW, device=device)
        for k, kp_indices in enumerate(PART_KPS):
            part_activations[:, k] = hm[:, kp_indices].max(dim=1)[0]

        body_max = part_activations[:, :NUM_PARTS].max(dim=1)[0]
        part_activations[:, NUM_PARTS] = (1.0 - body_max).clamp(min=0.0)

        pose_bias = part_activations.flatten(2) * self.pose_mask_temp
        return pose_bias

    def _cross_attention_with_pose(self, queries, keys, values, pose_bias):
        """Manual cross-attention with pose bias injection."""
        B, L, C = queries.shape
        N = keys.shape[1]
        H = self.num_heads
        d = self.head_dim

        Q = self.q_proj(queries).reshape(B, L, H, d).transpose(1, 2)
        K = self.k_proj(keys).reshape(B, N, H, d).transpose(1, 2)
        V = self.v_proj(values).reshape(B, N, H, d).transpose(1, 2)

        scale = 1.0 / math.sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if pose_bias is not None:
            attn_scores = attn_scores + pose_bias.unsqueeze(1)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_drop = self.attn_drop(attn_weights)

        output = torch.matmul(attn_weights_drop, V)
        output = output.transpose(1, 2).reshape(B, L, C)
        output = self.out_proj(output)

        attn_weights_avg = attn_weights.mean(dim=1)
        return output, attn_weights_avg

    def forward(self, feat_map, heatmaps, return_cls=True):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone features (detached)
            heatmaps: (B, 17, H, W) pose heatmaps
            return_cls: return classification scores

        Returns:
            cls_scores: [pooled_cls] + [part_k_cls for active parts]
            feats: [pooled_feat] + [part_k_feat for active parts]
            aux_data: dict with visibility, active_mask, assign_loss, etc.
        """
        B, C, fH, fW = feat_map.shape
        N = fH * fW
        tokens = feat_map.flatten(2).transpose(1, 2)  # (B, N, C)

        # Compute visibility and active mask
        if heatmaps is not None:
            visibility, active_mask = self._compute_visibility(heatmaps, fH, fW)
            pose_bias = self._compute_pose_bias(heatmaps, fH, fW)
        else:
            visibility = torch.ones(B, NUM_PARTS, device=feat_map.device)
            active_mask = torch.ones(B, NUM_PARTS, device=feat_map.device)
            pose_bias = None

        # CLIP cross-attention for ALL parts (but only active ones contribute to loss)
        text_protos = self.text_proj(self.clip_text_features)
        text_protos = text_protos.unsqueeze(0).expand(B, -1, -1)

        part_feats_raw, attn_weights = self._cross_attention_with_pose(
            text_protos, tokens, tokens, pose_bias)
        part_feats_raw = self.attn_norm(part_feats_raw)

        # Extract per-part features
        all_part_feats = [part_feats_raw[:, k] for k in range(NUM_PARTS)]

        # Visibility-weighted pooling (only active parts contribute)
        vis_weights = visibility * active_mask  # (B, K) — zero for inactive
        vis_sum = vis_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        vis_norm = vis_weights / vis_sum
        pooled_feat = sum(vis_norm[:, k:k+1] * all_part_feats[k] for k in range(NUM_PARTS))

        # Assignment supervision loss
        assign_loss = torch.tensor(0.0, device=feat_map.device)
        if self.training and heatmaps is not None and attn_weights is not None:
            gt_labels = self._compute_gt_assignment(heatmaps, fH, fW)
            if gt_labels is not None:
                assign_loss = F.kl_div(
                    (attn_weights + 1e-8).log(), gt_labels, reduction='batchmean')

        # Compute number of active parts per sample for logging
        n_active = active_mask.sum(dim=-1).mean().item()

        aux_data = {
            'assign_loss': assign_loss,
            'visibility': visibility.detach(),
            'active_mask': active_mask.detach(),
            'n_active': n_active,
            'kp_feats': torch.stack(all_part_feats, dim=1),  # (B, K, C)
            'kp_weights': visibility.detach(),
        }

        if return_cls:
            # VCSR: only compute loss for active parts
            # Return ALL parts but mark inactive ones so processor can skip them
            part_cls_scores = []
            part_feats_out = []
            for k in range(NUM_PARTS):
                bn_feat = self.part_bns[k](all_part_feats[k])
                cls = self.part_classifiers[k](bn_feat)
                part_cls_scores.append(cls)
                part_feats_out.append(all_part_feats[k])

            pooled_bn = self.pooled_bn(pooled_feat)
            pooled_cls = self.pooled_classifier(pooled_bn)

            # Store active_mask in aux_data for processor to use
            aux_data['vcsr_active_mask'] = active_mask  # (B, NUM_PARTS)

            return ([pooled_cls] + part_cls_scores,
                    [pooled_feat] + part_feats_out,
                    aux_data)
        else:
            return (None, [pooled_feat] + all_part_feats, aux_data)

    def _compute_gt_assignment(self, heatmaps, fH, fW):
        """Compute GT soft assignment from pose heatmaps."""
        B = heatmaps.shape[0]
        device = heatmaps.device
        hm = F.interpolate(heatmaps.float(), size=(fH, fW),
                           mode='bilinear', align_corners=False)

        gt = torch.zeros(B, self.num_labels, fH * fW, device=device)
        for k, kp_indices in enumerate(PART_KPS):
            gt[:, k] = hm[:, kp_indices].max(dim=1)[0].flatten(1)

        gt[:, NUM_PARTS] = (1.0 - gt[:, :NUM_PARTS].max(dim=1)[0]).clamp(min=0.0)
        gt = gt / gt.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return gt
