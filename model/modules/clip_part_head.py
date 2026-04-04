"""LGPA: Language-Grounded Part Assignment Head.

Uses frozen CLIP text embeddings as semantic body-part prototypes,
cross-attends backbone spatial tokens to these prototypes,
and uses pose heatmaps as spatial attention masks.

First to combine VLM semantic knowledge + geometric pose for ReID.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Body part text descriptions for CLIP encoding
PART_TEXTS = [
    "head of a person",
    "torso of a person",
    "arms of a person",
    "legs of a person",
    "background scene",
]
NUM_PARTS = len(PART_TEXTS) - 1  # 4 body parts + 1 background


class CLIPPartHead(nn.Module):
    """Language-Grounded Part Assignment via CLIP cross-attention.

    Args:
        feat_dim: backbone feature dimension (768)
        num_classes: number of identity classes
        clip_dim: CLIP text feature dimension (512 for ViT-B-32)
        num_heads: cross-attention heads
        pose_mask_temp: temperature for pose mask softness
    """

    def __init__(self, feat_dim=768, num_classes=702, clip_dim=512,
                 num_heads=8, pose_mask_temp=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.clip_dim = clip_dim
        self.num_parts = NUM_PARTS
        self.num_labels = NUM_PARTS + 1  # +1 for background
        self.pose_mask_temp = pose_mask_temp

        # Project CLIP text features to backbone dimension
        self.text_proj = nn.Linear(clip_dim, feat_dim)

        # Cross-attention: text queries attend to spatial tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(feat_dim)

        # Per-part BN + classifiers
        self.part_bns = nn.ModuleList([
            nn.BatchNorm1d(feat_dim) for _ in range(NUM_PARTS)])
        for bn in self.part_bns:
            bn.bias.requires_grad_(False)
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)

        self.part_classifiers = nn.ModuleList([
            nn.Linear(feat_dim, num_classes, bias=False)
            for _ in range(NUM_PARTS)])

        # Pooled part BN + classifier
        self.pooled_bn = nn.BatchNorm1d(feat_dim)
        self.pooled_bn.bias.requires_grad_(False)
        nn.init.constant_(self.pooled_bn.weight, 1.0)
        nn.init.constant_(self.pooled_bn.bias, 0.0)
        self.pooled_classifier = nn.Linear(feat_dim, num_classes, bias=False)

        # Pre-compute and register CLIP text features
        self._init_clip_features()

        print(f'[LGPA] CLIP Part Head: {NUM_PARTS} parts, '
              f'clip_dim={clip_dim}, num_heads={num_heads}')

    def _init_clip_features(self):
        """Pre-compute frozen CLIP text features."""
        try:
            import open_clip
            clip_model, _, _ = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            texts = tokenizer(PART_TEXTS)
            with torch.no_grad():
                text_features = clip_model.encode_text(texts)
                text_features = F.normalize(text_features, p=2, dim=-1)
            # Register as buffer (not trainable, moves with model)
            self.register_buffer('clip_text_features', text_features.float())
            print(f'[LGPA] CLIP text features loaded: {text_features.shape}')
        except Exception as e:
            print(f'[LGPA] WARNING: CLIP init failed: {e}. Using random init.')
            self.register_buffer('clip_text_features',
                                 torch.randn(self.num_labels, self.clip_dim))

    def _compute_pose_mask(self, heatmaps, fH, fW):
        """Generate per-part attention mask from pose heatmaps.

        For each body part, identifies which spatial tokens belong to it.

        Args:
            heatmaps: (B, 17, H, W)
            fH, fW: feature map spatial size

        Returns:
            mask_bias: (B, num_labels, fH*fW) attention bias
                       High value = this token belongs to this part
        """
        B = heatmaps.shape[0]
        device = heatmaps.device
        N = fH * fW

        # Resize heatmaps to feature map size
        hm = F.interpolate(heatmaps.float(), size=(fH, fW),
                           mode='bilinear', align_corners=False)

        # COCO keypoint to part mapping
        part_kps = [
            [0, 1, 2, 3, 4],      # head
            [5, 6, 11, 12],        # torso
            [5, 7, 8, 9, 10],      # arms (include shoulders)
            [11, 13, 14, 15, 16],  # legs (include hips)
        ]

        # Compute per-part activation
        part_activations = torch.zeros(B, self.num_labels, fH, fW, device=device)
        for k, kp_indices in enumerate(part_kps):
            part_activations[:, k] = hm[:, kp_indices].max(dim=1)[0]

        # Background: low activation everywhere
        body_max = part_activations[:, :NUM_PARTS].max(dim=1)[0]
        part_activations[:, NUM_PARTS] = (1.0 - body_max).clamp(min=0.0)

        # Convert to attention bias: higher = more likely this part
        mask_bias = part_activations.flatten(2)  # (B, num_labels, N)
        mask_bias = mask_bias * self.pose_mask_temp

        return mask_bias

    def forward(self, feat_map, scene_heatmaps, return_cls=True):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone features (NOT detached)
            scene_heatmaps: (B, 17, H, W) pose heatmaps
            return_cls: return classification scores

        Returns:
            cls_scores: [pooled_cls, part1_cls..partK_cls]
            feats: [pooled_feat, part1_feat..partK_feat]
            aux_data: dict with stats
        """
        B, C, fH, fW = feat_map.shape
        N = fH * fW
        tokens = feat_map.flatten(2).transpose(1, 2)  # (B, N, C)

        # Project CLIP text features to backbone dimension
        text_protos = self.text_proj(self.clip_text_features)  # (num_labels, C)
        text_protos = text_protos.unsqueeze(0).expand(B, -1, -1)  # (B, num_labels, C)

        # Compute pose-conditioned attention mask
        if scene_heatmaps is not None:
            pose_bias = self._compute_pose_mask(scene_heatmaps, fH, fW)
            # Convert to attention mask format for MultiheadAttention
            # attn_mask: (B*num_heads, num_labels, N) — additive bias
            # MultiheadAttention expects (L, S) or (B*num_heads, L, S)
            # We'll add it manually after computing raw attention
        else:
            pose_bias = None

        # Cross-attention: text queries → spatial keys/values
        # Q = text_protos (B, num_labels, C)
        # K = V = tokens (B, N, C)
        part_feats_raw, attn_weights = self.cross_attn(
            text_protos, tokens, tokens,
            need_weights=True, average_attn_weights=True
        )
        # part_feats_raw: (B, num_labels, C)
        # attn_weights: (B, num_labels, N)

        # Apply pose bias to re-weight attention (soft masking)
        if pose_bias is not None:
            pose_weights = F.softmax(pose_bias, dim=-1)  # (B, num_labels, N)
            pose_modulated = torch.bmm(pose_weights, tokens)  # (B, num_labels, C)

            # Blend cross-attn output with pose-modulated output
            part_feats_raw = 0.5 * part_feats_raw + 0.5 * pose_modulated

        part_feats_raw = self.attn_norm(part_feats_raw)

        # Extract body part features (exclude background)
        part_feats = [part_feats_raw[:, k] for k in range(NUM_PARTS)]  # list of (B, C)

        # Pose-derived visibility: use heatmap response per body part
        if scene_heatmaps is not None:
            hm_vis = F.interpolate(scene_heatmaps.float(), size=(fH, fW),
                                   mode='bilinear', align_corners=False)
            part_kps_vis = [
                [0, 1, 2, 3, 4],      # head
                [5, 6, 11, 12],        # torso
                [5, 7, 8, 9, 10],      # arms
                [11, 13, 14, 15, 16],  # legs
            ]
            vis_scores = []
            for kps in part_kps_vis:
                vis_scores.append(hm_vis[:, kps].max(dim=1)[0].mean(dim=(1, 2)))
            visibility = torch.stack(vis_scores, dim=1)  # (B, K)
            visibility = visibility / visibility.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            visibility = torch.ones(B, NUM_PARTS, device=feat_map.device) / NUM_PARTS

        # Pooled part feature (visibility-weighted)
        pooled_feat = sum(visibility[:, k:k+1] * part_feats[k] for k in range(NUM_PARTS))

        # Assignment supervision loss (pose-based GT)
        assign_loss = torch.tensor(0.0, device=feat_map.device)
        if self.training and scene_heatmaps is not None and attn_weights is not None:
            gt_labels = self._compute_gt_assignment(scene_heatmaps, fH, fW, attn_weights)
            if gt_labels is not None:
                # Soft CE between attention weights and GT
                assign_loss = F.kl_div(
                    (attn_weights + 1e-8).log(),
                    gt_labels,
                    reduction='batchmean'
                )

        aux_data = {
            'assign_loss': assign_loss,
            'kp_feats': torch.stack(part_feats, dim=1),  # (B, K, C) for MaxSim
            'kp_weights': visibility.detach(),
        }

        if return_cls:
            part_cls_scores = []
            for k in range(NUM_PARTS):
                bn_feat = self.part_bns[k](part_feats[k])
                cls = self.part_classifiers[k](bn_feat)
                part_cls_scores.append(cls)

            pooled_bn = self.pooled_bn(pooled_feat)
            pooled_cls = self.pooled_classifier(pooled_bn)

            return ([pooled_cls] + part_cls_scores,
                    [pooled_feat] + part_feats,
                    aux_data)
        else:
            return (None, [pooled_feat] + part_feats, aux_data)

    def _compute_gt_assignment(self, heatmaps, fH, fW, attn_weights):
        """Compute GT soft assignment from pose heatmaps."""
        B = heatmaps.shape[0]
        device = heatmaps.device

        hm = F.interpolate(heatmaps.float(), size=(fH, fW),
                           mode='bilinear', align_corners=False)

        part_kps = [
            [0, 1, 2, 3, 4],
            [5, 6, 11, 12],
            [5, 7, 8, 9, 10],
            [11, 13, 14, 15, 16],
        ]

        gt = torch.zeros(B, self.num_labels, fH * fW, device=device)
        for k, kp_indices in enumerate(part_kps):
            gt[:, k] = hm[:, kp_indices].max(dim=1)[0].flatten(1)

        # Background
        gt[:, NUM_PARTS] = (1.0 - gt[:, :NUM_PARTS].max(dim=1)[0]).clamp(min=0.0)

        # Normalize to distribution
        gt = gt / gt.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return gt
