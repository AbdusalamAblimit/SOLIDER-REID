"""LGPA: Language-Grounded Part Assignment Head.

Uses frozen CLIP text embeddings as semantic body-part prototypes,
cross-attends backbone spatial tokens to these prototypes with
pose heatmaps injected as additive attention bias.

First to combine VLM semantic knowledge + geometric pose for ReID.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Body part text descriptions for CLIP encoding (5 parts + background)
# Text semantics must match the COCO keypoint groups exactly.
PART_TEXTS = [
    "head and face of a person",
    "torso and chest of a person",
    "arms and hands of a person",
    "upper legs and thighs of a person",
    "lower legs and feet of a person",
    "background scene",
]
NUM_PARTS = len(PART_TEXTS) - 1  # 5 body parts + 1 background

# COCO keypoint to part mapping (5 parts)
# Each group's keypoints must semantically match the corresponding PART_TEXT.
PART_KPS = [
    [0, 1, 2, 3, 4],       # head: nose, eyes, ears
    [5, 6, 11, 12],         # torso: shoulders, hips
    [5, 6, 7, 8, 9, 10],    # arms: both shoulders, elbows, wrists
    [11, 12, 13, 14],       # upper legs: hips, knees
    [15, 16],               # lower legs: ankles
]


class CLIPPartHead(nn.Module):
    """Language-Grounded Part Assignment via CLIP cross-attention.

    Pose heatmaps are injected as additive bias in the cross-attention
    score matrix: attn = softmax(QK^T / sqrt(d) + pose_bias).

    Args:
        feat_dim: backbone feature dimension (768)
        num_classes: number of identity classes
        clip_dim: CLIP text feature dimension (512 for ViT-B-32)
        num_heads: cross-attention heads
        pose_mask_temp: temperature scaling for pose bias
    """

    def __init__(self, feat_dim=768, num_classes=702, clip_dim=512,
                 num_heads=8, pose_mask_temp=1.0,
                 query_mode='clip_frozen', query_seed=42):
        super().__init__()
        self.feat_dim = feat_dim
        self.clip_dim = clip_dim
        self.num_parts = NUM_PARTS
        self.num_labels = NUM_PARTS + 1  # +1 for background
        self.num_heads = num_heads
        self.pose_mask_temp = pose_mask_temp
        self.head_dim = feat_dim // num_heads
        self.query_mode = str(query_mode)
        self.query_seed = int(query_seed)
        valid_query_modes = {'clip_frozen', 'random_frozen', 'random_learned'}
        if self.query_mode not in valid_query_modes:
            raise ValueError(
                f'Unknown LGPA query_mode={self.query_mode!r}; '
                f'expected one of {sorted(valid_query_modes)}')

        # Project CLIP text features to backbone dimension
        self.text_proj = nn.Linear(clip_dim, feat_dim)

        # Manual cross-attention projections (so we can inject pose bias)
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.attn_drop = nn.Dropout(0.1)
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

        # Register the six query codes under the historical state-dict key.
        # random_frozen and random_learned start from exactly the same tensor;
        # only buffer-vs-Parameter registration differs.
        self._init_query_features()

        print(f'[LGPA] Part Head: {NUM_PARTS} parts, '
              f'clip_dim={clip_dim}, num_heads={num_heads}, '
              f'pose_temp={pose_mask_temp}, query_mode={self.query_mode}, '
              f'query_seed={self.query_seed}')

    def _init_query_features(self):
        if self.query_mode == 'clip_frozen':
            self._init_clip_features()
            return

        generator = torch.Generator(device='cpu').manual_seed(self.query_seed)
        query_features = F.normalize(
            torch.randn(self.num_labels, self.clip_dim, generator=generator),
            p=2, dim=-1).float()
        if self.query_mode == 'random_learned':
            self.clip_text_features = nn.Parameter(query_features)
        else:
            self.register_buffer('clip_text_features', query_features)
        print(f'[LGPA] Random query IDs initialized: {query_features.shape}, '
              f'seed={self.query_seed}, learned={self.query_mode == "random_learned"}')

    def _init_clip_features(self):
        """Load frozen CLIP text features. Try cached file first, then live CLIP."""
        import os
        cache_path = 'pretrained/clip_part_text_features.pt'
        if os.path.exists(cache_path):
            text_features = torch.load(cache_path, map_location='cpu')
            assert text_features.shape == (self.num_labels, self.clip_dim), \
                f'Cached features shape {text_features.shape} != expected ({self.num_labels}, {self.clip_dim})'
            self.register_buffer('clip_text_features', text_features.float())
            print(f'[LGPA] CLIP text features loaded from cache: {text_features.shape}')
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
        # Cache for future use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(text_features.float(), cache_path)
        print(f'[LGPA] CLIP text features loaded and cached: {text_features.shape}')

    def _compute_pose_bias(self, heatmaps, fH, fW):
        """Generate per-part attention bias from pose heatmaps.

        Returns additive bias for cross-attention: shape (B, num_labels, N).
        This is added to QK^T/sqrt(d) BEFORE softmax.

        Args:
            heatmaps: (B, 17, H, W) — TARGET person heatmaps
            fH, fW: feature map spatial size

        Returns:
            pose_bias: (B, num_labels, fH*fW)
        """
        B = heatmaps.shape[0]
        device = heatmaps.device

        # Resize heatmaps to feature map size
        hm = F.interpolate(heatmaps.float(), size=(fH, fW),
                           mode='bilinear', align_corners=False)

        # Compute per-part activation
        part_activations = torch.zeros(B, self.num_labels, fH, fW, device=device)
        for k, kp_indices in enumerate(PART_KPS):
            part_activations[:, k] = hm[:, kp_indices].max(dim=1)[0]

        # Background: inverse of body
        body_max = part_activations[:, :NUM_PARTS].max(dim=1)[0]
        part_activations[:, NUM_PARTS] = (1.0 - body_max).clamp(min=0.0)

        # Flatten and scale by temperature
        pose_bias = part_activations.flatten(2)  # (B, num_labels, N)
        pose_bias = pose_bias * self.pose_mask_temp

        return pose_bias

    def _cross_attention_with_pose(self, queries, keys, values, pose_bias):
        """Manual cross-attention with pose bias injection.

        attn_scores = QK^T / sqrt(d) + pose_bias
        attn_weights = softmax(attn_scores)
        output = attn_weights @ V

        Args:
            queries: (B, L, C) — text prototypes (L = num_labels)
            keys: (B, N, C) — spatial tokens
            values: (B, N, C) — spatial tokens
            pose_bias: (B, L, N) or None — additive attention bias

        Returns:
            output: (B, L, C)
            attn_weights: (B, L, N) — averaged across heads
        """
        B, L, C = queries.shape
        N = keys.shape[1]
        H = self.num_heads
        d = self.head_dim

        # Project Q, K, V
        Q = self.q_proj(queries).reshape(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        K = self.k_proj(keys).reshape(B, N, H, d).transpose(1, 2)     # (B, H, N, d)
        V = self.v_proj(values).reshape(B, N, H, d).transpose(1, 2)   # (B, H, N, d)

        # Attention scores: QK^T / sqrt(d)
        scale = 1.0 / math.sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, L, N)

        # Inject pose bias: broadcast across heads
        if pose_bias is not None:
            attn_scores = attn_scores + pose_bias.unsqueeze(1)  # (B, H, L, N)

        # Clamp scores to prevent late-training weight drift from producing +/-Inf
        # that softmax turns into NaN (e.g. +inf - max(+inf) = NaN). [-50, 50] keeps
        # exp in fp32-safe range; extreme differences still give clean 1-hot-ish softmax.
        attn_scores = attn_scores.clamp(min=-50.0, max=50.0)

        # Softmax (for V pooling) + log-softmax (for numerically stable KL-div)
        attn_weights_full = F.softmax(attn_scores, dim=-1)  # (B, H, L, N)
        log_attn_full = F.log_softmax(attn_scores, dim=-1)  # (B, H, L, N), uses log-sum-exp trick
        attn_weights_drop = self.attn_drop(attn_weights_full)

        # Weighted sum of values
        output = torch.matmul(attn_weights_drop, V)  # (B, H, L, d)
        output = output.transpose(1, 2).reshape(B, L, C)  # (B, L, C)
        output = self.out_proj(output)

        # Head-averaged attention (for downstream use) and its log (for loss).
        # log(mean_H(softmax)) = logsumexp_H(log_softmax) - log(H), stable vs softmax→+eps→log.
        attn_weights_avg = attn_weights_full.mean(dim=1)  # (B, L, N)
        log_attn_avg = torch.logsumexp(log_attn_full, dim=1) - math.log(self.num_heads)  # (B, L, N)
        # Clamp from below: extreme attn_scores (late-training drift) can push log to -inf.
        # kl_div hits `target * (log_target - (-inf)) = 0 * inf = NaN` when target is 0 but
        # log_q is -inf. exp(-30) ≈ 9e-14 — any finer granularity is meaningless in fp32.
        log_attn_avg = log_attn_avg.clamp(min=-30.0)

        return output, attn_weights_avg, log_attn_avg

    def forward(self, feat_map, target_heatmaps, return_cls=True):
        """
        Args:
            feat_map: (B, C, fH, fW) backbone features (NOT detached)
            target_heatmaps: (B, 17, H, W) TARGET person heatmaps
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

        # Compute pose bias for attention
        if target_heatmaps is not None:
            pose_bias = self._compute_pose_bias(target_heatmaps, fH, fW)
        else:
            pose_bias = None

        # Pose-conditioned cross-attention: text queries → spatial keys/values
        # attn = softmax(QK^T / sqrt(d) + pose_bias)
        part_feats_raw, attn_weights, log_attn_weights = self._cross_attention_with_pose(
            text_protos, tokens, tokens, pose_bias)
        # part_feats_raw: (B, num_labels, C)
        # attn_weights:      (B, num_labels, N) — head-averaged softmax, for visibility/debug
        # log_attn_weights:  (B, num_labels, N) — log of attn_weights via log-sum-exp, for KL-div

        part_feats_raw = self.attn_norm(part_feats_raw)

        # Extract body part features (exclude background)
        part_feats = [part_feats_raw[:, k] for k in range(NUM_PARTS)]

        # Pose-derived visibility: use heatmap response per body part
        if target_heatmaps is not None:
            hm_vis = F.interpolate(target_heatmaps.float(), size=(fH, fW),
                                   mode='bilinear', align_corners=False)
            vis_scores = []
            for kps in PART_KPS:
                vis_scores.append(hm_vis[:, kps].max(dim=1)[0].mean(dim=(1, 2)))
            visibility = torch.stack(vis_scores, dim=1)  # (B, K)
            visibility = visibility / visibility.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            visibility = torch.ones(B, NUM_PARTS, device=feat_map.device) / NUM_PARTS

        # Pooled part feature (visibility-weighted)
        pooled_feat = sum(visibility[:, k:k+1] * part_feats[k] for k in range(NUM_PARTS))

        # Assignment supervision loss (pose-based GT)
        assign_loss = torch.tensor(0.0, device=feat_map.device)
        if self.training and target_heatmaps is not None and log_attn_weights is not None:
            gt_labels = self._compute_gt_assignment(target_heatmaps, fH, fW)
            if gt_labels is not None and torch.isfinite(gt_labels).all():
                # KL(GT || predicted). log_attn_weights is log-sum-exp computed and
                # clamped at -30, so both forward and backward are bounded.
                raw_loss = F.kl_div(
                    log_attn_weights,
                    gt_labels,
                    reduction='batchmean'
                )
                # Defensive: if some rare numerical path still produces NaN/Inf
                # (e.g. gt_labels has 0 at a position where log_attn clamped at -30
                # still lets target * log interactions go wrong), skip rather than
                # letting it poison the LGPA head weights via backward.
                if torch.isfinite(raw_loss):
                    assign_loss = raw_loss

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

    def _compute_gt_assignment(self, heatmaps, fH, fW):
        """Compute GT soft assignment from pose heatmaps."""
        B = heatmaps.shape[0]
        device = heatmaps.device

        hm = F.interpolate(heatmaps.float(), size=(fH, fW),
                           mode='bilinear', align_corners=False)

        gt = torch.zeros(B, self.num_labels, fH * fW, device=device)
        for k, kp_indices in enumerate(PART_KPS):
            gt[:, k] = hm[:, kp_indices].max(dim=1)[0].flatten(1)

        # Background
        gt[:, NUM_PARTS] = (1.0 - gt[:, :NUM_PARTS].max(dim=1)[0]).clamp(min=0.0)

        # Normalize to distribution
        gt = gt / gt.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return gt
