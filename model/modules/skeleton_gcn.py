"""Skeleton Graph Convolutional Network for keypoint feature enhancement.

Uses the COCO human skeleton topology to propagate features between
keypoint locations. When a joint is occluded (low confidence), GCN
can propagate information from neighboring visible joints along
skeleton edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# COCO skeleton edges (0-indexed)
COCO_SKELETON_EDGES = [
    (0, 1), (0, 2),     # nose - eyes
    (1, 3), (2, 4),     # eyes - ears
    (5, 6),              # shoulders
    (5, 7), (7, 9),     # left arm: shoulder - elbow - wrist
    (6, 8), (8, 10),    # right arm: shoulder - elbow - wrist
    (5, 11), (6, 12),   # torso: shoulders - hips
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg: hip - knee - ankle
    (12, 14), (14, 16),  # right leg: hip - knee - ankle
]

# Additional edges for better connectivity
EXTRA_EDGES = [
    (0, 5), (0, 6),     # nose - shoulders (head-body connection)
]


class SkeletonGCN(nn.Module):
    """Graph Convolutional Network over COCO skeleton.

    Architecture:
    - Fixed adjacency matrix from COCO skeleton topology
    - 2-layer GCN with residual connection
    - Each layer: h = ReLU(A_norm @ h @ W)
    - Output: input + GCN_output (residual)

    Args:
        feat_dim: input/output feature dimension (768 for Swin-Tiny)
        hidden_dim: hidden layer dimension
        num_layers: number of GCN layers
        num_joints: number of keypoints (17 for COCO)
        use_extra_edges: add nose-shoulder connections
    """

    def __init__(self, feat_dim=768, hidden_dim=256, num_layers=2,
                 num_joints=17, use_extra_edges=True):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        # Build adjacency matrix
        edges = COCO_SKELETON_EDGES[:]
        if use_extra_edges:
            edges.extend(EXTRA_EDGES)

        adj = torch.zeros(num_joints, num_joints)
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        adj += torch.eye(num_joints)  # self-loops

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        degree = adj.sum(dim=1)
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
        adj_norm = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
        self.register_buffer('adj_norm', adj_norm)

        # GCN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = feat_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else feat_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        # Zero-init last layer for identity start
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        """
        Args:
            x: (B, num_joints, feat_dim) keypoint features

        Returns:
            (B, num_joints, feat_dim) enhanced keypoint features
        """
        h = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Graph convolution: aggregate from neighbors
            h = torch.matmul(self.adj_norm, h)  # (B, 17, dim)
            h = layer(h)  # (B, 17, out_dim)
            h = norm(h)
            if i < len(self.layers) - 1:
                h = F.relu(h, inplace=True)

        # Residual connection (zero-init makes this identity at start)
        return x + h


class SkeletonGCNHead(nn.Module):
    """Complete head: bilinear sample → optional GCN → confidence-weighted average.

    Replaces Part Pooling in the PDS Part branch. When ``use_gcn=False``,
    this becomes a pure keypoint-pooling head.

    Args:
        feat_dim: backbone feature dimension (768)
        hidden_dim: GCN hidden dimension
        num_layers: GCN layers
        num_classes: number of identity classes
        input_size: (H, W) of input images for coordinate mapping
    """

    def __init__(self, feat_dim, hidden_dim, num_layers, num_classes,
                 input_size=(384, 128), use_gcn=True,
                 kp_weight_mode='score', kp_triplet=False,
                 kp_learnable_attn=False,
                 sgmkc=False, sgmkc_ratio=0.3,
                 kp_uncertainty=False, kp_uncertainty_reg=0.1,
                 pke=False,
                 dpf=False,
                 mrkf=False, mrkf_s2_dim=384,
                 sgmt=False, sgmt_ratio=0.3, sgmt_threshold=0.3):
        super().__init__()
        self.feat_dim = feat_dim
        self.input_h, self.input_w = input_size
        self.num_joints = 17
        self.use_gcn = use_gcn
        self.kp_weight_mode = kp_weight_mode
        self.kp_triplet = kp_triplet
        self.kp_learnable_attn = kp_learnable_attn
        self.sgmkc = sgmkc
        self.sgmkc_ratio = sgmkc_ratio
        self.ttsfr = False  # Set by model init if POSE_TTSFR enabled
        self.ttsfr_threshold = 0.3
        self.kp_uncertainty = kp_uncertainty
        self.kp_uncertainty_reg = kp_uncertainty_reg
        self.pke = pke
        # DPF: Distributional Part Features — heatmap-weighted spatial pooling
        self.dpf = dpf
        # SGMT: Skeleton-Guided Masked Training
        self.sgmt = sgmt
        self.sgmt_ratio = sgmt_ratio
        self.sgmt_threshold = sgmt_threshold
        if self.sgmt:
            self.mask_token = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)
        # MRKF: Multi-Resolution Keypoint Features
        self.mrkf = mrkf
        if self.mrkf:
            # Project Stage 2 + Stage 3 features to a common dim, then fuse
            fuse_dim = 256
            self.mrkf_s2_proj = nn.Linear(mrkf_s2_dim, fuse_dim)
            self.mrkf_s3_proj = nn.Linear(feat_dim, fuse_dim)
            self.mrkf_fusion = nn.Linear(fuse_dim * 2, feat_dim)
            # Initialize fusion to approximate identity (output ≈ Stage 3 features)
            nn.init.zeros_(self.mrkf_fusion.weight)
            nn.init.zeros_(self.mrkf_fusion.bias)

        # Optional graph propagation over sampled keypoint features.
        if self.use_gcn:
            self.gcn = SkeletonGCN(
                feat_dim=feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_joints=17,
            )
        else:
            self.gcn = None

        # Learnable Keypoint Attention (LKA)
        if self.kp_learnable_attn:
            self.kp_attention = nn.Sequential(
                nn.Linear(17, 32),
                nn.ReLU(),
                nn.Linear(32, 17),
                nn.Sigmoid()
            )
            # Zero-init last layer → outputs ~0.5 initially (identity start)
            nn.init.zeros_(self.kp_attention[2].weight)
            nn.init.zeros_(self.kp_attention[2].bias)

        # Learned Keypoint Uncertainty head
        if self.kp_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()  # output in [0, 1], higher = more uncertain
            )

        # Probabilistic Keypoint Embeddings: per-keypoint log_sigma prediction
        if self.pke:
            self.sigma_head = nn.Linear(feat_dim, feat_dim)
            # Init: small sigma (log_sigma = -2 → sigma ≈ 0.14)
            nn.init.zeros_(self.sigma_head.weight)
            nn.init.constant_(self.sigma_head.bias, -2.0)

        # BN + Classifier for skeleton feature (uses mu only)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

        # Initialize BN
        self.bn.apply(self._init_bn)

    @staticmethod
    def _init_bn(m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def _sample_keypoint_features(self, feat_map, keypoints, scores,
                                  person_mask):
        """Bilinear sample features at keypoint locations.

        Args:
            feat_map: (B, C, fH, fW) feature map
            keypoints: (B, max_persons, 17, 2) pixel coordinates
            scores: (B, max_persons, 17) confidence scores
            person_mask: (B, max_persons) boolean mask

        Returns:
            kp_feats: (B, 17, C) sampled features
            kp_scores: (B, 17) confidence scores (from person 0)
        """
        B, C, fH, fW = feat_map.shape

        # Use person 0 (main person) keypoints
        kp_coords = keypoints[:, 0, :, :]  # (B, 17, 2) pixel coords
        kp_scores = scores[:, 0, :]  # (B, 17)

        # Map pixel coordinates to feature map coordinates
        # Normalize to [-1, 1] for grid_sample
        grid_x = kp_coords[:, :, 0] / self.input_w * 2 - 1  # (B, 17)
        grid_y = kp_coords[:, :, 1] / self.input_h * 2 - 1  # (B, 17)

        # Clamp to valid range
        grid_x = grid_x.clamp(-1, 1)
        grid_y = grid_y.clamp(-1, 1)

        # Build grid for grid_sample: (B, 17, 1, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)

        # Sample: (B, C, 17, 1)
        sampled = F.grid_sample(
            feat_map, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )
        # Reshape to (B, 17, C)
        kp_feats = sampled.squeeze(-1).permute(0, 2, 1)

        return kp_feats, kp_scores

    def _heatmap_pool_features(self, feat_map, heatmaps, scores, person_mask):
        """Heatmap-weighted spatial pooling for Distributional Part Features.

        Instead of sampling at keypoint centers, pools features over each
        body part's spatial region using the heatmap as a probability
        distribution. Computes both mean (identity signal) and variance
        (reliability indicator).

        Args:
            feat_map: (B, C, fH, fW) backbone feature map
            heatmaps: (B, max_persons, 17, hH, hW) pose heatmaps
            scores: (B, max_persons, 17) confidence scores
            person_mask: (B, max_persons) boolean

        Returns:
            kp_means: (B, 17, C) mean features per keypoint
            kp_vars: (B, 17, C) feature variance per keypoint
            kp_scores: (B, 17) confidence scores
        """
        B, C, fH, fW = feat_map.shape

        # Use person-0 heatmaps
        hm = heatmaps[:, 0]  # (B, 17, hH, hW)
        kp_scores = scores[:, 0, :]  # (B, 17)

        # Zero out if person-0 doesn't exist
        p0_mask = person_mask[:, 0].float()  # (B,)
        hm = hm * p0_mask.view(-1, 1, 1, 1)

        # Resize to feature map resolution
        hm = F.interpolate(hm, size=(fH, fW),
                           mode='bilinear', align_corners=False)

        # ReLU: raw ViTPose output can have small negatives
        hm = F.relu(hm)

        # Normalize each channel → spatial probability distribution
        hm_sum = hm.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        attn = hm / hm_sum  # (B, 17, fH, fW), each sums to 1

        # Mean: μ_k = Σ_{h,w} attn_k(h,w) · feat(h,w)
        # feat_map: (B, C, fH, fW) → (B, 1, C, fH*fW)
        # attn:     (B, 17, fH, fW) → (B, 17, 1, fH*fW)
        feat_flat = feat_map.view(B, C, -1).unsqueeze(1)   # (B, 1, C, fH*fW)
        attn_flat = attn.view(B, 17, -1).unsqueeze(2)      # (B, 17, 1, fH*fW)

        kp_means = (feat_flat * attn_flat).sum(dim=3)  # (B, 17, C)

        # Variance: σ²_k = Σ_{h,w} attn_k(h,w) · (feat(h,w) - μ_k)²
        residuals = feat_flat - kp_means.unsqueeze(3)  # (B, 17, C, fH*fW)
        kp_vars = (attn_flat * residuals.pow(2)).sum(dim=3)  # (B, 17, C)

        return kp_means, kp_vars, kp_scores

    def _ttsfr_recover(self, kp_feats, kp_scores, labels):
        """Training-Time Skeleton Feature Recovery.

        For each sample, find same-ID samples in batch and use their
        visible keypoint features to fill in occluded keypoints.

        Args:
            kp_feats: (B, 17, C) keypoint features
            kp_scores: (B, 17) confidence scores
            labels: (B,) identity labels

        Returns:
            recovered_feats: (B, 17, C) with occluded kp replaced
        """
        B = kp_feats.shape[0]
        thr = self.ttsfr_threshold
        recovered = kp_feats.clone()

        for i in range(B):
            # Find same-ID samples (exclude self)
            same_id = (labels == labels[i]).nonzero(as_tuple=True)[0]
            same_id = same_id[same_id != i]
            if len(same_id) == 0:
                continue

            for kp in range(17):
                if kp_scores[i, kp] >= thr:
                    continue  # Visible, no recovery needed

                # Find candidates where this kp is visible
                cand_vis = kp_scores[same_id, kp]
                visible = cand_vis >= thr
                if not visible.any():
                    continue

                # Average visible candidates' features (detached to avoid cross-sample grad)
                vis_indices = same_id[visible]
                vis_feats = kp_feats[vis_indices, kp].detach()
                vis_weights = kp_scores[vis_indices, kp]
                w = vis_weights / vis_weights.sum()
                recovered[i, kp] = (vis_feats * w.unsqueeze(1)).sum(dim=0)

        return recovered

    def _compute_kp_weights(self, pose_dict):
        """Compute keypoint weights based on kp_weight_mode.

        Returns:
            weights: (B, 17) per-keypoint weights for pooling
        """
        scores = pose_dict['scores'][:, 0, :]  # (B, 17) person 0

        if self.kp_weight_mode == 'score':
            return scores
        elif self.kp_weight_mode == 'visibility':
            return pose_dict['visibility'][:, 0, :]  # (B, 17)
        elif self.kp_weight_mode == 'score_visibility':
            vis = pose_dict['visibility'][:, 0, :]  # (B, 17)
            return scores * vis
        elif self.kp_weight_mode == 'binary_visibility':
            return pose_dict['visibility_binary'][:, 0, :]  # (B, 17)
        else:
            raise ValueError(
                f"Unknown kp_weight_mode: {self.kp_weight_mode}")

    def forward(self, feat_map, pose_dict, return_cls=True, label=None,
                stage2_feat=None):
        """
        Args:
            feat_map: (B, C, fH, fW) Part Stage 3 feature map
            pose_dict: dict with keypoints, scores, person_mask
            return_cls: whether to return classification scores
            label: identity labels (unused, kept for interface compat)
            stage2_feat: (B, C2, fH2, fW2) Stage 2 feature map for MRKF (optional)

        Returns:
            If training:
                cls_scores: list of 1 tensor [(B, num_classes)]
                feats: list of 1 tensor [(B, feat_dim)]
                None (part_valid placeholder)
            If testing:
                feats: list of 1 tensor [(B, feat_dim)]
        """
        keypoints = pose_dict['keypoints']
        scores = pose_dict['scores']
        person_mask = pose_dict['person_mask']

        # 1. Extract keypoint features — DPF or point sampling
        kp_vars = None  # DPF variance (None when using point sampling)
        if self.dpf and 'heatmaps' in pose_dict:
            # DPF: heatmap-weighted spatial pooling → distributional features
            kp_feats, kp_vars, kp_scores = self._heatmap_pool_features(
                feat_map, pose_dict['heatmaps'], scores, person_mask)
        else:
            # Standard: bilinear sampling at keypoint locations
            kp_feats, kp_scores = self._sample_keypoint_features(
                feat_map, keypoints, scores, person_mask)
        # kp_feats: (B, 17, C), kp_scores: (B, 17)
        # kp_vars: (B, 17, C) or None

        # 1.1. MRKF: fuse with Stage 2 keypoint features for higher spatial resolution
        if self.mrkf and stage2_feat is not None:
            # Sample keypoint features from Stage 2 (higher resolution, lower dim)
            kp_s2, _ = self._sample_keypoint_features(
                stage2_feat, keypoints, scores, person_mask)  # (B, 17, C2)
            # Project both to common dimension and fuse
            s2_proj = self.mrkf_s2_proj(kp_s2)       # (B, 17, 256)
            s3_proj = self.mrkf_s3_proj(kp_feats)     # (B, 17, 256)
            fused = torch.cat([s2_proj, s3_proj], dim=-1)  # (B, 17, 512)
            # Residual: kp_feats + fusion(concat) — starts as identity via zero-init
            kp_feats = kp_feats + self.mrkf_fusion(fused)   # (B, 17, C)

        # 1.5. TTSFR: Training-Time Skeleton Feature Recovery
        # Use same-ID samples in batch to fill occluded keypoints
        if self.training and self.ttsfr and label is not None:
            kp_feats = self._ttsfr_recover(kp_feats, kp_scores, label)

        # 1.6. SGMT: Skeleton-Guided Masked Training
        # Training: random mask → GCN recovers via message passing → ID loss
        # Testing: mask low-confidence kp → GCN recovers → then SGCFR
        if self.sgmt:
            if self.training:
                B_sz, N_kp, _ = kp_feats.shape
                num_mask = max(1, int(N_kp * self.sgmt_ratio))
                mask = torch.zeros(B_sz, N_kp, dtype=torch.bool,
                                   device=kp_feats.device)
                for b in range(B_sz):
                    idx = torch.randperm(N_kp, device=kp_feats.device)[:num_mask]
                    mask[b, idx] = True
                mt = self.mask_token.expand(B_sz, N_kp, -1)
                kp_feats = torch.where(mask.unsqueeze(-1), mt, kp_feats)
            else:
                # Test: mask keypoints with low confidence
                low_conf = kp_scores < self.sgmt_threshold
                if low_conf.any():
                    B_sz = kp_feats.shape[0]
                    mt = self.mask_token.expand(B_sz, self.num_joints, -1)
                    kp_feats = torch.where(low_conf.unsqueeze(-1), mt, kp_feats)

        # 2. Optional SGMKC masking (training only)
        sgmkc_mask = None
        kp_feats_original = None
        if self.training and self.sgmkc and self.sgmkc_ratio > 0:
            B_sz, N_kp, C_dim = kp_feats.shape
            # Save original features for reconstruction target
            kp_feats_original = kp_feats.detach().clone()
            # Generate random mask: True = keep, False = mask out
            sgmkc_mask = torch.rand(B_sz, N_kp, device=kp_feats.device) > self.sgmkc_ratio
            # Zero out masked keypoint features
            kp_feats = kp_feats * sgmkc_mask.unsqueeze(-1).float()

        # 2.5. SCFR: Support-Complete Feature Replacement
        scfr_stats = None
        if self.training and hasattr(self, '_scfr_bank') and self._scfr_bank is not None \
                and getattr(self, '_scfr_active', False) and label is not None:
            kp_feats, _, scfr_stats = self._scfr_bank.replace(
                kp_feats, kp_scores, label)

        # 3. Optional skeleton GCN (propagate along skeleton edges)
        if self.use_gcn:
            kp_feats_enhanced = self.gcn(kp_feats)  # (B, 17, C)
        else:
            kp_feats_enhanced = kp_feats

        # 4. Keypoint weighting for pooling
        # DPF precision weighting: use inverse variance as weights
        if self.dpf and kp_vars is not None:
            # Scalar precision per keypoint = 1 / mean_variance
            mean_var = kp_vars.mean(dim=2)  # (B, 17)
            kp_weights = 1.0 / (mean_var + 1e-3)  # (B, 17) — precision
            # Clip to prevent single keypoint domination
            kp_weights = kp_weights.clamp(max=1e3)
            # Also incorporate confidence scores (zero-confidence → zero weight)
            kp_weights = kp_weights * kp_scores.clamp(min=0)
        else:
            kp_weights = self._compute_kp_weights(pose_dict)  # (B, 17)

        # Learnable Keypoint Attention: modulate weights with learned attention
        if self.kp_learnable_attn:
            attn = self.kp_attention(kp_weights)  # (B, 17) → sigmoid → (B, 17)
            kp_weights = kp_weights * attn  # element-wise: confidence × attention

        # 4.5 Learned Keypoint Uncertainty: modulate weights with learned uncertainty
        kp_unc = None
        if self.kp_uncertainty:
            kp_unc = self.uncertainty_head(kp_feats_enhanced).squeeze(-1)  # (B, 17)
            # reliability = 1 - uncertainty
            reliability = (1.0 - kp_unc).clamp(min=0.01)
            kp_weights = kp_weights * reliability

        weights = kp_weights.clamp(min=1e-6).unsqueeze(-1)  # (B, 17, 1)
        skeleton_feat = (kp_feats_enhanced * weights).sum(dim=1) / \
                        weights.sum(dim=1).clamp(min=1e-6)  # (B, C)

        # PKE: compute per-keypoint log_sigma and pooled sigma
        skeleton_sigma = None
        if self.pke:
            log_sigma = self.sigma_head(kp_feats_enhanced)  # (B, 17, C)
            sigma = torch.exp(log_sigma.clamp(min=-5.0, max=5.0))  # (B, 17, C), clamp for stability
            # Inverse-variance weighted pooling for sigma
            skeleton_sigma = (sigma * weights).sum(dim=1) / \
                            weights.sum(dim=1).clamp(min=1e-6)  # (B, C)

        aux_data = {
            'kp_feats': kp_feats_enhanced,  # (B, 17, C)
            'kp_weights': kp_weights,       # (B, 17)
        }
        # DPF: export per-keypoint variance for probabilistic matching at test time
        if kp_vars is not None:
            aux_data['kp_vars'] = kp_vars  # (B, 17, C)
        if skeleton_sigma is not None:
            aux_data['sigma'] = skeleton_sigma  # (B, C)
        if kp_unc is not None:
            aux_data['kp_uncertainty'] = kp_unc  # (B, 17)
        # SGMKC: pass mask and original features for reconstruction loss
        if sgmkc_mask is not None:
            aux_data['sgmkc_mask'] = sgmkc_mask              # (B, 17) True=kept
            aux_data['sgmkc_original'] = kp_feats_original   # (B, 17, C)
        # SCFR: export replacement statistics
        if scfr_stats is not None:
            aux_data['scfr_stats'] = scfr_stats

        # PKE: for test, use sigma-weighted mu (precision-weighted feature)
        if self.pke and skeleton_sigma is not None and not self.training:
            # Precision weighting: mu / sigma → high-sigma dims contribute less to distance
            skeleton_feat_out = skeleton_feat / skeleton_sigma.clamp(min=0.01)
        else:
            skeleton_feat_out = skeleton_feat

        if return_cls:
            # BN + Classifier (uses mu only, not sigma)
            feat_bn = self.bn(skeleton_feat)
            aux_data['feat_bn'] = feat_bn
            cls_score = self.classifier(feat_bn)
            return [cls_score], [skeleton_feat], aux_data
        else:
            return None, [skeleton_feat_out], aux_data
