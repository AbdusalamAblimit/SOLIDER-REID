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
                 input_size=(384, 128), use_gcn=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.input_h, self.input_w = input_size
        self.num_joints = 17
        self.use_gcn = use_gcn

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

        # BN + Classifier for skeleton feature
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

    def forward(self, feat_map, pose_dict, return_cls=True, label=None):
        """
        Args:
            feat_map: (B, C, fH, fW) Part Stage 3 feature map
            pose_dict: dict with keypoints, scores, person_mask
            return_cls: whether to return classification scores
            label: identity labels (unused, kept for interface compat)

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

        # 1. Sample features at keypoint locations
        kp_feats, kp_scores = self._sample_keypoint_features(
            feat_map, keypoints, scores, person_mask)
        # kp_feats: (B, 17, C), kp_scores: (B, 17)

        # 2. Optional skeleton GCN
        if self.use_gcn:
            kp_feats_enhanced = self.gcn(kp_feats)  # (B, 17, C)
        else:
            kp_feats_enhanced = kp_feats

        # 3. Confidence-weighted average
        weights = kp_scores.clamp(min=1e-6).unsqueeze(-1)  # (B, 17, 1)
        skeleton_feat = (kp_feats_enhanced * weights).sum(dim=1) / \
                        weights.sum(dim=1).clamp(min=1e-6)  # (B, C)

        if return_cls:
            # BN + Classifier
            feat_bn = self.bn(skeleton_feat)
            cls_score = self.classifier(feat_bn)
            return [cls_score], [skeleton_feat], None
        else:
            return None, [skeleton_feat], None
