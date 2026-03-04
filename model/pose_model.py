"""
Pose-Guided Model Wrapper

Wraps the standard build_transformer with PosePartPool to add
visibility-aware part features. The global branch remains unchanged.

Architecture:
  Swin-Tiny → feature_map (B, H*W, 768)
                ├─ global avg pool → global_feat (768)
                └─ PosePartPool(kpts, vis) → part_feats (B, 5, 768)

Training outputs: [cls_global, cls_part0, ..., cls_part4], [global_feat, part0, ..., part4]
Eval outputs: {'global': feat_bn, 'parts': part_feats_bn, 'part_vis': vis}

Memory overhead: ~5 BNNeck + 5 classifiers ≈ 0.03GB
"""

import torch
import torch.nn as nn
from model.modules.pose_part_pool import PosePartPool


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class PoseGuidedModel(nn.Module):
    """Standard Swin-Tiny + PosePartPool for visibility-aware part features.

    The backbone is the same as build_transformer (single branch, no JPM).
    PosePartPool adds per-part features with minimal overhead.
    """

    def __init__(self, base_model, num_classes, cfg):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.neck_feat = cfg.TEST.NECK_FEAT

        # Get feature dimension from base model
        feat_dim = base_model.in_planes  # 768 for Swin-Tiny

        # Determine feature map spatial size from input size and stride
        input_h, input_w = cfg.INPUT.SIZE_TRAIN
        stride_h, stride_w = cfg.MODEL.STRIDE_SIZE
        self.feat_h = input_h // stride_h
        self.feat_w = input_w // stride_w

        n_parts = getattr(cfg.MODEL.POSE, 'N_KPTS_PARTS', 5)
        self.n_parts = n_parts

        # PosePartPool module
        self.pose_pool = PosePartPool(
            feat_dim=feat_dim,
            feat_h=self.feat_h,
            feat_w=self.feat_w,
            n_parts=n_parts,
            sigma=2.0,
            min_vis=0.3,
        )

        # Per-part BNNeck + classifier
        self.part_bnnecks = nn.ModuleList()
        self.part_classifiers = nn.ModuleList()
        for _ in range(n_parts):
            bn = nn.BatchNorm1d(feat_dim)
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)
            self.part_bnnecks.append(bn)
            cls = nn.Linear(feat_dim, num_classes, bias=False)
            cls.apply(weights_init_classifier)
            self.part_classifiers.append(cls)

        # Weight for part loss relative to global loss
        self.part_loss_weight = getattr(cfg.MODEL.POSE, 'LOCAL_LOSS_WEIGHT', 0.5)

    def forward(self, x, label=None, cam_label=None, view_label=None,
                kpts=None, vis=None):
        """
        Args:
            x: (B, 3, H, W) images
            label, cam_label, view_label: standard ReID labels
            kpts: (B, 17, 3) keypoints [norm_x, norm_y, confidence]
            vis: (B, 17) visibility from VisPredictHead

        Returns:
            Training: (scores_list, feats_list, extras_dict)
            Eval: (feat_dict, None)
        """
        # Run base model backbone directly to get feature map
        base = self.base_model.base
        feat_map_raw = base(x)  # tuple: (feat, None) or just feat

        if isinstance(feat_map_raw, tuple):
            feat_map = feat_map_raw[0]  # (B, H*W, D) for Swin
        else:
            feat_map = feat_map_raw

        # Global feature: standard avg pool
        B = feat_map.shape[0]
        if feat_map.dim() == 3:
            global_feat = feat_map.mean(dim=1)  # (B, D)
        else:
            global_feat = feat_map.mean(dim=(-2, -1))  # (B, D)

        # Global BN + classifier (from base model)
        global_bn = self.base_model.bottleneck(global_feat)

        if kpts is not None and vis is not None:
            # Part features via PosePartPool
            part_feats, part_vis = self.pose_pool(feat_map, kpts, vis)  # (B, K, D), (B, K)
        else:
            # Fallback: uniform horizontal stripe pooling
            D = global_feat.shape[1]
            if feat_map.dim() == 3:
                fm = feat_map.transpose(1, 2).reshape(B, D, self.feat_h, self.feat_w)
            else:
                fm = feat_map
            stripe_h = self.feat_h // self.n_parts
            parts = []
            for k in range(self.n_parts):
                start = k * stripe_h
                end = start + stripe_h if k < self.n_parts - 1 else self.feat_h
                parts.append(fm[:, :, start:end, :].mean(dim=(-2, -1)))
            part_feats = torch.stack(parts, dim=1)  # (B, K, D)
            part_vis = torch.ones(B, self.n_parts, device=feat_map.device)

        if self.training:
            # Global classification
            cls_global = self.base_model.classifier(global_bn)
            scores = [cls_global]
            feats = [global_feat]

            # Per-part classification
            for k in range(self.n_parts):
                pk_bn = self.part_bnnecks[k](part_feats[:, k])
                scores.append(self.part_classifiers[k](pk_bn))
                feats.append(part_feats[:, k])

            extras = {'part_vis': part_vis}
            return scores, feats, extras
        else:
            # Eval: return features
            part_bn_list = []
            for k in range(self.n_parts):
                part_bn_list.append(self.part_bnnecks[k](part_feats[:, k]))
            part_bn = torch.stack(part_bn_list, dim=1)

            if self.neck_feat == 'after':
                feat_dict = {
                    'global': global_bn,
                    'parts': part_bn,
                    'part_vis': part_vis,
                }
            else:
                feat_dict = {
                    'global': global_feat,
                    'parts': part_feats,
                    'part_vis': part_vis,
                }
            return feat_dict, None

    def load_param(self, trained_path):
        self.base_model.load_param(trained_path)

    def update_freeze_schedule(self, epoch):
        if hasattr(self.base_model, 'update_freeze_schedule'):
            self.base_model.update_freeze_schedule(epoch)
