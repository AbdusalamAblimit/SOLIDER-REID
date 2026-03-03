# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .part_loss import PartAveragedTripletLoss, PushLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    def _get_pose_branch_weights():
        default_weight = 0.5
        transformer_type = getattr(cfg.MODEL, 'TRANSFORMER_TYPE', '')
        if transformer_type and 'pose_swin' in transformer_type:
            local_weight = getattr(cfg.MODEL.POSE, 'LOCAL_LOSS_WEIGHT', default_weight)
        else:
            local_weight = default_weight
        try:
            local_weight = float(local_weight)
        except (TypeError, ValueError):
            local_weight = default_weight
        local_weight = max(0.0, min(1.0, local_weight))
        return 1.0 - local_weight, local_weight

    def _combine_pose_branch_losses(global_loss, local_losses):
        if not local_losses:
            return global_loss
        local_losses = [loss for loss in local_losses if loss is not None]
        if not local_losses:
            return global_loss
        local_loss = sum(local_losses) / len(local_losses)
        global_weight, local_weight = _get_pose_branch_weights()
        return global_weight * global_loss + local_weight * local_loss

    loss_strategy = getattr(cfg.MODEL.POSE, 'LOSS_STRATEGY', 'unified')
    # SPTrans may override loss strategy
    sptrans_cfg = getattr(cfg.MODEL, 'SPTRANS', None)
    if sptrans_cfg is not None:
        sptrans_strategy = getattr(sptrans_cfg, 'LOSS_STRATEGY', None)
        if sptrans_strategy:
            loss_strategy = sptrans_strategy
    # PAMS overrides loss strategy
    pams_cfg = getattr(cfg.MODEL, 'PAMS', None)
    if pams_cfg is not None and getattr(pams_cfg, 'ENABLE', False):
        loss_strategy = getattr(pams_cfg, 'LOSS_STRATEGY', 'pams')

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target,target_cam):
            return F.cross_entropy(score, target)

    #  elif cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'id_triplet', 'img_triplet']:
    elif 'triplet' in sampler:
        # PAMS loss strategy
        if loss_strategy == 'pams':
            pams_id_w = getattr(pams_cfg, 'ID_WEIGHT', 1.0) if pams_cfg else 1.0
            pams_tri_w = getattr(pams_cfg, 'TRI_WEIGHT', 1.0) if pams_cfg else 1.0
            pams_bpa_w = getattr(pams_cfg, 'BPA_WEIGHT', 1.0) if pams_cfg else 1.0
            pams_push_w = getattr(pams_cfg, 'PUSH_WEIGHT', 0.1) if pams_cfg else 0.1
            part_tri_loss_fn = PartAveragedTripletLoss(margin=cfg.SOLVER.MARGIN)
            push_loss_fn = PushLoss()
            ce = xent if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy

            def loss_func(score, feat, target, target_cam, extras=None):
                # ID loss: global + foreground
                id_loss = ce(score[0], target) + ce(score[1], target)

                # Part-averaged triplet loss
                # feat[2:] are BN-normalized part features; stack them
                part_feats_bn = torch.stack(feat[2:], dim=1)  # [B, K, D]
                part_vis = extras['part_vis'] if extras else None
                tri_loss = part_tri_loss_fn(part_feats_bn, target, part_vis)

                total = pams_id_w * id_loss + pams_tri_w * tri_loss

                # BPA supervision (only when available, i.e. training with pose)
                if extras and 'bpa_logits' in extras:
                    bpa_loss = F.cross_entropy(extras['bpa_logits'].float(), extras['bpa_targets'])
                    total = total + pams_bpa_w * bpa_loss

                # Push diversity loss
                push_loss = push_loss_fn(part_feats_bn)
                total = total + pams_push_w * push_loss

                return total

            return loss_func, center_criterion

        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                # Part Expert strategy: global + local ID+Triplet, plus per-part ID loss
                if loss_strategy == 'part_expert' and isinstance(score, list) and isinstance(feat, list):
                    ce = xent if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy
                    n_parts = getattr(cfg.MODEL.SPTRANS, 'N_PARTS', 5)
                    part_loss_w = getattr(cfg.MODEL.SPTRANS, 'PART_LOSS_WEIGHT', 0.2)
                    # Global: ID + Triplet
                    ID_G = ce(score[0], target)
                    TRI_G = triplet(feat[0], target)[0]
                    # Local (visibility-weighted avg): ID + Triplet
                    ID_L = ce(score[1], target)
                    TRI_L = triplet(feat[1], target)[0]
                    # Per-part: ID loss (score[2:2+K])
                    part_scores = score[2:2 + n_parts]
                    if part_scores:
                        PART_ID = sum(ce(s, target) for s in part_scores) / len(part_scores)
                    else:
                        PART_ID = torch.tensor(0.0, device=target.device)
                    global_w, local_w = _get_pose_branch_weights()
                    return cfg.MODEL.ID_LOSS_WEIGHT * (global_w * ID_G + local_w * ID_L + part_loss_w * PART_ID) + \
                           cfg.MODEL.TRIPLET_LOSS_WEIGHT * (global_w * TRI_G + local_w * TRI_L)

                # Split strategy: global=ID+Triplet, local=Triplet only
                if loss_strategy == 'split' and isinstance(score, list) and isinstance(feat, list):
                    ce = xent if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy
                    ID_LOSS = ce(score[0], target)
                    TRI_LOSS_G = triplet(feat[0], target)[0]
                    # L2 normalize local feat before triplet to prevent
                    # distance explosion with high-dim part features (e.g. 3840-d)
                    local_f = F.normalize(feat[1], p=2, dim=1) if len(feat) > 1 else feat[0]
                    TRI_LOSS_L = triplet(local_f, target)[0]
                    global_w, local_w = _get_pose_branch_weights()
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                           cfg.MODEL.TRIPLET_LOSS_WEIGHT * (global_w * TRI_LOSS_G + local_w * TRI_LOSS_L)

                # GiLt strategy: global=ID only, local=Triplet only
                if loss_strategy == 'gilt' and isinstance(score, list) and isinstance(feat, list):
                    ce = xent if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy
                    ID_LOSS = ce(score[0], target)
                    TRI_LOSS = triplet(feat[1], target)[0] if len(feat) > 1 else triplet(feat[0], target)[0]
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        local_id_losses = [xent(scor, target) for scor in score[1:]]
                        global_id_loss = xent(score[0], target)
                        ID_LOSS = _combine_pose_branch_losses(global_id_loss, local_id_losses)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            local_tri_losses = [triplet(feats, target)[0] for feats in feat[1:]]
                            global_tri_loss = triplet(feat[0], target)[0]
                            TRI_LOSS = _combine_pose_branch_losses(global_tri_loss, local_tri_losses)
                    else:
                            TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        local_id_losses = [F.cross_entropy(scor, target) for scor in score[1:]]
                        global_id_loss = F.cross_entropy(score[0], target)
                        ID_LOSS = _combine_pose_branch_losses(global_id_loss, local_id_losses)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            local_tri_losses = [triplet(feats, target)[0] for feats in feat[1:]]
                            global_tri_loss = triplet(feat[0], target)[0]
                            TRI_LOSS = _combine_pose_branch_losses(global_tri_loss, local_tri_losses)
                    else:
                            TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


