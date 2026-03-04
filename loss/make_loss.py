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

    # Check if VPReID loss strategy is needed
    vpreid_cfg = getattr(cfg.MODEL, 'VPREID', None)
    is_vpreid = vpreid_cfg is not None and getattr(vpreid_cfg, 'ENABLE', False)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target,target_cam):
            return F.cross_entropy(score, target)

    #  elif cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'id_triplet', 'img_triplet']:
    elif 'triplet' in sampler and is_vpreid:
        # VPReID loss: global ID + fg ID + per-part ID + part triplet + push
        vp_id_w = getattr(vpreid_cfg, 'ID_WEIGHT', 1.0)
        vp_tri_w = getattr(vpreid_cfg, 'TRI_WEIGHT', 1.0)
        vp_part_id_w = getattr(vpreid_cfg, 'PART_ID_WEIGHT', 0.5)
        vp_push_w = getattr(vpreid_cfg, 'PUSH_WEIGHT', 0.1)
        n_parts = getattr(vpreid_cfg, 'N_PARTS', 5)

        vp_margin = None if cfg.MODEL.NO_MARGIN else cfg.SOLVER.MARGIN
        part_tri_fn = PartAveragedTripletLoss(margin=vp_margin, normalize=True)
        push_fn = PushLoss()
        ce = xent if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy

        import logging
        _vp_logger = logging.getLogger("transreid.vpreid_loss")
        _vp_step = [0]

        def loss_func(score, feat, target, target_cam, extras=None):
            # ID loss: global + foreground
            id_loss = ce(score[0], target) + ce(score[1], target)

            # Per-part ID loss: score[2:2+K]
            part_id_loss = torch.tensor(0.0, device=target.device)
            part_scores = score[2:2 + n_parts]
            if part_scores:
                part_id_loss = sum(ce(s, target) for s in part_scores) / len(part_scores)

            # Part-averaged triplet loss
            part_feats_bn = torch.stack(feat[2:2 + n_parts], dim=1)  # [B, K, D]
            part_vis = extras['part_vis'] if extras else None
            tri_loss = part_tri_fn(part_feats_bn, target, part_vis)

            total = vp_id_w * id_loss + vp_part_id_w * part_id_loss + vp_tri_w * tri_loss

            # Push diversity loss
            push_loss = push_fn(part_feats_bn)
            total = total + vp_push_w * push_loss

            # Diagnostic logging
            _vp_step[0] += 1
            if _vp_step[0] % 20 == 1 or total.item() > 20.0:
                _vp_logger.info(
                    f"[LOSS] id={id_loss.item():.2f} pid={part_id_loss.item():.2f} "
                    f"tri={tri_loss.item():.2f} push={push_loss.item():.2f} "
                    f"total={total.item():.2f}"
                )

            return total

        return loss_func, center_criterion

    elif 'triplet' in sampler:
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
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


