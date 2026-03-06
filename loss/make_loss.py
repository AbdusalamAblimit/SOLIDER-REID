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

    # Check special loss strategies
    vpreid_cfg = getattr(cfg.MODEL, 'VPREID', None)
    is_vpreid = vpreid_cfg is not None and getattr(vpreid_cfg, 'ENABLE', False)

    pose_part_cfg = getattr(cfg.MODEL, 'POSE_PART', None)
    is_pose_part = pose_part_cfg is not None and getattr(pose_part_cfg, 'ENABLE', False)

    pcfc_cfg = getattr(cfg.MODEL, 'PCFC', None)
    is_pcfc = pcfc_cfg is not None and getattr(pcfc_cfg, 'ENABLE', False)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target, target_cam):
            return F.cross_entropy(score, target)

    elif 'triplet' in sampler and is_pcfc:
        # PCFC loss: global ID + Triplet + optional part ID + optional part Triplet + optional Push
        pcfc_part_id_w = getattr(pcfc_cfg, 'PART_ID_WEIGHT', 1.0)
        pcfc_part_tri_w = getattr(pcfc_cfg, 'PART_TRIPLET_WEIGHT', 0.0)
        pcfc_push_w = getattr(pcfc_cfg, 'PUSH_WEIGHT', 0.0)
        pcfc_vis_thr = getattr(pcfc_cfg, 'VIS_THRESHOLD', 0.3)
        pcfc_use_part = getattr(pcfc_cfg, 'USE_PART_LOSS', True)
        ce = xent if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy

        # Part triplet loss (GiLt-style)
        part_tri_fn = None
        if pcfc_part_tri_w > 0:
            part_tri_fn = PartAveragedTripletLoss(margin=None, normalize=True)

        # Push loss (part diversity)
        push_fn = None
        if pcfc_push_w > 0:
            push_fn = PushLoss()

        def loss_func(score, feat, target, target_cam, extras=None):
            ID_LOSS = ce(score, target)
            TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]
            total = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            part_id_loss = torch.tensor(0.0, device=target.device)
            part_tri_loss = torch.tensor(0.0, device=target.device)
            push_loss_val = torch.tensor(0.0, device=target.device)
            n_valid = 0
            if pcfc_use_part and extras and 'part_logits' in extras:
                part_logits = extras['part_logits']
                part_vis = extras['part_vis']
                for k, logits_k in enumerate(part_logits):
                    vis_k = part_vis[:, k]
                    mask = vis_k > pcfc_vis_thr
                    if mask.sum() > 0:
                        part_id_loss = part_id_loss + ce(logits_k[mask], target[mask])
                        n_valid += 1
                if n_valid > 0:
                    part_id_loss = part_id_loss / n_valid
                    total = total + pcfc_part_id_w * part_id_loss

            # Part triplet loss (GiLt-style)
            if part_tri_fn is not None and extras and 'part_feats' in extras:
                part_feats = extras['part_feats']  # [B, K, C]
                part_vis = extras['part_vis']
                part_tri_loss = part_tri_fn(part_feats, target, part_vis)
                total = total + pcfc_part_tri_w * part_tri_loss

            # Push loss (part diversity)
            if push_fn is not None and extras and 'part_feats' in extras:
                push_loss_val = push_fn(extras['part_feats'])
                total = total + pcfc_push_w * push_loss_val

            alpha_val = extras.get('attn_alpha', 0.0) if extras else 0.0
            components = {
                'id': ID_LOSS.item(),
                'tri': TRI_LOSS.item(),
                'pid': part_id_loss.item() if n_valid > 0 else 0.0,
                'ptri': part_tri_loss.item(),
                'push': push_loss_val.item(),
                'n_vis': n_valid,
                'alpha': alpha_val,
            }
            # Include PVFM beta values and KPE scale if present
            if extras:
                for k, v in extras.items():
                    if k.startswith('beta_') or k == 'kpe_scale':
                        components[k] = v
            loss_func.last_components = components
            return total

        loss_func.last_components = None
        return loss_func, center_criterion

    elif 'triplet' in sampler and is_pose_part:
        # PosePart loss: global ID + Triplet + visibility-weighted per-part ID
        pp_part_id_w = getattr(pose_part_cfg, 'PART_ID_WEIGHT', 0.5)
        pp_vis_thr = getattr(pose_part_cfg, 'VIS_THRESHOLD', 0.3)
        ce = xent if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy

        def loss_func(score, feat, target, target_cam, extras=None):
            # Global ID loss
            ID_LOSS = ce(score, target)
            # Global triplet loss
            TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

            total = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            # Visibility-weighted per-part ID loss
            if extras and 'part_logits' in extras:
                part_logits = extras['part_logits']  # list of [B, C]
                part_vis = extras['part_vis']  # [B, K]

                part_id_loss = torch.tensor(0.0, device=target.device)
                n_valid = 0
                for k, logits_k in enumerate(part_logits):
                    vis_k = part_vis[:, k]  # [B]
                    mask = vis_k > pp_vis_thr  # [B] bool

                    if mask.sum() > 0:
                        part_id_loss = part_id_loss + ce(logits_k[mask], target[mask])
                        n_valid += 1

                if n_valid > 0:
                    part_id_loss = part_id_loss / n_valid
                    total = total + pp_part_id_w * part_id_loss

                loss_func.last_components = {
                    'id': ID_LOSS.item(),
                    'tri': TRI_LOSS.item(),
                    'pid': part_id_loss.item() if n_valid > 0 else 0.0,
                    'n_vis': n_valid,
                }
            else:
                loss_func.last_components = {
                    'id': ID_LOSS.item(),
                    'tri': TRI_LOSS.item(),
                    'pid': 0.0,
                    'n_vis': 0,
                }

            return total

        loss_func.last_components = None
        return loss_func, center_criterion

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

            # Store loss components for external logging (processor.py)
            loss_func.last_components = {
                'id': id_loss.item(),
                'pid': part_id_loss.item(),
                'tri': tri_loss.item(),
                'push': push_loss.item(),
            }

            return total

        loss_func.last_components = None

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
