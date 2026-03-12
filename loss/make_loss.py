# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    pose_alpha = getattr(cfg.MODEL, 'POSE_PCRA_ALPHA', 0.0)
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss(pose_alpha=pose_alpha)
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN, pose_alpha=pose_alpha)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
        if pose_alpha > 0:
            print(f"[PCRA] Pose-Contrastive Representation Alignment enabled: alpha={pose_alpha}")
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target, target_cam, pose_sim=None):
            return F.cross_entropy(score, target)

    #  elif cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'id_triplet', 'img_triplet']:
    elif 'triplet' in sampler:
        def loss_func(score, feat, target, target_cam, pose_sim=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                ce_fn = xent if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy
                trp_norm = cfg.SOLVER.TRP_L2 if hasattr(cfg.SOLVER, 'TRP_L2') else False

                loss_details = {}

                if isinstance(score, list):
                    # Configurable global/part loss ratio via POSE_PART_WEIGHT
                    pw = getattr(cfg.MODEL, 'POSE_PART_WEIGHT', 1.0)
                    w_p = pw / (1.0 + pw)  # default: 0.5
                    w_g = 1.0 / (1.0 + pw)  # default: 0.5
                    global_id = ce_fn(score[0], target)
                    part_ids = [ce_fn(s, target) for s in score[1:]]
                    part_id_avg = sum(part_ids) / len(part_ids)
                    ID_LOSS = w_g * global_id + w_p * part_id_avg
                    loss_details['id_global'] = global_id.item()
                    loss_details['id_part'] = part_id_avg.item()
                else:
                    global_loss_scale = getattr(cfg.MODEL, 'GLOBAL_LOSS_SCALE', 1.0)
                    ID_LOSS = global_loss_scale * ce_fn(score, target)
                    loss_details['id_global'] = ID_LOSS.item()

                if isinstance(feat, list):
                    pt = getattr(cfg.MODEL, 'POSE_PART_TRI_WEIGHT', 1.0)
                    wt_p = pt / (1.0 + pt)
                    wt_g = 1.0 / (1.0 + pt)
                    # PCRA: pass pose_sim only to global triplet
                    global_tri = triplet(feat[0], target, pose_sim=pose_sim)[0]
                    part_tris = [triplet(f, target)[0] for f in feat[1:]]
                    part_tri_avg = sum(part_tris) / len(part_tris)
                    TRI_LOSS = wt_g * global_tri + wt_p * part_tri_avg
                    loss_details['tri_global'] = global_tri.item()
                    loss_details['tri_part'] = part_tri_avg.item()
                else:
                    global_loss_scale = getattr(cfg.MODEL, 'GLOBAL_LOSS_SCALE', 1.0)
                    TRI_LOSS = global_loss_scale * triplet(feat, target, normalize_feature=trp_norm, pose_sim=pose_sim)[0]
                    loss_details['tri_global'] = TRI_LOSS.item()

                total = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                        cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                loss_details['total'] = total.item()

                # Store details on the loss tensor for the processor to read
                total._loss_details = loss_details
                return total
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


