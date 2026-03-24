# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss, euclidean_dist, normalize, hard_example_mining
from .center_loss import CenterLoss


def _compute_maxsim_triplet(kp_feats, kp_weights, labels, margin=0.3, tau=0.05):
    """Set-to-Set Metric Learning via Soft-MaxSim Triplet.

    Replaces pooled-vector triplet with ColBERT-style late interaction triplet.
    Each query keypoint soft-aligns to the best gallery keypoint via temperature softmax.

    Args:
        kp_feats: (B, K, D) keypoint features from GCN branch
        kp_weights: (B, K) keypoint confidence scores
        labels: (B,) identity labels
        margin: triplet margin
        tau: softmax temperature (lower = sharper, closer to hard max)
    Returns:
        loss: scalar triplet loss
        stats: dict with diagnostic metrics
    """
    B, K, D = kp_feats.shape

    # L2 normalize each keypoint feature
    kp_norm = F.normalize(kp_feats, p=2, dim=2)  # (B, K, D)

    # Pairwise cosine similarity: (B, B, K, K)
    cos_all = torch.einsum('ikh,jlh->ijkl', kp_norm, kp_norm)

    # Soft attention: for each (i,j) pair, each query kp k attends over gallery kps l
    attn = F.softmax(cos_all / tau, dim=3)  # (B, B, K, K)

    # Attention-weighted similarity per query keypoint: (B, B, K)
    per_kp_sim = (attn * cos_all).sum(dim=3)

    # Confidence-weighted aggregation: (B, B)
    w = kp_weights.clamp(min=0.0)  # (B, K)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
    sim_matrix = torch.einsum('ijk,ik->ij', per_kp_sim, w) / w_sum  # (B, B)

    # Distance = 1 - similarity: (B, B)
    dist_matrix = 1.0 - sim_matrix

    # Hard mining: batch hard
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)

    # Hardest positive: max distance among same-ID (exclude self)
    pos_mask = label_eq.clone()
    pos_mask.fill_diagonal_(False)
    dist_ap = dist_matrix.clone()
    dist_ap[~pos_mask] = -1e9
    hardest_pos_dist, _ = dist_ap.max(dim=1)  # (B,)

    # Hardest negative: min distance among different-ID
    neg_mask = ~label_eq
    dist_an = dist_matrix.clone()
    dist_an[~neg_mask] = 1e9
    hardest_neg_dist, _ = dist_an.min(dim=1)  # (B,)

    # Triplet loss
    if margin is not None:
        trip_loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
    else:
        trip_loss = F.softplus(hardest_pos_dist - hardest_neg_dist)

    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        loss = (kp_feats * 0.0).sum()
        stats = {'d_ap': 0, 'd_an': 0, 'margin_gap': 0}
        return loss, stats

    loss = trip_loss[valid].mean()

    with torch.no_grad():
        d_ap_mean = hardest_pos_dist[valid].mean().item()
        d_an_mean = hardest_neg_dist[valid].mean().item()
        attn_entropy = -(attn * (attn + 1e-8).log()).sum(dim=3).mean().item()

    stats = {
        'd_ap': d_ap_mean,
        'd_an': d_an_mean,
        'margin_gap': d_an_mean - d_ap_mean,
        'attn_ent': attn_entropy,
    }
    return loss, stats


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target, target_cam, pose_sim=None):
            return F.cross_entropy(score, target)

    elif 'triplet' in sampler:
        def loss_func(score, feat, target, target_cam, pose_sim=None, kp_data=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                ce_fn = xent if cfg.MODEL.IF_LABELSMOOTH == 'on' else F.cross_entropy
                trp_norm = cfg.SOLVER.TRP_L2 if hasattr(cfg.SOLVER, 'TRP_L2') else False

                loss_details = {}

                if isinstance(score, list):
                    # Configurable global/part loss ratio via POSE_PART_WEIGHT
                    pw = getattr(cfg.MODEL, 'POSE_PART_WEIGHT', 1.0)
                    w_p = pw / (1.0 + pw)
                    w_g = 1.0 / (1.0 + pw)
                    global_id = ce_fn(score[0], target)

                    # Evidential DL: replace GCN branch CE with Dirichlet-based loss
                    evid_enabled = getattr(cfg.MODEL, 'POSE_EVIDENTIAL', False)
                    if evid_enabled and kp_data is not None:
                        from loss.evidential_loss import evidential_loss
                        evid_kl = float(getattr(cfg.MODEL, 'POSE_EVIDENTIAL_KL_REG', 0.1))
                        evid_anneal = float(getattr(cfg.MODEL, 'POSE_EVIDENTIAL_ANNEAL', 0.6))
                        evid_epoch = int(kp_data.get('epoch', 0))
                        total_ep = cfg.SOLVER.MAX_EPOCHS
                        num_cls = score[0].size(1)
                        # Apply evidential loss to each part's logits
                        part_evid_losses = []
                        all_evid_stats = None
                        for s in score[1:]:
                            el, es = evidential_loss(
                                s, target, num_cls, evid_epoch, total_ep,
                                kl_reg=evid_kl, anneal_ratio=evid_anneal)
                            part_evid_losses.append(el)
                            if all_evid_stats is None:
                                all_evid_stats = es
                        part_id_avg = sum(part_evid_losses) / len(part_evid_losses)
                        loss_details['evid_br'] = all_evid_stats['bayes_risk']
                        loss_details['evid_kl'] = all_evid_stats['kl']
                        loss_details['evid_unc'] = all_evid_stats['uncertainty']
                        loss_details['evid_ev'] = all_evid_stats['evidence']
                        loss_details['evid_ann'] = all_evid_stats['anneal']
                    else:
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
                    global_tri_base = triplet(feat[0], target)[0]

                    # MaxSim triplet: set-to-set metric learning
                    maxsim_tri_enabled = getattr(cfg.MODEL, 'POSE_MAXSIM_TRIPLET', False)
                    maxsim_tri_tau = float(getattr(cfg.MODEL, 'POSE_MAXSIM_TRIPLET_TEMP', 0.05))
                    maxsim_tri_additive = getattr(cfg.MODEL, 'POSE_MAXSIM_TRIPLET_ADDITIVE', False)
                    maxsim_tri_weight = float(getattr(cfg.MODEL, 'POSE_MAXSIM_TRIPLET_WEIGHT', 0.25))
                    if maxsim_tri_enabled and kp_data is not None:
                        maxsim_loss, maxsim_stats = _compute_maxsim_triplet(
                            kp_data['kp_feats'], kp_data['kp_weights'],
                            target, margin=triplet.margin, tau=maxsim_tri_tau)
                        loss_details['tri_maxsim'] = maxsim_loss.item()
                        loss_details['maxsim_d_ap'] = maxsim_stats['d_ap']
                        loss_details['maxsim_d_an'] = maxsim_stats['d_an']
                        loss_details['maxsim_margin'] = maxsim_stats['margin_gap']
                        loss_details['maxsim_ent'] = maxsim_stats['attn_ent']
                        if maxsim_tri_additive:
                            # Additive: keep pooled triplet + add MaxSim as auxiliary
                            part_tris = [triplet(f, target)[0] for f in feat[1:]]
                            part_tri_avg = sum(part_tris) / len(part_tris) + maxsim_tri_weight * maxsim_loss
                        else:
                            # Replace: MaxSim triplet replaces pooled part triplet
                            part_tri_avg = maxsim_loss
                    else:
                        # Normalize per-token features before triplet if many parts (per-token mode)
                        use_norm = len(feat) > 3  # >3 parts = per-token mode
                        part_tris = [triplet(f, target, normalize_feature=use_norm)[0] for f in feat[1:]]
                        part_tri_avg = sum(part_tris) / len(part_tris)
                    TRI_LOSS = wt_g * global_tri_base + wt_p * part_tri_avg
                    loss_details['tri_global'] = global_tri_base.item()
                    loss_details['tri_part'] = part_tri_avg.item()
                else:
                    global_loss_scale = getattr(cfg.MODEL, 'GLOBAL_LOSS_SCALE', 1.0)
                    TRI_LOSS = global_loss_scale * triplet(feat, target, normalize_feature=trp_norm)[0]
                    loss_details['tri_global'] = TRI_LOSS.item()

                total = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                        cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                loss_details['total'] = total.item()
                total._loss_details = loss_details
                return total
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
