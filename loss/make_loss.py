# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss, euclidean_dist, normalize
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

    def _compute_common_support_matrix(kp_weights):
        weights = kp_weights.detach().clamp(min=0)
        min_sum = torch.minimum(
            weights.unsqueeze(1), weights.unsqueeze(0)).sum(dim=-1)
        max_sum = torch.maximum(
            weights.unsqueeze(1), weights.unsqueeze(0)).sum(dim=-1)
        return min_sum / max_sum.clamp(min=1e-6)

    def _support_aware_hard_mining(dist_mat, labels, overlap_mat,
                                   min_overlap=0.3, mode='both'):
        assert dist_mat.size(0) == dist_mat.size(1)
        num_samples = dist_mat.size(0)

        label_mat = labels.expand(num_samples, num_samples)
        is_pos = label_mat.eq(label_mat.t())
        is_neg = label_mat.ne(label_mat.t())
        eye = torch.eye(num_samples, dtype=torch.bool, device=labels.device)
        is_pos = is_pos & ~eye

        dist_ap = []
        dist_an = []
        pos_overlaps = []
        neg_overlaps = []
        pos_fallback = 0
        neg_fallback = 0

        for idx in range(num_samples):
            pos_mask = is_pos[idx]
            neg_mask = is_neg[idx]

            if mode in ('pos', 'both'):
                pos_valid = pos_mask & (overlap_mat[idx] >= min_overlap)
                if not pos_valid.any():
                    pos_valid = pos_mask
                    pos_fallback += 1
            else:
                pos_valid = pos_mask

            if mode in ('neg', 'both'):
                neg_valid = neg_mask & (overlap_mat[idx] >= min_overlap)
                if not neg_valid.any():
                    neg_valid = neg_mask
                    neg_fallback += 1
            else:
                neg_valid = neg_mask

            pos_inds = torch.where(pos_valid)[0]
            neg_inds = torch.where(neg_valid)[0]

            if pos_inds.numel() == 0:
                pos_inds = torch.where(pos_mask)[0]
            if neg_inds.numel() == 0:
                neg_inds = torch.where(neg_mask)[0]

            pos_dists = dist_mat[idx, pos_inds]
            neg_dists = dist_mat[idx, neg_inds]

            pos_idx = pos_inds[torch.argmax(pos_dists)]
            neg_idx = neg_inds[torch.argmin(neg_dists)]

            dist_ap.append(dist_mat[idx, pos_idx])
            dist_an.append(dist_mat[idx, neg_idx])
            pos_overlaps.append(overlap_mat[idx, pos_idx])
            neg_overlaps.append(overlap_mat[idx, neg_idx])

        stats = {
            'pos_overlap': torch.stack(pos_overlaps).mean().item(),
            'neg_overlap': torch.stack(neg_overlaps).mean().item(),
            'pos_fallback': float(pos_fallback),
            'neg_fallback': float(neg_fallback),
        }
        return torch.stack(dist_ap), torch.stack(dist_an), stats

    def _compute_csgt_loss(global_feat, labels, kp_weights,
                           normalize_feature=False, pose_sim=None):
        feat_input = normalize(global_feat, axis=-1) if normalize_feature else global_feat
        dist_mat = euclidean_dist(feat_input, feat_input)

        if pose_sim is not None and triplet.pose_alpha > 0:
            dist_mat = dist_mat * (1 - triplet.pose_alpha * pose_sim)

        overlap_mat = _compute_common_support_matrix(kp_weights)
        mine_mode = getattr(cfg.MODEL, 'POSE_CSGT_MINE_MODE', 'both')
        min_overlap = getattr(cfg.MODEL, 'POSE_CSGT_MIN_OVERLAP', 0.3)
        dist_ap, dist_an, stats = _support_aware_hard_mining(
            dist_mat, labels, overlap_mat, min_overlap=min_overlap, mode=mine_mode)

        y = dist_an.new_ones(dist_an.size())
        if triplet.margin is not None:
            loss = triplet.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = triplet.ranking_loss(dist_an - dist_ap, y)
        return loss, stats

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target, target_cam, pose_sim=None):
            return F.cross_entropy(score, target)

    #  elif cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'id_triplet', 'img_triplet']:
    elif 'triplet' in sampler:
        def loss_func(score, feat, target, target_cam, pose_sim=None, kp_data=None):
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

                csgt_loss = None
                csgt_weight = getattr(cfg.MODEL, 'POSE_CSGT_WEIGHT', 1.0)
                if isinstance(feat, list):
                    pt = getattr(cfg.MODEL, 'POSE_PART_TRI_WEIGHT', 1.0)
                    wt_p = pt / (1.0 + pt)
                    wt_g = 1.0 / (1.0 + pt)
                    # PCRA: pass pose_sim only to global triplet
                    global_tri_base = triplet(feat[0], target, pose_sim=pose_sim)[0]
                    if getattr(cfg.MODEL, 'POSE_CSGT', False) and kp_data is not None:
                        csgt_loss, csgt_stats = _compute_csgt_loss(
                            feat[0], target, kp_data['kp_weights'],
                            normalize_feature=trp_norm, pose_sim=pose_sim)
                        loss_details['tri_csgt'] = csgt_loss.item()
                        loss_details['csgt_pos_overlap'] = csgt_stats['pos_overlap']
                        loss_details['csgt_neg_overlap'] = csgt_stats['neg_overlap']
                        loss_details['csgt_pos_fallback'] = csgt_stats['pos_fallback']
                        loss_details['csgt_neg_fallback'] = csgt_stats['neg_fallback']
                    part_tris = [triplet(f, target)[0] for f in feat[1:]]
                    part_tri_avg = sum(part_tris) / len(part_tris)
                    TRI_LOSS = wt_g * global_tri_base + wt_p * part_tri_avg
                    loss_details['tri_global'] = global_tri_base.item()
                    loss_details['tri_part'] = part_tri_avg.item()
                else:
                    global_loss_scale = getattr(cfg.MODEL, 'GLOBAL_LOSS_SCALE', 1.0)
                    TRI_LOSS = global_loss_scale * triplet(feat, target, normalize_feature=trp_norm, pose_sim=pose_sim)[0]
                    loss_details['tri_global'] = TRI_LOSS.item()

                total = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                        cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                if csgt_loss is not None:
                    total = total + csgt_weight * csgt_loss

                # Per-keypoint triplet loss (confidence-weighted)
                if kp_data is not None and 'weight' in kp_data:
                    kp_feats = kp_data['kp_feats']      # (B, 17, C)
                    kp_weights = kp_data['kp_weights']  # (B, 17)
                    kp_tri_w = kp_data['weight']
                    num_kp = kp_feats.size(1)
                    kp_tri_losses = []
                    for k in range(num_kp):
                        kp_feat_k = kp_feats[:, k, :]  # (B, C)
                        kp_tri_k = triplet(kp_feat_k, target)[0]
                        kp_tri_losses.append(kp_tri_k)
                    # Confidence-weighted average across keypoints
                    avg_conf = kp_weights.mean(dim=0)  # (17,)
                    avg_conf = avg_conf / avg_conf.sum().clamp(min=1e-6)
                    kp_tri_loss = sum(l * w for l, w in zip(kp_tri_losses, avg_conf))
                    total = total + kp_tri_w * kp_tri_loss
                    loss_details['tri_kp'] = kp_tri_loss.item()

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
