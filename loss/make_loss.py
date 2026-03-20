# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss, euclidean_dist, normalize, hard_example_mining
from .center_loss import CenterLoss


def _compute_paml_triplet(kp_feats, kp_weights, labels, margin_loss,
                          margin=None):
    """Pose-Aware Metric Learning: triplet loss with per-keypoint pairwise
    distance aggregated by confidence-weighted average.

    Instead of computing distance on the aggregated skeleton feature,
    compute per-keypoint L2 distances and aggregate using min(confidence_i,
    confidence_j) weights — matching CVK test-time distance logic.

    Args:
        kp_feats: (B, K, C) per-keypoint features from GCN
        kp_weights: (B, K) confidence scores
        labels: (B,) identity labels
        margin_loss: MarginRankingLoss or SoftMarginLoss instance
        margin: margin value (None for soft margin)

    Returns:
        loss: scalar triplet loss
    """
    B, K, C = kp_feats.shape

    # Per-keypoint pairwise L2 distance: list of K (B, B) matrices
    kp_dists = []
    for k in range(K):
        kp_k = kp_feats[:, k, :]  # (B, C)
        dist_k = euclidean_dist(kp_k, kp_k)  # (B, B)
        kp_dists.append(dist_k)
    kp_dist_stack = torch.stack(kp_dists, dim=-1)  # (B, B, K)

    # Confidence-weighted aggregation: weight = min(score_i, score_j)
    w = kp_weights.clamp(min=1e-6)  # (B, K)
    min_w = torch.minimum(w.unsqueeze(1), w.unsqueeze(0))  # (B, B, K)

    # Weighted average distance
    dist_mat = (kp_dist_stack * min_w).sum(dim=-1) / \
               min_w.sum(dim=-1).clamp(min=1e-6)  # (B, B)

    # Standard hard example mining + ranking loss
    dist_ap, dist_an = hard_example_mining(dist_mat, labels)
    y = dist_an.new_ones(dist_an.size())
    if margin is not None:
        loss = margin_loss(dist_an, dist_ap, y)
    else:
        loss = margin_loss(dist_an - dist_ap, y)

    return loss


def _compute_csrd_loss(global_feat, kp_feats, kp_weights, labels, tau=0.10,
                       teacher_kp_feats=None, anchor_weights=None):
    """Distill batch-wise CVK-style pair relations into the global embedding.

    Teacher:
        per-keypoint same-index distances aggregated with CVK-style confidence
        weights, detached from the graph branch.
    Student:
        pairwise distances in the normalized global embedding space.
    """
    feat_s = normalize(global_feat, axis=-1)
    dist_s = euclidean_dist(feat_s, feat_s)

    if teacher_kp_feats is None:
        teacher_kp_feats = kp_feats
    kp_f = F.normalize(teacher_kp_feats.detach(), dim=-1)
    per_kp_dist = [euclidean_dist(kp_f[:, k, :], kp_f[:, k, :])
                   for k in range(kp_f.size(1))]
    dist_k = torch.stack(per_kp_dist, dim=-1)  # (B, B, K)

    w = kp_weights.detach().clamp(min=0.0)
    pair_w = torch.sqrt(w.unsqueeze(1) * w.unsqueeze(0))  # (B, B, K)
    weight_sum = pair_w.sum(dim=-1)
    dist_t = (dist_k * pair_w).sum(dim=-1) / weight_sum.clamp(min=1e-6)
    dist_t = torch.where(weight_sum > 0, dist_t, dist_s.detach())

    B = dist_s.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=labels.device)
    same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    pos_mask = same_label & ~eye
    neg_mask = ~same_label

    if anchor_weights is not None:
        anchor_weights = anchor_weights.detach().to(dist_s.dtype).clamp(min=0.0)

    losses = []
    loss_weights = []
    for idx in range(B):
        if anchor_weights is None:
            anchor_w = dist_s.new_ones(())
        else:
            anchor_w = anchor_weights[idx]
            if anchor_w.item() <= 0:
                continue
        if pos_mask[idx].any():
            s_logp = F.log_softmax((-dist_s[idx, pos_mask[idx]]) / tau, dim=0)
            t_prob = F.softmax((-dist_t[idx, pos_mask[idx]].detach()) / tau, dim=0)
            losses.append(F.kl_div(s_logp, t_prob, reduction='batchmean'))
            loss_weights.append(anchor_w)
        if neg_mask[idx].any():
            s_logp = F.log_softmax((-dist_s[idx, neg_mask[idx]]) / tau, dim=0)
            t_prob = F.softmax((-dist_t[idx, neg_mask[idx]].detach()) / tau, dim=0)
            losses.append(F.kl_div(s_logp, t_prob, reduction='batchmean'))
            loss_weights.append(anchor_w)

    if losses:
        loss_stack = torch.stack(losses)
        weight_stack = torch.stack(loss_weights)
        loss = (loss_stack * weight_stack).sum() / weight_stack.sum().clamp(min=1e-12)
    else:
        loss = dist_s.new_zeros(())

    pos_teacher = dist_t[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_teacher = dist_t[neg_mask].mean().item() if neg_mask.any() else 0.0
    pos_student = dist_s[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_student = dist_s[neg_mask].mean().item() if neg_mask.any() else 0.0
    stats = {
        'teacher_gap': float(neg_teacher - pos_teacher),
        'student_gap': float(neg_student - pos_student),
        'valid_ratio': float((weight_sum > 0).float().mean().item()),
        'active_anchor_ratio': float((anchor_weights > 0).float().mean().item()) if anchor_weights is not None else 1.0,
        'mean_anchor_weight': float(anchor_weights.mean().item()) if anchor_weights is not None else 1.0,
    }
    return loss, stats


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
                csrd_loss = None
                csgt_weight = getattr(cfg.MODEL, 'POSE_CSGT_WEIGHT', 1.0)
                csrd_weight = getattr(cfg.MODEL, 'POSE_CSRD_WEIGHT', 0.5)
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
                    if getattr(cfg.MODEL, 'POSE_CSRD', False) and kp_data is not None:
                        csrd_warmup = getattr(cfg.MODEL, 'POSE_CSRD_WARMUP', 20)
                        epoch_now = int(kp_data.get('epoch', 0))
                        if epoch_now > csrd_warmup:
                            csrd_tau = getattr(cfg.MODEL, 'POSE_CSRD_TAU', 0.10)
                            csrd_loss, csrd_stats = _compute_csrd_loss(
                                feat[0], kp_data['kp_feats'], kp_data['kp_weights'],
                                target, tau=csrd_tau,
                                teacher_kp_feats=kp_data.get('csrd_teacher_feats'),
                                anchor_weights=kp_data.get('csrd_anchor_weights'))
                            loss_details['csrd'] = csrd_loss.item()
                            loss_details['csrd_tgap'] = csrd_stats['teacher_gap']
                            loss_details['csrd_sgap'] = csrd_stats['student_gap']
                            loss_details['csrd_vr'] = csrd_stats['valid_ratio']
                            loss_details['csrd_ar'] = csrd_stats['active_anchor_ratio']
                            loss_details['csrd_aw'] = csrd_stats['mean_anchor_weight']
                    # PAML: use per-keypoint pairwise distance for part triplet
                    paml_enabled = getattr(cfg.MODEL, 'POSE_PAML', False)
                    if paml_enabled and kp_data is not None:
                        part_tri_avg = _compute_paml_triplet(
                            kp_data['kp_feats'], kp_data['kp_weights'],
                            target, triplet.ranking_loss,
                            margin=triplet.margin)
                    else:
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
                if csrd_loss is not None:
                    total = total + csrd_weight * csrd_loss

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

                # Keypoint Dissimilar Loss (KDL) — prevent GCN feature collapse
                if getattr(cfg.MODEL, 'POSE_KP_DISSIMILAR', False) and kp_data is not None and 'kp_feats' in kp_data:
                    kdl_w = getattr(cfg.MODEL, 'POSE_KP_DISSIMILAR_WEIGHT', 0.1)
                    kp_f = kp_data['kp_feats']  # (B, 17, C)
                    kp_f_norm = F.normalize(kp_f, dim=-1)
                    # Pairwise cosine similarity matrix (B, 17, 17)
                    cos_sim = torch.bmm(kp_f_norm, kp_f_norm.transpose(1, 2))
                    # Mean of upper triangle (excluding diagonal) = average cross-kp similarity
                    mask = torch.triu(torch.ones(17, 17, device=cos_sim.device), diagonal=1).bool()
                    kdl_loss = cos_sim[:, mask].mean()  # minimize cross-kp similarity
                    total = total + kdl_w * kdl_loss
                    loss_details['kdl'] = kdl_loss.item()

                # PKE: add sigma regularization to prevent sigma collapse to zero
                if getattr(cfg.MODEL, 'POSE_PKE', False) and kp_data is not None and 'sigma' in kp_data:
                    sigma = kp_data['sigma']  # (B, C)
                    # Regularize: penalize too-small sigma (prevent collapse to deterministic)
                    # log(sigma).mean() → negative when sigma < 1
                    pke_reg = -sigma.log().clamp(min=-5).mean() * 0.01
                    total = total + pke_reg
                    loss_details['pke'] = sigma.mean().item()

                # Keypoint Uncertainty Regularization — prevent collapse to all-uncertain
                if getattr(cfg.MODEL, 'POSE_KP_UNCERTAINTY', False) and kp_data is not None and 'kp_uncertainty' in kp_data:
                    unc_reg_w = getattr(cfg.MODEL, 'POSE_KP_UNCERTAINTY_REG', 0.1)
                    kp_unc = kp_data['kp_uncertainty']  # (B, 17) in [0, 1]
                    # Penalize high mean uncertainty to prevent collapse
                    unc_reg = kp_unc.mean()
                    total = total + unc_reg_w * unc_reg
                    loss_details['unc'] = kp_unc.mean().item()

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
