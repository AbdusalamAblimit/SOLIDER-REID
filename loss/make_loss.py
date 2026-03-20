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
                       teacher_kp_feats=None, anchor_weights=None,
                       pair_weight_mode='none', pair_weight_alpha=1.0,
                       pair_top_ratio=0.25, target_mode='full',
                       queue_data=None):
    """Distill batch-wise CVK-style pair relations into the global embedding.

    Teacher:
        per-keypoint same-index distances aggregated with CVK-style confidence
        weights, detached from the graph branch.
    Student:
        pairwise distances in the normalized global embedding space.
    """
    feat_s = normalize(global_feat, axis=-1)
    dist_s = euclidean_dist(feat_s, feat_s)

    w = kp_weights.detach().clamp(min=0.0)
    pair_w = torch.sqrt(w.unsqueeze(1) * w.unsqueeze(0))  # (B, B, K)
    weight_sum = pair_w.sum(dim=-1)

    def _aggregate_teacher_dist(src_feats):
        kp_f = F.normalize(src_feats.detach(), dim=-1)
        per_kp_dist = [euclidean_dist(kp_f[:, k, :], kp_f[:, k, :])
                       for k in range(kp_f.size(1))]
        dist_k = torch.stack(per_kp_dist, dim=-1)  # (B, B, K)
        dist = (dist_k * pair_w).sum(dim=-1) / weight_sum.clamp(min=1e-6)
        return torch.where(weight_sum > 0, dist, dist_s.detach())

    dist_base = _aggregate_teacher_dist(kp_feats)
    if teacher_kp_feats is None:
        dist_t = dist_base
    else:
        dist_t = _aggregate_teacher_dist(teacher_kp_feats)

    pair_delta = None
    if pair_weight_mode in ('delta', 'delta_top', 'delta_top_exact') and teacher_kp_feats is not None:
        pair_delta = (dist_t - dist_base).abs().detach()

    queue_size = 0
    queue_ratio_means = []
    dist_s_q = None
    dist_base_q = None
    dist_t_q = None
    pos_mask_q = None
    neg_mask_q = None
    pair_delta_q = None
    if queue_data is not None and queue_data.get('labels') is not None:
        queue_labels = queue_data['labels']
        if queue_labels.numel() > 0:
            queue_size = int(queue_labels.numel())
            feat_q = normalize(queue_data['student_feat'], axis=-1)
            dist_s_q = euclidean_dist(feat_s, feat_q)

            w_q = queue_data['kp_weights'].detach().clamp(min=0.0)
            pair_w_q = torch.sqrt(w.unsqueeze(1) * w_q.unsqueeze(0))  # (B, Q, K)
            weight_sum_q = pair_w_q.sum(dim=-1)

            def _aggregate_teacher_cross(src_feats, ref_feats):
                kp_f_src = F.normalize(src_feats.detach(), dim=-1)
                kp_f_ref = F.normalize(ref_feats.detach(), dim=-1)
                per_kp_dist = [euclidean_dist(kp_f_src[:, k, :], kp_f_ref[:, k, :])
                               for k in range(kp_f_src.size(1))]
                dist_k = torch.stack(per_kp_dist, dim=-1)  # (B, Q, K)
                dist = (dist_k * pair_w_q).sum(dim=-1) / weight_sum_q.clamp(min=1e-6)
                return torch.where(weight_sum_q > 0, dist, dist_s_q.detach())

            dist_base_q = _aggregate_teacher_cross(kp_feats, queue_data['kp_feats'])
            if teacher_kp_feats is not None and queue_data.get('teacher_kp_feats') is not None:
                dist_t_q = _aggregate_teacher_cross(
                    teacher_kp_feats, queue_data['teacher_kp_feats'])
                if pair_weight_mode in ('delta', 'delta_top', 'delta_top_exact'):
                    pair_delta_q = (dist_t_q - dist_base_q).abs().detach()
            else:
                dist_t_q = dist_base_q

            pos_mask_q = labels.unsqueeze(1).eq(queue_labels.unsqueeze(0))
            neg_mask_q = ~pos_mask_q

    def _focus_from_delta(delta_vec):
        if delta_vec is None:
            return None, None, None
        scale = delta_vec.max().clamp(min=1e-6)
        focus = 1.0 + pair_weight_alpha * (delta_vec / scale)
        if pair_weight_mode in ('delta_top', 'delta_top_exact'):
            keep_num = max(1, int(math.ceil(delta_vec.numel() * pair_top_ratio)))
            top_vals, top_idx = torch.topk(delta_vec, k=keep_num, largest=True, sorted=False)
            sparse_focus = torch.full_like(focus, 1e-6)
            if pair_weight_mode == 'delta_top':
                keep_mask = delta_vec >= top_vals.min()
                sparse_focus[keep_mask] = focus[keep_mask]
                return sparse_focus, sparse_focus[keep_mask].mean(), keep_mask.float().mean()
            sparse_focus.scatter_(0, top_idx, focus[top_idx])
            keep_ratio = delta_vec.new_tensor(keep_num / max(1, delta_vec.numel()))
            return sparse_focus, focus[top_idx].mean(), keep_ratio
        return focus, focus.mean(), focus.new_ones(())

    def _distill_subset(student_dist, teacher_dist, base_dist, focus=None):
        if target_mode == 'residual':
            base_det = base_dist.detach()
            teacher_res = teacher_dist.detach() - base_det
            student_res = student_dist - base_det
            scale = teacher_res.abs().max().clamp(min=1e-6)
            point_loss = F.smooth_l1_loss(
                student_res / scale, teacher_res / scale, reduction='none')
            if focus is not None:
                focus_w = focus.detach().clamp(min=0.0)
                return (point_loss * focus_w).sum() / focus_w.sum().clamp(min=1e-6)
            return point_loss.mean()
        if target_mode == 'residual_kl':
            base_det = base_dist.detach()
            s_logits = (-(student_dist - base_det)) / tau
            t_logits = (-(teacher_dist.detach() - base_det)) / tau
            if focus is not None:
                log_focus = focus.clamp(min=1e-6).log()
                s_logits = s_logits + log_focus
                t_logits = t_logits + log_focus
            s_logp = F.log_softmax(s_logits, dim=0)
            t_prob = F.softmax(t_logits, dim=0)
            return F.kl_div(s_logp, t_prob, reduction='batchmean')
        s_logits = (-student_dist) / tau
        t_logits = (-teacher_dist.detach()) / tau
        if focus is not None:
            log_focus = focus.clamp(min=1e-6).log()
            s_logits = s_logits + log_focus
            t_logits = t_logits + log_focus
        s_logp = F.log_softmax(s_logits, dim=0)
        t_prob = F.softmax(t_logits, dim=0)
        return F.kl_div(s_logp, t_prob, reduction='batchmean')

    B = dist_s.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=labels.device)
    same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    pos_mask = same_label & ~eye
    neg_mask = ~same_label

    if anchor_weights is not None:
        anchor_weights = anchor_weights.detach().to(dist_s.dtype).clamp(min=0.0)

    losses = []
    loss_weights = []
    pair_delta_means = []
    pair_focus_means = []
    pair_select_ratio_means = []
    for idx in range(B):
        if anchor_weights is None:
            anchor_w = dist_s.new_ones(())
        else:
            anchor_w = anchor_weights[idx]
            if anchor_w.item() <= 0:
                continue
        pos_parts_s = []
        pos_parts_t = []
        pos_parts_b = []
        pos_parts_d = []
        pos_queue_count = 0
        pos_total_count = 0
        if pos_mask[idx].any():
            pos_parts_s.append(dist_s[idx, pos_mask[idx]])
            pos_parts_t.append(dist_t[idx, pos_mask[idx]])
            pos_parts_b.append(dist_base[idx, pos_mask[idx]])
            pos_total_count += int(pos_mask[idx].sum().item())
            if pair_delta is not None:
                pos_parts_d.append(pair_delta[idx, pos_mask[idx]])
        if pos_mask_q is not None and pos_mask_q[idx].any():
            pos_parts_s.append(dist_s_q[idx, pos_mask_q[idx]])
            pos_parts_t.append(dist_t_q[idx, pos_mask_q[idx]])
            pos_parts_b.append(dist_base_q[idx, pos_mask_q[idx]])
            pos_queue_count += int(pos_mask_q[idx].sum().item())
            pos_total_count += int(pos_mask_q[idx].sum().item())
            if pair_delta_q is not None:
                pos_parts_d.append(pair_delta_q[idx, pos_mask_q[idx]])
        if pos_parts_s:
            pos_focus = None
            if pos_parts_d:
                pos_delta = torch.cat(pos_parts_d, dim=0)
                pos_focus, pos_focus_mean, pos_select_ratio = _focus_from_delta(pos_delta)
                pair_delta_means.append(pos_delta.mean())
                if pos_focus_mean is not None:
                    pair_focus_means.append(pos_focus_mean)
                if pos_select_ratio is not None:
                    pair_select_ratio_means.append(pos_select_ratio)
            losses.append(_distill_subset(
                torch.cat(pos_parts_s, dim=0),
                torch.cat(pos_parts_t, dim=0),
                torch.cat(pos_parts_b, dim=0),
                focus=pos_focus))
            loss_weights.append(anchor_w)
            if pos_total_count > 0:
                queue_ratio_means.append(
                    dist_s.new_tensor(pos_queue_count / pos_total_count))

        neg_parts_s = []
        neg_parts_t = []
        neg_parts_b = []
        neg_parts_d = []
        neg_queue_count = 0
        neg_total_count = 0
        if neg_mask[idx].any():
            neg_parts_s.append(dist_s[idx, neg_mask[idx]])
            neg_parts_t.append(dist_t[idx, neg_mask[idx]])
            neg_parts_b.append(dist_base[idx, neg_mask[idx]])
            neg_total_count += int(neg_mask[idx].sum().item())
            if pair_delta is not None:
                neg_parts_d.append(pair_delta[idx, neg_mask[idx]])
        if neg_mask_q is not None and neg_mask_q[idx].any():
            neg_parts_s.append(dist_s_q[idx, neg_mask_q[idx]])
            neg_parts_t.append(dist_t_q[idx, neg_mask_q[idx]])
            neg_parts_b.append(dist_base_q[idx, neg_mask_q[idx]])
            neg_queue_count += int(neg_mask_q[idx].sum().item())
            neg_total_count += int(neg_mask_q[idx].sum().item())
            if pair_delta_q is not None:
                neg_parts_d.append(pair_delta_q[idx, neg_mask_q[idx]])
        if neg_parts_s:
            neg_focus = None
            if neg_parts_d:
                neg_delta = torch.cat(neg_parts_d, dim=0)
                neg_focus, neg_focus_mean, neg_select_ratio = _focus_from_delta(neg_delta)
                pair_delta_means.append(neg_delta.mean())
                if neg_focus_mean is not None:
                    pair_focus_means.append(neg_focus_mean)
                if neg_select_ratio is not None:
                    pair_select_ratio_means.append(neg_select_ratio)
            losses.append(_distill_subset(
                torch.cat(neg_parts_s, dim=0),
                torch.cat(neg_parts_t, dim=0),
                torch.cat(neg_parts_b, dim=0),
                focus=neg_focus))
            loss_weights.append(anchor_w)
            if neg_total_count > 0:
                queue_ratio_means.append(
                    dist_s.new_tensor(neg_queue_count / neg_total_count))

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
    pair_mask = ~eye
    teacher_residual = (dist_t - dist_base).abs()
    student_residual = (dist_s.detach() - dist_base).abs()
    stats = {
        'teacher_gap': float(neg_teacher - pos_teacher),
        'student_gap': float(neg_student - pos_student),
        'valid_ratio': float((weight_sum > 0).float().mean().item()),
        'active_anchor_ratio': float((anchor_weights > 0).float().mean().item()) if anchor_weights is not None else 1.0,
        'mean_anchor_weight': float(anchor_weights.mean().item()) if anchor_weights is not None else 1.0,
        'pair_delta': float(torch.stack(pair_delta_means).mean().item()) if pair_delta_means else 0.0,
        'pair_focus': float(torch.stack(pair_focus_means).mean().item()) if pair_focus_means else 1.0,
        'pair_select_ratio': float(torch.stack(pair_select_ratio_means).mean().item()) if pair_select_ratio_means else 1.0,
        'queue_size': float(queue_size),
        'queue_ratio': float(torch.stack(queue_ratio_means).mean().item()) if queue_ratio_means else 0.0,
        'teacher_residual': float(teacher_residual[pair_mask].mean().item()),
        'student_residual': float(student_residual[pair_mask].mean().item()),
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

    csrd_pair_weight_mode = getattr(cfg.MODEL, 'POSE_CSRD_PAIR_WEIGHT_MODE', 'none')
    if csrd_pair_weight_mode not in ('none', 'delta', 'delta_top', 'delta_top_exact'):
        raise ValueError(f"Unsupported POSE_CSRD_PAIR_WEIGHT_MODE: {csrd_pair_weight_mode}")
    csrd_target_mode = getattr(cfg.MODEL, 'POSE_CSRD_TARGET_MODE', 'full')
    if csrd_target_mode not in ('full', 'residual', 'residual_kl'):
        raise ValueError(f"Unsupported POSE_CSRD_TARGET_MODE: {csrd_target_mode}")
    csrd_pair_weight_alpha = getattr(cfg.MODEL, 'POSE_CSRD_PAIR_WEIGHT_ALPHA', 1.0)
    csrd_pair_top_ratio = getattr(cfg.MODEL, 'POSE_CSRD_PAIR_TOP_RATIO', 0.25)

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
                                anchor_weights=kp_data.get('csrd_anchor_weights'),
                                pair_weight_mode=csrd_pair_weight_mode,
                                pair_weight_alpha=csrd_pair_weight_alpha,
                                pair_top_ratio=csrd_pair_top_ratio,
                                target_mode=csrd_target_mode,
                                queue_data=kp_data.get('csrd_queue'))
                            loss_details['csrd'] = csrd_loss.item()
                            loss_details['csrd_tgap'] = csrd_stats['teacher_gap']
                            loss_details['csrd_sgap'] = csrd_stats['student_gap']
                            loss_details['csrd_vr'] = csrd_stats['valid_ratio']
                            loss_details['csrd_ar'] = csrd_stats['active_anchor_ratio']
                            loss_details['csrd_aw'] = csrd_stats['mean_anchor_weight']
                            loss_details['csrd_tr'] = csrd_stats['teacher_residual']
                            loss_details['csrd_gr'] = csrd_stats['student_residual']
                            if csrd_pair_weight_mode != 'none':
                                loss_details['csrd_pd'] = csrd_stats['pair_delta']
                                loss_details['csrd_pf'] = csrd_stats['pair_focus']
                                loss_details['csrd_psr'] = csrd_stats['pair_select_ratio']
                            if csrd_stats['queue_size'] > 0:
                                loss_details['csrd_qn'] = csrd_stats['queue_size']
                                loss_details['csrd_qr'] = csrd_stats['queue_ratio']
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
