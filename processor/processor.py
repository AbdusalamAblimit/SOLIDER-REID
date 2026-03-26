import math
import logging
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from model.modules.support_complete_bank import SupportCompleteBank
from model.modules.pair_adaptive_fusion import (
    build_pair_descriptors,
    build_query_competition_descriptors,
    build_query_context_descriptors,
    common_support_distance,
    euclidean_distance_tensor,
)


def _pose_to_device(pose_dict, device):
    """Move all tensors in pose_dict to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in pose_dict.items()}


def _flatten_eval_like_feat(feat):
    """Build a test-like feature for auxiliary losses/logging.

    For list features, mimic equal-concat using normalized branch features.
    """
    if isinstance(feat, (list, tuple)):
        parts = [F.normalize(f, dim=1) for f in feat]
        return torch.cat(parts, dim=1)
    return F.normalize(feat, dim=1)


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    use_pose = cfg.MODEL.POSE_ENABLED

    # LTCS / LPCS support banks
    ltcs_enabled = getattr(cfg.MODEL, 'POSE_LTCS', False)
    lpcs_enabled = getattr(cfg.MODEL, 'POSE_LPCS', False)
    if ltcs_enabled and lpcs_enabled:
        raise ValueError('POSE_LTCS and POSE_LPCS cannot be enabled together')

    ltcs_teacher_bank = None
    if ltcs_enabled:
        ltcs_weight = getattr(cfg.MODEL, 'POSE_LTCS_WEIGHT', 0.5)
        ltcs_warmup = getattr(cfg.MODEL, 'POSE_LTCS_WARMUP', 20)
        ltcs_st_low_thr = getattr(cfg.MODEL, 'POSE_LTCS_ST_LOW_THR', 0.3)
        ltcs_st_update_thr = getattr(cfg.MODEL, 'POSE_LTCS_ST_UPDATE_THR', 0.7)
        ltcs_st_mom = getattr(cfg.MODEL, 'POSE_LTCS_ST_MOM', 0.9)
        ltcs_st_min_count = getattr(cfg.MODEL, 'POSE_LTCS_ST_MIN_COUNT', 1)
        ltcs_st_update_stop_epoch = getattr(cfg.MODEL, 'POSE_LTCS_ST_UPDATE_STOP_EPOCH', -1)
        num_train_classes = len(set([d[1] for d in train_loader.dataset.dataset]))
        ltcs_teacher_bank = SupportCompleteBank(
            num_classes=num_train_classes, feat_dim=768, num_keypoints=17,
            low_thr=ltcs_st_low_thr, update_thr=ltcs_st_update_thr,
            momentum=ltcs_st_mom, min_count=ltcs_st_min_count,
        ).to(device)

    lpcs_teacher_bank = None
    if lpcs_enabled:
        lpcs_weight = getattr(cfg.MODEL, 'POSE_LPCS_WEIGHT', 0.5)
        lpcs_warmup = getattr(cfg.MODEL, 'POSE_LPCS_WARMUP', 20)
        lpcs_hidden = getattr(cfg.MODEL, 'POSE_LPCS_HIDDEN', 32)
        lpcs_delta_scale = getattr(cfg.MODEL, 'POSE_LPCS_DELTA_SCALE', 0.5)
        lpcs_head_mode = getattr(cfg.MODEL, 'POSE_LPCS_HEAD_MODE', 'residual')
        lpcs_conf_weight = float(getattr(cfg.MODEL, 'POSE_LPCS_CONF_WEIGHT', 0.25))
        lpcs_pair_mode = getattr(cfg.MODEL, 'POSE_LPCS_PAIR_MODE', 'all')
        lpcs_pair_top_ratio = float(getattr(cfg.MODEL, 'POSE_LPCS_PAIR_TOP_RATIO', 1.0))
        lpcs_rank_mode = getattr(cfg.MODEL, 'POSE_LPCS_RANK_MODE', 'all')
        lpcs_rank_top_ratio = float(getattr(cfg.MODEL, 'POSE_LPCS_RANK_TOP_RATIO', 1.0))
        lpcs_rank_tau = float(getattr(cfg.MODEL, 'POSE_LPCS_RANK_TAU', 8.0))
        lpcs_context_mode = getattr(cfg.MODEL, 'POSE_LPCS_CONTEXT_MODE', 'none')
        lpcs_cvk_global_weight = float(getattr(cfg.TEST, 'CVK_GLOBAL_WEIGHT', 1.0))
        lpcs_cvk_kp_weight = float(getattr(cfg.TEST, 'CVK_KP_WEIGHT', 1.0))
        lpcs_st_low_thr = getattr(cfg.MODEL, 'POSE_LPCS_ST_LOW_THR', 0.3)
        lpcs_st_update_thr = getattr(cfg.MODEL, 'POSE_LPCS_ST_UPDATE_THR', 0.7)
        lpcs_st_mom = getattr(cfg.MODEL, 'POSE_LPCS_ST_MOM', 0.9)
        lpcs_st_min_count = getattr(cfg.MODEL, 'POSE_LPCS_ST_MIN_COUNT', 1)
        lpcs_st_update_stop_epoch = getattr(cfg.MODEL, 'POSE_LPCS_ST_UPDATE_STOP_EPOCH', -1)
        num_train_classes = len(set([d[1] for d in train_loader.dataset.dataset]))
        lpcs_teacher_bank = SupportCompleteBank(
            num_classes=num_train_classes, feat_dim=768, num_keypoints=17,
            low_thr=lpcs_st_low_thr, update_thr=lpcs_st_update_thr,
            momentum=lpcs_st_mom, min_count=lpcs_st_min_count,
        ).to(device)

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    if use_pose:
        logger.info('Pose-guided training ENABLED')
    if ltcs_enabled:
        logger.info(f'[LTCS] enabled: weight={ltcs_weight}, warmup={ltcs_warmup}')
    if lpcs_enabled:
        logger.info(f'[LPCS] enabled: weight={lpcs_weight}, warmup={lpcs_warmup}, '
                    f'head_mode={lpcs_head_mode}, context_mode={lpcs_context_mode}')

    def _compute_ltcs_loss(ltcs_head, global_feat, kp_feats, kp_weights, teacher_kp_feats, labels):
        feat_g = F.normalize(global_feat.detach(), dim=-1)
        kp_base = F.normalize(kp_feats.detach(), dim=-1)
        kp_teacher = F.normalize(teacher_kp_feats.detach(), dim=-1)
        weights = kp_weights.detach().clamp(min=0.0)

        global_dist = euclidean_distance_tensor(feat_g, feat_g)
        base_dist, support_ratio = common_support_distance(
            kp_base, kp_base, weights, weights, fallback=global_dist, return_ratio=True)
        teacher_dist, _ = common_support_distance(
            kp_teacher, kp_teacher, weights, weights, fallback=global_dist, return_ratio=True)

        batch_size = global_feat.size(0)
        eye = torch.eye(batch_size, dtype=torch.bool, device=global_feat.device)
        q_vis_mean = weights.mean(dim=1, keepdim=True).expand(-1, batch_size)
        g_vis_mean = q_vis_mean.t()
        desc = build_pair_descriptors(
            global_dist, base_dist, support_ratio, q_vis_mean, g_vis_mean)
        alpha = ltcs_head(desc.view(-1, desc.shape[-1])).view(batch_size, batch_size)
        mixed_dist = (1.0 - alpha) * global_dist + alpha * base_dist

        informative = (global_dist - base_dist).abs().detach()
        informative = informative / informative.mean().clamp(min=1e-6)
        mask = ~eye
        loss_map = F.smooth_l1_loss(mixed_dist, teacher_dist.detach(), reduction='none')
        loss = (loss_map[mask] * informative[mask]).sum() / informative[mask].sum().clamp(min=1e-6)

        same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        pos_mask = same_label & ~eye
        neg_mask = ~same_label
        if pos_mask.any() and neg_mask.any():
            teacher_gap = float(teacher_dist[neg_mask].mean().item() - teacher_dist[pos_mask].mean().item())
            mixed_gap = float(mixed_dist[neg_mask].mean().item() - mixed_dist[pos_mask].mean().item())
        else:
            teacher_gap = 0.0
            mixed_gap = 0.0

        stats = {
            'alpha_mean': float(alpha[mask].mean().item()),
            'alpha_std': float(alpha[mask].std(unbiased=False).item()),
            'support_mean': float(support_ratio[mask].mean().item()),
            'before_error': float((global_dist.detach() - teacher_dist.detach()).abs()[mask].mean().item()),
            'after_error': float((mixed_dist.detach() - teacher_dist.detach()).abs()[mask].mean().item()),
            'teacher_gap': teacher_gap,
            'mixed_gap': mixed_gap,
        }
        return loss, stats

    def _compute_lpcs_loss(lpcs_head, global_feat, kp_feats, kp_weights, teacher_kp_feats, labels):
        def _select_top(values, ratio, largest=True):
            if values.numel() <= 1 or ratio >= 1.0:
                return torch.ones_like(values, dtype=torch.bool)
            keep = max(1, int(math.ceil(values.numel() * ratio)))
            top_idx = torch.topk(values, k=keep, largest=largest, sorted=False).indices
            mask = torch.zeros_like(values, dtype=torch.bool)
            mask[top_idx] = True
            return mask

        def _rank_decay_factors(values, largest=True):
            if values.numel() <= 1:
                return torch.ones_like(values)
            order = torch.argsort(values, descending=largest)
            ranks = torch.empty_like(order, dtype=values.dtype)
            ranks[order] = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
            return torch.exp(-ranks / lpcs_rank_tau)

        feat_g = F.normalize(global_feat.detach(), dim=-1)
        kp_base = F.normalize(kp_feats.detach(), dim=-1)
        kp_teacher = F.normalize(teacher_kp_feats.detach(), dim=-1)
        weights = kp_weights.detach().clamp(min=0.0)

        global_dist = euclidean_distance_tensor(feat_g, feat_g)
        kp_dist, support_ratio = common_support_distance(
            kp_base, kp_base, weights, weights, fallback=global_dist, return_ratio=True)
        teacher_kp_dist, _ = common_support_distance(
            kp_teacher, kp_teacher, weights, weights, fallback=global_dist, return_ratio=True)

        weight_sum = max(lpcs_cvk_global_weight + lpcs_cvk_kp_weight, 1e-6)
        base_dist = (lpcs_cvk_global_weight * global_dist + lpcs_cvk_kp_weight * kp_dist) / weight_sum
        teacher_dist = (lpcs_cvk_global_weight * global_dist + lpcs_cvk_kp_weight * teacher_kp_dist) / weight_sum
        pair_change = (teacher_dist.detach() - base_dist.detach()).abs()

        batch_size = global_feat.size(0)
        eye = torch.eye(batch_size, dtype=torch.bool, device=global_feat.device)
        q_vis_mean = weights.mean(dim=1, keepdim=True).expand(-1, batch_size)
        g_vis_mean = q_vis_mean.t()
        desc = build_pair_descriptors(
            global_dist, kp_dist, support_ratio, q_vis_mean, g_vis_mean)
        context_mean = 0.0
        if lpcs_context_mode == 'query_ctx':
            pair_gap = (kp_dist.detach() - base_dist.detach()).abs()
            row_ctx = build_query_context_descriptors(
                base_dist.detach(), support_ratio.detach(),
                pair_change=pair_gap, valid_mask=~eye)
            desc = torch.cat([desc, row_ctx], dim=-1)
            context_mean = float(row_ctx.abs().mean().item())
        elif lpcs_context_mode == 'comp_ctx':
            comp_ctx = build_query_competition_descriptors(
                base_dist.detach(), kp_dist.detach(),
                support_ratio.detach(), valid_mask=~eye)
            desc = torch.cat([desc, comp_ctx], dim=-1)
            context_mean = float(comp_ctx.abs().mean().item())

        if lpcs_head_mode == 'residual_conf':
            raw_delta, conf_logits = lpcs_head(desc.view(-1, desc.shape[-1]))
            raw_delta = raw_delta.view(batch_size, batch_size)
            conf_logits = conf_logits.view(batch_size, batch_size)
            conf = torch.sigmoid(conf_logits)
            delta = conf * raw_delta
        else:
            raw_delta = lpcs_head(desc.view(-1, desc.shape[-1])).view(batch_size, batch_size)
            conf_logits = None
            conf = None
            delta = raw_delta
        final_dist = base_dist + delta

        same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        pos_mask = same_label & ~eye
        neg_mask = ~same_label
        pair_weight = pair_change / pair_change[~eye].mean().clamp(min=1e-6)
        conf_target = 1.0 - torch.exp(-pair_change / pair_change[~eye].mean().clamp(min=1e-6))

        total_loss = torch.tensor(0.0, device=global_feat.device)
        total_weight = torch.tensor(0.0, device=global_feat.device)
        selected_pair_count = 0.0
        total_pair_count = 0.0
        selected_pair_weight_sum = 0.0
        total_pair_weight_sum = 0.0
        rank_selected_pair_count = 0.0
        rank_factor_sum = 0.0
        rank_factor_count = 0.0
        for idx in range(batch_size):
            pos = final_dist[idx][pos_mask[idx]]
            neg = final_dist[idx][neg_mask[idx]]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            pos_w_full = pair_weight[idx][pos_mask[idx]]
            neg_w_full = pair_weight[idx][neg_mask[idx]]

            if lpcs_pair_mode == 'delta_top':
                pos_sel = _select_top(pos_w_full, lpcs_pair_top_ratio)
                neg_sel = _select_top(neg_w_full, lpcs_pair_top_ratio)
            else:
                pos_sel = torch.ones_like(pos_w_full, dtype=torch.bool)
                neg_sel = torch.ones_like(neg_w_full, dtype=torch.bool)

            pos = pos[pos_sel]
            neg = neg[neg_sel]
            pos_w = pos_w_full[pos_sel]
            neg_w = neg_w_full[neg_sel]
            if pos.numel() == 0 or neg.numel() == 0:
                continue

            routed_pos_w = pos_w
            routed_neg_w = neg_w

            if lpcs_rank_mode == 'hard_top':
                pos_rank_sel = _select_top(pos, lpcs_rank_top_ratio, largest=True)
                neg_rank_sel = _select_top(neg, lpcs_rank_top_ratio, largest=False)
                pos_rank_factor = torch.ones_like(pos)
                neg_rank_factor = torch.ones_like(neg)
            elif lpcs_rank_mode == 'rank_decay':
                pos_rank_sel = torch.ones_like(pos, dtype=torch.bool)
                neg_rank_sel = torch.ones_like(neg, dtype=torch.bool)
                pos_rank_factor = _rank_decay_factors(pos, largest=True)
                neg_rank_factor = _rank_decay_factors(neg, largest=False)
            else:
                pos_rank_sel = torch.ones_like(pos, dtype=torch.bool)
                neg_rank_sel = torch.ones_like(neg, dtype=torch.bool)
                pos_rank_factor = torch.ones_like(pos)
                neg_rank_factor = torch.ones_like(neg)

            pos = pos[pos_rank_sel]
            neg = neg[neg_rank_sel]
            pos_w = pos_w[pos_rank_sel] * pos_rank_factor[pos_rank_sel]
            neg_w = neg_w[neg_rank_sel] * neg_rank_factor[neg_rank_sel]
            pos_rank_factor = pos_rank_factor[pos_rank_sel]
            neg_rank_factor = neg_rank_factor[neg_rank_sel]
            if pos.numel() == 0 or neg.numel() == 0:
                continue

            selected_pair_count += float(routed_pos_w.numel() + routed_neg_w.numel())
            total_pair_count += float(pos_w_full.numel() + neg_w_full.numel())
            rank_selected_pair_count += float(pos_w.numel() + neg_w.numel())
            selected_pair_weight_sum += float(routed_pos_w.sum().item() + routed_neg_w.sum().item())
            total_pair_weight_sum += float(pos_w_full.sum().item() + neg_w_full.sum().item())
            rank_factor_sum += float(pos_rank_factor.sum().item() + neg_rank_factor.sum().item())
            rank_factor_count += float(pos_rank_factor.numel() + neg_rank_factor.numel())
            rank_term = F.softplus(pos.unsqueeze(1) - neg.unsqueeze(0))
            rank_weight = torch.sqrt(pos_w.unsqueeze(1) * neg_w.unsqueeze(0))
            total_loss = total_loss + (rank_term * rank_weight).sum()
            total_weight = total_weight + rank_weight.sum()
        loss = total_loss / total_weight.clamp(min=1e-6)
        conf_loss = None
        if conf is not None:
            mask = ~eye
            conf_loss = F.binary_cross_entropy_with_logits(
                conf_logits[mask], conf_target[mask], reduction='none')
            conf_loss = (conf_loss * pair_weight[mask]).sum() / pair_weight[mask].sum().clamp(min=1e-6)
            loss = loss + lpcs_conf_weight * conf_loss

        if pos_mask.any() and neg_mask.any():
            base_gap = float(base_dist[neg_mask].mean().item() - base_dist[pos_mask].mean().item())
            final_gap = float(final_dist[neg_mask].mean().item() - final_dist[pos_mask].mean().item())
        else:
            base_gap = 0.0
            final_gap = 0.0

        mask = ~eye
        pair_selected_ratio = selected_pair_count / max(total_pair_count, 1e-6)
        pair_focus = 1.0
        if selected_pair_count > 0.0 and total_pair_weight_sum > 0.0:
            pair_focus = (selected_pair_weight_sum / selected_pair_count) / (
                total_pair_weight_sum / max(total_pair_count, 1e-6))
        rank_selected_ratio = rank_selected_pair_count / max(selected_pair_count, 1e-6)
        rank_weight_mean = rank_factor_sum / max(rank_factor_count, 1e-6)
        stats = {
            'delta_mean': float(delta[mask].mean().item()),
            'raw_delta_mean': float(raw_delta[mask].mean().item()),
            'delta_std': float(delta[mask].std(unbiased=False).item()),
            'support_mean': float(support_ratio[mask].mean().item()),
            'change_mean': float(pair_change[mask].mean().item()),
            'weight_mean': float(pair_weight[mask].mean().item()),
            'base_gap': base_gap,
            'final_gap': final_gap,
            'pair_selected_ratio': float(pair_selected_ratio),
            'pair_focus': float(pair_focus),
            'rank_selected_ratio': float(rank_selected_ratio),
            'rank_weight_mean': float(rank_weight_mean),
            'context_mean': float(context_mean),
        }
        if conf is not None:
            stats['conf_mean'] = float(conf[mask].mean().item())
            stats['conf_target_mean'] = float(conf_target[mask].mean().item())
            stats['conf_loss'] = float(conf_loss.item())
        return loss, stats

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    detail_meters = {}

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
    scaler = amp.GradScaler()

    # Backbone freeze warmup
    freeze_epochs = cfg.SOLVER.FREEZE_BACKBONE_EPOCHS
    backbone_frozen = False

    def _freeze_backbone(model):
        m = model.module if hasattr(model, 'module') else model
        for name, param in m.base.named_parameters():
            param.requires_grad = False
        frozen = sum(1 for p in m.base.parameters() if not p.requires_grad)
        total = sum(1 for p in m.base.parameters())
        logger.info(f'Backbone FROZEN: {frozen}/{total} params frozen')

    def _unfreeze_backbone(model):
        m = model.module if hasattr(model, 'module') else model
        for param in m.base.parameters():
            param.requires_grad = True
        logger.info('Backbone UNFROZEN: all params trainable')

    if freeze_epochs > 0:
        _freeze_backbone(model)
        backbone_frozen = True
        logger.info(f'Backbone freeze warmup: {freeze_epochs} epochs')

    # OA-SD: create EMA teacher model
    oa_sd_enabled = getattr(cfg.MODEL, 'POSE_OA_SD', False)
    ema_teacher = None
    ema_decay = float(getattr(cfg.MODEL, 'POSE_OA_SD_EMA_DECAY', 0.999))
    if oa_sd_enabled:
        import copy
        base_model = model.module if hasattr(model, 'module') else model
        ema_teacher = copy.deepcopy(base_model)
        ema_teacher.eval()
        for p in ema_teacher.parameters():
            p.requires_grad = False
        logger.info(f'[OA-SD] EMA teacher created (decay={ema_decay})')
        if not getattr(cfg.MODEL, 'POSE_LOWER_BODY_OCC', False):
            logger.warning('[OA-SD] WARNING: PLBOA is disabled. Teacher and student see near-identical images.')

    # train
    for epoch in range(1, epochs + 1):
        if backbone_frozen and epoch > freeze_epochs:
            _unfreeze_backbone(model)
            backbone_frozen = False

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in detail_meters.values():
            m.reset()
        evaluator.reset()
        model.train()

        for n_iter, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            if use_pose:
                img, vid, target_cam, target_view, pose_dict = batch_data
                pose_dict = _pose_to_device(pose_dict, device)
            else:
                img, vid, target_cam, target_view = batch_data
                pose_dict = None

            # Handle multi-view modes: img may be list of tensors
            # parallel_aug: 3 views, OA-SD: 2 views (student + teacher)
            # parallel_aug + OA-SD: 4 views (3 student + 1 teacher), standard: 1 view
            parallel_aug = isinstance(img, list) and len(img) >= 3
            oa_sd_mode = isinstance(img, list) and len(img) == 2
            # Combined mode: parallel_aug with OA-SD teacher view appended as 4th element
            parallel_oa_sd = parallel_aug and oa_sd_enabled and len(img) == 4
            if parallel_aug:
                if parallel_oa_sd:
                    img_views = [v.to(device) for v in img[:3]]  # 3 student views
                    img_teacher = img[3].to(device)               # teacher (clean pre-PLBOA)
                else:
                    img_views = [v.to(device) for v in img]
                batch_size = img_views[0].shape[0]
            elif oa_sd_mode:
                img_student = img[0].to(device)  # occluded (post-PLBOA)
                img_teacher = img[1].to(device)  # clean (pre-PLBOA)
                img = img_student
                batch_size = img.shape[0]
            else:
                img = img.to(device)
                batch_size = img.shape[0]
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                if parallel_aug and use_pose:
                    # 3-view parallel augmentation: forward all, average loss
                    all_scores, all_feats, all_recon, all_kpdata = [], [], [], []
                    for v_img in img_views:
                        m_out = model(v_img, label=target, cam_label=target_cam,
                                      view_label=target_view, pose_dict=pose_dict)
                        kd = None
                        if len(m_out) == 5:
                            s, f, fm_v, rl, kd = m_out
                        elif len(m_out) == 4:
                            s, f, fm_v, rl = m_out
                        else:
                            s, f, fm_v = m_out[:3]; rl = None
                        all_scores.append(s)
                        all_feats.append(f)
                        all_recon.append(rl)
                        all_kpdata.append(kd)
                    score, feat = all_scores[0], all_feats[0]
                    recon_loss = all_recon[0]
                    kp_data = all_kpdata[0]
                elif use_pose:
                    model_out = model(img, label=target, cam_label=target_cam,
                                      view_label=target_view, pose_dict=pose_dict)
                    kp_data = None
                    if len(model_out) == 5:
                        score, feat, feat_maps, recon_loss, kp_data = model_out
                    elif len(model_out) == 4:
                        score, feat, feat_maps, recon_loss = model_out
                    else:
                        score, feat, feat_maps = model_out
                        recon_loss = None
                else:
                    score, feat, _ = model(img, label=target, cam_label=target_cam,
                                           view_label=target_view)
                    recon_loss = None
                    kp_data = None

                # Prepare kp_aux_data for loss function
                maxsim_tri_enabled = getattr(cfg.MODEL, 'POSE_MAXSIM_TRIPLET', False)
                evid_enabled = getattr(cfg.MODEL, 'POSE_EVIDENTIAL', False)
                supcon_vis_enabled = getattr(cfg.MODEL, 'POSE_STR_SUPCON_VIS_WEIGHT', False)
                kp_aux_data = None
                if kp_data is not None and (maxsim_tri_enabled or evid_enabled or supcon_vis_enabled):
                    kp_aux_data = dict(kp_data)
                    kp_aux_data['epoch'] = epoch

                # LTCS teacher bank replacement
                if ltcs_enabled and ltcs_teacher_bank is not None and kp_data is not None and epoch > ltcs_warmup:
                    if kp_aux_data is None:
                        kp_aux_data = dict(kp_data)
                        kp_aux_data['epoch'] = epoch
                    kp_feats_ltcs = kp_data.get('kp_feats')
                    kp_w_ltcs = kp_data.get('kp_weights')
                    if kp_feats_ltcs is not None and kp_w_ltcs is not None:
                        teacher_feats_ltcs, _, teacher_stats_ltcs = ltcs_teacher_bank.replace(
                            kp_feats_ltcs, kp_w_ltcs, target)
                        kp_aux_data['ltcs_teacher_feats'] = teacher_feats_ltcs.detach()
                        kp_aux_data['ltcs_teacher_stats'] = teacher_stats_ltcs

                # LPCS teacher bank replacement
                if lpcs_enabled and lpcs_teacher_bank is not None and kp_data is not None and epoch > lpcs_warmup:
                    if kp_aux_data is None:
                        kp_aux_data = dict(kp_data)
                        kp_aux_data['epoch'] = epoch
                    kp_feats_lpcs = kp_data.get('kp_feats')
                    kp_w_lpcs = kp_data.get('kp_weights')
                    if kp_feats_lpcs is not None and kp_w_lpcs is not None:
                        teacher_feats_lpcs, _, teacher_stats_lpcs = lpcs_teacher_bank.replace(
                            kp_feats_lpcs, kp_w_lpcs, target)
                        kp_aux_data['lpcs_teacher_feats'] = teacher_feats_lpcs.detach()
                        kp_aux_data['lpcs_teacher_stats'] = teacher_stats_lpcs

                loss = loss_fn(score, feat, target, target_cam, kp_data=kp_aux_data)

                # STD-PR: log structural routing stats (token norms, self-attn diagnostics)
                if kp_data is not None and 'str_stats' in kp_data:
                    details = getattr(loss, '_loss_details', {})
                    for k, v in kp_data['str_stats'].items():
                        details[f'str_{k}'] = v
                    loss._loss_details = details

                # PNIS: log pose normalizer stats
                if kp_data is not None and 'pn_stats' in kp_data:
                    details = getattr(loss, '_loss_details', {})
                    pn = kp_data['pn_stats']
                    details['pn_alpha'] = pn['alpha']
                    details['pn_off'] = pn['offset_norm']
                    details['pn_ratio'] = pn['ratio']
                    loss._loss_details = details

                # LTCS loss
                if ltcs_enabled and kp_aux_data is not None and 'ltcs_teacher_feats' in kp_aux_data:
                    _m = model.module if hasattr(model, 'module') else model
                    global_feat_ltcs = feat[0] if isinstance(feat, list) else feat
                    ltcs_loss, ltcs_stats = _compute_ltcs_loss(
                        _m.ltcs_head, global_feat_ltcs,
                        kp_data.get('kp_feats'), kp_data.get('kp_weights'),
                        kp_aux_data['ltcs_teacher_feats'], target)
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + ltcs_weight * ltcs_loss
                    details['ltcs'] = ltcs_loss.item()
                    details['ltcs_a'] = ltcs_stats['alpha_mean']
                    details['ltcs_as'] = ltcs_stats['alpha_std']
                    details['ltcs_sm'] = ltcs_stats['support_mean']
                    details['ltcs_be'] = ltcs_stats['before_error']
                    details['ltcs_ae'] = ltcs_stats['after_error']
                    details['ltcs_tg'] = ltcs_stats['teacher_gap']
                    details['ltcs_mg'] = ltcs_stats['mixed_gap']
                    loss._loss_details = details

                # LPCS loss
                if lpcs_enabled and kp_aux_data is not None and 'lpcs_teacher_feats' in kp_aux_data:
                    _m = model.module if hasattr(model, 'module') else model
                    global_feat_lpcs = feat[0] if isinstance(feat, list) else feat
                    lpcs_loss, lpcs_stats = _compute_lpcs_loss(
                        _m.lpcs_head, global_feat_lpcs,
                        kp_data.get('kp_feats'), kp_data.get('kp_weights'),
                        kp_aux_data['lpcs_teacher_feats'], target)
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + lpcs_weight * lpcs_loss
                    details['lpcs'] = lpcs_loss.item()
                    details['lpcs_dm'] = lpcs_stats['delta_mean']
                    details['lpcs_rdm'] = lpcs_stats['raw_delta_mean']
                    details['lpcs_ds'] = lpcs_stats['delta_std']
                    details['lpcs_sm'] = lpcs_stats['support_mean']
                    details['lpcs_cm'] = lpcs_stats['change_mean']
                    details['lpcs_wm'] = lpcs_stats['weight_mean']
                    details['lpcs_bg'] = lpcs_stats['base_gap']
                    details['lpcs_fg'] = lpcs_stats['final_gap']
                    details['lpcs_psr'] = lpcs_stats['pair_selected_ratio']
                    details['lpcs_pf'] = lpcs_stats['pair_focus']
                    details['lpcs_rsr'] = lpcs_stats['rank_selected_ratio']
                    details['lpcs_rwm'] = lpcs_stats['rank_weight_mean']
                    details['lpcs_ctxm'] = lpcs_stats['context_mean']
                    if 'conf_mean' in lpcs_stats:
                        details['lpcs_cf'] = lpcs_stats['conf_mean']
                        details['lpcs_ctm'] = lpcs_stats['conf_target_mean']
                        details['lpcs_cl'] = lpcs_stats['conf_loss']
                    loss._loss_details = details

                if recon_loss is not None:
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + recon_loss
                    details['recon'] = recon_loss.item()
                    loss._loss_details = details

                # Parallel augmentation: add losses from view 2 and 3
                if parallel_aug and use_pose:
                    saved_details = getattr(loss, '_loss_details', {})
                    for vi in range(1, len(all_scores)):
                        v_loss = loss_fn(all_scores[vi], all_feats[vi], target, target_cam)
                        if all_recon[vi] is not None:
                            v_loss = v_loss + all_recon[vi]
                        loss = loss + v_loss
                    loss = loss / len(all_scores)
                    loss._loss_details = saved_details

                # OA-SD: Occlusion-Asymmetric Self-Distillation with EMA teacher
                if oa_sd_enabled and (oa_sd_mode or parallel_oa_sd) and use_pose and ema_teacher is not None:
                    oa_sd_weight = float(getattr(cfg.MODEL, 'POSE_OA_SD_WEIGHT', 1.0))
                    # EMA Teacher forward: clean image (no PLBOA), no grad
                    # Must set training=True temporarily so forward returns (score, feat, ...) not (test_feat, featmaps)
                    with torch.no_grad():
                        ema_teacher.train()
                        teacher_out = ema_teacher(img_teacher, label=target,
                                                 cam_label=target_cam,
                                                 view_label=target_view,
                                                 pose_dict=pose_dict)
                        ema_teacher.eval()
                        if len(teacher_out) == 5:
                            _, teacher_feat, _, _, _ = teacher_out
                        elif len(teacher_out) == 4:
                            _, teacher_feat, _, _ = teacher_out
                        else:
                            _, teacher_feat, _ = teacher_out[:3]
                    # Distillation: student features → teacher features
                    # For per-token: feat = [global, tok1, ..., tok6]
                    oa_sd_global_only = getattr(cfg.MODEL, 'POSE_OA_SD_GLOBAL_ONLY', False)
                    if oa_sd_global_only:
                        # Global-only distillation: only distill the global (pooled) feature
                        # This avoids gradient conflict with SupCon on per-token features
                        sf = feat[0] if isinstance(feat, list) else feat
                        tf = teacher_feat[0] if isinstance(teacher_feat, list) else teacher_feat
                        sf_norm = F.normalize(sf, p=2, dim=1)
                        tf_norm = F.normalize(tf.detach(), p=2, dim=1)
                        oa_sd_loss = (1.0 - (sf_norm * tf_norm).sum(dim=1)).mean()
                    elif isinstance(feat, list) and isinstance(teacher_feat, list):
                        # All-token distillation: distill global + each structural token
                        distill_losses = []
                        for sf, tf in zip(feat, teacher_feat):
                            sf_norm = F.normalize(sf, p=2, dim=1)
                            tf_norm = F.normalize(tf.detach(), p=2, dim=1)
                            d_loss = (1.0 - (sf_norm * tf_norm).sum(dim=1)).mean()
                            distill_losses.append(d_loss)
                        oa_sd_loss = sum(distill_losses) / len(distill_losses)
                    else:
                        # Fallback: single feature distillation
                        sf = feat[0] if isinstance(feat, list) else feat
                        tf = teacher_feat[0] if isinstance(teacher_feat, list) else teacher_feat
                        sf_norm = F.normalize(sf, p=2, dim=1)
                        tf_norm = F.normalize(tf.detach(), p=2, dim=1)
                        oa_sd_loss = (1.0 - (sf_norm * tf_norm).sum(dim=1)).mean()
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + oa_sd_weight * oa_sd_loss
                    details['oa_sd'] = oa_sd_loss.item()
                    loss._loss_details = details

                # SPLADE: auxiliary sparse CE + sparsity regularization
                splade_enabled = getattr(cfg.MODEL, 'POSE_SPLADE', False)
                if splade_enabled and kp_data is not None and 'splade_cls' in kp_data:
                    splade_reg_w = float(getattr(cfg.MODEL, 'POSE_SPLADE_REG', 0.01))
                    # Sparse CE loss (0.5 weight, same as part branch)
                    splade_ce = F.cross_entropy(kp_data['splade_cls'], target)
                    # Sparsity regularization (FLOPS-style: penalize total activation)
                    splade_reg_loss = kp_data['splade_reg']
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + 0.5 * splade_ce + splade_reg_w * splade_reg_loss
                    details['splade_ce'] = splade_ce.item()
                    details['splade_reg'] = splade_reg_loss.item()
                    details['splade_sp'] = kp_data.get('splade_sparsity', 0)
                    loss._loss_details = details

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # OA-SD: update EMA teacher after optimizer step
            if ema_teacher is not None:
                base_model = model.module if hasattr(model, 'module') else model
                with torch.no_grad():
                    for t_param, s_param in zip(ema_teacher.parameters(), base_model.parameters()):
                        t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1.0 - ema_decay)

            # Bank updates (after optimizer step)
            if ltcs_enabled and ltcs_teacher_bank is not None and kp_data is not None:
                kp_feats_ltcs = kp_data.get('kp_feats')
                kp_w_ltcs = kp_data.get('kp_weights')
                if kp_feats_ltcs is not None and kp_w_ltcs is not None:
                    if ltcs_st_update_stop_epoch < 0 or epoch <= ltcs_st_update_stop_epoch:
                        ltcs_teacher_bank.update(kp_feats_ltcs, kp_w_ltcs, target)

            if lpcs_enabled and lpcs_teacher_bank is not None and kp_data is not None:
                kp_feats_lpcs = kp_data.get('kp_feats')
                kp_w_lpcs = kp_data.get('kp_weights')
                if kp_feats_lpcs is not None and kp_w_lpcs is not None:
                    if lpcs_st_update_stop_epoch < 0 or epoch <= lpcs_st_update_stop_epoch:
                        lpcs_teacher_bank.update(kp_feats_lpcs, kp_w_lpcs, target)

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, 1)

            if hasattr(loss, '_loss_details'):
                for k, v in loss._loss_details.items():
                    if k not in detail_meters:
                        detail_meters[k] = AverageMeter()
                    detail_meters[k].update(v, batch_size)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        detail_str = ' '.join(f'{k}: {m.avg:.3f}' for k, m in detail_meters.items() if k != 'total')
                        log_msg = "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                            epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr)
                        if detail_str:
                            log_msg += f" | {detail_str}"
                        logger.info(log_msg)
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    detail_str = ' '.join(f'{k}: {m.avg:.3f}' for k, m in detail_meters.items() if k != 'total')
                    log_msg = "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr)
                    if detail_str:
                        log_msg += f" | {detail_str}"
                    logger.info(log_msg)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        epoch_time = time_per_batch * (n_iter + 1)
        remaining_epochs = epochs - epoch
        eta_seconds = remaining_epochs * epoch_time
        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s] ETA: {}h{}m"
                    .format(epoch, epoch_time, train_loader.batch_size / time_per_batch, eta_h, eta_m))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    _eval_model = model.module if hasattr(model, 'module') else model
                    evaluator.pair_fusion_head = getattr(_eval_model, 'ltcs_head', None)
                    evaluator.pair_residual_head = getattr(_eval_model, 'lpcs_head', None)
                    for n_iter, batch_data in enumerate(val_loader):
                        with torch.no_grad():
                            if use_pose:
                                img, vid, camid, camids, target_view, _, pose_dict = batch_data
                                pose_dict = _pose_to_device(pose_dict, device)
                            else:
                                img, vid, camid, camids, target_view, _ = batch_data
                                pose_dict = None
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            if use_pose:
                                feat, _ = model(img, cam_label=camids, view_label=target_view,
                                                pose_dict=pose_dict)
                            else:
                                feat, _ = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                _eval_model = model.module if hasattr(model, 'module') else model
                evaluator.pair_fusion_head = getattr(_eval_model, 'ltcs_head', None)
                evaluator.pair_residual_head = getattr(_eval_model, 'lpcs_head', None)
                for n_iter, batch_data in enumerate(val_loader):
                    with torch.no_grad():
                        if use_pose:
                            img, vid, camid, camids, target_view, _, pose_dict = batch_data
                            pose_dict = _pose_to_device(pose_dict, device)
                        else:
                            img, vid, camid, camids, target_view, _ = batch_data
                            pose_dict = None
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        if use_pose:
                            feat, _ = model(img, cam_label=camids, view_label=target_view,
                                            pose_dict=pose_dict)
                        else:
                            feat, _ = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")
    use_pose = cfg.MODEL.POSE_ENABLED

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                            reranking=cfg.TEST.RE_RANKING, cfg=cfg)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    _eval_model = model.module if hasattr(model, 'module') else model
    evaluator.pair_fusion_head = getattr(_eval_model, 'ltcs_head', None)
    evaluator.pair_residual_head = getattr(_eval_model, 'lpcs_head', None)

    model.eval()
    img_path_list = []

    for n_iter, batch_data in enumerate(val_loader):
        with torch.no_grad():
            if use_pose:
                img, pid, camid, camids, target_view, imgpath, pose_dict = batch_data
                pose_dict = _pose_to_device(pose_dict, device)
            else:
                img, pid, camid, camids, target_view, imgpath = batch_data
                pose_dict = None
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            if use_pose:
                feat, _ = model(img, cam_label=camids, view_label=target_view,
                                pose_dict=pose_dict)
            else:
                feat, _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
