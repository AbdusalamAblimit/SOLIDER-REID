import math
import logging
import os
import random
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.flip_test import flip_batch
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
    result = {}
    for k, v in pose_dict.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            # Nested pose dict (e.g. teacher_pose)
            result[k] = _pose_to_device(v, device)
        else:
            result[k] = v
    return result


def _extract_feat_flip(model, img, pose_dict, camids, target_view, use_pose, flip_test):
    """Forward once (default) or twice with horizontal flip, return averaged feature.

    When `flip_test=True`, features of original and flipped batch are averaged.
    Used by both mid-training eval and do_inference. MaxSim post-hoc eval stays
    in `scripts/eval_fliptest_maxsim.py` and is NOT invoked here.

    Fused-feature averaging requires **per-block L2-renorm** to preserve the
    equal_concat semantics: model output `[g_norm | p1_norm | ... | pN_norm]`
    has each block unit-norm; whole-vector averaging followed by a single global
    L2-normalize distorts the per-block weights (blocks with high flip-invariance
    get amplified, blocks with low flip-invariance get attenuated). The fix
    detects `equal_concat` via `model.pose_test_feat` and re-normalizes each
    C-dim block after averaging.

    Output handling:
    - Tensor `feat` in `equal_concat` mode: split into (B, n_blocks, C), average,
      F.normalize per block, reshape back. Preserves equal-weight cosine fusion.
    - Tensor `feat` in `global` / `gcn_only` / `part_only` (single block) modes:
      straight `(feat + feat_flip) / 2` — whole-vector renorm downstream is
      exactly correct for single block.
    - Tensor `feat` in `concat_scaled`: straight average (scale factors baked
      in by the model; whole-vector renorm downstream preserves relative weights).
    - Dict `feat` (cvk_* / maxsim* modes): average each field, then L2-renorm
      `global_feat` (dim=1) and `kp_feats` (dim=2) for downstream correctness.
    """
    if use_pose:
        feat, _ = model(img, cam_label=camids, view_label=target_view, pose_dict=pose_dict)
    else:
        feat, _ = model(img, cam_label=camids, view_label=target_view)
    if not flip_test:
        return feat
    img_f, pose_f = flip_batch(img, pose_dict if use_pose else None)
    if use_pose:
        feat_f, _ = model(img_f, cam_label=camids, view_label=target_view, pose_dict=pose_f)
    else:
        feat_f, _ = model(img_f, cam_label=camids, view_label=target_view)

    if isinstance(feat, dict) and isinstance(feat_f, dict):
        merged = dict(feat)
        for k in ('global_feat', 'kp_feats', 'kp_weights'):
            if k in feat and k in feat_f and isinstance(feat[k], torch.Tensor):
                merged[k] = (feat[k] + feat_f[k]) / 2.0
        # Re-normalize averaged feature blocks for MaxSim/CVK-style downstream
        # (they expect unit-norm global/kp; maxsim script also does this post-hoc).
        if 'global_feat' in merged and merged['global_feat'].dim() == 2:
            merged['global_feat'] = F.normalize(merged['global_feat'], p=2, dim=1)
        if 'kp_feats' in merged and merged['kp_feats'].dim() == 3:
            merged['kp_feats'] = F.normalize(merged['kp_feats'], p=2, dim=2)
        return merged

    avg = (feat + feat_f) / 2.0

    # Per-block L2-renorm for equal_concat mode only.
    mode = getattr(model, 'pose_test_feat', None)
    if mode == 'equal_concat':
        # Model feature dim (per-block C). Prefer in_planes which is set by
        # build_transformer (1024 for Swin-Base, 768 for Swin-Tiny/Small).
        C = getattr(model, 'in_planes', None) or getattr(model, 'num_features', None)
        if C is not None and avg.dim() == 2 and avg.shape[1] > C and avg.shape[1] % C == 0:
            n_blocks = avg.shape[1] // C
            avg = avg.view(avg.shape[0], n_blocks, C)
            avg = F.normalize(avg, p=2, dim=2)
            avg = avg.reshape(avg.shape[0], n_blocks * C).contiguous()
    return avg


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

    def _pose_hyper_lora_log(current_model):
        """Compact live audit for exp376; empty on every legacy config."""
        base_model = (current_model.module
                      if hasattr(current_model, 'module') else current_model)
        modules = getattr(base_model, 'pose_hyper_lora_modules', None)
        stats = getattr(base_model, '_last_pose_hyper_lora_stats', {})
        if not modules or not stats:
            return ''
        alphas = torch.stack([
            module.residual_scale.detach().float().cpu()
            for module in modules.values()])
        visibility = torch.stack([
            value['visibility_mean'].detach().float().cpu()
            for value in stats.values()])
        coefficients = torch.stack([
            value['coefficient_abs_mean'].detach().float().cpu()
            for value in stats.values()])
        delta_rms = torch.stack([
            value['delta_rms'].detach().float().cpu()
            for value in stats.values()])
        return (' | HyperLoRA alpha={:.3e}[{:.3e},{:.3e}] '
                'vis={:.3e} coeff={:.3e} delta_rms={:.3e}'.format(
                    alphas.mean().item(), alphas.min().item(),
                    alphas.max().item(), visibility.mean().item(),
                    coefficients.mean().item(), delta_rms.mean().item()))

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
    scaler = amp.GradScaler(
        init_scale=float(getattr(cfg.SOLVER, 'AMP_INIT_SCALE', 65536.0)))

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

    # OA-SD / OA-RD: create EMA teacher model (shared infrastructure)
    oa_sd_enabled = getattr(cfg.MODEL, 'POSE_OA_SD', False)
    oa_rd_enabled = getattr(cfg.MODEL, 'POSE_OA_RD', False)
    ema_teacher = None
    ema_decay = float(getattr(cfg.MODEL, 'POSE_OA_SD_EMA_DECAY', 0.999))
    if oa_sd_enabled or oa_rd_enabled:
        import copy
        base_model = model.module if hasattr(model, 'module') else model
        ema_teacher = copy.deepcopy(base_model)
        ema_teacher.eval()
        for p in ema_teacher.parameters():
            p.requires_grad = False
        mode_str = []
        if oa_sd_enabled: mode_str.append('OA-SD')
        if oa_rd_enabled: mode_str.append('OA-RD')
        logger.info(f'[{"+".join(mode_str)}] EMA teacher created (decay={ema_decay})')
        if not getattr(cfg.MODEL, 'POSE_LOWER_BODY_OCC', False):
            logger.warning('[OA-SD/RD] WARNING: PLBOA is disabled. Teacher and student see near-identical images.')

    # VC-Norm config guard (codex Medium-a): VCA is wired INSIDE the OA-SD branch
    # (it consumes the EMA teacher's per-keypoint tokens). With POSE_VCNORM=True
    # but POSE_OA_SD=False, the VCA block is never reached and alignment is
    # silently skipped — a no-op that would masquerade as "VC-Norm ran". Fail
    # loudly instead. VC-Norm also needs PLBOA to create the occluded-vs-clean
    # asymmetry it aligns away (without it the occluded cohort is empty).
    if getattr(cfg.MODEL, 'POSE_VCNORM', False):
        assert getattr(cfg.MODEL, 'POSE_OA_SD', False), (
            'POSE_VCNORM=True requires POSE_OA_SD=True: VCA consumes the OA-SD '
            'EMA teacher per-keypoint tokens (teacher_kp_data). Enable OA-SD or '
            'disable VC-Norm.')
        if not getattr(cfg.MODEL, 'POSE_LOWER_BODY_OCC', False):
            logger.warning(
                '[VC-Norm] WARNING: PLBOA (POSE_LOWER_BODY_OCC) is disabled. The '
                'occluded-vs-clean asymmetry VCA aligns away will be near-empty; '
                'the occluded student cohort (s_occ & t_vis) collapses to ~0 and '
                'VCA becomes a no-op (valid_k=0).')

    # PACI: Part Prototype Bank
    paci_enabled = getattr(cfg.MODEL, 'POSE_PACI', False)
    paci_bank = None
    if paci_enabled:
        from model.modules.part_prototype_bank import PartPrototypeBank
        base_m = model.module if hasattr(model, 'module') else model
        feat_dim = base_m.in_planes  # backbone feature dim
        paci_num_classes = base_m.classifier.weight.shape[0]  # from classifier output dim
        paci_bank = PartPrototypeBank(
            num_classes=paci_num_classes, num_parts=17, feat_dim=feat_dim,
            momentum=float(getattr(cfg.MODEL, 'POSE_PACI_MOMENTUM', 0.9)),
            vis_threshold=0.3,
        ).to('cuda')
        logger.info(f'[PACI] Part Prototype Bank created: {paci_num_classes} IDs x 17 parts x {feat_dim}D')

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
            # Combined mode: parallel_aug with OA-SD/OA-RD teacher view appended as 4th element
            parallel_oa_sd = parallel_aug and (oa_sd_enabled or oa_rd_enabled) and len(img) == 4
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

                # Structural Token Mixup (STM): swap body-part tokens between same-ID samples
                # Creates diverse token combinations to improve occlusion robustness
                stm_enabled = getattr(cfg.MODEL, 'POSE_STM', False)
                if stm_enabled and isinstance(feat, list) and len(feat) > 1:
                    stm_num_swap = int(getattr(cfg.MODEL, 'POSE_STM_NUM_SWAP', 2))
                    stm_prob = float(getattr(cfg.MODEL, 'POSE_STM_PROB', 0.5))
                    stm_weight = float(getattr(cfg.MODEL, 'POSE_STM_WEIGHT', 0.5))
                    num_parts = len(feat) - 1  # exclude global feat[0]
                    B = feat[0].shape[0]
                    num_instance = cfg.DATALOADER.NUM_INSTANCE
                    num_ids = B // num_instance

                    if random.random() < stm_prob and num_ids > 0:
                        # Fixed-count generation: each ID produces exactly num_instance mixed samples
                        # This ensures triplet loss's equal-positive-count requirement
                        mixed_scores_all = []
                        mixed_feats_all = []
                        mixed_labels = []

                        for id_idx in range(num_ids):
                            start = id_idx * num_instance
                            id_indices = list(range(start, start + num_instance))

                            for i in range(num_instance):
                                # Each sample gets a random partner from same ID
                                partners = [j for j in range(num_instance) if j != i]
                                j = random.choice(partners)
                                idx_i = id_indices[i]
                                idx_j = id_indices[j]

                                # Select random parts to swap (1-indexed, skip global at 0)
                                swap_parts = random.sample(range(1, num_parts + 1), min(stm_num_swap, num_parts))

                                # Create mixed score and feat
                                mixed_score_i = []
                                mixed_feat_i = []
                                for k in range(len(feat)):
                                    if k in swap_parts:
                                        mixed_score_i.append(score[k][idx_j:idx_j+1])
                                        mixed_feat_i.append(feat[k][idx_j:idx_j+1])
                                    else:
                                        mixed_score_i.append(score[k][idx_i:idx_i+1])
                                        mixed_feat_i.append(feat[k][idx_i:idx_i+1])

                                mixed_scores_all.append(mixed_score_i)
                                mixed_feats_all.append(mixed_feat_i)
                                mixed_labels.append(target[idx_i])

                        # Stack mixed samples: same shape as original batch (B samples, num_instance per ID)
                        stm_score = [torch.cat([ms[k] for ms in mixed_scores_all], dim=0) for k in range(len(score))]
                        stm_feat = [torch.cat([mf[k] for mf in mixed_feats_all], dim=0) for k in range(len(feat))]
                        stm_target = torch.stack(mixed_labels)
                        stm_cam = target_cam[:len(stm_target)]

                        # Compute loss on mixed batch (no kp_data — mixed tokens lack kp correspondence)
                        stm_loss = loss_fn(stm_score, stm_feat, stm_target, stm_cam)
                        details = getattr(loss, '_loss_details', {})
                        loss = loss + stm_weight * stm_loss
                        details['stm'] = stm_loss.item()
                        details['stm_n'] = len(mixed_labels)
                        loss._loss_details = details

                # STD-PR: log structural routing stats (token norms, self-attn diagnostics)
                if kp_data is not None and 'str_stats' in kp_data:
                    details = getattr(loss, '_loss_details', {})
                    for k, v in kp_data['str_stats'].items():
                        details[f'str_{k}'] = v
                    loss._loss_details = details

                # VC-Norm: log VCN affine gain/shift magnitudes (collapse check)
                if kp_data is not None and 'vcn_stats' in kp_data:
                    details = getattr(loss, '_loss_details', {})
                    for k, v in kp_data['vcn_stats'].items():
                        details[k] = v
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
                    # Use train() mode so forward returns (score, feat, ...) — same output structure as student
                    # But set BN/Dropout/DropPath to eval to avoid noise and running stats corruption
                    with torch.no_grad():
                        ema_teacher.train()
                        for m in ema_teacher.modules():
                            # BN: use running stats, don't update them
                            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                                m.eval()
                            # Dropout: no random dropping
                            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                                m.eval()
                            # DropPath: set drop_prob=0 temporarily
                            if hasattr(m, 'drop_prob') and not hasattr(m, '_saved_drop_prob'):
                                m._saved_drop_prob = m.drop_prob
                                m.drop_prob = 0.0
                        # Use clean (pre-PLBOA) pose for teacher if available
                        teacher_pose = pose_dict.get('teacher_pose', pose_dict)
                        teacher_out = ema_teacher(img_teacher, label=target,
                                                 cam_label=target_cam,
                                                 view_label=target_view,
                                                 pose_dict=teacher_pose)
                        # Restore DropPath
                        for m in ema_teacher.modules():
                            if hasattr(m, '_saved_drop_prob'):
                                m.drop_prob = m._saved_drop_prob
                                del m._saved_drop_prob
                        ema_teacher.eval()
                        teacher_kp_data = None
                        if len(teacher_out) == 5:
                            _, teacher_feat, _, _, teacher_kp_data = teacher_out
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

                    # BT-PKD: Backbone-Through Per-Keypoint Distillation
                    # Distill per-keypoint features from teacher (clean) to student (occluded)
                    # Student features are NON-detached → gradients flow to backbone
                    bt_pkd_enabled = getattr(cfg.MODEL, 'POSE_BT_PKD', False)
                    if bt_pkd_enabled and kp_data is not None and teacher_kp_data is not None:
                        bt_kp_feats = kp_data.get('bt_kp_feats')        # (B, 17, C) non-detached
                        t_kp_feats = teacher_kp_data.get('kp_feats')    # (B, 17, C) teacher's GCN output
                        t_kp_weights = teacher_kp_data.get('kp_weights')  # (B, 17) teacher confidence
                        if bt_kp_feats is not None and t_kp_feats is not None:
                            bt_pkd_weight = float(getattr(cfg.MODEL, 'POSE_BT_PKD_WEIGHT', 0.01))
                            # Cosine decay: reduce weight to 0 by decay_epoch
                            bt_pkd_decay_ep = int(getattr(cfg.MODEL, 'POSE_BT_PKD_DECAY_EPOCH', 0))
                            if bt_pkd_decay_ep > 0 and epoch > 0:
                                import math
                                if epoch >= bt_pkd_decay_ep:
                                    bt_pkd_weight = 0.0
                                else:
                                    bt_pkd_weight *= 0.5 * (1 + math.cos(math.pi * epoch / bt_pkd_decay_ep))
                            # L2 normalize both for cosine distillation
                            s_norm = F.normalize(bt_kp_feats, p=2, dim=2)    # (B, 17, C)
                            t_norm = F.normalize(t_kp_feats.detach(), p=2, dim=2)  # (B, 17, C)
                            # Per-keypoint cosine distance
                            per_kp_dist = 1.0 - (s_norm * t_norm).sum(dim=2)  # (B, 17)
                            # Weight by teacher keypoint confidence
                            if t_kp_weights is not None:
                                w = t_kp_weights.detach().clamp(min=0.0)  # (B, 17)
                                bt_pkd_loss = (per_kp_dist * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)
                            else:
                                bt_pkd_loss = per_kp_dist.mean(dim=1)
                            bt_pkd_loss = bt_pkd_loss.mean()
                            loss = loss + bt_pkd_weight * bt_pkd_loss
                            details['bt_pkd'] = bt_pkd_loss.item()
                            loss._loss_details = details

                    # VC-Norm: visibility-conditioned per-keypoint statistic alignment.
                    # Align occluded-student GCN-token norm statistics to the clean
                    # teacher's, collapsing the "occluded vs un-occluded" domain axis.
                    # Operates on GCN tokens (gcn_kp_feats), which the probe measured.
                    vcnorm_enabled = getattr(cfg.MODEL, 'POSE_VCNORM', False)
                    vcnorm_warmup = int(getattr(cfg.MODEL, 'POSE_VCNORM_WARMUP', 20))
                    if vcnorm_enabled and epoch > vcnorm_warmup \
                            and kp_data is not None and teacher_kp_data is not None:
                        s_kp = kp_data.get('gcn_kp_feats')
                        s_sc = kp_data.get('gcn_kp_weights')
                        t_kp = teacher_kp_data.get('gcn_kp_feats')
                        t_sc = teacher_kp_data.get('gcn_kp_weights')
                        if s_kp is not None and s_sc is not None \
                                and t_kp is not None and t_sc is not None:
                            from loss.vcnorm_loss import vcnorm_align_loss
                            vcn_weight = float(getattr(cfg.MODEL, 'POSE_VCNORM_WEIGHT', 0.5))
                            vcn_vis_thr = float(getattr(cfg.MODEL, 'POSE_VCNORM_VIS_THR', 0.3))
                            vca_loss, vca_stats = vcnorm_align_loss(
                                s_kp, t_kp, s_sc, t_sc, vis_thr=vcn_vis_thr)
                            loss = loss + vcn_weight * vca_loss
                            details['vca'] = vca_stats['vca_loss']
                            details['vca_md'] = vca_stats['vca_mean_dist']
                            details['vca_sd'] = vca_stats['vca_std_dist']
                            details['vca_vk'] = vca_stats['vca_valid_k']
                            details['vca_or'] = vca_stats['vca_occ_ratio']
                            loss._loss_details = details

                # OA-RD: Occlusion-Asymmetric Relational Distillation
                # Distills pairwise similarity STRUCTURE (not individual features) from teacher to student
                if oa_rd_enabled and (oa_sd_mode or parallel_oa_sd) and use_pose and ema_teacher is not None:
                    oa_rd_weight = float(getattr(cfg.MODEL, 'POSE_OA_RD_WEIGHT', 1.0))
                    oa_rd_temp = float(getattr(cfg.MODEL, 'POSE_OA_RD_TEMP', 0.1))

                    # Get teacher features if not already computed by OA-SD
                    if not oa_sd_enabled:
                        with torch.no_grad():
                            ema_teacher.train()
                            for m in ema_teacher.modules():
                                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                                    m.eval()
                                if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
                                    m.eval()
                                if hasattr(m, 'drop_prob') and not hasattr(m, '_saved_drop_prob'):
                                    m._saved_drop_prob = m.drop_prob
                                    m.drop_prob = 0.0
                            teacher_out = ema_teacher(img_teacher, label=target,
                                                     cam_label=target_cam,
                                                     view_label=target_view,
                                                     pose_dict=pose_dict)
                            for m in ema_teacher.modules():
                                if hasattr(m, '_saved_drop_prob'):
                                    m.drop_prob = m._saved_drop_prob
                                    del m._saved_drop_prob
                            ema_teacher.eval()
                            if len(teacher_out) == 5:
                                _, teacher_feat, _, _, _ = teacher_out
                            elif len(teacher_out) == 4:
                                _, teacher_feat, _, _ = teacher_out
                            else:
                                _, teacher_feat, _ = teacher_out[:3]

                    # Extract global features for relational distillation
                    s_global = feat[0] if isinstance(feat, list) else feat
                    t_global = teacher_feat[0] if isinstance(teacher_feat, list) else teacher_feat

                    # Compute pairwise cosine similarity matrices
                    s_norm = F.normalize(s_global, p=2, dim=1)  # (B, D)
                    t_norm = F.normalize(t_global.detach(), p=2, dim=1)  # (B, D)

                    sim_s = s_norm @ s_norm.t() / oa_rd_temp  # (B, B) student similarity
                    sim_t = t_norm @ t_norm.t() / oa_rd_temp  # (B, B) teacher similarity

                    # KL divergence: match row-normalized distributions
                    log_p_s = F.log_softmax(sim_s, dim=1)
                    p_t = F.softmax(sim_t, dim=1)
                    oa_rd_loss = F.kl_div(log_p_s, p_t, reduction='batchmean')

                    details = getattr(loss, '_loss_details', {})
                    loss = loss + oa_rd_weight * oa_rd_loss
                    details['oa_rd'] = oa_rd_loss.item()
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

                # FSDC: Feature-Space Diffusion Completion reconstruction loss
                fsdc_enabled = getattr(cfg.MODEL, 'POSE_FSDC', False)
                if fsdc_enabled and kp_data is not None and 'fsdc_loss' in kp_data:
                    fsdc_weight = float(getattr(cfg.MODEL, 'POSE_FSDC_WEIGHT', 0.5))
                    fsdc_loss = kp_data['fsdc_loss']
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + fsdc_weight * fsdc_loss
                    details['fsdc_recon'] = fsdc_loss.item()
                    fsdc_stats = kp_data.get('fsdc_stats', {})
                    details['fsdc_mask_ratio'] = fsdc_stats.get('mask_ratio', 0)
                    loss._loss_details = details

                # PPA: Part Assignment loss
                ppa_enabled = getattr(cfg.MODEL, 'POSE_PPA', False)
                if ppa_enabled and kp_data is not None and 'assign_loss' in kp_data:
                    assign_weight = float(getattr(cfg.MODEL, 'POSE_PPA_ASSIGN_WEIGHT', 0.5))
                    assign_loss = kp_data['assign_loss']
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + assign_weight * assign_loss
                    details['ppa_assign'] = assign_loss.item()
                    details['ppa_bg_ratio'] = kp_data.get('bg_ratio', 0)
                    details['ppa_entropy'] = kp_data.get('assign_entropy', 0)
                    loss._loss_details = details

                # LGPA: CLIP Part Assignment loss (same assign_loss key as PPA)
                lgpa_enabled = getattr(cfg.MODEL, 'POSE_LGPA', False)
                if lgpa_enabled and not ppa_enabled and kp_data is not None and 'assign_loss' in kp_data:
                    lgpa_assign_w = float(getattr(cfg.MODEL, 'POSE_LGPA_ASSIGN_WEIGHT', 0.5))
                    lgpa_assign_loss = kp_data['assign_loss']
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + lgpa_assign_w * lgpa_assign_loss
                    details['lgpa_assign'] = lgpa_assign_loss.item()
                    loss._loss_details = details

                # PBSR: pose supervises router parameters through a detached
                # backbone input. The retrieval loss remains the standard
                # single-global ID/triplet path.
                pbsr_enabled = getattr(cfg.MODEL, 'POSE_PBSR', False)
                if pbsr_enabled and kp_data is not None and 'pbsr_route_loss' in kp_data:
                    pbsr_route_w = float(getattr(
                        cfg.MODEL, 'POSE_PBSR_ROUTE_WEIGHT', 0.5))
                    pbsr_route_loss = kp_data['pbsr_route_loss']
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + pbsr_route_w * pbsr_route_loss
                    details['pbsr_route'] = pbsr_route_loss.item()
                    pbsr_stats = kp_data.get('pbsr_stats', {})
                    details['pbsr_alpha'] = pbsr_stats.get('write_scale', 0)
                    details['pbsr_entropy'] = pbsr_stats.get('route_entropy', 0)
                    details['pbsr_bg'] = pbsr_stats.get('background_share', 0)
                    details['pbsr_delta'] = pbsr_stats.get('delta_norm', 0)
                    loss._loss_details = details

                # VCSR: Visibility-Conditional assignment loss + diagnostics
                vcsr_enabled = getattr(cfg.MODEL, 'POSE_VCSR', False)
                if vcsr_enabled and kp_data is not None and 'assign_loss' in kp_data:
                    vcsr_assign_w = float(getattr(cfg.MODEL, 'POSE_VCSR_ASSIGN_WEIGHT', 0.5))
                    vcsr_assign_loss = kp_data['assign_loss']
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + vcsr_assign_w * vcsr_assign_loss
                    details['vcsr_assign'] = vcsr_assign_loss.item()
                    details['vcsr_n_active'] = kp_data.get('n_active', 0)
                    loss._loss_details = details

                # PKC: Per-Keypoint Contrastive loss on GCN keypoint features
                pkc_enabled = getattr(cfg.MODEL, 'POSE_PKC', False)
                if pkc_enabled and kp_data is not None and 'kp_feats' in kp_data:
                    pkc_weight = float(getattr(cfg.MODEL, 'POSE_PKC_WEIGHT', 0.5))
                    pkc_temp = float(getattr(cfg.MODEL, 'POSE_PKC_TEMP', 0.07))
                    pkc_vis_thr = float(getattr(cfg.MODEL, 'POSE_PKC_VIS_THR', 0.3))

                    kp_f = kp_data['kp_feats']     # (B, 17, C)
                    kp_w = kp_data['kp_weights']    # (B, 17)
                    B_kp, K_kp, C_kp = kp_f.shape

                    # Lazy-init SupCon for PKC
                    if not hasattr(do_train, '_pkc_supcon'):
                        from loss.supcon_loss import SupConLoss
                        do_train._pkc_supcon = SupConLoss(temperature=pkc_temp)

                    pkc_losses = []
                    for k_idx in range(K_kp):
                        # Visibility mask for this keypoint
                        vis_mask = kp_w[:, k_idx] > pkc_vis_thr  # (B,)
                        n_vis = vis_mask.sum().item()
                        if n_vis < 4:  # need at least 4 samples for SupCon
                            continue
                        feat_k = kp_f[vis_mask, k_idx, :]  # (n_vis, C)
                        label_k = target[vis_mask]
                        # Need at least 2 different IDs
                        if label_k.unique().shape[0] < 2:
                            continue
                        pkc_loss_k = do_train._pkc_supcon(feat_k, label_k)
                        pkc_losses.append(pkc_loss_k)

                    if pkc_losses:
                        pkc_loss = sum(pkc_losses) / len(pkc_losses)
                        details = getattr(loss, '_loss_details', {})
                        loss = loss + pkc_weight * pkc_loss
                        details['pkc'] = pkc_loss.item()
                        details['pkc_nk'] = len(pkc_losses)
                        loss._loss_details = details

                # OERL: Occlusion-Equivariant Representation Learning — Part Occlusion Invariance
                oerl_enabled = getattr(cfg.MODEL, 'POSE_OERL', False)
                if oerl_enabled and use_pose and feat_maps is not None and kp_data is not None:
                    oerl_weight = float(getattr(cfg.MODEL, 'POSE_OERL_WEIGHT', 1.0))
                    oerl_occ_ratio = float(getattr(cfg.MODEL, 'POSE_OERL_OCC_RATIO', 0.5))

                    # Feature-map-level Part Occlusion Invariance
                    # 1. Sample keypoint features from the original (non-detached) feature map
                    # 2. Create a randomly occluded version by masking feature map with pose heatmaps
                    # 3. Re-sample from occluded feature map
                    # 4. Align non-occluded keypoints

                    fm = feat_maps[-1]  # (B, C, fH, fW) — NON-detached backbone output
                    B_oerl, C_oerl, fH, fW = fm.shape
                    kp_coords = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2)
                    kp_scores = pose_dict['scores'][:, 0, :]  # (B, 17)
                    heatmaps = pose_dict['heatmaps'][:, 0, :, :, :]  # (B, 17, hH, hW) person 0
                    input_h, input_w = img.shape[2], img.shape[3]

                    # Sample clean keypoint features
                    grid_x = (kp_coords[:, :, 0] / input_w * 2 - 1).clamp(-1, 1)
                    grid_y = (kp_coords[:, :, 1] / input_h * 2 - 1).clamp(-1, 1)
                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, 17, 1, 2)
                    clean_kp = F.grid_sample(fm, grid, mode='bilinear',
                                             padding_mode='border', align_corners=True
                                             ).squeeze(-1).permute(0, 2, 1)  # (B, 17, C)

                    # Random per-sample occlusion: select keypoints to "occlude"
                    num_occ = max(3, int(17 * oerl_occ_ratio))
                    occ_mask = torch.zeros(B_oerl, 17, dtype=torch.bool, device=fm.device)
                    for b in range(B_oerl):
                        occ_idx = torch.randperm(17, device=fm.device)[:num_occ]
                        occ_mask[b, occ_idx] = True

                    # Create spatial occlusion mask from selected keypoints' heatmaps
                    hm_resized = F.interpolate(heatmaps, size=(fH, fW),
                                               mode='bilinear', align_corners=False)
                    hm_resized = F.relu(hm_resized)  # (B, 17, fH, fW)
                    # Aggregate heatmaps of occluded keypoints into a single occlusion map
                    occ_hm = (hm_resized * occ_mask.float().unsqueeze(2).unsqueeze(3)).max(dim=1)[0]  # (B, fH, fW)
                    # Normalize to [0, 1] and invert: 1 = keep, 0 = occlude
                    occ_max = occ_hm.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
                    spatial_mask = 1.0 - (occ_hm / occ_max).clamp(0, 1)  # (B, fH, fW)

                    # Apply spatial mask to feature map (soft occlusion)
                    fm_occluded = fm * spatial_mask.unsqueeze(1)  # (B, C, fH, fW)

                    # Sample keypoint features from occluded feature map
                    occ_kp = F.grid_sample(fm_occluded, grid, mode='bilinear',
                                           padding_mode='border', align_corners=True
                                           ).squeeze(-1).permute(0, 2, 1)  # (B, 17, C)

                    # Visible = NOT occluded AND has valid keypoint score
                    visible = (~occ_mask) & (kp_scores > 0.3)  # (B, 17)

                    if visible.any():
                        occ_norm = F.normalize(occ_kp, p=2, dim=2)
                        clean_norm = F.normalize(clean_kp, p=2, dim=2)
                        cos_sim = (occ_norm * clean_norm).sum(dim=2)  # (B, 17)
                        poi_loss = (1.0 - cos_sim[visible]).mean()

                        details = getattr(loss, '_loss_details', {})
                        loss = loss + oerl_weight * poi_loss
                        details['oerl'] = poi_loss.item()
                        details['oerl_nv'] = visible.float().sum().item() / visible.shape[0]
                        loss._loss_details = details

                # BA-PKC: Backbone-Aware Per-Keypoint Contrastive
                # Uses NON-detached keypoint features → gradients flow to backbone!
                ba_pkc_enabled = getattr(cfg.MODEL, 'POSE_BA_PKC', False)
                if ba_pkc_enabled and kp_data is not None and 'ba_kp_feats' in kp_data:
                    ba_pkc_weight = float(getattr(cfg.MODEL, 'POSE_BA_PKC_WEIGHT', 0.1))
                    ba_vis_thr = float(getattr(cfg.MODEL, 'POSE_PKC_VIS_THR', 0.3))

                    ba_kp_f = kp_data['ba_kp_feats']  # (B, 17, C) — NOT detached!
                    ba_kp_w = kp_data['kp_weights']    # (B, 17) — visibility weights
                    B_ba, K_ba, C_ba = ba_kp_f.shape

                    if not hasattr(do_train, '_ba_pkc_supcon'):
                        from loss.supcon_loss import SupConLoss
                        do_train._ba_pkc_supcon = SupConLoss(temperature=0.07)

                    ba_losses = []
                    for k_idx in range(K_ba):
                        vis_mask = ba_kp_w[:, k_idx] > ba_vis_thr
                        if vis_mask.sum().item() < 4:
                            continue
                        feat_k = ba_kp_f[vis_mask, k_idx, :]
                        label_k = target[vis_mask]
                        if label_k.unique().shape[0] < 2:
                            continue
                        ba_loss_k = do_train._ba_pkc_supcon(feat_k, label_k)
                        ba_losses.append(ba_loss_k)

                    if ba_losses:
                        ba_pkc_loss = sum(ba_losses) / len(ba_losses)
                        details = getattr(loss, '_loss_details', {})
                        loss = loss + ba_pkc_weight * ba_pkc_loss
                        details['ba_pkc'] = ba_pkc_loss.item()
                        details['ba_nk'] = len(ba_losses)
                        loss._loss_details = details

                # MST: MaxSim Triplet loss — directly optimize per-keypoint features for MaxSim
                mst_enabled = getattr(cfg.MODEL, 'POSE_MST', False)
                if mst_enabled and kp_data is not None and 'kp_feats' in kp_data:
                    mst_weight = float(getattr(cfg.MODEL, 'POSE_MST_WEIGHT', 0.5))
                    mst_margin = float(getattr(cfg.MODEL, 'POSE_MST_MARGIN', 0.3))
                    mst_vis_thr = float(getattr(cfg.MODEL, 'POSE_MST_VIS_THR', 0.3))

                    kp_f = kp_data['kp_feats']     # (B, 17, C)
                    kp_w = kp_data['kp_weights']    # (B, 17)
                    B_kp, K_kp, C_kp = kp_f.shape

                    # L2 normalize keypoint features
                    kp_fn = F.normalize(kp_f, p=2, dim=2)  # (B, 17, C)

                    # Compute pairwise MaxSim distance matrix: (B, B)
                    # For each pair (i,j): sim = mean_k max_l cos(kp_i_k, kp_j_l), weighted by visibility
                    # Efficient: sim_all = einsum('bkd,cjd->bkcj', kp_fn, kp_fn)  # (B, B, 17, 17)
                    # maxsim_per_k = sim_all.max(dim=3)[0]  # (B, B, 17) — best match for each query kp
                    # Weight by visibility and average
                    w_mask = (kp_w > mst_vis_thr).float()  # (B, 17)
                    w_eff = kp_w * w_mask  # (B, 17)
                    w_sum = w_eff.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)

                    # Chunk to avoid OOM on large B
                    chunk = min(B_kp, 32)
                    sim_chunks = []
                    for i in range(0, B_kp, chunk):
                        ie = min(i + chunk, B_kp)
                        q = kp_fn[i:ie]  # (c, 17, C)
                        # sim: (c, 17_q, B_g, 17_g) — cosine similarity between all kp pairs
                        s = torch.einsum('qkd,gjd->qkgj', q, kp_fn)  # (c, 17, B, 17)
                        # For each query kp, find best matching gallery kp
                        ms = s.max(dim=3)[0]  # (c, 17, B) — max over gallery kps
                        ms = ms.permute(0, 2, 1)  # (c, B, 17) — per query-gallery pair
                        # Weighted average over query keypoints
                        w_q = w_eff[i:ie]  # (c, 17)
                        w_s = w_q.sum(dim=1, keepdim=True).clamp(min=1.0)  # (c, 1)
                        sim = (ms * w_q.unsqueeze(1)).sum(dim=2) / w_s  # (c, B)
                        sim_chunks.append(sim)
                    maxsim_mat = torch.cat(sim_chunks, dim=0)  # (B, B)
                    dist_mat = 1.0 - maxsim_mat  # (B, B)

                    # Hard triplet mining
                    label_eq = (target.unsqueeze(0) == target.unsqueeze(1))
                    self_mask = ~torch.eye(B_kp, dtype=torch.bool, device=dist_mat.device)
                    pos_mask = label_eq & self_mask
                    neg_mask = ~label_eq

                    # Hardest positive: max distance among same-ID
                    pos_dist = dist_mat.clone()
                    pos_dist[~pos_mask] = -1.0
                    hardest_pos, _ = pos_dist.max(dim=1)  # (B,)

                    # Hardest negative: min distance among diff-ID
                    neg_dist = dist_mat.clone()
                    neg_dist[~neg_mask] = 1e6
                    hardest_neg, _ = neg_dist.min(dim=1)  # (B,)

                    # Only samples with valid positives
                    has_pos = pos_mask.any(dim=1)
                    if has_pos.any():
                        mst_loss = F.relu(hardest_pos[has_pos] - hardest_neg[has_pos] + mst_margin).mean()
                        details = getattr(loss, '_loss_details', {})
                        loss = loss + mst_weight * mst_loss
                        details['mst'] = mst_loss.item()
                        loss._loss_details = details

                # PACI: Part-Prototype Consistency Loss
                paci_warmup = int(getattr(cfg.MODEL, 'POSE_PACI_WARMUP', 5))
                if paci_enabled and paci_bank is not None and kp_data is not None \
                        and 'kp_feats' in kp_data and epoch > paci_warmup:
                    paci_weight = float(getattr(cfg.MODEL, 'POSE_PACI_WEIGHT', 0.5))
                    paci_margin = float(getattr(cfg.MODEL, 'POSE_PACI_MARGIN', 0.3))

                    kp_f = kp_data['kp_feats']     # (B, 17, C) detached GCN features
                    kp_w = kp_data['kp_weights']    # (B, 17)

                    # Get prototypes for current batch's identities
                    protos, proto_valid = paci_bank.get_prototypes(target)  # (B, 17, C), (B, 17)

                    # Get negative prototypes (random different IDs)
                    neg_protos = paci_bank.get_negative_prototypes(target, num_neg=4)

                    # Visible AND prototype initialized
                    can_use = (kp_w > 0.3) & proto_valid  # (B, 17)

                    if can_use.any() and neg_protos is not None:
                        kp_norm = F.normalize(kp_f, p=2, dim=2)
                        pos_norm = F.normalize(protos.detach(), p=2, dim=2)

                        # Positive: cosine similarity to own prototype
                        pos_sim = (kp_norm * pos_norm).sum(dim=2)  # (B, 17)

                        # Negative: cosine similarity to random other ID's prototype
                        # Use hardest negative (max similarity among neg IDs)
                        neg_norm = F.normalize(neg_protos.detach(), p=2, dim=3)  # (B, 4, 17, C)
                        neg_sim = torch.einsum('bkc,bnkc->bnk', kp_norm, neg_norm)  # (B, 4, 17)
                        hard_neg_sim = neg_sim.max(dim=1)[0]  # (B, 17)

                        # Triplet loss on visible+valid parts
                        paci_loss_per = F.relu(paci_margin - pos_sim + hard_neg_sim)  # (B, 17)
                        paci_loss = paci_loss_per[can_use].mean()

                        details = getattr(loss, '_loss_details', {})
                        loss = loss + paci_weight * paci_loss
                        details['paci'] = paci_loss.item()
                        details['paci_pos'] = pos_sim[can_use].mean().item()
                        details['paci_neg'] = hard_neg_sim[can_use].mean().item()
                        loss._loss_details = details

            if kp_data is not None and kp_data.get('clip_id_loss') is not None:
                clip_id_w = float(getattr(cfg.MODEL, 'POSE_CLIP_ID_WEIGHT', 1.0))
                details = getattr(loss, '_loss_details', {})
                loss = loss + clip_id_w * kp_data['clip_id_loss']
                details['clip_id'] = kp_data['clip_id_loss'].item()
                loss._loss_details = details

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # PACI: update prototype bank (after optimizer step, with detached features)
            if paci_enabled and paci_bank is not None and kp_data is not None \
                    and 'kp_feats' in kp_data:
                paci_bank.update(
                    kp_data['kp_feats'].detach(),
                    kp_data['kp_weights'].detach(),
                    target)

            # OA-SD: update EMA teacher after optimizer step (params + buffers)
            if ema_teacher is not None:
                base_model = model.module if hasattr(model, 'module') else model
                with torch.no_grad():
                    for t_param, s_param in zip(ema_teacher.parameters(), base_model.parameters()):
                        t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1.0 - ema_decay)
                    # Also EMA-update BN running stats (buffers)
                    for (t_name, t_buf), (s_name, s_buf) in zip(
                            ema_teacher.named_buffers(), base_model.named_buffers()):
                        if t_buf.dtype == torch.float32 and t_buf.shape == s_buf.shape:
                            t_buf.data.mul_(ema_decay).add_(s_buf.data, alpha=1.0 - ema_decay)

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
                        log_msg += _pose_hyper_lora_log(model)
                        logger.info(log_msg)
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    detail_str = ' '.join(f'{k}: {m.avg:.3f}' for k, m in detail_meters.items() if k != 'total')
                    log_msg = "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr)
                    if detail_str:
                        log_msg += f" | {detail_str}"
                    log_msg += _pose_hyper_lora_log(model)
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
            torch.cuda.empty_cache()  # Free training-reserved memory before eval
            flip_test = getattr(cfg.TEST, 'FLIP_TEST', True)
            if epoch == eval_period and not cfg.MODEL.DIST_TRAIN:
                logger.info("Eval protocol: flip_test={} feat_norm={}".format(
                    flip_test, cfg.TEST.FEAT_NORM))
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
                            feat = _extract_feat_flip(
                                model, img, pose_dict, camids, target_view,
                                use_pose, flip_test)
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
                        feat = _extract_feat_flip(
                            model, img, pose_dict, camids, target_view,
                            use_pose, flip_test)
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
    flip_test = getattr(cfg.TEST, 'FLIP_TEST', True)
    if flip_test:
        logger.info("Flip-test TTA: ON (cfg.TEST.FLIP_TEST)")

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
            feat = _extract_feat_flip(
                model, img, pose_dict, camids, target_view, use_pose, flip_test)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
