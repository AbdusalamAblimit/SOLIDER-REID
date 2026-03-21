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
from model.modules.pamc import pamc_consistency_loss
from model.modules.support_complete_bank import SupportCompleteBank
from model.modules.pair_adaptive_fusion import (
    build_pair_descriptors,
    common_support_distance,
    euclidean_distance_tensor,
)


def _pose_to_device(pose_dict, device):
    """Move all tensors in pose_dict to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in pose_dict.items()}


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
    pcra_alpha = getattr(cfg.MODEL, 'POSE_PCRA_ALPHA', 0.0)
    kp_triplet_enabled = getattr(cfg.MODEL, 'POSE_KP_TRIPLET', False)
    kp_triplet_weight = getattr(cfg.MODEL, 'POSE_KP_TRIPLET_WEIGHT', 1.0)
    csgt_enabled = getattr(cfg.MODEL, 'POSE_CSGT', False)
    csrd_enabled = getattr(cfg.MODEL, 'POSE_CSRD', False)
    pamc_enabled = getattr(cfg.MODEL, 'POSE_PAMC', False)
    pamc_weight = getattr(cfg.MODEL, 'POSE_PAMC_WEIGHT', 0.5)
    pamc_warmup = getattr(cfg.MODEL, 'POSE_PAMC_WARMUP', 10)

    # LSRM: Learned Skeleton Recovery Module (lives inside model, proper optimizer/scheduler)
    lsrm_enabled = getattr(cfg.MODEL, 'POSE_LSRM', False)
    lsrm_weight = getattr(cfg.MODEL, 'POSE_LSRM_WEIGHT', 0.5) if lsrm_enabled else 0
    lsrm_warmup = 20  # Don't train LSRM before GCN features are meaningful

    # PCQA: Pose Translation Module (inside model)
    ptm_enabled = getattr(cfg.MODEL, 'POSE_TRANSLATION', False)
    ptm_weight = getattr(cfg.MODEL, 'POSE_TRANSLATION_WEIGHT', 0.5) if ptm_enabled else 0
    ptm_warmup = 20
    ptm_norm = getattr(cfg.MODEL, 'POSE_TRANSLATION_NORM', False)

    # PAMN: Pose-Aware Matching Network
    pamn_enabled = getattr(cfg.MODEL, 'POSE_MATCHING_NETWORK', False)
    pamn_module = None
    if pamn_enabled:
        from model.modules.pose_matching_network import PoseMatchingNetwork
        pamn_weight = getattr(cfg.MODEL, 'POSE_MATCHING_NETWORK_WEIGHT', 0.5)
        pamn_module = PoseMatchingNetwork(num_keypoints=17, feat_dim=768).to(device)
        # Add PAMN params to optimizer
        pamn_params = [{'params': pamn_module.parameters(), 'lr': cfg.SOLVER.BASE_LR}]
        for pg in pamn_params:
            optimizer.add_param_group(pg)

    # Momentum Memory Contrastive Learning
    mm_enabled = getattr(cfg.MODEL, 'POSE_MOMENTUM_MEMORY', False)
    mm_memory = None
    if mm_enabled:
        from model.modules.momentum_memory import MomentumMemory
        mm_weight = getattr(cfg.MODEL, 'POSE_MOMENTUM_MEMORY_WEIGHT', 0.5)
        mm_temp = getattr(cfg.MODEL, 'POSE_MOMENTUM_MEMORY_TEMP', 0.05)
        mm_mom = getattr(cfg.MODEL, 'POSE_MOMENTUM_MEMORY_MOM', 0.1)
        num_train_classes = len(set([d[1] for d in train_loader.dataset.dataset]))
        mm_memory = MomentumMemory(
            feat_dim=768, num_classes=num_train_classes,
            momentum=mm_mom, temp=mm_temp).to(device)

    sckd_enabled = getattr(cfg.MODEL, 'POSE_SCKD', False)
    scfr_enabled = getattr(cfg.MODEL, 'POSE_SCFR', False)
    scrc_enabled = getattr(cfg.MODEL, 'POSE_SCRC', False)
    if (scfr_enabled or scrc_enabled) and not sckd_enabled:
        raise ValueError('POSE_SCFR/POSE_SCRC require POSE_SCKD=True to create the support bank')
    if scfr_enabled and scrc_enabled:
        raise ValueError('POSE_SCFR and POSE_SCRC cannot be enabled together')
    sckd_bank = None
    if sckd_enabled:
        sckd_weight = getattr(cfg.MODEL, 'POSE_SCKD_WEIGHT', 0.5)
        sckd_warmup = getattr(cfg.MODEL, 'POSE_SCKD_WARMUP', 20)
        sckd_low_thr = getattr(cfg.MODEL, 'POSE_SCKD_LOW_THR', 0.3)
        sckd_update_thr = getattr(cfg.MODEL, 'POSE_SCKD_UPDATE_THR', 0.5)
        sckd_mom = getattr(cfg.MODEL, 'POSE_SCKD_MOM', 0.9)
        sckd_min_count = getattr(cfg.MODEL, 'POSE_SCKD_MIN_COUNT', 1)
        sckd_update_stop_epoch = getattr(cfg.MODEL, 'POSE_SCKD_UPDATE_STOP_EPOCH', -1)
        num_train_classes = len(set([d[1] for d in train_loader.dataset.dataset]))
        sckd_bank = SupportCompleteBank(
            num_classes=num_train_classes,
            feat_dim=768,
            num_keypoints=17,
            low_thr=sckd_low_thr,
            update_thr=sckd_update_thr,
            momentum=sckd_mom,
            min_count=sckd_min_count,
        ).to(device)

    csrd_support_teacher = getattr(cfg.MODEL, 'POSE_CSRD_SUPPORT_TEACHER', False)
    csrd_teacher_bank = None
    csrd_target_mode = getattr(cfg.MODEL, 'POSE_CSRD_TARGET_MODE', 'full')
    csrd_anchor_weight_mode = getattr(cfg.MODEL, 'POSE_CSRD_ANCHOR_WEIGHT_MODE', 'none')
    csrd_pair_weight_mode = getattr(cfg.MODEL, 'POSE_CSRD_PAIR_WEIGHT_MODE', 'none')
    csrd_queue_size = getattr(cfg.MODEL, 'POSE_CSRD_QUEUE_SIZE', 0)
    if csrd_target_mode not in ('full', 'residual', 'residual_kl'):
        raise ValueError(f"Unsupported POSE_CSRD_TARGET_MODE: {csrd_target_mode}")
    if csrd_anchor_weight_mode not in ('none', 'replace_ratio', 'low_ratio'):
        raise ValueError(f"Unsupported POSE_CSRD_ANCHOR_WEIGHT_MODE: {csrd_anchor_weight_mode}")
    if csrd_pair_weight_mode not in ('none', 'delta', 'delta_top', 'delta_top_exact'):
        raise ValueError(f"Unsupported POSE_CSRD_PAIR_WEIGHT_MODE: {csrd_pair_weight_mode}")
    if csrd_anchor_weight_mode != 'none' and not csrd_support_teacher:
        raise ValueError('POSE_CSRD_ANCHOR_WEIGHT_MODE requires POSE_CSRD_SUPPORT_TEACHER=True')
    if csrd_pair_weight_mode != 'none' and not csrd_support_teacher:
        raise ValueError('POSE_CSRD_PAIR_WEIGHT_MODE requires POSE_CSRD_SUPPORT_TEACHER=True')
    if csrd_target_mode == 'residual' and not csrd_support_teacher:
        raise ValueError('POSE_CSRD_TARGET_MODE=residual requires POSE_CSRD_SUPPORT_TEACHER=True')
    if csrd_target_mode == 'residual_kl' and not csrd_support_teacher:
        raise ValueError('POSE_CSRD_TARGET_MODE=residual_kl requires POSE_CSRD_SUPPORT_TEACHER=True')
    if csrd_enabled and csrd_support_teacher:
        csrd_st_low_thr = getattr(cfg.MODEL, 'POSE_CSRD_ST_LOW_THR', 0.3)
        csrd_st_update_thr = getattr(cfg.MODEL, 'POSE_CSRD_ST_UPDATE_THR', 0.7)
        csrd_st_mom = getattr(cfg.MODEL, 'POSE_CSRD_ST_MOM', 0.9)
        csrd_st_min_count = getattr(cfg.MODEL, 'POSE_CSRD_ST_MIN_COUNT', 1)
        csrd_st_update_stop_epoch = getattr(cfg.MODEL, 'POSE_CSRD_ST_UPDATE_STOP_EPOCH', -1)
        num_train_classes = len(set([d[1] for d in train_loader.dataset.dataset]))
        csrd_teacher_bank = SupportCompleteBank(
            num_classes=num_train_classes,
            feat_dim=768,
            num_keypoints=17,
            low_thr=csrd_st_low_thr,
            update_thr=csrd_st_update_thr,
            momentum=csrd_st_mom,
            min_count=csrd_st_min_count,
        ).to(device)

    ltcs_enabled = getattr(cfg.MODEL, 'POSE_LTCS', False)
    lpcs_enabled = getattr(cfg.MODEL, 'POSE_LPCS', False)
    if ltcs_enabled and lpcs_enabled:
        raise ValueError('POSE_LTCS and POSE_LPCS cannot be enabled together')
    ltcs_teacher_bank = None
    if ltcs_enabled:
        ltcs_weight = getattr(cfg.MODEL, 'POSE_LTCS_WEIGHT', 0.5)
        ltcs_warmup = getattr(cfg.MODEL, 'POSE_LTCS_WARMUP', 20)
        ltcs_hidden = getattr(cfg.MODEL, 'POSE_LTCS_HIDDEN', 32)
        ltcs_st_low_thr = getattr(cfg.MODEL, 'POSE_LTCS_ST_LOW_THR', 0.3)
        ltcs_st_update_thr = getattr(cfg.MODEL, 'POSE_LTCS_ST_UPDATE_THR', 0.7)
        ltcs_st_mom = getattr(cfg.MODEL, 'POSE_LTCS_ST_MOM', 0.9)
        ltcs_st_min_count = getattr(cfg.MODEL, 'POSE_LTCS_ST_MIN_COUNT', 1)
        ltcs_st_update_stop_epoch = getattr(cfg.MODEL, 'POSE_LTCS_ST_UPDATE_STOP_EPOCH', -1)
        num_train_classes = len(set([d[1] for d in train_loader.dataset.dataset]))
        ltcs_teacher_bank = SupportCompleteBank(
            num_classes=num_train_classes,
            feat_dim=768,
            num_keypoints=17,
            low_thr=ltcs_st_low_thr,
            update_thr=ltcs_st_update_thr,
            momentum=ltcs_st_mom,
            min_count=ltcs_st_min_count,
        ).to(device)
    lpcs_teacher_bank = None
    if lpcs_enabled:
        lpcs_weight = getattr(cfg.MODEL, 'POSE_LPCS_WEIGHT', 0.5)
        lpcs_warmup = getattr(cfg.MODEL, 'POSE_LPCS_WARMUP', 20)
        lpcs_hidden = getattr(cfg.MODEL, 'POSE_LPCS_HIDDEN', 32)
        lpcs_delta_scale = getattr(cfg.MODEL, 'POSE_LPCS_DELTA_SCALE', 0.5)
        lpcs_st_low_thr = getattr(cfg.MODEL, 'POSE_LPCS_ST_LOW_THR', 0.3)
        lpcs_st_update_thr = getattr(cfg.MODEL, 'POSE_LPCS_ST_UPDATE_THR', 0.7)
        lpcs_st_mom = getattr(cfg.MODEL, 'POSE_LPCS_ST_MOM', 0.9)
        lpcs_st_min_count = getattr(cfg.MODEL, 'POSE_LPCS_ST_MIN_COUNT', 1)
        lpcs_st_update_stop_epoch = getattr(cfg.MODEL, 'POSE_LPCS_ST_UPDATE_STOP_EPOCH', -1)
        num_train_classes = len(set([d[1] for d in train_loader.dataset.dataset]))
        lpcs_teacher_bank = SupportCompleteBank(
            num_classes=num_train_classes,
            feat_dim=768,
            num_keypoints=17,
            low_thr=lpcs_st_low_thr,
            update_thr=lpcs_st_update_thr,
            momentum=lpcs_st_mom,
            min_count=lpcs_st_min_count,
        ).to(device)

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    if use_pose:
        logger.info('Pose-guided training ENABLED')
    if mm_enabled:
        logger.info(f'Momentum Memory enabled: weight={mm_weight}, temp={mm_temp}, mom={mm_mom}')
    if sckd_enabled:
        logger.info(f'[SCKD] enabled: weight={sckd_weight}, warmup={sckd_warmup}, '
                    f'low_thr={sckd_low_thr}, update_thr={sckd_update_thr}, '
                    f'mom={sckd_mom}, stop_epoch={sckd_update_stop_epoch}')
    if csrd_enabled and csrd_support_teacher:
        logger.info(f'[CSRD-ST] enabled: low_thr={csrd_st_low_thr}, '
                    f'update_thr={csrd_st_update_thr}, mom={csrd_st_mom}, '
                    f'min_count={csrd_st_min_count}, stop_epoch={csrd_st_update_stop_epoch}')
    if csrd_enabled:
        logger.info(f'[CSRD-TARGET] mode={csrd_target_mode}')
    if csrd_enabled and csrd_pair_weight_mode != 'none':
        csrd_pair_weight_alpha = getattr(cfg.MODEL, 'POSE_CSRD_PAIR_WEIGHT_ALPHA', 1.0)
        if csrd_pair_weight_mode in ('delta_top', 'delta_top_exact'):
            csrd_pair_top_ratio = getattr(cfg.MODEL, 'POSE_CSRD_PAIR_TOP_RATIO', 0.25)
            logger.info(f'[CSRD-PW] mode={csrd_pair_weight_mode}, '
                        f'alpha={csrd_pair_weight_alpha}, top_ratio={csrd_pair_top_ratio}')
        else:
            logger.info(f'[CSRD-PW] mode={csrd_pair_weight_mode}, '
                        f'alpha={csrd_pair_weight_alpha}')
    if csrd_enabled and csrd_queue_size > 0:
        logger.info(f'[CSRD-QUEUE] size={csrd_queue_size}')
    if ltcs_enabled:
        logger.info(f'[LTCS] enabled: weight={ltcs_weight}, warmup={ltcs_warmup}, '
                    f'hidden={ltcs_hidden}, low_thr={ltcs_st_low_thr}, '
                    f'update_thr={ltcs_st_update_thr}, mom={ltcs_st_mom}, '
                    f'min_count={ltcs_st_min_count}, stop_epoch={ltcs_st_update_stop_epoch}')
    if lpcs_enabled:
        logger.info(f'[LPCS] enabled: weight={lpcs_weight}, warmup={lpcs_warmup}, '
                    f'hidden={lpcs_hidden}, delta_scale={lpcs_delta_scale}, '
                    f'low_thr={lpcs_st_low_thr}, update_thr={lpcs_st_update_thr}, '
                    f'mom={lpcs_st_mom}, min_count={lpcs_st_min_count}, '
                    f'stop_epoch={lpcs_st_update_stop_epoch}')
    if scfr_enabled:
        logger.info(f'[SCFR] Feature replacement mode enabled (loss disabled, bank replaces features)')
    if scrc_enabled:
        scrc_hidden = getattr(cfg.MODEL, 'POSE_SCRC_HIDDEN', 128)
        logger.info(f'[SCRC] Residual completion mode enabled: hidden={scrc_hidden} '
                    f'(loss disabled, bank fuses support prior into low-vis keypoints)')
    if pamc_enabled:
        logger.info(f'[PAMC] Pose-Aware Masking Consistency: weight={pamc_weight}, warmup={pamc_warmup}')
    if pcra_alpha > 0:
        logger.info(f'[PCRA] Pose-Contrastive Representation Alignment: alpha={pcra_alpha}')

    csrd_queue = None
    if csrd_enabled and csrd_queue_size > 0:
        csrd_queue = {
            'student_feat': None,
            'kp_feats': None,
            'kp_weights': None,
            'teacher_kp_feats': None,
            'labels': None,
        }

    def _get_csrd_queue_payload():
        if csrd_queue is None or csrd_queue['labels'] is None:
            return None
        if csrd_queue['labels'].numel() == 0:
            return None
        return {k: v.detach() for k, v in csrd_queue.items()}

    def _enqueue_csrd_queue(student_feat, kp_feats, kp_weights, teacher_kp_feats, labels):
        if csrd_queue is None:
            return
        new_items = {
            'student_feat': student_feat.detach(),
            'kp_feats': kp_feats.detach(),
            'kp_weights': kp_weights.detach(),
            'teacher_kp_feats': teacher_kp_feats.detach(),
            'labels': labels.detach(),
        }
        for key, value in new_items.items():
            if csrd_queue[key] is None:
                csrd_queue[key] = value
            else:
                csrd_queue[key] = torch.cat([csrd_queue[key], value], dim=0)
                if csrd_queue[key].size(0) > csrd_queue_size:
                    csrd_queue[key] = csrd_queue[key][-csrd_queue_size:]

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
        feat_g = F.normalize(global_feat.detach(), dim=-1)
        kp_base = F.normalize(kp_feats.detach(), dim=-1)
        kp_teacher = F.normalize(teacher_kp_feats.detach(), dim=-1)
        weights = kp_weights.detach().clamp(min=0.0)

        global_dist = euclidean_distance_tensor(feat_g, feat_g)
        kp_dist, support_ratio = common_support_distance(
            kp_base, kp_base, weights, weights, fallback=global_dist, return_ratio=True)
        teacher_kp_dist, _ = common_support_distance(
            kp_teacher, kp_teacher, weights, weights, fallback=global_dist, return_ratio=True)

        base_dist = 0.5 * (global_dist + kp_dist)
        teacher_dist = 0.5 * (global_dist + teacher_kp_dist)

        batch_size = global_feat.size(0)
        eye = torch.eye(batch_size, dtype=torch.bool, device=global_feat.device)
        q_vis_mean = weights.mean(dim=1, keepdim=True).expand(-1, batch_size)
        g_vis_mean = q_vis_mean.t()
        desc = build_pair_descriptors(
            global_dist, kp_dist, support_ratio, q_vis_mean, g_vis_mean)
        delta = lpcs_head(desc.view(-1, desc.shape[-1])).view(batch_size, batch_size)
        final_dist = base_dist + delta

        same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        pos_mask = same_label & ~eye
        neg_mask = ~same_label
        pair_change = (teacher_dist.detach() - base_dist.detach()).abs()
        pair_weight = pair_change / pair_change[~eye].mean().clamp(min=1e-6)

        total_loss = torch.tensor(0.0, device=global_feat.device)
        total_weight = torch.tensor(0.0, device=global_feat.device)
        for idx in range(batch_size):
            pos = final_dist[idx][pos_mask[idx]]
            neg = final_dist[idx][neg_mask[idx]]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            pos_w = pair_weight[idx][pos_mask[idx]]
            neg_w = pair_weight[idx][neg_mask[idx]]
            rank_term = F.softplus(pos.unsqueeze(1) - neg.unsqueeze(0))
            rank_weight = torch.sqrt(pos_w.unsqueeze(1) * neg_w.unsqueeze(0))
            total_loss = total_loss + (rank_term * rank_weight).sum()
            total_weight = total_weight + rank_weight.sum()
        loss = total_loss / total_weight.clamp(min=1e-6)

        if pos_mask.any() and neg_mask.any():
            base_gap = float(base_dist[neg_mask].mean().item() - base_dist[pos_mask].mean().item())
            final_gap = float(final_dist[neg_mask].mean().item() - final_dist[pos_mask].mean().item())
        else:
            base_gap = 0.0
            final_gap = 0.0

        mask = ~eye
        stats = {
            'delta_mean': float(delta[mask].mean().item()),
            'delta_std': float(delta[mask].std(unbiased=False).item()),
            'support_mean': float(support_ratio[mask].mean().item()),
            'change_mean': float(pair_change[mask].mean().item()),
            'weight_mean': float(pair_weight[mask].mean().item()),
            'base_gap': base_gap,
            'final_gap': final_gap,
        }
        return loss, stats
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    # Per-component loss meters for detailed logging
    detail_meters = {}

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
    scaler = amp.GradScaler()

    # Backbone freeze warmup
    freeze_epochs = cfg.SOLVER.FREEZE_BACKBONE_EPOCHS
    backbone_frozen = False

    def _freeze_backbone(model):
        """Freeze backbone parameters, keep PSG/classifier/BN trainable."""
        m = model.module if hasattr(model, 'module') else model
        for name, param in m.base.named_parameters():
            param.requires_grad = False
        frozen = sum(1 for p in m.base.parameters() if not p.requires_grad)
        total = sum(1 for p in m.base.parameters())
        logger.info(f'Backbone FROZEN: {frozen}/{total} params frozen')

    def _unfreeze_backbone(model):
        """Unfreeze all backbone parameters."""
        m = model.module if hasattr(model, 'module') else model
        for param in m.base.parameters():
            param.requires_grad = True
        logger.info('Backbone UNFROZEN: all params trainable')

    if freeze_epochs > 0:
        _freeze_backbone(model)
        backbone_frozen = True
        logger.info(f'Backbone freeze warmup: {freeze_epochs} epochs')

    # train
    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after warmup
        if backbone_frozen and epoch > freeze_epochs:
            _unfreeze_backbone(model)
            backbone_frozen = False
        # Set current epoch for delayed stop_grad
        _model = model.module if hasattr(model, 'module') else model
        if hasattr(_model, 'current_epoch'):
            _model.current_epoch = epoch
            if hasattr(_model, 'stop_grad_epochs') and _model.stop_grad_epochs > 0:
                if epoch == _model.stop_grad_epochs + 1:
                    logger.info(f'[PDS] Epoch {epoch}: Part gradient RELEASED to shared stages')

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in detail_meters.values():
            m.reset()
        evaluator.reset()
        model.train()

        # SCFR: set/unset bank reference on skeleton_head each epoch
        if (scfr_enabled or scrc_enabled) and sckd_bank is not None:
            _m = model.module if hasattr(model, 'module') else model
            if hasattr(_m, 'skeleton_head'):
                _m.skeleton_head._scfr_bank = sckd_bank if scfr_enabled else None
                _m.skeleton_head._scfr_active = (epoch > sckd_warmup) if scfr_enabled else False
                _m.skeleton_head._scrc_bank = sckd_bank if scrc_enabled else None
                _m.skeleton_head._scrc_active = (epoch > sckd_warmup) if scrc_enabled else False

        for n_iter, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            if use_pose:
                img, vid, target_cam, target_view, pose_dict = batch_data
                pose_dict = _pose_to_device(pose_dict, device)
            else:
                img, vid, target_cam, target_view = batch_data
                pose_dict = None

            # Handle parallel augmentation: img may be list of 3 tensors
            parallel_aug = isinstance(img, list)
            if parallel_aug:
                img_views = [v.to(device) for v in img]
                batch_size = img_views[0].shape[0]
            else:
                img = img.to(device)
                batch_size = img.shape[0]
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                feat_maps = None  # captured for PACD
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
                    # Use first view's outputs as primary (for logging)
                    score, feat = all_scores[0], all_feats[0]
                    feat_maps = fm_v  # last view's feature maps for PACD
                    recon_loss = all_recon[0]
                    kp_data = all_kpdata[0]
                elif use_pose:
                    model_out = model(img, label=target, cam_label=target_cam,
                                      view_label=target_view,
                                      pose_dict=pose_dict)
                    # Handle optional return values
                    kp_data = None
                    feat_maps = None
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
                # PCRA: compute pose similarity matrix for triplet loss
                pose_sim = None
                if use_pose and pose_dict is not None and pcra_alpha > 0:
                    heatmaps = pose_dict['heatmaps']  # (B, max_persons, 17, H, W)
                    # Merge persons: max across person dim → (B, 17, H, W)
                    scene_hm = heatmaps.max(dim=1)[0]
                    # GAP → (B, 17) pose signature
                    pose_sig = scene_hm.mean(dim=(-2, -1))  # (B, 17)
                    # Cosine similarity matrix (B, B)
                    pose_sig_norm = F.normalize(pose_sig, p=2, dim=1)
                    pose_sim = torch.mm(pose_sig_norm, pose_sig_norm.t())  # (B, B)

                # Prepare per-keypoint triplet data
                paml_enabled = getattr(cfg.MODEL, 'POSE_PAML', False)
                kp_aux_data = None
                kdl_enabled = getattr(cfg.MODEL, 'POSE_KP_DISSIMILAR', False)
                lku_enabled = getattr(cfg.MODEL, 'POSE_KP_UNCERTAINTY', False)
                pke_enabled = getattr(cfg.MODEL, 'POSE_PKE', False)
                if kp_data is not None and (kp_triplet_enabled or csgt_enabled or csrd_enabled or paml_enabled or kdl_enabled or lku_enabled or pke_enabled):
                    kp_aux_data = dict(kp_data)
                    kp_aux_data['epoch'] = epoch
                    if kp_triplet_enabled:
                        kp_aux_data['weight'] = kp_triplet_weight
                    if csrd_enabled and csrd_support_teacher and csrd_teacher_bank is not None and epoch > getattr(cfg.MODEL, 'POSE_CSRD_WARMUP', 20):
                        kp_feats_csrd = kp_data.get('kp_feats')
                        kp_w_csrd = kp_data.get('kp_weights')
                        if kp_feats_csrd is not None and kp_w_csrd is not None:
                            teacher_feats, replace_mask, teacher_stats = csrd_teacher_bank.replace(
                                kp_feats_csrd, kp_w_csrd, target)
                            kp_aux_data['csrd_teacher_feats'] = teacher_feats.detach()
                            if csrd_anchor_weight_mode == 'replace_ratio':
                                anchor_weights = replace_mask.float().mean(dim=1)
                            elif csrd_anchor_weight_mode == 'low_ratio':
                                anchor_weights = (kp_w_csrd <= csrd_teacher_bank.low_thr).float().mean(dim=1)
                            else:
                                anchor_weights = None
                            if anchor_weights is not None:
                                kp_aux_data['csrd_anchor_weights'] = anchor_weights.detach()
                                teacher_stats['anchor_weight_mean'] = float(anchor_weights.mean().item())
                                teacher_stats['anchor_active_ratio'] = float((anchor_weights > 0).float().mean().item())
                            kp_aux_data['csrd_teacher_stats'] = teacher_stats
                            if csrd_queue_size > 0:
                                queue_payload = _get_csrd_queue_payload()
                                if queue_payload is not None:
                                    kp_aux_data['csrd_queue'] = queue_payload
                    if ltcs_enabled and ltcs_teacher_bank is not None and epoch > ltcs_warmup:
                        kp_feats_ltcs = kp_data.get('kp_feats')
                        kp_w_ltcs = kp_data.get('kp_weights')
                        if kp_feats_ltcs is not None and kp_w_ltcs is not None:
                            teacher_feats_ltcs, _, teacher_stats_ltcs = ltcs_teacher_bank.replace(
                                kp_feats_ltcs, kp_w_ltcs, target)
                            kp_aux_data['ltcs_teacher_feats'] = teacher_feats_ltcs.detach()
                            kp_aux_data['ltcs_teacher_stats'] = teacher_stats_ltcs
                    if lpcs_enabled and lpcs_teacher_bank is not None and epoch > lpcs_warmup:
                        kp_feats_lpcs = kp_data.get('kp_feats')
                        kp_w_lpcs = kp_data.get('kp_weights')
                        if kp_feats_lpcs is not None and kp_w_lpcs is not None:
                            teacher_feats_lpcs, _, teacher_stats_lpcs = lpcs_teacher_bank.replace(
                                kp_feats_lpcs, kp_w_lpcs, target)
                            kp_aux_data['lpcs_teacher_feats'] = teacher_feats_lpcs.detach()
                            kp_aux_data['lpcs_teacher_stats'] = teacher_stats_lpcs

                loss = loss_fn(score, feat, target, target_cam, pose_sim=pose_sim,
                               kp_data=kp_aux_data)
                if kp_aux_data is not None and 'csrd_teacher_stats' in kp_aux_data:
                    teacher_stats = kp_aux_data['csrd_teacher_stats']
                    details = getattr(loss, '_loss_details', {})
                    details['csrd_sr'] = teacher_stats['replace_ratio']
                    details['csrd_sn'] = float(teacher_stats['n_replaced'])
                    details['csrd_lowr'] = teacher_stats['low_ratio']
                    if 'anchor_weight_mean' in teacher_stats:
                        details['csrd_aw'] = teacher_stats['anchor_weight_mean']
                        details['csrd_ar'] = teacher_stats['anchor_active_ratio']
                    loss._loss_details = details
                if ltcs_enabled and kp_aux_data is not None and 'ltcs_teacher_feats' in kp_aux_data:
                    _m = model.module if hasattr(model, 'module') else model
                    global_feat_ltcs = feat[0] if isinstance(feat, list) else feat
                    ltcs_loss, ltcs_stats = _compute_ltcs_loss(
                        _m.ltcs_head,
                        global_feat_ltcs,
                        kp_data.get('kp_feats'),
                        kp_data.get('kp_weights'),
                        kp_aux_data['ltcs_teacher_feats'],
                        target,
                    )
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
                if lpcs_enabled and kp_aux_data is not None and 'lpcs_teacher_feats' in kp_aux_data:
                    _m = model.module if hasattr(model, 'module') else model
                    global_feat_lpcs = feat[0] if isinstance(feat, list) else feat
                    lpcs_loss, lpcs_stats = _compute_lpcs_loss(
                        _m.lpcs_head,
                        global_feat_lpcs,
                        kp_data.get('kp_feats'),
                        kp_data.get('kp_weights'),
                        kp_aux_data['lpcs_teacher_feats'],
                        target,
                    )
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + lpcs_weight * lpcs_loss
                    details['lpcs'] = lpcs_loss.item()
                    details['lpcs_dm'] = lpcs_stats['delta_mean']
                    details['lpcs_ds'] = lpcs_stats['delta_std']
                    details['lpcs_sm'] = lpcs_stats['support_mean']
                    details['lpcs_cm'] = lpcs_stats['change_mean']
                    details['lpcs_wm'] = lpcs_stats['weight_mean']
                    details['lpcs_bg'] = lpcs_stats['base_gap']
                    details['lpcs_fg'] = lpcs_stats['final_gap']
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
                        v_loss = loss_fn(all_scores[vi], all_feats[vi], target,
                                         target_cam, pose_sim=pose_sim)
                        if all_recon[vi] is not None:
                            v_loss = v_loss + all_recon[vi]
                        loss = loss + v_loss
                    loss = loss / len(all_scores)  # average over views
                    loss._loss_details = saved_details  # re-attach logging

                # SGMKC: reconstruction loss for masked keypoint completion
                if kp_data is not None and 'sgmkc_mask' in kp_data:
                    sgmkc_mask = kp_data['sgmkc_mask']        # (B, 17) True=kept
                    sgmkc_orig = kp_data['sgmkc_original']    # (B, 17, C)
                    sgmkc_pred = kp_data['kp_feats']          # (B, 17, C)
                    # Compute MSE only at masked (zeroed-out) positions
                    masked_positions = ~sgmkc_mask             # True = was masked
                    if masked_positions.any():
                        pred_masked = sgmkc_pred[masked_positions]   # (N_masked, C)
                        orig_masked = sgmkc_orig[masked_positions]   # (N_masked, C)
                        sgmkc_loss = F.mse_loss(pred_masked, orig_masked)
                        sgmkc_weight = getattr(cfg.MODEL, 'POSE_SGMKC_WEIGHT', 1.0)
                        details = getattr(loss, '_loss_details', {})
                        loss = loss + sgmkc_weight * sgmkc_loss
                        details['sgmkc'] = sgmkc_loss.item()
                        loss._loss_details = details

                # PAMC: Pose-Aware Masking Consistency
                if pamc_enabled and use_pose and epoch > pamc_warmup:
                    _m = model.module if hasattr(model, 'module') else model
                    if hasattr(_m, 'pamc_masker') and hasattr(_m, 'pamc_projector'):
                        # Get scene heatmaps for masking
                        pamc_scene_hm, _, _, _ = _m._prepare_pose(pose_dict)
                        # Create masked image (no grad needed for masking)
                        img_masked, _ = _m.pamc_masker(img, pamc_scene_hm)
                        # Switch to eval mode for deterministic target features
                        # (disables DropPath/StochasticDepth, BN uses running stats)
                        _m.eval()
                        with torch.no_grad():
                            masked_global_feat, _ = _m._run_backbone_with_psg(
                                img_masked, pamc_scene_hm)
                            if _m.reduce_feat_dim:
                                masked_global_feat = _m.fcneck(masked_global_feat)
                        _m.train()  # restore training mode
                        # Get original global feat (pre-BN, has grad from ID loss)
                        if isinstance(feat, list):
                            orig_global_feat = feat[0]  # global branch only
                        else:
                            orig_global_feat = feat
                        # Asymmetric consistency: projector(orig) predicts masked target
                        pamc_loss = pamc_consistency_loss(
                            orig_global_feat, masked_global_feat,
                            _m.pamc_projector)
                        details = getattr(loss, '_loss_details', {})
                        loss = loss + pamc_weight * pamc_loss
                        details['pamc'] = pamc_loss.item()
                        loss._loss_details = details

                # CIPGFR: Cross-Instance Pose-Guided Feature Recovery
                cipgfr_enabled = getattr(cfg.MODEL, 'POSE_CIPGFR', False)
                cipgfr_warmup = getattr(cfg.MODEL, 'POSE_CIPGFR_WARMUP', 20)
                if cipgfr_enabled and kp_data is not None and epoch > cipgfr_warmup:
                    kp_feats_all = kp_data.get('kp_feats')    # (B, 17, C)
                    kp_weights_all = kp_data.get('kp_weights')  # (B, 17)
                    if kp_feats_all is not None and kp_weights_all is not None:
                        cipgfr_weight = getattr(cfg.MODEL, 'POSE_CIPGFR_WEIGHT', 0.5)
                        cipgfr_thr = getattr(cfg.MODEL, 'POSE_CIPGFR_THRESHOLD', 0.3)
                        B_kp = kp_feats_all.shape[0]
                        cipgfr_loss = torch.tensor(0.0, device=kp_feats_all.device)
                        n_pairs = 0
                        # For each sample, find same-ID partner in batch
                        for i in range(B_kp):
                            # Find indices with same label
                            same_id = (target == target[i]).nonzero(as_tuple=True)[0]
                            same_id = same_id[same_id != i]  # exclude self
                            if len(same_id) == 0:
                                continue
                            j = same_id[torch.randint(len(same_id), (1,)).item()]
                            # i's occluded but j's visible keypoints
                            occ_i = kp_weights_all[i] < cipgfr_thr  # (17,) bool
                            vis_j = kp_weights_all[j] > cipgfr_thr  # (17,) bool
                            recovery_mask = occ_i & vis_j
                            if recovery_mask.sum() == 0:
                                continue
                            # Recovery: i's occluded feat → j's visible feat (detached)
                            cipgfr_loss = cipgfr_loss + F.mse_loss(
                                kp_feats_all[i][recovery_mask],
                                kp_feats_all[j][recovery_mask].detach())
                            n_pairs += 1
                        if n_pairs > 0:
                            cipgfr_loss = cipgfr_loss / n_pairs
                            details = getattr(loss, '_loss_details', {})
                            loss = loss + cipgfr_weight * cipgfr_loss
                            details['cipgfr'] = cipgfr_loss.item()
                            loss._loss_details = details

                # PCQA: Pose Translation Module loss
                if ptm_enabled and use_pose and pose_dict is not None and epoch > ptm_warmup:
                    _m = model.module if hasattr(model, 'module') else model
                    if hasattr(_m, 'ptm'):
                        # Use actual keypoint coordinates + scores as pose descriptor
                        ptm_kp = pose_dict['keypoints'][:, 0, :, :]  # (B, 17, 2) person 0
                        ptm_scores = pose_dict['scores'][:, 0, :]    # (B, 17) person 0
                        if isinstance(feat, list):
                            global_feat_ptm = feat[0]
                        else:
                            global_feat_ptm = feat
                        ptm_loss = _m.ptm.compute_training_loss(
                            global_feat_ptm, ptm_kp, ptm_scores, target,
                            normalize_coords=ptm_norm)
                        details = getattr(loss, '_loss_details', {})
                        loss = loss + ptm_weight * ptm_loss
                        details['ptm'] = ptm_loss.item()
                        loss._loss_details = details

                # LSRM: Learned Skeleton Recovery Module loss
                if lsrm_enabled and kp_data is not None and epoch > lsrm_warmup:
                    _m = model.module if hasattr(model, 'module') else model
                    if hasattr(_m, 'lsrm'):
                        kp_feats_lsrm = kp_data.get('kp_feats')
                        kp_weights_lsrm = kp_data.get('kp_weights')
                        if kp_feats_lsrm is not None and kp_weights_lsrm is not None:
                            lsrm_loss = _m.lsrm.compute_training_loss(
                                kp_feats_lsrm, kp_weights_lsrm, target)
                            details = getattr(loss, '_loss_details', {})
                            loss = loss + lsrm_weight * lsrm_loss
                            details['lsrm'] = lsrm_loss.item()
                            loss._loss_details = details

                # PAMN: Pose-Aware Matching Network loss
                if pamn_enabled and pamn_module is not None and kp_data is not None:
                    kp_feats_pamn = kp_data.get('kp_feats')
                    kp_weights_pamn = kp_data.get('kp_weights')
                    if kp_feats_pamn is not None and kp_weights_pamn is not None:
                        pamn_loss = pamn_module.compute_training_loss(
                            kp_feats_pamn.detach(), kp_weights_pamn.detach(), target)
                        details = getattr(loss, '_loss_details', {})
                        loss = loss + pamn_weight * pamn_loss
                        details['pamn'] = pamn_loss.item()
                        loss._loss_details = details

                # Momentum Memory Contrastive Loss
                if mm_enabled and mm_memory is not None:
                    if isinstance(feat, list):
                        mm_feat = feat[0]  # use global feature
                    else:
                        mm_feat = feat
                    mm_loss = mm_memory(mm_feat, target)
                    details = getattr(loss, '_loss_details', {})
                    loss = loss + mm_weight * mm_loss
                    details['mm'] = mm_loss.item()
                    loss._loss_details = details

                if sckd_enabled and sckd_bank is not None and kp_data is not None and epoch > sckd_warmup:
                    # SCFR mode: log replacement stats instead of computing loss
                    if scfr_enabled and kp_data.get('scfr_stats') is not None:
                        scfr_st = kp_data['scfr_stats']
                        details = getattr(loss, '_loss_details', {})
                        details['scfr_n'] = float(scfr_st['n_replaced'])
                        details['scfr_r'] = scfr_st['replace_ratio']
                        details['scfr_conf'] = scfr_st['proto_conf']
                        details['scfr_count'] = scfr_st['proto_count']
                        loss._loss_details = details
                    elif scrc_enabled and kp_data.get('scrc_stats') is not None:
                        scrc_st = kp_data['scrc_stats']
                        details = getattr(loss, '_loss_details', {})
                        details['scrc_n'] = float(scrc_st['n_fused'])
                        details['scrc_r'] = scrc_st['fuse_ratio']
                        details['scrc_g'] = scrc_st['gate_mean']
                        details['scrc_gm'] = scrc_st['gate_max']
                        details['scrc_dn'] = scrc_st['delta_norm']
                        details['scrc_conf'] = scrc_st['proto_conf']
                        details['scrc_count'] = scrc_st['proto_count']
                        loss._loss_details = details
                    elif not scfr_enabled and not scrc_enabled:
                        # Original SCKD distillation loss
                        kp_feats_sckd = kp_data.get('kp_feats')
                        kp_w_sckd = kp_data.get('kp_weights')
                        if kp_feats_sckd is not None and kp_w_sckd is not None:
                            sckd_loss, sckd_pairs, sckd_stats = sckd_bank.compute_loss(
                                kp_feats_sckd, kp_w_sckd, target)
                            if sckd_pairs > 0:
                                details = getattr(loss, '_loss_details', {})
                                loss = loss + sckd_weight * sckd_loss
                                details['sckd'] = sckd_loss.item()
                                details['sckd_pairs'] = float(sckd_pairs)
                                details['sckd_lowr'] = sckd_stats['low_ratio']
                                details['sckd_actr'] = sckd_stats['active_ratio']
                                details['sckd_eligr'] = sckd_stats['elig_ratio']
                                details['sckd_conf'] = sckd_stats['proto_conf']
                            details['sckd_count'] = sckd_stats['proto_count']
                            details['sckd_cos'] = sckd_stats['cosine']
                            loss._loss_details = details

                # SGRE: Skeleton-Guided Re-Encoding loss
                sgre_enabled = getattr(cfg.MODEL, 'POSE_SGRE', False)
                sgre_warmup = getattr(cfg.MODEL, 'POSE_SGRE_WARMUP', 20)
                if sgre_enabled and kp_data is not None and epoch > sgre_warmup:
                    _m = model.module if hasattr(model, 'module') else model
                    if hasattr(_m, 'sgre'):
                        kp_feats_sgre = kp_data.get('kp_feats')
                        kp_w_sgre = kp_data.get('kp_weights')
                        if kp_feats_sgre is not None:
                            sgre_weight = getattr(cfg.MODEL, 'POSE_SGRE_WEIGHT', 0.5)
                            sgre_loss = _m.sgre.compute_training_loss(
                                kp_feats_sgre.detach(), kp_w_sgre.detach(), target)
                            details = getattr(loss, '_loss_details', {})
                            loss = loss + sgre_weight * sgre_loss
                            details['sgre'] = sgre_loss.item()
                            loss._loss_details = details

                # PISD: Pose-Informed Self-Distillation at IMAGE level
                # Mask body parts on the actual image, re-run backbone,
                # enforce partial features ≈ full features
                pisd_enabled = getattr(cfg.MODEL, 'POSE_PISD', False)
                pisd_warmup = getattr(cfg.MODEL, 'POSE_PISD_WARMUP', 10)
                if pisd_enabled and use_pose and epoch > pisd_warmup and not parallel_aug:
                    pisd_weight = getattr(cfg.MODEL, 'POSE_PISD_WEIGHT', 0.3)
                    pisd_ratio = getattr(cfg.MODEL, 'POSE_PISD_MASK_RATIO', 0.4)
                    _m = model.module if hasattr(model, 'module') else model

                    # Create masked image using pose heatmaps at INPUT resolution
                    hm = pose_dict['heatmaps'].max(dim=1)[0]  # (B, 17, hH, hW)
                    hm_full = F.interpolate(hm, size=img.shape[2:],
                                             mode='bilinear', align_corners=False)
                    hm_full = torch.relu(hm_full)  # (B, 17, H, W)

                    # Select random keypoints to mask
                    B_img = img.shape[0]
                    num_mask = max(1, int(17 * pisd_ratio))
                    img_mask = torch.zeros(B_img, 1, img.shape[2], img.shape[3],
                                           device=img.device)
                    for b in range(B_img):
                        kp_idx = torch.randperm(17, device=img.device)[:num_mask]
                        selected = hm_full[b, kp_idx].sum(dim=0)  # (H, W)
                        # Use actual heatmap intensity as soft mask threshold
                        # Mask where body part response is strong
                        threshold = selected.quantile(0.7)  # top 30% of response
                        img_mask[b, 0] = (selected > threshold).float()

                    # Create masked image (fill with mean pixel value)
                    mean_pixel = img.mean(dim=(2, 3), keepdim=True)
                    img_masked = img * (1.0 - img_mask) + mean_pixel * img_mask

                    # Teacher: full features (already computed, detached)
                    if isinstance(feat, list):
                        feat_full = F.normalize(feat[0].detach(), dim=1)
                    else:
                        feat_full = F.normalize(feat.detach(), dim=1)

                    # Student: forward with masked image
                    # FIX: freeze BN running stats (don't change model.training
                    # which would switch the return format to eval mode)
                    bn_states = {}
                    for name, mod in _m.named_modules():
                        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            bn_states[name] = mod.training
                            mod.eval()
                    pisd_out = model(img_masked, label=target, cam_label=target_cam,
                                      view_label=target_view, pose_dict=pose_dict)
                    for name, mod in _m.named_modules():
                        if name in bn_states:
                            mod.train(bn_states[name])
                    # Extract global feature from student output
                    pisd_feat_raw = pisd_out[1]
                    if isinstance(pisd_feat_raw, list):
                        pisd_feat_raw = pisd_feat_raw[0]
                    pisd_feat_norm = F.normalize(pisd_feat_raw, dim=1)

                    # PISD loss: cosine distance
                    pisd_loss = (1.0 - (pisd_feat_norm * feat_full).sum(dim=1)).mean()

                    details = getattr(loss, '_loss_details', {})
                    loss = loss + pisd_weight * pisd_loss
                    details['pisd'] = pisd_loss.item()
                    loss._loss_details = details

                # PACD: Pose-Anchored Contrastive Distillation (feature map level)
                pacd_enabled = getattr(cfg.MODEL, 'POSE_PACD', False)
                pacd_warmup = getattr(cfg.MODEL, 'POSE_PACD_WARMUP', 10)
                if pacd_enabled and use_pose and feat_maps is not None and epoch > pacd_warmup:
                    pacd_weight = getattr(cfg.MODEL, 'POSE_PACD_WEIGHT', 0.3)
                    pacd_ratio = getattr(cfg.MODEL, 'POSE_PACD_MASK_RATIO', 0.4)
                    stage3_fm = feat_maps[-1] if isinstance(feat_maps, list) else feat_maps
                    B_fm, C_fm, fH, fW = stage3_fm.shape
                    _m = model.module if hasattr(model, 'module') else model

                    # Row-based masking: mask contiguous rows to simulate
                    # real occlusion (upper/lower body). Much stronger than
                    # 3×3 keypoint masking (~8%) or heatmap masking (76% bug).
                    # Target: ~50% of rows masked → strong but not destructive.
                    num_rows_mask = max(1, int(fH * pacd_ratio))  # 0.4*12=4 rows
                    body_mask = torch.zeros(B_fm, 1, fH, fW, device=stage3_fm.device)
                    for b in range(B_fm):
                        # Randomly choose starting row for contiguous block
                        start = torch.randint(0, fH - num_rows_mask + 1, (1,)).item()
                        body_mask[b, 0, start:start + num_rows_mask, :] = 1.0

                    # Mask: zero out selected keypoint positions
                    fm_masked = stage3_fm * (1.0 - body_mask)

                    # Pool with renormalization: average over UNMASKED positions only
                    keep_mask = (1.0 - body_mask)  # (B, 1, fH, fW)
                    n_keep = keep_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
                    feat_partial = (fm_masked).sum(dim=(2, 3)) / n_keep.squeeze(3).squeeze(2)
                    # feat_partial: (B, C)
                    if _m.reduce_feat_dim:
                        feat_partial = _m.fcneck(feat_partial)

                    # Teacher: full features (detached, L2-normalized for stable comparison)
                    if isinstance(feat, list):
                        feat_full = F.normalize(feat[0].detach(), dim=1)
                    else:
                        feat_full = F.normalize(feat.detach(), dim=1)
                    feat_partial_norm = F.normalize(feat_partial, dim=1)

                    # PACD loss: cosine distance (not MSE — invariant to magnitude)
                    pacd_loss = (1.0 - (feat_partial_norm * feat_full).sum(dim=1)).mean()

                    details = getattr(loss, '_loss_details', {})
                    loss = loss + pacd_weight * pacd_loss
                    details['pacd'] = pacd_loss.item()
                    loss._loss_details = details

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if sckd_enabled and sckd_bank is not None and kp_data is not None:
                kp_feats_sckd = kp_data.get('kp_feats')
                kp_w_sckd = kp_data.get('kp_weights')
                if kp_feats_sckd is not None and kp_w_sckd is not None:
                    if sckd_update_stop_epoch < 0 or epoch <= sckd_update_stop_epoch:
                        sckd_bank.update(kp_feats_sckd, kp_w_sckd, target)

            if csrd_enabled and csrd_support_teacher and csrd_teacher_bank is not None and kp_data is not None:
                kp_feats_csrd = kp_data.get('kp_feats')
                kp_w_csrd = kp_data.get('kp_weights')
                if kp_feats_csrd is not None and kp_w_csrd is not None:
                    if csrd_st_update_stop_epoch < 0 or epoch <= csrd_st_update_stop_epoch:
                        csrd_teacher_bank.update(kp_feats_csrd, kp_w_csrd, target)
                    if csrd_queue_size > 0 and epoch > getattr(cfg.MODEL, 'POSE_CSRD_WARMUP', 20):
                        queue_teacher_feats = None
                        if kp_aux_data is not None:
                            queue_teacher_feats = kp_aux_data.get('csrd_teacher_feats')
                        if queue_teacher_feats is not None:
                            queue_student_feat = feat[0] if isinstance(feat, list) else feat
                            _enqueue_csrd_queue(
                                queue_student_feat, kp_feats_csrd, kp_w_csrd,
                                queue_teacher_feats, target)

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

            # Track per-component losses if available
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
