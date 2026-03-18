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

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    if use_pose:
        logger.info('Pose-guided training ENABLED')
    if mm_enabled:
        logger.info(f'Momentum Memory enabled: weight={mm_weight}, temp={mm_temp}, mom={mm_mom}')
    if pamc_enabled:
        logger.info(f'[PAMC] Pose-Aware Masking Consistency: weight={pamc_weight}, warmup={pamc_warmup}')
    if pcra_alpha > 0:
        logger.info(f'[PCRA] Pose-Contrastive Representation Alignment: alpha={pcra_alpha}')
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
                if parallel_aug and use_pose:
                    # 3-view parallel augmentation: forward all, average loss
                    all_scores, all_feats, all_recon, all_kpdata = [], [], [], []
                    for v_img in img_views:
                        m_out = model(v_img, label=target, cam_label=target_cam,
                                      view_label=target_view, pose_dict=pose_dict)
                        kd = None
                        if len(m_out) == 5:
                            s, f, _, rl, kd = m_out
                        elif len(m_out) == 4:
                            s, f, _, rl = m_out
                        else:
                            s, f = m_out[:2]; rl = None
                        all_scores.append(s)
                        all_feats.append(f)
                        all_recon.append(rl)
                        all_kpdata.append(kd)
                    # Use first view's outputs as primary (for logging)
                    score, feat = all_scores[0], all_feats[0]
                    recon_loss = all_recon[0]
                    kp_data = all_kpdata[0]
                elif use_pose:
                    model_out = model(img, label=target, cam_label=target_cam,
                                      view_label=target_view,
                                      pose_dict=pose_dict)
                    # Handle optional return values
                    kp_data = None
                    if len(model_out) == 5:
                        score, feat, _, recon_loss, kp_data = model_out
                    elif len(model_out) == 4:
                        score, feat, _, recon_loss = model_out
                    else:
                        score, feat, _ = model_out
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
                if kp_data is not None and (kp_triplet_enabled or csgt_enabled or paml_enabled or kdl_enabled or lku_enabled or pke_enabled):
                    kp_aux_data = dict(kp_data)
                    if kp_triplet_enabled:
                        kp_aux_data['weight'] = kp_triplet_weight

                loss = loss_fn(score, feat, target, target_cam, pose_sim=pose_sim,
                               kp_data=kp_aux_data)
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

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

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
