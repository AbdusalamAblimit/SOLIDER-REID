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

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    if use_pose:
        logger.info('Pose-guided training ENABLED')
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

            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                if use_pose:
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
                        pamc_scene_hm, _ = _m._prepare_pose(pose_dict)
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

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            # Track per-component losses if available
            if hasattr(loss, '_loss_details'):
                for k, v in loss._loss_details.items():
                    if k not in detail_meters:
                        detail_meters[k] = AverageMeter()
                    detail_meters[k].update(v, img.shape[0])

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
