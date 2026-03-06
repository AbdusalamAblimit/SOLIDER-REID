import logging
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

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

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # ETA tracking
    epoch_times = []
    train_start_time = time.time()

    # Check if pose_part is enabled
    use_pose_part = getattr(cfg.MODEL, 'POSE_PART', None) and cfg.MODEL.POSE_PART.ENABLE

    # Backbone freeze support for VPReID
    freeze_epochs = getattr(cfg.SOLVER, 'FREEZE_BACKBONE_EPOCHS', 0)
    is_vpreid = hasattr(model, 'is_vpreid') and model.is_vpreid
    if freeze_epochs > 0 and is_vpreid:
        logger.info(f'Freezing backbone for first {freeze_epochs} epochs')
        for p in model.base.base.parameters():
            p.requires_grad = False

    # train
    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1 and is_vpreid:
            logger.info(f'Unfreezing backbone at epoch {epoch}')
            for p in model.base.base.parameters():
                p.requires_grad = True

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            # Unpack batch: standard (4 items) or pose-aware (6 items)
            if len(batch) == 6:
                img, vid, target_cam, target_view, kpts, vis = batch
                kpts = kpts.to(device)
                vis = vis.to(device)
            else:
                img, vid, target_cam, target_view = batch
                kpts = None
                vis = None

            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                if kpts is not None:
                    model_out = model(img, label=target, cam_label=target_cam,
                                      view_label=target_view, keypoints=kpts, visibility=vis)
                else:
                    model_out = model(img, label=target, cam_label=target_cam, view_label=target_view)

                # Detect output format:
                # PosePart: (score, global_feat, extras_dict) where extras has 'part_logits'
                # VPReID: (scores_list, feats_list, extras_dict) where extras has 'part_vis'
                # Standard: (score, feat, featmaps)
                if isinstance(model_out, tuple) and len(model_out) == 3 and isinstance(model_out[2], dict):
                    score, feat, extras = model_out
                    loss = loss_fn(score, feat, target, target_cam, extras=extras)
                else:
                    score, feat, _ = model_out
                    loss = loss_fn(score, feat, target, target_cam)

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

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                is_main = not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0
                if is_main:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    # Iter-level ETA
                    iter_elapsed = time.time() - start_time
                    iters_done = n_iter + 1
                    iters_total = len(train_loader)
                    iter_eta_s = iter_elapsed / iters_done * (iters_total - iters_done)
                    # Remaining epochs ETA (use previous epoch avg if available)
                    if epoch_times:
                        epoch_eta_s = (sum(epoch_times) / len(epoch_times)) * (epochs - epoch)
                    else:
                        epoch_eta_s = (iter_elapsed / iters_done * iters_total) * (epochs - epoch + 1)
                    total_eta_s = iter_eta_s + epoch_eta_s
                    eta_h = int(total_eta_s // 3600)
                    eta_m = int((total_eta_s % 3600) // 60)

                    msg = "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Lr: {:.2e}, ETA: {}h{:02d}m".format(
                        epoch, iters_done, iters_total, loss_meter.avg, acc_meter.avg, base_lr, eta_h, eta_m)

                    # Append loss breakdown if available
                    if hasattr(loss_fn, 'last_components') and loss_fn.last_components:
                        lc = loss_fn.last_components
                        if 'push' in lc:
                            # VPReID format
                            msg += " | id={:.2f} pid={:.2f} tri={:.2f} push={:.2f}".format(
                                lc['id'], lc['pid'], lc['tri'], lc['push'])
                        elif 'pid' in lc:
                            # PosePart / PCFC format
                            msg += " | id={:.2f} tri={:.2f} pid={:.2f} nv={}".format(
                                lc['id'], lc['tri'], lc['pid'], lc.get('n_vis', '?'))
                            if 'ptri' in lc and lc['ptri'] > 0:
                                msg += " ptri={:.2f}".format(lc['ptri'])
                            if 'alpha' in lc:
                                msg += " a={:.3f}".format(lc['alpha'])
                            if 'kpe_scale' in lc:
                                msg += " kpe={:.3f}".format(lc['kpe_scale'])
                            # Log PVFM beta values
                            betas = [f"{k}={v:.3f}" for k, v in lc.items() if k.startswith('beta_')]
                            if betas:
                                msg += " " + " ".join(betas)

                    logger.info(msg)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()

        # ETA estimation
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        elapsed = end_time - train_start_time
        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)
        elapsed_h = int(elapsed // 3600)
        elapsed_m = int((elapsed % 3600) // 60)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time: {:.0f}s, Speed: {:.1f}[samples/s], "
                    "Elapsed: {}h{}m, ETA: {}h{}m ({} epochs left)"
                    .format(epoch, epoch_time, train_loader.batch_size / time_per_batch,
                            elapsed_h, elapsed_m, eta_h, eta_m, remaining_epochs))

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
                    for n_iter, val_batch in enumerate(val_loader):
                        with torch.no_grad():
                            # Unpack val batch: standard (6) or pose-aware (8)
                            if len(val_batch) == 8:
                                img, vid, camid, camids, target_view, _, kpts, vis = val_batch
                                kpts = kpts.to(device)
                                vis = vis.to(device)
                            else:
                                img, vid, camid, camids, target_view, _ = val_batch
                                kpts = None
                                vis = None
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            if kpts is not None:
                                feat, _ = model(img, cam_label=camids, view_label=target_view,
                                                keypoints=kpts, visibility=vis)
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
                for n_iter, val_batch in enumerate(val_loader):
                    with torch.no_grad():
                        if len(val_batch) == 8:
                            img, vid, camid, camids, target_view, _, kpts, vis = val_batch
                            kpts = kpts.to(device)
                            vis = vis.to(device)
                        else:
                            img, vid, camid, camids, target_view, _ = val_batch
                            kpts = None
                            vis = None
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        if kpts is not None:
                            feat, _ = model(img, cam_label=camids, view_label=target_view,
                                            keypoints=kpts, visibility=vis)
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

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, val_batch in enumerate(val_loader):
        with torch.no_grad():
            if len(val_batch) == 8:
                img, pid, camid, camids, target_view, imgpath, kpts, vis = val_batch
                kpts = kpts.to(device)
                vis = vis.to(device)
            else:
                img, pid, camid, camids, target_view, imgpath = val_batch
                kpts = None
                vis = None
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            if kpts is not None:
                feat, _ = model(img, cam_label=camids, view_label=target_view,
                                keypoints=kpts, visibility=vis)
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
