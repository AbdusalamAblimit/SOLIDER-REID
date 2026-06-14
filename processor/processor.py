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
    objgate_on = bool(getattr(cfg, 'OBJGATE', None) is not None and cfg.OBJGATE.ENABLED)
    multihyp_on = bool(getattr(cfg, 'MULTIHYP', None) is not None and cfg.MULTIHYP.ENABLED)
    oss_on = bool(getattr(cfg, 'OSS', None) is not None and cfg.OSS.ENABLED)
    donor_on = bool(getattr(cfg, 'DONOR_DECOUPLE', None) is not None and cfg.DONOR_DECOUPLE.ENABLED)
    partial_evidence_on = bool(getattr(cfg, 'PARTIAL_EVIDENCE', None) is not None and cfg.PARTIAL_EVIDENCE.ENABLED)
    if sum([objgate_on, multihyp_on, oss_on, donor_on, partial_evidence_on]) > 1:
        raise ValueError("OBJGATE、MULTIHYP、OSS、DONOR_DECOUPLE、PARTIAL_EVIDENCE 都是单变量插件，请一次只开启一个。")
    if multihyp_on:
        from model.multihyp import multihyp_set_loss, multihyp_dshs_loss
        from utils.metrics import R1_mAP_eval_multihyp
    if oss_on:
        from model.occ_shortcut import occluder_shortcut_loss
    if donor_on:
        from model.donor_decouple import (
            build_donor_synth_batch,
            donor_counterfactual_loss,
            donor_orth_loss,
            donor_sameb_negative_loss,
        )
    if partial_evidence_on:
        from model.occ_shortcut import paste_occluder_batch
        from model.partial_evidence import partial_evidence_training_loss
        partial_evidence_pool = getattr(train_loader, 'partial_evidence_pool', None)
        if not partial_evidence_pool:
            raise ValueError("PARTIAL_EVIDENCE 需要 make_dataloader 构造 partial_evidence_pool。")

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

    if multihyp_on:
        evaluator = R1_mAP_eval_multihyp(num_query, cfg, max_rank=50)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        if objgate_on:
            _we = cfg.OBJGATE.LAMBDA_WARMUP_EPOCHS
            if epoch <= _we:
                cur_lambda = 0.0
            else:
                cur_lambda = min(1.0, (epoch - _we) / max(1, _we)) * cfg.OBJGATE.LAMBDA_TARGET
        else:
            cur_lambda = 0.0
        for n_iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            if objgate_on:
                img, vid, target_cam, target_view, obj_is_synth, obj_side, obj_ratio = batch
            elif oss_on:
                img, vid, target_cam, target_view, occ_id = batch
            else:
                img, vid, target_cam, target_view = batch
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            if oss_on:
                occ_id = occ_id.to(device)
            with amp.autocast(enabled=True):
                if objgate_on:
                    obj_is_synth = obj_is_synth.to(device)
                    obj_side = obj_side.to(device)
                    obj_ratio = obj_ratio.to(device)
                    score, feat, _, reg_loss = model(img, label=target, cam_label=target_cam, view_label=target_view,
                                                     obj_is_synth=obj_is_synth, obj_side=obj_side, obj_ratio=obj_ratio,
                                                     obj_lambda=cur_lambda)
                    loss = loss_fn(score, feat, target, target_cam) + reg_loss
                elif multihyp_on:
                    score, feat, _, slots = model(img, label=target, cam_label=target_cam, view_label=target_view)
                    set_loss = multihyp_set_loss(slots, target,
                                                 pos_margin=cfg.MULTIHYP.POS_MARGIN,
                                                 neg_margin=cfg.MULTIHYP.NEG_MARGIN,
                                                 div_w=cfg.MULTIHYP.DIV_W,
                                                 set_temp=cfg.MULTIHYP.SET_TEMP)
                    loss = loss_fn(score, feat, target, target_cam) + cfg.MULTIHYP.LOSS_W * set_loss
                    if cfg.MULTIHYP.DSHS_W > 0:                       # C2 DSHS（DSHS_W=0 时此分支不进，逐数值等于现损失）
                        dshs_loss = multihyp_dshs_loss(slots, feat.detach(), target,
                                                       set_margin=cfg.MULTIHYP.DSHS_MARGIN,
                                                       n_hard=cfg.MULTIHYP.DSHS_NHARD,
                                                       set_temp=cfg.MULTIHYP.SET_TEMP,
                                                       hard_by=cfg.MULTIHYP.DSHS_HARD)
                        loss = loss + cfg.MULTIHYP.DSHS_W * dshs_loss
                elif oss_on:
                    score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view)
                    loss = loss_fn(score, feat, target, target_cam)
                    # GRL alpha warmup（标准 DANN schedule：alpha 从 ~0 平滑升到 GRL_ALPHA，
                    # 让身份信号先建立、再逐渐加对抗，避免固定 alpha=1 时对抗 CE 顶起整体损失）
                    _p = (epoch - 1 + (n_iter + 1) / len(train_loader)) / max(1, epochs)
                    oss_alpha = cfg.OSS.GRL_ALPHA * (2.0 / (1.0 + np.exp(-10.0 * _p)) - 1.0)
                    raw_model = model.module if hasattr(model, 'module') else model
                    occ_logits, occ_loss = occluder_shortcut_loss(raw_model.oss_head, feat, occ_id, oss_alpha)
                    loss = loss + cfg.OSS.W * occ_loss
                elif donor_on:
                    score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view)
                    loss = loss_fn(score, feat, target, target_cam)
                    raw_model = model.module if hasattr(model, 'module') else model
                    synth_img, donor_label, donor_rect, donor_group, _ = build_donor_synth_batch(
                        img,
                        target,
                        paste_prob=cfg.DONOR_DECOUPLE.PASTE_PROB,
                        donor_repeat=cfg.DONOR_DECOUPLE.DONOR_REPEAT,
                        no_donor_label=int(raw_model.num_classes),
                    )
                    synth_score, synth_feat, _, donor_logits, donor_feat = model(
                        synth_img,
                        label=target,
                        cam_label=target_cam,
                        view_label=target_view,
                        donor_rects=donor_rect,
                        donor_aux=True,
                    )
                    synth_score_main = synth_score[0] if isinstance(synth_score, list) else synth_score
                    synth_id_loss = F.cross_entropy(synth_score_main, target)
                    cf_loss = donor_counterfactual_loss(synth_feat, feat)
                    sameb_loss = donor_sameb_negative_loss(
                        synth_feat, feat, target, donor_group,
                        margin=cfg.DONOR_DECOUPLE.NEG_MARGIN,
                    )
                    donor_cls_loss = F.cross_entropy(donor_logits, donor_label)
                    orth_loss = donor_orth_loss(synth_feat, donor_feat)
                    loss = loss \
                        + cfg.DONOR_DECOUPLE.SYN_ID_W * synth_id_loss \
                        + cfg.DONOR_DECOUPLE.CF_W * cf_loss \
                        + cfg.DONOR_DECOUPLE.SAMEB_NEG_W * sameb_loss \
                        + cfg.DONOR_DECOUPLE.DONOR_CLS_W * donor_cls_loss \
                        + cfg.DONOR_DECOUPLE.ORTH_W * orth_loss
                elif partial_evidence_on:
                    score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view)
                    loss = loss_fn(score, feat, target, target_cam)
                    synth_img, pe_occ_id, _, pe_evidence = paste_occluder_batch(
                        img,
                        partial_evidence_pool,
                        aug_prob=cfg.PARTIAL_EVIDENCE.PASTE_PROB,
                        return_metadata=True,
                    )
                    synth_score, synth_feat, _ = model(
                        synth_img,
                        label=target,
                        cam_label=target_cam,
                        view_label=target_view,
                    )
                    pe_loss = partial_evidence_training_loss(
                        synth_score,
                        synth_feat,
                        target,
                        target_cam,
                        pe_occ_id,
                        pe_evidence,
                        cfg,
                        loss_fn=loss_fn,
                    )
                    loss = loss + pe_loss
                else:
                    score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view )
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
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

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
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
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
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
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

    if bool(getattr(cfg, 'MULTIHYP', None) is not None and cfg.MULTIHYP.ENABLED):
        from utils.metrics import R1_mAP_eval_multihyp
        evaluator = R1_mAP_eval_multihyp(num_query, cfg, max_rank=50)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat , _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
