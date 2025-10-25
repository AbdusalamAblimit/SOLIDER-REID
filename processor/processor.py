import logging
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, EvalResultItem
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Sequence, Tuple


def _get_enabled_eval_branches(cfg) -> Tuple[str, ...]:
    branches = getattr(cfg.TEST, "ENABLED_FEATS", ())
    if isinstance(branches, str):
        branches = (branches,)
    else:
        branches = tuple(branches)
    return tuple(b for b in branches if b)


def _filter_eval_features(feat, enabled_branches: Sequence[str], logger: logging.Logger):
    if not isinstance(feat, dict) or not enabled_branches:
        return feat
    selected = OrderedDict()
    missing = []
    for name in enabled_branches:
        if name in feat:
            selected[name] = feat[name]
        else:
            missing.append(name)
    if missing:
        logger.warning("Requested evaluation features %s are not available in model outputs (%s).",
                       missing, list(feat.keys()))
    if not selected:
        raise ValueError(
            "None of the requested evaluation features {} are available in model outputs (keys: {}).".format(
                enabled_branches, list(feat.keys()))
        )
    return selected


def _results_to_mapping(results, enabled_branches: Sequence[str]):
    if isinstance(results, dict):
        return results
    metrics = results if isinstance(results, EvalResultItem) else None
    if metrics is None:
        raise TypeError("Unexpected evaluation result type: {}".format(type(results)))
    if enabled_branches and len(enabled_branches) == 1:
        name = enabled_branches[0]
    else:
        name = 'global'
    return {name: metrics}
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             tb_writer: Optional[SummaryWriter] = None):
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
    enabled_eval_branches = _get_enabled_eval_branches(cfg)
    if enabled_eval_branches:
        logger.info("Evaluating branches: %s", ", ".join(enabled_eval_branches))
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        # === TB: 让 Pose-Swin 每个 epoch 只抓一次（第一个 batch） ===
        is_main = (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0
        if tb_writer is not None and is_main:
            core = model.module if hasattr(model, "module") else model
            if hasattr(core, "base") and hasattr(core.base, "reset_pose_debug_epoch"):
                core.base.reset_pose_debug_epoch()
                
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        # 每个 epoch 开始后立即按照配置冻结/解冻对应模块
        core_model = model.module if hasattr(model, "module") else model
        if hasattr(core_model, "update_freeze_schedule"):
            core_model.update_freeze_schedule(epoch)
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
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

            # === TB: 训练标量（每 iter） ===
            if tb_writer is not None and is_main:
                step = (epoch - 1) * len(train_loader) + n_iter  # 全局 step
                cur_acc = float(acc.detach().item()) if torch.is_tensor(acc) else float(acc)
                tb_writer.add_scalar("train/loss", float(loss.item()), step)
                tb_writer.add_scalar("train/acc",  cur_acc,            step)
                tb_writer.add_scalar("train/lr",   float(optimizer.param_groups[0]["lr"]), step)

            # === TB: 每个 epoch 的第一个 batch 把 Pose 中间量写图（如果模型支持） ===
            if tb_writer is not None and is_main and n_iter == 0:
                core = model.module if hasattr(model, "module") else model
                if hasattr(core, "base") and hasattr(core.base, "tb_dump_pose"):
                    try:
                        core.base.tb_dump_pose(tb_writer, step=step, tag_prefix=f"pose/epoch_{epoch:03d}")
                    except Exception as e:
                        print(f"[TB] pose dump failed: {e}")

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
        if tb_writer is not None:
            tb_writer.flush()
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
            eval_results = None
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, _ = model(img, cam_label=camids, view_label=target_view)
                            feat = _filter_eval_features(feat, enabled_eval_branches, logger)
                            evaluator.update((feat, vid, camid))
                    raw_results = evaluator.compute()
                    eval_results = _results_to_mapping(raw_results, enabled_eval_branches)
                    for name, metrics in eval_results.items():
                        cmc = metrics.cmc
                        mAP = metrics.mAP
                        logger.info("Validation Results [{}] - Epoch: {}".format(name, epoch))
                        logger.info("mAP: {:.1%}".format(mAP))
                        for r in [1, 5, 10]:
                            logger.info("CMC curve [{}], Rank-{:<3}:{:.1%}".format(name, r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, _ = model(img, cam_label=camids, view_label=target_view)
                        feat = _filter_eval_features(feat, enabled_eval_branches, logger)
                        evaluator.update((feat, vid, camid))
                raw_results = evaluator.compute()
                eval_results = _results_to_mapping(raw_results, enabled_eval_branches)
                for name, metrics in eval_results.items():
                    cmc = metrics.cmc
                    mAP = metrics.mAP
                    logger.info("Validation Results [{}] - Epoch: {}".format(name, epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve [{}], Rank-{:<3}:{:.1%}".format(name, r, cmc[r - 1]))
                torch.cuda.empty_cache()
            # === TB: 评估标量（按 epoch） ===
            if tb_writer is not None and ((not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0) and eval_results is not None:
                for name, metrics in eval_results.items():
                    tb_writer.add_scalar(f"eval/mAP_{name}", float(metrics.mAP), epoch)
                    tb_writer.add_scalar(f"eval/Rank-1_{name}", float(metrics.cmc[0]), epoch)
                tb_writer.flush()
            eval_results = None

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()
    enabled_eval_branches = _get_enabled_eval_branches(cfg)
    if enabled_eval_branches:
        logger.info("Evaluating branches: %s", ", ".join(enabled_eval_branches))

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
            feat = _filter_eval_features(feat, enabled_eval_branches, logger)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    raw_results = evaluator.compute()
    metrics_map = _results_to_mapping(raw_results, enabled_eval_branches)
    summary = {}
    logger.info("Validation Results")
    for name, metrics in metrics_map.items():
        cmc = metrics.cmc
        mAP = metrics.mAP
        logger.info("  [{}] mAP: {:.1%}".format(name, mAP))
        for r in [1, 5, 10]:
            logger.info("  [{}] CMC curve, Rank-{:<3}:{:.1%}".format(name, r, cmc[r - 1]))
        rank5_index = 4 if cmc.shape[0] > 4 else (cmc.shape[0] - 1)
        summary[name] = (cmc[0], cmc[rank5_index])
    return summary


