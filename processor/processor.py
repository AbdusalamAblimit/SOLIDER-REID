import logging
import os
import random
import hashlib
import json
import stat
from pathlib import Path
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_elo_generic_evidence(
    path_value,
    expected_sha256,
    dataset_name,
    clip_checkpoint_sha256,
    codebook_sha256,
    pose_manifest_sha256,
    device,
):
    configured = Path(path_value)
    if not configured.is_absolute():
        raise ValueError("ELO generic evidence path must be absolute")
    resolved = configured.resolve(strict=True)
    if resolved != configured:
        raise RuntimeError("ELO generic evidence must use a canonical path")
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError("ELO generic evidence must be a unique regular file")
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("ELO generic evidence SHA mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("experiment") != "exp403_counterfactual_operator_ownership":
        raise RuntimeError("ELO generic evidence experiment tag mismatch")
    expected_metadata = {
        "format": "elo_generic_evidence_v1",
        "dataset": str(dataset_name),
        "split": "train",
        "clip_checkpoint_sha256": str(clip_checkpoint_sha256),
        "codebook_sha256": str(codebook_sha256),
        "pose_manifest_sha256": str(pose_manifest_sha256),
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise RuntimeError("ELO generic evidence metadata mismatch: " + key)
    count_by_slot = payload.get("count_by_slot")
    if (
        not isinstance(count_by_slot, list)
        or len(count_by_slot) != 5
        or any(not isinstance(value, int) or value <= 0 for value in count_by_slot)
    ):
        raise RuntimeError("ELO generic evidence count_by_slot is invalid")
    evidence = torch.as_tensor(payload.get("evidence"), dtype=torch.float32)
    if evidence.shape != (5, 16):
        raise RuntimeError("ELO generic evidence must have shape [5,16]")
    if not bool(torch.isfinite(evidence).all()):
        raise RuntimeError("ELO generic evidence is non-finite")
    return evidence.to(device=device), actual_sha256

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

    semantic_teacher = None
    semantic_teacher_diagnostic_done = False
    rich_evidence_enabled = bool(
        cfg.MODEL.TAPF.ENABLED
        and cfg.MODEL.TAPF.SEMANTIC_ENABLED
        and cfg.MODEL.TAPF.RICH_EVIDENCE_ENABLED
    )
    elo_cur_enabled = bool(
        rich_evidence_enabled and cfg.MODEL.TAPF.ELO_CUR_ENABLED
    )
    spk_enabled = bool(
        rich_evidence_enabled and cfg.MODEL.TAPF.SPK_ENABLED
    )
    picrd_enabled = bool(
        cfg.MODEL.TAPF.ENABLED and cfg.MODEL.TAPF.PICRD_ENABLED
    )
    picrd_cache = None
    if picrd_enabled:
        from model.pose_clip_relation import PoseClipRelationCache

        picrd_cache = PoseClipRelationCache(
            cfg.MODEL.TAPF.PICRD_CACHE,
            cfg.MODEL.TAPF.PICRD_CACHE_SHA256,
        )
        if len(picrd_cache) != len(train_loader.dataset):
            raise RuntimeError(
                "PICRD cache does not exactly cover the training dataset"
            )
        expected_picrd_paths = tuple(
            Path(record[0])
            .resolve()
            .relative_to(train_loader.dataset.pose_store.dataset_root)
            .as_posix()
            for record in train_loader.dataset.dataset
        )
        if (
            len(set(expected_picrd_paths)) != len(expected_picrd_paths)
            or set(expected_picrd_paths) != set(picrd_cache.paths)
        ):
            raise RuntimeError(
                "PICRD cache path set does not match the training dataset"
            )
        logger.info(
            "PICRD cache loaded: samples=%d SHA=%s",
            len(picrd_cache),
            picrd_cache.sha256,
        )
    generic_evidence = None
    if cfg.MODEL.TAPF.ENABLED and cfg.MODEL.TAPF.SEMANTIC_ENABLED:
        from model.clip_semantic_teacher import (
            FrozenClipSlotTeacher,
            FrozenRichClipEvidenceTeacher,
        )

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all()
        numpy_rng_state = np.random.get_state()
        python_rng_state = random.getstate()
        if rich_evidence_enabled:
            semantic_teacher = FrozenRichClipEvidenceTeacher(
                checkpoint=cfg.MODEL.TAPF.CLIP_CHECKPOINT,
                checkpoint_sha256=cfg.MODEL.TAPF.CLIP_CHECKPOINT_SHA256,
                codebook=cfg.MODEL.TAPF.RICH_CODEBOOK,
                codebook_sha256=cfg.MODEL.TAPF.RICH_CODEBOOK_SHA256,
                device=torch.device("cuda", local_rank),
                microbatch=cfg.MODEL.TAPF.CLIP_MICROBATCH,
            )
        else:
            semantic_teacher = FrozenClipSlotTeacher(
                checkpoint=cfg.MODEL.TAPF.CLIP_CHECKPOINT,
                checkpoint_sha256=cfg.MODEL.TAPF.CLIP_CHECKPOINT_SHA256,
                device=torch.device("cuda", local_rank),
                microbatch=cfg.MODEL.TAPF.CLIP_MICROBATCH,
            )
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_state)
        np.random.set_state(numpy_rng_state)
        random.setstate(python_rng_state)
        if rich_evidence_enabled:
            logger.info(
                "Frozen rich PC-MBCLS teacher loaded outside model/optimizer, checkpoint SHA: %s, codebook SHA: %s",
                semantic_teacher.checkpoint_sha256,
                semantic_teacher.codebook_sha256,
            )
        else:
            logger.info(
                "Frozen PC-MBCLS teacher loaded outside model/optimizer, SHA: %s",
                semantic_teacher.checkpoint_sha256,
            )
    if elo_cur_enabled:
        generic_evidence, generic_sha256 = _load_elo_generic_evidence(
            cfg.MODEL.TAPF.ELO_GENERIC_EVIDENCE,
            cfg.MODEL.TAPF.ELO_GENERIC_EVIDENCE_SHA256,
            cfg.DATASETS.NAMES,
            cfg.MODEL.TAPF.CLIP_CHECKPOINT_SHA256,
            cfg.MODEL.TAPF.RICH_CODEBOOK_SHA256,
            cfg.MODEL.TAPF.MANIFEST_SHA256,
            torch.device("cuda", local_rank),
        )
        logger.info(
            "Frozen train-split ELO generic evidence loaded outside model/state, SHA: %s",
            generic_sha256,
        )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    pose_loss_meter = AverageMeter()
    semantic_loss_meter = AverageMeter()
    region_mask_loss_meter = AverageMeter()
    presence_loss_meter = AverageMeter()
    q_loss_meter = AverageMeter()
    evidence_cos_loss_meter = AverageMeter()
    evidence_relation_loss_meter = AverageMeter()
    exec_loss_meter = AverageMeter()
    compatibility_loss_meter = AverageMeter()
    cur_loss_meter = AverageMeter()
    early_pose_loss_meter = AverageMeter()
    late_pose_loss_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        pose_loss_meter.reset()
        semantic_loss_meter.reset()
        region_mask_loss_meter.reset()
        presence_loss_meter.reset()
        q_loss_meter.reset()
        evidence_cos_loss_meter.reset()
        evidence_relation_loss_meter.reset()
        exec_loss_meter.reset()
        compatibility_loss_meter.reset()
        cur_loss_meter.reset()
        early_pose_loss_meter.reset()
        late_pose_loss_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, batch in enumerate(train_loader):
            if cfg.MODEL.TAPF.ENABLED:
                img, vid, target_cam, target_view, pose_batch = batch
                relative_paths = pose_batch.get("relative_paths")
                pose_batch = {
                    "keypoints": pose_batch["keypoints"].to(device),
                    "scores": pose_batch["scores"].to(device),
                    "valid": pose_batch["valid"].to(device),
                    **(
                        {"teacher_rgb": pose_batch["teacher_rgb"].to(device)}
                        if "teacher_rgb" in pose_batch
                        else {}
                    ),
                }
            else:
                img, vid, target_cam, target_view = batch
                pose_batch = None
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            if picrd_enabled:
                if relative_paths is None:
                    raise RuntimeError("PICRD batch is missing relative paths")
                clip_features, clip_valid = picrd_cache.lookup(relative_paths)
                pose_batch["clip_slot_features"] = clip_features.to(
                    device=device, non_blocking=True
                )
                pose_batch["clip_slot_valid"] = clip_valid.to(
                    device=device, non_blocking=True
                )
                pose_batch["identity"] = target
            if elo_cur_enabled:
                pose_batch["identity"] = target
                pose_batch["camera"] = target_cam
                pose_batch["generic_evidence"] = generic_evidence
            if semantic_teacher is not None:
                if "teacher_rgb" not in pose_batch:
                    raise RuntimeError("Semantic TAPF batch is missing pre-RE RGB")
                with torch.no_grad(), amp.autocast(enabled=True):
                    semantic_targets = semantic_teacher(
                        pose_batch["teacher_rgb"],
                        pose_batch["keypoints"],
                        pose_batch["scores"],
                        pose_batch["valid"],
                    )
                pose_batch["semantic_valid"] = semantic_targets[
                    "valid"
                ].detach().clone()
                pose_batch["semantic_teacher_mask"] = semantic_targets[
                    "region_masks"
                ].detach().clone()
                if rich_evidence_enabled:
                    pose_batch["semantic_teacher_evidence"] = semantic_targets[
                        "evidence_code"
                    ].detach().clone()
                else:
                    pose_batch["semantic_q_visible"] = semantic_targets[
                        "q_visible"
                    ].detach().clone()
                if (
                    not semantic_teacher_diagnostic_done
                    and rich_evidence_enabled
                ):
                    evidence = semantic_targets["evidence_code"].float()
                    evidence_valid = semantic_targets["valid"].bool()
                    valid_norm = evidence.norm(dim=-1)[evidence_valid]
                    logger.info(
                        "Rich evidence first batch: valid=%d, norm-mean=%.6f, norm-min=%.6f, norm-max=%.6f, basis-orthogonal-max-abs=%.3e",
                        int(evidence_valid.sum()),
                        valid_norm.mean().item() if valid_norm.numel() else 0.0,
                        valid_norm.min().item() if valid_norm.numel() else 0.0,
                        valid_norm.max().item() if valid_norm.numel() else 0.0,
                        semantic_teacher.basis_orthogonal_max_abs,
                    )
                    semantic_teacher_diagnostic_done = True
                if not semantic_teacher_diagnostic_done:
                    q_values = semantic_targets["q_visible"].float()
                    q_valid = semantic_targets["valid"].bool()
                    q_mean = []
                    q_std = []
                    q_entropy = []
                    q_constant_gap = []
                    for slot in range(q_values.shape[1]):
                        slot_values = q_values[:, slot][q_valid[:, slot]]
                        if slot_values.numel() == 0:
                            q_mean.append(None)
                            q_std.append(None)
                            q_entropy.append(None)
                            q_constant_gap.append(None)
                            continue
                        clipped = slot_values.clamp(1e-6, 1.0 - 1e-6)
                        mean_value = clipped.mean()
                        sample_entropy = -(
                            clipped * clipped.log()
                            + (1.0 - clipped) * (1.0 - clipped).log()
                        ).mean()
                        constant_entropy = -(
                            mean_value * mean_value.log()
                            + (1.0 - mean_value)
                            * (1.0 - mean_value).log()
                        )
                        q_mean.append(mean_value.item())
                        q_std.append(
                            clipped.std(unbiased=False).item()
                        )
                        q_entropy.append(sample_entropy.item())
                        q_constant_gap.append(
                            (constant_entropy - sample_entropy).item()
                        )
                    logger.info(
                        "Semantic q first-batch slots: mean=%s std=%s entropy=%s constant-prior-gap=%s",
                        [None if value is None else round(value, 6) for value in q_mean],
                        [None if value is None else round(value, 6) for value in q_std],
                        [None if value is None else round(value, 6) for value in q_entropy],
                        [None if value is None else round(value, 6) for value in q_constant_gap],
                    )
                    mean = torch.as_tensor(
                        cfg.INPUT.PIXEL_MEAN, device=img.device
                    ).view(1, 3, 1, 1)
                    std = torch.as_tensor(
                        cfg.INPUT.PIXEL_STD, device=img.device
                    ).view(1, 3, 1, 1)
                    post_erasing_rgb = (img.float() * std + mean).clamp(0.0, 1.0)
                    with torch.no_grad(), amp.autocast(enabled=True):
                        post_targets = semantic_teacher(
                            post_erasing_rgb,
                            pose_batch["keypoints"],
                            pose_batch["scores"],
                            pose_batch["valid"],
                        )
                    keep = semantic_targets["valid"] & post_targets["valid"]
                    valid_count = int(keep.sum().item())
                    if valid_count > 0:
                        logger.info(
                            "Semantic teacher boundary: valid=%d, preRE-q %.6f, postRE-q %.6f, paired-abs-diff %.6f",
                            valid_count,
                            semantic_targets["q_visible"][keep].mean().item(),
                            post_targets["q_visible"][keep].mean().item(),
                            (
                                semantic_targets["q_visible"][keep]
                                - post_targets["q_visible"][keep]
                            ).abs().mean().item(),
                        )
                    else:
                        logger.warning(
                            "Semantic teacher boundary: valid=0; pre/post-RE q diagnostic skipped"
                        )
                    semantic_teacher_diagnostic_done = True
            with amp.autocast(enabled=True):
                model_output = model(
                    img,
                    label=target,
                    cam_label=target_cam,
                    view_label=target_view,
                    pose_batch=pose_batch,
                    tapf_epoch=epoch,
                )
                if cfg.MODEL.TAPF.ENABLED:
                    score, feat, _, tapf_aux = model_output
                    reid_loss = loss_fn(score, feat, target, target_cam)
                    loss = reid_loss + (
                        cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * tapf_aux["pose_loss"]
                    )
                else:
                    score, feat, _ = model_output
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
            if cfg.MODEL.TAPF.ENABLED:
                pose_loss_meter.update(tapf_aux["pose_loss"].item(), img.shape[0])
                if picrd_enabled and n_iter == 0:
                    logger.info(
                        "PICRD epoch=%d first-batch loss/correct/wrong/generic/zero/rank=%.6f/%.6f/%.6f/%.6f/%.6f/%.6f shift=%d common-valid=%.4f",
                        epoch,
                        tapf_aux["picrd_loss"].item(),
                        tapf_aux["picrd_correct"].item(),
                        tapf_aux["picrd_wrong_rgb"].item(),
                        tapf_aux["picrd_generic"].item(),
                        tapf_aux["picrd_zero"].item(),
                        tapf_aux["picrd_ranking"].item(),
                        tapf_aux["picrd_wrong_shift"],
                        tapf_aux["picrd_common_valid_fraction"].item(),
                    )
                if tapf_aux.get("semantic_loss") is not None:
                    semantic_loss_meter.update(
                        tapf_aux["semantic_loss"].item(), img.shape[0]
                    )
                    region_mask_loss_meter.update(
                        tapf_aux["region_mask_loss"].item(), img.shape[0]
                    )
                    presence_loss_meter.update(
                        tapf_aux["presence_loss"].item(), img.shape[0]
                    )
                    if tapf_aux.get("q_loss") is not None:
                        q_loss_meter.update(
                            tapf_aux["q_loss"].item(), img.shape[0]
                        )
                    if tapf_aux.get("evidence_cos_loss") is not None:
                        evidence_cos_loss_meter.update(
                            tapf_aux["evidence_cos_loss"].item(), img.shape[0]
                        )
                        evidence_relation_loss_meter.update(
                            tapf_aux["evidence_relation_loss"].item(), img.shape[0]
                        )
                        if elo_cur_enabled:
                            compatibility_loss_meter.update(
                                tapf_aux["compatibility_loss"].item(),
                                img.shape[0],
                            )
                            cur_loss_meter.update(
                                tapf_aux["cur_loss"].item(), img.shape[0]
                            )
                        elif tapf_aux.get("exec_loss") is not None:
                            exec_loss_meter.update(
                                tapf_aux["exec_loss"].item(), img.shape[0]
                            )
                if cfg.MODEL.TAPF.HIERARCHICAL:
                    early_pose_loss_meter.update(
                        tapf_aux["early_pose_loss"].item(), img.shape[0]
                    )
                    late_pose_loss_meter.update(
                        tapf_aux["late_pose_loss"].item(), img.shape[0]
                    )

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        if cfg.MODEL.TAPF.ENABLED:
                            logger.info(
                                "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Pose: {:.3f}, Acc: {:.3f}, Student: {:.2f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, pose_loss_meter.avg, acc_meter.avg, tapf_aux["student_fraction"], base_lr)
                            )
                        else:
                            logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                        .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    if cfg.MODEL.TAPF.ENABLED:
                        if cfg.MODEL.TAPF.HIERARCHICAL:
                            early_gate_abs = torch.stack(
                                [
                                    delta.detach().float().abs().mean()
                                    for delta in tapf_aux["early_gate_deltas"]
                                ]
                            ).mean().item()
                            late_gate_abs = torch.stack(
                                [
                                    delta.detach().float().abs().mean()
                                    for delta in tapf_aux["late_gate_deltas"]
                                ]
                            ).mean().item()
                            logger.info(
                                "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Pose: {:.3f}, PoseEarly: {:.3f}, PoseLate: {:.3f}, Acc: {:.3f}, StudentEarly: {:.2f}, StudentLate: {:.2f}, ReliabilityEarly: {:.3f}, ReliabilityLate: {:.3f}, GateEarlyAbs: {:.3e}, GateLateAbs: {:.3e}, Base Lr: {:.2e}"
                                .format(
                                    epoch,
                                    (n_iter + 1),
                                    len(train_loader),
                                    loss_meter.avg,
                                    pose_loss_meter.avg,
                                    early_pose_loss_meter.avg,
                                    late_pose_loss_meter.avg,
                                    acc_meter.avg,
                                    tapf_aux["early_student_fraction"],
                                    tapf_aux["late_student_fraction"],
                                    tapf_aux["early_reliability"].detach().float().mean().item(),
                                    tapf_aux["late_reliability"].detach().float().mean().item(),
                                    early_gate_abs,
                                    late_gate_abs,
                                    base_lr,
                                )
                            )
                        else:
                            gate_abs = torch.stack(
                                [delta.detach().float().abs().mean() for delta in tapf_aux["gate_deltas"]]
                            ).mean().item()
                            if picrd_enabled:
                                logger.info(
                                    "Epoch[%d] Iter[%d/%d] Loss: %.3f, Pose: %.3f, PICRD: %.4f, RelC/W/G/Z: %.4f/%.4f/%.4f/%.4f, Rank: %.4f, CommonValid: %.3f, Acc: %.3f, Student: %.2f, GateAbs: %.3e, Base Lr: %.2e",
                                    epoch,
                                    n_iter + 1,
                                    len(train_loader),
                                    loss_meter.avg,
                                    pose_loss_meter.avg,
                                    tapf_aux["picrd_loss"].item(),
                                    tapf_aux["picrd_correct"].item(),
                                    tapf_aux["picrd_wrong_rgb"].item(),
                                    tapf_aux["picrd_generic"].item(),
                                    tapf_aux["picrd_zero"].item(),
                                    tapf_aux["picrd_ranking"].item(),
                                    tapf_aux["picrd_common_valid_fraction"].item(),
                                    acc_meter.avg,
                                    tapf_aux["student_fraction"],
                                    gate_abs,
                                    base_lr,
                                )
                            elif rich_evidence_enabled:
                                if spk_enabled:
                                    product_factor = tapf_aux[
                                        "semantic_product_factor"
                                    ].detach().float()
                                    product_delta = tapf_aux[
                                        "semantic_product_delta"
                                    ].detach().float()
                                    logger.info(
                                        "Epoch[%d] Iter[%d/%d] Loss: %.3f, Pose: %.3f, Semantic: %.3f, RegionMask: %.3f, Presence: %.3f, EvidenceCos: %.3f, EvidenceRel: %.3f, Acc: %.3f, Student: %.2f, Reliability: %.3f, SPKMean/Std/Min/Max: %.4f/%.4f/%.4f/%.4f, SPKDeltaAbs: %.3e, GateAbs: %.3e, Base Lr: %.2e",
                                        epoch,
                                        n_iter + 1,
                                        len(train_loader),
                                        loss_meter.avg,
                                        pose_loss_meter.avg,
                                        semantic_loss_meter.avg,
                                        region_mask_loss_meter.avg,
                                        presence_loss_meter.avg,
                                        evidence_cos_loss_meter.avg,
                                        evidence_relation_loss_meter.avg,
                                        acc_meter.avg,
                                        tapf_aux["student_fraction"],
                                        tapf_aux["reliability"].detach().float().mean().item(),
                                        product_factor.mean().item(),
                                        product_factor.std(unbiased=False).item(),
                                        product_factor.min().item(),
                                        product_factor.max().item(),
                                        product_delta.abs().mean().item(),
                                        gate_abs,
                                        base_lr,
                                    )
                                elif elo_cur_enabled:
                                    compatibility = tapf_aux[
                                        "compatibility_means"
                                    ]
                                    diagnostic_gaps = tapf_aux[
                                        "compatibility_diagnostic_gaps"
                                    ]
                                    cur_components = tapf_aux[
                                        "cur_component_losses"
                                    ]
                                    reference_utility = tapf_aux[
                                        "reference_utility_means"
                                    ]
                                    logger.info(
                                        "Epoch[%d] Iter[%d/%d] Loss: %.3f, Pose: %.3f, Semantic: %.3f, RegionMask: %.3f, Presence: %.3f, EvidenceCos: %.3f, EvidenceRel: %.3f, Compat: %.3f, CUR: %.3f, CompatC/W/G/N: %.4f/%.4f/%.4f/%.4f, RefGapWG/GN: %.4f/%.4f, CURW/G/N: %.4f/%.4f/%.4f, UtilityC/W/G/N: %.4f/%.4f/%.4f/%.4f, Eligible: %.3f, CoeffStd: %.3e, EffRank: %.3f, RNGExact: %d, Acc: %.3f, Student: %.2f, Reliability: %.3f, Rho: %.9f, BudgetAbs: %.3e, Base Lr: %.2e",
                                        epoch,
                                        n_iter + 1,
                                        len(train_loader),
                                        loss_meter.avg,
                                        pose_loss_meter.avg,
                                        semantic_loss_meter.avg,
                                        region_mask_loss_meter.avg,
                                        presence_loss_meter.avg,
                                        evidence_cos_loss_meter.avg,
                                        evidence_relation_loss_meter.avg,
                                        compatibility_loss_meter.avg,
                                        cur_loss_meter.avg,
                                        compatibility["correct"].item(),
                                        compatibility["wrong"].item(),
                                        compatibility["generic"].item(),
                                        compatibility["null"].item(),
                                        diagnostic_gaps["wrong_minus_generic"].item(),
                                        diagnostic_gaps["generic_minus_null"].item(),
                                        cur_components["wrong"].item(),
                                        cur_components["generic"].item(),
                                        cur_components["null"].item(),
                                        tapf_aux["correct_utility_mean"].item(),
                                        reference_utility["wrong"].item(),
                                        reference_utility["generic"].item(),
                                        reference_utility["null"].item(),
                                        tapf_aux["donor_eligible"].float().mean().item(),
                                        tapf_aux["coefficient_std"].item(),
                                        tapf_aux["coefficient_effective_rank"].item(),
                                        int(tapf_aux["reference_rng_exact"]),
                                        acc_meter.avg,
                                        tapf_aux["student_fraction"],
                                        tapf_aux["reliability"].detach().float().mean().item(),
                                        tapf_aux["rho"],
                                        gate_abs,
                                        base_lr,
                                    )
                                else:
                                    logger.info(
                                        "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Pose: {:.3f}, Semantic: {:.3f}, RegionMask: {:.3f}, Presence: {:.3f}, EvidenceCos: {:.3f}, EvidenceRel: {:.3f}, Exec: {:.3f}, Acc: {:.3f}, Student: {:.2f}, Reliability: {:.3f}, Rho: {:.9f}, BudgetAbs: {:.3e}, Base Lr: {:.2e}"
                                        .format(
                                            epoch,
                                            (n_iter + 1),
                                            len(train_loader),
                                            loss_meter.avg,
                                            pose_loss_meter.avg,
                                            semantic_loss_meter.avg,
                                            region_mask_loss_meter.avg,
                                            presence_loss_meter.avg,
                                            evidence_cos_loss_meter.avg,
                                            evidence_relation_loss_meter.avg,
                                            exec_loss_meter.avg,
                                            acc_meter.avg,
                                            tapf_aux["student_fraction"],
                                            tapf_aux["reliability"].detach().float().mean().item(),
                                            tapf_aux["rho"],
                                            gate_abs,
                                            base_lr,
                                        )
                                    )
                            elif cfg.MODEL.TAPF.SEMANTIC_ENABLED:
                                logger.info(
                                    "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Pose: {:.3f}, Semantic: {:.3f}, RegionMask: {:.3f}, Presence: {:.3f}, Q: {:.3f}, Acc: {:.3f}, Student: {:.2f}, Reliability: {:.3f}, GateAbs: {:.3e}, Base Lr: {:.2e}"
                                    .format(
                                        epoch,
                                        (n_iter + 1),
                                        len(train_loader),
                                        loss_meter.avg,
                                        pose_loss_meter.avg,
                                        semantic_loss_meter.avg,
                                        region_mask_loss_meter.avg,
                                        presence_loss_meter.avg,
                                        q_loss_meter.avg,
                                        acc_meter.avg,
                                        tapf_aux["student_fraction"],
                                        tapf_aux["reliability"].detach().float().mean().item(),
                                        gate_abs,
                                        base_lr,
                                    )
                                )
                            else:
                                logger.info(
                                    "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Pose: {:.3f}, Acc: {:.3f}, Student: {:.2f}, Reliability: {:.3f}, GateAbs: {:.3e}, Base Lr: {:.2e}"
                                    .format(
                                        epoch,
                                        (n_iter + 1),
                                        len(train_loader),
                                        loss_meter.avg,
                                        pose_loss_meter.avg,
                                        acc_meter.avg,
                                        tapf_aux["student_fraction"],
                                        tapf_aux["reliability"].detach().float().mean().item(),
                                        gate_abs,
                                        base_lr,
                                    )
                                )
                    else:
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
