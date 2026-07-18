#!/usr/bin/env python3
"""Real-runtime CUDA/AMP preflight for the first semantic TAPF training arm."""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def move_pose_batch(pose_batch, device):
    return {
        "keypoints": pose_batch["keypoints"].to(device),
        "scores": pose_batch["scores"].to(device),
        "valid": pose_batch["valid"].to(device),
        "teacher_rgb": pose_batch["teacher_rgb"].to(device),
    }


def make_teacher_targets(teacher, pose_batch):
    with torch.no_grad(), amp.autocast(enabled=True):
        values = teacher(
            pose_batch["teacher_rgb"],
            pose_batch["keypoints"],
            pose_batch["scores"],
            pose_batch["valid"],
        )
    pose_batch["semantic_q_visible"] = values["q_visible"].detach().clone()
    pose_batch["semantic_valid"] = values["valid"].detach().clone()
    pose_batch["semantic_teacher_mask"] = values[
        "region_masks"
    ].detach().clone()
    return values


def reference_parity(
    teacher,
    targets,
    pose_batch,
    phase0b,
    ontology,
    pcmbcls,
):
    count = min(4, len(pose_batch["teacher_rgb"]))
    renderer = ontology.ExclusiveRegionRenderer(phase0b, "hard-owner")
    reference_masks = []
    reference_pose_valid = []
    for index in range(count):
        mask, _, valid = renderer(
            pose_batch["keypoints"][index].cpu(),
            pose_batch["scores"][index].cpu(),
            pose_batch["valid"][index].cpu(),
        )
        reference_masks.append(mask)
        reference_pose_valid.append(valid)
    reference_masks = torch.stack(reference_masks).to(teacher.device)
    reference_pose_valid = torch.stack(reference_pose_valid).to(teacher.device)
    mask_max_abs = float(
        (reference_masks - targets["region_masks"][:count]).abs().max().item()
    )

    rgb = pose_batch["teacher_rgb"][:count].float()
    resized = F.interpolate(
        rgb,
        size=(224, 75),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    clip_images = teacher.mean.expand(count, 3, 224, 224).clone()
    clip_images[:, :, :, 74:149] = resized.clamp(0.0, 1.0)
    region_masks = F.interpolate(
        reference_masks.float(), size=(224, 75), mode="nearest"
    )
    clip_masks = torch.zeros(
        count, 5, 224, 224, device=teacher.device, dtype=torch.float32
    )
    clip_masks[:, :, :, 74:149] = region_masks
    grid_masks = F.avg_pool2d(clip_masks, kernel_size=14, stride=14)
    normalized = (clip_images - teacher.mean) / teacher.std
    with torch.no_grad(), amp.autocast(enabled=True):
        shared = pcmbcls.forward_shared_trunk(teacher.visual, normalized)
        region_features, readout_valid = pcmbcls.forward_regions(
            teacher.visual, shared.clone(), grid_masks
        )
        logits = torch.einsum(
            "brd,rsd->brs", region_features, teacher.text
        )
        reference_q = torch.softmax(logits / 0.07, dim=-1)[..., 0]
    reference_valid = readout_valid & reference_pose_valid
    reference_q = torch.where(
        reference_valid, reference_q.float(), torch.zeros_like(reference_q.float())
    )
    q_max_abs = float(
        (reference_q - targets["q_visible"][:count]).abs().max().item()
    )
    valid_exact = bool(
        torch.equal(reference_valid, targets["valid"][:count])
    )
    return {
        "mask_max_abs": mask_max_abs,
        "q_max_abs": q_max_abs,
        "valid_exact": valid_exact,
        "pass": mask_max_abs <= 1e-6 and q_max_abs <= 1e-6 and valid_exact,
    }


def summarize_q(q_batches, valid_batches):
    q = torch.cat(q_batches, dim=0)
    valid = torch.cat(valid_batches, dim=0)
    slots = []
    for slot in range(q.shape[1]):
        values = q[:, slot][valid[:, slot]].double().clamp(1e-8, 1.0 - 1e-8)
        if values.numel() == 0:
            slots.append({"valid": 0})
            continue
        mean = values.mean()
        entropy = -(
            values * values.log()
            + (1.0 - values) * (1.0 - values).log()
        ).mean()
        constant_entropy = -(
            mean * mean.log() + (1.0 - mean) * (1.0 - mean).log()
        )
        slots.append({
            "valid": int(values.numel()),
            "mean": float(mean),
            "std": float(values.std(unbiased=False)),
            "entropy": float(entropy),
            "constant_prior_bce_gap": float(constant_entropy - entropy),
        })
    return slots


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase0b-script", required=True)
    parser.add_argument("--ontology-script", required=True)
    parser.add_argument("--pcmbcls-script", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=24)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    os.chdir(str(repo_root))
    import sys

    sys.path.insert(0, str(repo_root))
    import open_clip
    from config import cfg as default_cfg
    from datasets import make_dataloader
    from loss import make_loss
    from model import make_model
    from model.clip_semantic_teacher import FrozenClipSlotTeacher
    from solver import make_optimizer

    cfg = default_cfg.clone()
    cfg.merge_from_file(str(Path(args.config).resolve()))
    cfg.freeze()
    if not cfg.MODEL.TAPF.SEMANTIC_ENABLED:
        raise RuntimeError("Semantic TAPF config is not enabled")
    if cfg.SOLVER.IMS_PER_BATCH != 64 or cfg.DATALOADER.NUM_WORKERS != 8:
        raise RuntimeError("Preflight requires formal batch64/8-worker config")
    formal_output = (repo_root / cfg.OUTPUT_DIR).resolve()
    if formal_output.exists():
        raise RuntimeError("Formal output already exists: %s" % formal_output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.__version__.startswith("1.13.1"):
        raise RuntimeError("Unexpected Torch runtime: %s" % torch.__version__)
    if open_clip.__version__ != "2.32.0":
        raise RuntimeError("Unexpected OpenCLIP runtime: %s" % open_clip.__version__)

    set_seed(cfg.SOLVER.SEED)
    train_loader, _, val_loader, _, num_classes, camera_num, view_num = (
        make_dataloader(cfg)
    )
    if train_loader.batch_size != 64 or train_loader.num_workers != 8:
        raise RuntimeError("Constructed loader does not preserve batch64/8 workers")
    device = torch.device("cuda", 0)
    teacher = FrozenClipSlotTeacher(
        checkpoint=cfg.MODEL.TAPF.CLIP_CHECKPOINT,
        checkpoint_sha256=cfg.MODEL.TAPF.CLIP_CHECKPOINT_SHA256,
        device=device,
        microbatch=cfg.MODEL.TAPF.CLIP_MICROBATCH,
    )
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    teacher_parameter_ids = {
        id(parameter) for parameter in teacher.visual.parameters()
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    teacher_isolated = bool(
        model_parameter_ids.isdisjoint(teacher_parameter_ids)
        and optimizer_parameter_ids.isdisjoint(teacher_parameter_ids)
    )
    state_keys = tuple(model.state_dict())
    teacher_absent_from_state = not any(
        "clip" in key.lower() or "teacher" in key.lower() for key in state_keys
    )

    phase0b = load_module("exp392_phase0b_preflight", args.phase0b_script)
    ontology = load_module("exp392_ontology_preflight", args.ontology_script)
    pcmbcls = load_module("exp392_pcmbcls_preflight", args.pcmbcls_script)

    scaler = amp.GradScaler()
    train_iterator = iter(train_loader)
    q_batches = []
    valid_batches = []
    q_head_grad = []
    consumer_grad = [[], []]
    scale_history = []
    loss_history = []
    step_records = []
    successful_updates = 0
    consecutive_updates = 0
    longest_consecutive_updates = 0
    parity = None
    last_aux = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    epoch_schedule = (1, 6, 10, 11)
    model.train()
    for step in range(args.steps):
        batch, train_iterator = next_batch(train_iterator, train_loader)
        img, vid, target_cam, target_view, raw_pose_batch = batch
        pose_batch = move_pose_batch(raw_pose_batch, device)
        targets = make_teacher_targets(teacher, pose_batch)
        if parity is None:
            parity = reference_parity(
                teacher, targets, pose_batch, phase0b, ontology, pcmbcls
            )
        q_batches.append(targets["q_visible"].detach().cpu())
        valid_batches.append(targets["valid"].detach().cpu())
        img = img.to(device)
        target = vid.to(device)
        target_cam = target_cam.to(device)
        target_view = target_view.to(device)
        optimizer.zero_grad(set_to_none=True)
        epoch = epoch_schedule[(step * len(epoch_schedule)) // args.steps]
        scale_before = float(scaler.get_scale())
        probes = {
            "q_head": model.base.tapf.anchor.support_head.weight,
            "consumer_0": model.base.tapf.psg_bank[0].expert,
            "consumer_1": model.base.tapf.psg_bank[1].expert,
            "backbone": model.base.patch_embed.projection.weight,
            "head": model.classifier.weight,
        }
        probe_before = {
            name: parameter.detach().clone()
            for name, parameter in probes.items()
        }
        with amp.autocast(enabled=True):
            score, feat, _, aux = model(
                img,
                label=target,
                cam_label=target_cam,
                view_label=target_view,
                pose_batch=pose_batch,
                tapf_epoch=epoch,
            )
            reid_loss = loss_fn(score, feat, target, target_cam)
            loss = reid_loss + cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * aux["pose_loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite loss at step %d" % step)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        found_inf = float(sum(
            value.detach().float().item()
            for value in scaler._per_optimizer_states[id(optimizer)][
                "found_inf_per_device"
            ].values()
        ))
        q_parameter = model.base.tapf.anchor.support_head.weight
        current_q_grad = float(q_parameter.grad.detach().float().norm().item())
        current_consumer_grad = []
        for index in range(2):
            gradient = model.base.tapf.psg_bank[index].expert.grad
            current_consumer_grad.append(
                0.0 if gradient is None else float(gradient.detach().float().norm().item())
            )
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        updated = {
            name: not torch.equal(parameter.detach(), probe_before[name])
            for name, parameter in probes.items()
        }
        if found_inf > 0.0:
            if scale_after >= scale_before or any(updated.values()):
                raise RuntimeError(
                    "Overflow step did not perform an exact skip at index %d" % step
                )
            consecutive_updates = 0
        else:
            if scale_after != scale_before or not all(updated.values()):
                raise RuntimeError(
                    "Finite step failed to update all probes at index %d: %s"
                    % (step, updated)
                )
            if not np.isfinite(current_q_grad) or current_q_grad <= 0.0:
                raise RuntimeError("Finite step has invalid q-head gradient")
            if any(
                not np.isfinite(value) or value <= 0.0
                for value in current_consumer_grad
            ):
                raise RuntimeError("Finite step missed a semantic consumer")
            q_head_grad.append(current_q_grad)
            for index, value in enumerate(current_consumer_grad):
                consumer_grad[index].append(value)
            successful_updates += 1
            consecutive_updates += 1
            longest_consecutive_updates = max(
                longest_consecutive_updates, consecutive_updates
            )
        scale_history.append(scale_after)
        loss_history.append(float(loss.detach().item()))
        step_records.append({
            "step": step + 1,
            "epoch_route": epoch,
            "loss": float(loss.detach().item()),
            "scale_before": scale_before,
            "scale_after": scale_after,
            "found_inf": found_inf,
            "updated": updated,
            "q_head_gradient_norm": current_q_grad,
            "consumer_gradient_norm": current_consumer_grad,
        })
        last_aux = aux
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - began
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    if last_aux is None:
        raise RuntimeError("No training steps executed")
    if successful_updates < 1:
        raise RuntimeError("Default GradScaler never reached a finite update")
    if longest_consecutive_updates < 8:
        raise RuntimeError("Fewer than eight consecutive finite updates")
    teacher_mask = last_aux["teacher_mask"].detach().float()
    student_mask = last_aux["student_mask"].detach().float()
    maximum = teacher_mask.flatten(2).amax(dim=-1)[..., None, None]
    foreground = teacher_mask > (0.05 * maximum)
    background = ~foreground
    mask_diagnostic = {
        "teacher_mass_mean": float(teacher_mask.flatten(2).sum(-1).mean().item()),
        "student_mass_mean": float(student_mask.flatten(2).sum(-1).mean().item()),
        "student_foreground_mean": float(student_mask[foreground].mean().item()),
        "student_background_mean": float(student_mask[background].mean().item()),
    }

    model.eval()
    validation = next(iter(val_loader))[0].to(device)
    with torch.no_grad(), amp.autocast(enabled=True):
        eval_feature, _ = model(validation)
    eval_rgb_only_finite = bool(torch.isfinite(eval_feature).all())
    model.train()

    state = model.state_dict()
    state_finite = all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in state.values()
    )
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
        checkpoint_path = Path(handle.name)
    try:
        torch.save(state, checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        reloaded = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
        )
        reloaded.load_state_dict(checkpoint_state, strict=True)
        strict_reload = True
    finally:
        checkpoint_path.unlink(missing_ok=True)

    q_summary = summarize_q(q_batches, valid_batches)
    gates = {
        "runtime_versions": (
            torch.__version__.startswith("1.13.1")
            and open_clip.__version__ == "2.32.0"
        ),
        "pcmbcls_parity": bool(parity["pass"]),
        "teacher_isolated": teacher_isolated and teacher_absent_from_state,
        "finite_recovery": (
            all(np.isfinite(loss_history))
            and all(np.isfinite(scale_history))
            and state_finite
            and successful_updates >= 1
            and longest_consecutive_updates >= 8
        ),
        "q_head_gradient": min(q_head_grad) > 0.0,
        "two_consumers": all(min(values) > 0.0 for values in consumer_grad),
        "memory": peak_reserved < 24 * 1024 ** 3,
        "rgb_only_eval": eval_rgb_only_finite,
        "checkpoint_strict": strict_reload,
        "loader": train_loader.batch_size == 64 and train_loader.num_workers == 8,
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "parity": parity,
        "q_slots": q_summary,
        "q_head_gradient_norm": {
            "min": min(q_head_grad),
            "max": max(q_head_grad),
            "mean": float(np.mean(q_head_grad)),
        },
        "consumer_expert_gradient_norm": [
            {"min": min(values), "max": max(values), "mean": float(np.mean(values))}
            for values in consumer_grad
        ],
        "mask_diagnostic": mask_diagnostic,
        "steps": args.steps,
        "successful_updates": successful_updates,
        "overflow_steps": args.steps - successful_updates,
        "longest_consecutive_updates": longest_consecutive_updates,
        "step_records": step_records,
        "batch_size": train_loader.batch_size,
        "workers": train_loader.num_workers,
        "elapsed_seconds": elapsed,
        "samples_per_second": args.steps * train_loader.batch_size / elapsed,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "state_tensor_count": len(state),
        "state_finite": state_finite,
        "teacher_absent_from_state": teacher_absent_from_state,
        "checkpoint_sha256": checkpoint_sha256,
        "eval_rgb_only_finite": eval_rgb_only_finite,
        "torch_version": torch.__version__,
        "open_clip_version": open_clip.__version__,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(__file__),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
