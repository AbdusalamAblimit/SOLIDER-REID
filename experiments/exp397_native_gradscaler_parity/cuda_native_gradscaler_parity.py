#!/usr/bin/env python3
"""Matched D0/rich native GradScaler dynamics gate for exp397."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
REPORTER_DEPENDENCY_SHA256 = (
    "6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164"
)
CLIP_SHA256 = "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
CODEBOOK_SHA256 = "fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a"
RUNTIME_FREEZE_SHA256 = (
    "3d38c99c7f06502d8b40467d2674c966723e5c913d2edf962c5a7088ec60cddb"
)
EXPECTED_INITIAL_SCALE = 65536.0
ATTEMPTS = 12
TAPF_EPOCHS = (1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6)
MIN_SUCCESSFUL_UPDATES = 10
RICH_SPECIFIC_GROUPS = (
    "mask_head",
    "presence_head",
    "evidence_head",
    "router0_token_projection",
    "router0_context_projection",
    "router0_evidence_projection",
    "router0_experts",
    "router1_token_projection",
    "router1_context_projection",
    "router1_evidence_projection",
    "router1_experts",
)
EXPECTED_SOURCE_SHA256 = {
    "model/tapf.py": "95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886",
    "model/clip_semantic_teacher.py": "c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5",
    "model/make_model.py": "6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03",
    "processor/processor.py": "be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1",
    "config/defaults.py": "a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5",
    "model/backbones/swin_transformer.py": "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "datasets/pose_dataset.py": "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc",
    "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml":
        "e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e",
    "configs/occluded_duke/swin_tiny_tapf_d0.yml":
        "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_text(command, cwd=None) -> str:
    return __import__("subprocess").check_output(
        command, cwd=cwd, text=True
    ).strip()


def load_reporter(path: Path):
    if sha256_file(path) != REPORTER_DEPENDENCY_SHA256:
        raise RuntimeError("exp396 reporter dependency SHA mismatch")
    spec = importlib.util.spec_from_file_location("exp397_reporter_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def clone_cpu_batch(batch):
    image, vid, camid, viewid, pose = batch
    pose_copy = {}
    for name, value in pose.items():
        if torch.is_tensor(value):
            pose_copy[name] = value.detach().cpu().clone()
        elif isinstance(value, (list, tuple)):
            pose_copy[name] = list(value)
        else:
            pose_copy[name] = value
    return (
        image.detach().cpu().clone(),
        vid.detach().cpu().clone(),
        camid.detach().cpu().clone(),
        viewid.detach().cpu().clone(),
        pose_copy,
    )


def batch_manifest(base, batch, index):
    image, target, camid, viewid, pose = batch
    return {
        "attempt": index + 1,
        "tapf_epoch": TAPF_EPOCHS[index],
        "relative_paths": pose["relative_paths"],
        "image_sha256": pose["image_sha256"],
        "pids": target.tolist(),
        "camids": camid.tolist(),
        "viewids": viewid.tolist(),
        "input_tensor_sha256": base.sha256_tensor(image),
        "keypoints_sha256": base.sha256_tensor(pose["keypoints"]),
        "scores_sha256": base.sha256_tensor(pose["scores"]),
        "valid_sha256": base.sha256_tensor(pose["valid"]),
        "teacher_rgb_sha256": base.sha256_tensor(pose["teacher_rgb"]),
    }


def prepare_step_rng(base, device):
    base.set_seed(3971234)
    states = []
    for _ in range(ATTEMPTS):
        states.append(base.rng_state())
        torch.rand(1)
        torch.rand(1, device=device)
        np.random.random()
        random.random()
    return states


def rich_forward_losses(base, model, loss_fn, batch, pose, tapf_epoch):
    image, target, camid, viewid = batch
    score, feature, _, aux = model(
        image,
        label=target,
        cam_label=camid,
        view_label=viewid,
        pose_batch=pose,
        tapf_epoch=tapf_epoch,
    )
    reid = loss_fn(score, feature, target, camid)
    exec_losses = aux["exec_losses"]
    if len(exec_losses) != 2:
        raise RuntimeError("exp397 requires exactly two exec losses")
    exec_mean = torch.stack(exec_losses).mean()
    semantic = torch.stack(
        [
            aux["region_mask_loss"],
            aux["presence_loss"],
            aux["evidence_cos_loss"],
            aux["evidence_relation_loss"],
            exec_mean,
        ]
    ).mean()
    pose_loss = aux["heatmap_loss"] + aux["confidence_loss"] + semantic
    total = reid + 0.1 * pose_loss
    gates = {
        "exec_mean_exact": base.tensor_formula_equivalent(
            exec_mean, aux["exec_loss"]
        ),
        "semantic_exact": base.tensor_formula_equivalent(
            semantic, aux["semantic_loss"]
        ),
        "pose_exact": base.tensor_formula_equivalent(pose_loss, aux["pose_loss"]),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Rich loss formula mismatch: {gates}")
    return {
        "reid": reid,
        "heatmap": aux["heatmap_loss"],
        "confidence": aux["confidence_loss"],
        "mask": aux["region_mask_loss"],
        "presence": aux["presence_loss"],
        "evidence_cosine": aux["evidence_cos_loss"],
        "evidence_relation": aux["evidence_relation_loss"],
        "exec_consumer0": exec_losses[0],
        "exec_consumer1": exec_losses[1],
        "pose": pose_loss,
        "total": total,
    }, gates


def d0_forward_losses(base, model, loss_fn, batch, pose, tapf_epoch):
    image, target, camid, viewid = batch
    score, feature, _, aux = model(
        image,
        label=target,
        cam_label=camid,
        view_label=viewid,
        pose_batch=pose,
        tapf_epoch=tapf_epoch,
    )
    reid = loss_fn(score, feature, target, camid)
    pose_loss = aux["heatmap_loss"] + aux["confidence_loss"]
    total = reid + 0.1 * pose_loss
    gates = {
        "pose_exact": base.tensor_formula_equivalent(pose_loss, aux["pose_loss"])
    }
    if not all(gates.values()):
        raise RuntimeError(f"D0 loss formula mismatch: {gates}")
    return {
        "reid": reid,
        "heatmap": aux["heatmap_loss"],
        "confidence": aux["confidence_loss"],
        "pose": pose_loss,
        "total": total,
    }, gates


def step_key(step):
    return (
        step["attempt"],
        step["tapf_epoch"],
        step["scale_before"],
        step["scale_after"],
        step["optimizer_skipped"],
    )


def evaluate_trajectories(d0_steps, rich_steps):
    def first_success(steps):
        return next(
            (step["attempt"] for step in steps if step["optimizer_succeeded"]),
            None,
        )

    d0_success = sum(step["optimizer_succeeded"] for step in d0_steps)
    rich_success = sum(step["optimizer_succeeded"] for step in rich_steps)
    d0_first = first_success(d0_steps)
    rich_first = first_success(rich_steps)
    all_steps = d0_steps + rich_steps
    rich_specific_finite = all(
        not set(step["nonfinite_groups"]) & set(RICH_SPECIFIC_GROUPS)
        for step in rich_steps
    )
    after_first_finite = all(
        not step["had_nonfinite"]
        for steps, first in ((d0_steps, d0_first), (rich_steps, rich_first))
        if first is not None
        for step in steps
        if step["attempt"] >= first
    )
    handoff = [step for step in all_steps if step["tapf_epoch"] == 6]
    gates = {
        "row_counts_exact": len(d0_steps) == len(rich_steps) == ATTEMPTS,
        "native_step_semantics_exact": all(
            step["had_nonfinite"] == step["optimizer_skipped"]
            and (
                step["scale_after"] == step["scale_before"] * 0.5
                if step["optimizer_skipped"]
                else step["scale_after"] == step["scale_before"]
            )
            for step in all_steps
        ),
        "matched_scale_skip_trajectory": [step_key(step) for step in d0_steps]
        == [step_key(step) for step in rich_steps],
        "minimum_successful_updates": (
            d0_success >= MIN_SUCCESSFUL_UPDATES
            and rich_success >= MIN_SUCCESSFUL_UPDATES
        ),
        "first_success_matched_and_early": (
            d0_first is not None
            and rich_first is not None
            and d0_first == rich_first
            and d0_first <= 3
        ),
        "after_first_success_all_finite": after_first_finite,
        "rich_specific_groups_always_finite": rich_specific_finite,
        "handoff_all_success_and_finite": len(handoff) == 12
        and all(
            step["optimizer_succeeded"] and not step["had_nonfinite"]
            for step in handoff
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "d0_successful_updates": d0_success,
        "rich_successful_updates": rich_success,
        "d0_first_success": d0_first,
        "rich_first_success": rich_first,
        "gates": gates,
    }


def run_dynamic_arm(
    base,
    arm,
    model,
    optimizer,
    loss_fn,
    cpu_batches,
    rich_targets,
    step_rng,
    device,
    scratch_root,
):
    group_parameters, group_names, coverage = base.parameter_groups(
        model, optimizer, arm
    )
    if not coverage["exact"]:
        raise RuntimeError(f"{arm} parameter coverage failed: {coverage}")
    scaler = torch.amp.GradScaler("cuda")
    if float(scaler.get_scale()) != EXPECTED_INITIAL_SCALE:
        raise RuntimeError("Default GradScaler initial scale mismatch")
    original_step = optimizer.step
    step_counter = {"calls": 0}

    def counted_step(*args, **kwargs):
        step_counter["calls"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    rows = []
    initial_model_state = base.module_state_sha256(model)
    initial_optimizer_state = base.optimizer_state_sha256(optimizer)
    current_model_state = initial_model_state
    current_optimizer_state = initial_optimizer_state
    try:
        for index, (cpu_batch, tapf_epoch) in enumerate(
            zip(cpu_batches, TAPF_EPOCHS)
        ):
            base.restore_rng(step_rng[index])
            rng_entry = base.rng_sha256(base.rng_state())
            image, target, camid, viewid, pose = base.transfer_batch(
                cpu_batch, device
            )
            batch = (image, target, camid, viewid)
            if arm == "rich":
                target_batch = rich_targets[index]
                pose_input = {
                    "keypoints": pose["keypoints"],
                    "scores": pose["scores"],
                    "valid": pose["valid"],
                    **{
                        name: value.to(device)
                        for name, value in target_batch.items()
                    },
                }
                forward = lambda: rich_forward_losses(
                    base,
                    model,
                    loss_fn,
                    batch,
                    pose_input,
                    tapf_epoch,
                )
            else:
                pose_input = {
                    "keypoints": pose["keypoints"],
                    "scores": pose["scores"],
                    "valid": pose["valid"],
                }
                forward = lambda: d0_forward_losses(
                    base,
                    model,
                    loss_fn,
                    batch,
                    pose_input,
                    tapf_epoch,
                )
            model.train()
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            versions_before = base.parameter_versions(model)
            calls_before = step_counter["calls"]
            scale_before = float(scaler.get_scale())
            with torch.amp.autocast("cuda", enabled=True):
                losses, formula_gates = forward()
                total = losses["total"]
            components = {
                name: base.scalar_record(value) for name, value in losses.items()
            }
            if not components["total"]["finite"]:
                raise RuntimeError(f"{arm} total loss non-finite before backward")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            report = base.gradient_report(
                group_parameters,
                scratch_root,
                f"{arm}_attempt{index + 1}_epoch{tapf_epoch}_unscaled",
            )
            nonfinite_groups = [
                name for name, item in report.items() if not item["all_finite"]
            ]
            had_nonfinite = bool(nonfinite_groups)
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            calls_after = step_counter["calls"]
            versions_after = base.parameter_versions(model)
            optimizer_succeeded = calls_after == calls_before + 1
            version_changed = sum(
                versions_after[name] != versions_before[name]
                for name in versions_before
            )
            model_state_after = base.module_state_sha256(model)
            optimizer_state_after = base.optimizer_state_sha256(optimizer)
            rows.append(
                {
                    "attempt": index + 1,
                    "tapf_epoch": tapf_epoch,
                    "rng_entry_sha256": rng_entry,
                    "components": components,
                    "formula_gates": formula_gates,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "had_nonfinite": had_nonfinite,
                    "nonfinite_groups": nonfinite_groups,
                    "gradient_report": report,
                    "optimizer_succeeded": optimizer_succeeded,
                    "optimizer_skipped": not optimizer_succeeded,
                    "optimizer_step_calls_delta": calls_after - calls_before,
                    "parameter_version_changed_count": version_changed,
                    "model_state_sha256_before": current_model_state,
                    "model_state_sha256_after": model_state_after,
                    "optimizer_state_sha256_before": current_optimizer_state,
                    "optimizer_state_sha256_after": optimizer_state_after,
                }
            )
            current_model_state = model_state_after
            current_optimizer_state = optimizer_state_after
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
    finally:
        optimizer.step = original_step
    return {
        "parameter_groups": group_names,
        "parameter_coverage": coverage,
        "initial_model_state_sha256": initial_model_state,
        "final_model_state_sha256": current_model_state,
        "initial_optimizer_state_sha256": initial_optimizer_state,
        "final_optimizer_state_sha256": current_optimizer_state,
        "optimizer_step_calls": step_counter["calls"],
        "rows": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--reporter-dependency", required=True)
    parser.add_argument(
        "--rich-config",
        default="configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml",
    )
    parser.add_argument(
        "--d0-config", default="configs/occluded_duke/swin_tiny_tapf_d0.yml"
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--runtime-freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run_gate(args):
    repo_root = Path(args.repo_root).resolve()
    dependency_path = Path(args.reporter_dependency).resolve()
    base = load_reporter(dependency_path)
    output = Path(args.output).resolve()
    runner = Path(args.runner).resolve()
    manifest_path = Path(args.manifest).resolve()
    if any(path.exists() for path in (output, runner, manifest_path)):
        raise RuntimeError("Refusing to overwrite exp397 output assets")
    if run_text(["git", "rev-parse", "HEAD"], repo_root) != SOURCE_COMMIT:
        raise RuntimeError("Source commit mismatch")
    if run_text(
        ["git", "status", "--porcelain", "--untracked-files=no"], repo_root
    ):
        raise RuntimeError("Source repository has tracked modifications")
    source_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Source/config SHA mismatch")

    clip_path = Path(args.clip).resolve()
    codebook_path = Path(args.codebook).resolve()
    runtime_freeze = Path(args.runtime_freeze).resolve()
    asset_gates = {
        "clip_fresh_name": "exp397" in clip_path.name,
        "codebook_fresh_name": "exp397" in codebook_path.name,
        "clip_regular": clip_path.is_file() and not clip_path.is_symlink(),
        "codebook_regular": codebook_path.is_file() and not codebook_path.is_symlink(),
        "dependency_regular": dependency_path.is_file()
        and not dependency_path.is_symlink(),
        "runtime_regular": runtime_freeze.is_file()
        and not runtime_freeze.is_symlink(),
        "clip_sha": sha256_file(clip_path) == CLIP_SHA256,
        "codebook_sha": sha256_file(codebook_path) == CODEBOOK_SHA256,
        "dependency_sha": sha256_file(dependency_path)
        == REPORTER_DEPENDENCY_SHA256,
        "runtime_sha": sha256_file(runtime_freeze) == RUNTIME_FREEZE_SHA256,
    }
    if not all(asset_gates.values()):
        raise RuntimeError(f"Asset contract failed: {asset_gates}")
    initial_processes = base.nvidia_processes()
    if initial_processes:
        raise RuntimeError(f"GPU already has compute process: {initial_processes}")

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from loss import make_loss
    from model import make_model
    from model.clip_semantic_teacher import FrozenRichClipEvidenceTeacher
    from solver import make_optimizer
    import cv2
    import open_clip
    import timm

    runtime_gates = {
        "torch": torch.__version__ == "2.6.0+cu124",
        "open_clip": open_clip.__version__ == "3.3.0",
        "opencv": cv2.__version__ == "4.13.0",
        "timm": timm.__version__ == "1.0.27",
    }
    if not all(runtime_gates.values()):
        raise RuntimeError(f"Runtime mismatch: {runtime_gates}")

    rich_cfg = base_cfg.clone()
    rich_cfg.merge_from_file(str(repo_root / args.rich_config))
    rich_cfg.defrost()
    rich_cfg.MODEL.TAPF.CLIP_CHECKPOINT = str(clip_path)
    rich_cfg.MODEL.TAPF.RICH_CODEBOOK = str(codebook_path)
    rich_cfg.freeze()
    d0_cfg = base_cfg.clone()
    d0_cfg.merge_from_file(str(repo_root / args.d0_config))
    d0_cfg.freeze()
    config_gates = {
        "batch64": int(rich_cfg.SOLVER.IMS_PER_BATCH)
        == int(d0_cfg.SOLVER.IMS_PER_BATCH)
        == 64,
        "seed1234": int(rich_cfg.SOLVER.SEED)
        == int(d0_cfg.SOLVER.SEED)
        == 1234,
        "workers8": int(rich_cfg.DATALOADER.NUM_WORKERS) == 8,
        "rich_enabled": bool(rich_cfg.MODEL.TAPF.RICH_EVIDENCE_ENABLED),
        "d0_nonsemantic": bool(d0_cfg.MODEL.TAPF.ENABLED)
        and not bool(d0_cfg.MODEL.TAPF.SEMANTIC_ENABLED),
    }
    if not all(config_gates.values()):
        raise RuntimeError(f"Config mismatch: {config_gates}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("exp397 requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.init()
    if "4090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Exclusive RTX 4090 required")
    torch.cuda.reset_peak_memory_stats(device)
    base.set_seed(1234)
    train_loader, _, _, _, num_classes, camera_num, view_num = make_dataloader(
        rich_cfg
    )
    iterator = iter(train_loader)
    cpu_batches = [clone_cpu_batch(next(iterator)) for _ in range(ATTEMPTS)]
    manifest = {
        "attempts": ATTEMPTS,
        "tapf_epochs": list(TAPF_EPOCHS),
        "batches": [
            batch_manifest(base, batch, index)
            for index, batch in enumerate(cpu_batches)
        ],
    }
    write_json(manifest_path, manifest)

    base.set_seed(1234)
    d0_model = make_model(
        d0_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=d0_cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    d0_loss_fn, d0_center = make_loss(d0_cfg, num_classes=num_classes)
    d0_optimizer, _ = make_optimizer(d0_cfg, d0_model, d0_center)
    base.set_seed(1234)
    rich_model = make_model(
        rich_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=rich_cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    rich_loss_fn, rich_center = make_loss(rich_cfg, num_classes=num_classes)
    rich_optimizer, _ = make_optimizer(rich_cfg, rich_model, rich_center)
    common_parity = base.common_state_parity(d0_model, rich_model)
    if not common_parity["exact"]:
        raise RuntimeError(f"Initial common state mismatch: {common_parity}")

    teacher_rng = base.rng_state()
    teacher = FrozenRichClipEvidenceTeacher(
        checkpoint=str(clip_path),
        checkpoint_sha256=CLIP_SHA256,
        codebook=str(codebook_path),
        codebook_sha256=CODEBOOK_SHA256,
        device=device,
        microbatch=rich_cfg.MODEL.TAPF.CLIP_MICROBATCH,
    )
    base.restore_rng(teacher_rng)
    teacher_versions_before = base.teacher_versions(teacher)
    teacher_state_before = base.module_state_sha256(teacher.visual)
    codebook_state_before = base.sha256_json(
        {
            "slot_means": base.sha256_tensor(teacher.slot_means),
            "shared_basis": base.sha256_tensor(teacher.shared_basis),
        }
    )
    rich_targets = []
    target_sha = []
    target_gates = []
    for cpu_batch in cpu_batches:
        _, _, _, _, pose = base.transfer_batch(cpu_batch, device)
        targets, raw_targets = base.teacher_targets(teacher, pose)
        valid = raw_targets["valid"].bool()
        valid_norm = raw_targets["evidence_code"].float().norm(dim=-1)[valid]
        gates = {
            "evidence_shape": tuple(raw_targets["evidence_code"].shape)
            == (64, 5, 16),
            "mask_shape": tuple(raw_targets["region_masks"].shape)
            == (64, 5, 96, 32),
            "valid_shape": tuple(valid.shape) == (64, 5),
            "finite": all(
                bool(torch.isfinite(value).all()) for value in targets.values()
            ),
            "invalid_zero": torch.equal(
                raw_targets["evidence_code"][~valid],
                torch.zeros_like(raw_targets["evidence_code"][~valid]),
            ),
            "valid_norm": valid_norm.numel() > 0
            and float((valid_norm - 1.0).abs().max()) < 1e-5,
        }
        if not all(gates.values()):
            raise RuntimeError(f"Teacher target contract failed: {gates}")
        target_gates.append(gates)
        target_sha.append(
            {name: base.sha256_tensor(value) for name, value in targets.items()}
        )
        rich_targets.append(
            {name: value.detach().cpu().clone() for name, value in targets.items()}
        )

    step_rng = prepare_step_rng(base, device)
    step_rng_sha = [base.rng_sha256(state) for state in step_rng]
    checkpoint_before = sorted(str(path) for path in output.parent.glob("*.pth"))
    scratch_prefix = ".exp397_gradient_scratch_"
    scratch_before = sorted(
        str(path) for path in output.parent.glob(f"{scratch_prefix}*")
    )
    if scratch_before:
        raise RuntimeError(f"Stale exp397 scratch exists: {scratch_before}")
    began = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=scratch_prefix, dir=output.parent
    ) as scratch_directory:
        scratch_root = Path(scratch_directory)
        d0_result = run_dynamic_arm(
            base,
            "d0",
            d0_model,
            d0_optimizer,
            d0_loss_fn,
            cpu_batches,
            rich_targets,
            step_rng,
            device,
            scratch_root,
        )
        rich_result = run_dynamic_arm(
            base,
            "rich",
            rich_model,
            rich_optimizer,
            rich_loss_fn,
            cpu_batches,
            rich_targets,
            step_rng,
            device,
            scratch_root,
        )
        if any(scratch_root.iterdir()):
            raise RuntimeError("exp397 reporter scratch not empty")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - began
    scratch_after = sorted(
        str(path) for path in output.parent.glob(f"{scratch_prefix}*")
    )
    evaluation = evaluate_trajectories(d0_result["rows"], rich_result["rows"])
    teacher_versions_after = base.teacher_versions(teacher)
    teacher_state_after = base.module_state_sha256(teacher.visual)
    codebook_state_after = base.sha256_json(
        {
            "slot_means": base.sha256_tensor(teacher.slot_means),
            "shared_basis": base.sha256_tensor(teacher.shared_basis),
        }
    )
    checkpoint_after = sorted(str(path) for path in output.parent.glob("*.pth"))
    validity_gates = {
        "source_runtime_assets_exact": all(asset_gates.values())
        and all(runtime_gates.values())
        and all(config_gates.values()),
        "common_initial_state_exact": common_parity["exact"],
        "parameter_coverage_exact": d0_result["parameter_coverage"]["exact"]
        and rich_result["parameter_coverage"]["exact"],
        "target_contract_exact": all(all(gate.values()) for gate in target_gates),
        "step_rng_entries_matched": [
            row["rng_entry_sha256"] for row in d0_result["rows"]
        ]
        == [row["rng_entry_sha256"] for row in rich_result["rows"]]
        == step_rng_sha,
        "teacher_versions_exact": teacher_versions_before
        == teacher_versions_after,
        "teacher_state_exact": teacher_state_before == teacher_state_after,
        "codebook_state_exact": codebook_state_before == codebook_state_after,
        "scratch_cleanup_exact": scratch_before == scratch_after == [],
        "checkpoint_zero": checkpoint_before == checkpoint_after == [],
        "trajectory_pass": evaluation["status"] == "PASS",
    }
    return {
        "status": "PASS" if all(validity_gates.values()) else "FAIL",
        "diagnostic_outcome": (
            "MATCHED_NATIVE_GRADSCALER_PASS"
            if evaluation["status"] == "PASS"
            else "NATIVE_GRADSCALER_PARITY_FAIL"
        ),
        "source_commit": SOURCE_COMMIT,
        "source_sha256": source_sha,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "reporter_dependency_sha256": sha256_file(dependency_path),
        "asset_gates": asset_gates,
        "runtime_gates": runtime_gates,
        "config_gates": config_gates,
        "device": torch.cuda.get_device_name(device),
        "attempts": ATTEMPTS,
        "tapf_epochs": list(TAPF_EPOCHS),
        "batch_manifest_sha256": sha256_file(manifest_path),
        "teacher_target_sha256": target_sha,
        "teacher_target_gates": target_gates,
        "common_initial_state_parity": common_parity,
        "d0": d0_result,
        "rich": rich_result,
        "trajectory_evaluation": evaluation,
        "teacher_state_sha256_before": teacher_state_before,
        "teacher_state_sha256_after": teacher_state_after,
        "codebook_state_sha256_before": codebook_state_before,
        "codebook_state_sha256_after": codebook_state_after,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "checkpoint_count": len(checkpoint_after),
        "scratch_before": scratch_before,
        "scratch_after": scratch_after,
        "in_process_final_gpu_processes": base.nvidia_processes(),
        "post_exit_gpu_audit_required": True,
        "validity_gates": validity_gates,
        "exp394_remains_sealed": True,
        "exp395_remains_sealed": True,
        "exp396_remains_sealed": True,
        "formal_training_authorized": False,
    }


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    runner = Path(args.runner).resolve()
    try:
        result = run_gate(args)
        exit_code = 0 if result["status"] == "PASS" else 1
    except Exception as error:
        result = {
            "status": "INVALID",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "formal_training_authorized": False,
            "exp394_remains_sealed": True,
            "exp395_remains_sealed": True,
            "exp396_remains_sealed": True,
        }
        exit_code = 1
    if output.exists() or runner.exists():
        raise RuntimeError("Refusing to overwrite exp397 result/runner")
    write_json(output, result)
    write_json(runner, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
