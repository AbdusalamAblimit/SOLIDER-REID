#!/usr/bin/env python3
"""Final rich-budget production CUDA preflight for exp400."""

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
ATTEMPTS = 32
STAGE_LENGTH = 16
STAGE_TAIL = 8
TAPF_EPOCHS = (1,) * STAGE_LENGTH + (6,) * STAGE_LENGTH
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
    spec = importlib.util.spec_from_file_location("exp400_reporter_base", path)
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
    base.set_seed(4001234)
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
        raise RuntimeError("exp400 requires exactly two exec losses")
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


def tensor_state_cpu(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def tensor_state_sha256(base, state) -> str:
    return base.sha256_json(
        [
            {"name": name, "tensor_sha256": base.sha256_tensor(value)}
            for name, value in sorted(state.items())
        ]
    )


def rich_pose_batch(base, cpu_batch, cpu_targets, device):
    _, _, _, _, pose = base.transfer_batch(cpu_batch, device)
    return {
        "keypoints": pose["keypoints"],
        "scores": pose["scores"],
        "valid": pose["valid"],
        **{name: value.to(device) for name, value in cpu_targets.items()},
    }


def descriptor_variant(
    base,
    model,
    model_state,
    saved_rng,
    image,
    target,
    camid,
    viewid,
    pose,
    epoch,
    bypass=(),
):
    model.load_state_dict(model_state, strict=True)
    base.restore_rng(saved_rng)
    model.train()
    tapf = model.base.tapf
    original = tapf.apply_gate
    had_instance_override = "apply_gate" in tapf.__dict__
    instance_override = tapf.__dict__.get("apply_gate")

    def selective(bank_index, tokens, hw_shape, state):
        if bank_index in bypass:
            state["gate_deltas"].append(torch.zeros_like(tokens))
            return tokens
        return original(bank_index, tokens, hw_shape, state)

    tapf.apply_gate = selective
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            output = model(
                image,
                label=target,
                cam_label=camid,
                view_label=viewid,
                pose_batch=pose,
                tapf_epoch=epoch,
            )
        descriptor = output[1].detach().clone()
    finally:
        if had_instance_override:
            tapf.__dict__["apply_gate"] = instance_override
        else:
            tapf.__dict__.pop("apply_gate", None)
        model.load_state_dict(model_state, strict=True)
        base.restore_rng(saved_rng)
        model.zero_grad(set_to_none=True)
    return descriptor


class ExplodingPose(dict):
    accesses = 0

    def __getitem__(self, key):
        type(self).accesses += 1
        raise RuntimeError("eval accessed external pose")

    def get(self, key, default=None):
        type(self).accesses += 1
        raise RuntimeError("eval accessed external pose")

    def __iter__(self):
        type(self).accesses += 1
        raise RuntimeError("eval iterated external pose")


def eval_descriptor(model, image, camid, viewid, pose_batch):
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        descriptor, _ = model(
            image,
            cam_label=camid,
            view_label=viewid,
            pose_batch=pose_batch,
        )
    return descriptor.detach().clone()


def parameter_group_state(base, group_parameters):
    state = {}
    for group_name, items in group_parameters.items():
        records = []
        seen = set()
        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("Parameter group item must be a (name, parameter) tuple")
            parameter_name, parameter = item
            if not isinstance(parameter_name, str) or not parameter_name:
                raise TypeError("Parameter group name must be a non-empty string")
            if parameter_name in seen:
                raise ValueError(f"Duplicate parameter name in group: {parameter_name}")
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError("Parameter group value must be torch.nn.Parameter")
            seen.add(parameter_name)
            records.append(
                {
                    "name": parameter_name,
                    "tensor_sha256": base.sha256_tensor(parameter.detach()),
                }
            )
        state[group_name] = base.sha256_json(records)
    return state


def evaluate_trajectories(d0_steps, rich_steps, rich_group_state):
    def first_success(steps):
        return next(
            (step["attempt"] for step in steps if step["optimizer_succeeded"]),
            None,
        )

    def stage_tail(steps, epoch):
        stage = [step for step in steps if step["tapf_epoch"] == epoch]
        return stage[-STAGE_TAIL:]

    def native_semantics(steps):
        if not steps or steps[0]["scale_before"] != EXPECTED_INITIAL_SCALE:
            return False
        for index, step in enumerate(steps):
            if step["had_nonfinite"] != step["optimizer_skipped"]:
                return False
            expected_delta = 0 if step["optimizer_skipped"] else 1
            if step.get("optimizer_step_calls_delta", expected_delta) != expected_delta:
                return False
            expected_after = (
                step["scale_before"] * 0.5
                if step["optimizer_skipped"]
                else step["scale_before"]
            )
            if step["scale_after"] != expected_after:
                return False
            if index and step["scale_before"] != steps[index - 1]["scale_after"]:
                return False
        return True

    all_steps = d0_steps + rich_steps
    pairs = list(zip(d0_steps, rich_steps))
    d0_success = sum(step["optimizer_succeeded"] for step in d0_steps)
    rich_success = sum(step["optimizer_succeeded"] for step in rich_steps)
    extra_rich_skips = [
        rich["attempt"]
        for d0, rich in pairs
        if d0["optimizer_succeeded"] and rich["optimizer_skipped"]
    ]
    rich_nonfinite_not_shared = {
        rich["attempt"]: sorted(
            set(rich["nonfinite_groups"]) - set(d0["nonfinite_groups"])
        )
        for d0, rich in pairs
        if set(rich["nonfinite_groups"]) - set(d0["nonfinite_groups"])
    }
    rich_specific_finite = all(
        not set(step["nonfinite_groups"]) & set(RICH_SPECIFIC_GROUPS)
        for step in rich_steps
    )
    rich_specific_active = {
        name: any(
            step["tapf_epoch"] == 6
            and step["optimizer_succeeded"]
            and step["gradient_report"][name]["all_finite"]
            and step["gradient_report"][name]["grad_nonzero_tensors"] > 0
            for step in rich_steps
        )
        for name in RICH_SPECIFIC_GROUPS
    }
    initial_group_state = rich_group_state["initial"]
    final_group_state = rich_group_state["final"]
    rich_specific_updated = {
        name: initial_group_state[name] != final_group_state[name]
        for name in RICH_SPECIFIC_GROUPS
    }
    stage_tail_gates = {
        f"{arm}_e{epoch}_tail{STAGE_TAIL}": len(stage_tail(steps, epoch))
        == STAGE_TAIL
        and all(
            step["optimizer_succeeded"] and not step["had_nonfinite"]
            for step in stage_tail(steps, epoch)
        )
        for arm, steps in (("d0", d0_steps), ("rich", rich_steps))
        for epoch in (1, 6)
    }
    gates = {
        "row_counts_and_schedule_exact": (
            len(d0_steps) == len(rich_steps) == ATTEMPTS
            and [step["tapf_epoch"] for step in d0_steps]
            == [step["tapf_epoch"] for step in rich_steps]
            == list(TAPF_EPOCHS)
        ),
        "native_step_semantics_exact": native_semantics(d0_steps)
        and native_semantics(rich_steps),
        "stage_tail_steady_state": all(stage_tail_gates.values()),
        "no_rich_extra_skip_on_d0_success": not extra_rich_skips,
        "rich_success_not_below_d0": rich_success >= d0_success,
        "rich_nonfinite_groups_shared_subset": not rich_nonfinite_not_shared,
        "rich_specific_groups_always_finite": rich_specific_finite,
        "rich_specific_groups_e6_active": all(rich_specific_active.values()),
        "rich_specific_group_state_updated": all(rich_specific_updated.values()),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "d0_successful_updates": d0_success,
        "rich_successful_updates": rich_success,
        "d0_first_success": first_success(d0_steps),
        "rich_first_success": first_success(rich_steps),
        "extra_rich_skip_attempts": extra_rich_skips,
        "rich_nonfinite_not_shared": rich_nonfinite_not_shared,
        "stage_tail_gates": stage_tail_gates,
        "rich_specific_active": rich_specific_active,
        "rich_specific_updated": rich_specific_updated,
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
    initial_group_state = parameter_group_state(base, group_parameters)
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
        "group_state_sha256": {
            "initial": initial_group_state,
            "final": parameter_group_state(base, group_parameters),
        },
        "optimizer_step_calls": step_counter["calls"],
        "rows": rows,
    }


def run_terminal_audit(
    base,
    model,
    cfg,
    make_model,
    num_classes,
    camera_num,
    view_num,
    cpu_batches,
    rich_targets,
    teacher,
    teacher_versions_before,
    teacher_state_before,
    codebook_state_before,
    repo_root,
    clip_path,
    codebook_path,
    dependency_path,
    runtime_freeze,
    device,
):
    final_state = tensor_state_cpu(model)
    final_state_sha = tensor_state_sha256(base, final_state)
    diagnostic_rng = base.rng_state()
    diagnostic_rng_sha = base.rng_sha256(diagnostic_rng)
    tapf = model.base.tapf
    apply_gate_override_before = tapf.__dict__.get("apply_gate")
    apply_gate_had_override_before = "apply_gate" in tapf.__dict__

    image1, target1, camid1, viewid1, _ = base.transfer_batch(
        cpu_batches[0], device
    )
    pose1 = rich_pose_batch(base, cpu_batches[0], rich_targets[0], device)
    image6, target6, camid6, viewid6, _ = base.transfer_batch(
        cpu_batches[STAGE_LENGTH], device
    )
    pose6 = rich_pose_batch(
        base,
        cpu_batches[STAGE_LENGTH],
        rich_targets[STAGE_LENGTH],
        device,
    )
    reloaded = None
    try:
        epoch1_full = descriptor_variant(
            base,
            model,
            final_state,
            diagnostic_rng,
            image1,
            target1,
            camid1,
            viewid1,
            pose1,
            epoch=1,
            bypass=(),
        )
        epoch1_all_bypass = descriptor_variant(
            base,
            model,
            final_state,
            diagnostic_rng,
            image1,
            target1,
            camid1,
            viewid1,
            pose1,
            epoch=1,
            bypass=(0, 1),
        )
        epoch6_full = descriptor_variant(
            base,
            model,
            final_state,
            diagnostic_rng,
            image6,
            target6,
            camid6,
            viewid6,
            pose6,
            epoch=6,
            bypass=(),
        )
        epoch6_all_bypass = descriptor_variant(
            base,
            model,
            final_state,
            diagnostic_rng,
            image6,
            target6,
            camid6,
            viewid6,
            pose6,
            epoch=6,
            bypass=(0, 1),
        )
        epoch6_bypass0 = descriptor_variant(
            base,
            model,
            final_state,
            diagnostic_rng,
            image6,
            target6,
            camid6,
            viewid6,
            pose6,
            epoch=6,
            bypass=(0,),
        )
        epoch6_bypass1 = descriptor_variant(
            base,
            model,
            final_state,
            diagnostic_rng,
            image6,
            target6,
            camid6,
            viewid6,
            pose6,
            epoch=6,
            bypass=(1,),
        )

        reload_rng = base.rng_state()
        reloaded = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
        ).to(device)
        incompatible = reloaded.load_state_dict(final_state, strict=True)
        base.restore_rng(reload_rng)
        model.load_state_dict(final_state, strict=True)
        base.restore_rng(diagnostic_rng)
        correct_pose = pose6
        shuffled_pose = {
            name: value.roll(1, 0) if torch.is_tensor(value) else value
            for name, value in pose6.items()
        }
        ExplodingPose.accesses = 0
        eval_correct = eval_descriptor(
            model, image6, camid6, viewid6, correct_pose
        )
        eval_shuffle = eval_descriptor(
            model, image6, camid6, viewid6, shuffled_pose
        )
        eval_none = eval_descriptor(model, image6, camid6, viewid6, None)
        eval_exploding = eval_descriptor(
            model, image6, camid6, viewid6, ExplodingPose()
        )
        reload_descriptor = eval_descriptor(
            reloaded, image6, camid6, viewid6, None
        )
        eval_correct_end = eval_descriptor(
            model, image6, camid6, viewid6, correct_pose
        )
    finally:
        if reloaded is not None:
            del reloaded
        model.load_state_dict(final_state, strict=True)
        base.restore_rng(diagnostic_rng)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    state_names = tuple(final_state)
    forbidden_components = {"teacher", "clip", "codebook", "text", "pose_batch"}
    all_bypass_delta = epoch6_full - epoch6_all_bypass
    bypass0_delta = epoch6_full - epoch6_bypass0
    bypass1_delta = epoch6_full - epoch6_bypass1
    descriptor_values = (
        epoch1_full,
        epoch1_all_bypass,
        epoch6_full,
        epoch6_all_bypass,
        epoch6_bypass0,
        epoch6_bypass1,
        eval_correct,
        eval_shuffle,
        eval_none,
        eval_exploding,
        reload_descriptor,
        eval_correct_end,
    )
    source_sha_after = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    teacher_versions_after = base.teacher_versions(teacher)
    teacher_state_after = base.module_state_sha256(teacher.visual)
    codebook_state_after = base.sha256_json(
        {
            "slot_means": base.sha256_tensor(teacher.slot_means),
            "shared_basis": base.sha256_tensor(teacher.shared_basis),
        }
    )
    apply_gate_had_override_after = "apply_gate" in tapf.__dict__
    apply_gate_override_after = tapf.__dict__.get("apply_gate")
    apply_gate_restore_exact = (
        apply_gate_had_override_before == apply_gate_had_override_after
        and (
            not apply_gate_had_override_before
            or apply_gate_override_before is apply_gate_override_after
        )
    )
    gates = {
        "strict_reload": (
            not incompatible.missing_keys and not incompatible.unexpected_keys
        ),
        "reload_descriptor_exact": torch.equal(eval_none, reload_descriptor),
        "rgb_correct_shuffle_exact": torch.equal(eval_correct, eval_shuffle),
        "rgb_correct_none_exact": torch.equal(eval_correct, eval_none),
        "rgb_correct_exploding_exact": torch.equal(eval_correct, eval_exploding),
        "rgb_correct_start_end_exact": torch.equal(
            eval_correct, eval_correct_end
        ),
        "exploding_pose_access_zero": ExplodingPose.accesses == 0,
        "epoch1_rho_zero_exact": float(tapf.rho_at_epoch(1, True)) == 0.0,
        "epoch1_full_all_bypass_exact": torch.equal(
            epoch1_full, epoch1_all_bypass
        ),
        "epoch6_rho_nonzero": float(tapf.rho_at_epoch(6, True)) > 0.0,
        "epoch6_all_bypass_nonzero": not torch.equal(
            epoch6_full, epoch6_all_bypass
        ),
        "epoch6_consumer0_nonzero": not torch.equal(
            epoch6_full, epoch6_bypass0
        ),
        "epoch6_consumer1_nonzero": not torch.equal(
            epoch6_full, epoch6_bypass1
        ),
        "all_bypass_mean_l2_positive": float(
            all_bypass_delta.float().norm(dim=1).mean()
        ) > 0.0,
        "consumer0_max_abs_positive": float(bypass0_delta.abs().max()) > 0.0,
        "consumer1_max_abs_positive": float(bypass1_delta.abs().max()) > 0.0,
        "all_descriptors_finite": all(
            bool(torch.isfinite(value).all()) for value in descriptor_values
        ),
        "state_teacher_free": all(
            not (set(name.lower().split(".")) & forbidden_components)
            for name in state_names
        ),
        "evidence_head_retained": any(
            "anchor.evidence_head" in name for name in state_names
        ),
        "two_routers_retained": all(
            any(
                f"psg_bank.{index}.evidence_projection" in name
                for name in state_names
            )
            for index in (0, 1)
        ),
        "final_state_finite": all(
            bool(torch.isfinite(value).all()) for value in final_state.values()
        ),
        "diagnostic_state_exact": tensor_state_sha256(
            base, tensor_state_cpu(model)
        ) == final_state_sha,
        "diagnostic_rng_exact": base.rng_sha256(base.rng_state())
        == diagnostic_rng_sha,
        "apply_gate_restore_exact": apply_gate_restore_exact,
        "teacher_versions_exact": teacher_versions_before
        == teacher_versions_after,
        "teacher_state_exact": teacher_state_before == teacher_state_after,
        "teacher_grads_none": all(
            parameter.grad is None for parameter in teacher.visual.parameters()
        ),
        "codebook_state_exact": codebook_state_before == codebook_state_after,
        "source_sha_unchanged": source_sha_after == EXPECTED_SOURCE_SHA256,
        "asset_sha_unchanged": (
            sha256_file(clip_path) == CLIP_SHA256
            and sha256_file(codebook_path) == CODEBOOK_SHA256
            and sha256_file(dependency_path) == REPORTER_DEPENDENCY_SHA256
            and sha256_file(runtime_freeze) == RUNTIME_FREEZE_SHA256
        ),
        "tracked_clean": not bool(
            run_text(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                repo_root,
            )
        ),
    }
    diagnostics = {
        "final_state_sha256": final_state_sha,
        "final_module_state_sha256": base.module_state_sha256(model),
        "state_names_count": len(state_names),
        "epoch1_rho": float(tapf.rho_at_epoch(1, True)),
        "epoch6_rho": float(tapf.rho_at_epoch(6, True)),
        "epoch6_all_bypass_max_abs": float(all_bypass_delta.abs().max()),
        "epoch6_all_bypass_mean_l2": float(
            all_bypass_delta.float().norm(dim=1).mean()
        ),
        "epoch6_bypass0_max_abs": float(bypass0_delta.abs().max()),
        "epoch6_bypass1_max_abs": float(bypass1_delta.abs().max()),
        "exploding_pose_accesses": ExplodingPose.accesses,
        "source_sha256_after": source_sha_after,
        "teacher_state_sha256_after": teacher_state_after,
        "codebook_state_sha256_after": codebook_state_after,
        "diagnostic_rng_sha256": diagnostic_rng_sha,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "diagnostics": diagnostics,
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
        raise RuntimeError("Refusing to overwrite exp400 output assets")
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
        "clip_fresh_name": "exp400" in clip_path.name,
        "codebook_fresh_name": "exp400" in codebook_path.name,
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
        raise RuntimeError("exp400 requires CUDA")
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
        "stage_length": STAGE_LENGTH,
        "stage_tail": STAGE_TAIL,
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
    scratch_prefix = ".exp400_gradient_scratch_"
    scratch_before = sorted(
        str(path) for path in output.parent.glob(f"{scratch_prefix}*")
    )
    if scratch_before:
        raise RuntimeError(f"Stale exp400 scratch exists: {scratch_before}")
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
            raise RuntimeError("exp400 reporter scratch not empty")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - began
    scratch_after = sorted(
        str(path) for path in output.parent.glob(f"{scratch_prefix}*")
    )
    evaluation = evaluate_trajectories(
        d0_result["rows"],
        rich_result["rows"],
        rich_result["group_state_sha256"],
    )
    terminal = run_terminal_audit(
        base,
        rich_model,
        rich_cfg,
        make_model,
        num_classes,
        camera_num,
        view_num,
        cpu_batches,
        rich_targets,
        teacher,
        teacher_versions_before,
        teacher_state_before,
        codebook_state_before,
        repo_root,
        clip_path,
        codebook_path,
        dependency_path,
        runtime_freeze,
        device,
    )
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
        "terminal_pass": terminal["status"] == "PASS",
    }
    passed = all(validity_gates.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "diagnostic_outcome": (
            "FINAL_PRODUCTION_PREFLIGHT_PASS"
            if passed
            else "FINAL_PRODUCTION_PREFLIGHT_FAIL"
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
        "terminal_evaluation": terminal,
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
        "exp397_remains_sealed": True,
        "exp398_remains_sealed": True,
        "exp399_remains_sealed": True,
        "production_preflight_authorized": passed,
        "formal_training_authorized": passed,
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
            "exp397_remains_sealed": True,
            "exp398_remains_sealed": True,
            "exp399_remains_sealed": True,
            "production_preflight_authorized": False,
        }
        exit_code = 1
    if output.exists() or runner.exists():
        raise RuntimeError("Refusing to overwrite exp400 result/runner")
    write_json(output, result)
    write_json(runner, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
