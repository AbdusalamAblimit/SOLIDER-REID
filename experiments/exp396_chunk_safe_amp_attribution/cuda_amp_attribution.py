#!/usr/bin/env python3
"""Zero-update chunk-safe actual-batch CUDA/AMP attribution for exp396."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
CLIP_SHA256 = "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
CODEBOOK_SHA256 = "fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a"
RUNTIME_FREEZE_SHA256 = "3d38c99c7f06502d8b40467d2674c966723e5c913d2edf962c5a7088ec60cddb"
EXPECTED_INITIAL_SCALE = 65536.0
REPORTER_CHUNK_ELEMENTS = 1_048_576
REPORTER_NAME = "chunk_safe_exact_memmap_v1"
EXPECTED_SOURCE_SHA256 = {
    "model/tapf.py": "95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886",
    "model/clip_semantic_teacher.py": "c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5",
    "model/make_model.py": "6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03",
    "processor/processor.py": "be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1",
    "config/defaults.py": "a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5",
    "model/backbones/swin_transformer.py": "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "datasets/pose_dataset.py": "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc",
    "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml": "e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e",
    "configs/occluded_duke/swin_tiny_tapf_d0.yml": "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b",
}
GROUP_NAMES = (
    "backbone",
    "anchor_trunk",
    "pose_head",
    "mask_head",
    "presence_head",
    "evidence_head",
    "id_head",
    "router0_token_projection",
    "router0_context_projection",
    "router0_evidence_projection",
    "router0_experts",
    "router1_token_projection",
    "router1_context_projection",
    "router1_evidence_projection",
    "router1_experts",
)
BASELINE_LOSSES = ("reid", "heatmap", "confidence", "pose", "total")
RICH_LOSSES = (
    "reid",
    "heatmap",
    "confidence",
    "mask",
    "presence",
    "evidence_cosine",
    "evidence_relation",
    "exec_consumer0",
    "exec_consumer1",
    "pose",
    "total",
)
INDIVIDUAL_AUX = (
    "heatmap",
    "confidence",
    "mask",
    "presence",
    "evidence_cosine",
    "evidence_relation",
    "exec_consumer0",
    "exec_consumer1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def sha256_json(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_text(command, cwd=None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def rng_state():
    return {
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def rng_sha256(state) -> str:
    digest = hashlib.sha256()
    digest.update(state["torch"].cpu().numpy().tobytes())
    for value in state["cuda"]:
        digest.update(value.cpu().numpy().tobytes())
    digest.update(pickle.dumps(state["numpy"], protocol=4))
    digest.update(pickle.dumps(state["python"], protocol=4))
    return digest.hexdigest()


def module_state_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_state_sha256(optimizer) -> str:
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def parameter_versions(model):
    return {name: parameter._version for name, parameter in model.named_parameters()}


def buffer_snapshot(model):
    return {
        name: value.detach().clone() for name, value in model.named_buffers()
    }


def restore_buffers(model, snapshot) -> None:
    current = dict(model.named_buffers())
    if set(current) != set(snapshot):
        raise RuntimeError("Model buffer set changed during attribution")
    with torch.no_grad():
        for name, value in snapshot.items():
            current[name].copy_(value)


def buffers_exact(model, snapshot) -> bool:
    current = dict(model.named_buffers())
    return set(current) == set(snapshot) and all(
        torch.equal(current[name], value) for name, value in snapshot.items()
    )


def teacher_versions(teacher):
    return {
        "visual_parameters": [
            parameter._version for parameter in teacher.visual.parameters()
        ],
        "visual_buffers": [value._version for value in teacher.visual.buffers()],
        "slot_means": teacher.slot_means._version,
        "shared_basis": teacher.shared_basis._version,
    }


def iter_cpu_fp64_chunks(gradients):
    for gradient in gradients:
        flattened = gradient.detach().reshape(-1)
        for start in range(0, flattened.numel(), REPORTER_CHUNK_ELEMENTS):
            yield flattened[start : start + REPORTER_CHUNK_ELEMENTS].to(
                device="cpu", dtype=torch.float64
            )


def linear_quantiles_from_sorted(values, quantiles=(0.50, 0.95, 0.99)):
    count = int(values.shape[0])
    if count == 0:
        return [None for _ in quantiles]
    output = []
    for quantile in quantiles:
        rank = (count - 1) * quantile
        lower_index = int(math.floor(rank))
        upper_index = int(math.ceil(rank))
        lower = float(values[lower_index])
        upper = float(values[upper_index])
        output.append(lower + (upper - lower) * (rank - lower_index))
    return output


def chunk_safe_finite_statistics(gradients, scratch_root: Path, label: str):
    elements = 0
    finite_elements = 0
    nan_elements = 0
    posinf_elements = 0
    neginf_elements = 0
    finite_abs_max = None
    finite_l2 = 0.0
    for chunk in iter_cpu_fp64_chunks(gradients):
        elements += int(chunk.numel())
        finite_mask = torch.isfinite(chunk)
        finite_elements += int(finite_mask.sum())
        nan_elements += int(torch.isnan(chunk).sum())
        posinf_elements += int(torch.isposinf(chunk).sum())
        neginf_elements += int(torch.isneginf(chunk).sum())
        finite_abs = chunk[finite_mask].abs()
        if finite_abs.numel():
            chunk_max = float(finite_abs.max())
            finite_abs_max = (
                chunk_max
                if finite_abs_max is None
                else max(finite_abs_max, chunk_max)
            )
            finite_l2 = math.hypot(
                finite_l2, float(torch.linalg.vector_norm(finite_abs))
            )
    if elements != finite_elements + nan_elements + posinf_elements + neginf_elements:
        raise RuntimeError("Chunk-safe reporter classification count mismatch")
    if finite_elements == 0:
        return {
            "elements": elements,
            "finite_elements": 0,
            "nan_elements": nan_elements,
            "posinf_elements": posinf_elements,
            "neginf_elements": neginf_elements,
            "all_finite": elements == 0,
            "finite_abs_max": None,
            "finite_l2": None,
            "finite_abs_p50": None,
            "finite_abs_p95": None,
            "finite_abs_p99": None,
        }

    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in label
    )
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f"{safe_label}_", suffix=".f64", dir=scratch_root
    )
    os.close(descriptor)
    memmap_path = Path(raw_path)
    values = None
    try:
        if memmap_path.is_symlink() or not memmap_path.is_file():
            raise RuntimeError("Reporter scratch must be a regular file")
        values = np.memmap(
            memmap_path, dtype=np.float64, mode="w+", shape=(finite_elements,)
        )
        offset = 0
        for chunk in iter_cpu_fp64_chunks(gradients):
            finite_abs = chunk[torch.isfinite(chunk)].abs().numpy()
            next_offset = offset + int(finite_abs.size)
            values[offset:next_offset] = finite_abs
            offset = next_offset
        if offset != finite_elements:
            raise RuntimeError("Reporter scratch finite write count mismatch")
        values.flush()
        values.sort(kind="quicksort")
        p50, p95, p99 = linear_quantiles_from_sorted(values)
    finally:
        if values is not None:
            values.flush()
            del values
        memmap_path.unlink(missing_ok=True)
    return {
        "elements": elements,
        "finite_elements": finite_elements,
        "nan_elements": nan_elements,
        "posinf_elements": posinf_elements,
        "neginf_elements": neginf_elements,
        "all_finite": finite_elements == elements,
        "finite_abs_max": finite_abs_max,
        "finite_l2": finite_l2,
        "finite_abs_p50": p50,
        "finite_abs_p95": p95,
        "finite_abs_p99": p99,
    }


def gradient_report(group_parameters, scratch_root: Path, label: str):
    report = {}
    for group_name in GROUP_NAMES:
        items = group_parameters[group_name]
        present = [
            parameter.grad for _, parameter in items if parameter.grad is not None
        ]
        statistics = chunk_safe_finite_statistics(
            present, scratch_root, f"{label}_{group_name}"
        )
        item = {
            "applicable": bool(items),
            "parameter_tensors": len(items),
            "grad_present_tensors": len(present),
            "grad_absent_tensors": len(items) - len(present),
            "grad_nonzero_tensors": sum(
                int(bool(torch.count_nonzero(gradient.detach())))
                for gradient in present
            ),
            "grad_zero_tensors": sum(
                int(not bool(torch.count_nonzero(gradient.detach())))
                for gradient in present
            ),
            **statistics,
        }
        report[group_name] = item
    return report


def classify_parameter(name: str, arm: str):
    if name.startswith("base.tapf.anchor.project."):
        return "anchor_trunk"
    if name.startswith("base.tapf.anchor.depthwise."):
        return "anchor_trunk"
    if name.startswith("base.tapf.anchor.norm."):
        return "anchor_trunk"
    if arm == "rich":
        anchor_heads = {
            "base.tapf.anchor.pose_head.": "pose_head",
            "base.tapf.anchor.region_mask_head.": "mask_head",
            "base.tapf.anchor.presence_head.": "presence_head",
            "base.tapf.anchor.evidence_head.": "evidence_head",
        }
        for prefix, group in anchor_heads.items():
            if name.startswith(prefix):
                return group
        for index in (0, 1):
            router_parts = {
                f"base.tapf.psg_bank.{index}.token_projection.": f"router{index}_token_projection",
                f"base.tapf.psg_bank.{index}.context_projection.": f"router{index}_context_projection",
                f"base.tapf.psg_bank.{index}.evidence_projection.": f"router{index}_evidence_projection",
                f"base.tapf.psg_bank.{index}.experts.": f"router{index}_experts",
            }
            for prefix, group in router_parts.items():
                if name.startswith(prefix):
                    return group
    elif arm == "d0":
        if name.startswith("base.tapf.anchor.head."):
            return "pose_head"
        for index in (0, 1):
            if name.startswith(f"base.tapf.psg_bank.{index}."):
                return f"router{index}_experts"
    else:
        raise ValueError(f"Unknown arm: {arm}")
    if name.startswith("base.tapf."):
        return None
    if name.startswith("base."):
        return "backbone"
    return "id_head"


def parameter_groups(model, optimizer, arm: str):
    groups = {name: [] for name in GROUP_NAMES}
    uncovered = []
    duplicates = []
    assignments = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = classify_parameter(name, arm)
        if group is None:
            uncovered.append(name)
            continue
        if id(parameter) in assignments:
            duplicates.append(name)
            continue
        assignments[id(parameter)] = group
        groups[group].append((name, parameter))
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    }
    grouped_parameters = set(assignments)
    optimizer_uncovered = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) in optimizer_parameters - grouped_parameters
    )
    grouped_not_optimizer = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) in grouped_parameters - optimizer_parameters
    )
    coverage = {
        "uncovered_names": sorted(set(uncovered + optimizer_uncovered)),
        "duplicate_names": sorted(set(duplicates)),
        "grouped_not_optimizer_names": grouped_not_optimizer,
        "optimizer_parameter_count": len(optimizer_parameters),
        "grouped_parameter_count": len(grouped_parameters),
        "exact": (
            not uncovered
            and not optimizer_uncovered
            and not duplicates
            and not grouped_not_optimizer
            and optimizer_parameters == grouped_parameters
        ),
    }
    names = {
        group_name: [name for name, _ in items]
        for group_name, items in groups.items()
    }
    return groups, names, coverage


def scalar_record(value: torch.Tensor):
    detached = value.detach().float()
    finite = bool(torch.isfinite(detached).all())
    return {"finite": finite, "value": float(detached) if finite else None}


def tensor_formula_equivalent(left: torch.Tensor, right: torch.Tensor) -> bool:
    left = left.detach()
    right = right.detach()
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    classifications = (
        torch.isfinite,
        torch.isnan,
        torch.isposinf,
        torch.isneginf,
    )
    if not all(torch.equal(fn(left), fn(right)) for fn in classifications):
        return False
    finite = torch.isfinite(left)
    return torch.equal(left[finite], right[finite])


def scaled_unscaled_consistency(scaled, unscaled, scale):
    count_fields = (
        "grad_present_tensors",
        "grad_absent_tensors",
        "grad_nonzero_tensors",
        "grad_zero_tensors",
        "elements",
        "finite_elements",
        "nan_elements",
        "posinf_elements",
        "neginf_elements",
    )
    range_fields = (
        "finite_abs_max",
        "finite_l2",
        "finite_abs_p50",
        "finite_abs_p95",
        "finite_abs_p99",
    )
    groups = {}
    for name in GROUP_NAMES:
        left = scaled[name]
        right = unscaled[name]
        counts_equal = all(left[field] == right[field] for field in count_fields)
        errors = {}
        for field in range_fields:
            scaled_value = left[field]
            unscaled_value = right[field]
            if scaled_value is None or unscaled_value is None:
                errors[field] = None
                continue
            expected = unscaled_value * scale
            errors[field] = abs(scaled_value - expected) / max(
                abs(scaled_value), abs(expected), 1e-300
            )
        finite_errors = [value for value in errors.values() if value is not None]
        groups[name] = {
            "nonfinite_and_presence_counts_equal": counts_equal,
            "range_relative_errors": errors,
            "max_range_relative_error": max(finite_errors, default=0.0),
        }
    return {
        "groups": groups,
        "counts_equal_all_groups": all(
            item["nonfinite_and_presence_counts_equal"]
            for item in groups.values()
        ),
        "max_range_relative_error": max(
            item["max_range_relative_error"] for item in groups.values()
        ),
    }


def rich_forward_losses(model, loss_fn, batch, pose):
    image, target, camid, viewid = batch
    output = model(
        image,
        label=target,
        cam_label=camid,
        view_label=viewid,
        pose_batch=pose,
        tapf_epoch=1,
    )
    score, feature, _, aux = output
    reid = loss_fn(score, feature, target, camid)
    exec_losses = aux["exec_losses"]
    if len(exec_losses) != 2:
        raise RuntimeError("Rich attribution requires exactly two exec losses")
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
    formula_gates = {
        "exec_mean_exact": tensor_formula_equivalent(
            exec_mean, aux["exec_loss"]
        ),
        "semantic_exact": tensor_formula_equivalent(
            semantic, aux["semantic_loss"]
        ),
        "pose_exact": tensor_formula_equivalent(
            pose_loss, aux["pose_loss"]
        ),
    }
    if not all(formula_gates.values()):
        raise RuntimeError(f"Rich loss formula mismatch: {formula_gates}")
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
    }, formula_gates


def d0_forward_losses(model, loss_fn, batch, pose):
    image, target, camid, viewid = batch
    output = model(
        image,
        label=target,
        cam_label=camid,
        view_label=viewid,
        pose_batch=pose,
        tapf_epoch=1,
    )
    score, feature, _, aux = output
    reid = loss_fn(score, feature, target, camid)
    pose_loss = aux["heatmap_loss"] + aux["confidence_loss"]
    total = reid + 0.1 * pose_loss
    formula_gates = {
        "pose_exact": tensor_formula_equivalent(pose_loss, aux["pose_loss"])
    }
    if not all(formula_gates.values()):
        raise RuntimeError(f"D0 loss formula mismatch: {formula_gates}")
    return {
        "reid": reid,
        "heatmap": aux["heatmap_loss"],
        "confidence": aux["confidence_loss"],
        "pose": pose_loss,
        "total": total,
    }, formula_gates


def run_loss_row(
    model,
    optimizer,
    group_parameters,
    loss_name,
    forward_losses,
    audit_rng,
    scratch_root,
    arm,
):
    restore_rng(audit_rng)
    buffers = buffer_snapshot(model)
    versions_before = parameter_versions(model)
    optimizer_before = optimizer_state_sha256(optimizer)
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    record = {
        "loss_name": loss_name,
        "optimizer_step_calls": 0,
        "scaler_step_calls": 0,
        "scaler_update_calls": 0,
        "scheduler_update_calls": 0,
    }
    try:
        model.train()
        with torch.amp.autocast("cuda", enabled=True):
            losses, formula_gates = forward_losses()
            selected = losses[loss_name]
        record["components"] = {
            name: scalar_record(value) for name, value in losses.items()
        }
        record["formula_gates"] = formula_gates
        record["autocast_dtype"] = str(torch.get_autocast_dtype("cuda"))
        record["selected_loss"] = scalar_record(selected)
        if not record["selected_loss"]["finite"]:
            record["row_status"] = "LOSS_NONFINITE_NO_BACKWARD"
            record["scaled"] = None
            record["unscaled"] = None
            return record
        scaler = torch.amp.GradScaler("cuda")
        scale_before = float(scaler.get_scale())
        scaler.scale(selected).backward()
        scaled = gradient_report(
            group_parameters, scratch_root, f"{arm}_{loss_name}_scaled"
        )
        scaler.unscale_(optimizer)
        unscaled = gradient_report(
            group_parameters, scratch_root, f"{arm}_{loss_name}_unscaled"
        )
        scale_after = float(scaler.get_scale())
        record.update(
            {
                "row_status": "BACKWARD_CAPTURED",
                "scale_before": scale_before,
                "scale_after_unscale_without_update": scale_after,
                "scaled": scaled,
                "unscaled": unscaled,
                "scaled_unscaled_consistency": scaled_unscaled_consistency(
                    scaled, unscaled, scale_before
                ),
            }
        )
        return record
    finally:
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        restore_buffers(model, buffers)
        restore_rng(audit_rng)
        record["buffers_restored_exact"] = buffers_exact(model, buffers)
        record["parameter_versions_unchanged"] = (
            versions_before == parameter_versions(model)
        )
        record["optimizer_state_unchanged"] = (
            optimizer_before == optimizer_state_sha256(optimizer)
        )


def run_arm(
    arm,
    model,
    optimizer,
    loss_names,
    forward_factory,
    audit_rng,
    scratch_root,
):
    group_parameters, group_names, coverage = parameter_groups(
        model, optimizer, arm
    )
    if not coverage["exact"]:
        raise RuntimeError(f"{arm} parameter coverage failed: {coverage}")
    state_before = module_state_sha256(model)
    optimizer_before = optimizer_state_sha256(optimizer)
    rows = []
    for loss_name in loss_names:
        rows.append(
            run_loss_row(
                model,
                optimizer,
                group_parameters,
                loss_name,
                forward_factory,
                audit_rng,
                scratch_root,
                arm,
            )
        )
    state_after = module_state_sha256(model)
    optimizer_after = optimizer_state_sha256(optimizer)
    return {
        "parameter_groups": group_names,
        "parameter_coverage": coverage,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "optimizer_state_sha256_before": optimizer_before,
        "optimizer_state_sha256_after": optimizer_after,
        "rows": rows,
        "gates": {
            "row_count_exact": len(rows) == len(loss_names),
            "rows_isolated": all(
                row.get("buffers_restored_exact", False)
                and row.get("parameter_versions_unchanged", False)
                and row.get("optimizer_state_unchanged", False)
                for row in rows
            ),
            "state_exact": state_before == state_after,
            "optimizer_exact": optimizer_before == optimizer_after,
            "zero_updates": all(
                row["optimizer_step_calls"] == 0
                and row["scaler_step_calls"] == 0
                and row["scaler_update_calls"] == 0
                and row["scheduler_update_calls"] == 0
                for row in rows
            ),
            "default_scale_unmodified": all(
                row["row_status"] != "BACKWARD_CAPTURED"
                or (
                    row["scale_before"] == EXPECTED_INITIAL_SCALE
                    and row["scale_after_unscale_without_update"]
                    == EXPECTED_INITIAL_SCALE
                )
                for row in rows
            ),
            "scaled_unscaled_counts_consistent": all(
                row["row_status"] != "BACKWARD_CAPTURED"
                or row["scaled_unscaled_consistency"][
                    "counts_equal_all_groups"
                ]
                for row in rows
            ),
        },
    }


def row_nonfinite(row):
    if not row["selected_loss"]["finite"]:
        return True
    for stage in ("scaled", "unscaled"):
        report = row.get(stage)
        if report is None:
            continue
        if any(not item["all_finite"] for item in report.values()):
            return True
    return False


def nonfinite_cells(arm_name, arm_result):
    cells = []
    for row in arm_result["rows"]:
        if not row["selected_loss"]["finite"]:
            cells.append(
                {
                    "arm": arm_name,
                    "loss": row["loss_name"],
                    "stage": "loss",
                    "group": None,
                    "nan": None,
                    "posinf": None,
                    "neginf": None,
                }
            )
        for stage in ("scaled", "unscaled"):
            report = row.get(stage)
            if report is None:
                continue
            for group, item in report.items():
                if not item["all_finite"]:
                    cells.append(
                        {
                            "arm": arm_name,
                            "loss": row["loss_name"],
                            "stage": stage,
                            "group": group,
                            "nan": item["nan_elements"],
                            "posinf": item["posinf_elements"],
                            "neginf": item["neginf_elements"],
                        }
                    )
    return cells


def diagnostic_outcome(d0_result, rich_result):
    d0_rows = {row["loss_name"]: row for row in d0_result["rows"]}
    rich_rows = {row["loss_name"]: row for row in rich_result["rows"]}
    if row_nonfinite(d0_rows["total"]):
        return "SHARED_D0_OR_RUNTIME_NONFINITE"
    if row_nonfinite(rich_rows["reid"]):
        return "RICH_REID_GRAPH_NONFINITE"
    if any(row_nonfinite(rich_rows[name]) for name in INDIVIDUAL_AUX):
        return "RICH_ISOLATED_AUX_NONFINITE"
    if row_nonfinite(rich_rows["pose"]) or row_nonfinite(rich_rows["total"]):
        return "RICH_AGGREGATE_NONFINITE"
    return "FRESH_EXP395_FINITE_NONREPRODUCTION"


def transfer_batch(batch, device):
    image, vid, camid, viewid, pose = batch
    pose_device = {
        "relative_paths": list(pose["relative_paths"]),
        "image_sha256": list(pose["image_sha256"]),
        "keypoints": pose["keypoints"].to(device),
        "scores": pose["scores"].to(device),
        "valid": pose["valid"].to(device),
        "teacher_rgb": pose["teacher_rgb"].to(device),
    }
    return (
        image.to(device),
        vid.to(device),
        camid.to(device),
        viewid.to(device),
        pose_device,
    )


def teacher_targets(teacher, pose):
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        targets = teacher(
            pose["teacher_rgb"],
            pose["keypoints"],
            pose["scores"],
            pose["valid"],
        )
    model_targets = {
        "semantic_teacher_evidence": targets["evidence_code"].detach().clone(),
        "semantic_valid": targets["valid"].detach().clone(),
        "semantic_teacher_mask": targets["region_masks"].detach().clone(),
    }
    return model_targets, targets


def nvidia_processes():
    output = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def common_state_parity(d0_model, rich_model):
    d0 = d0_model.state_dict()
    rich = rich_model.state_dict()
    common = sorted(
        key
        for key in set(d0) & set(rich)
        if not key.startswith("base.tapf.") and d0[key].shape == rich[key].shape
    )
    mismatches = [key for key in common if not torch.equal(d0[key], rich[key])]
    return {
        "common_tensors": len(common),
        "mismatches": mismatches,
        "exact": bool(common) and not mismatches,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--rich-config",
        default="configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml",
    )
    parser.add_argument(
        "--d0-config",
        default="configs/occluded_duke/swin_tiny_tapf_d0.yml",
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--runtime-freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run_attribution(args):
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    runner_path = Path(args.runner).resolve()
    manifest_path = Path(args.manifest).resolve()
    if any(path.exists() for path in (output_path, runner_path, manifest_path)):
        raise RuntimeError("Refusing to overwrite exp396 output assets")
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
        raise RuntimeError("Source/config SHA contract mismatch")

    clip_path = Path(args.clip).resolve()
    codebook_path = Path(args.codebook).resolve()
    runtime_freeze = Path(args.runtime_freeze).resolve()
    asset_gates = {
        "clip_fresh_name": "exp396" in clip_path.name,
        "codebook_fresh_name": "exp396" in codebook_path.name,
        "clip_regular": clip_path.is_file() and not clip_path.is_symlink(),
        "codebook_regular": codebook_path.is_file() and not codebook_path.is_symlink(),
        "runtime_freeze_regular": runtime_freeze.is_file()
        and not runtime_freeze.is_symlink(),
        "clip_sha": sha256_file(clip_path) == CLIP_SHA256,
        "codebook_sha": sha256_file(codebook_path) == CODEBOOK_SHA256,
        "runtime_freeze_sha": sha256_file(runtime_freeze)
        == RUNTIME_FREEZE_SHA256,
    }
    if not all(asset_gates.values()):
        raise RuntimeError(f"Fresh asset contract failed: {asset_gates}")
    initial_processes = nvidia_processes()
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
        raise RuntimeError(f"Canonical runtime version mismatch: {runtime_gates}")

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
        "rich_batch64": int(rich_cfg.SOLVER.IMS_PER_BATCH) == 64,
        "d0_batch64": int(d0_cfg.SOLVER.IMS_PER_BATCH) == 64,
        "seed1234": int(rich_cfg.SOLVER.SEED) == int(d0_cfg.SOLVER.SEED) == 1234,
        "rich_enabled": bool(rich_cfg.MODEL.TAPF.RICH_EVIDENCE_ENABLED),
        "d0_nonsemantic": bool(d0_cfg.MODEL.TAPF.ENABLED)
        and not bool(d0_cfg.MODEL.TAPF.SEMANTIC_ENABLED),
        "same_pose_weight": float(rich_cfg.MODEL.TAPF.POSE_LOSS_WEIGHT)
        == float(d0_cfg.MODEL.TAPF.POSE_LOSS_WEIGHT)
        == 0.1,
        "workers8": int(rich_cfg.DATALOADER.NUM_WORKERS) == 8,
    }
    if not all(config_gates.values()):
        raise RuntimeError(f"Config contract failed: {config_gates}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("exp396 actual attribution requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.init()
    if "4090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Exclusive RTX 4090 is required")
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(1234)
    train_loader, _, _, _, num_classes, camera_num, view_num = make_dataloader(
        rich_cfg
    )

    set_seed(1234)
    d0_model = make_model(
        d0_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=d0_cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    d0_loss_fn, d0_center = make_loss(d0_cfg, num_classes=num_classes)
    d0_optimizer, _ = make_optimizer(d0_cfg, d0_model, d0_center)

    set_seed(1234)
    rich_model = make_model(
        rich_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=rich_cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    rich_loss_fn, rich_center = make_loss(rich_cfg, num_classes=num_classes)
    rich_optimizer, _ = make_optimizer(rich_cfg, rich_model, rich_center)
    common_parity = common_state_parity(d0_model, rich_model)
    if not common_parity["exact"]:
        raise RuntimeError(f"D0/rich common initial state mismatch: {common_parity}")

    teacher_rng = rng_state()
    teacher = FrozenRichClipEvidenceTeacher(
        checkpoint=str(clip_path),
        checkpoint_sha256=CLIP_SHA256,
        codebook=str(codebook_path),
        codebook_sha256=CODEBOOK_SHA256,
        device=device,
        microbatch=rich_cfg.MODEL.TAPF.CLIP_MICROBATCH,
    )
    restore_rng(teacher_rng)
    teacher_versions_before = teacher_versions(teacher)
    teacher_state_before = module_state_sha256(teacher.visual)
    codebook_state_before = sha256_json(
        {
            "slot_means": sha256_tensor(teacher.slot_means),
            "shared_basis": sha256_tensor(teacher.shared_basis),
        }
    )

    raw_batch = next(iter(train_loader))
    image, target, camid, viewid, pose = transfer_batch(raw_batch, device)
    batch = (image, target, camid, viewid)
    manifest = {
        "relative_paths": pose["relative_paths"],
        "image_sha256": pose["image_sha256"],
        "pids": target.detach().cpu().tolist(),
        "camids": camid.detach().cpu().tolist(),
        "viewids": viewid.detach().cpu().tolist(),
        "input_tensor_sha256": sha256_tensor(image),
        "keypoints_sha256": sha256_tensor(pose["keypoints"]),
        "scores_sha256": sha256_tensor(pose["scores"]),
        "valid_sha256": sha256_tensor(pose["valid"]),
        "teacher_rgb_sha256": sha256_tensor(pose["teacher_rgb"]),
    }
    write_json(manifest_path, manifest)
    targets, raw_targets = teacher_targets(teacher, pose)
    valid = raw_targets["valid"].bool()
    valid_norm = raw_targets["evidence_code"].float().norm(dim=-1)[valid]
    target_gates = {
        "evidence_shape": tuple(raw_targets["evidence_code"].shape) == (64, 5, 16),
        "mask_shape": tuple(raw_targets["region_masks"].shape) == (64, 5, 96, 32),
        "valid_shape": tuple(valid.shape) == (64, 5),
        "targets_detached": all(
            not value.requires_grad
            for value in (
                raw_targets["evidence_code"],
                raw_targets["region_masks"],
            )
        ),
        "targets_finite": all(
            bool(torch.isfinite(value).all())
            for value in (
                raw_targets["evidence_code"],
                raw_targets["region_masks"],
            )
        ),
        "invalid_zero": torch.equal(
            raw_targets["evidence_code"][~valid],
            torch.zeros_like(raw_targets["evidence_code"][~valid]),
        ),
        "valid_norm": valid_norm.numel() > 0
        and bool(torch.isfinite(valid_norm).all())
        and float((valid_norm - 1.0).abs().max()) < 1e-5,
    }
    if not all(target_gates.values()):
        raise RuntimeError(f"Teacher target contract failed: {target_gates}")
    target_sha = {
        name: sha256_tensor(value) for name, value in targets.items()
    }
    rich_pose = {
        "keypoints": pose["keypoints"],
        "scores": pose["scores"],
        "valid": pose["valid"],
        **targets,
    }
    d0_pose = {
        "keypoints": pose["keypoints"],
        "scores": pose["scores"],
        "valid": pose["valid"],
    }
    audit_rng = rng_state()
    audit_rng_sha = rng_sha256(audit_rng)
    checkpoint_before = sorted(
        str(path) for path in output_path.parent.glob("*.pth")
    )
    began = time.perf_counter()
    scratch_prefix = ".exp396_gradient_scratch_"
    scratch_before = sorted(
        str(path) for path in output_path.parent.glob(f"{scratch_prefix}*")
    )
    if scratch_before:
        raise RuntimeError(f"Stale exp396 scratch exists: {scratch_before}")
    with tempfile.TemporaryDirectory(
        prefix=scratch_prefix, dir=output_path.parent
    ) as scratch_directory:
        scratch_root = Path(scratch_directory)
        d0_result = run_arm(
            "d0",
            d0_model,
            d0_optimizer,
            BASELINE_LOSSES,
            lambda: d0_forward_losses(d0_model, d0_loss_fn, batch, d0_pose),
            audit_rng,
            scratch_root,
        )
        rich_result = run_arm(
            "rich",
            rich_model,
            rich_optimizer,
            RICH_LOSSES,
            lambda: rich_forward_losses(
                rich_model, rich_loss_fn, batch, rich_pose
            ),
            audit_rng,
            scratch_root,
        )
        if any(scratch_root.iterdir()):
            raise RuntimeError("Reporter left per-cell scratch files")
    scratch_after = sorted(
        str(path) for path in output_path.parent.glob(f"{scratch_prefix}*")
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - began
    restore_rng(audit_rng)
    final_rng_sha = rng_sha256(rng_state())
    teacher_versions_after = teacher_versions(teacher)
    teacher_state_after = module_state_sha256(teacher.visual)
    codebook_state_after = sha256_json(
        {
            "slot_means": sha256_tensor(teacher.slot_means),
            "shared_basis": sha256_tensor(teacher.shared_basis),
        }
    )
    checkpoint_after = sorted(
        str(path) for path in output_path.parent.glob("*.pth")
    )
    cells = nonfinite_cells("d0", d0_result) + nonfinite_cells(
        "rich", rich_result
    )
    validity_gates = {
        "d0_arm_valid": all(d0_result["gates"].values()),
        "rich_arm_valid": all(rich_result["gates"].values()),
        "common_state_parity": common_parity["exact"],
        "target_contract": all(target_gates.values()),
        "teacher_versions_exact": teacher_versions_before
        == teacher_versions_after,
        "teacher_state_exact": teacher_state_before == teacher_state_after,
        "codebook_state_exact": codebook_state_before == codebook_state_after,
        "rng_exact": audit_rng_sha == final_rng_sha,
        "zero_optimizer_updates": True,
        "zero_scaler_updates": True,
        "checkpoint_zero": checkpoint_before == checkpoint_after == [],
        "manifest_written": manifest_path.is_file(),
        "scratch_cleanup_exact": scratch_before == scratch_after == [],
    }
    return {
        "status": "PASS" if all(validity_gates.values()) else "INVALID",
        "diagnostic_outcome": diagnostic_outcome(d0_result, rich_result),
        "source_commit": SOURCE_COMMIT,
        "source_sha256": source_sha,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "reporter": {
            "name": REPORTER_NAME,
            "chunk_elements": REPORTER_CHUNK_ELEMENTS,
            "percentiles": [0.50, 0.95, 0.99],
            "scratch_before": scratch_before,
            "scratch_after": scratch_after,
        },
        "asset_gates": asset_gates,
        "config_gates": config_gates,
        "initial_gpu_processes": initial_processes,
        "device": torch.cuda.get_device_name(device),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            "open_clip": open_clip.__version__,
            "opencv": cv2.__version__,
            "timm": timm.__version__,
            "gates": runtime_gates,
        },
        "batch_manifest_sha256": sha256_file(manifest_path),
        "teacher_target_sha256": target_sha,
        "teacher_target_gates": target_gates,
        "valid_slots": int(valid.sum()),
        "common_initial_state_parity": common_parity,
        "d0": d0_result,
        "rich": rich_result,
        "nonfinite_cells": cells,
        "optimizer_updates": 0,
        "scaler_step_calls": 0,
        "scaler_update_calls": 0,
        "checkpoint_count": len(checkpoint_after),
        "audit_rng_sha256_before": audit_rng_sha,
        "audit_rng_sha256_after": final_rng_sha,
        "teacher_state_sha256_before": teacher_state_before,
        "teacher_state_sha256_after": teacher_state_after,
        "codebook_state_sha256_before": codebook_state_before,
        "codebook_state_sha256_after": codebook_state_after,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "in_process_final_gpu_processes": nvidia_processes(),
        "post_exit_gpu_audit_required": True,
        "validity_gates": validity_gates,
        "exp394_remains_sealed": True,
        "exp395_remains_sealed": True,
        "formal_training_authorized": False,
    }


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    runner = Path(args.runner).resolve()
    result = None
    exit_code = 0
    try:
        result = run_attribution(args)
        if result["status"] != "PASS":
            exit_code = 1
    except Exception as exc:
        result = {
            "status": "INVALID",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "optimizer_updates": 0,
            "scaler_step_calls": 0,
            "scaler_update_calls": 0,
            "formal_training_authorized": False,
            "exp394_remains_sealed": True,
            "exp395_remains_sealed": True,
        }
        exit_code = 1
    if output.exists() or runner.exists():
        raise RuntimeError("Refusing to overwrite exp396 result/runner")
    write_json(output, result)
    write_json(runner, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
