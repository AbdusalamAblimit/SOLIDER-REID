#!/usr/bin/env python3
"""Actual-batch CUDA/AMP preflight for exp394 rich evidence budget TAPF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
RHO_STAR = 0.08075544983148575
EXPECTED_SHA256 = {
    "model/tapf.py": "95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886",
    "model/clip_semantic_teacher.py": "c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5",
    "model/make_model.py": "6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03",
    "processor/processor.py": "be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1",
    "config/defaults.py": "a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5",
    "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml": "e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e",
    "model/backbones/swin_transformer.py": "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "datasets/pose_dataset.py": "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc",
}
CLIP_SHA256 = "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
CODEBOOK_SHA256 = "fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a"
STEPS = 24
TEACHER_STEPS = 12
PEAK_LIMIT_BYTES = 22 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def tensor_state_cpu(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def state_sha256(state) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_snapshot(parameters):
    return [parameter.detach().cpu().clone() for parameter in parameters]


def parameter_trajectory(parameters, initial):
    changed = 0
    maximum = 0.0
    finite = True
    for parameter, before in zip(parameters, initial):
        after = parameter.detach().cpu()
        difference = (after.float() - before.float()).abs()
        changed += int(bool(torch.count_nonzero(difference)))
        maximum = max(maximum, float(difference.max()))
        finite &= bool(torch.isfinite(after).all())
    return {
        "parameters": len(parameters),
        "changed": changed,
        "max_abs": maximum,
        "finite": finite,
    }


def parameter_groups(model):
    tapf = model.base.tapf
    tapf_ids = {id(parameter) for parameter in tapf.parameters()}
    backbone_live_candidates = [
        parameter
        for name, parameter in model.base.named_parameters()
        if id(parameter) not in tapf_ids
        and parameter.requires_grad
        and not name.startswith(("norm0.", "norm1.", "norm2."))
    ]
    anchor_trunk = [
        *tapf.anchor.project.parameters(),
        *tapf.anchor.depthwise.parameters(),
        *tapf.anchor.norm.parameters(),
    ]
    mask_presence = [
        *tapf.anchor.region_mask_head.parameters(),
        *tapf.anchor.presence_head.parameters(),
    ]
    pose_heads = list(tapf.anchor.pose_head.parameters())
    router_groups = {}
    for index, router in enumerate(tapf.psg_bank):
        router_groups[f"router{index}_projections"] = [
            *router.token_projection.parameters(),
            *router.context_projection.parameters(),
            *router.evidence_projection.parameters(),
        ]
        router_groups[f"router{index}_experts"] = list(router.experts.parameters())
    groups = {
        "backbone": backbone_live_candidates,
        "anchor_trunk": anchor_trunk,
        "mask_presence_heads": mask_presence,
        "pose_heads": pose_heads,
        "evidence_head": list(tapf.anchor.evidence_head.parameters()),
        "id_head": [
            parameter
            for module in (model.bottleneck, model.classifier)
            for parameter in module.parameters()
            if parameter.requires_grad
        ],
        **router_groups,
    }
    return groups


def grad_report(groups):
    report = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters]
        present = [gradient for gradient in gradients if gradient is not None]
        nonzero = [
            gradient
            for gradient in present
            if bool(torch.count_nonzero(gradient.detach()))
        ]
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in present)
        report[name] = {
            "parameters": len(parameters),
            "grad_present": len(present),
            "grad_nonzero": len(nonzero),
            "grad_abs_max": max(
                (float(gradient.detach().abs().max()) for gradient in present),
                default=0.0,
            ),
            "finite": finite,
        }
    return report


def group_active(report, name):
    item = report[name]
    return (
        item["parameters"] > 0
        and item["grad_present"] == item["parameters"]
        and item["grad_nonzero"] == item["parameters"]
        and item["grad_abs_max"] > 0
        and item["finite"]
    )


def group_off(report, name):
    item = report[name]
    return item["grad_nonzero"] == 0 and item["finite"]


def transfer_batch(batch, device):
    image, vid, camid, viewid, pose = batch
    pose_device = {
        "relative_paths": pose["relative_paths"],
        "image_sha256": pose["image_sha256"],
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
    return {
        "semantic_teacher_evidence": targets["evidence_code"].detach().clone(),
        "semantic_valid": targets["valid"].detach().clone(),
        "semantic_teacher_mask": targets["region_masks"].detach().clone(),
    }, targets


def model_pose_batch(pose, targets):
    return {
        "keypoints": pose["keypoints"],
        "scores": pose["scores"],
        "valid": pose["valid"],
        **targets,
    }


def forward_train(model, loss_fn, image, target, camid, viewid, pose, epoch):
    output = model(
        image,
        label=target,
        cam_label=camid,
        view_label=viewid,
        pose_batch=pose,
        tapf_epoch=epoch,
    )
    score, feature, _, aux = output
    reid = loss_fn(score, feature, target, camid)
    total = reid + 0.1 * aux["pose_loss"]
    return score, feature, aux, reid, total


def descriptor_variant(
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
    restore_rng(saved_rng)
    model.train()
    tapf = model.base.tapf
    original = tapf.apply_gate

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
        tapf.apply_gate = original
        model.load_state_dict(model_state, strict=True)
        restore_rng(saved_rng)
        model.zero_grad(set_to_none=True)
    return descriptor


def isolated_gradients(
    model,
    model_state,
    saved_rng,
    groups,
    loss_fn,
    batch,
    pose,
):
    image, target, camid, viewid = batch
    output = {}
    for name in ("evidence", "mask_presence", "exec", "reid"):
        model.load_state_dict(model_state, strict=True)
        restore_rng(saved_rng)
        model.zero_grad(set_to_none=True)
        model.train()
        with torch.amp.autocast("cuda", enabled=True):
            _, _, aux, reid, _ = forward_train(
                model, loss_fn, image, target, camid, viewid, pose, epoch=6
            )
            if name == "evidence":
                loss = aux["evidence_cos_loss"] + aux["evidence_relation_loss"]
            elif name == "mask_presence":
                loss = aux["region_mask_loss"] + aux["presence_loss"]
            elif name == "exec":
                loss = aux["exec_loss"]
            else:
                loss = reid
        loss.backward()
        output[name] = {
            "loss": float(loss.detach()),
            "gradients": grad_report(groups),
            "heatmap_loss": float(aux["heatmap_loss"].detach()),
            "confidence_loss": float(aux["confidence_loss"].detach()),
        }
    model.load_state_dict(model_state, strict=True)
    restore_rng(saved_rng)
    model.zero_grad(set_to_none=True)
    return output


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


def nvidia_processes():
    output = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def run_preflight(args):
    repo_root = Path(args.repo_root).resolve()
    config_path = (repo_root / args.config).resolve()
    output_path = Path(args.output).resolve()
    if run_text(["git", "rev-parse", "HEAD"], repo_root) != SOURCE_COMMIT:
        raise RuntimeError("Source commit mismatch")
    if run_text(
        ["git", "status", "--porcelain", "--untracked-files=no"], repo_root
    ):
        raise RuntimeError("Source repository has tracked modifications")
    actual_sha = {
        relative: sha256_file(repo_root / relative) for relative in EXPECTED_SHA256
    }
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError("Source SHA contract mismatch")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if output_path.exists():
        raise RuntimeError("Preflight result already exists")
    initial_processes = nvidia_processes()
    if initial_processes:
        raise RuntimeError("GPU already has a compute process")

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    from config import cfg
    from datasets import make_dataloader
    from loss import make_loss
    from model import make_model
    from model.clip_semantic_teacher import FrozenRichClipEvidenceTeacher
    from solver import make_optimizer

    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    config_gates = {
        "batch64": int(cfg.SOLVER.IMS_PER_BATCH) == 64,
        "seed1234": int(cfg.SOLVER.SEED) == 1234,
        "sgd": str(cfg.SOLVER.OPTIMIZER_NAME) == "SGD",
        "lr": float(cfg.SOLVER.BASE_LR) == 0.0008,
        "teacher_handoff": (
            int(cfg.MODEL.TAPF.TEACHER_EPOCHS) == 5
            and int(cfg.MODEL.TAPF.HANDOFF_EPOCHS) == 5
        ),
        "rho": float(cfg.MODEL.TAPF.RESIDUAL_RHO) == RHO_STAR,
        "rich_on": bool(cfg.MODEL.TAPF.RICH_EVIDENCE_ENABLED),
        "semantic_on": bool(cfg.MODEL.TAPF.SEMANTIC_ENABLED),
        "loss_weight": float(cfg.MODEL.TAPF.POSE_LOSS_WEIGHT) == 0.1,
        "checkpoint_sha": cfg.MODEL.TAPF.CLIP_CHECKPOINT_SHA256 == CLIP_SHA256,
        "codebook_sha": cfg.MODEL.TAPF.RICH_CODEBOOK_SHA256 == CODEBOOK_SHA256,
        "workers8": int(cfg.DATALOADER.NUM_WORKERS) == 8,
    }
    if not all(config_gates.values()):
        raise RuntimeError("Merged config contract failed")
    clip_path = Path(cfg.MODEL.TAPF.CLIP_CHECKPOINT)
    codebook_path = Path(cfg.MODEL.TAPF.RICH_CODEBOOK)
    asset_gates = {
        "clip_regular": clip_path.is_file() and not clip_path.is_symlink(),
        "clip_sha": sha256_file(clip_path) == CLIP_SHA256,
        "codebook_sha": sha256_file(codebook_path) == CODEBOOK_SHA256,
    }
    if not all(asset_gates.values()):
        raise RuntimeError("Teacher asset contract failed")
    codebook_payload = json.loads(codebook_path.read_text(encoding="utf-8"))
    means = torch.as_tensor(codebook_payload["slot_means"], dtype=torch.float64)
    basis = torch.as_tensor(codebook_payload["shared_basis"], dtype=torch.float64)
    asset_gates.update(
        {
            "codebook_shape": means.shape == (5, 768) and basis.shape == (16, 768),
            "codebook_finite": bool(
                torch.isfinite(means).all() and torch.isfinite(basis).all()
            ),
            "basis_orthogonal": float(
                (basis @ basis.T - torch.eye(16)).abs().max()
            )
            <= 1e-8,
        }
    )
    if not all(asset_gates.values()):
        raise RuntimeError("Codebook content contract failed")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA preflight requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    device_name = torch.cuda.get_device_name(device)
    if "4090" not in device_name:
        raise RuntimeError(f"Expected the exclusive RTX 4090, got {device_name}")
    set_seed(int(cfg.SOLVER.SEED))

    train_loader, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    scaler = torch.amp.GradScaler("cuda")

    teacher_rng = rng_state()
    teacher = FrozenRichClipEvidenceTeacher(
        checkpoint=cfg.MODEL.TAPF.CLIP_CHECKPOINT,
        checkpoint_sha256=cfg.MODEL.TAPF.CLIP_CHECKPOINT_SHA256,
        codebook=cfg.MODEL.TAPF.RICH_CODEBOOK,
        codebook_sha256=cfg.MODEL.TAPF.RICH_CODEBOOK_SHA256,
        device=device,
        microbatch=cfg.MODEL.TAPF.CLIP_MICROBATCH,
    )
    restore_rng(teacher_rng)
    tapf = model.base.tapf
    router_hook_counts_before = [
        len(router._forward_pre_hooks) for router in tapf.psg_bank
    ]
    groups = parameter_groups(model)
    initial_group_state = {
        name: parameter_snapshot(parameters) for name, parameters in groups.items()
    }
    initial_model_state = tensor_state_cpu(model)
    initial_state_sha = state_sha256(initial_model_state)
    teacher_versions = [parameter._version for parameter in teacher.visual.parameters()]
    teacher_codebook_versions = (
        teacher.slot_means._version,
        teacher.shared_basis._version,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    teacher_parameter_ids = {id(parameter) for parameter in teacher.visual.parameters()}
    isolation_gates = {
        "teacher_not_module_child": all(
            child is not teacher for child in model.modules()
        ),
        "teacher_not_optimizer": not bool(teacher_parameter_ids & optimizer_ids),
        "teacher_all_frozen": teacher.all_parameters_frozen(),
        "rho_not_state": all(
            "rho" not in name.lower()
            for name, _ in tuple(model.named_parameters())
            + tuple(model.named_buffers())
        ),
        "consumer_storage_independent": all(
            left.data_ptr() != right.data_ptr()
            for left, right in zip(
                tapf.psg_bank[0].parameters(), tapf.psg_bank[1].parameters()
            )
        ),
        "initial_state_finite": all(
            bool(torch.isfinite(value).all()) for value in initial_model_state.values()
        ),
    }
    if not all(isolation_gates.values()):
        raise RuntimeError("Initial isolation contract failed")

    manifest = []
    step_records = []
    iterator = iter(train_loader)
    diagnostic_batch = None
    diagnostic_pose = None
    first_epoch1 = None
    first_epoch6 = None
    router_inputs = None
    teacher_seconds = 0.0
    model_seconds = 0.0
    began_all = time.perf_counter()

    for step in range(1, STEPS + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw_batch = next(iterator)
        image, target, camid, viewid, pose = transfer_batch(raw_batch, device)
        manifest.append(
            {
                "step": step,
                "relative_paths": list(pose["relative_paths"]),
                "image_sha256": list(pose["image_sha256"]),
                "pids": target.detach().cpu().tolist(),
            }
        )
        teacher_start = time.perf_counter()
        targets, raw_targets = teacher_targets(teacher, pose)
        torch.cuda.synchronize(device)
        teacher_seconds += time.perf_counter() - teacher_start
        train_pose = model_pose_batch(pose, targets)
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
            "valid_norm": (
                valid_norm.numel() > 0
                and bool(torch.isfinite(valid_norm).all())
                and float((valid_norm - 1.0).abs().max()) < 1e-5
            ),
        }
        if not all(target_gates.values()):
            raise RuntimeError(f"Teacher target contract failed at step {step}")
        epoch = 1 if step <= TEACHER_STEPS else 6
        expected_rho = 0.0 if epoch == 1 else RHO_STAR / 5.0

        before_small = {
            "id": model.classifier.weight.detach().clone(),
            "evidence": tapf.anchor.evidence_head.weight.detach().clone(),
            "router0": tapf.psg_bank[0].experts[0].weight.detach().clone(),
            "router1": tapf.psg_bank[1].experts[0].weight.detach().clone(),
        }
        captured = []
        handles = []
        if step == TEACHER_STEPS + 1:
            def capture_input(module, inputs):
                captured.append(
                    tuple(
                        item.detach().clone() if torch.is_tensor(item) else item
                        for item in inputs
                    )
                )
            handles = [router.register_forward_pre_hook(capture_input) for router in tapf.psg_bank]

        optimizer.zero_grad(set_to_none=True)
        model.train()
        model_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=True):
            _, feature, aux, reid_loss, total_loss = forward_train(
                model,
                loss_fn,
                image,
                target,
                camid,
                viewid,
                train_pose,
                epoch,
            )
        for handle in handles:
            handle.remove()
        if step == TEACHER_STEPS + 1:
            router_inputs = captured
        scale_before = float(scaler.get_scale())
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        gradient_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        if not gradient_finite:
            raise RuntimeError(f"Non-finite gradient at step {step}")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        model_seconds += time.perf_counter() - model_start
        scale_after = float(scaler.get_scale())
        after_small = {
            "id": model.classifier.weight.detach(),
            "evidence": tapf.anchor.evidence_head.weight.detach(),
            "router0": tapf.psg_bank[0].experts[0].weight.detach(),
            "router1": tapf.psg_bank[1].experts[0].weight.detach(),
        }
        changed_small = {
            key: not torch.equal(before_small[key], after_small[key])
            for key in before_small
        }
        step_gates = {
            "rho_exact": aux["rho"] == expected_rho,
            "loss_finite": all(
                bool(torch.isfinite(value).all())
                for value in (
                    total_loss,
                    reid_loss,
                    aux["pose_loss"],
                    aux["semantic_loss"],
                    aux["evidence_cos_loss"],
                    aux["evidence_relation_loss"],
                    aux["exec_loss"],
                )
            ),
            "two_router_calls": len(aux["gate_deltas"]) == 2,
            "router_finite": all(
                bool(torch.isfinite(value).all())
                for value in aux["gate_deltas"]
            ),
            "teacher_identity": (
                epoch != 1
                or all(
                    torch.equal(delta, torch.zeros_like(delta))
                    for delta in aux["gate_deltas"]
                )
            ),
            "handoff_nonzero": (
                epoch == 1
                or any(bool(torch.count_nonzero(delta)) for delta in aux["gate_deltas"])
            ),
            "gradient_finite": gradient_finite,
            "amp_no_skip": scale_after >= scale_before,
            "id_updated": changed_small["id"],
            "aux_updated": (
                changed_small["evidence"]
                and changed_small["router0"]
                and changed_small["router1"]
            ),
        }
        if not all(step_gates.values()):
            raise RuntimeError(f"Optimizer step contract failed at step {step}: {step_gates}")
        step_records.append(
            {
                "step": step,
                "epoch": epoch,
                "rho": aux["rho"],
                "loss": float(total_loss.detach()),
                "reid_loss": float(reid_loss.detach()),
                "pose_loss": float(aux["pose_loss"].detach()),
                "semantic_loss": float(aux["semantic_loss"].detach()),
                "evidence_cos_loss": float(aux["evidence_cos_loss"].detach()),
                "evidence_relation_loss": float(aux["evidence_relation_loss"].detach()),
                "exec_loss": float(aux["exec_loss"].detach()),
                "scale_before": scale_before,
                "scale_after": scale_after,
                "gate_abs": [
                    float(delta.detach().float().abs().mean())
                    for delta in aux["gate_deltas"]
                ],
                "valid_slots": int(valid.sum()),
                "gates": step_gates,
            }
        )
        if step == 1:
            first_epoch1 = (image, target, camid, viewid, train_pose)
        if step == TEACHER_STEPS + 1:
            first_epoch6 = (image, target, camid, viewid, train_pose)
        if step == STEPS:
            diagnostic_batch = (image, target, camid, viewid)
            diagnostic_pose = train_pose

    elapsed_all = time.perf_counter() - began_all
    final_state = tensor_state_cpu(model)
    final_state_sha = state_sha256(final_state)
    trajectories = {
        name: parameter_trajectory(groups[name], initial_group_state[name])
        for name in groups
    }

    diagnostic_rng = rng_state()
    epoch1_state = final_state
    epoch1_full = descriptor_variant(
        model, epoch1_state, diagnostic_rng, *first_epoch1, epoch=1, bypass=()
    )
    epoch1_bypass = descriptor_variant(
        model, epoch1_state, diagnostic_rng, *first_epoch1, epoch=1, bypass=(0, 1)
    )
    epoch6_full = descriptor_variant(
        model, final_state, diagnostic_rng, *first_epoch6, epoch=6, bypass=()
    )
    epoch6_all_bypass = descriptor_variant(
        model, final_state, diagnostic_rng, *first_epoch6, epoch=6, bypass=(0, 1)
    )
    epoch6_bypass0 = descriptor_variant(
        model, final_state, diagnostic_rng, *first_epoch6, epoch=6, bypass=(0,)
    )
    epoch6_bypass1 = descriptor_variant(
        model, final_state, diagnostic_rng, *first_epoch6, epoch=6, bypass=(1,)
    )
    descriptor_gates = {
        "epoch1_full_bypass_exact": torch.equal(epoch1_full, epoch1_bypass),
        "epoch6_all_bypass_nonzero": not torch.equal(epoch6_full, epoch6_all_bypass),
        "epoch6_consumer0_nonzero": not torch.equal(epoch6_full, epoch6_bypass0),
        "epoch6_consumer1_nonzero": not torch.equal(epoch6_full, epoch6_bypass1),
        "descriptors_finite": all(
            bool(torch.isfinite(value).all())
            for value in (
                epoch1_full,
                epoch1_bypass,
                epoch6_full,
                epoch6_all_bypass,
                epoch6_bypass0,
                epoch6_bypass1,
            )
        ),
    }
    descriptor_diagnostics = {
        "epoch6_all_bypass_max_abs": float(
            (epoch6_full - epoch6_all_bypass).abs().max()
        ),
        "epoch6_all_bypass_mean_l2": float(
            (epoch6_full - epoch6_all_bypass).float().norm(dim=1).mean()
        ),
        "epoch6_bypass0_max_abs": float((epoch6_full - epoch6_bypass0).abs().max()),
        "epoch6_bypass1_max_abs": float((epoch6_full - epoch6_bypass1).abs().max()),
    }
    if not all(descriptor_gates.values()):
        raise RuntimeError("Descriptor bypass contract failed")

    if router_inputs is None or len(router_inputs) != 2:
        raise RuntimeError("Could not capture both router inputs")
    proposal_diagnostics = []
    proposal_gates = []
    for index, inputs in enumerate(router_inputs):
        tokens, hw_shape, mask, presence, evidence, rho = inputs
        router = tapf.psg_bank[index]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            correct = router.branch(tokens, hw_shape, mask, presence, evidence)
            wrong = router.branch(
                tokens, hw_shape, mask, presence, evidence.roll(1, 0)
            )
            static = router.branch(
                tokens, hw_shape, mask, presence, torch.zeros_like(evidence)
            )
            null_mask_output, null_mask_delta, null_mask = router(
                tokens,
                hw_shape,
                torch.zeros_like(mask),
                presence,
                evidence,
                RHO_STAR,
            )
            null_presence_output, null_presence_delta, null_presence = router(
                tokens,
                hw_shape,
                mask,
                torch.zeros_like(presence),
                evidence,
                RHO_STAR,
            )
        valid_normalized = correct["normalized_proposal"][correct["slot_valid"]]
        normalized_rms = valid_normalized.float().square().mean(-1).sqrt()
        proposal_float = correct["proposal"].float()
        expected_normalized = (
            proposal_float
            / (
                proposal_float.square()
                .mean(dim=-1, keepdim=True)
                .sqrt()
                .detach()
                + 1e-6
            )
        ).to(tokens.dtype)
        expected_normalized = torch.where(
            correct["slot_valid"][:, :, None, None],
            expected_normalized,
            torch.zeros_like(expected_normalized),
        )
        gates = {
            "correct_wrong_nonexact": not torch.equal(
                correct["proposal"], wrong["proposal"]
            ),
            "correct_static_nonexact": not torch.equal(
                correct["proposal"], static["proposal"]
            ),
            "all_finite": all(
                bool(torch.isfinite(value).all())
                for value in (
                    correct["proposal"],
                    correct["normalized_proposal"],
                    correct["unit_delta"],
                    wrong["proposal"],
                    static["proposal"],
                )
            ),
            "fp32_rms_formula_exact": torch.equal(
                correct["normalized_proposal"], expected_normalized
            ),
            "valid_rms_finite": (
                normalized_rms.numel() > 0
                and bool(torch.isfinite(normalized_rms).all())
            ),
            "valid_rms_near_one": (
                normalized_rms.numel() > 0
                and float((normalized_rms - 1.0).abs().max()) < 1e-3
            ),
            "null_mask_identity": torch.equal(null_mask_output, tokens),
            "null_mask_delta_zero": torch.equal(
                null_mask_delta, torch.zeros_like(null_mask_delta)
            ),
            "null_mask_normalized_zero": torch.equal(
                null_mask["normalized_proposal"],
                torch.zeros_like(null_mask["normalized_proposal"]),
            ),
            "null_presence_identity": torch.equal(null_presence_output, tokens),
            "null_presence_delta_zero": torch.equal(
                null_presence_delta, torch.zeros_like(null_presence_delta)
            ),
            "null_presence_normalized_zero": torch.equal(
                null_presence["normalized_proposal"],
                torch.zeros_like(null_presence["normalized_proposal"]),
            ),
        }
        proposal_gates.append(gates)
        proposal_diagnostics.append(
            {
                "consumer": index,
                "gates": gates,
                "correct_wrong_max_abs": float(
                    (correct["proposal"] - wrong["proposal"]).abs().max()
                ),
                "correct_static_max_abs": float(
                    (correct["proposal"] - static["proposal"]).abs().max()
                ),
                "normalized_rms_min": float(normalized_rms.min()),
                "normalized_rms_max": float(normalized_rms.max()),
                "rho_capture": rho,
            }
        )
    if not all(all(gates.values()) for gates in proposal_gates):
        raise RuntimeError("Proposal/NULL contract failed")

    isolated = isolated_gradients(
        model,
        final_state,
        diagnostic_rng,
        groups,
        loss_fn,
        diagnostic_batch,
        diagnostic_pose,
    )
    router_names = (
        "router0_projections",
        "router0_experts",
        "router1_projections",
        "router1_experts",
    )
    evidence_report = isolated["evidence"]["gradients"]
    mask_report = isolated["mask_presence"]["gradients"]
    exec_report = isolated["exec"]["gradients"]
    reid_report = isolated["reid"]["gradients"]
    ownership_gates = {
        "evidence_updates_head": group_active(evidence_report, "evidence_head"),
        "evidence_blocks_others": all(
            group_off(evidence_report, name)
            for name in (
                "backbone",
                "anchor_trunk",
                "mask_presence_heads",
                "pose_heads",
                "id_head",
                *router_names,
            )
        ),
        "mask_updates_anchor": (
            group_active(mask_report, "anchor_trunk")
            and group_active(mask_report, "mask_presence_heads")
        ),
        "mask_blocks_others": all(
            group_off(mask_report, name)
            for name in (
                "backbone",
                "pose_heads",
                "evidence_head",
                "id_head",
                *router_names,
            )
        ),
        "exec_updates_owned": (
            group_active(exec_report, "evidence_head")
            and all(group_active(exec_report, name) for name in router_names)
        ),
        "exec_blocks_others": all(
            group_off(exec_report, name)
            for name in (
                "backbone",
                "anchor_trunk",
                "mask_presence_heads",
                "pose_heads",
                "id_head",
            )
        ),
        "reid_updates_owned": (
            group_active(reid_report, "backbone")
            and group_active(reid_report, "id_head")
            and all(group_active(reid_report, name) for name in router_names)
        ),
        "reid_blocks_anchor": all(
            group_off(reid_report, name)
            for name in (
                "anchor_trunk",
                "mask_presence_heads",
                "pose_heads",
                "evidence_head",
            )
        ),
    }
    if not all(ownership_gates.values()):
        raise RuntimeError(f"Gradient ownership contract failed: {ownership_gates}")

    model.load_state_dict(final_state, strict=True)
    restore_rng(diagnostic_rng)
    model.zero_grad(set_to_none=True)
    teacher_isolation_final = {
        "versions_exact": teacher_versions
        == [parameter._version for parameter in teacher.visual.parameters()],
        "codebook_versions_exact": teacher_codebook_versions
        == (teacher.slot_means._version, teacher.shared_basis._version),
        "all_grads_none": all(
            parameter.grad is None for parameter in teacher.visual.parameters()
        ),
    }
    if not all(teacher_isolation_final.values()):
        raise RuntimeError("Teacher changed during preflight")
    memory_before_teacher_delete = int(torch.cuda.memory_allocated(device))
    del teacher
    torch.cuda.empty_cache()
    memory_after_teacher_delete = int(torch.cuda.memory_allocated(device))

    reload_rng = rng_state()
    reloaded = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device)
    incompatible = reloaded.load_state_dict(final_state, strict=True)
    restore_rng(reload_rng)
    image, target, camid, viewid = diagnostic_batch
    correct_pose = diagnostic_pose
    shuffled_pose = {
        key: value.roll(1, 0) if torch.is_tensor(value) else value
        for key, value in diagnostic_pose.items()
    }
    ExplodingPose.accesses = 0
    eval_correct = eval_descriptor(model, image, camid, viewid, correct_pose)
    eval_shuffle = eval_descriptor(model, image, camid, viewid, shuffled_pose)
    eval_none = eval_descriptor(model, image, camid, viewid, None)
    eval_exploding = eval_descriptor(model, image, camid, viewid, ExplodingPose())
    reload_descriptor = eval_descriptor(reloaded, image, camid, viewid, None)
    eval_correct_end = eval_descriptor(model, image, camid, viewid, correct_pose)
    state_names = tuple(final_state)
    forbidden_components = {"teacher", "clip", "codebook", "text", "pose_batch"}
    terminal_gates = {
        "strict_reload": (
            not incompatible.missing_keys and not incompatible.unexpected_keys
        ),
        "reload_descriptor_exact": torch.equal(eval_none, reload_descriptor),
        "rgb_correct_shuffle_exact": torch.equal(eval_correct, eval_shuffle),
        "rgb_correct_none_exact": torch.equal(eval_correct, eval_none),
        "rgb_correct_exploding_exact": torch.equal(eval_correct, eval_exploding),
        "correct_start_end_exact": torch.equal(eval_correct, eval_correct_end),
        "exploding_access_zero": ExplodingPose.accesses == 0,
        "state_teacher_free": all(
            not (set(name.lower().split(".")) & forbidden_components)
            for name in state_names
        ),
        "evidence_head_retained": any("anchor.evidence_head" in name for name in state_names),
        "routers_retained": all(
            any(f"psg_bank.{index}.evidence_projection" in name for name in state_names)
            for index in (0, 1)
        ),
        "final_state_finite": all(
            bool(torch.isfinite(value).all()) for value in final_state.values()
        ),
        "source_sha_unchanged": {
            relative: sha256_file(repo_root / relative) for relative in EXPECTED_SHA256
        }
        == EXPECTED_SHA256,
        "asset_sha_unchanged": (
            sha256_file(clip_path) == CLIP_SHA256
            and sha256_file(codebook_path) == CODEBOOK_SHA256
        ),
        "diagnostic_state_exact": state_sha256(model.state_dict())
        == final_state_sha,
        "router_hooks_restored": router_hook_counts_before
        == [len(router._forward_pre_hooks) for router in tapf.psg_bank],
        "teacher_memory_reclaimed": memory_after_teacher_delete
        < memory_before_teacher_delete,
        "tracked_clean": not bool(
            run_text(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                repo_root,
            )
        ),
    }
    del reloaded
    torch.cuda.empty_cache()
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    terminal_gates["peak_under_22_gib"] = peak_allocated < PEAK_LIMIT_BYTES
    checkpoint_count = 0
    output_dir = (repo_root / str(cfg.OUTPUT_DIR)).resolve()
    if output_dir.exists():
        checkpoint_count = len(list(output_dir.rglob("*.pth")))
    terminal_gates["checkpoint_zero"] = checkpoint_count == 0
    if not all(terminal_gates.values()):
        raise RuntimeError(f"Terminal contract failed: {terminal_gates}")

    trajectory_gates = {
        "backbone": trajectories["backbone"]["changed"] > 0,
        "anchor": (
            trajectories["anchor_trunk"]["changed"] > 0
            and trajectories["mask_presence_heads"]["changed"] > 0
            and trajectories["evidence_head"]["changed"] > 0
        ),
        "id_head": trajectories["id_head"]["changed"] > 0,
        "pose_heads": trajectories["pose_heads"]["changed"] > 0,
        "two_routers": all(
            trajectories[name]["changed"] == trajectories[name]["parameters"]
            and trajectories[name]["finite"]
            for name in router_names
        ),
        "all_finite": all(item["finite"] for item in trajectories.values()),
    }
    if not all(trajectory_gates.values()):
        raise RuntimeError(f"Parameter trajectory contract failed: {trajectory_gates}")

    gates = {
        "source": actual_sha == EXPECTED_SHA256,
        "config": all(config_gates.values()),
        "assets": all(asset_gates.values()),
        "initial_isolation": all(isolation_gates.values()),
        "steps": len(step_records) == STEPS
        and all(all(item["gates"].values()) for item in step_records),
        "descriptors": all(descriptor_gates.values()),
        "proposals_null": all(all(item["gates"].values()) for item in proposal_diagnostics),
        "ownership": all(ownership_gates.values()),
        "teacher_isolation": all(teacher_isolation_final.values()),
        "trajectories": all(trajectory_gates.values()),
        "terminal": all(terminal_gates.values()),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "verdict": "CUDA_AMP_PREFLIGHT_PASS" if all(gates.values()) else "CUDA_AMP_PREFLIGHT_FAIL",
        "gates": gates,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": actual_sha,
        "config_gates": config_gates,
        "asset_gates": asset_gates,
        "isolation_gates": isolation_gates,
        "step_records": step_records,
        "batch_manifest_sha256": sha256_json(manifest),
        "batch_manifest": manifest,
        "descriptor_gates": descriptor_gates,
        "descriptor_diagnostics": descriptor_diagnostics,
        "proposal_diagnostics": proposal_diagnostics,
        "ownership_gates": ownership_gates,
        "isolated_gradients": isolated,
        "teacher_isolation_final": teacher_isolation_final,
        "trajectories": trajectories,
        "trajectory_gates": trajectory_gates,
        "terminal_gates": terminal_gates,
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": final_state_sha,
        "state_changed": initial_state_sha != final_state_sha,
        "checkpoint_count": checkpoint_count,
        "teacher_seconds": teacher_seconds,
        "model_seconds": model_seconds,
        "elapsed_seconds": elapsed_all,
        "throughput_images_per_second": STEPS * 64 / elapsed_all,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "memory_after_teacher_delete_bytes": memory_after_teacher_delete,
        "memory_before_teacher_delete_bytes": memory_before_teacher_delete,
        "initial_gpu_processes": initial_processes,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    if result["status"] != "PASS":
        raise RuntimeError("CUDA/AMP preflight gates failed")
    return result


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    try:
        result = run_preflight(args)
        write_json(output, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        failure = {
            "status": "FAIL",
            "verdict": "CUDA_AMP_PREFLIGHT_FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        write_json(output, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
