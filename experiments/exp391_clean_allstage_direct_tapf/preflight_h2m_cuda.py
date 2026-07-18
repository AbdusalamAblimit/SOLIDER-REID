"""Run H2-M real paired batch64/8-worker formal-schedule CUDA/AMP gate."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


GROUPS = (
    "swin",
    "early_anchor",
    "late_anchor",
    "early_psg",
    "late_psg",
    "head",
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_finite(name, value):
    values = value if isinstance(value, (list, tuple)) else (value,)
    for index, tensor in enumerate(values):
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
            raise RuntimeError("Non-finite {}[{}]".format(name, index))


def parameter_group(name):
    if name.startswith("base.tapf.early_anchor."):
        return "early_anchor"
    if name.startswith("base.tapf.anchor."):
        return "late_anchor"
    if name.startswith("base.tapf.early_psg_bank."):
        return "early_psg"
    if name.startswith("base.tapf.psg_bank."):
        return "late_psg"
    if name.startswith("base."):
        return "swin"
    return "head"


def group_gradient_stats(model):
    stats = {
        name: {
            "tensor_count": 0,
            "nonzero_tensors": 0,
            "nonfinite_tensors": 0,
            "finite_abs_sum": 0.0,
        }
        for name in GROUPS
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = stats[parameter_group(name)]
        gradient = parameter.grad.detach().float()
        group["tensor_count"] += 1
        group["nonzero_tensors"] += int(torch.count_nonzero(gradient).item() > 0)
        if torch.isfinite(gradient).all():
            group["finite_abs_sum"] += float(gradient.double().abs().sum().item())
        else:
            group["nonfinite_tensors"] += 1
    return stats


def parameter_snapshots(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def changed_parameter_summary(model, initial):
    result = {
        name: {"changed_tensors": 0, "total_tensors": 0} for name in GROUPS
    }
    for name, parameter in model.named_parameters():
        group = result[parameter_group(name)]
        group["total_tensors"] += 1
        group["changed_tensors"] += int(
            not torch.equal(parameter.detach().cpu(), initial[name])
        )
    return result


def optimizer_state_summary(optimizer):
    tensor_count = 0
    nonfinite_count = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                tensor_count += 1
                nonfinite_count += int(not torch.isfinite(value).all())
    return {
        "state_entries": len(optimizer.state),
        "tensor_count": tensor_count,
        "nonfinite_count": nonfinite_count,
    }


def select_probes(model):
    preferred = {
        "early_psg": "base.tapf.early_psg_bank.0.output_projection.weight",
        "late_psg": "base.tapf.psg_bank.0.output_projection.weight",
    }
    probes = {}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        if group in preferred:
            if name == preferred[group]:
                probes[group] = (name, parameter)
        elif group not in probes:
            probes[group] = (name, parameter)
    missing = set(GROUPS) - set(probes)
    if missing:
        raise RuntimeError("Missing probe groups: {}".format(sorted(missing)))
    return probes


def found_inf_value(scaler, optimizer):
    values = scaler._per_optimizer_states[id(optimizer)][
        "found_inf_per_device"
    ].values()
    return float(sum(value.detach().float().item() for value in values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    config_path = Path(args.config)
    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    if cfg.SOLVER.IMS_PER_BATCH != 64 or cfg.DATALOADER.NUM_WORKERS != 8:
        raise RuntimeError("Gate requires formal batch64/8-worker config")
    if not cfg.MODEL.TAPF.HIERARCHICAL:
        raise RuntimeError("Gate requires hierarchical TAPF")
    if cfg.MODEL.TAPF.POSE_LOSS_REDUCTION != "mean":
        raise RuntimeError("Gate requires H2-M mean reduction")

    set_seed(cfg.SOLVER.SEED)
    (
        train_loader,
        _,
        _,
        _,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)
    if train_loader.batch_size != 64 or train_loader.num_workers != 8:
        raise RuntimeError("Unexpected real DataLoader geometry")

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).cuda().train()
    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    del scheduler
    initial_lrs = sorted(set(float(group["lr"]) for group in optimizer.param_groups))
    if len(initial_lrs) != 1 or not math.isclose(
        initial_lrs[0], 8.0e-06, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise RuntimeError("Unexpected formal initial LR: {}".format(initial_lrs))
    scaler = amp.GradScaler()
    probes = select_probes(model)
    initial_parameters = parameter_snapshots(model)
    iterator = iter(train_loader)

    records = []
    successful_updates = 0
    consecutive_updates = 0
    longest_consecutive_updates = 0
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(1, args.steps + 1):
            images, labels, cameras, views, pose_batch = next(iterator)
            if tuple(images.shape) != (64, 3, 384, 128):
                raise RuntimeError("Unexpected image batch shape")
            if tuple(pose_batch["keypoints"].shape) != (64, 17, 2):
                raise RuntimeError("Unexpected keypoint batch shape")
            if tuple(pose_batch["scores"].shape) != (64, 17):
                raise RuntimeError("Unexpected score batch shape")
            if tuple(pose_batch["valid"].shape) != (64, 17):
                raise RuntimeError("Unexpected valid batch shape")
            require_finite("input_images", images)
            require_finite("input_keypoints", pose_batch["keypoints"])
            require_finite("input_scores", pose_batch["scores"])

            images = images.cuda(non_blocking=False)
            labels = labels.cuda(non_blocking=False)
            cameras = cameras.cuda(non_blocking=False)
            views = views.cuda(non_blocking=False)
            cuda_pose = {
                "keypoints": pose_batch["keypoints"].cuda(non_blocking=False),
                "scores": pose_batch["scores"].cuda(non_blocking=False),
                "valid": pose_batch["valid"].cuda(non_blocking=False),
            }
            optimizer.zero_grad()
            probe_before = {
                group: parameter.detach().clone()
                for group, (_, parameter) in probes.items()
            }
            scale_before = float(scaler.get_scale())
            torch.cuda.synchronize()
            started = time.perf_counter()
            with amp.autocast(enabled=True):
                score, feature, featmaps, aux = model(
                    images,
                    label=labels,
                    cam_label=cameras,
                    view_label=views,
                    pose_batch=cuda_pose,
                    tapf_epoch=1,
                )
                reid_loss = loss_fn(score, feature, labels, cameras)
                total_loss = reid_loss + cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * aux[
                    "pose_loss"
                ]

            for name, value in (
                ("score", score),
                ("feature", feature),
                ("featmaps", featmaps),
                ("reid_loss", reid_loss),
                ("pose_loss", aux["pose_loss"]),
                ("early_pose_loss", aux["early_pose_loss"]),
                ("late_pose_loss", aux["late_pose_loss"]),
                ("early_student_field", aux["early_student_field"]),
                ("late_student_field", aux["late_student_field"]),
                ("early_teacher_field", aux["early_teacher_field"]),
                ("late_teacher_field", aux["late_teacher_field"]),
                ("early_reliability", aux["early_reliability"]),
                ("late_reliability", aux["late_reliability"]),
                ("early_gates", aux["early_gate_deltas"]),
                ("late_gates", aux["late_gate_deltas"]),
                ("total_loss", total_loss),
            ):
                require_finite(name, value)
            if len(aux["early_gate_deltas"]) != 6 or len(aux["late_gate_deltas"]) != 2:
                raise RuntimeError("H2-M must consume 6/2 independent gates")
            if aux["early_student_fraction"] != 0.0 or aux["late_student_fraction"] != 0.0:
                raise RuntimeError("Epoch-1 route must be teacher-only")

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            found_inf = found_inf_value(scaler, optimizer)
            gradient_stats = group_gradient_stats(model)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            scale_after = float(scaler.get_scale())
            updated = {
                group: not torch.equal(parameter.detach(), probe_before[group])
                for group, (_, parameter) in probes.items()
            }

            if found_inf > 0.0:
                if scale_after >= scale_before or any(updated.values()):
                    raise RuntimeError("Overflow step did not skip exactly")
                consecutive_updates = 0
            else:
                if scale_after != scale_before or not all(updated.values()):
                    raise RuntimeError("Finite step failed probe update: {}".format(updated))
                successful_updates += 1
                consecutive_updates += 1
                longest_consecutive_updates = max(
                    longest_consecutive_updates, consecutive_updates
                )

            records.append(
                {
                    "step": step,
                    "loss": float(total_loss.detach().float()),
                    "reid_loss": float(reid_loss.detach().float()),
                    "pose_loss": float(aux["pose_loss"].detach().float()),
                    "early_pose_loss": float(aux["early_pose_loss"].detach().float()),
                    "late_pose_loss": float(aux["late_pose_loss"].detach().float()),
                    "early_reliability_mean": float(
                        aux["early_reliability"].detach().float().mean()
                    ),
                    "late_reliability_mean": float(
                        aux["late_reliability"].detach().float().mean()
                    ),
                    "early_gate_abs_mean": [
                        float(value.detach().float().abs().mean())
                        for value in aux["early_gate_deltas"]
                    ],
                    "late_gate_abs_mean": [
                        float(value.detach().float().abs().mean())
                        for value in aux["late_gate_deltas"]
                    ],
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "found_inf": found_inf,
                    "updated_probes": updated,
                    "gradient_stats": gradient_stats,
                    "elapsed_seconds": elapsed,
                }
            )
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()

    if successful_updates < 1 or longest_consecutive_updates < 8:
        raise RuntimeError("Insufficient finite update sequence")
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError("Non-finite final parameter: {}".format(name))
    optimizer_summary = optimizer_state_summary(optimizer)
    if optimizer_summary["nonfinite_count"]:
        raise RuntimeError("Non-finite optimizer state")
    changed = changed_parameter_summary(model, initial_parameters)
    for group in GROUPS:
        if changed[group]["changed_tensors"] == 0:
            raise RuntimeError("No final parameter update in {}".format(group))

    elapsed_values = [record["elapsed_seconds"] for record in records]
    result = {
        "status": "EXP391_H2M_REAL_BATCH64_CUDA_AMP_PASS",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(__file__),
        "batch_size": train_loader.batch_size,
        "workers": train_loader.num_workers,
        "steps": args.steps,
        "formal_initial_lrs": initial_lrs,
        "successful_updates": successful_updates,
        "overflow_steps": args.steps - successful_updates,
        "longest_consecutive_updates": longest_consecutive_updates,
        "initial_scale": records[0]["scale_before"],
        "final_scale": records[-1]["scale_after"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "mean_step_seconds": float(np.mean(elapsed_values)),
        "median_step_seconds": float(np.median(elapsed_values)),
        "changed_parameters": changed,
        "optimizer": optimizer_summary,
        "probe_names": {group: value[0] for group, value in probes.items()},
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    print("successful_updates={}".format(successful_updates))
    print("overflow_steps={}".format(args.steps - successful_updates))
    print("longest_consecutive_updates={}".format(longest_consecutive_updates))
    print("scale={}->{}".format(result["initial_scale"], result["final_scale"]))
    print("changed_parameters={}".format(json.dumps(changed, sort_keys=True)))
    print("output={}".format(output_path))


if __name__ == "__main__":
    main()
