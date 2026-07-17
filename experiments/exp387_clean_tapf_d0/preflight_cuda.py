"""Run the real paired batch64 CUDA/AMP stability gate for exp387."""

import argparse
import hashlib
import json
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
    if name.startswith("base.tapf.anchor."):
        return "anchor"
    if name.startswith("base.tapf.psg_bank."):
        return "psg"
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
        for name in ("swin", "anchor", "psg", "head")
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = stats[parameter_group(name)]
        gradient = parameter.grad.detach().float()
        group["tensor_count"] += 1
        if torch.count_nonzero(gradient).item():
            group["nonzero_tensors"] += 1
        if torch.isfinite(gradient).all():
            group["finite_abs_sum"] += float(gradient.double().abs().sum().item())
        else:
            group["nonfinite_tensors"] += 1
    return stats


def group_parameter_snapshots(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def changed_parameter_summary(model, initial):
    summary = {
        name: {"changed_tensors": 0, "total_tensors": 0}
        for name in ("swin", "anchor", "psg", "head")
    }
    for name, parameter in model.named_parameters():
        group = summary[parameter_group(name)]
        group["total_tensors"] += 1
        if not torch.equal(parameter.detach().cpu(), initial[name]):
            group["changed_tensors"] += 1
    return summary


def optimizer_state_summary(optimizer):
    tensor_count = 0
    nonfinite_count = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                tensor_count += 1
                if not torch.isfinite(value).all():
                    nonfinite_count += 1
    return {
        "state_entries": len(optimizer.state),
        "tensor_count": tensor_count,
        "nonfinite_count": nonfinite_count,
    }


def select_probes(model):
    probes = {}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        if group == "psg":
            if name == "base.tapf.psg_bank.0.output_projection.weight":
                probes[group] = (name, parameter)
        elif group not in probes:
            probes[group] = (name, parameter)
    missing = set(("swin", "anchor", "psg", "head")) - set(probes)
    if missing:
        raise RuntimeError("Missing parameter probe groups: {}".format(sorted(missing)))
    return probes


def found_inf_value(scaler, optimizer):
    state = scaler._per_optimizer_states[id(optimizer)]
    values = state["found_inf_per_device"].values()
    return float(sum(value.detach().float().item() for value in values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/occluded_duke/swin_tiny_tapf_d0.yml"
    )
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    config_path = Path(args.config)
    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    if cfg.SOLVER.IMS_PER_BATCH != 64 or cfg.DATALOADER.NUM_WORKERS != 8:
        raise RuntimeError("This gate requires the formal batch64/8-worker config")
    if not cfg.MODEL.TAPF.ENABLED:
        raise RuntimeError("TAPF must be enabled")

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
    ).cuda()
    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    model.train()
    scaler = amp.GradScaler()
    probes = select_probes(model)
    initial_parameters = group_parameter_snapshots(model)
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
                raise RuntimeError("Unexpected paired keypoint batch shape")
            if tuple(pose_batch["scores"].shape) != (64, 17):
                raise RuntimeError("Unexpected paired score batch shape")
            if tuple(pose_batch["valid"].shape) != (64, 17):
                raise RuntimeError("Unexpected paired valid batch shape")
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
                score, feature, featmaps, tapf_aux = model(
                    images,
                    label=labels,
                    cam_label=cameras,
                    view_label=views,
                    pose_batch=cuda_pose,
                    tapf_epoch=1,
                )
                reid_loss = loss_fn(score, feature, labels, cameras)
                total_loss = reid_loss + (
                    cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * tapf_aux["pose_loss"]
                )

            require_finite("score", score)
            require_finite("feature", feature)
            require_finite("featmaps", featmaps)
            require_finite("reid_loss", reid_loss)
            require_finite("pose_loss", tapf_aux["pose_loss"])
            require_finite("student_field", tapf_aux["student_field"])
            require_finite("teacher_field", tapf_aux["teacher_field"])
            require_finite("consumer_field", tapf_aux["consumer_field"])
            require_finite("reliability", tapf_aux["reliability"])
            require_finite("gate_deltas", tapf_aux["gate_deltas"])
            require_finite("total_loss", total_loss)
            if len(tapf_aux["gate_deltas"]) != 2:
                raise RuntimeError("Both independent PSG banks must be consumed")
            if tapf_aux["student_fraction"] != 0.0:
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
                if scale_after >= scale_before:
                    raise RuntimeError("Overflow did not reduce GradScaler scale")
                if any(updated.values()):
                    raise RuntimeError("An overflow step changed a parameter probe")
                consecutive_updates = 0
            else:
                if scale_after != scale_before:
                    raise RuntimeError("Finite step unexpectedly changed GradScaler scale")
                if not all(updated.values()):
                    raise RuntimeError(
                        "Finite step failed to update probes: {}".format(updated)
                    )
                successful_updates += 1
                consecutive_updates += 1
                longest_consecutive_updates = max(
                    longest_consecutive_updates, consecutive_updates
                )

            records.append(
                {
                    "step": step,
                    "loss": float(total_loss.detach().float().item()),
                    "reid_loss": float(reid_loss.detach().float().item()),
                    "pose_loss": float(tapf_aux["pose_loss"].detach().float().item()),
                    "reliability_mean": float(
                        tapf_aux["reliability"].detach().float().mean().item()
                    ),
                    "student_field_mean": float(
                        tapf_aux["student_field"].detach().float().mean().item()
                    ),
                    "gate_abs_mean": [
                        float(delta.detach().float().abs().mean().item())
                        for delta in tapf_aux["gate_deltas"]
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

    if successful_updates < 1:
        raise RuntimeError("Default GradScaler never reached a finite optimizer update")
    if longest_consecutive_updates < 8:
        raise RuntimeError("Fewer than eight consecutive finite optimizer updates")

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError("Non-finite final model parameter: {}".format(name))
    optimizer_summary = optimizer_state_summary(optimizer)
    if optimizer_summary["nonfinite_count"]:
        raise RuntimeError("Non-finite optimizer state")
    changed_summary = changed_parameter_summary(model, initial_parameters)
    for group in ("swin", "anchor", "psg", "head"):
        if changed_summary[group]["changed_tensors"] == 0:
            raise RuntimeError("No final parameter update in {}".format(group))

    elapsed_values = [record["elapsed_seconds"] for record in records]
    result = {
        "status": "EXP387_REAL_BATCH64_CUDA_AMP_PASS",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(__file__),
        "batch_size": train_loader.batch_size,
        "workers": train_loader.num_workers,
        "steps": args.steps,
        "successful_updates": successful_updates,
        "overflow_steps": args.steps - successful_updates,
        "longest_consecutive_updates": longest_consecutive_updates,
        "initial_scale": records[0]["scale_before"],
        "final_scale": records[-1]["scale_after"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "mean_step_seconds": float(np.mean(elapsed_values)),
        "median_step_seconds": float(np.median(elapsed_values)),
        "changed_parameters": changed_summary,
        "optimizer": optimizer_summary,
        "probe_names": {group: item[0] for group, item in probes.items()},
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
    print("peak_allocated_bytes={}".format(result["peak_allocated_bytes"]))
    print("peak_reserved_bytes={}".format(result["peak_reserved_bytes"]))
    print("mean_step_seconds={:.6f}".format(result["mean_step_seconds"]))
    print("changed_parameters={}".format(json.dumps(changed_summary, sort_keys=True)))
    print("output={}".format(output_path))


if __name__ == "__main__":
    main()
