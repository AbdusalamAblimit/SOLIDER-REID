"""Measure matched B0/D0 model-only compute, memory, and throughput."""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from mmengine.analysis import get_model_complexity_info
from torch.cuda import amp

from config import cfg as default_cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer


NUM_CLASSES = 702
CAMERA_NUM = 8
VIEW_NUM = 1
TEACHER_PATH = "/home/afr/reid-clean/weights/solider_swin_tiny_tea.pth"
B0_CONFIG = "configs/occluded_duke/swin_tiny.yml"
D0_CONFIG = "configs/occluded_duke/swin_tiny_tapf_d0.yml"
DATASET_ROOT = None
SEMANTIC_WEIGHT = None


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


def tensor_sha256(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_cfg(tapf):
    config = default_cfg.clone()
    config.merge_from_file(D0_CONFIG if tapf else B0_CONFIG)
    config.defrost()
    config.MODEL.PRETRAIN_CHOICE = "self"
    config.MODEL.PRETRAIN_PATH = TEACHER_PATH
    if DATASET_ROOT is not None:
        config.DATASETS.ROOT_DIR = DATASET_ROOT
    if SEMANTIC_WEIGHT is not None:
        config.MODEL.SEMANTIC_WEIGHT = SEMANTIC_WEIGHT
    config.freeze()
    return config


def load_fixed_real_batches():
    config = make_cfg(True)
    set_seed(config.SOLVER.SEED)
    train_loader, _, val_loader, _, _, _, _ = make_dataloader(config)
    train_iterator = iter(train_loader)
    try:
        train_batch = next(train_iterator)
    finally:
        if hasattr(train_iterator, "_shutdown_workers"):
            train_iterator._shutdown_workers()
    val_iterator = iter(val_loader)
    try:
        val_batch = next(val_iterator)
    finally:
        if hasattr(val_iterator, "_shutdown_workers"):
            val_iterator._shutdown_workers()
    if tuple(train_batch[0].shape) != (64, 3, 384, 128):
        raise RuntimeError("Unexpected fixed train batch")
    if tuple(val_batch[0].shape) != (256, 3, 384, 128):
        raise RuntimeError("Unexpected fixed eval batch")
    return train_batch, val_batch[0]


def move_train_batch(cpu_batch, tapf):
    images, labels, cameras, views, cpu_pose = cpu_batch
    images = images.cuda()
    labels = labels.cuda()
    cameras = cameras.cuda()
    views = views.cuda()
    pose = None
    if tapf:
        pose = {
            "keypoints": cpu_pose["keypoints"].cuda(),
            "scores": cpu_pose["scores"].cuda(),
            "valid": cpu_pose["valid"].cuda(),
        }
    return images, labels, cameras, views, pose


def found_inf_value(scaler, optimizer):
    state = scaler._per_optimizer_states[id(optimizer)]
    return float(
        sum(
            value.detach().float().item()
            for value in state["found_inf_per_device"].values()
        )
    )


def train_step(model, optimizer, scaler, loss_fn, config, batch, tapf):
    images, labels, cameras, views, pose = batch
    optimizer.zero_grad()
    with amp.autocast(enabled=True):
        output = model(
            images,
            label=labels,
            cam_label=cameras,
            view_label=views,
            pose_batch=pose,
            tapf_epoch=11,
        )
        if tapf:
            score, feature, _, aux = output
            loss = loss_fn(score, feature, labels, cameras)
            loss = loss + config.MODEL.TAPF.POSE_LOSS_WEIGHT * aux["pose_loss"]
        else:
            score, feature, _ = output
            loss = loss_fn(score, feature, labels, cameras)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite efficiency loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if found_inf_value(scaler, optimizer) != 0.0:
        raise RuntimeError("Safe-scale efficiency step overflowed")
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().item())


def train_benchmark(config, tapf, cpu_batch, warmup_steps, measured_steps):
    torch.cuda.empty_cache()
    set_seed(config.SOLVER.SEED)
    model = make_model(
        config,
        NUM_CLASSES,
        CAMERA_NUM,
        VIEW_NUM,
        config.MODEL.SEMANTIC_WEIGHT,
    ).cuda().train()
    loss_fn, center_criterion = make_loss(config, NUM_CLASSES)
    optimizer, _ = make_optimizer(config, model, center_criterion)
    scaler = amp.GradScaler(init_scale=1.0)
    batch = move_train_batch(cpu_batch, tapf)

    for _ in range(warmup_steps):
        train_step(model, optimizer, scaler, loss_fn, config, batch, tapf)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    losses = []
    for _ in range(measured_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        losses.append(
            train_step(model, optimizer, scaler, loss_fn, config, batch, tapf)
        )
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    result = {
        "batch_size": config.SOLVER.IMS_PER_BATCH,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "mean_step_ms": statistics.mean(timings),
        "median_step_ms": statistics.median(timings),
        "samples_per_second": (
            1000.0 * config.SOLVER.IMS_PER_BATCH / statistics.mean(timings)
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "final_loss": losses[-1],
        "final_scale": float(scaler.get_scale()),
    }
    del batch, scaler, optimizer, center_criterion, loss_fn, model
    torch.cuda.empty_cache()
    return result


def eval_benchmark(config, tapf, cpu_images, warmup_steps, measured_steps):
    torch.cuda.empty_cache()
    set_seed(config.SOLVER.SEED)
    model = make_model(
        config,
        NUM_CLASSES,
        CAMERA_NUM,
        VIEW_NUM,
        config.MODEL.SEMANTIC_WEIGHT,
    ).cuda().eval()
    images = cpu_images.cuda()

    with torch.no_grad():
        for _ in range(warmup_steps):
            descriptor, _ = model(images, pose_batch=None, tapf_epoch=None)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(measured_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            descriptor, _ = model(images, pose_batch=None, tapf_epoch=None)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    if tuple(descriptor.shape) != (config.TEST.IMS_PER_BATCH, 768):
        raise RuntimeError("Unexpected eval descriptor shape")
    if not torch.isfinite(descriptor).all():
        raise RuntimeError("Non-finite eval descriptor")
    result = {
        "batch_size": config.TEST.IMS_PER_BATCH,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "mean_step_ms": statistics.mean(timings),
        "median_step_ms": statistics.median(timings),
        "images_per_second": (
            1000.0 * config.TEST.IMS_PER_BATCH / statistics.mean(timings)
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "descriptor_shape": list(descriptor.shape),
        "pose_input": None,
    }

    complexity = get_model_complexity_info(
        model,
        inputs=(torch.randn(1, 3, *config.INPUT.SIZE_TEST, device="cuda"),),
        show_table=False,
        show_arch=False,
    )
    result["supported_op_flops"] = int(complexity["flops"])
    result["analyzer_activations"] = int(complexity["activations"])
    result["parameters"] = int(complexity["params"])
    del descriptor, images, model
    torch.cuda.empty_cache()
    return result


def audit_arm(
    tapf,
    train_batch,
    eval_images,
    train_warmup,
    train_steps,
    eval_warmup,
    eval_steps,
):
    config = make_cfg(tapf)
    return {
        "config": D0_CONFIG if tapf else B0_CONFIG,
        "config_sha256": sha256_file(D0_CONFIG if tapf else B0_CONFIG),
        "train": train_benchmark(
            config, tapf, train_batch, train_warmup, train_steps
        ),
        "eval_rgb_only": eval_benchmark(
            config, tapf, eval_images, eval_warmup, eval_steps
        ),
    }


def relative_delta(d0, b0):
    return {
        key: d0[key] - b0[key]
        for key in (
            "mean_step_ms",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        )
    }


def main():
    global NUM_CLASSES, CAMERA_NUM, VIEW_NUM, B0_CONFIG, D0_CONFIG
    global DATASET_ROOT, SEMANTIC_WEIGHT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--b0-config", default=B0_CONFIG)
    parser.add_argument("--d0-config", default=D0_CONFIG)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--camera-num", type=int, default=CAMERA_NUM)
    parser.add_argument("--view-num", type=int, default=VIEW_NUM)
    parser.add_argument("--dataset-root")
    parser.add_argument("--semantic-weight", type=float)
    parser.add_argument("--train-warmup", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--eval-warmup", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=20)
    args = parser.parse_args()

    B0_CONFIG = args.b0_config
    D0_CONFIG = args.d0_config
    NUM_CLASSES = args.num_classes
    CAMERA_NUM = args.camera_num
    VIEW_NUM = args.view_num
    DATASET_ROOT = args.dataset_root
    SEMANTIC_WEIGHT = args.semantic_weight

    train_batch, eval_images = load_fixed_real_batches()
    b0 = audit_arm(
        False,
        train_batch,
        eval_images,
        args.train_warmup,
        args.train_steps,
        args.eval_warmup,
        args.eval_steps,
    )
    d0 = audit_arm(
        True,
        train_batch,
        eval_images,
        args.train_warmup,
        args.train_steps,
        args.eval_warmup,
        args.eval_steps,
    )
    train_delta = relative_delta(d0["train"], b0["train"])
    eval_delta = relative_delta(d0["eval_rgb_only"], b0["eval_rgb_only"])
    flop_delta = (
        d0["eval_rgb_only"]["supported_op_flops"]
        - b0["eval_rgb_only"]["supported_op_flops"]
    )
    parameter_delta = (
        d0["eval_rgb_only"]["parameters"]
        - b0["eval_rgb_only"]["parameters"]
    )
    if parameter_delta != 105442:
        raise RuntimeError("Unexpected TAPF parameter delta")
    if flop_delta <= 0:
        raise RuntimeError("Expected positive TAPF FLOP delta")

    result = {
        "status": "TAPF_MATCHED_EFFICIENCY_PASS",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "script_sha256": sha256_file(__file__),
        "measurement_scope": {
            "train": "one fixed real paired batch; model forward+backward+SGD step, AMP, scale=1",
            "eval": "one fixed real RGB-only validation batch; model forward, FP32",
            "flops": "MMEngine supported-operator trace; unsupported elementwise ops excluded equally",
        },
        "fixed_batch_sha256": {
            "train_rgb": tensor_sha256(train_batch[0]),
            "train_pid": tensor_sha256(train_batch[1]),
            "train_pose_keypoints": tensor_sha256(train_batch[4]["keypoints"]),
            "eval_rgb": tensor_sha256(eval_images),
        },
        "b0": b0,
        "d0": d0,
        "delta": {
            "parameters": parameter_delta,
            "parameter_percent": 100.0 * parameter_delta / b0["eval_rgb_only"]["parameters"],
            "supported_op_flops": flop_delta,
            "supported_op_flops_percent": (
                100.0 * flop_delta / b0["eval_rgb_only"]["supported_op_flops"]
            ),
            "train": train_delta,
            "eval_rgb_only": eval_delta,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    print("parameter_delta={}".format(parameter_delta))
    print("supported_op_flops_delta={}".format(flop_delta))
    print("train_delta={}".format(json.dumps(train_delta, sort_keys=True)))
    print("eval_delta={}".format(json.dumps(eval_delta, sort_keys=True)))
    print("output={}".format(output_path))


if __name__ == "__main__":
    main()
