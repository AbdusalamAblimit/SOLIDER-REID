"""Audit full-model TAPF routes, isolation, overflow, and RGB-only state."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer


class ExplodingPose:
    def __getitem__(self, key):
        raise AssertionError("Evaluation must not read external pose")


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


def parameter_group(name):
    if name.startswith("base.tapf.anchor."):
        return "anchor"
    if name.startswith("base.tapf.psg_bank."):
        return "psg"
    if name.startswith("base."):
        return "swin"
    return "head"


def gradient_summary(model):
    summary = {
        group: {"tensor_count": 0, "nonzero_tensors": 0, "abs_sum": 0.0}
        for group in ("swin", "anchor", "psg", "head")
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().double()
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Non-finite isolation gradient: {}".format(name))
        group = summary[parameter_group(name)]
        group["tensor_count"] += 1
        if torch.count_nonzero(gradient).item():
            group["nonzero_tensors"] += 1
        group["abs_sum"] += float(gradient.abs().sum().item())
    return summary


def require_only_nonzero(summary, expected):
    for group, values in summary.items():
        nonzero = values["nonzero_tensors"] > 0 and values["abs_sum"] > 0.0
        if group in expected and not nonzero:
            raise RuntimeError("Expected nonzero gradient group: {}".format(group))
        if group not in expected and nonzero:
            raise RuntimeError("Unexpected gradient route into {}".format(group))


def clone_parameters(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def require_parameters_equal(model, expected):
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter.detach().cpu(), expected[name]):
            raise RuntimeError("Skipped overflow changed parameter: {}".format(name))


def clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested(item) for item in value)
    return copy.deepcopy(value)


def require_nested_equal(actual, expected, path="root"):
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.equal(actual.detach().cpu(), expected):
            raise RuntimeError("Optimizer state mismatch at {}".format(path))
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or list(actual) != list(expected):
            raise RuntimeError("Optimizer mapping mismatch at {}".format(path))
        for key in expected:
            require_nested_equal(actual[key], expected[key], "{}.{}".format(path, key))
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise RuntimeError("Optimizer sequence mismatch at {}".format(path))
        for index, (left, right) in enumerate(zip(actual, expected)):
            require_nested_equal(left, right, "{}[{}]".format(path, index))
        return
    if actual != expected:
        raise RuntimeError("Optimizer scalar mismatch at {}".format(path))


def found_inf_value(scaler, optimizer):
    state = scaler._per_optimizer_states[id(optimizer)]
    return float(
        sum(
            value.detach().float().item()
            for value in state["found_inf_per_device"].values()
        )
    )


def make_synthetic_batch(batch_size=4):
    generator = torch.Generator().manual_seed(20260718)
    images = torch.randn(
        batch_size, 3, 384, 128, generator=generator, dtype=torch.float32
    ).cuda()
    labels = torch.tensor([index // 2 for index in range(batch_size)], device="cuda")
    cameras = torch.tensor([index % 2 for index in range(batch_size)], device="cuda")
    views = torch.ones(batch_size, dtype=torch.long, device="cuda")
    keypoints = torch.zeros(batch_size, 17, 2, device="cuda")
    keypoints[..., 0] = torch.linspace(8.0, 120.0, 17, device="cuda")
    keypoints[..., 1] = torch.linspace(16.0, 368.0, 17, device="cuda")
    keypoints += torch.arange(batch_size, device="cuda")[:, None, None] * 0.25
    scores = torch.linspace(0.2, 1.1, 17, device="cuda").repeat(batch_size, 1)
    valid = torch.ones(batch_size, 17, dtype=torch.bool, device="cuda")
    valid[:, -1] = False
    pose = {"keypoints": keypoints, "scores": scores, "valid": valid}
    return images, labels, cameras, views, pose


def forward_train(model, batch, epoch):
    images, labels, cameras, views, pose = batch
    return model(
        images,
        label=labels,
        cam_label=cameras,
        view_label=views,
        pose_batch=pose,
        tapf_epoch=epoch,
    )


def route_audit(model, batch):
    counts = [0, 0]
    hooks = []
    for index, bank in enumerate(model.base.tapf.psg_bank):
        def count_call(module, inputs, output, bank_index=index):
            del module, inputs, output
            counts[bank_index] += 1

        hooks.append(bank.register_forward_hook(count_call))

    records = {}
    model.train()
    try:
        for epoch, expected_fraction in ((1, 0.0), (6, 0.2), (10, 1.0), (11, 1.0)):
            before = tuple(counts)
            with torch.no_grad(), amp.autocast(enabled=True):
                score, feature, featmaps, aux = forward_train(model, batch, epoch)
            for name, value in (
                ("score", score),
                ("feature", feature),
                ("student_field", aux["student_field"]),
                ("teacher_field", aux["teacher_field"]),
                ("consumer_field", aux["consumer_field"]),
                ("pose_loss", aux["pose_loss"]),
            ):
                values = value if isinstance(value, (list, tuple)) else (value,)
                if any(not torch.isfinite(item).all() for item in values):
                    raise RuntimeError("Non-finite route output: {}".format(name))
            if any(not torch.isfinite(item).all() for item in featmaps):
                raise RuntimeError("Non-finite route featmap")
            if len(aux["gate_deltas"]) != 2:
                raise RuntimeError("Expected exactly two gate deltas")
            if aux["student_fraction"] != expected_fraction:
                raise RuntimeError("Unexpected student fraction at e{}".format(epoch))
            if epoch == 1:
                expected_field = aux["teacher_field"]
            elif epoch == 6:
                expected_field = (
                    0.8 * aux["teacher_field"] + 0.2 * aux["student_field"]
                )
            else:
                expected_field = aux["student_field"]
            if not torch.equal(aux["consumer_field"], expected_field):
                raise RuntimeError("Consumer field route mismatch at e{}".format(epoch))
            after = tuple(counts)
            if tuple(right - left for left, right in zip(before, after)) != (1, 1):
                raise RuntimeError("Each PSG bank must be consumed exactly once")
            records[str(epoch)] = {
                "student_fraction": aux["student_fraction"],
                "bank_calls": [right - left for left, right in zip(before, after)],
                "pose_loss": float(aux["pose_loss"].float().item()),
                "student_field_mean": float(aux["student_field"].float().mean().item()),
                "teacher_field_mean": float(aux["teacher_field"].float().mean().item()),
                "gate_abs_mean": [
                    float(delta.float().abs().mean().item())
                    for delta in aux["gate_deltas"]
                ],
            }
    finally:
        for hook in hooks:
            hook.remove()
    return records


def gradient_audit(model, loss_fn, batch):
    images, labels, cameras, _, _ = batch
    del images

    model.train()
    model.zero_grad(set_to_none=True)
    _, _, _, pose_aux = forward_train(model, batch, epoch=11)
    pose_aux["pose_loss"].backward()
    pose_summary = gradient_summary(model)
    require_only_nonzero(pose_summary, {"anchor"})

    model.zero_grad(set_to_none=True)
    score, feature, _, _ = forward_train(model, batch, epoch=11)
    reid_loss = loss_fn(score, feature, labels, cameras)
    reid_loss.backward()
    reid_summary = gradient_summary(model)
    require_only_nonzero(reid_summary, {"swin", "psg", "head"})
    model.zero_grad(set_to_none=True)
    return {
        "pose_loss": pose_summary,
        "reid_loss": reid_summary,
        "reid_loss_value": float(reid_loss.detach().item()),
    }


def overflow_audit(model, optimizer, loss_fn, batch, pose_weight):
    _, labels, cameras, _, _ = batch
    scaler = amp.GradScaler(init_scale=1.0)
    model.train()
    optimizer.zero_grad()
    with amp.autocast(enabled=True):
        score, feature, _, aux = forward_train(model, batch, epoch=11)
        finite_loss = loss_fn(score, feature, labels, cameras)
        finite_loss = finite_loss + pose_weight * aux["pose_loss"]
    scaler.scale(finite_loss).backward()
    scaler.unscale_(optimizer)
    finite_found_inf = found_inf_value(scaler, optimizer)
    if finite_found_inf != 0.0:
        raise RuntimeError("Safe-scale warmup step was not finite")
    scaler.step(optimizer)
    scaler.update()

    parameters_before = clone_parameters(model)
    optimizer_before = clone_nested(optimizer.state_dict())
    scale_before = float(scaler.get_scale())
    optimizer.zero_grad()
    with amp.autocast(enabled=True):
        score, feature, _, aux = forward_train(model, batch, epoch=11)
        finite_base = loss_fn(score, feature, labels, cameras)
        finite_base = finite_base + pose_weight * aux["pose_loss"]
        overflow_loss = finite_base * torch.tensor(float("inf"), device="cuda")
    scaler.scale(overflow_loss).backward()
    scaler.unscale_(optimizer)
    overflow_found_inf = found_inf_value(scaler, optimizer)
    if overflow_found_inf <= 0.0:
        raise RuntimeError("Artificial non-finite did not trigger GradScaler")
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if scale_after >= scale_before:
        raise RuntimeError("Overflow did not lower GradScaler scale")
    require_parameters_equal(model, parameters_before)
    require_nested_equal(optimizer.state_dict(), optimizer_before)
    optimizer.zero_grad(set_to_none=True)
    return {
        "finite_warmup_found_inf": finite_found_inf,
        "overflow_found_inf": overflow_found_inf,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "parameter_count_checked": len(parameters_before),
        "optimizer_state_entries_checked": len(optimizer_before["state"]),
    }


def eval_capture(model, images, pose):
    model.eval()
    with torch.no_grad():
        descriptor, _ = model(images, pose_batch=pose, tapf_epoch=None)
        _, _, aux = model.base(images, pose_batch=pose, tapf_epoch=None)
    return {
        "descriptor": descriptor.detach().clone(),
        "student_field": aux["student_field"].detach().clone(),
        "gate_deltas": [delta.detach().clone() for delta in aux["gate_deltas"]],
    }


def require_capture_equal(actual, expected, label):
    if not torch.equal(actual["descriptor"], expected["descriptor"]):
        raise RuntimeError("Descriptor mismatch: {}".format(label))
    if not torch.equal(actual["student_field"], expected["student_field"]):
        raise RuntimeError("Student field mismatch: {}".format(label))
    if len(actual["gate_deltas"]) != len(expected["gate_deltas"]):
        raise RuntimeError("Gate count mismatch: {}".format(label))
    for index, (left, right) in enumerate(
        zip(actual["gate_deltas"], expected["gate_deltas"])
    ):
        if not torch.equal(left, right):
            raise RuntimeError("Gate mismatch {} bank {}".format(label, index))


def state_and_pose_free_audit(
    model, model_factory, images, correct_pose, checkpoint_path
):
    shuffled_pose = {
        key: value.flip(0) for key, value in correct_pose.items()
    }
    reference = eval_capture(model, images, correct_pose)
    variants = {
        "shuffle": shuffled_pose,
        "none": None,
        "exploding": ExplodingPose(),
    }
    for name, pose in variants.items():
        require_capture_equal(eval_capture(model, images, pose), reference, name)

    torch.save(model.state_dict(), checkpoint_path)
    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    for name, tensor in loaded_state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError("Non-finite checkpoint tensor: {}".format(name))
    restored = model_factory().cuda()
    incompatible = restored.load_state_dict(loaded_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict load returned incompatible keys")
    restored_capture = eval_capture(restored, images, None)
    require_capture_equal(restored_capture, reference, "strict_roundtrip")
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "state_tensor_count": len(loaded_state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "pose_variants_exact": ["correct", "shuffle", "none", "exploding"],
        "descriptor_shape": list(reference["descriptor"].shape),
        "student_field_shape": list(reference["student_field"].shape),
        "gate_shapes": [list(delta.shape) for delta in reference["gate_deltas"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config_path = Path("configs/occluded_duke/swin_tiny_tapf_d0.yml")
    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)
    (
        train_loader,
        train_loader_normal,
        val_loader,
        num_query,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)
    if hasattr(val_loader.dataset, "pose_store"):
        raise RuntimeError("Query/gallery dataset must not own a pose store")
    if hasattr(train_loader_normal.dataset, "pose_store"):
        raise RuntimeError("Normal RGB train evaluator must not own a pose store")
    if val_loader.dataset.__class__.__name__ != "ImageDataset":
        raise RuntimeError("Query/gallery must use the RGB ImageDataset")

    def model_factory():
        return make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
        )

    model = model_factory().cuda()
    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    batch = make_synthetic_batch()

    routes = route_audit(model, batch)
    gradients = gradient_audit(model, loss_fn, batch)
    overflow = overflow_audit(
        model,
        optimizer,
        loss_fn,
        batch,
        cfg.MODEL.TAPF.POSE_LOSS_WEIGHT,
    )
    state_pose_free = state_and_pose_free_audit(
        model,
        model_factory,
        batch[0],
        batch[4],
        Path(args.checkpoint),
    )

    result = {
        "status": "EXP387_FULL_SEMANTICS_PASS",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(__file__),
        "dataset_boundary": {
            "num_query": num_query,
            "val_dataset": val_loader.dataset.__class__.__name__,
            "val_has_pose_store": hasattr(val_loader.dataset, "pose_store"),
            "normal_train_has_pose_store": hasattr(
                train_loader_normal.dataset, "pose_store"
            ),
        },
        "routes": routes,
        "gradients": gradients,
        "overflow": overflow,
        "state_pose_free": state_pose_free,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    print("route_epochs={}".format(sorted(routes)))
    print("pose_gradient_groups=['anchor']")
    print("reid_gradient_groups=['swin', 'psg', 'head']")
    print("overflow={}".format(json.dumps(overflow, sort_keys=True)))
    print("checkpoint_sha256={}".format(state_pose_free["checkpoint_sha256"]))
    print("pose_variants_exact={}".format(state_pose_free["pose_variants_exact"]))
    print("output={}".format(output_path))


if __name__ == "__main__":
    main()
