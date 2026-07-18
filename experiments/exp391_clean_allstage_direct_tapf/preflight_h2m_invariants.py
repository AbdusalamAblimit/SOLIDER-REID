"""Audit that H2-M changes only hierarchical pose-loss reduction."""

import argparse
import copy
import hashlib
import json
import random
import types
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp

from config import cfg as default_cfg
from loss import make_loss
from model import make_model
from solver import make_optimizer


class ExplodingPose:
    def __init__(self):
        self.accesses = 0

    def __getitem__(self, key):
        self.accesses += 1
        raise AssertionError("Evaluation must not read external pose")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_config(path):
    config = default_cfg.clone()
    config.merge_from_file(path)
    config.freeze()
    return config


def make_model_and_optimizer(config):
    model = make_model(
        config,
        num_class=702,
        camera_num=8,
        view_num=1,
        semantic_weight=config.MODEL.SEMANTIC_WEIGHT,
    ).cuda()
    loss_fn, center_criterion = make_loss(config, num_classes=702)
    optimizer, _ = make_optimizer(config, model, center_criterion)
    return model, loss_fn, optimizer


def synthetic_batch(batch_size=4):
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


def tensor_sha(named_tensors):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        count += 1
    return digest.hexdigest(), count


def require_named_tensors_equal(left, right, label):
    left = list(left)
    right = list(right)
    if [name for name, _ in left] != [name for name, _ in right]:
        raise RuntimeError("Tensor names differ for {}".format(label))
    for (name, a), (_, b) in zip(left, right):
        if not torch.equal(a.detach().cpu(), b.detach().cpu()):
            raise RuntimeError("Tensor mismatch for {} {}".format(label, name))


def optimizer_description(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        {
            "names": [names[id(parameter)] for parameter in group["params"]],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "momentum": group.get("momentum", 0.0),
        }
        for group in optimizer.param_groups
    ]


def normalize_mean_config(sum_config, mean_config):
    normalized = mean_config.clone()
    normalized.defrost()
    normalized.OUTPUT_DIR = sum_config.OUTPUT_DIR
    normalized.MODEL.TAPF.POSE_LOSS_REDUCTION = "sum"
    normalized.freeze()
    if normalized.dump() != sum_config.dump():
        raise RuntimeError("H2-M config changes more than reduction/output")


def forward_once(model, batch, epoch):
    images, labels, cameras, views, pose = batch
    score, feature, featmaps, aux = model(
        images,
        label=labels,
        cam_label=cameras,
        view_label=views,
        pose_batch=pose,
        tapf_epoch=epoch,
    )
    return score, feature, featmaps, aux


def exact_forward_tensors(output):
    score, feature, featmaps, aux = output
    tensors = [("score", score), ("feature", feature)]
    tensors.extend(
        ("featmap{}".format(index), value)
        for index, value in enumerate(featmaps)
    )
    for name in (
        "early_pose_loss",
        "late_pose_loss",
        "early_consumer_field",
        "late_consumer_field",
        "early_student_field",
        "late_student_field",
        "early_teacher_field",
        "late_teacher_field",
        "early_reliability",
        "late_reliability",
    ):
        tensors.append((name, aux[name]))
    tensors.extend(
        ("early_gate{}".format(index), value)
        for index, value in enumerate(aux["early_gate_deltas"])
    )
    tensors.extend(
        ("late_gate{}".format(index), value)
        for index, value in enumerate(aux["late_gate_deltas"])
    )
    return tensors


def gradient_pair_audit(
    sum_model, mean_model, sum_loss_fn, mean_loss_fn, batch, epoch, weight
):
    sum_model.zero_grad(set_to_none=True)
    mean_model.zero_grad(set_to_none=True)
    torch.manual_seed(9300 + epoch)
    torch.cuda.manual_seed_all(9300 + epoch)
    sum_output = forward_once(sum_model, batch, epoch)
    torch.manual_seed(9300 + epoch)
    torch.cuda.manual_seed_all(9300 + epoch)
    mean_output = forward_once(mean_model, batch, epoch)
    require_named_tensors_equal(
        exact_forward_tensors(sum_output), exact_forward_tensors(mean_output), "epoch{}".format(epoch)
    )
    sum_aux = sum_output[3]
    mean_aux = mean_output[3]
    if not torch.equal(mean_aux["pose_loss"] * 2.0, sum_aux["pose_loss"]):
        raise RuntimeError("Mean pose loss is not exact half at epoch {}".format(epoch))
    sum_reid = sum_loss_fn(sum_output[0], sum_output[1], batch[1], batch[2])
    mean_reid = mean_loss_fn(mean_output[0], mean_output[1], batch[1], batch[2])
    if not torch.equal(sum_reid, mean_reid):
        raise RuntimeError("ReID loss differs at epoch {}".format(epoch))
    (sum_reid + weight * sum_aux["pose_loss"]).backward()
    (mean_reid + weight * mean_aux["pose_loss"]).backward()

    sum_parameters = dict(sum_model.named_parameters())
    mean_parameters = dict(mean_model.named_parameters())
    if list(sum_parameters) != list(mean_parameters):
        raise RuntimeError("Parameter names differ")
    anchor_count = 0
    common_count = 0
    for name in sum_parameters:
        left = sum_parameters[name].grad
        right = mean_parameters[name].grad
        if (left is None) != (right is None):
            raise RuntimeError("Gradient presence differs for {}".format(name))
        if left is None:
            continue
        is_anchor = name.startswith("base.tapf.anchor.") or name.startswith(
            "base.tapf.early_anchor."
        )
        if is_anchor:
            if not torch.equal(right * 2.0, left):
                raise RuntimeError("Anchor gradient is not exact half: {}".format(name))
            anchor_count += 1
        else:
            if not torch.equal(right, left):
                raise RuntimeError("Non-anchor gradient differs: {}".format(name))
            common_count += 1
    if anchor_count != 16 or common_count == 0:
        raise RuntimeError("Unexpected gradient coverage")
    return {
        "epoch": epoch,
        "student_fraction": float(mean_aux["early_student_fraction"]),
        "early_gate_count": len(mean_aux["early_gate_deltas"]),
        "late_gate_count": len(mean_aux["late_gate_deltas"]),
        "anchor_gradient_tensors": anchor_count,
        "common_gradient_tensors": common_count,
        "sum_pose_loss": float(sum_aux["pose_loss"].detach()),
        "mean_pose_loss": float(mean_aux["pose_loss"].detach()),
    }


def eval_capture(model, images, pose):
    model.eval()
    with torch.no_grad():
        descriptor, _ = model(images, pose_batch=pose, tapf_epoch=None)
        _, _, aux = model.base(images, pose_batch=pose, tapf_epoch=None)
    return {
        "descriptor": descriptor.detach().cpu(),
        "early_field": aux["early_student_field"].detach().cpu(),
        "late_field": aux["late_student_field"].detach().cpu(),
        "early_gates": [value.detach().cpu() for value in aux["early_gate_deltas"]],
        "late_gates": [value.detach().cpu() for value in aux["late_gate_deltas"]],
    }


def require_capture_equal(actual, expected, label):
    for name in ("descriptor", "early_field", "late_field"):
        if not torch.equal(actual[name], expected[name]):
            raise RuntimeError("{} differs for {}".format(name, label))
    for name in ("early_gates", "late_gates"):
        if len(actual[name]) != len(expected[name]):
            raise RuntimeError("Gate count differs for {}".format(label))
        for left, right in zip(actual[name], expected[name]):
            if not torch.equal(left, right):
                raise RuntimeError("{} differs for {}".format(name, label))


def pose_free_audit(model, images, pose):
    reference = eval_capture(model, images, pose)
    exploding = ExplodingPose()
    variants = {
        "shuffle": {name: value.flip(0) for name, value in pose.items()},
        "none": None,
        "exploding": exploding,
    }
    for label, variant in variants.items():
        require_capture_equal(eval_capture(model, images, variant), reference, label)
    if exploding.accesses != 0:
        raise RuntimeError("Exploding pose was accessed")
    return {
        "variants": ["correct", "shuffle", "none", "exploding"],
        "exploding_accesses": exploding.accesses,
        "descriptor_shape": list(reference["descriptor"].shape),
        "early_field_shape": list(reference["early_field"].shape),
        "late_field_shape": list(reference["late_field"].shape),
        "early_gate_count": len(reference["early_gates"]),
        "late_gate_count": len(reference["late_gates"]),
    }


def consumer_path_audit(model, images):
    model.eval()
    generator = torch.Generator().manual_seed(391)
    for bank in list(model.base.tapf.early_psg_bank) + list(model.base.tapf.psg_bank):
        with torch.no_grad():
            values = torch.randn(
                bank.output_projection.weight.shape, generator=generator
            ) * 1.0e-3
            bank.output_projection.weight.copy_(values.to(bank.output_projection.weight.device))
    with torch.no_grad():
        reference, _ = model(images, pose_batch=None, tapf_epoch=None)
    consumers = [
        ("early{}".format(index), bank)
        for index, bank in enumerate(model.base.tapf.early_psg_bank)
    ] + [
        ("late{}".format(index), bank)
        for index, bank in enumerate(model.base.tapf.psg_bank)
    ]
    records = []
    for name, bank in consumers:
        original = bank.forward

        def bypass(self, tokens, hw_shape, field):
            del self, hw_shape, field
            return tokens, torch.zeros_like(tokens)

        bank.forward = types.MethodType(bypass, bank)
        try:
            with torch.no_grad():
                descriptor, _ = model(images, pose_batch=None, tapf_epoch=None)
        finally:
            bank.forward = original
        delta = float((descriptor.detach().cpu() - reference.detach().cpu()).abs().max())
        if not np.isfinite(delta) or delta <= 0.0:
            raise RuntimeError("Dead consumer path: {}".format(name))
        records.append({"consumer": name, "descriptor_max_abs_delta": delta})
    if len(records) != 8:
        raise RuntimeError("Expected eight consumer paths")
    return records


def strict_roundtrip(model, config, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    for name, tensor in state.items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
            tensor
        ).all():
            raise RuntimeError("Non-finite state tensor: {}".format(name))
    set_seed(config.SOLVER.SEED)
    restored, _, _ = make_model_and_optimizer(config)
    incompatible = restored.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict roundtrip failed")
    return {
        "state_tensors": len(state),
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sum-config", required=True)
    parser.add_argument("--mean-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sum_config = make_config(args.sum_config)
    mean_config = make_config(args.mean_config)
    normalize_mean_config(sum_config, mean_config)
    if sum_config.MODEL.TAPF.POSE_LOSS_REDUCTION != "sum":
        raise RuntimeError("Legacy config must resolve to sum")
    if mean_config.MODEL.TAPF.POSE_LOSS_REDUCTION != "mean":
        raise RuntimeError("H2-M config must resolve to mean")

    set_seed(sum_config.SOLVER.SEED)
    sum_model, sum_loss_fn, sum_optimizer = make_model_and_optimizer(sum_config)
    sum_cpu_rng = torch.get_rng_state().clone()
    sum_cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    set_seed(mean_config.SOLVER.SEED)
    mean_model, mean_loss_fn, mean_optimizer = make_model_and_optimizer(mean_config)
    mean_cpu_rng = torch.get_rng_state().clone()
    mean_cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]

    require_named_tensors_equal(
        sum_model.state_dict().items(), mean_model.state_dict().items(), "initial_state"
    )
    if not torch.equal(sum_cpu_rng, mean_cpu_rng) or any(
        not torch.equal(left, right)
        for left, right in zip(sum_cuda_rng, mean_cuda_rng)
    ):
        raise RuntimeError("Construction RNG differs")
    if optimizer_description(sum_model, sum_optimizer) != optimizer_description(
        mean_model, mean_optimizer
    ):
        raise RuntimeError("Optimizer membership differs")

    initial_state = copy.deepcopy(sum_model.state_dict())
    batch = synthetic_batch()
    epoch_records = []
    for epoch in (1, 6, 10, 11):
        sum_model.load_state_dict(initial_state, strict=True)
        mean_model.load_state_dict(initial_state, strict=True)
        sum_model.train()
        mean_model.train()
        epoch_records.append(
            gradient_pair_audit(
                sum_model,
                mean_model,
                sum_loss_fn,
                mean_loss_fn,
                batch,
                epoch,
                mean_config.MODEL.TAPF.POSE_LOSS_WEIGHT,
            )
        )

    mean_model.load_state_dict(initial_state, strict=True)
    pose_free = pose_free_audit(mean_model, batch[0], batch[4])
    mean_model.load_state_dict(initial_state, strict=True)
    consumer_paths = consumer_path_audit(mean_model, batch[0])
    mean_model.load_state_dict(initial_state, strict=True)
    strict = strict_roundtrip(mean_model, mean_config, Path(args.checkpoint))
    initial_sha, initial_count = tensor_sha(initial_state.items())
    result = {
        "status": "EXP391_H2M_INVARIANTS_PASS",
        "config_single_variable": ["MODEL.TAPF.POSE_LOSS_REDUCTION", "OUTPUT_DIR"],
        "state_init_exact": True,
        "initial_state_sha256": initial_sha,
        "initial_state_tensors": initial_count,
        "construction_rng_exact": True,
        "optimizer_exact": True,
        "optimizer_parameter_groups": len(sum_optimizer.param_groups),
        "epochs": epoch_records,
        "pose_free": pose_free,
        "consumer_paths": consumer_paths,
        "strict_roundtrip": strict,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
