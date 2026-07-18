"""Audit a sealed clean D0 checkpoint without external pose at evaluation."""

import argparse
import hashlib
import json
import random
import types
from pathlib import Path

import numpy as np
import torch

from config import cfg
from datasets import make_dataloader
from model import make_model


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
    torch.backends.cudnn.benchmark = True


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def changed_summary(model, initial):
    groups = {
        name: {"changed": 0, "total": 0}
        for name in ("anchor", "psg", "swin", "head")
    }
    per_bank = [{"changed": 0, "total": 0} for _ in range(2)]
    for name, parameter in model.named_parameters():
        if name.startswith("base.tapf.anchor."):
            group = "anchor"
        elif name.startswith("base.tapf.psg_bank."):
            group = "psg"
        elif name.startswith("base."):
            group = "swin"
        else:
            group = "head"
        groups[group]["total"] += 1
        changed = not torch.equal(parameter.detach().cpu(), initial[name])
        groups[group]["changed"] += int(changed)
        if group == "psg":
            index = int(name.split(".")[3])
            per_bank[index]["total"] += 1
            per_bank[index]["changed"] += int(changed)
    if groups["anchor"] != {"changed": 8, "total": 8}:
        raise RuntimeError("Unexpected anchor trajectory: {}".format(groups["anchor"]))
    if groups["psg"] != {"changed": 4, "total": 4}:
        raise RuntimeError("Unexpected PSG trajectory: {}".format(groups["psg"]))
    for index, stats in enumerate(per_bank):
        if stats != {"changed": 2, "total": 2}:
            raise RuntimeError("Inactive PSG bank {}: {}".format(index, stats))
    return groups, per_bank


def bank_independence(banks):
    equal = all(
        torch.equal(left.detach().cpu(), right.detach().cpu())
        for left, right in zip(banks[0].parameters(), banks[1].parameters())
    )
    if equal:
        raise RuntimeError("The two PSG banks are identical")
    return {"left": 0, "right": 1, "equal": equal}


def synthetic_inputs(batch_size=4):
    generator = torch.Generator().manual_seed(20260718)
    images = torch.randn(
        batch_size, 3, 384, 128, generator=generator, dtype=torch.float32
    ).cuda()
    keypoints = torch.zeros(batch_size, 17, 2, device="cuda")
    keypoints[..., 0] = torch.linspace(8.0, 120.0, 17, device="cuda")
    keypoints[..., 1] = torch.linspace(16.0, 368.0, 17, device="cuda")
    scores = torch.linspace(0.2, 1.1, 17, device="cuda").repeat(batch_size, 1)
    valid = torch.ones(batch_size, 17, dtype=torch.bool, device="cuda")
    return images, {"keypoints": keypoints, "scores": scores, "valid": valid}


def capture(model, images, pose):
    model.eval()
    with torch.no_grad():
        descriptor, _ = model(images, pose_batch=pose, tapf_epoch=None)
        _, _, aux = model.base(images, pose_batch=pose, tapf_epoch=None)
    tensors = {
        "descriptor": descriptor.detach().cpu(),
        "field": aux["student_field"].detach().cpu(),
        "gates": [item.detach().cpu() for item in aux["gate_deltas"]],
    }
    if not torch.isfinite(tensors["descriptor"]).all():
        raise RuntimeError("Non-finite descriptor")
    if not torch.isfinite(tensors["field"]).all():
        raise RuntimeError("Non-finite student field")
    if any(not torch.isfinite(item).all() for item in tensors["gates"]):
        raise RuntimeError("Non-finite gate")
    return tensors


def require_capture_equal(actual, expected, label):
    for name in ("descriptor", "field"):
        if not torch.equal(actual[name], expected[name]):
            raise RuntimeError("{} mismatch for {}".format(name, label))
    if len(actual["gates"]) != len(expected["gates"]):
        raise RuntimeError("Gate count mismatch for {}".format(label))
    for index, (left, right) in enumerate(zip(actual["gates"], expected["gates"])):
        if not torch.equal(left, right):
            raise RuntimeError("Gate mismatch for {} bank {}".format(label, index))


def consumer_path_audit(model, images, reference_descriptor):
    records = []
    for index, bank in enumerate(model.base.tapf.psg_bank):
        original_forward = bank.forward

        def bypass(self, tokens, hw_shape, field):
            del self, hw_shape, field
            return tokens, torch.zeros_like(tokens)

        bank.forward = types.MethodType(bypass, bank)
        try:
            with torch.no_grad():
                descriptor, _ = model(images, pose_batch=None, tapf_epoch=None)
        finally:
            bank.forward = original_forward
        if not torch.isfinite(descriptor).all():
            raise RuntimeError("Non-finite bypass descriptor for bank {}".format(index))
        max_abs_delta = float(
            (descriptor.detach().cpu() - reference_descriptor).abs().max().item()
        )
        if max_abs_delta <= 0.0:
            raise RuntimeError("Dead final consumer bank {}".format(index))
        records.append(
            {"consumer": "late{}".format(index), "descriptor_max_abs_delta": max_abs_delta}
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)
    loaders = make_dataloader(cfg)
    _, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = loaders
    if hasattr(train_loader_normal.dataset, "pose_store") or hasattr(
        val_loader.dataset, "pose_store"
    ):
        raise RuntimeError("RGB-only evaluator unexpectedly owns a pose store")

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    floating_count = 0
    for name, tensor in checkpoint.items():
        if tensor.is_floating_point() or tensor.is_complex():
            floating_count += 1
            if not torch.isfinite(tensor).all():
                raise RuntimeError("Non-finite checkpoint tensor: {}".format(name))
    incompatible = model.load_state_dict(checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint load failed")
    groups, per_bank = changed_summary(model, initial)
    independence = bank_independence(model.base.tapf.psg_bank)

    model = model.cuda().eval()
    images, correct_pose = synthetic_inputs()
    reference = capture(model, images, correct_pose)
    shuffled_pose = {name: value.flip(0) for name, value in correct_pose.items()}
    exploding = ExplodingPose()
    for label, pose in (("shuffle", shuffled_pose), ("none", None), ("exploding", exploding)):
        require_capture_equal(capture(model, images, pose), reference, label)
    if exploding.accesses != 0:
        raise RuntimeError("Evaluation accessed exploding pose")
    path_records = consumer_path_audit(model, images, reference["descriptor"])

    result = {
        "status": "EXP390_D0_FINAL_AUDIT_PASS",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_state_tensors": len(checkpoint),
        "checkpoint_floating_tensors": floating_count,
        "strict_missing": list(incompatible.missing_keys),
        "strict_unexpected": list(incompatible.unexpected_keys),
        "parameter_trajectory": groups,
        "psg_per_bank_trajectory": per_bank,
        "psg_pairwise_independence": independence,
        "pose_variants_exact": ["correct", "shuffle", "none", "exploding"],
        "exploding_pose_accesses": exploding.accesses,
        "descriptor_shape": list(reference["descriptor"].shape),
        "field_shape": list(reference["field"].shape),
        "gate_shapes": [list(item.shape) for item in reference["gates"]],
        "consumer_paths": path_records,
        "dataset_boundary": {
            "num_query": num_query,
            "normal_train_dataset": train_loader_normal.dataset.__class__.__name__,
            "validation_dataset": val_loader.dataset.__class__.__name__,
            "normal_train_has_pose_store": hasattr(train_loader_normal.dataset, "pose_store"),
            "validation_has_pose_store": hasattr(val_loader.dataset, "pose_store"),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
