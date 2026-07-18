"""Audit H2-M gradient ownership and exact optimizer-step overflow skip."""

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
    torch.backends.cudnn.benchmark = False


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    scores = torch.linspace(0.2, 1.1, 17, device="cuda").repeat(batch_size, 1)
    valid = torch.ones(batch_size, 17, dtype=torch.bool, device="cuda")
    pose = {"keypoints": keypoints, "scores": scores, "valid": valid}
    return images, labels, cameras, views, pose


def make_components():
    model = make_model(
        cfg,
        num_class=702,
        camera_num=8,
        view_num=1,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).cuda().train()
    loss_fn, center_criterion = make_loss(cfg, num_classes=702)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    return model, loss_fn, optimizer


def forward_losses(model, loss_fn, batch, epoch):
    images, labels, cameras, views, pose = batch
    score, feature, _, aux = model(
        images,
        label=labels,
        cam_label=cameras,
        view_label=views,
        pose_batch=pose,
        tapf_epoch=epoch,
    )
    reid_loss = loss_fn(score, feature, labels, cameras)
    return reid_loss, aux["pose_loss"], aux


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


def nonzero_gradient_groups(model):
    result = {
        name: {"present": 0, "nonzero": 0, "nonfinite": 0}
        for name in (
            "early_anchor",
            "late_anchor",
            "early_psg",
            "late_psg",
            "swin",
            "head",
        )
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = result[parameter_group(name)]
        group["present"] += 1
        group["nonzero"] += int(torch.count_nonzero(parameter.grad).item() > 0)
        group["nonfinite"] += int(not torch.isfinite(parameter.grad).all())
    return result


def gradient_ownership(batch):
    set_seed(cfg.SOLVER.SEED)
    pose_model, pose_loss_fn, _ = make_components()
    pose_model.zero_grad(set_to_none=True)
    _, pose_loss, _ = forward_losses(pose_model, pose_loss_fn, batch, epoch=11)
    pose_loss.backward()
    pose_groups = nonzero_gradient_groups(pose_model)
    if pose_groups["early_anchor"]["nonzero"] != 8:
        raise RuntimeError("Pose loss does not own all early-anchor tensors")
    if pose_groups["late_anchor"]["nonzero"] != 8:
        raise RuntimeError("Pose loss does not own all late-anchor tensors")
    for group in ("early_psg", "late_psg", "swin", "head"):
        if pose_groups[group]["present"] != 0:
            raise RuntimeError("Pose loss leaked into {}".format(group))

    set_seed(cfg.SOLVER.SEED)
    reid_model, reid_loss_fn, _ = make_components()
    reid_model.zero_grad(set_to_none=True)
    reid_loss, _, _ = forward_losses(reid_model, reid_loss_fn, batch, epoch=11)
    reid_loss.backward()
    reid_groups = nonzero_gradient_groups(reid_model)
    for group in ("early_anchor", "late_anchor"):
        if reid_groups[group]["present"] != 0:
            raise RuntimeError("ReID loss leaked into {}".format(group))
    for group in ("early_psg", "late_psg", "swin", "head"):
        if reid_groups[group]["nonzero"] == 0:
            raise RuntimeError("ReID loss missed {}".format(group))
    return {"pose_loss": pose_groups, "reid_loss": reid_groups}


def require_nested_equal(left, right, label):
    if type(left) is not type(right):
        raise RuntimeError("Type mismatch for {}".format(label))
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            raise RuntimeError("Tensor mismatch for {}".format(label))
    elif isinstance(left, dict):
        if list(left) != list(right):
            raise RuntimeError("Key mismatch for {}".format(label))
        for key in left:
            require_nested_equal(left[key], right[key], "{}.{}".format(label, key))
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError("Length mismatch for {}".format(label))
        for index, (a, b) in enumerate(zip(left, right)):
            require_nested_equal(a, b, "{}.{}".format(label, index))
    elif left != right:
        raise RuntimeError("Value mismatch for {}".format(label))


def optimizer_tensor_count(optimizer):
    return sum(
        int(torch.is_tensor(value))
        for state in optimizer.state.values()
        for value in state.values()
    )


def exact_overflow_skip(batch):
    set_seed(cfg.SOLVER.SEED)
    model, loss_fn, optimizer = make_components()
    optimizer.zero_grad(set_to_none=True)
    reid_loss, pose_loss, _ = forward_losses(model, loss_fn, batch, epoch=11)
    finite_loss = reid_loss + cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * pose_loss
    finite_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if len(optimizer.state) != 205 or optimizer_tensor_count(optimizer) != 205:
        raise RuntimeError(
            "Unexpected populated optimizer state: {}/{}".format(
                len(optimizer.state), optimizer_tensor_count(optimizer)
            )
        )

    scaler = amp.GradScaler()
    optimizer.zero_grad(set_to_none=True)
    with amp.autocast(enabled=True):
        reid_loss, pose_loss, aux = forward_losses(model, loss_fn, batch, epoch=11)
        total_loss = reid_loss + cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * pose_loss
    if not torch.isfinite(total_loss) or not torch.isfinite(aux["pose_loss"]):
        raise RuntimeError("Overflow audit forward is non-finite")
    scaler.scale(total_loss).backward()
    injected_name = None
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            parameter.grad.view(-1)[0] = float("inf")
            injected_name = name
            break
    if injected_name is None:
        raise RuntimeError("No gradient available for overflow injection")
    scale_before = float(scaler.get_scale())
    scaler.unscale_(optimizer)
    found_inf = float(
        sum(
            value.detach().float().item()
            for value in scaler._per_optimizer_states[id(optimizer)][
                "found_inf_per_device"
            ].values()
        )
    )
    if found_inf <= 0.0:
        raise RuntimeError("Injected overflow was not detected")
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    require_nested_equal(model_before, model.state_dict(), "model_state")
    require_nested_equal(optimizer_before, optimizer.state_dict(), "optimizer_state")
    if scale_after >= scale_before:
        raise RuntimeError("Overflow did not reduce GradScaler scale")
    return {
        "model_state_tensors": len(model_before),
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_state_tensors": optimizer_tensor_count(optimizer),
        "injected_parameter": injected_name,
        "found_inf": found_inf,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "model_exact_skip": True,
        "optimizer_exact_skip": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    if not cfg.MODEL.TAPF.HIERARCHICAL:
        raise RuntimeError("H2-M overflow audit requires hierarchical TAPF")
    if cfg.MODEL.TAPF.POSE_LOSS_REDUCTION != "mean":
        raise RuntimeError("H2-M overflow audit requires mean reduction")

    batch = synthetic_batch()
    ownership = gradient_ownership(batch)
    overflow = exact_overflow_skip(batch)
    result = {
        "status": "EXP391_H2M_OVERFLOW_PASS",
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(__file__),
        "gradient_ownership": ownership,
        "overflow": overflow,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
