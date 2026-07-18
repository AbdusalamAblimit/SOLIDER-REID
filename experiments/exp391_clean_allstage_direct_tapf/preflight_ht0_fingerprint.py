"""Emit a deterministic legacy HT0 training fingerprint for old/new parity."""

import argparse
import hashlib
import json
import random

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


def update_tensor(digest, name, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())


def tensor_hash(named_tensors):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in named_tensors:
        update_tensor(digest, name, tensor)
        count += 1
    return digest.hexdigest(), count


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()
    if not cfg.MODEL.TAPF.ENABLED or not cfg.MODEL.TAPF.HIERARCHICAL:
        raise RuntimeError("HT0 parity requires hierarchical TAPF")

    set_seed(cfg.SOLVER.SEED)
    model = make_model(
        cfg,
        num_class=702,
        camera_num=8,
        view_num=1,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).cuda().train()
    loss_fn, center_criterion = make_loss(cfg, num_classes=702)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    batch = synthetic_batch()
    scaler = amp.GradScaler()

    initial_state, state_count = tensor_hash(model.state_dict().items())
    step_records = []
    for step in range(args.steps):
        torch.manual_seed(9100 + step)
        torch.cuda.manual_seed_all(9100 + step)
        optimizer.zero_grad(set_to_none=True)
        images, labels, cameras, views, pose = batch
        with amp.autocast(enabled=True):
            score, feature, featmaps, aux = model(
                images,
                label=labels,
                cam_label=cameras,
                view_label=views,
                pose_batch=pose,
                tapf_epoch=11,
            )
            loss = loss_fn(score, feature, labels, cameras)
            loss = loss + cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * aux["pose_loss"]
        named_outputs = [
            ("score", score),
            ("feature", feature),
            ("pose_loss", aux["pose_loss"]),
            ("early_pose_loss", aux["early_pose_loss"]),
            ("late_pose_loss", aux["late_pose_loss"]),
            ("early_student_field", aux["early_student_field"]),
            ("late_student_field", aux["late_student_field"]),
        ]
        named_outputs.extend(
            ("featmap{}".format(index), value)
            for index, value in enumerate(featmaps)
        )
        named_outputs.extend(
            ("early_gate{}".format(index), value)
            for index, value in enumerate(aux["early_gate_deltas"])
        )
        named_outputs.extend(
            ("late_gate{}".format(index), value)
            for index, value in enumerate(aux["late_gate_deltas"])
        )
        output_sha, output_count = tensor_hash(named_outputs)
        scaler.scale(loss).backward()
        gradient_sha, gradient_count = tensor_hash(
            (name, parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        )
        scaler.step(optimizer)
        scaler.update()
        state_sha, _ = tensor_hash(model.state_dict().items())
        step_records.append(
            {
                "gradient_count": gradient_count,
                "gradient_sha256": gradient_sha,
                "loss": float(loss.detach()),
                "output_count": output_count,
                "output_sha256": output_sha,
                "scale": float(scaler.get_scale()),
                "state_sha256": state_sha,
            }
        )

    final_state, _ = tensor_hash(model.state_dict().items())
    momentum_sha, momentum_count = tensor_hash(
        (parameter_names[id(parameter)] + ".momentum", state["momentum_buffer"])
        for parameter, state in optimizer.state.items()
        if "momentum_buffer" in state
    )
    result = {
        "final_state_sha256": final_state,
        "initial_state_sha256": initial_state,
        "momentum_count": momentum_count,
        "momentum_sha256": momentum_sha,
        "state_count": state_count,
        "steps": step_records,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    with open(args.output, "w") as handle:
        handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
