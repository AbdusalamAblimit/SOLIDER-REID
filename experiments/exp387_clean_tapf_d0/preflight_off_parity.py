"""Emit deterministic fingerprints for legacy-vs-config-off parity."""

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


def update_tensor(digest, name, tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())


def hash_named_tensors(named_tensors):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in named_tensors:
        update_tensor(digest, name, tensor)
        count += 1
    return digest.hexdigest(), count


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg.merge_from_file("configs/occluded_duke/swin_tiny.yml")
    cfg.MODEL.PRETRAIN_CHOICE = "self"
    cfg.MODEL.PRETRAIN_PATH = "/home/afr/reid-clean/weights/solider_swin_tiny_tea.pth"
    cfg.freeze()

    set_seed(1234)
    model = make_model(
        cfg,
        num_class=702,
        camera_num=8,
        view_num=1,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    initial_hash, state_count = hash_named_tensors(model.state_dict().items())
    state_keys = list(model.state_dict().keys())
    cpu_rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    cuda_rng = hashlib.sha256(
        b"".join(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    ).hexdigest()

    loss_fn, center_criterion = make_loss(cfg, num_classes=702)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_groups = [
        {
            "names": [parameter_names[id(parameter)] for parameter in group["params"]],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "momentum": group.get("momentum", 0.0),
        }
        for group in optimizer.param_groups
    ]

    model.cuda().train()
    generator = torch.Generator().manual_seed(20260717)
    images = torch.randn(4, 3, 384, 128, generator=generator).cuda()
    labels = torch.tensor([0, 0, 1, 1], device="cuda")
    cameras = torch.tensor([0, 1, 0, 1], device="cuda")
    views = torch.ones(4, dtype=torch.long, device="cuda")
    scaler = amp.GradScaler()
    losses = []
    output_hashes = []
    gradient_hashes = []

    for step in range(10):
        torch.manual_seed(9000 + step)
        torch.cuda.manual_seed_all(9000 + step)
        optimizer.zero_grad()
        with amp.autocast(enabled=True):
            score, feature, featmaps = model(
                images,
                label=labels,
                cam_label=cameras,
                view_label=views,
            )
            loss = loss_fn(score, feature, labels, cameras)
        losses.append(float(loss.detach()))
        output_hash, _ = hash_named_tensors(
            [("score", score), ("feature", feature)]
            + [("featmap{}".format(index), value) for index, value in enumerate(featmaps)]
        )
        output_hashes.append(output_hash)
        scaler.scale(loss).backward()
        gradient_hash, gradient_count = hash_named_tensors(
            (name, parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        )
        gradient_hashes.append(gradient_hash)
        scaler.step(optimizer)
        scaler.update()

    final_hash, _ = hash_named_tensors(model.state_dict().items())
    momentum_hash, momentum_count = hash_named_tensors(
        (parameter_names[id(parameter)] + ".momentum", state["momentum_buffer"])
        for parameter, state in optimizer.state.items()
        if "momentum_buffer" in state
    )
    result = {
        "state_count": state_count,
        "state_keys": state_keys,
        "initial_state_sha256": initial_hash,
        "post_construction_cpu_rng_sha256": cpu_rng,
        "post_construction_cuda_rng_sha256": cuda_rng,
        "optimizer_groups": optimizer_groups,
        "losses": losses,
        "output_sha256": output_hashes,
        "gradient_sha256": gradient_hashes,
        "gradient_count": gradient_count,
        "final_state_sha256": final_hash,
        "momentum_sha256": momentum_hash,
        "momentum_count": momentum_count,
        "grad_scaler": scaler.state_dict(),
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({key: value for key, value in result.items() if key not in ("state_keys", "optimizer_groups")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
