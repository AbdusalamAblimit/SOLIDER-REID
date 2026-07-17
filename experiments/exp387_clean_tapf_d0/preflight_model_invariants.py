"""Audit matched construction, state, RNG, and optimizer membership."""

import hashlib
import random

import numpy as np
import torch
import torch.nn as nn

from config import cfg as default_cfg
from model import make_model
from solver import make_optimizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_cfg(tapf):
    config = default_cfg.clone()
    config.merge_from_file(
        "configs/occluded_duke/swin_tiny_tapf_d0.yml"
        if tapf
        else "configs/occluded_duke/swin_tiny.yml"
    )
    config.defrost()
    config.MODEL.PRETRAIN_CHOICE = "self"
    config.MODEL.PRETRAIN_PATH = (
        "/home/afr/reid-clean/weights/solider_swin_tiny_tea.pth"
    )
    config.freeze()
    return config


def tensor_sha(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_description(config, model):
    dummy_center = nn.Linear(1, 1)
    optimizer, _ = make_optimizer(config, model, dummy_center)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        (
            names[id(group["params"][0])],
            group["lr"],
            group["weight_decay"],
            group.get("momentum", 0.0),
        )
        for group in optimizer.param_groups
    ]


def main():
    b0_cfg = make_cfg(False)
    d0_cfg = make_cfg(True)

    set_seed(1234)
    b0 = make_model(b0_cfg, 702, 8, 1, b0_cfg.MODEL.SEMANTIC_WEIGHT)
    b0_cpu_rng = torch.get_rng_state().clone()
    b0_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

    set_seed(1234)
    d0 = make_model(d0_cfg, 702, 8, 1, d0_cfg.MODEL.SEMANTIC_WEIGHT)
    d0_cpu_rng = torch.get_rng_state().clone()
    d0_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

    b0_state = b0.state_dict()
    d0_common = {
        name: tensor
        for name, tensor in d0.state_dict().items()
        if not name.startswith("base.tapf.")
    }
    if list(b0_state) != list(d0_common):
        raise RuntimeError("Common state key/order mismatch")
    for name in b0_state:
        if not torch.equal(b0_state[name], d0_common[name]):
            raise RuntimeError("Common state mismatch: {}".format(name))
    if not torch.equal(b0_cpu_rng, d0_cpu_rng):
        raise RuntimeError("Post-construction CPU RNG mismatch")
    if len(b0_cuda_rng) != len(d0_cuda_rng) or any(
        not torch.equal(left, right) for left, right in zip(b0_cuda_rng, d0_cuda_rng)
    ):
        raise RuntimeError("Post-construction CUDA RNG mismatch")

    b0_optimizer = optimizer_description(b0_cfg, b0)
    d0_optimizer = [
        item
        for item in optimizer_description(d0_cfg, d0)
        if not item[0].startswith("base.tapf.")
    ]
    if b0_optimizer != d0_optimizer:
        raise RuntimeError("Common optimizer order/hyperparameter mismatch")

    tapf_names = [
        name for name, _ in d0.named_parameters() if name.startswith("base.tapf.")
    ]
    d0_optimizer_all = optimizer_description(d0_cfg, d0)
    optimized_names = [item[0] for item in d0_optimizer_all]
    if len(optimized_names) != len(set(optimized_names)):
        raise RuntimeError("Duplicate optimizer parameter")
    if not set(tapf_names).issubset(optimized_names):
        raise RuntimeError("TAPF parameter missing from optimizer")

    tapf = d0.base.tapf
    if len(tapf.psg_bank) != 2:
        raise RuntimeError("Expected two PSG banks")
    if any(
        torch.count_nonzero(gate.output_projection.weight).item() != 0
        for gate in tapf.psg_bank
    ):
        raise RuntimeError("PSG final projection is not zero initialized")
    bank0 = {id(parameter) for parameter in tapf.psg_bank[0].parameters()}
    bank1 = {id(parameter) for parameter in tapf.psg_bank[1].parameters()}
    if not bank0.isdisjoint(bank1):
        raise RuntimeError("PSG banks share parameters")

    b0_parameters = sum(parameter.numel() for parameter in b0.parameters())
    d0_parameters = sum(parameter.numel() for parameter in d0.parameters())
    tapf_parameters = sum(parameter.numel() for parameter in tapf.parameters())
    if d0_parameters - b0_parameters != tapf_parameters:
        raise RuntimeError("TAPF parameter accounting mismatch")

    print("EXP387_MODEL_INVARIANTS_PASS")
    print("common_state_count={}".format(len(b0_state)))
    print("common_state_sha256={}".format(tensor_sha(b0_state.items())))
    print("common_optimizer_parameters={}".format(len(b0_optimizer)))
    print("tapf_optimizer_parameters={}".format(len(tapf_names)))
    print("b0_parameters={}".format(b0_parameters))
    print("d0_parameters={}".format(d0_parameters))
    print("tapf_parameters={}".format(tapf_parameters))
    print("parameter_overhead_percent={:.6f}".format(100.0 * tapf_parameters / b0_parameters))


if __name__ == "__main__":
    main()
