#!/usr/bin/env python3
"""Read-only final checkpoint audit for exp392 Semantic TAPF C0."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


class ExplodingPose:
    def __getitem__(self, key):
        raise AssertionError("RGB-only evaluation touched a training-only target")

    def __iter__(self):
        raise AssertionError("RGB-only evaluation iterated a training-only target")

    def keys(self):
        raise AssertionError("RGB-only evaluation inspected a training-only target")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def group_trajectory(initial, final, prefix):
    keys = sorted(
        key
        for key, value in final.items()
        if key.startswith(prefix) and value.is_floating_point()
    )
    if not keys:
        raise RuntimeError("No floating tensors for trajectory prefix: " + prefix)
    squared = 0.0
    initial_squared = 0.0
    maximum = 0.0
    changed = []
    for key in keys:
        before = initial[key].detach().float().cpu()
        after = final[key].detach().float().cpu()
        difference = after - before
        squared += float(difference.square().sum().item())
        initial_squared += float(before.square().sum().item())
        maximum = max(maximum, float(difference.abs().max().item()))
        if not torch.equal(before, after):
            changed.append(key)
    return {
        "prefix": prefix,
        "tensor_count": len(keys),
        "changed_tensor_count": len(changed),
        "all_tensors_changed": len(changed) == len(keys),
        "l2_delta": float(squared ** 0.5),
        "relative_l2_delta": float(
            (squared ** 0.5) / max(initial_squared ** 0.5, 1e-12)
        ),
        "max_abs_delta": maximum,
        "changed_keys": changed,
    }


def tensor_difference(reference, candidate):
    difference = candidate.detach().float() - reference.detach().float()
    return {
        "exact_equal": bool(torch.equal(reference, candidate)),
        "l2": float(difference.norm().item()),
        "max_abs": float(difference.abs().max().item()),
    }


def router_null_contract(router, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    tokens = torch.randn(
        2,
        24,
        router.feature_channels,
        device=device,
        generator=generator,
    )
    mask = torch.rand(
        2,
        router.region_count,
        12,
        4,
        device=device,
        generator=generator,
    )
    support = torch.rand(
        2,
        router.region_count,
        device=device,
        generator=generator,
    )
    zero_mask_output, zero_mask_delta = router(
        tokens, (6, 4), torch.zeros_like(mask), support
    )
    zero_q_output, zero_q_delta = router(
        tokens, (6, 4), mask, torch.zeros_like(support)
    )
    return {
        "zero_mask_output_exact_identity": bool(
            torch.equal(zero_mask_output, tokens)
        ),
        "zero_mask_delta_exact_zero": bool(
            torch.equal(zero_mask_delta, torch.zeros_like(zero_mask_delta))
        ),
        "zero_q_output_exact_identity": bool(torch.equal(zero_q_output, tokens)),
        "zero_q_delta_exact_zero": bool(
            torch.equal(zero_q_delta, torch.zeros_like(zero_q_delta))
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    os.chdir(str(repo_root))
    sys.path.insert(0, str(repo_root))

    from config import cfg as default_cfg
    from datasets import make_dataloader
    from model import make_model

    cfg = default_cfg.clone()
    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    if not cfg.MODEL.TAPF.ENABLED or not cfg.MODEL.TAPF.SEMANTIC_ENABLED:
        raise RuntimeError("Final audit requires the semantic TAPF config")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the exact final runtime audit")

    set_seed(cfg.SOLVER.SEED)
    (
        train_loader,
        _,
        val_loader,
        _,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    initial_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint_state, dict):
        raise RuntimeError("Checkpoint is not a state dictionary")
    incompatible = model.load_state_dict(checkpoint_state, strict=True)
    strict_load = not incompatible.missing_keys and not incompatible.unexpected_keys
    final_state = model.state_dict()
    state_finite = all(
        (not value.is_floating_point()) or bool(torch.isfinite(value).all())
        for value in final_state.values()
    )
    forbidden_terms = (
        "clip",
        "teacher",
        "prototype",
        "open_clip",
        "text_encoder",
        "text_prototype",
        "prompt_bank",
    )
    forbidden_keys = sorted(
        key
        for key in final_state
        if any(term in key.lower() for term in forbidden_terms)
    )

    trajectories = {
        "anchor": group_trajectory(
            initial_state, final_state, "base.tapf.anchor."
        ),
        "q_support_head": group_trajectory(
            initial_state, final_state, "base.tapf.anchor.support_head."
        ),
        "consumer_0": group_trajectory(
            initial_state, final_state, "base.tapf.psg_bank.0."
        ),
        "consumer_1": group_trajectory(
            initial_state, final_state, "base.tapf.psg_bank.1."
        ),
        "consumer_0_expert": group_trajectory(
            initial_state, final_state, "base.tapf.psg_bank.0.expert"
        ),
        "consumer_1_expert": group_trajectory(
            initial_state, final_state, "base.tapf.psg_bank.1.expert"
        ),
    }

    device = torch.device("cuda", 0)
    model = model.to(device).eval()
    validation_batch = next(iter(val_loader))
    images = validation_batch[0][: args.sample_count].to(device)
    camera_labels = validation_batch[3][: args.sample_count].to(device)
    view_labels = validation_batch[4][: args.sample_count].to(device)
    if images.shape[0] < 2:
        raise RuntimeError("Audit requires at least two validation images")

    pose_correct = {
        "keypoints": torch.zeros(images.shape[0], 17, 2, device=device),
        "scores": torch.ones(images.shape[0], 17, device=device),
        "valid": torch.ones(
            images.shape[0], 17, dtype=torch.bool, device=device
        ),
    }
    pose_shuffle = {
        key: value.flip(0) for key, value in pose_correct.items()
    }

    def descriptor(pose_batch):
        with torch.no_grad():
            feature, _ = model(
                images,
                cam_label=camera_labels,
                view_label=view_labels,
                pose_batch=pose_batch,
            )
        return feature.detach()

    rgb_none = descriptor(None)
    rgb_correct = descriptor(pose_correct)
    rgb_shuffle = descriptor(pose_shuffle)
    rgb_exploding = descriptor(ExplodingPose())
    rgb_counterfactuals = {
        "correct_vs_none": tensor_difference(rgb_none, rgb_correct),
        "shuffle_vs_none": tensor_difference(rgb_none, rgb_shuffle),
        "exploding_vs_none": tensor_difference(rgb_none, rgb_exploding),
    }

    with torch.no_grad():
        base_feature, _, tapf_state = model.base(
            images, pose_batch=ExplodingPose(), tapf_epoch=None
        )
    base_matches_descriptor = bool(torch.equal(base_feature, rgb_none))
    student_support = tapf_state["student_support"].detach().float()
    student_mask = tapf_state["student_mask"].detach().float()
    student_presence = tapf_state["student_presence"].detach().float()
    gate_deltas = [delta.detach().float() for delta in tapf_state["gate_deltas"]]
    state_summary = {
        "support_mean": float(student_support.mean().item()),
        "support_std": float(student_support.std(unbiased=False).item()),
        "support_min": float(student_support.min().item()),
        "support_max": float(student_support.max().item()),
        "mask_mean": float(student_mask.mean().item()),
        "mask_std": float(student_mask.std(unbiased=False).item()),
        "presence_mean": float(student_presence.mean().item()),
        "presence_nonzero_fraction": float((student_presence > 0).float().mean().item()),
        "gate_delta_abs_mean": [
            float(delta.abs().mean().item()) for delta in gate_deltas
        ],
        "gate_delta_max_abs": [
            float(delta.abs().max().item()) for delta in gate_deltas
        ],
        "descriptor_matrix_rank": int(
            torch.linalg.matrix_rank(rgb_none.detach().float()).item()
        ),
    }

    reachability = []
    for index, router in enumerate(model.base.tapf.psg_bank):
        saved_expert = router.expert.detach().clone()
        with torch.no_grad():
            router.expert.zero_()
        ablated = descriptor(None)
        with torch.no_grad():
            router.expert.copy_(saved_expert)
        comparison = tensor_difference(rgb_none, ablated)
        comparison["consumer_index"] = index
        comparison["reaches_final_descriptor"] = not comparison["exact_equal"]
        reachability.append(comparison)

    null_contracts = [
        router_null_contract(router, device, cfg.SOLVER.SEED + index)
        for index, router in enumerate(model.base.tapf.psg_bank)
    ]

    gates = {
        "strict_checkpoint_load": strict_load,
        "all_floating_state_finite": state_finite,
        "teacher_absent_from_state": not forbidden_keys,
        "anchor_trajectory": trajectories["anchor"]["l2_delta"] > 0.0,
        "q_head_trajectory": trajectories["q_support_head"]["l2_delta"] > 0.0,
        "consumer_0_trajectory": trajectories["consumer_0"]["l2_delta"] > 0.0,
        "consumer_1_trajectory": trajectories["consumer_1"]["l2_delta"] > 0.0,
        "rgb_only_descriptor_finite": bool(torch.isfinite(rgb_none).all()),
        "rgb_pose_counterfactual_exact": all(
            item["exact_equal"] for item in rgb_counterfactuals.values()
        ),
        "base_descriptor_contract": base_matches_descriptor,
        "two_consumers_reach_descriptor": all(
            item["reaches_final_descriptor"] for item in reachability
        ),
        "two_gate_deltas_finite_nonzero": len(gate_deltas) == 2
        and all(bool(torch.isfinite(delta).all()) and bool((delta != 0).any()) for delta in gate_deltas),
        "null_identity_exact": all(
            all(contract.values()) for contract in null_contracts
        ),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "state_tensor_count": len(final_state),
            "forbidden_state_keys": forbidden_keys,
        },
        "config_sha256": sha256_file(config_path),
        "execution_head": os.popen("git rev-parse HEAD").read().strip(),
        "seed": int(cfg.SOLVER.SEED),
        "sample_count": int(images.shape[0]),
        "trajectories": trajectories,
        "rgb_counterfactuals": rgb_counterfactuals,
        "state_summary": state_summary,
        "consumer_reachability": reachability,
        "null_contracts": null_contracts,
        "torch_version": torch.__version__,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
