#!/usr/bin/env python3
"""Frozen full-validation attribution for the sealed exp392 C0 checkpoint."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
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
    from utils.metrics import R1_mAP_eval

    cfg = default_cfg.clone()
    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    if not cfg.MODEL.TAPF.ENABLED or not cfg.MODEL.TAPF.SEMANTIC_ENABLED:
        raise RuntimeError("Phase 0D requires Semantic TAPF C0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_seed(cfg.SOLVER.SEED)
    (
        _,
        _,
        val_loader,
        num_query,
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
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_state, strict=True)
    state_finite = all(
        (not value.is_floating_point()) or bool(torch.isfinite(value).all())
        for value in model.state_dict().values()
    )
    device = torch.device("cuda", 0)
    model = model.to(device).eval()
    tapf = model.base.tapf
    original_prepare = tapf.prepare
    base_state_sha = state_sha256(model)

    control = {
        "arm": "correct_start",
        "static_q": None,
        "collect_q": True,
    }
    q_sum = torch.zeros(5, dtype=torch.float64)
    q_square_sum = torch.zeros(5, dtype=torch.float64)
    q_count = 0

    def patched_prepare(self, source_feature, pose_batch, image_hw, epoch, training):
        nonlocal q_sum, q_square_sum, q_count
        if training:
            raise AssertionError("Frozen attribution entered a training path")
        state = original_prepare(
            source_feature,
            pose_batch=pose_batch,
            image_hw=image_hw,
            epoch=epoch,
            training=training,
        )
        support = state["consumer_support"]
        mask = state["consumer_mask"]
        if control["collect_q"]:
            values = support.detach().double().cpu()
            q_sum += values.sum(dim=0)
            q_square_sum += values.square().sum(dim=0)
            q_count += values.shape[0]

        arm = control["arm"]
        if arm == "static_slot_q":
            static_q = control["static_q"].to(
                device=support.device, dtype=support.dtype
            )
            support = static_q[None].expand_as(support)
        elif arm == "q_one":
            support = torch.ones_like(support)
        elif arm == "spatial_constant_mask":
            mask = mask.mean(dim=(-2, -1), keepdim=True).expand_as(mask)
        elif arm == "slot_cycle":
            mask = torch.roll(mask, shifts=1, dims=1)
            support = torch.roll(support, shifts=1, dims=1)
        elif arm not in {
            "correct_start",
            "correct_end",
            "expert_mean",
            "router_0_bypass",
            "router_1_bypass",
            "all_router_bypass",
        }:
            raise RuntimeError("Unknown attribution arm: " + arm)

        state["consumer_mask"] = mask.detach()
        state["consumer_support"] = support.detach()
        state["consumer_field"] = (
            state["consumer_mask"] * state["consumer_support"][..., None, None]
        ).detach()
        return state

    tapf.prepare = types.MethodType(patched_prepare, tapf)

    def evaluate(arm):
        evaluator = R1_mAP_eval(
            num_query,
            max_rank=50,
            feat_norm=cfg.TEST.FEAT_NORM,
            reranking=cfg.TEST.RE_RANKING,
        )
        evaluator.reset()
        descriptors = []
        started = time.perf_counter()
        with torch.no_grad():
            for image, pid, camid, camids, target_view, _ in val_loader:
                image = image.to(device)
                camids = camids.to(device)
                target_view = target_view.to(device)
                feature, _ = model(
                    image,
                    cam_label=camids,
                    view_label=target_view,
                    pose_batch=None,
                )
                if not bool(torch.isfinite(feature).all()):
                    raise RuntimeError("Non-finite descriptor in arm " + arm)
                descriptors.append(feature.detach().cpu())
                evaluator.update((feature, pid, camid))
        cmc, m_ap, _, _, _, _, _ = evaluator.compute()
        descriptor = torch.cat(descriptors, dim=0)
        result = {
            "mAP": float(m_ap * 100.0),
            "rank1": float(cmc[0] * 100.0),
            "rank5": float(cmc[4] * 100.0),
            "rank10": float(cmc[9] * 100.0),
            "descriptor_count": int(descriptor.shape[0]),
            "descriptor_finite": bool(torch.isfinite(descriptor).all()),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        print("ARM", arm, json.dumps(result, sort_keys=True), flush=True)
        return result, descriptor

    arm_order = [
        "correct_start",
        "static_slot_q",
        "q_one",
        "spatial_constant_mask",
        "slot_cycle",
        "expert_mean",
        "router_0_bypass",
        "router_1_bypass",
        "all_router_bypass",
        "correct_end",
    ]
    arm_results = {}
    arm_state_sha = {}
    correct_start_descriptor = None
    correct_end_descriptor = None
    original_experts = [
        router.expert.detach().clone() for router in tapf.psg_bank
    ]

    for arm in arm_order:
        control["arm"] = arm
        control["collect_q"] = arm == "correct_start"
        if arm == "static_slot_q":
            if q_count <= 0:
                raise RuntimeError("Correct pass did not collect q")
            control["static_q"] = (q_sum / q_count).float()

        if arm == "expert_mean":
            for router, expert in zip(tapf.psg_bank, original_experts):
                with torch.no_grad():
                    router.expert.copy_(
                        expert.mean(dim=0, keepdim=True).expand_as(expert)
                    )
        elif arm == "router_0_bypass":
            with torch.no_grad():
                tapf.psg_bank[0].expert.zero_()
        elif arm == "router_1_bypass":
            with torch.no_grad():
                tapf.psg_bank[1].expert.zero_()
        elif arm == "all_router_bypass":
            with torch.no_grad():
                for router in tapf.psg_bank:
                    router.expert.zero_()

        result, descriptor = evaluate(arm)
        arm_results[arm] = result
        if arm == "correct_start":
            correct_start_descriptor = descriptor
        elif arm == "correct_end":
            correct_end_descriptor = descriptor

        if arm in {
            "expert_mean",
            "router_0_bypass",
            "router_1_bypass",
            "all_router_bypass",
        }:
            with torch.no_grad():
                for router, expert in zip(tapf.psg_bank, original_experts):
                    router.expert.copy_(expert)
        arm_state_sha[arm] = state_sha256(model)
        if arm_state_sha[arm] != base_state_sha:
            raise RuntimeError("Model state was not restored after arm " + arm)

    if correct_start_descriptor is None or correct_end_descriptor is None:
        raise RuntimeError("Missing correct start/end descriptors")
    correct_descriptor_exact = bool(
        torch.equal(correct_start_descriptor, correct_end_descriptor)
    )
    correct_metric_exact = all(
        arm_results["correct_start"][key] == arm_results["correct_end"][key]
        for key in ("mAP", "rank1", "rank5", "rank10")
    )
    q_mean = q_sum / q_count
    q_variance = q_square_sum / q_count - q_mean.square()
    correct = arm_results["correct_start"]
    for arm, result in arm_results.items():
        result["delta_vs_correct"] = {
            key: float(result[key] - correct[key])
            for key in ("mAP", "rank1", "rank5", "rank10")
        }

    final_state_sha = state_sha256(model)
    gates = {
        "strict_state_finite": state_finite,
        "correct_descriptor_exact": correct_descriptor_exact,
        "correct_metric_exact": correct_metric_exact,
        "state_sha_exact": final_state_sha == base_state_sha
        and all(value == base_state_sha for value in arm_state_sha.values()),
        "all_descriptors_finite": all(
            item["descriptor_finite"] for item in arm_results.values()
        ),
        "all_arms_complete": set(arm_results) == set(arm_order),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "arm_order": arm_order,
        "arms": arm_results,
        "q_control": {
            "count": q_count,
            "slot_mean": [float(value) for value in q_mean.tolist()],
            "slot_std": [
                float(value) for value in q_variance.clamp_min(0).sqrt().tolist()
            ],
        },
        "base_state_sha256": base_state_sha,
        "final_state_sha256": final_state_sha,
        "arm_state_sha256": arm_state_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "execution_head": os.popen("git rev-parse HEAD").read().strip(),
        "torch_version": torch.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
