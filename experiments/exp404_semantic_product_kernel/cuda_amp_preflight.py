#!/usr/bin/env python3
"""Single necessary actual-batch64 CUDA/AMP preflight for exp404 SPK."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.cuda import amp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from model.clip_semantic_teacher import FrozenRichClipEvidenceTeacher
from model.tapf import (
    EvidenceBudgetRouter,
    EvidenceOwnedLowRankRouter,
    PoseSpatialGate,
)
from solver import make_optimizer


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def finite_nonzero(tensor):
    return (
        isinstance(tensor, torch.Tensor)
        and bool(torch.isfinite(tensor).all())
        and float(tensor.detach().float().norm()) > 0
    )


def compute_pids():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return sorted(
        int(line.strip())
        for line in output.splitlines()
        if line.strip().isdigit()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("Preflight output must be fresh")
    if not str(output).startswith("/home/afr/"):
        raise RuntimeError("Preflight output must remain under /home/afr")

    external_pids_before = compute_pids()
    if external_pids_before:
        raise RuntimeError("CUDA preflight requires an idle exclusive GPU")

    cfg.merge_from_file(args.config)
    cfg.freeze()
    seed = int(cfg.SOLVER.SEED)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if "4090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Exclusive RTX 4090 required")
    torch.cuda.reset_peak_memory_stats(device)

    train_loader, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    batch = next(iter(train_loader))
    images, identities, cameras, views, pose_batch = batch
    images = images.to(device)
    identities = identities.to(device)
    cameras = cameras.to(device)
    views = views.to(device)
    pose_batch = {
        "keypoints": pose_batch["keypoints"].to(device),
        "scores": pose_batch["scores"].to(device),
        "valid": pose_batch["valid"].to(device),
        "teacher_rgb": pose_batch["teacher_rgb"].to(device),
    }
    if images.shape[0] != 64:
        raise RuntimeError("Actual preflight batch is not 64")

    teacher = FrozenRichClipEvidenceTeacher(
        checkpoint=cfg.MODEL.TAPF.CLIP_CHECKPOINT,
        checkpoint_sha256=cfg.MODEL.TAPF.CLIP_CHECKPOINT_SHA256,
        codebook=cfg.MODEL.TAPF.RICH_CODEBOOK,
        codebook_sha256=cfg.MODEL.TAPF.RICH_CODEBOOK_SHA256,
        device=device,
        microbatch=cfg.MODEL.TAPF.CLIP_MICROBATCH,
    )
    with torch.no_grad(), amp.autocast(enabled=True):
        targets = teacher(
            pose_batch["teacher_rgb"],
            pose_batch["keypoints"],
            pose_batch["scores"],
            pose_batch["valid"],
        )
    pose_batch["semantic_valid"] = targets["valid"].detach().clone()
    pose_batch["semantic_teacher_mask"] = targets["region_masks"].detach().clone()
    pose_batch["semantic_teacher_evidence"] = targets[
        "evidence_code"
    ].detach().clone()
    del teacher
    torch.cuda.empty_cache()

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    ).to(device).train()
    loss_fn, center = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center)
    scaler = amp.GradScaler()
    tapf = model.base.tapf
    state_keys = list(model.state_dict())
    evidence_head_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".tapf.anchor.evidence_head." in name
    }
    capture = {}

    def capture_spk_input(module, inputs):
        del module
        capture["raw_global"] = inputs[0]
        capture["evidence"] = inputs[1]
        capture["presence"] = inputs[2]

    hook = model.semantic_product_kernel.register_forward_pre_hook(
        capture_spk_input
    )
    attempts = []
    success = None
    for attempt in range(1, args.max_attempts + 1):
        optimizer.zero_grad()
        before = {
            name: parameter.detach().clone()
            for name, parameter in evidence_head_parameters.items()
        }
        with amp.autocast(enabled=True):
            score, feature, _, tapf_aux = model(
                images,
                label=identities,
                cam_label=cameras,
                view_label=views,
                pose_batch=pose_batch,
                tapf_epoch=6,
            )
            reid_loss = loss_fn(score, feature, identities, cameras)
            loss = reid_loss + cfg.MODEL.TAPF.POSE_LOSS_WEIGHT * tapf_aux[
                "pose_loss"
            ]
        feature.retain_grad()
        tapf_aux["student_evidence"].retain_grad()
        tapf_aux["semantic_product_factor"].retain_grad()
        raw_global = capture["raw_global"]
        raw_evidence = capture["evidence"]
        raw_presence = capture["presence"]
        correct_descriptor = feature.detach().clone()
        with torch.no_grad():
            null_descriptor, null_factor = model.semantic_product_kernel(
                raw_global.detach(),
                torch.zeros_like(raw_evidence),
                raw_presence.detach(),
            )
            sign = torch.where(
                torch.arange(16, device=device) % 2 == 0,
                torch.ones(16, device=device),
                -torch.ones(16, device=device),
            )
            random_key = raw_evidence.detach().roll(shifts=1, dims=-1) * sign
            random_descriptor, random_factor = model.semantic_product_kernel(
                raw_global.detach(), random_key, raw_presence.detach()
            )
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        evidence_gradient = tapf_aux["student_evidence"].grad
        feature_gradient = feature.grad
        factor_gradient = tapf_aux["semantic_product_factor"].grad
        evidence_head_gradients = {
            name: 0.0 if parameter.grad is None else float(parameter.grad.norm())
            for name, parameter in evidence_head_parameters.items()
        }
        group_feature_gradients = feature_gradient.detach().float().reshape(
            64, 16, 48
        ).norm(dim=(0, 2))
        group_factor_gradients = factor_gradient.detach().float().norm(dim=0)
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        updated = {
            name: not torch.equal(before[name], parameter.detach())
            for name, parameter in evidence_head_parameters.items()
        }
        record = {
            "attempt": attempt,
            "loss": float(loss.detach()),
            "scale_before": scale_before,
            "scale_after": scale_after,
            "all_evidence_head_updated": bool(updated) and all(updated.values()),
            "evidence_grad_finite_nonzero": finite_nonzero(evidence_gradient),
            "feature_16_group_grad_finite_nonzero": bool(
                torch.isfinite(group_feature_gradients).all()
                and torch.all(group_feature_gradients > 0)
            ),
            "factor_16_group_grad_finite_nonzero": bool(
                torch.isfinite(group_factor_gradients).all()
                and torch.all(group_factor_gradients > 0)
            ),
        }
        attempts.append(record)
        if all(
            record[key]
            for key in (
                "all_evidence_head_updated",
                "evidence_grad_finite_nonzero",
                "feature_16_group_grad_finite_nonzero",
                "factor_16_group_grad_finite_nonzero",
            )
        ):
            success = {
                "attempt": attempt,
                "loss": float(loss.detach()),
                "reid_loss": float(reid_loss.detach()),
                "pose_loss": float(tapf_aux["pose_loss"].detach()),
                "student_evidence_grad_norm": float(evidence_gradient.norm()),
                "bound_feature_grad_norm": float(feature_gradient.norm()),
                "factor_grad_norm": float(factor_gradient.norm()),
                "evidence_head_grad_norm": evidence_head_gradients,
                "feature_group_grad_norm": group_feature_gradients.tolist(),
                "factor_group_grad_norm": group_factor_gradients.tolist(),
                "factor_mean": float(
                    tapf_aux["semantic_product_factor"].detach().mean()
                ),
                "factor_std": float(
                    tapf_aux["semantic_product_factor"].detach().std(
                        unbiased=False
                    )
                ),
                "factor_min": float(
                    tapf_aux["semantic_product_factor"].detach().min()
                ),
                "factor_max": float(
                    tapf_aux["semantic_product_factor"].detach().max()
                ),
                "correct_vs_null_abs": float(
                    (correct_descriptor - null_descriptor).abs().mean()
                ),
                "correct_vs_random_abs": float(
                    (correct_descriptor - random_descriptor).abs().mean()
                ),
                "null_factor_exact_one": torch.equal(
                    null_factor, torch.ones_like(null_factor)
                ),
                "null_descriptor_exact_raw": torch.equal(
                    null_descriptor, raw_global.detach()
                ),
                "random_factor_active": not torch.equal(
                    random_factor,
                    tapf_aux["semantic_product_factor"].detach(),
                ),
                "rho_exact_zero": float(tapf_aux["rho"]) == 0.0,
            }
            break
    hook.remove()
    if success is None:
        success = {}

    model.eval()
    with torch.no_grad(), amp.autocast(enabled=True):
        eval_none, _ = model(
            images[:2], cam_label=cameras[:2], view_label=views[:2]
        )
        exploding = {
            "keypoints": torch.full((2, 17, 2), float("nan"), device=device)
        }
        eval_exploding, _ = model(
            images[:2],
            cam_label=cameras[:2],
            view_label=views[:2],
            pose_batch=exploding,
        )

    current_pid = os.getpid()
    compute_pids_after = compute_pids()
    gates = {
        "exclusive_gpu_before": external_pids_before == [],
        "exclusive_gpu_after": compute_pids_after == [current_pid],
        "actual_rtx4090": "4090" in torch.cuda.get_device_name(device),
        "actual_batch64": images.shape[0] == 64,
        "teacher_valid_nonzero": bool(targets["valid"].any()),
        "successful_amp_update": bool(success),
        "all_evidence_head_updated": bool(success)
        and attempts[-1]["all_evidence_head_updated"],
        "student_evidence_grad_finite_nonzero": bool(success)
        and attempts[-1]["evidence_grad_finite_nonzero"],
        "feature_16_group_grad_finite_nonzero": bool(success)
        and attempts[-1]["feature_16_group_grad_finite_nonzero"],
        "factor_16_group_grad_finite_nonzero": bool(success)
        and attempts[-1]["factor_16_group_grad_finite_nonzero"],
        "factor_finite_positive_active": bool(success)
        and success["factor_min"] > 0
        and success["factor_std"] > 0
        and math.isclose(success["factor_mean"], 1.0, abs_tol=1e-6),
        "correct_null_intervention_active": bool(success)
        and success["correct_vs_null_abs"] > 0,
        "correct_random_intervention_active": bool(success)
        and success["correct_vs_random_abs"] > 0
        and success["random_factor_active"],
        "null_factor_exact_one": bool(success)
        and success["null_factor_exact_one"],
        "null_bypass_exact_raw": bool(success)
        and success["null_descriptor_exact_raw"],
        "rho_exact_zero": bool(success) and success["rho_exact_zero"],
        "spk_zero_parameters": not any(
            name.startswith("semantic_product_kernel.")
            for name, _ in model.named_parameters()
        ),
        "two_d0_spatial_gates": len(tapf.psg_bank) == 2
        and all(isinstance(gate, PoseSpatialGate) for gate in tapf.psg_bank),
        "no_c0_elo_router_modules": not any(
            isinstance(module, (EvidenceBudgetRouter, EvidenceOwnedLowRankRouter))
            for module in tapf.modules()
        ),
        "no_c0_elo_router_state": not any(
            marker in key
            for key in state_keys
            for marker in ("experts", "evidence_projection", "context_query")
        ),
        "teacher_free_state": not any(
            marker in key.lower()
            for key in state_keys
            for marker in ("teacher", "generic", "codebook", "clip")
        ),
        "rgb_only_eval_finite": bool(torch.isfinite(eval_none).all()),
        "none_exploding_pose_exact": torch.equal(eval_none, eval_exploding),
        "no_checkpoint": not any(Path(cfg.OUTPUT_DIR).glob("*.pth")),
        "official_data_read_only_path": str(cfg.DATASETS.ROOT_DIR).startswith(
            "/mnt1/afrdata"
        ),
        "frozen_pose_read_only_path": str(
            cfg.MODEL.TAPF.ARTIFACT_DIR
        ).startswith("/mnt1/afrderived"),
    }
    passed = all(gates.values())
    result = {
        "experiment": "exp404_semantic_product_kernel",
        "status": "CUDA_AMP_PREFLIGHT_PASS" if passed else "CUDA_AMP_PREFLIGHT_FAIL",
        "formal_training_authorized": passed,
        "gate_count": len(gates),
        "gate_pass_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "attempts": attempts,
        "success": success,
        "device": torch.cuda.get_device_name(device),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "model_state_key_count": len(state_keys),
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(__file__),
    }
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
