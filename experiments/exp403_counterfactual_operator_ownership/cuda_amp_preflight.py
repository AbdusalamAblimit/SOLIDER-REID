#!/usr/bin/env python3
"""Single necessary actual-batch64 CUDA/AMP preflight for exp403."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
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
from processor.processor import _load_elo_generic_evidence
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


def finite_tensor(value):
    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-attempts", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("Preflight output must be fresh")

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
    generic, generic_sha = _load_elo_generic_evidence(
        cfg.MODEL.TAPF.ELO_GENERIC_EVIDENCE,
        cfg.MODEL.TAPF.ELO_GENERIC_EVIDENCE_SHA256,
        cfg.DATASETS.NAMES,
        cfg.MODEL.TAPF.CLIP_CHECKPOINT_SHA256,
        cfg.MODEL.TAPF.RICH_CODEBOOK_SHA256,
        cfg.MODEL.TAPF.MANIFEST_SHA256,
        device,
    )
    pose_batch["identity"] = identities
    pose_batch["camera"] = cameras
    pose_batch["generic_evidence"] = generic

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
    production_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".tapf.psg_bank." in name
    }
    state_keys = list(model.state_dict())
    attempts = []
    success = None
    for attempt in range(1, args.max_attempts + 1):
        optimizer.zero_grad()
        before = {
            name: parameter.detach().clone()
            for name, parameter in production_parameters.items()
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
        tapf_aux["student_evidence"].retain_grad()
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {
            name: 0.0 if parameter.grad is None else float(parameter.grad.norm())
            for name, parameter in production_parameters.items()
        }
        gradient_finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in production_parameters.values()
        )
        student_grad = tapf_aux["student_evidence"].grad
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        updated = {
            name: not torch.equal(before[name], parameter.detach())
            for name, parameter in production_parameters.items()
        }
        actual_update = all(updated.values())
        record = {
            "attempt": attempt,
            "loss": float(loss.detach()),
            "scale_before": scale_before,
            "scale_after": scale_after,
            "all_production_grad_finite_nonzero": gradient_finite
            and all(value > 0 for value in gradients.values()),
            "all_production_updated": actual_update,
            "reference_rng_exact": bool(tapf_aux["reference_rng_exact"]),
        }
        attempts.append(record)
        if actual_update:
            success = {
                "attempt": attempt,
                "loss": float(loss.detach()),
                "reid_loss": float(reid_loss.detach()),
                "pose_loss": float(tapf_aux["pose_loss"].detach()),
                "compatibility_loss": float(
                    tapf_aux["compatibility_loss"].detach()
                ),
                "cur_loss": float(tapf_aux["cur_loss"].detach()),
                "rho": float(tapf_aux["rho"]),
                "budget_abs": float(
                    torch.stack(
                        [value.detach().float().abs().mean()
                         for value in tapf_aux["gate_deltas"]]
                    ).mean()
                ),
                "eligible_ratio": float(
                    tapf_aux["donor_eligible"].float().mean()
                ),
                "student_evidence_grad_norm": (
                    0.0 if student_grad is None else float(student_grad.norm())
                ),
                "production_grad_norm": gradients,
                "reference_rng_exact": bool(tapf_aux["reference_rng_exact"]),
                "reference_no_grad": all(
                    not value.requires_grad
                    for value in tapf_aux["reference_descriptors"].values()
                ),
                "all_tapf_values_finite": all(
                    finite_tensor(tapf_aux[key])
                    for key in (
                        "pose_loss",
                        "semantic_loss",
                        "evidence_cos_loss",
                        "evidence_relation_loss",
                        "compatibility_loss",
                        "cur_loss",
                    )
                ),
            }
            break
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

    donor = tapf_aux["donor_index"] if success else torch.empty(0, device=device)
    eligible = donor >= 0
    donor_contract = bool(eligible.any()) and bool(
        torch.all(cameras[donor[eligible]] == cameras[eligible])
        and torch.all(identities[donor[eligible]] != identities[eligible])
    )
    gates = {
        "actual_batch64": images.shape[0] == 64,
        "teacher_valid_nonzero": bool(targets["valid"].any()),
        "generic_sha_exact": generic_sha
        == cfg.MODEL.TAPF.ELO_GENERIC_EVIDENCE_SHA256,
        "successful_amp_update": bool(success),
        "all_production_grad_finite_nonzero": bool(success)
        and attempts[-1]["all_production_grad_finite_nonzero"],
        "all_production_updated": bool(success)
        and attempts[-1]["all_production_updated"],
        "correct_evidence_grad_nonzero": bool(success)
        and success["student_evidence_grad_norm"] > 0,
        "reference_no_grad": bool(success) and success["reference_no_grad"],
        "reference_rng_exact": bool(success) and success["reference_rng_exact"],
        "tapf_values_finite": bool(success) and success["all_tapf_values_finite"],
        "donor_contract": donor_contract,
        "no_slot_expert_state": not any("experts" in key for key in state_keys),
        "teacher_generic_free_state": not any(
            marker in key.lower()
            for key in state_keys
            for marker in ("teacher", "generic", "codebook", "clip")
        ),
        "rgb_only_eval_finite": bool(torch.isfinite(eval_none).all()),
        "none_exploding_pose_exact": torch.equal(eval_none, eval_exploding),
        "no_checkpoint": not any(Path(cfg.OUTPUT_DIR).glob("*.pth")),
    }
    passed = all(gates.values())
    result = {
        "experiment": "exp403_counterfactual_operator_ownership",
        "status": "CUDA_AMP_PREFLIGHT_PASS" if passed else "CUDA_AMP_PREFLIGHT_FAIL",
        "formal_training_authorized": passed,
        "gate_count": len(gates),
        "gate_pass_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "attempts": attempts,
        "success": success,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "model_state_key_count": len(state_keys),
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(__file__),
    }
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
