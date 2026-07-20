#!/usr/bin/env python3
"""Static config/source gate before the single exp404 CUDA preflight."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg as defaults


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_contract(config_path: Path, preflight_path: Path):
    cuda_before = torch.cuda.is_initialized()
    candidate = defaults.clone()
    candidate.merge_from_file(str(config_path))
    candidate.freeze()
    source = preflight_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(preflight_path))
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    tapf = candidate.MODEL.TAPF
    frozen_recipe = {
        "backbone_swin_tiny": candidate.MODEL.TRANSFORMER_TYPE
        == "swin_tiny_patch4_window7_224",
        "dataset_occluded_duke": str(candidate.DATASETS.NAMES)
        == "occluded_duke",
        "batch64": candidate.SOLVER.IMS_PER_BATCH == 64,
        "seed1234": candidate.SOLVER.SEED == 1234,
        "epoch120": candidate.SOLVER.MAX_EPOCHS == 120,
        "workers8": candidate.DATALOADER.NUM_WORKERS == 8,
        "lr_exact": candidate.SOLVER.BASE_LR == 0.0008,
        "pose_weight_exact": tapf.POSE_LOSS_WEIGHT == 0.1,
        "handoff_exact": tapf.TEACHER_EPOCHS == 5
        and tapf.HANDOFF_EPOCHS == 5,
        "checkpoint_only_e120": candidate.SOLVER.CHECKPOINT_PERIOD == 120,
    }
    mechanism = {
        "tapf_rich_semantic_enabled": tapf.ENABLED
        and tapf.SEMANTIC_ENABLED
        and tapf.RICH_EVIDENCE_ENABLED,
        "spk_enabled": tapf.SPK_ENABLED,
        "spk_groups16": tapf.SPK_GROUPS == 16,
        "elo_disabled": not tapf.ELO_CUR_ENABLED,
        "no_generic_asset": not tapf.ELO_GENERIC_EVIDENCE
        and not tapf.ELO_GENERIC_EVIDENCE_SHA256,
        "official_data_path": str(candidate.DATASETS.ROOT_DIR).startswith(
            "/mnt1/afrdata"
        ),
        "frozen_pose_path": str(tapf.ARTIFACT_DIR).startswith(
            "/mnt1/afrderived"
        ),
        "fresh_asset_paths": all(
            str(value).startswith("/home/afr/reid-clean/formal/exp404_spk/")
            for value in (
                candidate.MODEL.PRETRAIN_PATH,
                tapf.CLIP_CHECKPOINT,
                tapf.RICH_CODEBOOK,
            )
        ),
        "fresh_output_path": str(candidate.OUTPUT_DIR).startswith(
            "/home/afr/SOLIDER-REID-exp404-spk-formal-v1/"
        ),
    }
    source_gates = {
        "source_ast_valid": isinstance(tree, ast.Module),
        "actual_batch_guard": "Actual preflight batch is not 64"
        in string_literals,
        "exclusive_gpu_guard": "CUDA preflight requires an idle exclusive GPU"
        in string_literals,
        "rtx4090_guard": "Exclusive RTX 4090 required" in string_literals,
        "fresh_output_guard": "Preflight output must be fresh" in string_literals,
        "default_gradscaler": "scaler = amp.GradScaler()" in source,
        "spk_capture_hook": "register_forward_pre_hook" in source,
        "all_16_group_gradients": "feature_16_group_grad_finite_nonzero"
        in source
        and "factor_16_group_grad_finite_nonzero" in source,
        "null_and_random_interventions": "null_descriptor_exact_raw" in source
        and "random_factor_active" in source,
        "rgb_only_eval_contract": "none_exploding_pose_exact" in source,
        "no_performance_early_stop": "mAP" not in source
        and "rank1" not in source.lower(),
        "no_resume_or_checkpoint_load": "resume" not in source.lower()
        and "load_state_dict" not in source,
        "no_elo_generic_loader": "_load_elo_generic_evidence" not in source,
        "attempt_bound4": 'default=4' in source,
    }
    gates = {**frozen_recipe, **mechanism, **source_gates}
    passed = all(bool(value) for value in gates.values())
    return {
        "experiment": "exp404_semantic_product_kernel",
        "status": "CUDA_PREFLIGHT_STATIC_PASS" if passed else "CUDA_PREFLIGHT_STATIC_FAIL",
        "cuda_execution_authorized": passed,
        "formal_training_authorized": False,
        "gate_count": len(gates),
        "gate_pass_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "config_sha256": sha256_file(config_path),
        "preflight_sha256": sha256_file(preflight_path),
        "contract_sha256": sha256_file(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Static output must be fresh")
    result = run_contract(args.config.resolve(), args.preflight.resolve())
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "CUDA_PREFLIGHT_STATIC_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
