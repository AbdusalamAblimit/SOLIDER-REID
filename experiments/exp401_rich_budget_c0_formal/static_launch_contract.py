#!/usr/bin/env python3
"""CPU-only exact launch contract for exp401 formal e120."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path

import torch
import yaml


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
PREFLIGHT_RESULT_SHA256 = (
    "3935eb6df97ae832770316eff27cbfc757e4d2bd305b789d0b9b97835659a02f"
)
CLIP_SHA256 = "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
CODEBOOK_SHA256 = "fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a"
EXPECTED_SOURCE_SHA256 = {
    "model/tapf.py": "95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886",
    "model/clip_semantic_teacher.py": "c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5",
    "model/make_model.py": "6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03",
    "processor/processor.py": "be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1",
    "config/defaults.py": "a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5",
    "model/backbones/swin_transformer.py": "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "datasets/pose_dataset.py": "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc",
    "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml":
        "e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_text(command, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_path(value, path, replacement):
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def get_path(value, path):
    current = value
    for key in path:
        current = current[key]
    return current


def run_contract(repo_root: Path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("exp401 static contract requires hidden CUDA")
    cuda_before = torch.cuda.is_initialized()
    baseline_path = (
        repo_root
        / "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml"
    )
    candidate_path = (
        repo_root
        / "experiments/exp401_rich_budget_c0_formal/"
        "swin_tiny_tapf_rich_budget_c0_formal.yml"
    )
    preflight_path = (
        repo_root
        / "experiments/exp400_final_production_preflight/"
        "exp400_cuda_final.result.json"
    )
    baseline = load_yaml(baseline_path)
    candidate = load_yaml(candidate_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    normalized = copy.deepcopy(candidate)
    allowed_paths = (
        ("MODEL", "TAPF", "CLIP_CHECKPOINT"),
        ("MODEL", "TAPF", "RICH_CODEBOOK"),
        ("OUTPUT_DIR",),
    )
    changed_paths = []
    for path in allowed_paths:
        if get_path(candidate, path) != get_path(baseline, path):
            changed_paths.append(".".join(path))
        set_path(normalized, path, get_path(baseline, path))

    source_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    tapf = candidate["MODEL"]["TAPF"]
    solver = candidate["SOLVER"]
    gates = {
        "source_commit_exists": run_text(
            ["git", "cat-file", "-t", SOURCE_COMMIT], repo_root
        ) == "commit",
        "source_sha_exact": source_sha == EXPECTED_SOURCE_SHA256,
        "preflight_result_sha_exact": sha256_file(preflight_path)
        == PREFLIGHT_RESULT_SHA256,
        "preflight_formal_authorized": (
            preflight["status"] == "PASS"
            and preflight["diagnostic_outcome"]
            == "FINAL_PRODUCTION_PREFLIGHT_PASS"
            and preflight["formal_training_authorized"] is True
        ),
        "only_fresh_paths_and_output_changed": normalized == baseline
        and changed_paths
        == [
            "MODEL.TAPF.CLIP_CHECKPOINT",
            "MODEL.TAPF.RICH_CODEBOOK",
            "OUTPUT_DIR",
        ],
        "source_backbone_fixed": candidate["MODEL"]["TRANSFORMER_TYPE"]
        == "swin_tiny_patch4_window7_224",
        "rich_route_exact": (
            tapf["ENABLED"] is True
            and tapf["SEMANTIC_ENABLED"] is True
            and tapf["SEMANTIC_REZERO"] is False
            and tapf["RICH_EVIDENCE_ENABLED"] is True
            and tapf["HIERARCHICAL"] is False
        ),
        "rho_schedule_exact": (
            float(tapf["RESIDUAL_RHO"]) == 0.08075544983148575
            and int(tapf["TEACHER_EPOCHS"]) == 5
            and int(tapf["HANDOFF_EPOCHS"]) == 5
        ),
        "formal_recipe_exact": (
            int(solver["MAX_EPOCHS"]) == 120
            and int(solver["IMS_PER_BATCH"]) == 64
            and int(solver["SEED"]) == 1234
            and int(solver["CHECKPOINT_PERIOD"]) == 120
            and int(solver["EVAL_PERIOD"]) == 10
            and int(candidate["DATALOADER"]["NUM_WORKERS"]) == 8
        ),
        "optimizer_schedule_exact": (
            solver["OPTIMIZER_NAME"] == "SGD"
            and float(solver["BASE_LR"]) == 0.0008
            and int(solver["WARMUP_EPOCHS"]) == 20
            and solver["WARMUP_METHOD"] == "cosine"
            and float(solver["WEIGHT_DECAY"]) == 1e-4
            and float(solver["WEIGHT_DECAY_BIAS"]) == 1e-4
        ),
        "loss_weight_exact": float(tapf["POSE_LOSS_WEIGHT"]) == 0.1,
        "fresh_clip_contract": (
            "exp401" in tapf["CLIP_CHECKPOINT"]
            and tapf["CLIP_CHECKPOINT"].startswith("/home/afr/")
            and tapf["CLIP_CHECKPOINT_SHA256"] == CLIP_SHA256
        ),
        "fresh_codebook_contract": (
            "exp401" in tapf["RICH_CODEBOOK"]
            and tapf["RICH_CODEBOOK"].startswith("/home/afr/")
            and tapf["RICH_CODEBOOK_SHA256"] == CODEBOOK_SHA256
        ),
        "official_data_read_only_paths": (
            "/mnt1/afrdata" in candidate["DATASETS"]["ROOT_DIR"]
            and tapf["ARTIFACT_DIR"].startswith("/mnt1/afrderived/")
        ),
        "fresh_output_contract": (
            candidate["OUTPUT_DIR"].startswith("/home/afr/")
            and "exp401" in candidate["OUTPUT_DIR"]
        ),
        "no_resume_or_test_weight": candidate["TEST"]["WEIGHT"] == "",
        "final_only_checkpoint": int(solver["CHECKPOINT_PERIOD"])
        == int(solver["MAX_EPOCHS"]),
    }
    cuda_after = torch.cuda.is_initialized()
    gates["cuda_never_initialized"] = not cuda_before and not cuda_after
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "git_head": run_text(["git", "rev-parse", "HEAD"], repo_root),
        "candidate_config_sha256": sha256_file(candidate_path),
        "baseline_config_sha256": sha256_file(baseline_path),
        "preflight_result_sha256": sha256_file(preflight_path),
        "static_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_sha256": source_sha,
        "changed_paths": changed_paths,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "checkpoint_count": 0,
        "gates": gates,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    runner = Path(args.runner).resolve()
    if output.exists() or runner.exists():
        raise RuntimeError("Refusing to overwrite exp401 static assets")
    try:
        result = run_contract(repo_root)
    except Exception as error:
        result = {
            "status": "INVALID",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "checkpoint_count": 0,
        }
    write_json(output, result)
    write_json(runner, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
