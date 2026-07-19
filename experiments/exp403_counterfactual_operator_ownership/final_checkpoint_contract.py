#!/usr/bin/env python3
"""Freeze the naturally completed exp403 checkpoint before GPU evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import torch


SOURCE_FILES = (
    "model/tapf.py",
    "model/make_model.py",
    "model/backbones/swin_transformer.py",
    "datasets/bases.py",
    "datasets/occluded_duke.py",
    "utils/metrics.py",
    "config/defaults.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor state entry: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_text(command, cwd=None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--audit-wrapper", required=True)
    parser.add_argument("--base-audit", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--base-core", required=True)
    parser.add_argument("--postflight", required=True)
    parser.add_argument("--execution-wrapper", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    config = Path(args.config).resolve()
    output = Path(args.output).resolve()
    runner = Path(args.runner).resolve()
    contract_path = Path(args.contract).resolve()
    checkpoint = output / "transformer_120.pth"
    if contract_path.exists() or contract_path.with_suffix(".json.tmp").exists():
        raise FileExistsError("Checkpoint contract path must be fresh")
    checkpoint_files = sorted(
        path.resolve()
        for path in output.glob("*")
        if path.is_file() and path.suffix in {".pth", ".pt", ".ckpt"}
    )
    compute = [
        line for line in run_text([
            "nvidia-smi", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]).splitlines() if line.strip()
    ]
    main_pid_path = config.parent / "formal_main.pid"
    main_pid = int(main_pid_path.read_text().strip())
    train_processes = [str(main_pid)] if Path(f"/proc/{main_pid}").exists() else []
    state = torch.load(str(checkpoint), map_location="cpu")
    names = tuple(state)
    runner_text = runner.read_text(encoding="utf-8", errors="replace")
    source_sha = {name: sha256_file(repo / name) for name in SOURCE_FILES}
    required_router_suffixes = (
        "down_projection.weight", "context_projection.weight",
        "evidence_projection.weight", "up_projection.weight",
        "context_query.weight", "evidence_key.weight",
    )
    gates = {
        "repo_tracked_clean": not run_text(
            ["git", "status", "--porcelain", "--untracked-files=no"], repo
        ),
        "repo_all_clean": not run_text(
            ["git", "status", "--porcelain"], repo
        ),
        "train_processes_zero": not train_processes,
        "gpu_compute_processes_zero": not compute,
        "epoch120_done": "Epoch 120 done." in runner_text,
        "final_eval_present": runner_text.count("mAP:") >= 24,
        "checkpoint_unique": checkpoint_files == [checkpoint.resolve()],
        "checkpoint_regular": checkpoint.is_file() and not checkpoint.is_symlink(),
        "state_all_tensors": all(torch.is_tensor(value) for value in state.values()),
        "state_finite": all(
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in state.values()
        ),
        "state_teacher_free": all(
            not ({"teacher", "clip", "codebook", "text", "pose_batch"}
                 & set(name.lower().split(".")))
            for name in names
        ),
        "evidence_head_retained": any("anchor.evidence_head" in name for name in names),
        "two_elo_routers_retained": all(
            all(any(f"psg_bank.{bank}.{suffix}" in name for name in names)
                for suffix in required_router_suffixes)
            for bank in (0, 1)
        ),
        "static_experts_absent": not any(".experts." in name for name in names),
        "runner_fatal_zero": not re.search(
            r"(?i)traceback|runtimeerror|out of memory|cuda error|non[-_ ]?finite",
            runner_text,
        ),
    }
    paths = {
        "audit_wrapper": Path(args.audit_wrapper).resolve(),
        "base_audit": Path(args.base_audit).resolve(),
        "core": Path(args.core).resolve(),
        "base_core": Path(args.base_core).resolve(),
        "postflight": Path(args.postflight).resolve(),
        "execution_wrapper": Path(args.execution_wrapper).resolve(),
    }
    payload = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "repo": {"path": str(repo), "head": run_text(["git", "rev-parse", "HEAD"], repo)},
        "checkpoint": {
            "path": str(checkpoint),
            "state_count": len(state),
            "state_sha256": tensor_state_sha256(state),
        },
        "sha256": {
            "checkpoint": sha256_file(checkpoint),
            "config": sha256_file(config),
            "runner": sha256_file(runner),
            "checkpoint_contract_script": sha256_file(Path(__file__).resolve()),
            "source": source_sha,
            **{name: sha256_file(path) for name, path in paths.items()},
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "gpu_compute_pids": compute,
        "train_processes": train_processes,
    }
    atomic_json(contract_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
