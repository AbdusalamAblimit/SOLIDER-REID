#!/usr/bin/env python3
"""CPU-only AST contract for the unexecuted exp395 CUDA attribution script."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
EXPECTED_SOURCE_SHA256 = {
    "model/tapf.py": "95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886",
    "model/clip_semantic_teacher.py": "c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5",
    "model/make_model.py": "6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03",
    "processor/processor.py": "be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1",
    "config/defaults.py": "a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5",
    "model/backbones/swin_transformer.py": "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "datasets/pose_dataset.py": "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc",
    "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml": "e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e",
    "configs/occluded_duke/swin_tiny_tapf_d0.yml": "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b",
}
EXPECTED_GROUPS = (
    "backbone",
    "anchor_trunk",
    "pose_head",
    "mask_head",
    "presence_head",
    "evidence_head",
    "id_head",
    "router0_token_projection",
    "router0_context_projection",
    "router0_evidence_projection",
    "router0_experts",
    "router1_token_projection",
    "router1_context_projection",
    "router1_evidence_projection",
    "router1_experts",
)
EXPECTED_BASELINE_LOSSES = (
    "reid",
    "heatmap",
    "confidence",
    "pose",
    "total",
)
EXPECTED_RICH_LOSSES = (
    "reid",
    "heatmap",
    "confidence",
    "mask",
    "presence",
    "evidence_cosine",
    "evidence_relation",
    "exec_consumer0",
    "exec_consumer1",
    "pose",
    "total",
)


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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_text(command, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def literal_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise KeyError(name)


def call_paths(tree):
    paths = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        parts = []
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        paths.append(".".join(reversed(parts)))
    return paths


def function_source(source, tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise KeyError(name)


def run_contract(repo_root: Path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("Static contract requires hidden CUDA")
    cuda_before = torch.cuda.is_initialized()
    script_path = (
        repo_root
        / "experiments/exp395_amp_gradient_attribution/cuda_amp_attribution.py"
    )
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = call_paths(tree)
    row_source = function_source(source, tree, "run_loss_row")
    arm_source = function_source(source, tree, "run_arm")
    main_source = function_source(source, tree, "main")
    source_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    groups = literal_assignment(tree, "GROUP_NAMES")
    baseline_losses = literal_assignment(tree, "BASELINE_LOSSES")
    rich_losses = literal_assignment(tree, "RICH_LOSSES")
    initial_scale = literal_assignment(tree, "EXPECTED_INITIAL_SCALE")
    scaled_capture = row_source.index("scaled = gradient_report")
    unscale = row_source.index("scaler.unscale_(optimizer)")
    unscaled_capture = row_source.index("unscaled = gradient_report")
    gates = {
        "source_commit_exists": run_text(
            ["git", "cat-file", "-t", SOURCE_COMMIT], repo_root
        )
        == "commit",
        "production_source_sha_exact": source_sha == EXPECTED_SOURCE_SHA256,
        "group_count_and_order_exact": groups == EXPECTED_GROUPS,
        "baseline_loss_count_and_order_exact": baseline_losses
        == EXPECTED_BASELINE_LOSSES,
        "rich_loss_count_and_order_exact": rich_losses == EXPECTED_RICH_LOSSES,
        "scaled_capture_before_unscale": scaled_capture < unscale,
        "unscaled_capture_after_unscale": unscale < unscaled_capture,
        "fresh_scaler_per_row": 'torch.amp.GradScaler("cuda")' in row_source,
        "default_scale_not_overridden": initial_scale == 65536.0
        and "init_scale" not in source,
        "scaled_unscaled_consistency_recorded": (
            "scaled_unscaled_consistency(" in row_source
            and '"scaled_unscaled_counts_consistent"' in source
        ),
        "no_optimizer_step_call": not any(
            path.endswith("optimizer.step") or path == "optimizer.step"
            for path in calls
        ),
        "no_scaler_step_call": "scaler.step" not in calls,
        "no_scaler_update_call": "scaler.update" not in calls,
        "no_scheduler_step_call": not any(
            path.endswith("scheduler.step") for path in calls
        ),
        "no_retain_graph": "retain_graph" not in source,
        "no_checkpoint_load": "torch.load" not in calls,
        "only_in_memory_torch_save": calls.count("torch.save") == 1
        and "torch.save(optimizer.state_dict(), buffer)" in source,
        "individual_exec_losses_exposed": (
            '"exec_consumer0": exec_losses[0]' in source
            and '"exec_consumer1": exec_losses[1]' in source
        ),
        "d0_psg_mapping_explicit": (
            'return f"router{index}_experts"' in source
            and 'elif arm == "d0":' in source
        ),
        "d0_and_rich_arms_present": (
            'run_arm(\n        "d0"' in source
            and 'run_arm(\n        "rich"' in source
        ),
        "parameter_coverage_precedes_backward": (
            arm_source.index('if not coverage["exact"]:')
            < arm_source.index("run_loss_row(")
        ),
        "fresh_asset_names_enforced": (
            '"exp395" in clip_path.name' in source
            and '"exp395" in codebook_path.name' in source
            and "not clip_path.is_symlink()" in source
            and "not codebook_path.is_symlink()" in source
        ),
        "canonical_runtime_versions_enforced": all(
            value in source
            for value in (
                'torch.__version__ == "2.6.0+cu124"',
                'open_clip.__version__ == "3.3.0"',
                'cv2.__version__ == "4.13.0"',
                'timm.__version__ == "1.0.27"',
            )
        ),
        "zero_update_counters_frozen": (
            '"optimizer_updates": 0' in source
            and '"scaler_step_calls": 0' in source
            and '"scaler_update_calls": 0' in source
        ),
        "formal_training_never_authorized": (
            main_source.count('"formal_training_authorized": False') >= 1
            and '"formal_training_authorized": True' not in source
        ),
        "output_assets_are_result_runner_manifest_only": (
            "write_json(manifest_path, manifest)" in source
            and "write_json(output, result)" in source
            and "write_json(runner, result)" in source
        ),
        "post_exit_gpu_audit_required": '"post_exit_gpu_audit_required": True'
        in source,
        "exp394_sealed_boundary_retained": '"exp394_remains_sealed": True'
        in source,
    }
    cuda_after = torch.cuda.is_initialized()
    gates["cuda_never_initialized"] = not cuda_before and not cuda_after
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "git_head": run_text(["git", "rev-parse", "HEAD"], repo_root),
        "cuda_script_sha256": sha256_file(script_path),
        "static_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_sha256": source_sha,
        "group_names": list(groups),
        "baseline_losses": list(baseline_losses),
        "rich_losses": list(rich_losses),
        "call_count": len(calls),
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "optimizer_updates": 0,
        "checkpoint_count": 0,
        "cuda_execution_authorized": False,
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
        raise RuntimeError("Refusing to overwrite static result/runner")
    result = run_contract(repo_root)
    write_json(output, result)
    write_json(runner, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
