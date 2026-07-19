#!/usr/bin/env python3
"""CPU-only static contract for exp398 baseline-relative AMP stability."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import traceback
from pathlib import Path

import torch


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
EXPECTED_REPORTER_SHA256 = (
    "6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164"
)
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
    "configs/occluded_duke/swin_tiny_tapf_d0.yml":
        "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b",
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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp398_static_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def function_source(source, tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
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


def synthetic_group_state(module, inactive=()):
    initial = {name: f"{name}:initial" for name in module.RICH_SPECIFIC_GROUPS}
    final = {
        name: initial[name] if name in inactive else f"{name}:final"
        for name in module.RICH_SPECIFIC_GROUPS
    }
    return {"initial": initial, "final": final}


def synthetic_steps(module, skips, rich_nonfinite=None, inactive=()):
    rich_nonfinite = rich_nonfinite or {}
    rows = []
    scale = 65536.0
    for attempt in range(1, 33):
        skipped = attempt in skips or attempt in rich_nonfinite
        before = scale
        after = before * 0.5 if skipped else before
        extra_groups = list(rich_nonfinite.get(attempt, ()))
        nonfinite_groups = (["backbone"] if attempt in skips else []) + extra_groups
        report = {
            name: {
                "all_finite": name not in extra_groups,
                "grad_nonzero_tensors": int(
                    attempt > 16 and not skipped and name not in inactive
                ),
            }
            for name in module.RICH_SPECIFIC_GROUPS
        }
        rows.append(
            {
                "attempt": attempt,
                "tapf_epoch": 1 if attempt <= 16 else 6,
                "scale_before": before,
                "scale_after": after,
                "had_nonfinite": skipped,
                "nonfinite_groups": nonfinite_groups,
                "gradient_report": report,
                "optimizer_succeeded": not skipped,
                "optimizer_skipped": skipped,
                "optimizer_step_calls_delta": 0 if skipped else 1,
            }
        )
        scale = after
    return rows


def run_contract(repo_root: Path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("exp398 static contract requires hidden CUDA")
    cuda_before = torch.cuda.is_initialized()
    script_path = (
        repo_root
        / "experiments/exp398_baseline_relative_amp_stability/cuda_baseline_relative_amp_stability.py"
    )
    reporter_path = (
        repo_root
        / "experiments/exp396_chunk_safe_amp_attribution/cuda_amp_attribution.py"
    )
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = call_paths(tree)
    arm_source = function_source(source, tree, "run_dynamic_arm")
    gate_source = function_source(source, tree, "run_gate")
    main_source = function_source(source, tree, "main")
    module = load_module(script_path)

    source_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    shared_skips = {1, 2, 3, 4, 5, 17}
    base_rows = synthetic_steps(module, shared_skips)
    matched = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(module, shared_skips),
        synthetic_group_state(module),
    )
    extra_skip = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(module, shared_skips | {10}),
        synthetic_group_state(module),
    )
    persistent_rows = synthetic_steps(module, shared_skips | {16, 32})
    persistent_tail = module.evaluate_trajectories(
        persistent_rows,
        synthetic_steps(module, shared_skips | {16, 32}),
        synthetic_group_state(module),
    )
    rich_specific = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(
            module,
            shared_skips,
            rich_nonfinite={18: ("evidence_head",)},
        ),
        synthetic_group_state(module),
    )
    inactive = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(module, shared_skips, inactive=("evidence_head",)),
        synthetic_group_state(module, inactive=("evidence_head",)),
    )
    rich_better = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(module, {1, 2, 3, 4, 17}),
        synthetic_group_state(module),
    )

    gates = {
        "source_commit_exists": run_text(
            ["git", "cat-file", "-t", SOURCE_COMMIT], repo_root
        ) == "commit",
        "source_sha_exact": source_sha == EXPECTED_SOURCE_SHA256,
        "reporter_dependency_sha_exact": sha256_file(reporter_path)
        == EXPECTED_REPORTER_SHA256,
        "attempts_and_schedule_exact": (
            module.ATTEMPTS == 32
            and module.STAGE_LENGTH == 16
            and module.STAGE_TAIL == 8
            and tuple(module.TAPF_EPOCHS) == (1,) * 16 + (6,) * 16
        ),
        "rich_specific_group_count_exact": len(module.RICH_SPECIFIC_GROUPS) == 11,
        "fresh_default_scaler_only": (
            arm_source.count('torch.amp.GradScaler("cuda")') == 1
            and "init_scale" not in source
            and "growth_factor" not in source
            and "backoff_factor" not in source
            and "growth_interval" not in source
            and "._scale" not in source
            and "._growth_tracker" not in source
        ),
        "native_step_and_update_once_in_loop": (
            arm_source.count("scaler.step(optimizer)") == 1
            and arm_source.count("scaler.update()") == 1
            and arm_source.index("scaler.step(optimizer)")
            < arm_source.index("scaler.update()")
        ),
        "unscale_and_report_before_step": (
            arm_source.index("scaler.unscale_(optimizer)")
            < arm_source.index("base.gradient_report(")
            < arm_source.index("scaler.step(optimizer)")
        ),
        "single_materialized_loader_for_both_arms": (
            "cpu_batches = [clone_cpu_batch(next(iterator))" in gate_source
            and gate_source.count("cpu_batches,") >= 2
        ),
        "matched_rng_states_present": (
            "step_rng = prepare_step_rng(base, device)" in gate_source
            and '"step_rng_entries_matched"' in gate_source
        ),
        "group_state_capture_present": (
            "initial_group_state = parameter_group_state" in arm_source
            and '"group_state_sha256"' in arm_source
            and '"rich_specific_group_state_updated"' in source
        ),
        "baseline_relative_evaluator_present": all(
            token in source
            for token in (
                '"stage_tail_steady_state"',
                '"no_rich_extra_skip_on_d0_success"',
                '"rich_success_not_below_d0"',
                '"rich_nonfinite_groups_shared_subset"',
                '"rich_specific_groups_e6_active"',
            )
        ),
        "no_scheduler_step": not any(
            path.endswith("scheduler.step") for path in calls
        ),
        "no_checkpoint_load_or_save": (
            "torch.load" not in calls and "torch.save" not in calls
        ),
        "formal_training_not_authorized": (
            '"formal_training_authorized": True' not in source
            and '"formal_training_authorized": False' in main_source
        ),
        "production_preflight_only_gated": (
            '"production_preflight_authorized": passed' in gate_source
            and '"production_preflight_authorized": False' in main_source
        ),
        "matched_synthetic_pass": matched["status"] == "PASS"
        and all(matched["gates"].values()),
        "rich_extra_skip_fails": extra_skip["status"] == "FAIL"
        and not extra_skip["gates"]["no_rich_extra_skip_on_d0_success"],
        "persistent_tail_failure_fails": persistent_tail["status"] == "FAIL"
        and not persistent_tail["gates"]["stage_tail_steady_state"],
        "rich_specific_nonfinite_fails": rich_specific["status"] == "FAIL"
        and not rich_specific["gates"]["rich_specific_groups_always_finite"]
        and not rich_specific["gates"]["rich_nonfinite_groups_shared_subset"],
        "rich_specific_inactive_fails": inactive["status"] == "FAIL"
        and not inactive["gates"]["rich_specific_groups_e6_active"]
        and not inactive["gates"]["rich_specific_group_state_updated"],
        "rich_better_than_d0_passes": rich_better["status"] == "PASS"
        and all(rich_better["gates"].values()),
        "sealed_boundaries_present": all(
            token in source
            for token in (
                '"exp394_remains_sealed": True',
                '"exp395_remains_sealed": True',
                '"exp396_remains_sealed": True',
                '"exp397_remains_sealed": True',
            )
        ),
    }
    cuda_after = torch.cuda.is_initialized()
    gates["cuda_never_initialized"] = not cuda_before and not cuda_after
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "git_head": run_text(["git", "rev-parse", "HEAD"], repo_root),
        "cuda_script_sha256": sha256_file(script_path),
        "static_script_sha256": sha256_file(Path(__file__).resolve()),
        "reporter_dependency_sha256": sha256_file(reporter_path),
        "source_sha256": source_sha,
        "matched_evaluation": matched,
        "extra_skip_evaluation": extra_skip,
        "persistent_tail_evaluation": persistent_tail,
        "rich_specific_failure_evaluation": rich_specific,
        "rich_specific_inactive_evaluation": inactive,
        "rich_better_evaluation": rich_better,
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
        raise RuntimeError("Refusing to overwrite exp398 static assets")
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
