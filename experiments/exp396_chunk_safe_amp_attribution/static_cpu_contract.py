#!/usr/bin/env python3
"""CPU-only contract for the exp396 chunk-safe exact gradient reporter."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import tempfile
import traceback
from pathlib import Path

import torch


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
EXPECTED_CHUNK_ELEMENTS = 1_048_576
OVERSIZE_ELEMENTS = 16_777_217
EXPECTED_EXP395_SHA256 = {
    "experiments/exp395_amp_gradient_attribution/cuda_amp_attribution.py":
        "64840b710db587720aa8807571212b246af3eabb54306bd5aa1bbf692f5ea08b",
    "experiments/exp395_amp_gradient_attribution/cuda_actual_invalid_result.json":
        "cdffff60b1b6e04e6bb0b13bb54e12518380421675c59c2f2c785f1b7a5adb75",
    "experiments/exp395_amp_gradient_attribution/cuda_actual_invalid.runner.json":
        "cdffff60b1b6e04e6bb0b13bb54e12518380421675c59c2f2c785f1b7a5adb75",
    "experiments/exp395_amp_gradient_attribution/cuda_actual_manifest.json":
        "3a0ef5d98dd6387b330958bbfb1e9d893e60745e8857237bbbbe375778886c64",
}
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
EXPECTED_D0_LOSSES = ("reid", "heatmap", "confidence", "pose", "total")
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_text(command, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def load_reporter(path: Path):
    spec = importlib.util.spec_from_file_location("exp396_cuda_reporter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def literal_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise KeyError(name)


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


def reference_statistics(gradients):
    flattened = (
        torch.cat([gradient.detach().cpu().to(torch.float64).flatten()
                   for gradient in gradients])
        if gradients
        else torch.empty(0, dtype=torch.float64)
    )
    finite_mask = torch.isfinite(flattened)
    finite_abs = flattened[finite_mask].abs()
    quantiles = (
        torch.quantile(
            finite_abs,
            torch.tensor([0.50, 0.95, 0.99], dtype=torch.float64),
            interpolation="linear",
        )
        if finite_abs.numel()
        else None
    )
    return {
        "elements": int(flattened.numel()),
        "finite_elements": int(finite_mask.sum()),
        "nan_elements": int(torch.isnan(flattened).sum()),
        "posinf_elements": int(torch.isposinf(flattened).sum()),
        "neginf_elements": int(torch.isneginf(flattened).sum()),
        "all_finite": bool(finite_mask.all()) if flattened.numel() else True,
        "finite_abs_max": float(finite_abs.max()) if finite_abs.numel() else None,
        "finite_l2": (
            float(torch.linalg.vector_norm(finite_abs))
            if finite_abs.numel()
            else None
        ),
        "finite_abs_p50": float(quantiles[0]) if quantiles is not None else None,
        "finite_abs_p95": float(quantiles[1]) if quantiles is not None else None,
        "finite_abs_p99": float(quantiles[2]) if quantiles is not None else None,
    }


def scalar_close(left, right, relative=1e-12):
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=relative, abs_tol=0.0)


def statistics_match(actual, expected):
    exact_fields = (
        "elements",
        "finite_elements",
        "nan_elements",
        "posinf_elements",
        "neginf_elements",
        "all_finite",
    )
    range_fields = (
        "finite_abs_max",
        "finite_l2",
        "finite_abs_p50",
        "finite_abs_p95",
        "finite_abs_p99",
    )
    return all(actual[field] == expected[field] for field in exact_fields) and all(
        scalar_close(actual[field], expected[field]) for field in range_fields
    )


def monotonic_quantile(values: torch.Tensor, quantile: float) -> float:
    rank = (values.numel() - 1) * quantile
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower = float(values[lower_index])
    upper = float(values[upper_index])
    return lower + (upper - lower) * (rank - lower_index)


def run_contract(repo_root: Path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("exp396 static contract requires hidden CUDA")
    cuda_before = torch.cuda.is_initialized()
    script_path = (
        repo_root
        / "experiments/exp396_chunk_safe_amp_attribution/cuda_amp_attribution.py"
    )
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = call_paths(tree)
    reporter_source = function_source(
        source, tree, "chunk_safe_finite_statistics"
    )
    gradient_report_source = function_source(source, tree, "gradient_report")
    row_source = function_source(source, tree, "run_loss_row")
    main_source = function_source(source, tree, "main")
    module = load_reporter(script_path)

    exp395_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_EXP395_SHA256
    }
    source_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }

    small_gradients = [
        torch.tensor(
            [0.0, -1.0, 2.0, float("nan"), float("inf"), -float("inf")],
            dtype=torch.float64,
        ),
        torch.tensor([-3.5, 0.25, 9.0, 0.0, -0.125], dtype=torch.float32),
    ]
    small_before = [gradient.clone() for gradient in small_gradients]
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        small_actual = module.chunk_safe_finite_statistics(
            small_gradients, scratch, "small"
        )
        small_scratch_empty = not any(scratch.iterdir())
    small_expected = reference_statistics(small_gradients)
    small_inputs_exact = all(
        torch.allclose(before, after, rtol=0.0, atol=0.0, equal_nan=True)
        for before, after in zip(small_before, small_gradients)
    )

    multi_gradients = [
        torch.linspace(
            -2.0,
            3.0,
            steps=EXPECTED_CHUNK_ELEMENTS + 3,
            dtype=torch.float64,
        ),
        torch.tensor([7.0, -11.0, 0.0, 0.5], dtype=torch.float64),
    ]
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        multi_actual = module.chunk_safe_finite_statistics(
            multi_gradients, scratch, "multi_chunk"
        )
        multi_scratch_empty = not any(scratch.iterdir())
    multi_expected = reference_statistics(multi_gradients)

    empty_actual = module.chunk_safe_finite_statistics([], Path(tempfile.gettempdir()), "empty")

    parameters = [
        torch.nn.Parameter(torch.zeros_like(small_gradients[0])),
        torch.nn.Parameter(torch.zeros_like(small_gradients[1])),
    ]
    for parameter, gradient in zip(parameters, small_gradients):
        parameter.grad = gradient.clone()
    groups = {name: [] for name in EXPECTED_GROUPS}
    groups["backbone"] = [("p0", parameters[0]), ("p1", parameters[1])]
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        full_report = module.gradient_report(groups, scratch, "full_report")
        full_report_scratch_empty = not any(scratch.iterdir())
    backbone = full_report["backbone"]

    oversized = torch.arange(OVERSIZE_ELEMENTS, dtype=torch.float32)
    oversized_before_sha = module.sha256_tensor(oversized)
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        oversized_actual = module.chunk_safe_finite_statistics(
            [oversized], scratch, "oversized"
        )
        oversized_scratch_empty = not any(scratch.iterdir())
    oversized_after_sha = module.sha256_tensor(oversized)
    oversized_expected_quantiles = [
        monotonic_quantile(oversized, quantile)
        for quantile in (0.50, 0.95, 0.99)
    ]

    injected_scratch_empty = False
    injected_exception_seen = False
    original_quantile = module.linear_quantiles_from_sorted
    try:
        def injected_failure(*_args, **_kwargs):
            raise RuntimeError("injected reporter failure")

        module.linear_quantiles_from_sorted = injected_failure
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            try:
                module.chunk_safe_finite_statistics(
                    [torch.arange(4097, dtype=torch.float64)],
                    scratch,
                    "injected",
                )
            except RuntimeError as error:
                injected_exception_seen = str(error) == "injected reporter failure"
            injected_scratch_empty = not any(scratch.iterdir())
    finally:
        module.linear_quantiles_from_sorted = original_quantile

    oversized_quantiles = [
        oversized_actual["finite_abs_p50"],
        oversized_actual["finite_abs_p95"],
        oversized_actual["finite_abs_p99"],
    ]
    gates = {
        "source_commit_exists": run_text(
            ["git", "cat-file", "-t", SOURCE_COMMIT], repo_root
        ) == "commit",
        "exp395_failure_assets_exact": exp395_sha == EXPECTED_EXP395_SHA256,
        "production_source_sha_exact": source_sha == EXPECTED_SOURCE_SHA256,
        "chunk_size_exact": module.REPORTER_CHUNK_ELEMENTS
        == EXPECTED_CHUNK_ELEMENTS,
        "reporter_name_exact": module.REPORTER_NAME
        == "chunk_safe_exact_memmap_v1",
        "loss_and_group_order_exact": (
            tuple(module.GROUP_NAMES) == EXPECTED_GROUPS
            and tuple(module.BASELINE_LOSSES) == EXPECTED_D0_LOSSES
            and tuple(module.RICH_LOSSES) == EXPECTED_RICH_LOSSES
        ),
        "production_reporter_has_no_torch_quantile": "torch.quantile" not in source,
        "production_reporter_has_no_full_torch_cat": (
            "torch.cat" not in reporter_source
            and "torch.cat" not in gradient_report_source
        ),
        "memmap_exact_sort_and_unlink_present": all(
            token in reporter_source
            for token in ("np.memmap(", 'values.sort(kind="quicksort")',
                          "memmap_path.unlink(missing_ok=True)")
        ),
        "small_statistics_match_reference": statistics_match(
            small_actual, small_expected
        ),
        "small_inputs_unchanged": small_inputs_exact,
        "small_scratch_clean": small_scratch_empty,
        "multi_chunk_statistics_match_reference": statistics_match(
            multi_actual, multi_expected
        ),
        "multi_chunk_scratch_clean": multi_scratch_empty,
        "empty_semantics_exact": (
            empty_actual["elements"] == 0
            and empty_actual["finite_elements"] == 0
            and empty_actual["all_finite"]
            and all(
                empty_actual[field] is None
                for field in (
                    "finite_abs_max", "finite_l2", "finite_abs_p50",
                    "finite_abs_p95", "finite_abs_p99",
                )
            )
        ),
        "full_report_tensor_counts_exact": (
            backbone["parameter_tensors"] == 2
            and backbone["grad_present_tensors"] == 2
            and backbone["grad_absent_tensors"] == 0
            and backbone["grad_nonzero_tensors"] == 2
            and backbone["grad_zero_tensors"] == 0
            and statistics_match(backbone, small_expected)
        ),
        "full_report_scratch_clean": full_report_scratch_empty,
        "oversized_count_exact": (
            oversized_actual["elements"] == OVERSIZE_ELEMENTS
            and oversized_actual["finite_elements"] == OVERSIZE_ELEMENTS
            and oversized_actual["all_finite"]
        ),
        "oversized_quantiles_analytic_exact": oversized_quantiles
        == oversized_expected_quantiles,
        "oversized_input_unchanged": oversized_before_sha == oversized_after_sha,
        "oversized_scratch_clean": oversized_scratch_empty,
        "injected_exception_cleanup_exact": (
            injected_exception_seen and injected_scratch_empty
        ),
        "scaled_capture_before_unscale": (
            row_source.index("scaled = gradient_report")
            < row_source.index("scaler.unscale_(optimizer)")
            < row_source.index("unscaled = gradient_report")
        ),
        "fresh_default_scaler_unchanged": (
            'torch.amp.GradScaler("cuda")' in row_source
            and "init_scale" not in source
            and module.EXPECTED_INITIAL_SCALE == 65536.0
        ),
        "no_optimizer_or_scaler_updates": (
            not any(path.endswith("optimizer.step") for path in calls)
            and "scaler.step" not in calls
            and "scaler.update" not in calls
            and not any(path.endswith("scheduler.step") for path in calls)
        ),
        "fresh_exp396_assets_required": (
            '"exp396" in clip_path.name' in source
            and '"exp396" in codebook_path.name' in source
        ),
        "scratch_temporary_directory_and_final_gate": (
            "tempfile.TemporaryDirectory(" in source
            and '"scratch_cleanup_exact"' in source
        ),
        "formal_training_not_authorized": (
            '"formal_training_authorized": True' not in source
            and '"formal_training_authorized": False' in main_source
        ),
        "exp394_and_exp395_remain_sealed": (
            '"exp394_remains_sealed": True' in source
            and '"exp395_remains_sealed": True' in source
        ),
    }
    del oversized
    cuda_after = torch.cuda.is_initialized()
    gates["cuda_never_initialized"] = not cuda_before and not cuda_after
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "git_head": run_text(["git", "rev-parse", "HEAD"], repo_root),
        "cuda_script_sha256": sha256_file(script_path),
        "static_script_sha256": sha256_file(Path(__file__).resolve()),
        "exp395_sha256": exp395_sha,
        "source_sha256": source_sha,
        "chunk_elements": EXPECTED_CHUNK_ELEMENTS,
        "oversize_elements": OVERSIZE_ELEMENTS,
        "small_actual": small_actual,
        "small_expected": small_expected,
        "multi_actual": multi_actual,
        "multi_expected": multi_expected,
        "oversized_summary": oversized_actual,
        "oversized_expected_quantiles": oversized_expected_quantiles,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "optimizer_updates": 0,
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
        raise RuntimeError("Refusing to overwrite exp396 static assets")
    try:
        result = run_contract(repo_root)
    except Exception as error:
        result = {
            "status": "INVALID",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "optimizer_updates": 0,
            "checkpoint_count": 0,
        }
    write_json(output, result)
    write_json(runner, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
