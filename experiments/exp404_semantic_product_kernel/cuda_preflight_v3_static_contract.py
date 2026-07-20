#!/usr/bin/env python3
"""Static authorization gate for default-GradScaler exp404 CUDA v3."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
EXPECTED = {
    "config": "2bd191ef96da0158a57f917831ea70627f1fef163397219ce1168e3e30bb297d",
    "v1_core": "fb0a21168bef619a561bb77da0a2e5fe9216fde114ea7c34705c3fec544b7fe7",
    "v2_result": "d49e9421052675193eacb91828918033cbeefcd60a6702d2b31aad82c3a20c29",
    "production_v3_result": "56dc8a29957674034c9fb53b0894e686dfbc861c6c7668c3bffda2feed274603",
    "tapf": "72ff5a609c7a080d848e96a2c12239795388441cc13b85519ef2cbf42f04bf2a",
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("V3 static output must be fresh")

    paths = {
        "config": DIRECTORY / "swin_tiny_spk_formal.yml",
        "v1_core": DIRECTORY / "cuda_amp_preflight.py",
        "v2_result": DIRECTORY / "cuda_amp_preflight_v2_result.json",
        "production_v3_result": DIRECTORY / "production_cpu_v3_result.json",
        "tapf": ROOT / "model/tapf.py",
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    v2 = json.loads(paths["v2_result"].read_text(encoding="utf-8"))
    production = json.loads(
        paths["production_v3_result"].read_text(encoding="utf-8")
    )
    core_source = paths["v1_core"].read_text(encoding="utf-8")
    wrapper_path = DIRECTORY / "cuda_amp_preflight_v3.py"
    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    wrapper_tree = ast.parse(wrapper_source, filename=str(wrapper_path))
    expected_scale_before = [65536.0, 32768.0, 16384.0, 8192.0]
    expected_scale_after = [32768.0, 16384.0, 8192.0, 4096.0]
    gates = {
        "cuda_not_initialized": not torch.cuda.is_initialized(),
        "frozen_sources_exact": actual == EXPECTED,
        "v2_complete_fail_record": v2["status"] == "CUDA_AMP_PREFLIGHT_FAIL"
        and not v2["formal_training_authorized"]
        and len(v2["attempts"]) == 4,
        "v2_native_scale_sequence_exact": [
            record["scale_before"] for record in v2["attempts"]
        ]
        == expected_scale_before
        and [record["scale_after"] for record in v2["attempts"]]
        == expected_scale_after,
        "v2_all_target_grads_each_attempt": all(
            record["evidence_grad_finite_nonzero"]
            and record["feature_16_group_grad_finite_nonzero"]
            and record["factor_16_group_grad_finite_nonzero"]
            for record in v2["attempts"]
        ),
        "production_v3_still_pass": production["status"]
        == "PRODUCTION_CPU_V3_PASS"
        and production["gate_pass_count"] == production["gate_count"] == 49,
        "core_uses_default_gradscaler": "scaler = amp.GradScaler()"
        in core_source
        and "init_scale" not in core_source,
        "v3_wrapper_ast_valid": isinstance(wrapper_tree, ast.Module),
        "v3_execution_tag": "exp404_cuda_amp_preflight_v3" in wrapper_source,
        "v3_natural_backoff_tag": "default_gradscaler_natural_backoff_max8"
        in wrapper_source,
        "v3_attempts_frozen8": 'sys.argv.extend(["--max-attempts", "8"])'
        in wrapper_source,
        "v3_rejects_cli_attempt_override": (
            'if "--max-attempts" in sys.argv' in wrapper_source
        ),
        "v3_delegates_unchanged_core": 'with_name("cuda_amp_preflight.py")'
        in wrapper_source,
        "no_v3_actual_result_local": not (
            DIRECTORY / "cuda_amp_preflight_v3_result.json"
        ).exists(),
    }
    passed = all(gates.values())
    result = {
        "experiment": "exp404_semantic_product_kernel",
        "execution": "exp404_cuda_amp_preflight_v3",
        "status": "CUDA_PREFLIGHT_V3_STATIC_PASS" if passed else "CUDA_PREFLIGHT_V3_STATIC_FAIL",
        "cuda_execution_authorized": passed,
        "formal_training_authorized": False,
        "gate_count": len(gates),
        "gate_pass_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "source_sha256": {
            **actual,
            "v3_wrapper": sha256_file(wrapper_path),
            "contract": sha256_file(Path(__file__)),
        },
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
