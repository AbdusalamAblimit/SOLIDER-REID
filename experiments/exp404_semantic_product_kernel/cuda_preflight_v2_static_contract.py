#!/usr/bin/env python3
"""Static authorization gate for fresh exp404 CUDA preflight v2."""

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
    "v1_invalid": "9958ec661fcaaea20499be04e0450085d76ec3ec5094e8df03179ccff426b498",
    "production_v3_contract": "ce85da278b551a66cacaddd14b3fda79bff356fcee4f7aeff717a927710534ef",
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
        raise RuntimeError("V2 static output must be fresh")

    paths = {
        "config": DIRECTORY / "swin_tiny_spk_formal.yml",
        "v1_core": DIRECTORY / "cuda_amp_preflight.py",
        "v1_invalid": DIRECTORY / "cuda_amp_preflight_v1_sealed_invalid.json",
        "production_v3_contract": DIRECTORY / "production_cpu_v3_contract.py",
        "production_v3_result": DIRECTORY / "production_cpu_v3_result.json",
        "tapf": ROOT / "model/tapf.py",
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    v1_record = json.loads(paths["v1_invalid"].read_text(encoding="utf-8"))
    v3_result = json.loads(paths["production_v3_result"].read_text(encoding="utf-8"))
    wrapper_path = DIRECTORY / "cuda_amp_preflight_v2.py"
    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    wrapper_tree = ast.parse(wrapper_source, filename=str(wrapper_path))
    gates = {
        "cuda_not_initialized": not torch.cuda.is_initialized(),
        "frozen_sources_exact": actual == EXPECTED,
        "v1_sealed_invalid": v1_record["status"] == "SEALED_INVALID_RUNTIME"
        and not v1_record["formal_training_started"],
        "production_v3_all_pass": v3_result["status"]
        == "PRODUCTION_CPU_V3_PASS"
        and v3_result["gate_count"] == 49
        and v3_result["gate_pass_count"] == 49,
        "joint_field_shape_gate_present": v3_result["gates"][
            "train_consumer_joint_field_shape_exact"
        ],
        "five_slot_mutant_caught": v3_result["gates"][
            "five_slot_region_field_mutant_caught"
        ],
        "v2_wrapper_ast_valid": isinstance(wrapper_tree, ast.Module),
        "v2_execution_tag": "exp404_cuda_amp_preflight_v2" in wrapper_source,
        "v2_contract_tag": "joint_field_v3" in wrapper_source,
        "v2_delegates_frozen_core": 'with_name("cuda_amp_preflight.py")'
        in wrapper_source,
        "no_preflight_output_local": not (
            DIRECTORY / "cuda_amp_preflight_v2_result.json"
        ).exists(),
    }
    passed = all(gates.values())
    result = {
        "experiment": "exp404_semantic_product_kernel",
        "execution": "exp404_cuda_amp_preflight_v2",
        "status": "CUDA_PREFLIGHT_V2_STATIC_PASS" if passed else "CUDA_PREFLIGHT_V2_STATIC_FAIL",
        "cuda_execution_authorized": passed,
        "formal_training_authorized": False,
        "gate_count": len(gates),
        "gate_pass_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "source_sha256": {
            **actual,
            "v2_wrapper": sha256_file(wrapper_path),
            "contract": sha256_file(Path(__file__)),
        },
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
