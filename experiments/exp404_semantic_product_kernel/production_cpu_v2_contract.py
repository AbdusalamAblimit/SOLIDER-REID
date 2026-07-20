#!/usr/bin/env python3
"""Reporter-corrected v2 production CPU contract for exp404 SPK."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V1_CONTRACT = Path(__file__).with_name("production_cpu_v1_contract.py")
V1_RESULT = Path(__file__).with_name("production_cpu_v1_result.json")


def load_v1():
    spec = importlib.util.spec_from_file_location("exp404_production_v1", V1_CONTRACT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding_precedes_bnneck(path: Path) -> tuple[bool, int, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    build_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "build_transformer"
    )
    forward = next(
        node
        for node in build_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    binding_lines = []
    bottleneck_lines = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        function = node.value.func
        if not isinstance(function, ast.Attribute):
            continue
        if function.attr == "semantic_product_kernel":
            binding_lines.append(node.lineno)
        if function.attr == "bottleneck":
            bottleneck_lines.append(node.lineno)
    if len(binding_lines) != 1 or len(bottleneck_lines) != 1:
        return False, -1, -1
    binding_line = binding_lines[0]
    bottleneck_line = bottleneck_lines[0]
    return binding_line < bottleneck_line, binding_line, bottleneck_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v1 = load_v1()
    payload = v1.run_contract(args.output.resolve())
    binding_ok, binding_line, bottleneck_line = binding_precedes_bnneck(
        ROOT / "model/make_model.py"
    )
    payload["gates"]["binding_before_bnneck"] = binding_ok
    payload["gate_pass_count"] = sum(
        bool(value) for value in payload["gates"].values()
    )
    passed = payload["gate_pass_count"] == payload["gate_count"]
    payload["status"] = "PRODUCTION_CPU_PASS" if passed else "PRODUCTION_CPU_FAIL"
    payload["reporter_revision"] = {
        "version": 2,
        "reason": (
            "v1 used unrestricted string.index and matched the constructor "
            "BNNeck before build_transformer.forward"
        ),
        "binding_line": binding_line,
        "bottleneck_line": bottleneck_line,
        "v1_contract_sha256": v1.sha256_file(V1_CONTRACT),
        "v1_result_sha256": v1.sha256_file(V1_RESULT),
        "v1_status": json.loads(V1_RESULT.read_text(encoding="utf-8"))["status"],
    }
    v1.atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PRODUCTION_CPU_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
