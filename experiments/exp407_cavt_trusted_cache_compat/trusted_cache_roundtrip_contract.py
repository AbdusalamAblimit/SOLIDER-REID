#!/usr/bin/env python3
"""Targeted MMPOSE-ABU roundtrip contract for exp407 trusted caches."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import torch


EXPECTED_RUNTIME = Path("/usr/local/anaconda3/envs/mmpose-abu/bin/python")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--scratch-dir", required=True)
    return parser.parse_args()


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("exp407_roundtrip_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load exp407 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_loader_contract(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "write_cache_once"
    ]
    if len(functions) != 1:
        raise RuntimeError("write_cache_once must exist exactly once")
    calls = [
        node for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]
    if len(calls) != 1:
        raise RuntimeError("write_cache_once must call torch.load exactly once")
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    value = keywords.get("weights_only")
    if not isinstance(value, ast.Constant) or value.value is not False:
        raise RuntimeError("trusted cache self-check must set weights_only=False")


def main() -> None:
    args = parse_args()
    if Path(sys.executable).resolve() != EXPECTED_RUNTIME.resolve():
        raise RuntimeError("contract requires fixed MMPOSE-ABU runtime")
    runner_path = Path(args.runner).resolve()
    scratch = Path(args.scratch_dir).resolve()
    if Path("/home/afr") not in (scratch, *scratch.parents):
        raise RuntimeError("remote scratch must remain under /home/afr")
    scratch.mkdir(parents=True, exist_ok=False)
    verify_loader_contract(runner_path)
    runner = load_runner(runner_path)
    payload = {
        "schema": "exp407-p0b-preflight-cache-v1",
        "execution": "exp407-p0b-preflight-v1",
        "formal": False,
        "tensor": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        "metadata": {
            "paths": ("a.jpg", "b.jpg"),
            "controls": ["correct", "wrong-mask", "zero"],
            "nested": {"decision": "PREFLIGHT_ONLY_PASS", "value": 0.125},
        },
    }
    shas = []
    for index in (1, 2):
        path = scratch / ("roundtrip-%d.pt" % index)
        shas.append(runner.write_cache_once(path, payload))
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if loaded["schema"] != payload["schema"]:
            raise RuntimeError("published schema mismatch")
        if loaded["metadata"] != payload["metadata"]:
            raise RuntimeError("published metadata mismatch")
        if not torch.equal(loaded["tensor"], payload["tensor"]):
            raise RuntimeError("published tensor mismatch")
        if path.with_suffix(path.suffix + ".tmp").exists():
            raise RuntimeError("temporary file leaked after publication")
    if shas[0] != shas[1]:
        raise RuntimeError("fresh roundtrips are not byte-exact")
    print(json.dumps({
        "schema": "exp407-trusted-cache-roundtrip-v1",
        "status": "PASS",
        "torch": torch.__version__,
        "sha256": shas[0],
        "cases": {
            "fixed_runtime": True,
            "weights_only_false_ast": True,
            "mixed_payload_roundtrip": True,
            "byte_exact_repeat": True,
            "temporary_removed": True,
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
