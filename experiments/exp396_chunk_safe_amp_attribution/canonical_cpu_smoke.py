#!/usr/bin/env python3
"""Canonical-runtime CPU-only oversized smoke for the exp396 reporter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path

import torch


EXPECTED_SCRIPT_SHA256 = (
    "6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164"
)
OVERSIZE_ELEMENTS = 16_777_217


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp396_canonical_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def expected_quantile(values: torch.Tensor, quantile: float) -> float:
    rank = (values.numel() - 1) * quantile
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower = float(values[lower_index])
    upper = float(values[upper_index])
    return lower + (upper - lower) * (rank - lower_index)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--scratch-root", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("Canonical CPU smoke requires hidden CUDA")
    cuda_before = torch.cuda.is_initialized()
    script = Path(args.script).resolve()
    scratch_parent = Path(args.scratch_root).resolve()
    if sha256_file(script) != EXPECTED_SCRIPT_SHA256:
        raise RuntimeError("exp396 production script SHA mismatch")
    module = load_module(script)
    values = torch.arange(OVERSIZE_ELEMENTS, dtype=torch.float32)
    expected = [expected_quantile(values, q) for q in (0.50, 0.95, 0.99)]
    with tempfile.TemporaryDirectory(
        prefix=".exp396_canonical_cpu_smoke_", dir=scratch_parent
    ) as directory:
        scratch = Path(directory)
        report = module.chunk_safe_finite_statistics(
            [values], scratch, "canonical_oversized"
        )
        per_cell_scratch_empty = not any(scratch.iterdir())
    root_scratch_empty = not any(
        scratch_parent.glob(".exp396_canonical_cpu_smoke_*")
    )
    actual = [
        report["finite_abs_p50"],
        report["finite_abs_p95"],
        report["finite_abs_p99"],
    ]
    cuda_after = torch.cuda.is_initialized()
    gates = {
        "count_exact": report["elements"] == OVERSIZE_ELEMENTS,
        "all_finite": report["all_finite"],
        "quantiles_exact": actual == expected,
        "per_cell_scratch_empty": per_cell_scratch_empty,
        "root_scratch_empty": root_scratch_empty,
        "cuda_never_initialized": not cuda_before and not cuda_after,
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "python": os.sys.version,
        "torch": torch.__version__,
        "numpy": module.np.__version__,
        "script_sha256": sha256_file(script),
        "oversize_elements": OVERSIZE_ELEMENTS,
        "actual_quantiles": actual,
        "expected_quantiles": expected,
        "gates": gates,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
