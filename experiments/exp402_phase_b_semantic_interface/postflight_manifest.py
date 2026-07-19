#!/usr/bin/env python3
"""Post-exit manifest for one exp402 CUDA execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ANOMALY_PATTERNS = {
    "nan_or_inf": re.compile(
        r"(?i)(?<![A-Za-z])(?:nan|[+-]?inf)(?![A-Za-z])"
    ),
    "traceback": re.compile(r"(?i)traceback"),
    "runtime_error": re.compile(r"(?i)runtimeerror"),
    "oom": re.compile(r"(?i)out of memory|\boom\b"),
    "nonfinite": re.compile(r"(?i)non[-_ ]?finite"),
    "overflow": re.compile(r"(?i)overflow"),
    "amp_numeric_warning": re.compile(
        r"(?i)(?:gradscaler|autocast).*(?:nan|inf|overflow|non[-_ ]?finite)"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(command) -> str:
    return subprocess.check_output(command, text=True).strip()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args):
    result_path = Path(args.result).resolve()
    runner_path = Path(args.runner).resolve()
    manifest_path = Path(args.manifest).resolve()
    audit_script = Path(args.audit_script).resolve()
    core_path = Path(args.core).resolve()
    script_path = Path(__file__).resolve()
    for path in (result_path, runner_path, audit_script, core_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    if Path("/home/afr") not in manifest_path.parents:
        raise RuntimeError("Manifest must remain under /home/afr")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    runner_text = runner_path.read_text(encoding="utf-8", errors="replace")
    anomaly_counts = {
        name: len(pattern.findall(runner_text))
        for name, pattern in ANOMALY_PATTERNS.items()
    }
    compute_output = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    compute_pids = [
        line.strip() for line in compute_output.splitlines() if line.strip()
    ]
    gpu_rows = run_text(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    process_exited = not Path(f"/proc/{int(args.audit_pid)}").exists()
    gates = {
        "audit_exit_zero": int(args.exit_code) == 0,
        "result_status_pass": result.get("status") == "PASS",
        "audit_process_exited": process_exited,
        "gpu_compute_processes_zero": not compute_pids,
        "runner_anomalies_zero": not any(anomaly_counts.values()),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "decision": result.get("decision"),
        "mode": result.get("mode"),
        "gates": gates,
        "audit_pid": int(args.audit_pid),
        "audit_exit_code": int(args.exit_code),
        "gpu": {
            "compute_pids": compute_pids,
            "rows_memory_mib_utilization_percent": gpu_rows,
        },
        "anomaly_counts": anomaly_counts,
        "sha256": {
            "result": sha256_file(result_path),
            "runner": sha256_file(runner_path),
            "audit_script": sha256_file(audit_script),
            "core": sha256_file(core_path),
            "postflight_script": sha256_file(script_path),
        },
        "paths": {
            "result": str(result_path),
            "runner": str(runner_path),
            "audit_script": str(audit_script),
            "core": str(core_path),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audit-script", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--audit-pid", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    if manifest_path.exists() or temporary.exists():
        raise FileExistsError("Postflight manifest path must be fresh")
    result = run(args)
    atomic_json(manifest_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
