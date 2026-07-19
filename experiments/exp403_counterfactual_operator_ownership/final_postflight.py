#!/usr/bin/env python3
"""Post-exit manifest for the once-only exp403 final audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


PATTERNS = {
    "nan_or_inf": re.compile(r"(?i)(?<![A-Za-z])(?:nan|[+-]?inf)(?![A-Za-z])"),
    "traceback": re.compile(r"(?i)traceback"),
    "runtime_error": re.compile(r"(?i)runtimeerror"),
    "oom": re.compile(r"(?i)out of memory|\boom\b"),
    "nonfinite": re.compile(r"(?i)non[-_ ]?finite"),
    "overflow": re.compile(r"(?i)overflow"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for name in ("result", "runner", "manifest", "contract", "audit_wrapper",
                 "base_audit", "core", "base_core", "execution_wrapper"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--audit-pid", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)).resolve() for name in (
        "result", "runner", "manifest", "contract", "audit_wrapper",
        "base_audit", "core", "base_core", "execution_wrapper",
    )}
    if paths["manifest"].exists() or paths["manifest"].with_suffix(".json.tmp").exists():
        raise FileExistsError("Manifest path must be fresh")
    result = json.loads(paths["result"].read_text())
    contract = json.loads(paths["contract"].read_text())
    runner_text = paths["runner"].read_text(errors="replace")
    anomalies = {name: len(pattern.findall(runner_text)) for name, pattern in PATTERNS.items()}
    compute = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    gates = {
        "audit_exit_zero": args.exit_code == 0,
        "measurement_status_pass": result.get("status") == "PASS",
        "checkpoint_contract_pass": contract.get("status") == "PASS",
        "audit_process_exited": not Path(f"/proc/{args.audit_pid}").exists(),
        "gpu_compute_processes_zero": not [line for line in compute if line.strip()],
        "runner_anomalies_zero": not any(anomalies.values()),
        "contract_sha_exact": result.get("assets", {}).get("contract_sha256")
        == sha256_file(paths["contract"]),
    }
    payload = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "decision": result.get("decision"),
        "gates": gates,
        "audit_pid": args.audit_pid,
        "audit_exit_code": args.exit_code,
        "anomaly_counts": anomalies,
        "gpu_compute_pids": [line for line in compute if line.strip()],
        "sha256": {name: sha256_file(path) for name, path in paths.items() if name != "manifest"},
    }
    atomic_json(paths["manifest"], payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
