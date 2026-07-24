#!/usr/bin/env python3
"""CPU-only OOF adjudicator for the once-only exp416 PC-NEC fuel audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
for import_root in (REPOSITORY_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import fuel_core as core
import fuel_io


SCHEMA = "exp416-pcnec-audit-v1"
BANK_SCHEMA = "exp416-pcnec-bank-v1"
GEOMETRY_SCHEMA = "exp416-pcnec-geometry-v1"
FUEL_SCHEMA = "exp416-pcnec-fuel-cache-v1"
EXPECTED_TRAIN_COUNT = 15618

AUROC_AP_MIN_DELTA = 0.03
D0_MAP_R1_MIN_DELTA = 0.01
CONTROL_MAP_R1_MIN_DELTA = 0.005

BANK_FIELDS = {
    "schema",
    "relative_paths",
    "raw_pids",
    "relabeled_pids",
    "camids",
    "image_sha256",
    "d0_global",
    "query_indices",
    "candidate_indices",
    "pair_is_impostor",
    "d0_distance",
    "query_offsets",
}
FUEL_FIELDS = {
    "schema",
    "relative_paths",
    "image_sha256",
    "availability",
    "instance_rectangles",
    "canonical_rectangles",
    "d0_slot",
    "instance_clip",
    "canonical_clip",
    "global_clip",
    "instance_raw_hist",
    "canonical_raw_hist",
    "pair_row",
    "query_indices",
    "candidate_indices",
    "pair_energy",
    "common",
    "undecided",
    "wrong_donor_indices",
    "wrong_donor_invalid",
    "arm_names",
}


def _load_npz_bound(path, expected_sha256, fields, schema):
    resolved = Path(path).resolve(strict=True)
    if fuel_io.sha256_file(resolved) != str(expected_sha256):
        raise RuntimeError("input SHA256 mismatch: " + str(resolved))
    arrays = fuel_io.load_npz_exact(resolved, fields)
    if str(arrays["schema"].item()) != str(schema):
        raise RuntimeError("input schema mismatch: " + str(resolved))
    return resolved, arrays


def _load_geometry_summary(path, expected_sha256):
    resolved = Path(path).resolve(strict=True)
    if fuel_io.sha256_file(resolved) != str(expected_sha256):
        raise RuntimeError("geometry summary SHA256 mismatch")
    payload = fuel_io.readback_json(resolved)
    if payload.get("schema") != GEOMETRY_SCHEMA:
        raise RuntimeError("geometry summary schema mismatch")
    return resolved, payload


def build_query_payloads(bank, fuel):
    paths = tuple(str(value) for value in bank["relative_paths"].tolist())
    count = len(paths)
    if count != EXPECTED_TRAIN_COUNT:
        raise RuntimeError("unexpected train row count")
    if (
        tuple(str(value) for value in fuel["relative_paths"].tolist()) != paths
        or not np.array_equal(bank["image_sha256"], fuel["image_sha256"])
    ):
        raise RuntimeError("fuel cache changed RGB row identity/order")
    pair_count = len(bank["query_indices"])
    if (
        fuel["pair_row"].shape != (pair_count,)
        or not np.array_equal(
            fuel["pair_row"], np.arange(pair_count, dtype=fuel["pair_row"].dtype)
        )
        or not np.array_equal(bank["query_indices"], fuel["query_indices"])
        or not np.array_equal(bank["candidate_indices"], fuel["candidate_indices"])
    ):
        raise RuntimeError("fuel cache changed sealed pair rows")
    if tuple(str(value) for value in fuel["arm_names"].tolist()) != core.ARM_NAMES:
        raise RuntimeError("fuel cache arm order differs from frozen core")
    energy = np.asarray(fuel["pair_energy"], dtype=np.float64)
    if energy.shape != (pair_count, len(core.ARM_NAMES)):
        raise RuntimeError("fuel pair energy shape mismatch")
    if not np.isfinite(energy).all() or bool((energy < 0.0).any()):
        raise RuntimeError("fuel pair energy is non-finite or negative")
    if fuel["common"].shape != (pair_count, core.SLOT_COUNT):
        raise RuntimeError("fuel common-slot bitmap shape mismatch")
    if fuel["undecided"].shape != (pair_count,):
        raise RuntimeError("fuel UNDECIDED shape mismatch")
    if not np.array_equal(fuel["undecided"], ~fuel["common"].any(axis=1)):
        raise RuntimeError("fuel UNDECIDED differs from common-slot bitmap")

    offsets = np.asarray(bank["query_offsets"], dtype=np.int64)
    if (
        offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != pair_count
        or bool((np.diff(offsets) <= 0).any())
    ):
        raise RuntimeError("sealed query offsets are invalid")
    queries = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        start, stop = int(start), int(stop)
        query_indices = bank["query_indices"][start:stop]
        if not bool((query_indices == query_indices[0]).all()):
            raise RuntimeError("query offset spans multiple query rows")
        query_index = int(query_indices[0])
        candidate_index = bank["candidate_indices"][start:stop].astype(
            np.int64, copy=False
        )
        labels = bank["pair_is_impostor"][start:stop].astype(
            np.bool_, copy=False
        )
        if not bool(labels.any()) or not bool((~labels).any()):
            raise RuntimeError("query bank lacks one binary class")
        query_path = paths[query_index]
        arm_energy = {
            name: energy[start:stop, column].copy()
            for column, name in enumerate(core.ARM_NAMES)
        }
        if not np.array_equal(
            arm_energy["d0_only"],
            bank["d0_distance"][start:stop].astype(np.float64),
        ):
            raise RuntimeError("D0-only arm differs from sealed bank distance")
        queries.append(
            {
                "query_id": core.stable_hash_hex(
                    core.QUERY_ORDER_SALT, query_path
                ),
                "query_pid": int(bank["raw_pids"][query_index]),
                "d0_distance": bank["d0_distance"][start:stop].astype(
                    np.float64
                ),
                "candidate_paths": tuple(paths[row] for row in candidate_index),
                "impostor_positive": labels.copy(),
                "arm_energy": arm_energy,
            }
        )
    query_ids = [query["query_id"] for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError("query IDs are not unique")
    return queries


def evaluate_all_arms(queries):
    outputs = {
        arm: core.evaluate_arm_oof(queries, arm)
        for arm in core.ARM_NAMES
        if arm != "d0_only"
    }
    outputs["d0_only"] = core.evaluate_d0_only(queries)
    if set(outputs) != set(core.ARM_NAMES):
        raise RuntimeError("OOF output omitted a frozen arm")
    summaries = {name: value["summary"] for name, value in outputs.items()}
    strongest = core.select_strongest_controls(summaries)
    return outputs, summaries, strongest


def adjudicate(outputs, summaries, strongest, geometry_gate_pass):
    if set(strongest) != set(core.MAIN_METRICS):
        raise ValueError("strongest-control metric set mismatch")
    for metric in core.MAIN_METRICS:
        entry = strongest[metric]
        if set(entry) != {"arm", "value"}:
            raise ValueError("strongest-control entry schema mismatch")
        arm = str(entry["arm"])
        if arm not in core.CONTROL_ORDER:
            raise ValueError("strongest arm is not a frozen control")
        if not math.isclose(
            float(entry["value"]),
            float(summaries[arm][metric]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("strongest-control value differs from summary")
    correct = summaries["correct"]
    strict_control_wins = {}
    for metric in core.MAIN_METRICS:
        strict_control_wins[metric] = {
            control: bool(correct[metric] > summaries[control][metric])
            for control in core.CONTROL_ORDER
        }

    strongest_deltas = {
        metric: float(
            correct[metric]
            - summaries[strongest[metric]["arm"]][metric]
        )
        for metric in core.MAIN_METRICS
    }
    d0_deltas = {
        metric: float(correct[metric] - summaries["d0_only"][metric])
        for metric in ("mAP", "R1")
    }
    bootstrap_specs = (
        ("auroc", strongest["auroc"]["arm"]),
        ("average_precision", strongest["average_precision"]["arm"]),
        ("mAP", "d0_only"),
        ("R1", "d0_only"),
        ("mAP", strongest["mAP"]["arm"]),
        ("R1", strongest["R1"]["arm"]),
    )
    bootstraps = []
    for metric, control in bootstrap_specs:
        bootstraps.append(
            core.paired_pid_bootstrap(
                outputs["correct"]["rows"],
                outputs[control]["rows"],
                metric=metric,
                control_name=control,
            )
        )

    gates = {
        "auroc_delta_vs_strongest_ge_0_03": bool(
            strongest_deltas["auroc"] >= AUROC_AP_MIN_DELTA
        ),
        "average_precision_delta_vs_strongest_ge_0_03": bool(
            strongest_deltas["average_precision"] >= AUROC_AP_MIN_DELTA
        ),
        "mAP_delta_vs_d0_ge_0_01": bool(
            d0_deltas["mAP"] >= D0_MAP_R1_MIN_DELTA
        ),
        "R1_delta_vs_d0_ge_0_01": bool(
            d0_deltas["R1"] >= D0_MAP_R1_MIN_DELTA
        ),
        "mAP_delta_vs_strongest_ge_0_005": bool(
            strongest_deltas["mAP"] >= CONTROL_MAP_R1_MIN_DELTA
        ),
        "R1_delta_vs_strongest_ge_0_005": bool(
            strongest_deltas["R1"] >= CONTROL_MAP_R1_MIN_DELTA
        ),
        "six_bootstrap_lower_bounds_gt_zero": bool(
            all(item["one_sided_95_lower"] > 0.0 for item in bootstraps)
        ),
        "geometry_gate_pass": bool(geometry_gate_pass),
        "correct_strictly_beats_every_control_every_metric": bool(
            all(
                passed
                for metric_values in strict_control_wins.values()
                for passed in metric_values.values()
            )
        ),
    }
    go = bool(all(gates.values()))
    return {
        "strongest_deltas": strongest_deltas,
        "d0_deltas": d0_deltas,
        "strict_control_wins": strict_control_wins,
        "bootstraps": bootstraps,
        "gates": gates,
        "go": go,
        "verdict": (
            "PC-NEC FUEL GO / CERTIFICATE DESIGN AUTHORIZED / TRAINING NO-START"
            if go
            else "PC-NEC FUEL NO-GO / TRAINING NO-START / NO CANDIDATE"
        ),
    }


def _serializable_oof(outputs):
    query_ids = tuple(
        str(row["query_id"]) for row in outputs["correct"]["rows"]
    )
    query_pids = np.asarray(
        [int(row["query_pid"]) for row in outputs["correct"]["rows"]],
        dtype=np.int32,
    )
    metrics = np.empty(
        (len(query_ids), len(core.ARM_NAMES), len(core.MAIN_METRICS)),
        dtype=np.float64,
    )
    lambdas = np.empty((len(query_ids), len(core.ARM_NAMES)), dtype=np.float64)
    folds = np.empty((len(query_ids), len(core.ARM_NAMES)), dtype=np.int8)
    for arm_index, arm in enumerate(core.ARM_NAMES):
        rows = outputs[arm]["rows"]
        if tuple(str(row["query_id"]) for row in rows) != query_ids:
            raise RuntimeError("OOF arm query order differs")
        for query_index, row in enumerate(rows):
            if int(row["query_pid"]) != int(query_pids[query_index]):
                raise RuntimeError("OOF arm query PID differs")
            for metric_index, metric in enumerate(core.MAIN_METRICS):
                metrics[query_index, arm_index, metric_index] = float(
                    row[metric]
                )
            lambdas[query_index, arm_index] = float(row["lambda"])
            folds[query_index, arm_index] = int(row["fold"])
    return {
        "schema": np.asarray(SCHEMA),
        "query_ids": np.asarray(query_ids, dtype=np.str_),
        "query_pids": query_pids,
        "arm_names": np.asarray(core.ARM_NAMES, dtype=np.str_),
        "metric_names": np.asarray(core.MAIN_METRICS, dtype=np.str_),
        "query_metrics": metrics,
        "query_lambdas": lambdas,
        "query_folds": folds,
    }


def run_self_test():
    fold_to_pid = {}
    pid = 0
    while len(fold_to_pid) < core.FOLD_COUNT:
        fold_to_pid.setdefault(core.pid_fold(pid), pid)
        pid += 1
    queries = []
    for fold in range(core.FOLD_COUNT):
        query_pid = fold_to_pid[fold]
        arm_energy = {}
        for arm in core.ARM_NAMES:
            if arm == "correct":
                arm_energy[arm] = np.asarray((0.0, 1.0, 0.9))
            elif arm == "d0_only":
                arm_energy[arm] = np.asarray((0.9, 0.1, 0.2))
            else:
                arm_energy[arm] = np.zeros(3, dtype=np.float64)
        queries.append(
            {
                "query_id": "self-test-{}".format(fold),
                "query_pid": int(query_pid),
                "d0_distance": np.asarray((0.9, 0.1, 0.2)),
                "candidate_paths": (
                    "genuine-{}".format(fold),
                    "impostor-a-{}".format(fold),
                    "impostor-b-{}".format(fold),
                ),
                "impostor_positive": np.asarray((0, 1, 1), dtype=np.bool_),
                "arm_energy": arm_energy,
            }
        )
    outputs, summaries, strongest = evaluate_all_arms(queries)
    result = adjudicate(outputs, summaries, strongest, True)
    assert result["go"]
    arrays = _serializable_oof(outputs)
    assert arrays["query_metrics"].shape == (
        core.FOLD_COUNT,
        len(core.ARM_NAMES),
        len(core.MAIN_METRICS),
    )
    broken = dict(strongest)
    broken["auroc"] = {"arm": "correct", "value": 1.0}
    try:
        adjudicate(outputs, summaries, broken, True)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid strongest-control arm was accepted")
    print("EXP416_FUEL_AUDIT_SELF_TEST=PASS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--bank")
    parser.add_argument("--bank-sha256")
    parser.add_argument("--geometry-summary")
    parser.add_argument("--geometry-summary-sha256")
    parser.add_argument("--fuel-cache")
    parser.add_argument("--fuel-cache-sha256")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    required = (
        "bank",
        "bank_sha256",
        "geometry_summary",
        "geometry_summary_sha256",
        "fuel_cache",
        "fuel_cache_sha256",
        "output_dir",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError("missing formal arguments: " + ",".join(missing))
    if fuel_io.cuda_compute_processes():
        raise RuntimeError("CPU adjudicator requires no concurrent CUDA process")

    bank_path, bank = _load_npz_bound(
        args.bank, args.bank_sha256, BANK_FIELDS, BANK_SCHEMA
    )
    geometry_path, geometry = _load_geometry_summary(
        args.geometry_summary, args.geometry_summary_sha256
    )
    fuel_path, fuel = _load_npz_bound(
        args.fuel_cache, args.fuel_cache_sha256, FUEL_FIELDS, FUEL_SCHEMA
    )
    queries = build_query_payloads(bank, fuel)
    outputs, summaries, strongest = evaluate_all_arms(queries)
    adjudication = adjudicate(
        outputs, summaries, strongest, geometry.get("geometry_gate_pass", False)
    )
    oof_arrays = _serializable_oof(outputs)

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute() or output_dir.exists():
        raise FileExistsError("audit output directory must be fresh and absolute")
    output_dir.mkdir(mode=0o755, parents=False)
    oof_path = output_dir / "oof_metrics.npz"
    fuel_io.atomic_npz(oof_path, oof_arrays)
    fuel_io.readback_npz(oof_path, oof_arrays)
    result = {
        "schema": SCHEMA,
        "verdict": adjudication["verdict"],
        "go": adjudication["go"],
        "inputs": {
            "bank": str(bank_path),
            "bank_sha256": str(args.bank_sha256),
            "geometry_summary": str(geometry_path),
            "geometry_summary_sha256": str(args.geometry_summary_sha256),
            "fuel_cache": str(fuel_path),
            "fuel_cache_sha256": str(args.fuel_cache_sha256),
        },
        "counts": {
            "train_rows": int(len(bank["relative_paths"])),
            "pair_rows": int(len(bank["query_indices"])),
            "fixed_queries": int(len(queries)),
            "query_pids": int(len({query["query_pid"] for query in queries})),
            "undecided_pairs": int(fuel["undecided"].sum()),
            "wrong_donor_invalid_pairs": int(
                fuel["wrong_donor_invalid"].sum()
            ),
        },
        "geometry_gate_pass": bool(geometry.get("geometry_gate_pass", False)),
        "arm_summaries": summaries,
        "selected_lambda_by_fold": {
            arm: output.get("selected", {})
            for arm, output in outputs.items()
            if arm != "d0_only"
        },
        "strongest_controls": strongest,
        "adjudication": adjudication,
        "oof_metrics": str(oof_path),
        "oof_metrics_sha256": fuel_io.sha256_file(oof_path),
        "source_sha256": {
            "fuel_audit.py": fuel_io.sha256_file(Path(__file__).resolve()),
            "fuel_core.py": fuel_io.sha256_file(SCRIPT_DIR / "fuel_core.py"),
            "fuel_io.py": fuel_io.sha256_file(SCRIPT_DIR / "fuel_io.py"),
        },
        "cuda_process_count": 0,
    }
    result_path = output_dir / "result.json"
    fuel_io.atomic_json(result_path, result)
    fuel_io.readback_json(result_path, result)
    manifest = {
        "schema": SCHEMA,
        "result_json": str(result_path),
        "result_json_sha256": fuel_io.sha256_file(result_path),
        "oof_metrics_npz": str(oof_path),
        "oof_metrics_npz_sha256": fuel_io.sha256_file(oof_path),
        "verdict": adjudication["verdict"],
    }
    manifest_path = output_dir / "manifest.json"
    fuel_io.atomic_json(manifest_path, manifest)
    fuel_io.readback_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
