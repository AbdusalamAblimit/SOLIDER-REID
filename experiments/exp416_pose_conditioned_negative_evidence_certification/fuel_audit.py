#!/usr/bin/env python3
"""CPU-only OOF adjudicator for the once-only exp416 PC-NEC fuel audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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
BANK_SCHEMA = "exp416-pcnec-candidate-bank-v1"
GEOMETRY_SCHEMA = "exp416-pcnec-geometry-v1"
FUEL_SCHEMA = "exp416-pcnec-fuel-cache-v1"
EXPECTED_INTERPRETER = Path("/usr/local/anaconda3/envs/mmpose-abu/bin/python")
FIXED_REPOSITORY_ROOT = Path("/home/afr/SOLIDER-REID-exp416-pcnec-formal-v1")
FIXED_BANK = Path(
    "/home/afr/reid-clean/assets/exp416-pcnec-candidate-bank-v1/"
    "candidate_bank.npz"
)
FIXED_GEOMETRY_SUMMARY = Path(
    "/home/afr/reid-clean/assets/exp416-pcnec-geometry-v1/summary.json"
)
FIXED_FUEL_CACHE = Path(
    "/home/afr/reid-clean/assets/exp416-pcnec-fuel-v1/fuel_cache.npz"
)
FIXED_OUTPUT_DIR = Path(
    "/home/afr/reid-clean/assets/exp416-pcnec-audit-v1"
)
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
    "bank_sha256",
    "geometry_sha256",
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
SLOT_COUNT = 5
MIN_QUERY_COVERAGE = 0.80
MIN_COMMON_PAIRS_PER_SLOT = 100000
MIN_QUERY_PIDS_PER_SLOT = 300


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


def recompute_geometry_gate(bank, fuel, geometry, bank_sha256):
    if (
        fuel["bank_sha256"].shape != (1,)
        or fuel["geometry_sha256"].shape != (1,)
    ):
        raise RuntimeError("fuel provenance SHA fields must be scalar strings")
    fuel_bank_sha = str(fuel["bank_sha256"].item())
    fuel_geometry_sha = str(fuel["geometry_sha256"].item())
    if fuel_bank_sha != str(bank_sha256):
        raise RuntimeError("fuel cache bank SHA differs from current bank")
    if str(geometry.get("bank_sha256")) != str(bank_sha256):
        raise RuntimeError("geometry summary bank SHA differs from current bank")
    if str(geometry.get("geometry_npz_sha256")) != fuel_geometry_sha:
        raise RuntimeError("fuel cache geometry SHA differs from geometry summary")

    count = len(bank["relative_paths"])
    availability = np.asarray(fuel["availability"])
    if availability.dtype != np.bool_ or availability.shape != (
        count,
        SLOT_COUNT,
    ):
        raise RuntimeError("fuel availability schema mismatch")
    query = bank["query_indices"].astype(np.int64, copy=False)
    candidate = bank["candidate_indices"].astype(np.int64, copy=False)
    common = availability[query] & availability[candidate]
    if not np.array_equal(fuel["common"], common):
        raise RuntimeError("fuel common slots differ from current bank/availability")
    unique_queries = np.unique(query)
    covered_queries = np.unique(query[common.any(axis=1)])
    query_coverage = float(len(covered_queries) / len(unique_queries))
    pair_counts = common.sum(axis=0).astype(np.int64)
    raw_pids = bank["raw_pids"].astype(np.int64, copy=False)
    pid_counts = np.asarray(
        [
            len(np.unique(raw_pids[np.unique(query[common[:, slot]])]))
            for slot in range(SLOT_COUNT)
        ],
        dtype=np.int64,
    )
    coverage = geometry.get("coverage")
    if not isinstance(coverage, dict):
        raise RuntimeError("geometry summary coverage receipt is missing")
    if (
        float(coverage.get("query_coverage", -1.0)) != query_coverage
        or coverage.get("common_pair_count_by_slot") != pair_counts.tolist()
        or coverage.get("query_pid_count_by_slot") != pid_counts.tolist()
    ):
        raise RuntimeError(
            "geometry summary coverage differs from current bank/fuel"
        )
    gate = bool(
        query_coverage >= MIN_QUERY_COVERAGE
        and bool((pair_counts >= MIN_COMMON_PAIRS_PER_SLOT).all())
        and bool((pid_counts >= MIN_QUERY_PIDS_PER_SLOT).all())
    )
    if bool(geometry.get("geometry_gate_pass", False)) != gate:
        raise RuntimeError("geometry summary gate differs from recomputation")
    return {
        "geometry_gate_pass": gate,
        "query_coverage": query_coverage,
        "common_pair_count_by_slot": pair_counts.tolist(),
        "query_pid_count_by_slot": pid_counts.tolist(),
    }


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
        true_rows = candidate_index[~labels]
        expected_camera_quota = core._camera_matched_impostor_quota(
            true_rows,
            bank["camids"].astype(np.int64, copy=False),
            int(labels.sum()),
        )
        observed_camera_quota = {
            int(camera): int(count)
            for camera, count in zip(
                *np.unique(
                    bank["camids"][candidate_index[labels]],
                    return_counts=True,
                )
            )
        }
        if observed_camera_quota != expected_camera_quota:
            raise RuntimeError(
                "candidate bank impostor camera quota differs from genuine"
            )
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
    control_rows = {
        name: outputs[name]["rows"] for name in core.CONTROL_ORDER
    }
    bootstraps = [
        core.simultaneous_control_pid_bootstrap(
            outputs["correct"]["rows"],
            control_rows,
            metric=metric,
        )
        for metric in ("auroc", "average_precision")
    ]
    bootstraps.extend(
        core.paired_pid_bootstrap(
            outputs["correct"]["rows"],
            outputs["d0_only"]["rows"],
            metric=metric,
            control_name="d0_only",
        )
        for metric in ("mAP", "R1")
    )
    bootstraps.extend(
        core.simultaneous_control_pid_bootstrap(
            outputs["correct"]["rows"],
            control_rows,
            metric=metric,
        )
        for metric in ("mAP", "R1")
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
    mock_bank = {
        "relative_paths": np.asarray(("a", "b", "c"), dtype=np.str_),
        "raw_pids": np.asarray((10, 10, 20), dtype=np.int64),
        "query_indices": np.asarray((0, 0), dtype=np.int32),
        "candidate_indices": np.asarray((1, 2), dtype=np.int32),
    }
    mock_availability = np.asarray(
        (
            (1, 1, 1, 1, 1),
            (1, 1, 1, 1, 0),
            (1, 0, 0, 0, 0),
        ),
        dtype=np.bool_,
    )
    mock_common = mock_availability[[0, 0]] & mock_availability[[1, 2]]
    mock_fuel = {
        "bank_sha256": np.asarray(["b" * 64]),
        "geometry_sha256": np.asarray(["g" * 64]),
        "availability": mock_availability,
        "common": mock_common,
    }
    mock_geometry = {
        "bank_sha256": "b" * 64,
        "geometry_npz_sha256": "g" * 64,
        "geometry_gate_pass": False,
        "coverage": {
            "query_coverage": 1.0,
            "common_pair_count_by_slot": [2, 1, 1, 1, 0],
            "query_pid_count_by_slot": [1, 1, 1, 1, 0],
        },
    }
    receipt = recompute_geometry_gate(
        mock_bank, mock_fuel, mock_geometry, "b" * 64
    )
    assert receipt["geometry_gate_pass"] is False
    injected = dict(mock_geometry)
    injected["bank_sha256"] = "x" * 64
    try:
        recompute_geometry_gate(
            mock_bank, mock_fuel, injected, "b" * 64
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale geometry summary binding was accepted")
    print("EXP416_FUEL_AUDIT_SELF_TEST=PASS")


def _git_file_is_tracked(repository, path):
    result = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--error-unmatch",
            str(Path(path).resolve().relative_to(Path(repository).resolve())),
        ),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def validate_formal(args):
    if not args.formal:
        raise RuntimeError("non-self-test adjudication requires --formal")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("formal audit requires PYTHONDONTWRITEBYTECODE=1")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("formal audit requires PYTHONHASHSEED=0")
    if Path(sys.executable).resolve() != EXPECTED_INTERPRETER.resolve(strict=True):
        raise RuntimeError("formal audit interpreter mismatch")
    if REPOSITORY_ROOT.resolve(strict=True) != FIXED_REPOSITORY_ROOT:
        raise RuntimeError("formal audit repository path mismatch")
    exact_paths = {
        "bank": FIXED_BANK,
        "geometry_summary": FIXED_GEOMETRY_SUMMARY,
        "fuel_cache": FIXED_FUEL_CACHE,
        "output_dir": FIXED_OUTPUT_DIR,
    }
    for name, expected in exact_paths.items():
        if Path(getattr(args, name)).expanduser() != expected:
            raise RuntimeError("formal audit fixed path mismatch: " + name)
    if FIXED_OUTPUT_DIR.exists():
        raise FileExistsError("formal audit output must be fresh")
    if not FIXED_OUTPUT_DIR.parent.is_dir():
        raise NotADirectoryError(FIXED_OUTPUT_DIR.parent)
    for path in (FIXED_BANK, FIXED_GEOMETRY_SUMMARY, FIXED_FUEL_CACHE):
        path.resolve(strict=True)
    head = fuel_io.git_head(REPOSITORY_ROOT)
    if not args.expected_head or str(args.expected_head) != head:
        raise RuntimeError("formal audit HEAD mismatch")
    if fuel_io.git_tracked_status(REPOSITORY_ROOT):
        raise RuntimeError("formal audit tracked worktree is dirty")
    if fuel_io.git_index_status(REPOSITORY_ROOT):
        raise RuntimeError("formal audit index is dirty")
    expected_sources = {
        "fuel_audit.py": str(args.expected_fuel_audit_sha256),
        "fuel_core.py": str(args.expected_fuel_core_sha256),
        "fuel_io.py": str(args.expected_fuel_io_sha256),
    }
    for filename, expected in expected_sources.items():
        path = SCRIPT_DIR / filename
        if not expected or fuel_io.sha256_file(path) != expected:
            raise RuntimeError("formal audit source SHA mismatch: " + filename)
        if not _git_file_is_tracked(REPOSITORY_ROOT, path):
            raise RuntimeError("formal audit source is untracked: " + filename)
    fuel_io.assert_no_cuda_compute_processes()
    return {
        "head": head,
        "source_sha256": {
            filename: fuel_io.sha256_file(SCRIPT_DIR / filename)
            for filename in expected_sources
        },
    }


def _validate_upstream_receipts(
    *, bank_path, bank_sha256, geometry_path, geometry, fuel_path,
    fuel_sha256, formal_head
):
    bank_receipt_path = bank_path.parent / "receipt.json"
    bank_manifest_path = bank_path.parent / "manifest.json"
    bank_receipt = fuel_io.readback_json(bank_receipt_path)
    bank_manifest = fuel_io.readback_json(bank_manifest_path)
    if (
        bank_receipt.get("schema")
        != "exp416-pcnec-candidate-bank-receipt-v1"
        or bank_manifest.get("schema")
        != "exp416-pcnec-candidate-bank-manifest-v1"
        or bank_manifest.get("files", {})
        .get("candidate_bank.npz", {})
        .get("sha256")
        != str(bank_sha256)
        or bank_manifest.get("files", {}).get("receipt.json", {}).get("sha256")
        != fuel_io.sha256_file(bank_receipt_path)
        or bank_receipt.get("provenance", {}).get("formal_head")
        != formal_head
    ):
        raise RuntimeError("candidate artifact provenance binding mismatch")

    geometry_manifest_path = geometry_path.parent / "manifest.json"
    geometry_manifest = fuel_io.readback_json(geometry_manifest_path)
    if (
        geometry.get("formal_head") != formal_head
        or geometry_manifest.get("formal_head") != formal_head
        or geometry_manifest.get("geometry_npz_sha256")
        != geometry.get("geometry_npz_sha256")
        or geometry_manifest.get("summary_json_sha256")
        != fuel_io.sha256_file(geometry_path)
    ):
        raise RuntimeError("geometry artifact provenance binding mismatch")

    fuel_receipt_path = fuel_path.parent / "receipt.json"
    fuel_manifest_path = fuel_path.parent / "manifest.json"
    fuel_receipt = fuel_io.readback_json(fuel_receipt_path)
    fuel_manifest = fuel_io.readback_json(fuel_manifest_path)
    if (
        fuel_receipt.get("formal_head") != formal_head
        or fuel_receipt.get("bank_sha256") != str(bank_sha256)
        or fuel_receipt.get("geometry_sha256")
        != geometry.get("geometry_npz_sha256")
        or fuel_manifest.get("formal_head") != formal_head
        or fuel_manifest.get("files", {})
        .get("fuel_cache.npz", {})
        .get("sha256")
        != str(fuel_sha256)
        or fuel_manifest.get("files", {}).get("receipt.json", {}).get("sha256")
        != fuel_io.sha256_file(fuel_receipt_path)
    ):
        raise RuntimeError("fuel artifact provenance binding mismatch")


def _write_failure(output_dir, *, stage, error, validated):
    failure = {
        "schema": "exp416-pcnec-audit-failure-v1",
        "status": "FAILED",
        "stage": str(stage),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "resume_allowed": False,
        "formal_head": validated["head"],
        "source_sha256": validated["source_sha256"],
    }
    path = output_dir / "failure.json"
    if not path.exists() and not path.with_name(path.name + ".tmp").exists():
        fuel_io.atomic_json(path, failure)
        fuel_io.readback_json(path, failure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--bank")
    parser.add_argument("--bank-sha256")
    parser.add_argument("--geometry-summary")
    parser.add_argument("--geometry-summary-sha256")
    parser.add_argument("--fuel-cache")
    parser.add_argument("--fuel-cache-sha256")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-fuel-audit-sha256")
    parser.add_argument("--expected-fuel-core-sha256")
    parser.add_argument("--expected-fuel-io-sha256")
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
        "expected_head",
        "expected_fuel_audit_sha256",
        "expected_fuel_core_sha256",
        "expected_fuel_io_sha256",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError("missing formal arguments: " + ",".join(missing))
    validated = validate_formal(args)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute() or output_dir.exists():
        raise FileExistsError("audit output directory must be fresh and absolute")
    output_dir.mkdir(mode=0o755, parents=False)
    started = {
        "schema": "exp416-pcnec-audit-started-v1",
        "status": "STARTED",
        "resume_allowed": False,
        "formal_head": validated["head"],
        "source_sha256": validated["source_sha256"],
        "inputs": {
            "bank": str(args.bank),
            "bank_sha256": str(args.bank_sha256),
            "geometry_summary": str(args.geometry_summary),
            "geometry_summary_sha256": str(args.geometry_summary_sha256),
            "fuel_cache": str(args.fuel_cache),
            "fuel_cache_sha256": str(args.fuel_cache_sha256),
        },
    }
    stage = "started_write"
    try:
        started_path = output_dir / "started.json"
        fuel_io.atomic_json(started_path, started)
        fuel_io.readback_json(started_path, started)
        stage = "input_read_and_binding"
        bank_path, bank = _load_npz_bound(
            args.bank, args.bank_sha256, BANK_FIELDS, BANK_SCHEMA
        )
        geometry_path, geometry = _load_geometry_summary(
            args.geometry_summary, args.geometry_summary_sha256
        )
        fuel_path, fuel = _load_npz_bound(
            args.fuel_cache, args.fuel_cache_sha256, FUEL_FIELDS, FUEL_SCHEMA
        )
        _validate_upstream_receipts(
            bank_path=bank_path,
            bank_sha256=args.bank_sha256,
            geometry_path=geometry_path,
            geometry=geometry,
            fuel_path=fuel_path,
            fuel_sha256=args.fuel_cache_sha256,
            formal_head=validated["head"],
        )
        geometry_receipt = recompute_geometry_gate(
            bank, fuel, geometry, args.bank_sha256
        )
        stage = "query_payload_construction"
        queries = build_query_payloads(bank, fuel)
        stage = "oof_and_bootstrap"
        outputs, summaries, strongest = evaluate_all_arms(queries)
        adjudication = adjudicate(
            outputs,
            summaries,
            strongest,
            geometry_receipt["geometry_gate_pass"],
        )
        oof_arrays = _serializable_oof(outputs)
        stage = "result_write"
        oof_path = output_dir / "oof_metrics.npz"
        fuel_io.atomic_npz(oof_path, oof_arrays)
        fuel_io.readback_npz(oof_path, oof_arrays)
        result = {
            "schema": SCHEMA,
            "verdict": adjudication["verdict"],
            "go": adjudication["go"],
            "formal_head": validated["head"],
            "inputs": {
                "bank": str(bank_path),
                "bank_sha256": str(args.bank_sha256),
                "geometry_summary": str(geometry_path),
                "geometry_summary_sha256": str(
                    args.geometry_summary_sha256
                ),
                "fuel_cache": str(fuel_path),
                "fuel_cache_sha256": str(args.fuel_cache_sha256),
            },
            "counts": {
                "train_rows": int(len(bank["relative_paths"])),
                "pair_rows": int(len(bank["query_indices"])),
                "fixed_queries": int(len(queries)),
                "query_pids": int(
                    len({query["query_pid"] for query in queries})
                ),
                "undecided_pairs": int(fuel["undecided"].sum()),
                "wrong_donor_invalid_pairs": int(
                    fuel["wrong_donor_invalid"].sum()
                ),
            },
            "geometry": geometry_receipt,
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
            "source_sha256": validated["source_sha256"],
            "cuda_process_count": 0,
        }
        result_path = output_dir / "result.json"
        fuel_io.atomic_json(result_path, result)
        fuel_io.readback_json(result_path, result)
        manifest = {
            "schema": SCHEMA,
            "formal_head": validated["head"],
            "source_sha256": validated["source_sha256"],
            "files": {
                name: {
                    "bytes": int((output_dir / name).stat().st_size),
                    "sha256": fuel_io.sha256_file(output_dir / name),
                }
                for name in (
                    "started.json",
                    "oof_metrics.npz",
                    "result.json",
                )
            },
            "verdict": adjudication["verdict"],
            "resume_allowed": False,
        }
        manifest_path = output_dir / "manifest.json"
        fuel_io.atomic_json(manifest_path, manifest)
        fuel_io.readback_json(manifest_path, manifest)
        fuel_io.seal_directory(output_dir)
    except BaseException as error:
        try:
            _write_failure(
                output_dir,
                stage=stage,
                error=error,
                validated=validated,
            )
            fuel_io.seal_directory(output_dir)
        except BaseException:
            pass
        raise
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
