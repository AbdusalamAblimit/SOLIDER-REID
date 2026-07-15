"""Pure-CPU formal preflight for the exp374 publication state machine.

The tests use only temporary directories, two tiny synthetic arms, JSON, and
small NumPy arrays.  They do not construct a model, read a dataset or
checkpoint, access an accelerator, or invoke either full execution phase.

Writing this file does not authorize executing it.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from experiments.exp374_psg_fuel_gate import audit_gate_a as runner
from experiments.exp374_psg_fuel_gate import protocol


ROWS = (
    {"seed": 42, "arm": "bypass"},
    {"seed": 1234, "arm": "bypass"},
)


def _assert_gate_code(expected_code, callable_, *args, **kwargs):
    with pytest.raises(protocol.GateProtocolError) as caught:
        callable_(*args, **kwargs)
    assert caught.value.code == expected_code


def _provenance(row):
    return {
        "execution_sha256": "synthetic-execution",
        "checkpoint_sha256": f"checkpoint-{row['seed']}",
        "checkpoint_state_audit_sha256": f"state-{row['seed']}",
        "config_file_sha256": "config-file",
        "resolved_config_sha256": "resolved-config",
        "prepared_artifact_manifest_sha256": "prepared",
        "row": dict(row),
        "arm_id": runner.schedule_arm_id(row),
        "mapping": {},
    }


def _publish_synthetic_arm(execution_dir: Path, row):
    arms_root = execution_dir / "arms"
    arms_root.mkdir(exist_ok=True)
    arm_id = runner.schedule_arm_id(row)
    temporary = execution_dir / f".{arm_id}.tmp-arm"
    published = arms_root / arm_id
    provenance = _provenance(row)
    temporary.mkdir()
    protocol.atomic_write_json(temporary / "summary.json", {
        "status": "PASS",
        "row": dict(row),
        "arm_id": arm_id,
        "mapping": {},
        "provenance": provenance,
    })
    np.savez(
        temporary / "per_query.npz",
        AP=np.asarray([0.25, 0.75], dtype=np.float64),
        R1_indicator=np.asarray([0.0, 1.0], dtype=np.float64),
        margin=np.asarray([-0.1, 0.2], dtype=np.float64),
    )
    runner.publish_arm(temporary, published, provenance)
    return published, provenance


def _publish_two_arms(execution_dir: Path):
    published = [_publish_synthetic_arm(execution_dir, row) for row in ROWS]
    arm_ids = [runner.schedule_arm_id(row) for row in ROWS]
    return published, arm_ids


def _snapshot(directory: Path):
    return {
        str(path.relative_to(directory)): (
            protocol.sha256_file(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(value for value in directory.rglob("*") if value.is_file())
    }


def _results_payload():
    return {
        "schema": "synthetic-exp374-results",
        "decision": "GO",
        "per_seed": {42: {"effect": 1.0}, 1234: {"effect": 1.1}},
    }


def _aggregate_arrays():
    return {
        "seed_42_correct_AP": np.asarray([0.5, 0.6], dtype=np.float64),
        "seed_1234_correct_AP": np.asarray([0.4, 0.7], dtype=np.float64),
    }


def _resign_results(results_dir: Path):
    hashes = runner.arm_artifact_hashes(results_dir)
    protocol.atomic_write_json(results_dir / "COMPLETE.json", {
        "status": "PASS",
        "artifact_sha256": hashes,
    })
    return hashes


def test_exclusive_lock_rejects_competitor_without_releasing_owner(tmp_path):
    lock = tmp_path / "RUN.lock"
    with runner.exclusive_execution_lock(tmp_path, "owner"):
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload == {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "phase": "owner",
        }
        with pytest.raises(FileExistsError):
            with runner.exclusive_execution_lock(tmp_path, "competitor"):
                pass
        assert lock.is_file()
        assert json.loads(lock.read_text(encoding="utf-8")) == payload

    assert not lock.exists()
    with runner.exclusive_execution_lock(tmp_path, "reacquired"):
        assert lock.is_file()
    assert not lock.exists()


def test_exclusive_lock_releases_after_body_and_entry_failures(tmp_path):
    lock = tmp_path / "RUN.lock"
    with pytest.raises(LookupError, match="synthetic body failure"):
        with runner.exclusive_execution_lock(tmp_path, "body-failure"):
            raise LookupError("synthetic body failure")
    assert not lock.exists()

    with mock.patch.object(
        runner.os,
        "write",
        side_effect=OSError("synthetic entry failure"),
    ):
        with pytest.raises(OSError, match="synthetic entry failure"):
            with runner.exclusive_execution_lock(tmp_path, "entry-failure"):
                pass
    assert not lock.exists()

    with runner.exclusive_execution_lock(tmp_path, "after-failures"):
        assert lock.is_file()
    assert not lock.exists()


def test_two_arm_run_completion_binds_artifacts_markers_and_manifest(tmp_path):
    published, arm_ids = _publish_two_arms(tmp_path)
    for arm_dir, provenance in published:
        marker = runner.verify_published_arm(arm_dir, provenance)
        assert marker["artifact_sha256"] == runner.arm_artifact_hashes(arm_dir)

    arm_manifest, run_payload = runner.publish_run_completion(tmp_path, arm_ids)

    for arm_id in arm_ids:
        marker_path = tmp_path / "arms" / arm_id / "COMPLETE.json"
        assert arm_manifest[arm_id] == protocol.sha256_file(marker_path)
    assert run_payload == {
        "status": "PASS",
        "published_arms": 2,
        "metrics_summarized": False,
        "arm_manifest_sha256": protocol.sha256_bytes(
            protocol.canonical_json_bytes(arm_manifest)),
    }
    assert runner.verify_run_completion(tmp_path, arm_ids) == arm_manifest


def test_two_level_hash_chain_rejects_artifact_and_marker_mutation(tmp_path):
    published, arm_ids = _publish_two_arms(tmp_path)
    runner.publish_run_completion(tmp_path, arm_ids)
    first_dir, first_provenance = published[0]

    (first_dir / "per_query.npz").write_bytes(b"mutated artifact")
    _assert_gate_code(
        "E_ARM_HASH",
        runner.verify_published_arm,
        first_dir,
        first_provenance,
    )
    assert runner.verify_run_completion(tmp_path, arm_ids)

    marker_path = first_dir / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact_sha256"] = runner.arm_artifact_hashes(first_dir)
    protocol.atomic_write_json(marker_path, marker)
    _assert_gate_code(
        "E_RUN_ARM_MANIFEST",
        runner.verify_run_completion,
        tmp_path,
        arm_ids,
    )


@pytest.mark.parametrize(
    "missing_name",
    ["RUN_COMPLETE.json", "RUN_ARM_MANIFEST.json"],
)
def test_run_completion_rejects_missing_marker(tmp_path, missing_name):
    _published, arm_ids = _publish_two_arms(tmp_path)
    runner.publish_run_completion(tmp_path, arm_ids)
    (tmp_path / missing_name).unlink()

    _assert_gate_code(
        "E_RUN_INCOMPLETE",
        runner.verify_run_completion,
        tmp_path,
        arm_ids,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "FAILED"),
        ("published_arms", 3),
        ("metrics_summarized", True),
        ("arm_manifest_sha256", "wrong-manifest"),
    ],
)
def test_run_completion_rejects_payload_drift(tmp_path, field, value):
    _published, arm_ids = _publish_two_arms(tmp_path)
    runner.publish_run_completion(tmp_path, arm_ids)
    marker_path = tmp_path / "RUN_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = value
    protocol.atomic_write_json(marker_path, marker)

    _assert_gate_code(
        "E_RUN_INCOMPLETE",
        runner.verify_run_completion,
        tmp_path,
        arm_ids,
    )


def test_run_completion_rejects_manifest_ids_and_arm_marker_drift(tmp_path):
    _published, arm_ids = _publish_two_arms(tmp_path)
    runner.publish_run_completion(tmp_path, arm_ids)
    manifest_path = tmp_path / "RUN_ARM_MANIFEST.json"
    run_path = tmp_path / "RUN_COMPLETE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(arm_ids[-1])
    protocol.atomic_write_json(manifest_path, manifest)
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    run_payload["arm_manifest_sha256"] = protocol.sha256_bytes(
        protocol.canonical_json_bytes(manifest))
    protocol.atomic_write_json(run_path, run_payload)

    _assert_gate_code(
        "E_RUN_ARM_MANIFEST",
        runner.verify_run_completion,
        tmp_path,
        arm_ids,
    )

    manifest, _payload = runner.publish_run_completion(tmp_path, arm_ids)
    assert set(manifest) == set(arm_ids)
    marker_path = tmp_path / "arms" / arm_ids[0] / "COMPLETE.json"
    marker_path.write_bytes(marker_path.read_bytes() + b" ")
    _assert_gate_code(
        "E_RUN_ARM_MANIFEST",
        runner.verify_run_completion,
        tmp_path,
        arm_ids,
    )


def test_published_arm_resume_verification_is_read_only(tmp_path):
    arm_dir, provenance = _publish_synthetic_arm(tmp_path, ROWS[0])
    before = _snapshot(arm_dir)

    first = runner.verify_published_arm(arm_dir, provenance)
    second = runner.verify_published_arm(arm_dir, provenance)

    assert first == second
    assert _snapshot(arm_dir) == before


def test_published_arm_resume_rejects_corruption_and_semantic_resigning(tmp_path):
    arm_dir, provenance = _publish_synthetic_arm(tmp_path, ROWS[0])
    summary_path = arm_dir / "summary.json"
    summary_path.write_bytes(b"{}\n")
    _assert_gate_code(
        "E_ARM_HASH", runner.verify_published_arm, arm_dir, provenance)

    summary = {
        "status": "PASS",
        "row": {"seed": 2024, "arm": "bypass"},
        "arm_id": provenance["arm_id"],
        "mapping": {},
        "provenance": provenance,
    }
    protocol.atomic_write_json(summary_path, summary)
    marker_path = arm_dir / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact_sha256"] = runner.arm_artifact_hashes(arm_dir)
    protocol.atomic_write_json(marker_path, marker)
    _assert_gate_code(
        "E_ARM_PROVENANCE",
        runner.verify_published_arm,
        arm_dir,
        provenance,
    )


def test_published_arm_resume_rejects_missing_complete_marker(tmp_path):
    arm_dir, provenance = _publish_synthetic_arm(tmp_path, ROWS[0])
    (arm_dir / "COMPLETE.json").unlink()

    _assert_gate_code(
        "E_ARM_INCOMPLETE",
        runner.verify_published_arm,
        arm_dir,
        provenance,
    )


def test_results_publish_survives_crash_and_is_reused_without_writes(tmp_path):
    results = _results_payload()
    arrays = _aggregate_arrays()
    hashes = runner.publish_or_verify_results(tmp_path, results, arrays)
    results_dir = tmp_path / "results"

    assert runner.verify_results_directory(results_dir)["artifact_sha256"] == hashes
    assert not (tmp_path / "COMPLETE").exists()
    before = _snapshot(results_dir)
    with mock.patch.object(
        runner,
        "publish_directory",
        side_effect=AssertionError("must reuse published results"),
    ):
        reused = runner.publish_or_verify_results(tmp_path, results, arrays)

    assert reused == hashes
    assert _snapshot(results_dir) == before
    assert not (tmp_path / "COMPLETE").exists()


def test_results_reuse_rejects_unsigned_and_self_signed_json_drift(tmp_path):
    results = _results_payload()
    arrays = _aggregate_arrays()
    runner.publish_or_verify_results(tmp_path, results, arrays)
    results_dir = tmp_path / "results"
    result_path = results_dir / "gate_a_results.json"
    protocol.atomic_write_json(result_path, {"decision": "NO_GO"})

    _assert_gate_code(
        "E_RESULTS_HASH",
        runner.publish_or_verify_results,
        tmp_path,
        results,
        arrays,
    )

    _resign_results(results_dir)
    assert runner.verify_results_directory(results_dir)
    _assert_gate_code(
        "E_RESULTS_DRIFT",
        runner.publish_or_verify_results,
        tmp_path,
        results,
        arrays,
    )


def test_results_reuse_rejects_self_signed_npz_key_and_value_drift(tmp_path):
    results = _results_payload()
    arrays = _aggregate_arrays()
    runner.publish_or_verify_results(tmp_path, results, arrays)
    results_dir = tmp_path / "results"
    aggregate_path = results_dir / "primary_query_aggregates.npz"

    np.savez(aggregate_path, unexpected=np.asarray([1.0], dtype=np.float64))
    _resign_results(results_dir)
    _assert_gate_code(
        "E_RESULTS_DRIFT",
        runner.publish_or_verify_results,
        tmp_path,
        results,
        arrays,
    )

    np.savez(
        aggregate_path,
        seed_42_correct_AP=np.asarray([9.0, 9.0], dtype=np.float64),
        seed_1234_correct_AP=arrays["seed_1234_correct_AP"],
    )
    _resign_results(results_dir)
    _assert_gate_code(
        "E_RESULTS_DRIFT",
        runner.publish_or_verify_results,
        tmp_path,
        results,
        arrays,
    )


def test_failure_manifest_after_results_publish_is_nonreportable_and_non_destructive(
    tmp_path,
):
    results = _results_payload()
    arrays = _aggregate_arrays()
    hashes = runner.publish_or_verify_results(tmp_path, results, arrays)
    results_before = _snapshot(tmp_path / "results")
    args = SimpleNamespace(execution_dir=str(tmp_path), phase="summarize")

    runner._write_failure_manifest(
        args,
        protocol.GateProtocolError("E_SYNTHETIC", "after results publish"),
    )

    failure_path = tmp_path / "FAILED.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED_NONREPORTABLE"
    assert failure["phase"] == "summarize"
    assert failure["error_code"] == "E_SYNTHETIC"
    assert failure["failed_arm_published"] is False
    assert _snapshot(tmp_path / "results") == results_before
    assert runner.verify_results_directory(
        tmp_path / "results")["artifact_sha256"] == hashes

    sentinel = b'{"sentinel":"preserve-after-complete"}\n'
    failure_path.write_bytes(sentinel)
    protocol.atomic_write_json(tmp_path / "COMPLETE", {"status": "COMPLETE"})
    runner._write_failure_manifest(
        args,
        protocol.GateProtocolError("E_LATE", "must not overwrite"),
    )
    assert failure_path.read_bytes() == sentinel
