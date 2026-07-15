"""Pure synthetic unit tests for the exp374 Gate-A audit runner.

These tests intentionally use only temporary directories, tiny byte payloads,
and small NumPy/Torch objects.  They must remain independent of the ReID
dataset, real checkpoints, CUDA, inference, and training.

The file is written for a later, separately authorized test phase.  Creating
it does not authorize executing it.

The full 492-arm ``summarize_phase`` recovery path is deliberately excluded:
it remains an execution preflight because exercising it here would duplicate a
heavy formal schedule.  The lightweight result hash, normalization, semantic
comparison, and crash-state primitives are covered below.
"""

from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from experiments.exp374_psg_fuel_gate import audit_gate_a as runner
from experiments.exp374_psg_fuel_gate import protocol


class GateCodeMixin:
    """Assert a fail-closed protocol error without depending on its message."""

    def assert_gate_code(self, expected_code, callable_, *args, **kwargs):
        with self.assertRaises(protocol.GateProtocolError) as caught:
            callable_(*args, **kwargs)
        self.assertEqual(caught.exception.code, expected_code)


class FlatLogAndCheckpointBytesTests(GateCodeMixin, unittest.TestCase):
    @staticmethod
    def synthetic_alias_state(offset):
        state = {}
        for block_index, key in enumerate(("s3_b0", "s3_b1")):
            for suffix_index, suffix in enumerate((
                "encoder.0.weight",
                "encoder.0.bias",
                "encoder.2.weight",
                "encoder.2.bias",
            )):
                value = torch.tensor(
                    [float(offset + 10 * block_index + suffix_index)],
                    dtype=torch.float32,
                )
                state[f"psg_modules_dict.{key}.{suffix}"] = value
                state[f"psg_modules.{block_index}.{suffix}"] = value.clone()
        return state

    def test_flat_log_parser_accepts_one_pair_and_ignores_rank10(self):
        payload = (
            b"Validation Results\n"
            b"mAP: 58.3%\n"
            b"CMC curve, Rank-1: 68.1%\n"
            b"Rank-10: 84.2%\n"
        )

        parsed = runner.parse_flat_log_metrics_bytes(payload, "synthetic.log")

        self.assertEqual(parsed, {
            "mAP": 58.3,
            "R1": 68.1,
            "mAP_occurrences": 1,
            "R1_occurrences": 1,
        })

    def test_flat_log_parser_rejects_missing_metric(self):
        self.assert_gate_code(
            "E_FLAT_LOG_PARSE",
            runner.parse_flat_log_metrics_bytes,
            b"mAP: 58.3%\n",
            "missing-rank1.log",
        )

    def test_flat_log_parser_rejects_duplicate_metric(self):
        payload = b"mAP: 58.3%\nmAP: 58.4%\nRank-1: 68.1%\n"
        self.assert_gate_code(
            "E_FLAT_LOG_AMBIGUOUS",
            runner.parse_flat_log_metrics_bytes,
            payload,
            "duplicate-map.log",
        )

    def test_flat_log_parser_rejects_out_of_range_metric(self):
        self.assert_gate_code(
            "E_FLAT_LOG_PARSE",
            runner.parse_flat_log_metrics_bytes,
            b"mAP: 101.0%\nRank-1: 68.1%\n",
            "out-of-range.log",
        )

    def test_flat_log_parser_uses_strict_utf8(self):
        with self.assertRaises(UnicodeDecodeError):
            runner.parse_flat_log_metrics_bytes(
                b"mAP: 58.3%\nRank-1: 68.1%\n\xff",
                "non-utf8.log",
            )

    def test_checkpoint_key_normalization_rejects_collision(self):
        state = {
            "weight": torch.tensor([1.0]),
            "module.weight": torch.tensor([1.0]),
        }

        self.assert_gate_code(
            "E_CHECKPOINT_KEY_COLLISION",
            runner._normalized_checkpoint_state,
            state,
        )

    def test_checkpoint_bytes_loader_and_normalizer_use_synthetic_payload(self):
        buffer = io.BytesIO()
        torch.save(
            {"state_dict": {"module.weight": torch.tensor([1.0, 2.0])}},
            buffer,
        )

        raw = runner._torch_load_checkpoint_bytes(buffer.getvalue())
        state = runner._normalized_checkpoint_state(raw)

        self.assertEqual(set(state), {"weight"})
        self.assertTrue(torch.equal(state["weight"], torch.tensor([1.0, 2.0])))

    def test_checkpoint_state_rejects_non_tensor_value(self):
        self.assert_gate_code(
            "E_CHECKPOINT_FORMAT",
            runner._normalized_checkpoint_state,
            {"weight": "not-a-tensor"},
        )

    def test_default_checkpoint_logs_bind_flat_checkpoint_evaluations(self):
        expected = {
            seed: (
                "/home/afr/SOLIDER-REID/log/multiseed/"
                f"exp007_psg_seed{seed}/test_default.txt"
            )
            for seed in (42, 1234, 2024)
        }

        actual = {
            int(spec["seed"]): str(spec["flat_log"])
            for spec in runner.DEFAULT_CHECKPOINTS
        }

        self.assertEqual(actual, expected)
        self.assertTrue(all(Path(path).name == "test_default.txt" for path in actual.values()))

    def test_checkpoint_specs_freezes_sha_log_parse_and_aliases_from_one_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_specs = []
            asset_paths = []
            for offset, seed in enumerate((42, 1234, 2024)):
                weight = root / f"seed_{seed}.pth"
                flat_log = root / f"seed_{seed}_flat.log"
                train_log = root / f"seed_{seed}_train.log"
                torch.save(self.synthetic_alias_state(offset), weight)
                expected_map = 50.0 + offset
                expected_r1 = 60.0 + offset
                flat_log.write_bytes(
                    f"mAP: {expected_map:.1f}%\nRank-1: {expected_r1:.1f}%\n".encode())
                train_log.write_bytes(f"synthetic seed={seed}\n".encode())
                source_specs.append({
                    "seed": seed,
                    "weight": str(weight),
                    "weight_sha256": protocol.sha256_bytes(weight.read_bytes()),
                    "flat_log": str(flat_log),
                    "train_log": str(train_log),
                    "expected_mAP": expected_map,
                    "expected_R1": expected_r1,
                })
                asset_paths.extend((weight.resolve(), flat_log.resolve(), train_log.resolve()))
            checkpoint_manifest = root / "checkpoint_manifest.json"
            checkpoint_manifest.write_text(json.dumps(source_specs), encoding="utf-8")

            original_read_bytes = Path.read_bytes
            read_counts = {str(path): 0 for path in asset_paths}

            def counted_read_bytes(path):
                resolved = str(path.resolve())
                if resolved in read_counts:
                    read_counts[resolved] += 1
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", new=counted_read_bytes):
                frozen = runner.checkpoint_specs(str(checkpoint_manifest))

            self.assertEqual([int(spec["seed"]) for spec in frozen], [42, 1234, 2024])
            self.assertTrue(all(count == 1 for count in read_counts.values()))
            for spec in frozen:
                self.assertEqual(spec["flat_log_metrics"]["mAP_occurrences"], 1)
                self.assertEqual(spec["flat_log_metrics"]["R1_occurrences"], 1)
                self.assertEqual(len(spec["psg_alias_audit"]), 8)


class ScheduleAndArmIdTests(GateCodeMixin, unittest.TestCase):
    def test_frozen_schedule_has_492_unique_arm_ids(self):
        schedule = protocol.core_schedule([42, 1234, 2024])

        self.assertEqual(len(schedule), 492)
        self.assertEqual(
            len({runner.schedule_arm_id(row) for row in schedule}),
            492,
        )
        for seed in (42, 1234, 2024):
            rows = [row for row in schedule if int(row["seed"]) == seed]
            self.assertEqual(len(rows), 164)
            self.assertEqual(rows[0], {
                "seed": seed, "arm": "correct", "position": "start",
            })
            self.assertEqual(rows[-1], {
                "seed": seed, "arm": "correct", "position": "end",
            })
            self.assertEqual(sum(row["arm"] == "shuffle" for row in rows), 20)
            self.assertEqual(sum(row["arm"] == "centroid" for row in rows), 1)
            self.assertEqual(sum(row["arm"] == "bypass" for row in rows), 1)
            self.assertEqual(sum(row["arm"] == "group" for row in rows), 140)

    def test_arm_ids_follow_frozen_format(self):
        cases = (
            ({"seed": 42, "arm": "correct", "position": "start"},
             "seed_42__correct_start"),
            ({"seed": 42, "arm": "shuffle", "mapping": 3},
             "seed_42__shuffle_m03"),
            ({"seed": 42, "arm": "centroid"}, "seed_42__centroid"),
            ({"seed": 42, "arm": "bypass"}, "seed_42__bypass"),
            ({"seed": 42, "arm": "group", "group": "head", "mapping": 19},
             "seed_42__group_head_m19"),
        )

        for row, expected in cases:
            with self.subTest(row=row):
                self.assertEqual(runner.schedule_arm_id(row), expected)

    def test_schedule_rejects_wrong_seed_count(self):
        self.assert_gate_code(
            "E_SCHEDULE_SEEDS",
            protocol.core_schedule,
            [42, 1234],
        )


class AtomicPublishTests(GateCodeMixin, unittest.TestCase):
    def test_atomic_write_replaces_complete_payload_without_temp_residue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "state.json"
            target.write_bytes(b"old")

            protocol.atomic_write_bytes(target, b"new-complete-payload")

            self.assertEqual(target.read_bytes(), b"new-complete-payload")
            self.assertEqual(list(root.glob(f".{target.name}.*.tmp")), [])

    def test_publish_directory_is_atomic_and_create_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / ".arm.tmp"
            published = root / "arm"
            temporary.mkdir()
            (temporary / "a.bin").write_bytes(b"a")
            (temporary / "b.bin").write_bytes(b"b")

            protocol.publish_directory(temporary, published)

            self.assertFalse(temporary.exists())
            self.assertEqual((published / "a.bin").read_bytes(), b"a")
            self.assertEqual((published / "b.bin").read_bytes(), b"b")

            second = root / ".second.tmp"
            second.mkdir()
            self.assert_gate_code(
                "E_PUBLISH_EXISTS",
                protocol.publish_directory,
                second,
                published,
            )
            self.assertTrue(second.exists())

    def test_publish_directory_keeps_temporary_on_rename_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / ".arm.tmp"
            published = root / "arm"
            temporary.mkdir()
            (temporary / "payload.bin").write_bytes(b"payload")

            with mock.patch.object(protocol.os, "replace", side_effect=OSError("crash")):
                with self.assertRaises(OSError):
                    protocol.publish_directory(temporary, published)

            self.assertTrue(temporary.is_dir())
            self.assertFalse(published.exists())


class ArmProvenanceTests(GateCodeMixin, unittest.TestCase):
    @staticmethod
    def provenance(row):
        mapping = {}
        if row["arm"] in {"shuffle", "group"}:
            mapping = {
                "query_mapping_sha256": "synthetic-query-mapping",
                "gallery_mapping_sha256": "synthetic-gallery-mapping",
            }
        return {
            "execution_sha256": "execution",
            "checkpoint_sha256": "checkpoint",
            "checkpoint_state_audit_sha256": "state-audit",
            "config_file_sha256": "config-file",
            "resolved_config_sha256": "resolved-config",
            "prepared_artifact_manifest_sha256": "prepared",
            "row": dict(row),
            "arm_id": runner.schedule_arm_id(row),
            "mapping": mapping,
        }

    @staticmethod
    def write_summary(temporary, row, provenance, status):
        protocol.atomic_write_json(temporary / "summary.json", {
            "status": status,
            "row": dict(row),
            "arm_id": runner.schedule_arm_id(row),
            "mapping": dict(provenance["mapping"]),
            "provenance": dict(provenance),
        })

    @staticmethod
    def write_per_query(temporary):
        np.savez(
            temporary / "per_query.npz",
            AP=np.asarray([0.25, 0.75], dtype=np.float64),
            R1_indicator=np.asarray([0.0, 1.0], dtype=np.float64),
            margin=np.asarray([-0.1, 0.2], dtype=np.float64),
        )

    def publish_synthetic_arm(
        self,
        root,
        row,
        status="PASS",
        include_metrics=True,
        include_actual=False,
    ):
        provenance = self.provenance(row)
        temporary = root / f".{runner.schedule_arm_id(row)}.tmp-arm"
        published = root / runner.schedule_arm_id(row)
        temporary.mkdir()
        self.write_summary(temporary, row, provenance, status)
        if include_metrics:
            self.write_per_query(temporary)
        if include_actual:
            np.save(
                temporary / "actual_psg_input.npy",
                np.zeros((2, 17, 12, 4), dtype=np.float32),
                allow_pickle=False,
            )
        runner.publish_arm(
            temporary,
            published,
            provenance,
            status=status,
        )
        return published, provenance

    def test_pass_arm_requires_exact_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"seed": 42, "arm": "shuffle", "mapping": 0}
            published, provenance = self.publish_synthetic_arm(root, row)

            marker = runner.verify_published_arm(published, provenance)
            self.assertEqual(marker["status"], "PASS")

            wrong = dict(provenance)
            wrong["checkpoint_sha256"] = "another-checkpoint"
            self.assert_gate_code(
                "E_ARM_PROVENANCE",
                runner.verify_published_arm,
                published,
                wrong,
            )

    def test_expected_provenance_binds_selected_query_and_gallery_mapping_rows(self):
        query_mappings = np.arange(20 * 3, dtype=np.int32).reshape(20, 3)
        gallery_mappings = np.arange(20 * 4, dtype=np.int32).reshape(20, 4)
        scenes = SimpleNamespace(
            query_mappings=query_mappings,
            gallery_mappings=gallery_mappings,
        )
        manifest = {
            "config_file_sha256": "config-file",
            "resolved_config_sha256": "resolved-config",
            "prepared_artifact_sha256": {"prepared.bin": "prepared-sha"},
        }
        spec = {
            "weight_sha256": "checkpoint",
            "psg_alias_audit": {"canonical": "alias"},
        }
        row = {"seed": 42, "arm": "shuffle", "mapping": 7}

        provenance = runner.expected_arm_provenance(manifest, spec, row, scenes)

        self.assertEqual(provenance["mapping"], {
            "query_mapping_sha256": runner._array_sha256(query_mappings[7]),
            "gallery_mapping_sha256": runner._array_sha256(gallery_mappings[7]),
        })
        other = runner.expected_arm_provenance(
            manifest,
            spec,
            {"seed": 42, "arm": "shuffle", "mapping": 8},
            scenes,
        )
        self.assertNotEqual(provenance["mapping"], other["mapping"])

    def test_summary_mapping_drift_is_rejected_even_when_artifact_hash_is_self_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"seed": 42, "arm": "shuffle", "mapping": 0}
            provenance = self.provenance(row)
            temporary = root / f".{runner.schedule_arm_id(row)}.tmp-arm"
            published = root / runner.schedule_arm_id(row)
            temporary.mkdir()
            protocol.atomic_write_json(temporary / "summary.json", {
                "status": "PASS",
                "row": dict(row),
                "arm_id": runner.schedule_arm_id(row),
                "mapping": {
                    "query_mapping_sha256": "wrong-query",
                    "gallery_mapping_sha256": "wrong-gallery",
                },
                "provenance": dict(provenance),
            })
            self.write_per_query(temporary)
            runner.publish_arm(temporary, published, provenance)

            self.assert_gate_code(
                "E_ARM_PROVENANCE",
                runner.verify_published_arm,
                published,
                provenance,
            )

    def test_artifact_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"seed": 42, "arm": "shuffle", "mapping": 0}
            published, provenance = self.publish_synthetic_arm(root, row)
            (published / "summary.json").write_text("{}\n", encoding="utf-8")

            self.assert_gate_code(
                "E_ARM_HASH",
                runner.verify_published_arm,
                published,
                provenance,
            )

    def test_renamed_arm_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"seed": 42, "arm": "shuffle", "mapping": 0}
            published, provenance = self.publish_synthetic_arm(root, row)
            renamed = root / "seed_42__shuffle_m01"
            published.rename(renamed)

            self.assert_gate_code(
                "E_ARM_PROVENANCE",
                runner.verify_published_arm,
                renamed,
                provenance,
            )

    def test_invalid_secondary_is_allowed_only_for_metric_free_centroid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            centroid = {"seed": 42, "arm": "centroid"}
            published, provenance = self.publish_synthetic_arm(
                root,
                centroid,
                status="INVALID_SECONDARY",
                include_metrics=False,
            )
            marker = runner.verify_published_arm(published, provenance)
            self.assertEqual(marker["status"], "INVALID_SECONDARY")

            primary = {"seed": 1234, "arm": "shuffle", "mapping": 0}
            invalid_primary, primary_provenance = self.publish_synthetic_arm(
                root,
                primary,
                status="INVALID_SECONDARY",
                include_metrics=False,
            )
            self.assert_gate_code(
                "E_ARM_STATUS",
                runner.verify_published_arm,
                invalid_primary,
                primary_provenance,
            )

    def test_invalid_centroid_must_not_publish_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"seed": 42, "arm": "centroid"}
            published, provenance = self.publish_synthetic_arm(
                root,
                row,
                status="INVALID_SECONDARY",
                include_metrics=True,
            )

            self.assert_gate_code(
                "E_ARM_STATUS",
                runner.verify_published_arm,
                published,
                provenance,
            )

    def test_correct_start_is_the_only_arm_allowed_to_store_actual_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = {"seed": 42, "arm": "correct", "position": "start"}
            published, provenance = self.publish_synthetic_arm(
                root,
                start,
                include_actual=True,
            )
            self.assertEqual(
                runner.verify_published_arm(published, provenance)["status"],
                "PASS",
            )

            end = {"seed": 1234, "arm": "correct", "position": "end"}
            invalid_end, end_provenance = self.publish_synthetic_arm(
                root,
                end,
                include_actual=True,
            )
            self.assert_gate_code(
                "E_ARM_FILES",
                runner.verify_published_arm,
                invalid_end,
                end_provenance,
            )


class ExecutionResumeTests(GateCodeMixin, unittest.TestCase):
    @staticmethod
    def base_manifest(prepared_hashes=None):
        return {
            "schema": "synthetic-exp374",
            "prepared_artifact_sha256": prepared_hashes or {},
        }

    def test_execution_create_is_exclusive_and_exact_resume_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            manifest = self.base_manifest()
            execution, execution_sha = protocol.create_execution_directory(
                output_root, manifest, None)
            stale = execution / ".seed_42__shuffle_m00.tmp-arm"
            stale.mkdir()
            (stale / "partial.bin").write_bytes(b"partial")

            resumed, resumed_sha = protocol.create_execution_directory(
                output_root, manifest, execution)

            self.assertEqual(resumed, execution)
            self.assertEqual(resumed_sha, execution_sha)
            # create_execution_directory validates identity only.  Arm-temp
            # cleanup belongs to run_phase after all frozen checks pass.
            self.assertTrue(stale.is_dir())
            self.assertEqual((stale / "partial.bin").read_bytes(), b"partial")
            with self.assertRaises(FileExistsError):
                protocol.create_execution_directory(output_root, manifest, None)

    def test_execution_resume_rejects_frozen_manifest_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            manifest = self.base_manifest()
            execution, _sha = protocol.create_execution_directory(
                output_root, manifest, None)
            protocol.atomic_write_json(
                execution / "premetric_manifest.json",
                {"schema": "tampered", "prepared_artifact_sha256": {}},
            )

            self.assert_gate_code(
                "E_RESUME_HASH_DRIFT",
                protocol.create_execution_directory,
                output_root,
                manifest,
                execution,
            )

    def test_execution_resume_rejects_complete_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            manifest = self.base_manifest()
            execution, _sha = protocol.create_execution_directory(
                output_root, manifest, None)
            protocol.atomic_write_json(execution / "COMPLETE", {"status": "COMPLETE"})

            self.assert_gate_code(
                "E_RESUME_COMPLETE",
                protocol.create_execution_directory,
                output_root,
                manifest,
                execution,
            )

    def make_prepared_execution(self, output_root):
        payload = b"synthetic-prepared-artifact"
        prepared_hashes = {"scene.bin": protocol.sha256_bytes(payload)}
        manifest = self.base_manifest(prepared_hashes)
        execution, execution_sha = protocol.create_execution_directory(
            output_root, manifest, None)
        prepared = execution / "prepared"
        prepared.mkdir()
        protocol.atomic_write_bytes(prepared / "scene.bin", payload)
        protocol.atomic_write_json(execution / "PREPARED.json", {
            "execution_sha256": execution_sha,
            "prepared_artifact_sha256": prepared_hashes,
        })
        return execution

    def test_verify_prepared_artifacts_detects_hash_and_complete_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = self.make_prepared_execution(Path(directory))
            manifest = runner.verify_prepared_artifacts(execution)
            self.assertEqual(manifest["schema"], "synthetic-exp374")

            (execution / "prepared" / "scene.bin").write_bytes(b"mutated")
            self.assert_gate_code(
                "E_RESUME_HASH_DRIFT",
                runner.verify_prepared_artifacts,
                execution,
            )

        with tempfile.TemporaryDirectory() as directory:
            execution = self.make_prepared_execution(Path(directory))
            protocol.atomic_write_json(execution / "COMPLETE", {"status": "COMPLETE"})
            self.assert_gate_code(
                "E_RESUME_COMPLETE",
                runner.verify_prepared_artifacts,
                execution,
            )


class ResultsAndCrashStateTests(GateCodeMixin, unittest.TestCase):
    @staticmethod
    def publish_results(root, result_payload=None):
        temporary = root / ".results.tmp"
        results = root / "results"
        temporary.mkdir()
        protocol.atomic_write_json(
            temporary / "gate_a_results.json",
            result_payload or {"status": "COMPLETE"},
        )
        np.savez(
            temporary / "primary_query_aggregates.npz",
            seed_42_correct_AP=np.asarray([0.5, 0.6], dtype=np.float64),
        )
        hashes = runner.arm_artifact_hashes(temporary)
        protocol.atomic_write_json(temporary / "COMPLETE.json", {
            "status": "PASS",
            "artifact_sha256": hashes,
        })
        protocol.publish_directory(temporary, results)
        return results, hashes

    def test_results_directory_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, hashes = self.publish_results(root)

            marker = runner.verify_results_directory(results)
            self.assertEqual(marker["artifact_sha256"], hashes)

            (results / "gate_a_results.json").write_text(
                '{"status":"tampered"}\n', encoding="utf-8")
            self.assert_gate_code(
                "E_RESULTS_HASH",
                runner.verify_results_directory,
                results,
            )

    def test_results_json_round_trip_normalizes_integer_keys_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = {
                "per_seed": {
                    42: {"mAP": 58.0},
                    1234: {"mAP": 58.3},
                    2024: {"mAP": 58.1},
                },
            }
            results, _hashes = self.publish_results(root, current)
            frozen = json.loads(
                (results / "gate_a_results.json").read_text(encoding="utf-8"))
            normalized = json.loads(protocol.canonical_json_bytes(current))

            self.assertEqual(frozen, normalized)

    def test_published_results_remain_reusable_if_execution_complete_was_not_written(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = Path(directory)
            current = {"per_seed": {42: {"mAP": 58.0}}}
            results, hashes = self.publish_results(execution, current)

            self.assertFalse((execution / "COMPLETE").exists())
            marker = runner.verify_results_directory(results)
            frozen = json.loads(
                (results / "gate_a_results.json").read_text(encoding="utf-8"))
            normalized = json.loads(protocol.canonical_json_bytes(current))
            with np.load(
                results / "primary_query_aggregates.npz",
                allow_pickle=False,
            ) as payload:
                aggregate = payload["seed_42_correct_AP"].copy()

            self.assertEqual(marker["artifact_sha256"], hashes)
            self.assertEqual(frozen, normalized)
            self.assertTrue(np.array_equal(
                aggregate,
                np.asarray([0.5, 0.6], dtype=np.float64),
            ))

    def test_self_hashed_semantic_results_drift_survives_hash_check_but_not_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = Path(directory)
            expected = {"decision": "GO", "per_seed": {42: 0.5}}
            results, _hashes = self.publish_results(execution, expected)
            protocol.atomic_write_json(
                results / "gate_a_results.json",
                {"decision": "NO_GO", "per_seed": {42: -0.5}},
            )
            resigned_hashes = runner.arm_artifact_hashes(results)
            protocol.atomic_write_json(results / "COMPLETE.json", {
                "status": "PASS",
                "artifact_sha256": resigned_hashes,
            })

            marker = runner.verify_results_directory(results)
            frozen = json.loads(
                (results / "gate_a_results.json").read_text(encoding="utf-8"))
            normalized_expected = json.loads(protocol.canonical_json_bytes(expected))

            self.assertEqual(marker["artifact_sha256"], resigned_hashes)
            self.assertNotEqual(frozen, normalized_expected)

    def test_archive_transient_state_preserves_history_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = Path(directory)
            protocol.atomic_write_json(execution / "FAILED.json", {"attempt": 1})
            protocol.atomic_write_json(execution / "RUN_PROGRESS.json", {"arm": 1})

            runner.archive_transient_state(execution)

            self.assertFalse((execution / "FAILED.json").exists())
            self.assertFalse((execution / "RUN_PROGRESS.json").exists())
            self.assertEqual(
                json.loads((execution / "state_history/FAILED_0001.json").read_text()),
                {"attempt": 1},
            )
            self.assertTrue((execution / "state_history/RUN_PROGRESS_0001.json").is_file())

            protocol.atomic_write_json(execution / "FAILED.json", {"attempt": 2})
            runner.archive_transient_state(execution)
            self.assertEqual(
                json.loads((execution / "state_history/FAILED_0002.json").read_text()),
                {"attempt": 2},
            )

    def test_stale_arm_temporaries_are_removed_without_touching_published_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = Path(directory)
            stale_directory = execution / ".seed_42__shuffle_m00.tmp-arm"
            stale_file = execution / ".seed_42__shuffle_m01.tmp-arm"
            published = execution / "seed_42__shuffle_m00"
            stale_directory.mkdir()
            stale_file.write_bytes(b"partial")
            published.mkdir()
            (published / "payload.bin").write_bytes(b"published")

            runner._remove_stale_arm_temporaries(execution)

            self.assertFalse(stale_directory.exists())
            self.assertFalse(stale_file.exists())
            self.assertEqual((published / "payload.bin").read_bytes(), b"published")

    def test_failure_manifest_is_nonreportable_and_complete_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            execution = Path(directory)
            args = SimpleNamespace(execution_dir=str(execution), phase="run")
            error = protocol.GateProtocolError("E_SYNTHETIC", "expected failure")

            runner._write_failure_manifest(args, error)

            failure_path = execution / "FAILED.json"
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "FAILED_NONREPORTABLE")
            self.assertEqual(failure["phase"], "run")
            self.assertEqual(failure["error_code"], "E_SYNTHETIC")
            self.assertFalse(failure["failed_arm_published"])

            sentinel = b'{"sentinel":"must-not-change"}\n'
            failure_path.write_bytes(sentinel)
            protocol.atomic_write_json(execution / "COMPLETE", {"status": "COMPLETE"})
            runner._write_failure_manifest(
                args,
                protocol.GateProtocolError("E_LATE", "after complete"),
            )
            self.assertEqual(failure_path.read_bytes(), sentinel)


if __name__ == "__main__":
    unittest.main()
