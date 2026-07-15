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
import copy
import hashlib
import tempfile
import unittest
from dataclasses import replace
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


def synthetic_quick_identity(root):
    """Create the production-shaped identity snapshot for 19 real files."""

    identity_root = Path(root) / ".synthetic-quick-identity"
    identity_root.mkdir(exist_ok=True)
    registry = {}
    for index in range(19):
        path = identity_root / f"asset_{index:02d}.bin"
        if not path.exists():
            path.write_bytes(f"synthetic-quick-asset-{index}\n".encode("utf-8"))
        _report, identity, _unused = runner._stable_regular_file(path)
        registry[str(path)] = identity
    return registry


class StableRegularFileProtocolTests(GateCodeMixin, unittest.TestCase):
    def test_initial_lstat_and_open_errors_use_stable_io_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"stable\n")
            with mock.patch.object(
                Path, "lstat", side_effect=PermissionError("initial"),
            ):
                self.assert_gate_code(
                    "E_RELATION_FILE_IO",
                    runner._stable_regular_file,
                    path,
                )
            with mock.patch.object(
                Path, "open", side_effect=PermissionError("open"),
            ):
                self.assert_gate_code(
                    "E_RELATION_FILE_IO",
                    runner._stable_regular_file,
                    path,
                )

    def test_after_read_and_recheck_errors_use_stable_toctou_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"stable\n")
            before = path.lstat()
            with mock.patch.object(
                Path,
                "lstat",
                side_effect=(before, FileNotFoundError("after")),
            ):
                self.assert_gate_code(
                    "E_RELATION_FILE_TOCTOU",
                    runner._stable_regular_file,
                    path,
                )

            _report, identity, _payload = runner._stable_regular_file(path)
            with mock.patch.object(
                Path, "lstat", side_effect=FileNotFoundError("recheck"),
            ):
                self.assert_gate_code(
                    "E_RELATION_FILE_TOCTOU",
                    runner._recheck_identities,
                    {str(path): identity},
                )

    def test_nonregular_initial_and_recheck_assets_use_stable_type_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assert_gate_code(
                "E_RELATION_FILE_TYPE",
                runner._stable_regular_file,
                root,
            )
            self.assert_gate_code(
                "E_RELATION_FILE_TYPE",
                runner._recheck_identities,
                {str(root): (0, 0, 0, 0, 0, 0, 0)},
            )


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
            42: (
                "/home/afr/SOLIDER-REID/log/multiseed/"
                "exp007_psg_seed42/test_default.txt",
                57.5,
                66.7,
            ),
            1234: (
                "/home/afr/SOLIDER-REID/log/multiseed/"
                "exp007_psg_seed1234/test_default.txt",
                58.3,
                68.1,
            ),
            2024: (
                "/home/afr/SOLIDER-REID/log/multiseed/"
                "exp007_psg_seed2024/test_default.txt",
                58.0,
                68.4,
            ),
        }

        actual = {
            int(spec["seed"]): (
                str(spec["flat_log"]),
                float(spec["expected_mAP"]),
                float(spec["expected_R1"]),
            )
            for spec in runner.DEFAULT_CHECKPOINTS
        }

        self.assertEqual(actual, expected)

    def test_checkpoint_specs_rejects_flat_log_metric_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_specs = []
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
            source_specs[0]["expected_mAP"] = 99.0

            with mock.patch.object(runner, "DEFAULT_CHECKPOINTS", tuple(source_specs)):
                self.assert_gate_code(
                    "E_FLAT_LOG_MANIFEST",
                    runner.checkpoint_specs,
                    None,
                )

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


class SignedSceneAuditTests(GateCodeMixin, unittest.TestCase):
    @staticmethod
    def signed_batch():
        scene = torch.zeros((2, 17, 96, 32), dtype=torch.float32)
        # Broad, sample-distinct regions remain visible after the frozen
        # 96x32 -> 12x4 bilinear resize.  Single-pixel negatives can be
        # legitimately missed by that sampling grid and are not a valid
        # nonzero-Delta fixture.
        scene[0, 10, 0:16, 0:16] = -0.25
        scene[1, 6, 32:64, 16:32] = -0.07
        scene[0, 0, 16:48, 8:24] = 1.25
        scene[1, 1, 48:80, 0:16] = 0.75
        return scene

    @staticmethod
    def actual_audit(scene):
        state = runner._new_actual_space_audit()
        runner._update_actual_space_audit(state, scene)
        return runner._finalize_actual_space_audit(
            state, "synthetic", len(scene), torch.device("cpu"))

    @staticmethod
    def expected_digest(value):
        digest = hashlib.sha256()
        for sample in value:
            array = np.asarray(sample.detach().cpu().numpy(), dtype=np.dtype("<f4"))
            digest.update(np.ascontiguousarray(array).tobytes(order="C"))
        return digest.hexdigest()

    def test_signed_audit_freezes_raw_sign_and_actual_space_provenance(self):
        scene = self.signed_batch()
        frozen_raw = scene.clone()
        state = runner._new_signed_scene_audit()
        runner._update_signed_scene_audit(state, scene)
        audit = runner._finalize_signed_scene_audit(state, "synthetic", 2)
        actual_audit = self.actual_audit(scene)
        audit["actual_space"] = actual_audit

        self.assertTrue(torch.equal(scene, frozen_raw))
        self.assertEqual(audit["transform"], "positive_part_v1")
        self.assertEqual(audit["sample_order"], "dataset_index_0_to_N_minus_1")
        self.assertEqual(audit["raw_shape"], [2, 17, 96, 32])
        self.assertEqual(audit["raw_dtype"], "<f4")
        self.assertEqual(audit["raw_element_count"], 2 * 17 * 96 * 32)
        self.assertAlmostEqual(audit["raw_min"], -0.25)
        self.assertEqual(audit["negative_element_count"], 768)
        self.assertEqual(audit["negative_sample_count"], 2)
        self.assertEqual(audit["negative_sample_channel_count"], 2)
        self.assertEqual(audit["negative_channel_indices_0based"], [6, 10])
        self.assertAlmostEqual(audit["negative_absolute_mass"], 99.84, places=4)
        self.assertEqual(actual_audit["compute_backend"], "torch_cpu_test_only")
        self.assertEqual(
            actual_audit["active_psg_blocks"]["s3_b0"],
            actual_audit["active_psg_blocks"]["s3_b1"],
        )
        block = actual_audit["active_psg_blocks"]["s3_b0"]
        self.assertEqual(block["shape"], [2, 17, 12, 4])
        self.assertEqual(block["dtype"], "<f4")
        self.assertEqual(block["element_count"], 2 * 17 * 12 * 4)
        self.assertGreater(block["delta_max_abs"], 0.0)
        self.assertGreater(block["delta_sum_abs"], 0.0)
        self.assertGreater(block["delta_mean_abs"], 0.0)
        for name in ("sraw_sha256", "spos_sha256", "delta_sha256"):
            self.assertEqual(len(block[name]), 64)
        self.assertNotEqual(block["sraw_sha256"], block["spos_sha256"])
        sraw = protocol.actual_psg_input(scene, (12, 4))
        spos = protocol.actual_psg_input(scene.clamp_min(0.0), (12, 4))
        delta = sraw - spos
        self.assertEqual(block["sraw_sha256"], self.expected_digest(sraw))
        self.assertEqual(block["spos_sha256"], self.expected_digest(spos))
        self.assertEqual(block["delta_sha256"], self.expected_digest(delta))
        split_hashes = runner._actual_input_split_sha256(sraw, num_query=1)
        self.assertEqual(split_hashes, {
            "query": self.expected_digest(sraw[:1]),
            "gallery": self.expected_digest(sraw[1:]),
        })

    def test_signed_audit_is_sample_ordered_and_canonical(self):
        scene = self.signed_batch()
        whole = runner._new_signed_scene_audit()
        runner._update_signed_scene_audit(whole, scene)
        whole_audit = runner._finalize_signed_scene_audit(whole, "synthetic", 2)
        whole_actual = self.actual_audit(scene)

        streamed = runner._new_signed_scene_audit()
        runner._update_signed_scene_audit(streamed, scene[:1])
        runner._update_signed_scene_audit(streamed, scene[1:])
        streamed_audit = runner._finalize_signed_scene_audit(streamed, "synthetic", 2)
        streamed_actual_state = runner._new_actual_space_audit()
        runner._update_actual_space_audit(streamed_actual_state, scene[:1])
        runner._update_actual_space_audit(streamed_actual_state, scene[1:])
        streamed_actual = runner._finalize_actual_space_audit(
            streamed_actual_state, "synthetic", 2, torch.device("cpu"))
        self.assertEqual(
            protocol.canonical_json_bytes(whole_audit),
            protocol.canonical_json_bytes(streamed_audit),
        )
        self.assertEqual(
            protocol.canonical_json_bytes(whole_actual),
            protocol.canonical_json_bytes(streamed_actual),
        )

        reversed_scene = scene.flip(0).contiguous()
        reversed_sign = runner._new_signed_scene_audit()
        runner._update_signed_scene_audit(reversed_sign, reversed_scene)
        reversed_sign_audit = runner._finalize_signed_scene_audit(
            reversed_sign, "synthetic", 2)
        reversed_actual = self.actual_audit(reversed_scene)
        for name in (
            "raw_min",
            "negative_element_count",
            "negative_sample_count",
            "negative_sample_channel_count",
            "negative_channel_indices_0based",
            "negative_absolute_mass",
        ):
            self.assertEqual(whole_audit[name], reversed_sign_audit[name])
        for block_name in runner.ACTIVE_PSG_BLOCK_SHAPES:
            original = whole_actual["active_psg_blocks"][block_name]
            reversed_block = reversed_actual["active_psg_blocks"][block_name]
            for name in ("sraw_sha256", "spos_sha256", "delta_sha256"):
                self.assertNotEqual(original[name], reversed_block[name])

    def test_wrong_correct_hook_sha_fails_before_metric_computation(self):
        features = torch.zeros((2, 768), dtype=torch.float32)
        actual = torch.zeros((2, 17, 12, 4), dtype=torch.float32)
        wrong = {
            "correct_actual_sraw_sha256": {
                split: {key: "0" * 64 for key in runner.ACTIVE_PSG_BLOCK_SHAPES}
                for split in ("query", "gallery")
            },
        }
        with mock.patch.object(runner, "_metric_payload") as metric:
            self.assert_gate_code(
                "E_HOOK_PREMETRIC_DRIFT",
                runner._audited_metric_payload,
                features,
                [1, 2],
                [1, 2],
                1,
                "correct",
                actual,
                2,
                wrong,
            )
        metric.assert_not_called()


class PreparedSignedRawProvenanceTests(unittest.TestCase):
    def test_correct_shuffle_and_group_preserve_signed_raw_values(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            query = np.zeros((2, 17, 96, 32), dtype=np.float32)
            gallery = np.zeros((2, 17, 96, 32), dtype=np.float32)
            query[0, 6, 1, 1] = -0.1
            query[1, 6, 1, 1] = -0.2
            gallery[0, 10, 2, 2] = -0.3
            gallery[1, 10, 2, 2] = -0.4
            np.save(prepared / "query_scene_heatmaps.npy", query, allow_pickle=False)
            np.save(prepared / "gallery_scene_heatmaps.npy", gallery, allow_pickle=False)
            mappings = np.tile(np.asarray([[1, 0]], dtype=np.int32), (20, 1))
            np.save(prepared / "query_mappings.npy", mappings, allow_pickle=False)
            np.save(prepared / "gallery_mappings.npy", mappings, allow_pickle=False)
            scenes = runner.PreparedSceneAccess(prepared, num_query=2)
            rows = np.arange(4, dtype=np.int64)

            correct = scenes.scenes_for_rows(rows, {"arm": "correct"})
            shuffle = scenes.scenes_for_rows(
                rows, {"arm": "shuffle", "mapping": 0})
            group = scenes.scenes_for_rows(
                rows, {"arm": "group", "mapping": 0, "group": "shoulder"})

            self.assertTrue(np.array_equal(correct[:2], query))
            self.assertTrue(np.array_equal(correct[2:], gallery))
            self.assertAlmostEqual(float(shuffle[0, 6, 1, 1]), -0.2)
            self.assertAlmostEqual(float(shuffle[1, 6, 1, 1]), -0.1)
            self.assertAlmostEqual(float(shuffle[2, 10, 2, 2]), -0.4)
            self.assertAlmostEqual(float(shuffle[3, 10, 2, 2]), -0.3)
            self.assertTrue(np.array_equal(group[:, 5:7], shuffle[:, 5:7]))
            self.assertTrue(np.array_equal(group[:, :5], correct[:, :5]))
            self.assertTrue(np.array_equal(group[:, 7:], correct[:, 7:]))


class PoseAssetManifestV2Tests(GateCodeMixin, unittest.TestCase):
    IMAGE_NAME = "0001_c2_f0000003.jpg"

    @staticmethod
    def make_dataset(root, target_person_idx=0, persons=None):
        pose_root = Path(root) / "query"
        pose_root.mkdir()
        names = [f"person_{index:02d}.npz" for index in range(8)]
        for index, name in enumerate(names):
            (pose_root / name).write_bytes(f"pose-{index}\n".encode("utf-8"))
        selected = list(names if persons is None else persons)
        return SimpleNamespace(
            pose_dir=str(pose_root),
            max_persons=6,
            index={
                PoseAssetManifestV2Tests.IMAGE_NAME: {
                    "persons": selected,
                    "target_person_idx": target_person_idx,
                },
            },
        ), names

    def test_first_six_and_conditional_target_first_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, names = self.make_dataset(directory, target_person_idx=4)

            manifest = runner._pose_asset_manifest_v2(
                dataset, self.IMAGE_NAME, {}, {})

            expected = [names[4], names[0], names[1], names[2], names[3], names[5]]
            self.assertEqual(
                manifest["effective_pose_person_relpaths"],
                tuple(f"pose_data/query/{name}" for name in expected),
            )
            self.assertEqual(len(manifest["full_pose_person_paths"]), 8)
            self.assertEqual(len(manifest["effective_pose_person_paths"]), 6)
            self.assertFalse(manifest["target_outside_effective"])
            self.assertEqual(
                (manifest["source_pid"], manifest["source_camid"],
                 manifest["source_frame_id"]),
                (1, 1, 3),
            )

    def test_target_outside_first_six_does_not_reorder_effective_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, names = self.make_dataset(directory, target_person_idx=6)

            manifest = runner._pose_asset_manifest_v2(
                dataset, self.IMAGE_NAME, {})

            self.assertEqual(
                manifest["effective_pose_person_relpaths"],
                tuple(f"pose_data/query/{name}" for name in names[:6]),
            )
            self.assertEqual(manifest["target_person_idx"], 6)
            self.assertTrue(manifest["target_outside_effective"])
            self.assertEqual(len(manifest["full_pose_person_sha256"]), 8)
            self.assertEqual(len(manifest["effective_pose_person_sha256"]), 6)

    def test_invalid_target_indices_are_rejected(self):
        for invalid in (-1, 8, True, 1.5, "1"):
            with self.subTest(target_person_idx=invalid):
                with tempfile.TemporaryDirectory() as directory:
                    dataset, _names = self.make_dataset(
                        directory, target_person_idx=invalid)
                    self.assert_gate_code(
                        "E_POSE_TARGET_INDEX",
                        runner._pose_asset_manifest_v2,
                        dataset,
                        self.IMAGE_NAME,
                        {},
                    )

    def test_non_basename_pose_entries_are_rejected(self):
        invalid_entries = (
            "/absolute/person.npz",
            "../person.npz",
            "./person.npz",
            "nested/person.npz",
            "nested\\person.npz",
            "pose_data/query/person.npz",
        )
        for invalid in invalid_entries:
            with self.subTest(person=invalid):
                with tempfile.TemporaryDirectory() as directory:
                    dataset, _names = self.make_dataset(
                        directory, persons=[invalid])
                    self.assert_gate_code(
                        "E_POSE_PERSON_PATH",
                        runner._pose_asset_manifest_v2,
                        dataset,
                        self.IMAGE_NAME,
                        {},
                    )


class SceneRecordV2Tests(GateCodeMixin, unittest.TestCase):
    @staticmethod
    def metadata_row(index, split="query"):
        return {
            "schema": runner.SCENE_METADATA_SCHEMA,
            "index": index,
            "split": split,
            "path": f"/synthetic/{split}/{index:04d}_c1_f0000001.jpg",
            "rgb_sha256": f"rgb-{index}",
            "pose_path_sha256": f"pose-path-{index}",
            "pose_content_sha256": f"pose-content-{index}",
            "pid": index + 1,
            "camid": 0,
            "viewid": 1,
            "person_count": 2,
            "frame": 1,
            "report": {"total_L1": 1.0},
            "source_pid": index + 1,
            "source_camid": 0,
            "source_frame_id": 1,
            "target_person_idx": 1,
            "full_pose_person_relpaths": ["pose_data/query/a.npz", "pose_data/query/b.npz"],
            "full_pose_person_paths": ["/synthetic/query/a.npz", "/synthetic/query/b.npz"],
            "full_pose_person_sha256": ["pose-a", "pose-b"],
            "effective_pose_person_relpaths": [
                "pose_data/query/b.npz", "pose_data/query/a.npz"],
            "effective_pose_person_paths": [
                "/synthetic/query/b.npz", "/synthetic/query/a.npz"],
            "effective_pose_person_sha256": ["pose-b", "pose-a"],
        }

    @staticmethod
    def write_payload(root, metadata, continuous):
        protocol.atomic_write_json(Path(root) / "query_metadata.json", metadata)
        np.save(
            Path(root) / "query_continuous.npy",
            continuous,
            allow_pickle=False,
        )

    def test_continuous_npy_is_the_only_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            continuous = np.arange(2 * 95, dtype=np.dtype("<f8")).reshape(2, 95)
            metadata = [self.metadata_row(0), self.metadata_row(1)]
            self.write_payload(directory, metadata, continuous)

            records = runner.load_scene_records(Path(directory), "query")

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].continuous, tuple(continuous[0]))
            self.assertEqual(records[1].continuous, tuple(continuous[1]))
            self.assertNotIn("continuous", metadata[0])

    def test_metadata_schema_is_exact_nonempty_and_ordered(self):
        mutations = {
            "v1": lambda rows: rows[0].__setitem__("schema", "exp374-scene-metadata-v1"),
            "missing": lambda rows: rows[0].pop("report"),
            "extra_continuous": lambda rows: rows[0].__setitem__("continuous", [0.0] * 95),
            "out_of_order": lambda rows: rows[1].__setitem__("index", 0),
            "empty_full_tuple": lambda rows: rows[0].__setitem__(
                "full_pose_person_relpaths", []),
            "constituent_length_drift": lambda rows: rows[0].__setitem__(
                "effective_pose_person_sha256", ["pose-b"]),
        }
        for label, mutate in mutations.items():
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as directory:
                    metadata = [self.metadata_row(0), self.metadata_row(1)]
                    mutate(metadata)
                    continuous = np.zeros((2, 95), dtype=np.dtype("<f8"))
                    self.write_payload(directory, metadata, continuous)
                    self.assert_gate_code(
                        "E_METADATA_SCHEMA_V2",
                        runner.load_scene_records,
                        Path(directory),
                        "query",
                    )

        with tempfile.TemporaryDirectory() as directory:
            self.write_payload(
                directory, [], np.zeros((0, 95), dtype=np.dtype("<f8")))
            self.assert_gate_code(
                "E_METADATA_SCHEMA_V2",
                runner.load_scene_records,
                Path(directory),
                "query",
            )

    def test_continuous_cache_requires_exact_count_shape_dtype_and_finite_values(self):
        wrong_values = {
            "count": np.zeros((1, 95), dtype=np.dtype("<f8")),
            "shape": np.zeros((2, 94), dtype=np.dtype("<f8")),
            "dtype": np.zeros((2, 95), dtype=np.dtype("<f4")),
            "nonfinite": np.full((2, 95), np.nan, dtype=np.dtype("<f8")),
        }
        for label, continuous in wrong_values.items():
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as directory:
                    metadata = [self.metadata_row(0), self.metadata_row(1)]
                    self.write_payload(directory, metadata, continuous)
                    self.assert_gate_code(
                        "E_CONTINUOUS_CACHE_V2",
                        runner.load_scene_records,
                        Path(directory),
                        "query",
                    )


class SplitRelationAuditV2Tests(GateCodeMixin, unittest.TestCase):
    SPLITS = ("train", "query", "gallery")

    @staticmethod
    def make_record(
        split,
        index,
        basename,
        token,
        pid,
        full_person_count=2,
        target_person_idx=0,
    ):
        full_names = tuple(
            f"{basename}.person{position}.npz"
            for position in range(full_person_count))
        full_relpaths = tuple(
            f"pose_data/{split}/{name}" for name in full_names)
        full_paths = tuple(
            f"/synthetic/pose/{split}/{name}" for name in full_names)
        full_content = tuple(
            protocol.sha256_bytes(
                f"pose-person-content-{token}-{position}".encode("utf-8"))
            for position in range(full_person_count))
        effective_indices = list(range(min(full_person_count, 6)))
        if 0 < target_person_idx < len(effective_indices):
            target = effective_indices.pop(target_person_idx)
            effective_indices.insert(0, target)
        return protocol.SceneRecord(
            metadata_schema=runner.SCENE_METADATA_SCHEMA,
            index=index,
            split=split,
            path=f"/synthetic/rgb/{split}/{basename}",
            rgb_sha256=protocol.sha256_bytes(
                f"rgb-{token}".encode("utf-8")),
            pose_path_sha256=protocol.sha256_bytes(
                protocol.canonical_json_bytes(list(full_paths))),
            pose_content_sha256=protocol.sha256_bytes(
                protocol.canonical_json_bytes(list(full_content))),
            pid=pid,
            camid=index % 2,
            viewid=1,
            person_count=len(effective_indices),
            continuous=tuple(float(index) + value / 1000.0 for value in range(95)),
            frame=100 + index,
            report={"token": token, "total_L1": 1.0},
            source_pid=pid,
            source_camid=index % 2,
            source_frame_id=100 + index,
            target_person_idx=target_person_idx,
            full_pose_person_relpaths=full_relpaths,
            full_pose_person_paths=full_paths,
            full_pose_person_sha256=full_content,
            effective_pose_person_relpaths=tuple(
                full_relpaths[position] for position in effective_indices),
            effective_pose_person_paths=tuple(
                full_paths[position] for position in effective_indices),
            effective_pose_person_sha256=tuple(
                full_content[position] for position in effective_indices),
        )

    @staticmethod
    def mirror_gallery(query_record, index=None):
        gallery_index = query_record.index if index is None else index
        gallery = SplitRelationAuditV2Tests.make_record(
            "gallery",
            gallery_index,
            Path(query_record.path).name,
            f"gallery-mirror-{gallery_index}",
            query_record.pid,
            full_person_count=len(query_record.full_pose_person_paths),
            target_person_idx=query_record.target_person_idx,
        )
        return replace(
            gallery,
            rgb_sha256=query_record.rgb_sha256,
            pose_content_sha256=query_record.pose_content_sha256,
            camid=query_record.camid,
            viewid=query_record.viewid,
            person_count=query_record.person_count,
            continuous=query_record.continuous,
            frame=query_record.frame,
            report=dict(query_record.report),
            source_pid=query_record.source_pid,
            source_camid=query_record.source_camid,
            source_frame_id=query_record.source_frame_id,
            target_person_idx=query_record.target_person_idx,
            full_pose_person_relpaths=tuple(
                f"pose_data/gallery/{Path(value).name}"
                for value in query_record.full_pose_person_relpaths),
            full_pose_person_sha256=query_record.full_pose_person_sha256,
            effective_pose_person_relpaths=tuple(
                f"pose_data/gallery/{Path(value).name}"
                for value in query_record.effective_pose_person_relpaths),
            effective_pose_person_sha256=query_record.effective_pose_person_sha256,
        )

    @staticmethod
    def replace_full_assets(record, *, paths=None, content=None):
        """Replace constituents while preserving every derived projection."""

        full_paths = tuple(
            record.full_pose_person_paths if paths is None else paths)
        full_content = tuple(
            record.full_pose_person_sha256 if content is None else content)
        full_relpaths = tuple(
            f"pose_data/{record.split}/{Path(path).name}"
            for path in full_paths)
        effective_indices = list(range(min(len(full_paths), 6)))
        if 0 < record.target_person_idx < len(effective_indices):
            target = effective_indices.pop(record.target_person_idx)
            effective_indices.insert(0, target)
        return replace(
            record,
            pose_path_sha256=protocol.sha256_bytes(
                protocol.canonical_json_bytes(list(full_paths))),
            pose_content_sha256=protocol.sha256_bytes(
                protocol.canonical_json_bytes(list(full_content))),
            person_count=len(effective_indices),
            full_pose_person_relpaths=full_relpaths,
            full_pose_person_paths=full_paths,
            full_pose_person_sha256=full_content,
            effective_pose_person_relpaths=tuple(
                full_relpaths[position] for position in effective_indices),
            effective_pose_person_paths=tuple(
                full_paths[position] for position in effective_indices),
            effective_pose_person_sha256=tuple(
                full_content[position] for position in effective_indices),
        )

    @classmethod
    def records_fixture(cls, mirror=False, full_person_count=2):
        train = cls.make_record(
            "train", 0, "0001_c1_f0000001.jpg", "train-0", 1,
            full_person_count=full_person_count)
        query = cls.make_record(
            "query", 0, "0002_c2_f0000002.jpg", "query-0", 2,
            full_person_count=full_person_count)
        gallery = (
            cls.mirror_gallery(query)
            if mirror
            else cls.make_record(
                "gallery", 0, "0003_c3_f0000003.jpg", "gallery-0", 3,
                full_person_count=full_person_count)
        )
        return {"train": [train], "query": [query], "gallery": [gallery]}

    @classmethod
    def two_mirror_fixture(cls, full_person_count=2):
        records = cls.records_fixture(
            mirror=True, full_person_count=full_person_count)
        query = cls.make_record(
            "query", 1, "0004_c4_f0000004.jpg", "query-1", 4,
            full_person_count=full_person_count)
        records["query"].append(query)
        records["gallery"].append(cls.mirror_gallery(query))
        return records

    @staticmethod
    def official_lists_for(records):
        return {
            split: [Path(record.path).name for record in records[split]]
            for split in SplitRelationAuditV2Tests.SPLITS
        }

    @staticmethod
    def official_report_for(official_lists):
        output = {}
        for split in SplitRelationAuditV2Tests.SPLITS:
            names = list(official_lists[split])
            raw = "".join(f"{name}\n" for name in names).encode("utf-8")
            canonical = protocol.canonical_json_bytes(sorted(names))
            expected = runner.OFFICIAL_SPLITS[split]
            output[split] = {
                "rgb_root": str(expected["root"]),
                "list": str(expected["list"]),
                "count": len(names),
                "raw_bytes": len(raw),
                "raw_sha256": protocol.sha256_bytes(raw),
                "canonical_bytes": len(canonical),
                "canonical_sha256": protocol.sha256_bytes(canonical),
                "pose_index_bytes": 100 + len(names),
                "pose_index_sha256": protocol.sha256_bytes(
                    f"pose-index-{split}-{len(names)}".encode("utf-8")),
            }
        return output

    @staticmethod
    def cache_arrays_for(records):
        caches = {name: {} for name in ("heatmaps", "scores", "nuisance")}
        for split in SplitRelationAuditV2Tests.SPLITS:
            heatmaps = []
            scores = []
            nuisance = []
            for record in records[split]:
                value = float(sum(ord(char) for char in record.rgb_sha256) % 17)
                heatmaps.append(np.full((17, 96, 32), value, dtype=np.dtype("<f4")))
                scores.append(np.full((17,), value, dtype=np.dtype("<f4")))
                nuisance.append(np.full((95,), value, dtype=np.dtype("<f8")))
            caches["heatmaps"][split] = np.stack(heatmaps)
            caches["scores"][split] = np.stack(scores)
            caches["nuisance"][split] = np.stack(nuisance)
        return caches

    @classmethod
    def audit(cls, records, official_lists=None, official_report=None, caches=None):
        lists = official_lists or cls.official_lists_for(records)
        report = official_report or cls.official_report_for(lists)
        arrays = caches or cls.cache_arrays_for(records)
        return runner.audit_split_relations_v2(records, lists, report, arrays)

    @staticmethod
    def assert_self_hash(testcase, report):
        without_self = dict(report)
        frozen = without_self.pop("relation_report_sha256")
        testcase.assertEqual(
            frozen,
            protocol.sha256_bytes(protocol.canonical_json_bytes(without_self)),
        )

    def test_query_gallery_zero_overlap_passes_structure_layer(self):
        report = self.audit(self.records_fixture(mirror=False))

        self.assertEqual(report["relations"]["allowed_pair_count"], 0)
        self.assertEqual(report["relations"]["forbidden_pair_count"], 0)
        self.assertEqual(report["pairs"], [])
        self.assertEqual(
            report["cross_split"]["query_gallery"]["forbidden_overlap_count"],
            0,
        )
        self.assert_self_hash(self, report)

    def test_one_query_one_gallery_official_mirror_passes_projections(self):
        report = self.audit(self.records_fixture(mirror=True))
        relations = report["relations"]
        endpoint = relations["query_gallery_endpoint_pairs"]

        self.assertEqual(relations["allowed_pair_count"], 1)
        self.assertEqual(relations["junk_true_count"], 1)
        self.assertEqual(relations["forbidden_pair_count"], 0)
        self.assertTrue(endpoint["equal"])
        self.assertEqual(endpoint["rgb"], endpoint["pose"])
        self.assertEqual(endpoint["rgb"]["count"], 1)
        self.assertEqual(relations["query_gallery_joint_metadata_pairs"]["count"], 1)
        self.assertEqual(relations["query_gallery_joint_pairs"]["count"], 1)
        self.assertEqual(
            (report["pairs"][0]["query_index"], report["pairs"][0]["gallery_index"]),
            (0, 0),
        )
        self.assert_self_hash(self, report)

    def test_all_eight_within_split_duplicate_classes_fail_closed(self):
        cases = (
            ("path", "E_RELATION_WITHIN_PATH_DUPLICATE"),
            ("rgb_sha256", "E_RELATION_WITHIN_RGB_CONTENT_DUPLICATE"),
            ("pose_path_sha256", "E_RELATION_WITHIN_POSE_PATH_DUPLICATE"),
            ("pose_content_sha256", "E_RELATION_WITHIN_POSE_CONTENT_DUPLICATE"),
            (
                "effective_pose_person_paths",
                "E_RELATION_WITHIN_EFFECTIVE_CONSTITUENT_PATH_DUPLICATE",
            ),
            (
                "effective_pose_person_sha256",
                "E_RELATION_WITHIN_EFFECTIVE_CONSTITUENT_CONTENT_DUPLICATE",
            ),
            (
                "full_pose_person_paths",
                "E_RELATION_WITHIN_FULL_CONSTITUENT_PATH_DUPLICATE",
            ),
            (
                "full_pose_person_sha256",
                "E_RELATION_WITHIN_FULL_CONSTITUENT_CONTENT_DUPLICATE",
            ),
        )
        for field, error_code in cases:
            with self.subTest(field=field):
                records = self.records_fixture(
                    mirror=False, full_person_count=7)
                first = records["train"][0]
                second = self.make_record(
                    "train", 1, "0005_c5_f0000005.jpg", "train-1", 5,
                    full_person_count=7)
                if field in {"path", "rgb_sha256"}:
                    changed = replace(
                        second, **{field: getattr(first, field)})
                elif field == "pose_path_sha256":
                    changed = self.replace_full_assets(
                        second, paths=first.full_pose_person_paths)
                elif field == "pose_content_sha256":
                    changed = self.replace_full_assets(
                        second, content=first.full_pose_person_sha256)
                elif field.endswith("paths"):
                    position = 0 if field.startswith("effective") else 6
                    values = list(second.full_pose_person_paths)
                    values[position] = first.full_pose_person_paths[position]
                    changed = self.replace_full_assets(second, paths=values)
                else:
                    position = 0 if field.startswith("effective") else 6
                    values = list(second.full_pose_person_sha256)
                    values[position] = first.full_pose_person_sha256[position]
                    changed = self.replace_full_assets(second, content=values)
                records["train"].append(changed)
                self.assert_gate_code(
                    error_code,
                    self.audit,
                    records,
                )

    def test_train_eval_alias_matrix_fails_closed(self):
        cases = (
            ("path", "E_RELATION_TRAIN_EVAL_PATH_ALIAS"),
            ("rgb_sha256", "E_RELATION_TRAIN_EVAL_RGB_CONTENT_ALIAS"),
            ("pose_path_sha256", "E_RELATION_TRAIN_EVAL_POSE_PATH_ALIAS"),
            ("pose_content_sha256", "E_RELATION_TRAIN_EVAL_POSE_CONTENT_ALIAS"),
            (
                "effective_pose_person_paths",
                "E_RELATION_TRAIN_EVAL_EFFECTIVE_CONSTITUENT_PATH_ALIAS",
            ),
            (
                "effective_pose_person_sha256",
                "E_RELATION_TRAIN_EVAL_EFFECTIVE_CONSTITUENT_CONTENT_ALIAS",
            ),
            (
                "full_pose_person_paths",
                "E_RELATION_TRAIN_EVAL_FULL_CONSTITUENT_PATH_ALIAS",
            ),
            (
                "full_pose_person_sha256",
                "E_RELATION_TRAIN_EVAL_FULL_CONSTITUENT_CONTENT_ALIAS",
            ),
            ("source_pid", "E_RELATION_TRAIN_EVAL_SOURCE_PID_ALIAS"),
        )
        for eval_split in ("query", "gallery"):
            for field, error_code in cases:
                with self.subTest(eval_split=eval_split, field=field):
                    records = self.records_fixture(
                        mirror=False, full_person_count=7)
                    train = records["train"][0]
                    evaluation = records[eval_split][0]
                    if field in {"path", "rgb_sha256", "source_pid"}:
                        changed = replace(
                            evaluation, **{field: getattr(train, field)})
                    elif field == "pose_path_sha256":
                        changed = self.replace_full_assets(
                            evaluation, paths=train.full_pose_person_paths)
                    elif field == "pose_content_sha256":
                        changed = self.replace_full_assets(
                            evaluation, content=train.full_pose_person_sha256)
                    elif field.endswith("paths"):
                        position = 0 if field.startswith("effective") else 6
                        values = list(evaluation.full_pose_person_paths)
                        values[position] = train.full_pose_person_paths[position]
                        changed = self.replace_full_assets(
                            evaluation, paths=values)
                    else:
                        position = 0 if field.startswith("effective") else 6
                        values = list(evaluation.full_pose_person_sha256)
                        values[position] = train.full_pose_person_sha256[position]
                        changed = self.replace_full_assets(
                            evaluation, content=values)
                    records[eval_split][0] = changed
                    self.assert_gate_code(
                        error_code,
                        self.audit,
                        records,
                    )

    def test_query_gallery_path_and_pose_bundle_aliases_fail_closed(self):
        cases = (
            ("path", "E_RELATION_QUERY_GALLERY_RGB_PATH_ALIAS"),
            ("pose_path_sha256", "E_RELATION_QUERY_GALLERY_POSE_PATH_ALIAS"),
            (
                "effective_pose_person_paths",
                "E_RELATION_QUERY_GALLERY_EFFECTIVE_CONSTITUENT_PATH_ALIAS",
            ),
            (
                "full_pose_person_paths",
                "E_RELATION_QUERY_GALLERY_FULL_CONSTITUENT_PATH_ALIAS",
            ),
        )
        for field, error_code in cases:
            with self.subTest(field=field):
                records = self.records_fixture(
                    mirror=True, full_person_count=7)
                query = records["query"][0]
                gallery = records["gallery"][0]
                if field == "path":
                    changed = replace(gallery, path=query.path)
                elif field == "pose_path_sha256":
                    changed = self.replace_full_assets(
                        gallery, paths=query.full_pose_person_paths)
                else:
                    position = 0 if field.startswith("effective") else 6
                    values = list(gallery.full_pose_person_paths)
                    values[position] = query.full_pose_person_paths[position]
                    changed = self.replace_full_assets(gallery, paths=values)
                records["gallery"][0] = changed
                self.assert_gate_code(
                    error_code,
                    self.audit,
                    records,
                )

    def test_partial_nonendpoint_constituent_content_overlap_fails_closed(self):
        cases = (
            (
                "effective_pose_person_sha256",
                "E_RELATION_QUERY_GALLERY_EFFECTIVE_CONSTITUENT_CONTENT_ALIAS",
            ),
            (
                "full_pose_person_sha256",
                "E_RELATION_QUERY_GALLERY_FULL_CONSTITUENT_CONTENT_ALIAS",
            ),
        )
        for field, error_code in cases:
            with self.subTest(field=field):
                records = self.records_fixture(
                    mirror=False, full_person_count=7)
                query = records["query"][0]
                gallery = records["gallery"][0]
                position = 0 if field.startswith("effective") else 6
                values = list(gallery.full_pose_person_sha256)
                values[position] = query.full_pose_person_sha256[position]
                records["gallery"][0] = self.replace_full_assets(
                    gallery, content=values)
                self.assert_gate_code(
                    error_code,
                    self.audit,
                    records,
                )

    def test_cross_endpoint_constituent_content_and_path_fail_closed(self):
        cases = (
            (
                "effective_pose_person_paths",
                "E_RELATION_QUERY_GALLERY_EFFECTIVE_CONSTITUENT_PATH_ALIAS",
            ),
            (
                "full_pose_person_paths",
                "E_RELATION_QUERY_GALLERY_FULL_CONSTITUENT_PATH_ALIAS",
            ),
            (
                "effective_pose_person_sha256",
                "E_RELATION_QUERY_GALLERY_ENDPOINT_MISMATCH",
            ),
            (
                "full_pose_person_sha256",
                "E_RELATION_QUERY_GALLERY_ENDPOINT_MISMATCH",
            ),
        )
        for field, error_code in cases:
            with self.subTest(field=field):
                records = self.two_mirror_fixture(full_person_count=7)
                first = records["gallery"][0]
                second = records["gallery"][1]
                position = 0 if field.startswith("effective") else 6
                if field.endswith("paths"):
                    cross_endpoint_query = records["query"][1]
                    first_values = list(first.full_pose_person_paths)
                    first_values[position] = (
                        cross_endpoint_query.full_pose_person_paths[position])
                    records["gallery"][0] = self.replace_full_assets(
                        first, paths=first_values)
                else:
                    first_values = list(first.full_pose_person_sha256)
                    second_values = list(second.full_pose_person_sha256)
                    first_values[position], second_values[position] = (
                        second_values[position], first_values[position])
                    records["gallery"][0] = self.replace_full_assets(
                        first, content=first_values)
                    records["gallery"][1] = self.replace_full_assets(
                        second, content=second_values)
                self.assert_gate_code(
                    error_code,
                    self.audit,
                    records,
                )

    def test_query_query_gallery_gallery_and_three_member_groups_are_rejected(self):
        for case in ("query_query", "gallery_gallery", "three_member"):
            with self.subTest(case=case):
                records = self.records_fixture(mirror=False)
                if case in {"query_query", "three_member"}:
                    source = records["query"][0]
                    second = self.make_record(
                        "query", 1, "0006_c6_f0000006.jpg", "query-1", 6)
                    content = list(second.full_pose_person_sha256)
                    content[0] = source.full_pose_person_sha256[0]
                    records["query"].append(
                        self.replace_full_assets(second, content=content))
                if case in {"gallery_gallery", "three_member"}:
                    source = (
                        records["query"][0]
                        if case == "three_member"
                        else records["gallery"][0]
                    )
                    second = self.make_record(
                        "gallery", 1, "0007_c7_f0000007.jpg", "gallery-1", 7)
                    content = list(second.full_pose_person_sha256)
                    content[0] = source.full_pose_person_sha256[0]
                    records["gallery"].append(
                        self.replace_full_assets(second, content=content))
                self.assert_gate_code(
                    "E_RELATION_WITHIN_EFFECTIVE_CONSTITUENT_CONTENT_DUPLICATE",
                    self.audit,
                    records,
                )

    def test_pair_count_projection_mismatch_fails_closed(self):
        records = self.records_fixture(mirror=False)
        query_name = Path(records["query"][0].path).name
        records["gallery"][0] = replace(
            records["gallery"][0],
            path=f"/synthetic/rgb/gallery/{query_name}",
        )

        self.assert_gate_code(
            "E_RELATION_PAIR_BASENAME_PROJECTION",
            self.audit,
            records,
        )

    def test_rgb_pose_endpoint_mismatch_fails_closed(self):
        records = self.two_mirror_fixture()
        first = records["gallery"][0]
        second = records["gallery"][1]
        records["gallery"][0] = self.replace_full_assets(
            first, content=second.full_pose_person_sha256)
        records["gallery"][1] = self.replace_full_assets(
            second, content=first.full_pose_person_sha256)

        self.assert_gate_code(
            "E_RELATION_QUERY_GALLERY_ENDPOINT_MISMATCH",
            self.audit,
            records,
        )

    def test_mirror_metadata_drift_matrix_fails_closed(self):
        cases = (
            ("basename", "E_RELATION_PAIR_BASENAME"),
            ("pid", "E_RELATION_PAIR_NOT_JUNK"),
            ("camid", "E_RELATION_PAIR_NOT_JUNK"),
            ("viewid", "E_RELATION_PAIR_METADATA"),
            ("person_count", "E_RELATION_EFFECTIVE_PROJECTION"),
            ("frame", "E_RELATION_PAIR_METADATA"),
            ("report", "E_RELATION_PAIR_METADATA"),
            ("source_pid", "E_RELATION_PAIR_METADATA"),
            ("source_camid", "E_RELATION_PAIR_METADATA"),
            ("source_frame_id", "E_RELATION_PAIR_METADATA"),
            ("target_person_idx", "E_RELATION_PAIR_METADATA"),
        )
        for field, error_code in cases:
            with self.subTest(field=field):
                records = self.records_fixture(
                    mirror=True,
                    full_person_count=(8 if field == "target_person_idx" else 2),
                )
                if field == "target_person_idx":
                    records["query"][0] = replace(
                        records["query"][0], target_person_idx=6)
                gallery = records["gallery"][0]
                if field == "basename":
                    changed = replace(
                        gallery,
                        path="/synthetic/rgb/gallery/9999_c8_f9999999.jpg",
                    )
                elif field == "person_count":
                    changed = replace(
                        gallery,
                        person_count=1,
                        effective_pose_person_relpaths=(
                            gallery.effective_pose_person_relpaths[0],),
                        effective_pose_person_paths=(
                            gallery.effective_pose_person_paths[0],),
                        effective_pose_person_sha256=(
                            gallery.effective_pose_person_sha256[0],),
                    )
                elif field == "report":
                    changed = replace(gallery, report={"drift": True})
                elif field == "target_person_idx":
                    changed = replace(
                        gallery,
                        target_person_idx=7,
                    )
                else:
                    changed = replace(
                        gallery, **{field: int(getattr(gallery, field)) + 1})
                records["gallery"][0] = changed
                self.assert_gate_code(
                    error_code,
                    self.audit,
                    records,
                )

    def test_cache_dtype_shape_byteorder_and_nonfinite_matrix_fails_closed(self):
        cases = []
        for family, dtype, wrong_dtype in (
            ("heatmaps", "<f4", "<f8"),
            ("scores", "<f4", "<f8"),
            ("nuisance", "<f8", "<f4"),
        ):
            cases.extend((
                (family, "dtype", wrong_dtype, "E_RELATION_CACHE_DTYPE"),
                (family, "shape", dtype, "E_RELATION_CACHE_SHAPE"),
                (family, "byteorder", ">" + dtype[1:], "E_RELATION_CACHE_DTYPE"),
                (family, "nonfinite", dtype, "E_RELATION_ARRAY_NONFINITE"),
            ))

        for family, mutation, dtype, error_code in cases:
            with self.subTest(family=family, mutation=mutation):
                records = self.records_fixture(mirror=True)
                caches = self.cache_arrays_for(records)
                value = caches[family]["query"]
                if mutation in {"dtype", "byteorder"}:
                    caches[family]["query"] = value.astype(np.dtype(dtype))
                elif mutation == "shape":
                    caches[family]["query"] = value[..., :-1]
                else:
                    changed = value.copy()
                    changed.reshape(-1)[0] = np.nan
                    caches[family]["query"] = changed
                self.assert_gate_code(
                    error_code,
                    self.audit,
                    records,
                    caches=caches,
                )

    def test_coordinated_query_gallery_cache_drift_changes_frozen_projection(self):
        records = self.records_fixture(mirror=True)
        baseline_caches = self.cache_arrays_for(records)
        baseline = self.audit(records, caches=baseline_caches)
        drifted_caches = copy.deepcopy(baseline_caches)
        drifted_caches["heatmaps"]["query"] += np.float32(1.0)
        drifted_caches["heatmaps"]["gallery"] += np.float32(1.0)

        drifted = self.audit(records, caches=drifted_caches)

        self.assertEqual(drifted["relations"]["allowed_pair_count"], 1)
        self.assertNotEqual(
            baseline["relations"]["query_gallery_joint_pairs"]["sha256"],
            drifted["relations"]["query_gallery_joint_pairs"]["sha256"],
        )
        self.assertNotEqual(
            baseline["relation_report_sha256"],
            drifted["relation_report_sha256"],
        )

    def test_canonical_output_ignores_mapping_insertion_order(self):
        records = self.two_mirror_fixture()
        official_lists = self.official_lists_for(records)
        official_report = self.official_report_for(official_lists)
        caches = self.cache_arrays_for(records)
        baseline = self.audit(
            records,
            official_lists=official_lists,
            official_report=official_report,
            caches=caches,
        )

        reversed_records = dict(reversed(list(records.items())))
        reversed_lists = dict(reversed(list(official_lists.items())))
        reversed_report = dict(reversed(list(official_report.items())))
        reversed_caches = {
            family: dict(reversed(list(by_split.items())))
            for family, by_split in reversed(list(caches.items()))
        }
        reordered = self.audit(
            reversed_records,
            official_lists=reversed_lists,
            official_report=reversed_report,
            caches=reversed_caches,
        )

        self.assertEqual(
            protocol.canonical_json_bytes(baseline),
            protocol.canonical_json_bytes(reordered),
        )


class OfficialExactRelationV1Tests(GateCodeMixin, unittest.TestCase):
    SPLITS = ("train", "query", "gallery")

    @staticmethod
    def resign(report):
        report.pop("relation_report_sha256", None)
        report["relation_report_sha256"] = protocol.sha256_bytes(
            protocol.canonical_json_bytes(report))
        return report

    @staticmethod
    def summary_constant(summary):
        return (
            summary["count"],
            summary["canonical_bytes"],
            summary["sha256"],
        )

    @classmethod
    def contract_constants(cls, report):
        official_splits = {}
        for split, row in report["official_lists"].items():
            official_splits[split] = {
                "root": row["rgb_root"],
                "list": row["list"],
                "count": row["count"],
                "raw_bytes": row["raw_bytes"],
                "raw_sha256": row["raw_sha256"],
                "canonical_bytes": row["canonical_bytes"],
                "canonical_sha256": row["canonical_sha256"],
                "pose_index_bytes": row["pose_index_bytes"],
                "pose_index_sha256": row["pose_index_sha256"],
            }
        relations = report["relations"]
        relation_exact = {
            "shared_basename": cls.summary_constant(
                relations["query_gallery_shared_basenames"]),
            "shared_rgb_legacy": cls.summary_constant(
                relations["query_gallery_shared_rgb_sha256_legacy"]),
            "shared_rgb": cls.summary_constant(
                relations["query_gallery_shared_rgb_sha256"]),
            "endpoint_pairs": cls.summary_constant(
                relations["query_gallery_endpoint_pairs"]["rgb"]),
            "joint_metadata": cls.summary_constant(
                relations["query_gallery_joint_metadata_pairs"]),
            "joint_pairs": cls.summary_constant(
                relations["query_gallery_joint_pairs"]),
        }
        return {
            "OFFICIAL_SPLITS": official_splits,
            "OFFICIAL_SOURCE_PID_COUNTS": {
                split: report["within_split"][split]["source_pid_count"]
                for split in cls.SPLITS
            },
            "OFFICIAL_QUERY_GALLERY_COUNTS": dict(
                report["cross_split"]["query_gallery"]),
            "OFFICIAL_ALLOWED_PAIR_COUNT": len(report["pairs"]),
            "RELATION_EXACT": relation_exact,
        }

    @classmethod
    def contract_patch(cls, report, **overrides):
        constants = cls.contract_constants(report)
        constants.update(overrides)
        return mock.patch.multiple(runner, **constants)

    def official_fixture(self, pair_count=1):
        records = (
            SplitRelationAuditV2Tests.records_fixture(mirror=True)
            if pair_count == 1
            else SplitRelationAuditV2Tests.two_mirror_fixture()
        )
        return SplitRelationAuditV2Tests.audit(records)

    @staticmethod
    def canonical_summary(value):
        payload = protocol.canonical_json_bytes(value)
        return {
            "count": len(value),
            "canonical_bytes": len(payload),
            "sha256": protocol.sha256_bytes(payload),
        }

    def assert_official_code(
        self,
        expected_code,
        report,
        contract_report=None,
        **overrides,
    ):
        frozen_contract = report if contract_report is None else contract_report
        with self.contract_patch(frozen_contract, **overrides):
            self.assert_gate_code(
                expected_code,
                runner.assert_occluded_duke_official_v1,
                report,
            )

    def test_synthetic_official_report_passes_all_exact_closures(self):
        report = self.official_fixture()

        with self.contract_patch(report):
            runner.assert_occluded_duke_official_v1(report)

    def test_every_exact_mapping_rejects_extra_or_missing_keys(self):
        mutations = (
            (
                "report_extra",
                "E_OFFICIAL_REPORT_SCHEMA",
                lambda value: value.__setitem__("extra", 1),
            ),
            (
                "report_missing",
                "E_OFFICIAL_REPORT_SCHEMA",
                lambda value: value.pop("pairs"),
            ),
            (
                "official_list_extra",
                "E_OFFICIAL_LIST_SCHEMA",
                lambda value: value["official_lists"]["train"].__setitem__(
                    "extra", 1),
            ),
            (
                "within_extra",
                "E_OFFICIAL_WITHIN_SCHEMA",
                lambda value: value["within_split"]["train"].__setitem__(
                    "extra", 1),
            ),
            (
                "cross_extra",
                "E_OFFICIAL_CROSS_SCHEMA",
                lambda value: value["cross_split"]["query_gallery"].__setitem__(
                    "extra", 1),
            ),
            (
                "relations_extra",
                "E_OFFICIAL_RELATIONS_SCHEMA",
                lambda value: value["relations"].__setitem__("extra", 1),
            ),
            (
                "summary_extra",
                "E_OFFICIAL_SUMMARY_SCHEMA",
                lambda value: value["relations"][
                    "query_gallery_joint_pairs"].__setitem__("extra", 1),
            ),
            (
                "pair_extra",
                "E_OFFICIAL_PAIR_SCHEMA",
                lambda value: value["pairs"][0].__setitem__("extra", 1),
            ),
            (
                "official_source_missing",
                "E_OFFICIAL_REPORT_SCHEMA",
                lambda value: value["official_source"].pop("commit"),
            ),
            (
                "split_counts_missing",
                "E_OFFICIAL_REPORT_SCHEMA",
                lambda value: value["split_counts"].pop("gallery"),
            ),
            (
                "official_lists_root_missing",
                "E_OFFICIAL_LIST_SCHEMA",
                lambda value: value["official_lists"].pop("gallery"),
            ),
            (
                "official_list_row_missing",
                "E_OFFICIAL_LIST_SCHEMA",
                lambda value: value["official_lists"]["train"].pop(
                    "canonical_sha256"),
            ),
            (
                "within_root_missing",
                "E_OFFICIAL_WITHIN_SCHEMA",
                lambda value: value["within_split"].pop("gallery"),
            ),
            (
                "within_row_missing",
                "E_OFFICIAL_WITHIN_SCHEMA",
                lambda value: value["within_split"]["train"].pop(
                    "path_duplicate_count"),
            ),
            (
                "cross_root_missing",
                "E_OFFICIAL_CROSS_SCHEMA",
                lambda value: value["cross_split"].pop("train_gallery"),
            ),
            (
                "cross_row_missing",
                "E_OFFICIAL_CROSS_SCHEMA",
                lambda value: value["cross_split"]["query_gallery"].pop(
                    "path_overlap_count"),
            ),
            (
                "relations_missing",
                "E_OFFICIAL_RELATIONS_SCHEMA",
                lambda value: value["relations"].pop("allowed_pair_count"),
            ),
            (
                "summary_missing",
                "E_OFFICIAL_SUMMARY_SCHEMA",
                lambda value: value["relations"][
                    "query_gallery_joint_pairs"].pop("sha256"),
            ),
            (
                "endpoint_missing",
                "E_OFFICIAL_RELATIONS_SCHEMA",
                lambda value: value["relations"][
                    "query_gallery_endpoint_pairs"].pop("pose"),
            ),
            (
                "record_sets_missing",
                "E_OFFICIAL_RELATIONS_SCHEMA",
                lambda value: value["relations"]["split_record_sets"].pop(
                    "gallery"),
            ),
            (
                "pair_missing",
                "E_OFFICIAL_PAIR_SCHEMA",
                lambda value: value["pairs"][0].pop("report"),
            ),
        )
        baseline = self.official_fixture()
        for label, error_code, mutate in mutations:
            with self.subTest(case=label):
                changed = copy.deepcopy(baseline)
                mutate(changed)
                self.assert_official_code(
                    error_code, changed, contract_report=baseline)

    def test_bad_nested_types_normalize_to_stable_gate_codes(self):
        mutations = (
            (
                "report",
                "E_OFFICIAL_REPORT_SCHEMA",
                lambda value: value.__setitem__("split_counts", []),
            ),
            (
                "list_bool_count",
                "E_OFFICIAL_LIST_SCHEMA",
                lambda value: value["official_lists"]["train"].__setitem__(
                    "count", True),
            ),
            (
                "list_bad_sha",
                "E_OFFICIAL_LIST_SCHEMA",
                lambda value: value["official_lists"]["train"].__setitem__(
                    "raw_sha256", "not-a-sha"),
            ),
            (
                "within_bool",
                "E_OFFICIAL_WITHIN_SCHEMA",
                lambda value: value["within_split"]["train"].__setitem__(
                    "source_pid_count", True),
            ),
            (
                "cross_negative",
                "E_OFFICIAL_CROSS_SCHEMA",
                lambda value: value["cross_split"]["query_gallery"].__setitem__(
                    "path_overlap_count", -1),
            ),
            (
                "relation_bool",
                "E_OFFICIAL_RELATIONS_SCHEMA",
                lambda value: value["relations"].__setitem__(
                    "allowed_pair_count", True),
            ),
            (
                "summary_bool",
                "E_OFFICIAL_SUMMARY_SCHEMA",
                lambda value: value["relations"][
                    "query_gallery_joint_pairs"].__setitem__("count", True),
            ),
            (
                "endpoint_equal_int",
                "E_OFFICIAL_RELATIONS_SCHEMA",
                lambda value: value["relations"][
                    "query_gallery_endpoint_pairs"].__setitem__("equal", 1),
            ),
            (
                "pairs_mapping",
                "E_OFFICIAL_PAIR_SCHEMA",
                lambda value: value.__setitem__("pairs", {}),
            ),
            (
                "pair_integer_bool",
                "E_OFFICIAL_PAIR_SCHEMA",
                lambda value: value["pairs"][0].__setitem__("pid", True),
            ),
            (
                "pair_bad_sha",
                "E_OFFICIAL_PAIR_SCHEMA",
                lambda value: value["pairs"][0].__setitem__(
                    "rgb_sha256", "bad"),
            ),
        )
        baseline = self.official_fixture()
        for label, error_code, mutate in mutations:
            with self.subTest(case=label):
                changed = copy.deepcopy(baseline)
                mutate(changed)
                self.assert_official_code(
                    error_code, changed, contract_report=baseline)

    def test_official_source_split_list_and_overlap_constants_are_frozen(self):
        mutations = (
            (
                "source",
                "E_OFFICIAL_SOURCE",
                lambda value: value["official_source"].__setitem__(
                    "commit", "0" * 40),
            ),
            (
                "split_count",
                "E_OFFICIAL_SPLIT_COUNT",
                lambda value: (
                    value["split_counts"].__setitem__("train", 2),
                    value["relations"]["split_record_sets"]["train"].__setitem__(
                        "count", 2),
                ),
            ),
            (
                "list_digest",
                "E_OFFICIAL_LIST_DIGEST",
                lambda value: value["official_lists"]["train"].__setitem__(
                    "raw_bytes",
                    value["official_lists"]["train"]["raw_bytes"] + 1,
                ),
            ),
            (
                "target_outside",
                "E_OFFICIAL_TARGET_OUTSIDE",
                lambda value: value["within_split"]["train"].__setitem__(
                    "target_outside_effective_count", 1),
            ),
            (
                "within_duplicate",
                "E_OFFICIAL_WITHIN_DUPLICATE",
                lambda value: value["within_split"]["train"].__setitem__(
                    "path_duplicate_count", 1),
            ),
            (
                "source_pid_count",
                "E_OFFICIAL_SOURCE_PID_COUNT",
                lambda value: value["within_split"]["train"].__setitem__(
                    "source_pid_count", 2),
            ),
            (
                "train_eval_alias",
                "E_OFFICIAL_TRAIN_EVAL_ALIAS",
                lambda value: value["cross_split"]["train_query"].__setitem__(
                    "source_pid_overlap_count", 1),
            ),
            (
                "query_gallery_counts",
                "E_OFFICIAL_QUERY_GALLERY_RELATION",
                lambda value: value["cross_split"][
                    "query_gallery"].__setitem__("source_pid_overlap_count", 0),
            ),
        )
        baseline = self.official_fixture()
        for label, error_code, mutate in mutations:
            with self.subTest(case=label):
                changed = copy.deepcopy(baseline)
                mutate(changed)
                self.resign(changed)
                self.assert_official_code(
                    error_code, changed, contract_report=baseline)

    def test_pair_rows_are_the_authority_for_all_relation_summaries(self):
        baseline = self.official_fixture()
        changed = copy.deepcopy(baseline)
        changed["pairs"] = []
        self.resign(changed)

        self.assert_official_code(
            "E_OFFICIAL_PAIR_RELATION_CLOSURE",
            changed,
            contract_report=baseline,
        )

    def test_pair_field_modification_and_single_sided_summary_drift_fail_closed(self):
        baseline = self.official_fixture()
        for label, mutate in (
            (
                "pair_field",
                lambda value: value["pairs"][0]["report"].__setitem__(
                    "tampered", True),
            ),
            (
                "single_sided_summary",
                lambda value: value["relations"][
                    "query_gallery_joint_pairs"].__setitem__(
                        "sha256", "0" * 64),
            ),
        ):
            with self.subTest(case=label):
                changed = copy.deepcopy(baseline)
                mutate(changed)
                self.resign(changed)
                self.assert_official_code(
                    "E_OFFICIAL_PAIR_RELATION_CLOSURE",
                    changed,
                    contract_report=baseline,
                )

    def test_pair_reordering_and_duplication_fail_pair_schema(self):
        baseline = self.official_fixture(pair_count=2)
        mutations = (
            ("reordered", lambda value: value["pairs"].reverse()),
            (
                "duplicate",
                lambda value: value["pairs"].append(
                    copy.deepcopy(value["pairs"][0])),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(case=label):
                changed = copy.deepcopy(baseline)
                mutate(changed)
                self.resign(changed)
                self.assert_official_code(
                    "E_OFFICIAL_PAIR_SCHEMA",
                    changed,
                    contract_report=baseline,
                )

    def test_coordinated_pair_relation_and_self_resign_still_hits_exact_digest(self):
        baseline = self.official_fixture()
        changed = copy.deepcopy(baseline)
        changed["pairs"][0]["report"]["coordinated_tamper"] = True
        changed["relations"]["query_gallery_joint_pairs"] = (
            self.canonical_summary(changed["pairs"]))
        self.resign(changed)

        self.assert_official_code(
            "E_OFFICIAL_RELATION_DIGEST",
            changed,
            contract_report=baseline,
        )

    def test_frozen_relation_endpoint_and_pair_count_constants_are_enforced(self):
        report = self.official_fixture()
        relation_exact = copy.deepcopy(
            self.contract_constants(report)["RELATION_EXACT"])
        shared_rgb = list(relation_exact["shared_rgb"])
        shared_rgb[2] = "0" * 64
        relation_exact["shared_rgb"] = tuple(shared_rgb)
        self.assert_official_code(
            "E_OFFICIAL_RELATION_DIGEST",
            report,
            RELATION_EXACT=relation_exact,
        )

        endpoint_exact = copy.deepcopy(relation_exact)
        endpoint = list(endpoint_exact["endpoint_pairs"])
        endpoint[2] = "1" * 64
        endpoint_exact["endpoint_pairs"] = tuple(endpoint)
        endpoint_exact["shared_rgb"] = self.summary_constant(
            report["relations"]["query_gallery_shared_rgb_sha256"])
        self.assert_official_code(
            "E_OFFICIAL_ENDPOINT_DIGEST",
            report,
            RELATION_EXACT=endpoint_exact,
        )

        self.assert_official_code(
            "E_OFFICIAL_PAIR_COUNT",
            report,
            OFFICIAL_ALLOWED_PAIR_COUNT=2,
        )

    def test_relation_self_hash_is_independently_enforced(self):
        report = self.official_fixture()
        report["relation_report_sha256"] = "0" * 64

        self.assert_official_code(
            "E_RELATION_REPORT_SELF_HASH",
            report,
        )


class BurnedExecutionPreflightTests(GateCodeMixin, unittest.TestCase):
    def test_main_failure_path_never_rewrites_burned_execution(self):
        for descendant in (False, True):
            with self.subTest(descendant=descendant):
                with tempfile.TemporaryDirectory() as directory:
                    burned = Path(directory) / "burned-a02"
                    execution = burned / "child" if descendant else burned
                    execution.mkdir(parents=True)
                    failed = execution / "FAILED.json"
                    sentinel = b'{"status":"ORIGINAL_MAIN_FAILURE"}\n'
                    failed.write_bytes(sentinel)
                    args = SimpleNamespace(
                        phase="run",
                        execution_dir=str(execution),
                        device="cuda",
                    )
                    failure = runner.GateProtocolError(
                        "E_A02_BURNED_EXECUTION", str(execution))

                    with mock.patch.object(
                        runner, "parse_args", return_value=args,
                    ), mock.patch.object(
                        runner, "run_phase", side_effect=failure,
                    ), mock.patch.object(
                        runner, "BURNED_A02_ROOT", burned,
                    ), mock.patch.object(
                        runner, "atomic_write_json",
                    ) as atomic_write:
                        with self.assertRaises(
                            runner.GateProtocolError,
                        ) as caught:
                            runner.main()

                    self.assertEqual(
                        caught.exception.code, "E_A02_BURNED_EXECUTION")
                    atomic_write.assert_not_called()
                    self.assertEqual(failed.read_bytes(), sentinel)

    def test_failure_writer_never_rewrites_burned_execution(self):
        for descendant in (False, True):
            with self.subTest(descendant=descendant):
                with tempfile.TemporaryDirectory() as directory:
                    burned = Path(directory) / "burned-a02"
                    execution = burned / "child" if descendant else burned
                    execution.mkdir(parents=True)
                    failed = execution / "FAILED.json"
                    sentinel = b'{"status":"ORIGINAL_A02_FAILURE"}\n'
                    failed.write_bytes(sentinel)
                    args = SimpleNamespace(
                        phase="run", execution_dir=str(execution))

                    with mock.patch.object(
                        runner, "BURNED_A02_ROOT", burned,
                    ), mock.patch.object(
                        runner, "atomic_write_json",
                    ) as atomic_write:
                        runner._write_failure_manifest(
                            args,
                            runner.GateProtocolError(
                                "E_SYNTHETIC", "must not be persisted"),
                        )

                    atomic_write.assert_not_called()
                    self.assertEqual(failed.read_bytes(), sentinel)

    def test_prepare_resume_rejects_burned_root_before_output_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.yml"
            config_file.write_bytes(b"synthetic: true\n")
            burned = root / "burned-a02"
            output_root = root / "new-output"
            args = SimpleNamespace(
                config_file=str(config_file),
                opts=[],
                resume=str(burned / "descendant"),
                output_root=str(output_root),
            )
            with mock.patch.object(
                runner.os, "chdir",
            ), mock.patch.object(
                runner, "resolved_config", return_value=object(),
            ), mock.patch.object(
                runner, "BURNED_A02_ROOT", burned,
            ), mock.patch.object(
                Path, "mkdir",
            ) as mkdir, mock.patch.object(
                runner.tempfile, "mkdtemp",
            ) as mkdtemp:
                self.assert_gate_code(
                    "E_A02_BURNED_EXECUTION",
                    runner.prepare_phase,
                    args,
                )
            mkdir.assert_not_called()
            mkdtemp.assert_not_called()
            self.assertFalse(output_root.exists())

    def test_run_and_summarize_reject_burned_execution_before_consumers(self):
        for phase_function in (runner.run_phase, runner.summarize_phase):
            for descendant in (False, True):
                with self.subTest(
                    phase=phase_function.__name__, descendant=descendant):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        burned = root / "burned-a02"
                        execution = burned / "child" if descendant else burned
                        args = SimpleNamespace(
                            execution_dir=str(execution), device="cuda")
                        with mock.patch.object(
                            runner.os, "chdir",
                        ), mock.patch.object(
                            runner, "BURNED_A02_ROOT", burned,
                        ), mock.patch.object(
                            runner, "exclusive_execution_lock",
                        ) as lock, mock.patch.object(
                            runner, "verify_prepared_artifacts",
                        ) as verify, mock.patch.object(
                            runner.torch.cuda, "is_available",
                        ) as cuda_available:
                            self.assert_gate_code(
                                "E_A02_BURNED_EXECUTION",
                                phase_function,
                                args,
                            )
                        lock.assert_not_called()
                        verify.assert_not_called()
                        cuda_available.assert_not_called()


class RelationArtifactTripleTests(GateCodeMixin, unittest.TestCase):
    @staticmethod
    def make_triple(root):
        prepared = Path(root) / "prepared"
        prepared.mkdir()
        report = {"schema": "synthetic-relation-v2", "value": 1}
        payload = protocol.canonical_json_bytes(report)
        digest = protocol.sha256_bytes(payload)
        protocol.atomic_write_bytes(prepared / "split_relations.json", payload)
        manifest = {
            "dataset": {
                "split_relations": dict(report),
                "split_relations_artifact": {
                    "relpath": "split_relations.json",
                    "bytes": len(payload),
                    "sha256": digest,
                },
            },
            "prepared_artifact_sha256": {"split_relations.json": digest},
        }
        return manifest, prepared, report

    def verify_triple(self, manifest, prepared):
        with mock.patch.object(runner, "assert_occluded_duke_official_v1"):
            return runner._verify_relation_artifact_triple(manifest, prepared)

    def test_valid_object_artifact_and_prepared_hash_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, prepared, report = self.make_triple(directory)
            self.assertEqual(self.verify_triple(manifest, prepared), report)

    def test_object_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, prepared, _report = self.make_triple(directory)
            manifest["dataset"]["split_relations"]["value"] = 2
            self.assert_gate_code(
                "E_RELATION_PREPARED_TRIPLE",
                self.verify_triple,
                manifest,
                prepared,
            )

    def test_artifact_descriptor_bytes_and_sha_drift_are_rejected(self):
        for field, value in (("bytes", 1), ("sha256", "wrong-sha")):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    manifest, prepared, _report = self.make_triple(directory)
                    manifest["dataset"]["split_relations_artifact"][field] = value
                    self.assert_gate_code(
                        "E_RELATION_PREPARED_TRIPLE",
                        self.verify_triple,
                        manifest,
                        prepared,
                    )

    def test_prepared_hash_and_artifact_file_drift_are_rejected(self):
        for label in ("prepared_sha", "artifact_bytes"):
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as directory:
                    manifest, prepared, _report = self.make_triple(directory)
                    if label == "prepared_sha":
                        manifest["prepared_artifact_sha256"][
                            "split_relations.json"] = "wrong-sha"
                    else:
                        (prepared / "split_relations.json").write_bytes(b"tampered\n")
                    self.assert_gate_code(
                        "E_RELATION_PREPARED_TRIPLE",
                        self.verify_triple,
                        manifest,
                        prepared,
                    )


class LexicalRelationAssetPathTests(GateCodeMixin, unittest.TestCase):
    @staticmethod
    def dataset(root):
        root = Path(root)
        return SimpleNamespace(
            dataset_dir=str(root),
            train_list=str(root / "train.list"),
            query_list=str(root / "query.list"),
            gallery_list=str(root / "gallery.list"),
            train_dir=str(root / "bounding_box_train"),
            query_dir=str(root / "query"),
            gallery_dir=str(root / "bounding_box_test"),
        )

    @staticmethod
    def split_datasets(root):
        root = Path(root)
        return {
            split: SimpleNamespace(
                max_persons=6,
                pose_dir=str(root / "pose_data" / split),
            )
            for split in ("train", "query", "gallery")
        }

    @staticmethod
    def patched_official_lists():
        return (
            {"train": [], "query": [], "gallery": []},
            {"train": {}, "query": {}, "gallery": {}},
        )

    def test_official_list_rejects_alternate_lexical_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            alternate = root / "alternate-train.list"
            alternate.write_bytes(b"0001_c1_f0000001.jpg\n")
            dataset = self.dataset(root)
            dataset.train_list = str(alternate)

            self.assert_gate_code(
                "E_OFFICIAL_LIST_PATH",
                runner._official_lists_v2,
                dataset,
                {},
            )

    def test_official_list_rejects_final_component_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            actual = root / "actual-train.list"
            actual.write_bytes(b"0001_c1_f0000001.jpg\n")
            (root / "train.list").symlink_to(actual)
            dataset = self.dataset(root)

            self.assert_gate_code(
                "E_OFFICIAL_LIST_PATH",
                runner._official_lists_v2,
                dataset,
                {},
            )

    def test_official_list_rejects_prefix_parent_traversal_spelling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            dataset = self.dataset(root)
            dataset.train_list = str(
                root / "prefix" / ".." / "train.list")

            self.assert_gate_code(
                "E_OFFICIAL_LIST_PATH",
                runner._official_lists_v2,
                dataset,
                {},
            )

    def test_rgb_root_rejects_final_component_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "dataset"
            root.mkdir()
            actual = workspace / "actual-train-rgb"
            actual.mkdir()
            (root / "bounding_box_train").symlink_to(
                actual, target_is_directory=True)
            dataset = self.dataset(root)

            with mock.patch.object(
                runner,
                "_official_lists_v2",
                return_value=self.patched_official_lists(),
            ):
                self.assert_gate_code(
                    "E_RELATION_RGB_ROOT",
                    runner.build_relation_report_v2,
                    dataset,
                    self.split_datasets(root),
                    workspace / "prepared",
                )

    def test_rgb_root_rejects_prefix_parent_traversal_spelling(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "dataset"
            root.mkdir()
            dataset = self.dataset(root)
            dataset.train_dir = str(
                root / "prefix" / ".." / "bounding_box_train")

            with mock.patch.object(
                runner,
                "_official_lists_v2",
                return_value=self.patched_official_lists(),
            ):
                self.assert_gate_code(
                    "E_RELATION_RGB_ROOT",
                    runner.build_relation_report_v2,
                    dataset,
                    self.split_datasets(root),
                    workspace / "prepared",
                )

    def test_pose_base_and_split_reject_final_component_symlink(self):
        for symlink_scope in ("base", "split"):
            with self.subTest(scope=symlink_scope):
                with tempfile.TemporaryDirectory() as directory:
                    workspace = Path(directory)
                    root = workspace / "dataset"
                    root.mkdir()
                    (root / "bounding_box_train").mkdir()
                    if symlink_scope == "base":
                        actual = workspace / "actual-pose-base"
                        actual.mkdir()
                        (root / "pose_data").symlink_to(
                            actual, target_is_directory=True)
                    else:
                        pose_base = root / "pose_data"
                        pose_base.mkdir()
                        actual = workspace / "actual-pose-train"
                        actual.mkdir()
                        (pose_base / "train").symlink_to(
                            actual, target_is_directory=True)
                    dataset = self.dataset(root)

                    with mock.patch.object(
                        runner,
                        "_official_lists_v2",
                        return_value=self.patched_official_lists(),
                    ):
                        self.assert_gate_code(
                            "E_RELATION_POSE_ROOT",
                            runner.build_relation_report_v2,
                            dataset,
                            self.split_datasets(root),
                            workspace / "prepared",
                        )

    def test_pose_root_rejects_prefix_parent_traversal_spelling(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "dataset"
            root.mkdir()
            (root / "bounding_box_train").mkdir()
            dataset = self.dataset(root)
            split_datasets = self.split_datasets(root)
            split_datasets["train"].pose_dir = str(
                root / "prefix" / ".." / "pose_data" / "train")

            with mock.patch.object(
                runner,
                "_official_lists_v2",
                return_value=self.patched_official_lists(),
            ):
                self.assert_gate_code(
                    "E_RELATION_POSE_ROOT",
                    runner.build_relation_report_v2,
                    dataset,
                    split_datasets,
                    workspace / "prepared",
                )

    def test_deployment_symlink_root_accepts_canonical_real_children(self):
        class PassedLexicalGate(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            real_root = workspace / "real-dataset"
            real_root.mkdir()
            deployment_root = workspace / "deployment-dataset"
            deployment_root.symlink_to(real_root, target_is_directory=True)
            (real_root / "train.list").write_bytes(
                b"0001_c1_f0000001.jpg\n")
            dataset = self.dataset(deployment_root)

            with mock.patch.object(
                runner,
                "_stable_regular_file",
                side_effect=PassedLexicalGate("list gate passed"),
            ):
                with self.assertRaisesRegex(
                    PassedLexicalGate, "list gate passed"):
                    runner._official_lists_v2(dataset, {})

            (real_root / "bounding_box_train").mkdir()
            (real_root / "pose_data").mkdir()
            (real_root / "pose_data" / "train").mkdir()
            with mock.patch.object(
                runner,
                "_official_lists_v2",
                return_value=self.patched_official_lists(),
            ), mock.patch.object(
                runner,
                "_stable_json",
                side_effect=PassedLexicalGate("root gates passed"),
            ):
                with self.assertRaisesRegex(
                    PassedLexicalGate, "root gates passed"):
                    runner.build_relation_report_v2(
                        dataset,
                        self.split_datasets(deployment_root),
                        workspace / "prepared",
                    )


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
            quick_identity=synthetic_quick_identity(root),
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
            "dataset": {
                "cache": {
                    split: {
                        "signed_raw_audit": {
                            "actual_space": {
                                "active_psg_blocks": {
                                    key: {"sraw_sha256": f"{split}-{key}-sraw"}
                                    for key in runner.ACTIVE_PSG_BLOCK_SHAPES
                                },
                            },
                        },
                    }
                    for split in ("query", "gallery")
                },
            },
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
        self.assertEqual(provenance["correct_actual_sraw_sha256"]["query"], {
            "s3_b0": "query-s3_b0-sraw",
            "s3_b1": "query-s3_b1-sraw",
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
            runner.publish_arm(
                temporary,
                published,
                provenance,
                quick_identity=synthetic_quick_identity(root),
            )

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

    def test_quick_identity_drift_blocks_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"seed": 42, "arm": "shuffle", "mapping": 0}
            provenance = self.provenance(row)
            temporary = root / f".{runner.schedule_arm_id(row)}.tmp-arm"
            published = root / runner.schedule_arm_id(row)
            temporary.mkdir()
            self.write_summary(temporary, row, provenance, "PASS")
            self.write_per_query(temporary)
            quick_identity = synthetic_quick_identity(root)
            changed = Path(sorted(quick_identity)[0])
            changed.write_bytes(changed.read_bytes() + b"drift")

            self.assert_gate_code(
                "E_RELATION_FILE_TOCTOU",
                runner.publish_arm,
                temporary,
                published,
                provenance,
                quick_identity=quick_identity,
            )
            self.assertTrue(temporary.is_dir())
            self.assertFalse(published.exists())

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

    def test_runtime_centroid_protocol_failure_is_secondary_and_primary_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcomes = []
            quick_identity = synthetic_quick_identity(root)
            rows = (
                {"seed": 42, "arm": "centroid"},
                {"seed": 42, "arm": "shuffle", "mapping": 0},
            )
            for row in rows:
                provenance = self.provenance(row)
                temporary = root / f".{runner.schedule_arm_id(row)}.tmp-arm"
                published = root / runner.schedule_arm_id(row)
                temporary.mkdir()

                def synthetic_extract(current=row, temp=temporary, prov=provenance):
                    if current["arm"] == "centroid":
                        (temp / "partial.bin").write_bytes(b"must-be-removed")
                        raise protocol.GateProtocolError(
                            "E_CENTROID_NEGATIVE_L1", "synthetic crop")
                    self.write_summary(temp, current, prov, "PASS")
                    self.write_per_query(temp)

                outcomes.append(runner.extract_and_publish_unpublished_arm(
                    synthetic_extract,
                    temporary,
                    published,
                    row,
                    provenance,
                    quick_identity,
                ))

            self.assertEqual(outcomes, ["INVALID_SECONDARY", "PASS"])
            centroid_dir = root / "seed_42__centroid"
            centroid_summary = json.loads(
                (centroid_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(centroid_summary["status"], "INVALID_SECONDARY")
            self.assertEqual(
                centroid_summary["reason"]["error_code"],
                "E_CENTROID_NEGATIVE_L1",
            )
            self.assertEqual(
                centroid_summary["reason"]["phase"], "runtime_extract")
            self.assertFalse((centroid_dir / "per_query.npz").exists())
            self.assertFalse((centroid_dir / "partial.bin").exists())
            self.assertEqual(
                runner.verify_published_arm(
                    centroid_dir, self.provenance(rows[0]))["status"],
                "INVALID_SECONDARY",
            )
            self.assertEqual(
                runner.verify_published_arm(
                    root / "seed_42__shuffle_m00", self.provenance(rows[1]))["status"],
                "PASS",
            )

    def test_runtime_primary_protocol_failure_remains_global_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"seed": 42, "arm": "shuffle", "mapping": 0}
            provenance = self.provenance(row)
            quick_identity = synthetic_quick_identity(root)
            temporary = root / ".seed_42__shuffle_m00.tmp-arm"
            published = root / "seed_42__shuffle_m00"
            temporary.mkdir()

            def fail_primary():
                (temporary / "partial.bin").write_bytes(b"preserved-for-diagnosis")
                raise protocol.GateProtocolError("E_PRIMARY", "synthetic")

            self.assert_gate_code(
                "E_PRIMARY",
                runner.extract_and_publish_unpublished_arm,
                fail_primary,
                temporary,
                published,
                row,
                provenance,
                quick_identity,
            )
            self.assertTrue((temporary / "partial.bin").is_file())
            self.assertFalse(published.exists())

    def test_runtime_centroid_integrity_failure_remains_global_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"seed": 42, "arm": "centroid"}
            provenance = self.provenance(row)
            quick_identity = synthetic_quick_identity(root)
            temporary = root / ".seed_42__centroid.tmp-arm"
            published = root / "seed_42__centroid"
            temporary.mkdir()

            def fail_integrity():
                (temporary / "partial.bin").write_bytes(b"preserved-for-diagnosis")
                raise protocol.GateProtocolError(
                    "E_RUNTIME_RGB_TOCTOU", "synthetic integrity failure")

            self.assert_gate_code(
                "E_RUNTIME_RGB_TOCTOU",
                runner.extract_and_publish_unpublished_arm,
                fail_integrity,
                temporary,
                published,
                row,
                provenance,
                quick_identity,
            )
            self.assertTrue((temporary / "partial.bin").is_file())
            self.assertFalse(published.exists())

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

    def test_execution_resume_rejects_signed_premetric_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            signed = {
                "transform": "positive_part_v1",
                "negative_channel_indices_0based": [6, 10],
                "active_psg_blocks": {
                    "s3_b0": {"sraw_sha256": "raw", "delta_sum_abs": 1.0},
                },
            }
            manifest = self.base_manifest()
            manifest["dataset"] = {"cache": {"train": {"signed_raw_audit": signed}}}
            execution, _sha = protocol.create_execution_directory(
                output_root, manifest, None)
            frozen = json.loads(
                (execution / "premetric_manifest.json").read_text(encoding="utf-8"))
            frozen["dataset"]["cache"]["train"]["signed_raw_audit"][
                "negative_channel_indices_0based"] = [6]
            protocol.atomic_write_json(
                execution / "premetric_manifest.json", frozen)

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
