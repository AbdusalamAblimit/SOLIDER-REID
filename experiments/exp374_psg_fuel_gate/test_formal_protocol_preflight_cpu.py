"""CPU-only formal preflight for the complete exp374 matching protocol.

This test uses a fixed synthetic fixture and the real candidate builder, 1,000
baseline matchings, 20 Gumbel min-cost matchings, pair-quality gates, Hamming
gate, and persistence seam.  It does not load a dataset, model, checkpoint, or
CUDA tensor.  Merely creating this file does not authorize executing it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import importlib.metadata
import numpy as np
import pytest
import torch

from experiments.exp374_psg_fuel_gate import protocol as p


FIXTURE_COUNT = 128
NUISANCE_DIMENSION = 95
ANCHOR_CHUNK = 16
EXPECTED_EDGE_COUNT = FIXTURE_COUNT * (FIXTURE_COUNT - 1)
MAX_TEMPORARY_BYTES = 16 * 1024 * 1024


def _fixture_records() -> list[p.SceneRecord]:
    return [
        p.SceneRecord(
            metadata_schema=p.SCENE_METADATA_SCHEMA_V2,
            index=index,
            split="query",
            path=f"/synthetic/q_{index:03d}.jpg",
            rgb_sha256=f"rgb-{index:03d}",
            pose_path_sha256=f"pose-path-{index:03d}",
            pose_content_sha256=f"pose-content-{index:03d}",
            pid=index,
            camid=0,
            viewid=0,
            person_count=1,
            continuous=tuple([0.0] * NUISANCE_DIMENSION),
            frame=0,
            report={},
            source_pid=index,
            source_camid=0,
            source_frame_id=0,
            target_person_idx=0,
            full_pose_person_relpaths=(
                f"pose_data/query/person-{index:03d}.npz",
            ),
            full_pose_person_paths=(
                f"/synthetic/pose_data/query/person-{index:03d}.npz",
            ),
            full_pose_person_sha256=(f"person-content-{index:03d}",),
            effective_pose_person_relpaths=(
                f"pose_data/query/person-{index:03d}.npz",
            ),
            effective_pose_person_paths=(
                f"/synthetic/pose_data/query/person-{index:03d}.npz",
            ),
            effective_pose_person_sha256=(f"person-content-{index:03d}",),
        )
        for index in range(FIXTURE_COUNT)
    ]


def _empty_summary() -> dict[str, object]:
    payload = p.canonical_json_bytes([])
    return {
        "count": 0,
        "canonical_bytes": len(payload),
        "sha256": p.sha256_bytes(payload),
    }


def _within(records: Sequence[p.SceneRecord]) -> dict[str, int]:
    def duplicates(values: Sequence[object]) -> int:
        return len(values) - len(set(values))

    full_paths = [value for record in records for value in record.full_pose_person_paths]
    full_content = [value for record in records for value in record.full_pose_person_sha256]
    effective_paths = [
        value for record in records for value in record.effective_pose_person_paths
    ]
    effective_content = [
        value for record in records for value in record.effective_pose_person_sha256
    ]
    return {
        "path_duplicate_count": duplicates([record.path for record in records]),
        "rgb_sha256_duplicate_count": duplicates(
            [record.rgb_sha256 for record in records]),
        "pose_path_sha256_duplicate_count": duplicates(
            [record.pose_path_sha256 for record in records]),
        "pose_content_sha256_duplicate_count": duplicates(
            [record.pose_content_sha256 for record in records]),
        "full_pose_person_path_duplicate_count": duplicates(full_paths),
        "full_pose_person_content_duplicate_count": duplicates(full_content),
        "effective_pose_person_path_duplicate_count": duplicates(effective_paths),
        "effective_pose_person_content_duplicate_count": duplicates(effective_content),
        "source_pid_count": len({record.source_pid for record in records}),
        "target_outside_effective_count": 0,
    }


def _fixture_relation_report(
    records: Sequence[p.SceneRecord],
) -> dict[str, object]:
    empty_summary = _empty_summary()
    empty_records: list[p.SceneRecord] = []
    report: dict[str, object] = {
        "schema": p.SPLIT_RELATION_SCHEMA_V2,
        "official_source": {},
        "official_lists": {},
        "split_counts": {
            "train": 0,
            "query": len(records),
            "gallery": 0,
        },
        "within_split": {
            "train": _within(empty_records),
            "query": _within(records),
            "gallery": _within(empty_records),
        },
        "cross_split": {},
        "relations": {
            "query_gallery_shared_basenames": dict(empty_summary),
            "query_gallery_shared_rgb_sha256_legacy": dict(empty_summary),
            "query_gallery_shared_rgb_sha256": dict(empty_summary),
            "query_gallery_endpoint_pairs": {
                "equal": True,
                "rgb": dict(empty_summary),
                "pose": dict(empty_summary),
            },
            "query_gallery_joint_metadata_pairs": dict(empty_summary),
            "query_gallery_joint_pairs": dict(empty_summary),
            "split_record_sets": {
                "train": p.canonical_scene_record_set_summary(empty_records),
                "query": p.canonical_scene_record_set_summary(records),
                "gallery": p.canonical_scene_record_set_summary(empty_records),
            },
            "allowed_pair_count": 0,
            "junk_true_count": 0,
            "junk_false_count": 0,
            "forbidden_pair_count": 0,
        },
        "pairs": [],
    }
    report["relation_report_sha256"] = p.sha256_bytes(
        p.canonical_json_bytes(report))
    return report


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = p.canonical_json_bytes({
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    })
    return p.sha256_bytes(header + value.tobytes(order="C"))


def _mapping_hashes(mappings: Sequence[np.ndarray]) -> list[str]:
    return [_array_sha256(np.asarray(mapping, dtype=np.int64)) for mapping in mappings]


def _assert_bijection_and_edges(
    mapping: np.ndarray,
    adjacency: Sequence[Sequence[int]],
) -> None:
    assert mapping.shape == (FIXTURE_COUNT,)
    assert sorted(mapping.tolist()) == list(range(FIXTURE_COUNT))
    for left, right in enumerate(mapping.tolist()):
        assert right != left
        assert right in adjacency[left]


def _pairwise_hamming_values(mappings: Sequence[np.ndarray]) -> list[float]:
    return [
        float(np.mean(np.asarray(mappings[first]) != np.asarray(mappings[second])))
        for first in range(len(mappings))
        for second in range(first + 1, len(mappings))
    ]


def _assert_quality_round_trip(
    records: Sequence[p.SceneRecord],
    standardized: np.ndarray,
    mappings: Sequence[np.ndarray],
    base_edges: Mapping[Tuple[int, int], float],
    baseline_mean_costs: Sequence[float],
    frozen_audits: Sequence[Mapping[str, object]],
) -> None:
    exact_keys = (
        "mean_cost",
        "p95_cost",
        "max_dimension_median_abs_z",
        "baseline_median_mean_cost",
        "max_marginal_mean_error",
        "max_empirical_ks",
    )
    for mapping_index, mapping in enumerate(mappings):
        recomputed = p.audit_mapping(
            records,
            standardized,
            np.asarray(mapping, dtype=np.int64),
            base_edges,
            baseline_mean_costs,
        )
        for key in exact_keys:
            assert recomputed[key] == pytest.approx(
                float(frozen_audits[mapping_index][key]), abs=1e-15)
        assert abs(recomputed["mean_cost"]) <= 1e-12
        assert abs(recomputed["p95_cost"]) <= 1e-12
        assert recomputed["max_dimension_median_abs_z"] == 0.0
        assert recomputed["max_marginal_mean_error"] == 0.0
        assert recomputed["max_empirical_ks"] == 0.0
        assert np.isfinite(float(
            frozen_audits[mapping_index]["randomized_objective_float64"]))


def _assert_payloads_reproducible(first: Mapping[str, object],
                                  second: Mapping[str, object]) -> None:
    np.testing.assert_array_equal(first["standardized"], second["standardized"])
    assert first["scaler"] == second["scaler"]
    assert first["selected_adjacency"] == second["selected_adjacency"]
    assert first["base_edges"] == second["base_edges"]
    assert first["strata"] == second["strata"]
    assert first["eta_by_seed"] == second["eta_by_seed"]
    assert first["solver"] == second["solver"]
    np.testing.assert_array_equal(
        np.asarray(first["baseline_mean_costs"], dtype=np.float64),
        np.asarray(second["baseline_mean_costs"], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.stack(first["mappings"]),
        np.stack(second["mappings"]),
    )
    np.testing.assert_array_equal(
        np.stack(first["randomized_edge_costs"]),
        np.stack(second["randomized_edge_costs"]),
    )
    assert first["mapping_audits"] == second["mapping_audits"]
    assert first["minimum_hamming"] == second["minimum_hamming"]
    assert first["effective_unique_count"] == second["effective_unique_count"]


def test_formal_protocol_preflight_cpu(tmp_path: Path) -> None:
    # Structural resource limits are deterministic and avoid brittle wall-time
    # or process-RSS thresholds that vary across CI and BLAS implementations.
    assert ANCHOR_CHUNK * FIXTURE_COUNT * NUISANCE_DIMENSION * 8 < 2 * 1024 * 1024
    assert EXPECTED_EDGE_COUNT == 16_256
    assert len(p.MAPPING_SEEDS) * EXPECTED_EDGE_COUNT * 8 < 3 * 1024 * 1024
    assert len(p.BASELINE_SEEDS) == 20
    assert len(p.MAPPING_SEEDS) == 1

    device = torch.device("cpu")
    assert device.type == "cpu"
    records = _fixture_records()
    assert len(records) == FIXTURE_COUNT
    assert len({record.path for record in records}) == FIXTURE_COUNT
    assert len({record.pid for record in records}) == FIXTURE_COUNT
    assert len({record.rgb_sha256 for record in records}) == FIXTURE_COUNT
    assert len({record.pose_path_sha256 for record in records}) == FIXTURE_COUNT
    assert len({record.pose_content_sha256 for record in records}) == FIXTURE_COUNT
    fixture_hash = p.canonical_scene_record_set_summary(records)["sha256"]
    assert fixture_hash == p.canonical_scene_record_set_summary(
        _fixture_records())["sha256"]
    relation_report = _fixture_relation_report(records)
    assert p.validate_relation_report_self_hash(relation_report) == (
        relation_report["relation_report_sha256"]
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert torch.get_num_threads() == 1
        first = p.prepare_split_mappings(
            records,
            device=device,
            anchor_chunk=ANCHOR_CHUNK,
            relation_report=relation_report,
            split="query",
        )
        second = p.prepare_split_mappings(
            records,
            device=device,
            anchor_chunk=ANCHOR_CHUNK,
            relation_report=relation_report,
            split="query",
        )
    finally:
        torch.set_num_threads(previous_threads)

    _assert_payloads_reproducible(first, second)
    assert first["solver"]["scipy_version"] == importlib.metadata.version("scipy")
    assert first["scaler"]["constant_dims"] == list(range(NUISANCE_DIMENSION))
    assert np.array_equal(
        first["standardized"],
        np.zeros((FIXTURE_COUNT, NUISANCE_DIMENSION), dtype=np.float64),
    )
    assert first["strata"] == {
        "1": {
            "count": FIXTURE_COUNT,
            "selected_k": FIXTURE_COUNT - 1,
            "indices": list(range(FIXTURE_COUNT)),
        },
    }

    adjacency = first["selected_adjacency"]
    assert len(adjacency) == FIXTURE_COUNT
    for left, donors in enumerate(adjacency):
        assert donors == [right for right in range(FIXTURE_COUNT) if right != left]
    base_edges = first["base_edges"]
    assert len(base_edges) == EXPECTED_EDGE_COUNT
    assert max(abs(float(value)) for value in base_edges.values()) <= 1e-15
    assert all(eta == {"1": 0.01} for eta in first["eta_by_seed"])

    edge_order = sorted(base_edges)
    randomized_costs = np.stack(first["randomized_edge_costs"])
    assert randomized_costs.shape == (len(p.MAPPING_SEEDS), EXPECTED_EDGE_COUNT)
    for mapping_index, seed in enumerate(p.MAPPING_SEEDS):
        noise = np.random.Generator(
            np.random.PCG64DXSM(seed)).gumbel(size=EXPECTED_EDGE_COUNT)
        expected = np.asarray([
            float(base_edges[edge]) + 0.01 * noise[edge_index]
            for edge_index, edge in enumerate(edge_order)
        ], dtype=np.float64)
        np.testing.assert_array_equal(randomized_costs[mapping_index], expected)

    mappings = [np.asarray(mapping, dtype=np.int64) for mapping in first["mappings"]]
    assert len(mappings) == len(p.MAPPING_SEEDS)
    assert len(set(_mapping_hashes(mappings))) == 1
    for mapping in mappings:
        _assert_bijection_and_edges(mapping, adjacency)
    hamming_values = _pairwise_hamming_values(mappings)
    assert hamming_values == []
    assert first["minimum_hamming"] == pytest.approx(1.0, abs=0.0)
    assert first["effective_unique_count"] == 1

    baseline = np.asarray(first["baseline_mean_costs"], dtype=np.float64)
    assert baseline.shape == (len(p.BASELINE_SEEDS),)
    assert np.isfinite(baseline).all()
    assert np.max(np.abs(baseline)) <= 1e-12
    for seed in (p.BASELINE_SEEDS[0], p.BASELINE_SEEDS[len(p.BASELINE_SEEDS) // 2],
                 p.BASELINE_SEEDS[-1]):
        baseline_first = p.randomized_full_matching(adjacency, seed)
        baseline_second = p.randomized_full_matching(adjacency, seed)
        np.testing.assert_array_equal(baseline_first, baseline_second)
        _assert_bijection_and_edges(baseline_first, adjacency)

    _assert_quality_round_trip(
        records,
        np.asarray(first["standardized"], dtype=np.float64),
        mappings,
        base_edges,
        baseline,
        first["mapping_audits"],
    )
    edge_lookup_by_mapping = [
        {
            edge: float(randomized_costs[mapping_index, edge_index])
            for edge_index, edge in enumerate(edge_order)
        }
        for mapping_index in range(len(p.MAPPING_SEEDS))
    ]
    for mapping_index, mapping in enumerate(mappings):
        objective = float(np.sum(np.asarray([
            edge_lookup_by_mapping[mapping_index][(left, int(right))]
            for left, right in enumerate(mapping)
        ], dtype=np.float64), dtype=np.float64))
        assert objective == pytest.approx(
            float(first["mapping_audits"][mapping_index][
                "randomized_objective_float64"]), abs=1e-15)

    # Lazy import: only the persistence helper is called.  No dataset, model,
    # checkpoint, CUDA, inference, or training function is constructed/called.
    from experiments.exp374_psg_fuel_gate.audit_gate_a import save_mapping_payload

    manifest = save_mapping_payload(tmp_path, "query", first)
    assert manifest == {
        "mappings": "query_mappings.npy",
        "candidate_graph": "query_candidate_graph.npz",
        "audit": "query_mapping_audit.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == set(manifest.values())
    total_bytes = sum(path.stat().st_size for path in tmp_path.iterdir())
    assert total_bytes <= MAX_TEMPORARY_BYTES

    persisted_mappings = np.load(tmp_path / manifest["mappings"], allow_pickle=False)
    assert persisted_mappings.dtype == np.int32
    assert persisted_mappings.shape == (len(p.MAPPING_SEEDS), FIXTURE_COUNT)
    np.testing.assert_array_equal(persisted_mappings, np.stack(mappings).astype(np.int32))

    with np.load(tmp_path / manifest["candidate_graph"], allow_pickle=False) as graph:
        assert set(graph.files) == {
            "offsets", "donors", "edge_left", "edge_right", "edge_cost",
            "randomized_edge_cost", "baseline_mean_costs",
        }
        offsets = graph["offsets"].copy()
        donors = graph["donors"].copy()
        persisted_edges = list(zip(
            graph["edge_left"].tolist(), graph["edge_right"].tolist()))
        assert offsets.shape == (FIXTURE_COUNT + 1,)
        np.testing.assert_array_equal(
            np.diff(offsets),
            np.full(FIXTURE_COUNT, FIXTURE_COUNT - 1, dtype=offsets.dtype),
        )
        assert donors.shape == (EXPECTED_EDGE_COUNT,)
        assert persisted_edges == edge_order
        np.testing.assert_allclose(
            graph["edge_cost"],
            np.asarray([base_edges[edge] for edge in edge_order], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(graph["randomized_edge_cost"], randomized_costs)
        np.testing.assert_array_equal(graph["baseline_mean_costs"], baseline)
        for left in range(FIXTURE_COUNT):
            start, stop = int(offsets[left]), int(offsets[left + 1])
            assert donors[start:stop].tolist() == adjacency[left]

    frozen_audit = json.loads(
        (tmp_path / manifest["audit"]).read_text(encoding="utf-8"))
    assert frozen_audit["mapping_seeds"] == list(p.MAPPING_SEEDS)
    assert frozen_audit["baseline_seeds"] == list(p.BASELINE_SEEDS)
    assert frozen_audit["solver"] == first["solver"]
    assert frozen_audit["candidate_k_sequence"] == list(p.K_SEQUENCE)
    assert "all_cost_arithmetic=float64" in frozen_audit["cost_formula_version"]
    assert frozen_audit["minimum_hamming"] == pytest.approx(
        float(first["minimum_hamming"]), abs=0.0)
    assert frozen_audit["effective_unique_count"] == 1

    persisted_as_int64 = [row.astype(np.int64, copy=False) for row in persisted_mappings]
    persisted_hamming = _pairwise_hamming_values(persisted_as_int64)
    assert persisted_hamming == hamming_values
    _assert_quality_round_trip(
        records,
        np.asarray(first["standardized"], dtype=np.float64),
        persisted_as_int64,
        base_edges,
        baseline,
        frozen_audit["mapping_audits"],
    )
