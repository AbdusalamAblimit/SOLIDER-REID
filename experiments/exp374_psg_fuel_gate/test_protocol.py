"""Pure-synthetic unit tests for the frozen exp374 Gate-A protocol.

The fixtures deliberately avoid real datasets and checkpoints.  They bind the
production helpers directly and assert stable ``GateProtocolError.code`` values
for every fail-closed path covered here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pytest
import torch

from experiments.exp374_psg_fuel_gate import protocol as p


def _record(
    index: int,
    path: str,
    *,
    pid: int | None = None,
    camid: int = 0,
    viewid: int = 0,
    person_count: int = 1,
    frame: int = 0,
    split: str = "query",
    metadata_schema: str = p.SCENE_METADATA_SCHEMA_V2,
    continuous: Tuple[float, ...] | None = None,
    full_pose_person_relpaths: Tuple[str, ...] | None = None,
    full_pose_person_paths: Tuple[str, ...] | None = None,
    full_pose_person_sha256: Tuple[str, ...] | None = None,
    effective_pose_person_relpaths: Tuple[str, ...] | None = None,
    effective_pose_person_paths: Tuple[str, ...] | None = None,
    effective_pose_person_sha256: Tuple[str, ...] | None = None,
) -> p.SceneRecord:
    resolved_pid = index if pid is None else pid
    person_relpaths = tuple(
        f"pose_data/{split}/person-{index}-{slot}.npz"
        for slot in range(person_count)
    )
    person_paths = tuple(
        f"/synthetic/pose_data/{split}/person-{index}-{slot}.npz"
        for slot in range(person_count)
    )
    person_sha256 = tuple(
        f"person-content-{split}-{index}-{slot}"
        for slot in range(person_count)
    )
    return p.SceneRecord(
        metadata_schema=metadata_schema,
        index=index,
        split=split,
        path=path,
        rgb_sha256=f"rgb-{index}",
        pose_path_sha256=f"pose-path-{index}",
        pose_content_sha256=f"pose-content-{index}",
        pid=resolved_pid,
        camid=camid,
        viewid=viewid,
        person_count=person_count,
        continuous=tuple([0.0] * 95) if continuous is None else continuous,
        frame=frame,
        report={},
        source_pid=resolved_pid,
        source_camid=camid,
        source_frame_id=frame,
        target_person_idx=0,
        full_pose_person_relpaths=(
            person_relpaths
            if full_pose_person_relpaths is None else full_pose_person_relpaths
        ),
        full_pose_person_paths=(
            person_paths if full_pose_person_paths is None else full_pose_person_paths
        ),
        full_pose_person_sha256=(
            person_sha256 if full_pose_person_sha256 is None else full_pose_person_sha256
        ),
        effective_pose_person_relpaths=(
            person_relpaths
            if effective_pose_person_relpaths is None
            else effective_pose_person_relpaths
        ),
        effective_pose_person_paths=(
            person_paths
            if effective_pose_person_paths is None
            else effective_pose_person_paths
        ),
        effective_pose_person_sha256=(
            person_sha256
            if effective_pose_person_sha256 is None
            else effective_pose_person_sha256
        ),
    )


def _summary(payload: object) -> Dict[str, object]:
    canonical = p.canonical_json_bytes(payload)
    return {
        "count": len(payload) if isinstance(payload, (list, tuple)) else 0,
        "canonical_bytes": len(canonical),
        "sha256": p.sha256_bytes(canonical),
    }


def _within(records: Sequence[p.SceneRecord]) -> Dict[str, int]:
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
        "target_outside_effective_count": sum(
            record.target_person_idx >= len(record.effective_pose_person_paths)
            for record in records
        ),
    }


def _relation_report(
    records: Sequence[p.SceneRecord],
    *,
    split: str = "query",
) -> Dict[str, object]:
    empty_summary = _summary([])
    split_records = {
        name: list(records) if name == split else []
        for name in ("train", "query", "gallery")
    }
    report: Dict[str, object] = {
        "schema": p.SPLIT_RELATION_SCHEMA_V2,
        "official_source": {},
        "official_lists": {},
        "split_counts": {
            name: len(values) for name, values in split_records.items()
        },
        "within_split": {
            name: _within(values) for name, values in split_records.items()
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
                name: p.canonical_scene_record_set_summary(values)
                for name, values in split_records.items()
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


def _rehash_relation_report(report: Mapping[str, object]) -> Dict[str, object]:
    payload = dict(report)
    payload.pop("relation_report_sha256", None)
    payload["relation_report_sha256"] = p.sha256_bytes(
        p.canonical_json_bytes(payload))
    return payload


def _relation_token(
    records: Sequence[p.SceneRecord],
    *,
    split: str = "query",
) -> p._ValidatedRelationToken:
    return p._validated_relation_token(
        records,
        relation_report=_relation_report(records, split=split),
        split=split,
    )


def _assert_code(expected: str, function, *args, **kwargs) -> None:
    with pytest.raises(p.GateProtocolError) as captured:
        function(*args, **kwargs)
    assert captured.value.code == expected


def test_historical_signed_person_merge_below_and_at_capacity() -> None:
    from model.modules.pose_utils import merge_person_heatmaps

    heatmaps = torch.full((2, 6, 17, 2, 2), -0.6, dtype=torch.float32)
    for person in range(6):
        heatmaps[:, person] -= 0.01 * person
    heatmaps[:, 4, :, 0, 0] = 0.4
    heatmaps[:, 5, :, 0, 0] = 0.8
    person_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ], dtype=torch.float32)
    before = heatmaps.clone()

    merged = merge_person_heatmaps(heatmaps, person_mask)

    # Historical zero masking dominates an all-negative location below the
    # six-person capacity, whereas a full six-person scene retains raw signs.
    assert torch.equal(merged[0, :, 1, 1], torch.zeros(17))
    assert torch.equal(merged[1, :, 1, 1], torch.full((17,), -0.6))
    assert torch.equal(merged[0, :, 0, 0], torch.full((17,), 0.4))
    assert torch.equal(merged[1, :, 0, 0], torch.full((17,), 0.8))
    assert torch.equal(heatmaps, before)


def test_signed_nuisance_uses_positive_view_without_mutating_raw() -> None:
    raw = torch.full((17, 4, 4), -0.25, dtype=torch.float32)
    raw[:, 1, 1] = 0.5
    raw[6, 2, 2] = -7.5e-5
    scores = torch.linspace(0.1, 0.9, 17, dtype=torch.float32)
    before = raw.clone()
    before_sha = p.sha256_tensor(raw)

    signed = p.summarize_scene(raw, scores)
    explicit_positive = p.summarize_scene(raw.clamp_min(0.0), scores)

    assert signed == explicit_positive
    assert torch.equal(raw, before)
    assert p.sha256_tensor(raw) == before_sha

    all_negative = torch.full((17, 4, 4), -1e-4, dtype=torch.float32)
    _assert_code("E_MATCH_EMPTY_SUPPORT", p.summarize_scene, all_negative, scores)


def test_robust_scale_constant_mad_and_winsor() -> None:
    matrix = np.full((5, 95), 7.0, dtype=np.float64)
    matrix[:, 0] = [0.0, 1.0, 2.0, 3.0, 1000.0]

    standardized, audit = p.robust_scale(matrix)

    assert standardized.dtype == np.float64
    np.testing.assert_allclose(
        standardized[:, 0],
        np.asarray([-2.0 / 1.4826, -1.0 / 1.4826, 0.0, 1.0 / 1.4826, 5.0]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.array_equal(standardized[:, 1:], np.zeros((5, 94)))
    assert audit["median"][0] == 2.0
    assert audit["mad"][0] == 1.0
    assert audit["constant_dims"] == list(range(1, 95))
    assert np.isfinite(standardized).all()

    _assert_code("E_NUISANCE_MATRIX", p.robust_scale, matrix[:, :94])
    nonfinite = matrix.copy()
    nonfinite[0, 0] = np.nan
    _assert_code("E_NUISANCE_NONFINITE", p.robust_scale, nonfinite)


def test_exact_sparse_candidates_and_hall_gate() -> None:
    records = [
        _record(0, "d"),
        _record(1, "b"),
        _record(2, "a"),
        _record(3, "c"),
    ]
    token = _relation_token(records)
    candidates, costs = p.exact_sparse_candidates(
        np.zeros((4, 95), dtype=np.float64),
        device=torch.device("cpu"),
        anchor_chunk=2,
        token=token,
        global_indices=tuple(range(4)),
        local_records=records,
    )

    for left, donors in enumerate(candidates):
        expected = sorted(
            (right for right in range(4) if right != left),
            key=lambda right: records[right].path,
        )
        assert donors == expected
        for right in donors:
            assert abs(costs[(left, right)]) <= 1e-15

    selected, selected_k = p._selected_k_adjacency(candidates)
    assert selected == candidates
    assert selected_k == 3

    ineligible = [_record(0, "x", pid=7), _record(1, "y", pid=7)]
    ineligible_token = _relation_token(ineligible)
    _assert_code(
        "E_MATCH_NO_ELIGIBLE",
        p.exact_sparse_candidates,
        np.zeros((2, 95), dtype=np.float64),
        torch.device("cpu"),
        token=ineligible_token,
        global_indices=(0, 1),
        local_records=ineligible,
    )
    _assert_code("E_MATCH_HALL", p._selected_k_adjacency, [[0], [0]])


def test_exact_sparse_nonzero_continuous_camera_and_frame_cost() -> None:
    records = [
        _record(0, "a", camid=0, frame=0),
        _record(1, "b", camid=0, frame=0),
        _record(2, "c", camid=1, frame=0),
        _record(3, "d", camid=1, frame=1),
    ]
    standardized = np.stack([
        np.zeros(95, dtype=np.float64),
        np.ones(95, dtype=np.float64),
        np.full(95, 2.0, dtype=np.float64),
        np.full(95, 5.0, dtype=np.float64),
    ])

    token = _relation_token(records)
    candidates, costs = p.exact_sparse_candidates(
        standardized,
        device=torch.device("cpu"),
        anchor_chunk=2,
        token=token,
        global_indices=tuple(range(4)),
        local_records=records,
    )

    assert set(candidates[0]) == {1, 2, 3}
    assert costs[(0, 1)] == pytest.approx(1.0, abs=1e-12)
    assert costs[(0, 2)] == pytest.approx(2.25, abs=1e-12)
    assert costs[(0, 3)] == pytest.approx(5.50, abs=1e-12)
    np.testing.assert_allclose(
        p.base_cost(
            standardized[0],
            standardized[1:],
            camera_penalty=np.asarray([0.0, 1.0, 1.0]),
            frame_penalty=np.asarray([0.0, 0.0, 1.0]),
        ),
        [1.0, 2.25, 5.50],
        rtol=0.0,
        atol=1e-12,
    )


def test_strict_v2_records_and_relation_report_fail_closed() -> None:
    records = [_record(index, f"q{index}") for index in range(4)]
    report = _relation_report(records)
    token = p._validated_relation_token(
        records,
        relation_report=report,
        split="query",
    )
    assert token.split == "query"
    assert token.full_record_set_sha256 == (
        report["relations"]["split_record_sets"]["query"]["sha256"]
    )

    v1_record = replace(records[0], metadata_schema="exp374-scene-metadata-v1")
    _assert_code(
        "E_MATCH_RELATION_TOKEN",
        p._validated_relation_token,
        [v1_record, *records[1:]],
        relation_report=report,
        split="query",
    )
    for field_name in (
        "full_pose_person_relpaths",
        "full_pose_person_paths",
        "full_pose_person_sha256",
        "effective_pose_person_relpaths",
        "effective_pose_person_paths",
        "effective_pose_person_sha256",
    ):
        empty_tuple_record = replace(records[0], **{field_name: ()})
        _assert_code(
            "E_MATCH_RELATION_TOKEN",
            p._validated_relation_token,
            [empty_tuple_record, *records[1:]],
            relation_report=report,
            split="query",
        )

    self_hash_drift = dict(report)
    self_hash_drift["relation_report_sha256"] = "drift"
    _assert_code(
        "E_MATCH_RELATION_TOKEN",
        p._validated_relation_token,
        records,
        relation_report=self_hash_drift,
        split="query",
    )
    _assert_code(
        "E_MATCH_RELATION_TOKEN",
        p._validated_relation_token,
        records,
        relation_report=report,
        split="gallery",
    )

    summary_drift = dict(report)
    relations = dict(summary_drift["relations"])
    record_sets = dict(relations["split_record_sets"])
    query_summary = dict(record_sets["query"])
    query_summary["sha256"] = "drift"
    record_sets["query"] = query_summary
    relations["split_record_sets"] = record_sets
    summary_drift["relations"] = relations
    summary_drift = _rehash_relation_report(summary_drift)
    _assert_code(
        "E_MATCH_RELATION_TOKEN",
        p._validated_relation_token,
        records,
        relation_report=summary_drift,
        split="query",
    )


@pytest.mark.parametrize(
    "field_name,replacement_value",
    [
        ("camid", 9),
        ("frame", 15),
        ("continuous", tuple([1.0] + [0.0] * 94)),
        ("full_pose_person_relpaths", ("pose_data/query/drift.npz",)),
        ("full_pose_person_paths", ("/synthetic/drift.npz",)),
        ("full_pose_person_sha256", ("drift-full-content",)),
        ("effective_pose_person_relpaths", ("pose_data/query/drift-effective.npz",)),
        ("effective_pose_person_paths", ("/synthetic/drift-effective.npz",)),
        ("effective_pose_person_sha256", ("drift-effective-content",)),
    ],
)
def test_full_record_digest_binds_cost_and_constituent_fields(
    field_name: str,
    replacement_value: object,
) -> None:
    records = [_record(index, f"q{index}") for index in range(4)]
    report = _relation_report(records)
    drifted = list(records)
    drifted[0] = replace(records[0], **{field_name: replacement_value})
    _assert_code(
        "E_MATCH_RELATION_TOKEN",
        p._validated_relation_token,
        drifted,
        relation_report=report,
        split="query",
    )


def test_exact_sparse_token_identity_and_complete_stratum_gate() -> None:
    records = [_record(index, f"q{index}") for index in range(4)]
    token = _relation_token(records)
    standardized = np.zeros((4, 95), dtype=np.float64)
    with pytest.raises(TypeError):
        p.exact_sparse_candidates(
            standardized,
            torch.device("cpu"),
            global_indices=(0, 1, 2, 3),
            local_records=records,
        )
    fake_token = replace(token, _factory_identity=object())
    _assert_code(
        "E_MATCH_RELATION_TOKEN",
        p.exact_sparse_candidates,
        standardized,
        torch.device("cpu"),
        token=fake_token,
        global_indices=(0, 1, 2, 3),
        local_records=records,
    )

    invalid_calls = [
        ("omission", (0, 1, 2), records[:3]),
        ("superset", (0, 1, 2, 3, 4), [*records, records[0]]),
        ("reorder", (1, 0, 2, 3), [records[1], records[0], records[2], records[3]]),
        ("duplicate", (0, 1, 1, 3), [records[0], records[1], records[1], records[3]]),
        ("out_of_bounds", (0, 1, 2, 4), records),
        ("wrong_subset", (0, 2), [records[0], records[2]]),
    ]
    for _case, global_indices, local_records in invalid_calls:
        _assert_code(
            "E_MATCH_RELATION_TOKEN",
            p.exact_sparse_candidates,
            np.zeros((len(local_records), 95), dtype=np.float64),
            torch.device("cpu"),
            token=token,
            global_indices=global_indices,
            local_records=local_records,
        )

    for global_indices, local_records in (
        ((0, 1), records[:2]),
        ((2, 3), records[2:]),
    ):
        _assert_code(
            "E_MATCH_RELATION_TOKEN",
            p.exact_sparse_candidates,
            np.zeros((2, 95), dtype=np.float64),
            torch.device("cpu"),
            token=token,
            global_indices=global_indices,
            local_records=local_records,
        )

    cross_records = [
        _record(0, "q0", person_count=1),
        _record(1, "q1", person_count=2),
        _record(2, "q2", person_count=1),
        _record(3, "q3", person_count=2),
    ]
    cross_token = _relation_token(cross_records)
    _assert_code(
        "E_MATCH_RELATION_TOKEN",
        p.exact_sparse_candidates,
        np.zeros((2, 95), dtype=np.float64),
        torch.device("cpu"),
        token=cross_token,
        global_indices=(0, 1),
        local_records=cross_records[:2],
    )


def test_cpu_eligible_pair_matches_token_gated_gpu_candidate_edges() -> None:
    records = [_record(index, f"q{index}") for index in range(4)]
    token = _relation_token(records)
    candidates, _costs = p.exact_sparse_candidates(
        np.zeros((4, 95), dtype=np.float64),
        device=torch.device("cpu"),
        anchor_chunk=2,
        token=token,
        global_indices=(0, 1, 2, 3),
        local_records=records,
    )
    for left, anchor in enumerate(records):
        for right, donor in enumerate(records):
            assert (right in candidates[left]) is p.eligible_pair(anchor, donor)

    constituent_fields = (
        "full_pose_person_paths",
        "full_pose_person_sha256",
        "effective_pose_person_paths",
        "effective_pose_person_sha256",
    )
    for field_name in constituent_fields:
        drifted = list(records)
        drifted[1] = replace(
            records[1],
            **{field_name: getattr(records[0], field_name)},
        )
        assert not p.eligible_pair(drifted[0], drifted[1])
        _assert_code(
            "E_MATCH_RELATION_TOKEN",
            p._validated_relation_token,
            drifted,
            relation_report=_relation_report(drifted),
            split="query",
        )
        _assert_code(
            "E_MATCH_RELATION_TOKEN",
            p.exact_sparse_candidates,
            np.zeros((4, 95), dtype=np.float64),
            torch.device("cpu"),
            token=token,
            global_indices=(0, 1, 2, 3),
            local_records=drifted,
        )


def test_randomized_full_matching_is_reproducible_and_bijective() -> None:
    adjacency = [
        [right for right in range(6) if right != left]
        for left in range(6)
    ]

    first = p.randomized_full_matching(adjacency, seed=374123)
    second = p.randomized_full_matching(adjacency, seed=374123)

    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(6))
    assert all(int(right) in adjacency[left] for left, right in enumerate(first))
    _assert_code(
        "E_BASELINE_PARTIAL",
        p.randomized_full_matching,
        [[0], [0]],
        374123,
    )


def test_minimum_cost_matching_uses_supplied_global_tie_rank() -> None:
    adjacency = [[0, 1], [0, 1]]
    costs = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}
    ranks = {(0, 0): 3, (0, 1): 0, (1, 0): 1, (1, 1): 2}

    matching = p.minimum_cost_full_matching(
        adjacency,
        costs,
        tie_break_ranks=ranks,
        tie_break_denominator=4,
    )
    assert matching.tolist() == [1, 0]

    missing_cost = dict(costs)
    missing_cost.pop((1, 1))
    _assert_code(
        "E_MATCH_EDGE",
        p.minimum_cost_full_matching,
        adjacency,
        missing_cost,
    )
    duplicate_ranks = dict(ranks)
    duplicate_ranks[(1, 1)] = duplicate_ranks[(1, 0)]
    _assert_code(
        "E_MATCH_TIE_RANK",
        p.minimum_cost_full_matching,
        adjacency,
        costs,
        duplicate_ranks,
        4,
    )


def _install_tiny_mapping_stubs(
    monkeypatch: pytest.MonkeyPatch,
    hamming: float,
) -> list[Tuple[Dict[Tuple[int, int], int], int]]:
    def fake_candidates(
        _standardized: np.ndarray,
        device: torch.device,
        anchor_chunk: int = 16,
        *,
        token: p._ValidatedRelationToken,
        global_indices: Sequence[int],
        local_records: Sequence[p.SceneRecord],
    ):
        del device, anchor_chunk
        assert token.split == "query"
        assert tuple(record.index for record in local_records) == tuple(global_indices)
        names = tuple(record.path for record in local_records)
        if names == ("z0", "a1"):
            return [[1], [0]], {(0, 1): 10.0, (1, 0): 20.0}
        if names == ("a0", "z1"):
            return [[1], [0]], {(0, 1): 30.0, (1, 0): 40.0}
        raise AssertionError(names)

    tie_calls: list[Tuple[Dict[Tuple[int, int], int], int]] = []

    def fake_minimum_cost(
        adjacency: Sequence[Sequence[int]],
        edge_costs: Mapping[Tuple[int, int], float],
        tie_break_ranks: Mapping[Tuple[int, int], int] | None = None,
        tie_break_denominator: int | None = None,
    ) -> np.ndarray:
        assert adjacency == [[1], [0]]
        assert set(edge_costs) == {(0, 1), (1, 0)}
        assert tie_break_ranks is not None
        assert tie_break_denominator is not None
        tie_calls.append((dict(tie_break_ranks), int(tie_break_denominator)))
        return np.asarray([1, 0], dtype=np.int64)

    monkeypatch.setattr(p, "exact_sparse_candidates", fake_candidates)
    monkeypatch.setattr(p, "minimum_cost_full_matching", fake_minimum_cost)
    monkeypatch.setattr(
        p,
        "randomized_full_matching",
        lambda _adjacency, _seed: np.asarray([2, 3, 0, 1], dtype=np.int64),
    )
    monkeypatch.setattr(p, "audit_mapping", lambda *_args, **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(p, "pairwise_hamming", lambda _mappings: hamming)
    return tie_calls


def _tiny_mapping_records() -> list[p.SceneRecord]:
    return [
        _record(0, "z0", person_count=1),
        _record(1, "a0", person_count=2),
        _record(2, "a1", person_count=1),
        _record(3, "z1", person_count=2),
    ]


def test_global_tie_rank_gumbel_order_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    tie_calls = _install_tiny_mapping_stubs(monkeypatch, hamming=0.90)
    records = _tiny_mapping_records()
    payload = p.prepare_split_mappings(
        records,
        device=torch.device("cpu"),
        anchor_chunk=2,
        relation_report=_relation_report(records),
        split="query",
    )

    assert len(tie_calls) == 2 * len(p.MAPPING_SEEDS)
    for call in range(0, len(tie_calls), 2):
        first_ranks, first_denominator = tie_calls[call]
        second_ranks, second_denominator = tie_calls[call + 1]
        assert first_denominator == second_denominator == 4
        assert set(first_ranks.values()) | set(second_ranks.values()) == {0, 1, 2, 3}

    for mapping_index, seed in enumerate(p.MAPPING_SEEDS):
        noise = np.random.Generator(np.random.PCG64DXSM(seed)).gumbel(size=4)
        expected = np.asarray([
            10.0 + 1.25 * noise[2],
            30.0 + 1.25 * noise[0],
            20.0 + 1.25 * noise[1],
            40.0 + 1.25 * noise[3],
        ], dtype=np.float64)
        np.testing.assert_array_equal(payload["randomized_edge_costs"][mapping_index], expected)

    # Persistence is implemented by the runner, so this is the sole cross-file
    # seam in an otherwise protocol-only test module.
    from experiments.exp374_psg_fuel_gate.audit_gate_a import save_mapping_payload

    save_mapping_payload(tmp_path, "query", payload)
    with np.load(tmp_path / "query_candidate_graph.npz", allow_pickle=False) as graph:
        assert list(zip(graph["edge_left"].tolist(), graph["edge_right"].tolist())) == [
            (0, 2), (1, 3), (2, 0), (3, 1),
        ]
        np.testing.assert_array_equal(
            graph["randomized_edge_cost"],
            np.stack(payload["randomized_edge_costs"]),
        )


def test_prepare_split_single_mapping_records_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_mapping_stubs(monkeypatch, hamming=0.89)
    records = _tiny_mapping_records()
    payload = p.prepare_split_mappings(
        records,
        torch.device("cpu"),
        2,
        relation_report=_relation_report(records),
        split="query",
    )
    assert len(payload["mappings"]) == 1
    assert payload["minimum_hamming"] == pytest.approx(0.89)
    assert payload["effective_unique_count"] == 1


def test_pair_quality_audit_and_pairwise_hamming() -> None:
    records = [_record(index, f"r{index}") for index in range(4)]
    mapping = np.asarray([1, 0, 3, 2], dtype=np.int64)

    def edges(cost: float) -> Dict[Tuple[int, int], float]:
        return {(left, int(right)): cost for left, right in enumerate(mapping)}

    standardized = np.zeros((4, 95), dtype=np.float64)
    audit = p.audit_mapping(records, standardized, mapping, edges(0.1), [1.0])
    assert audit["mean_cost"] == pytest.approx(0.1)
    assert audit["p95_cost"] == pytest.approx(0.1)
    assert audit["max_dimension_median_abs_z"] == 0.0
    assert audit["max_marginal_mean_error"] == 0.0
    assert audit["max_empirical_ks"] == 0.0

    dimension_failure = standardized.copy()
    dimension_failure[:, 0] = [0.0, 1.0, 0.0, 1.0]
    _assert_code(
        "E_PAIR_DIM", p.audit_mapping,
        records, dimension_failure, mapping, edges(0.1), [1.0])
    _assert_code(
        "E_PAIR_P95", p.audit_mapping,
        records, standardized, mapping, edges(2.0), [10.0])
    _assert_code(
        "E_PAIR_BASELINE", p.audit_mapping,
        records, standardized, mapping, edges(0.6), [0.5])

    hard_failure = list(records)
    hard_failure[1] = _record(1, "r1", pid=records[0].pid)
    _assert_code(
        "E_MATCH_HARD", p.audit_mapping,
        hard_failure, standardized, mapping, edges(0.1), [1.0])

    first = np.arange(10, dtype=np.int64)
    second = np.asarray([0, 2, 3, 4, 5, 6, 7, 8, 9, 1], dtype=np.int64)
    assert p.pairwise_hamming([first, second]) == pytest.approx(0.90)


def test_intervention_strength_and_fail_closed_audits() -> None:
    correct_scene = torch.zeros((2, 17, 4, 4), dtype=torch.float32)
    donor_scene = torch.ones((2, 17, 4, 4), dtype=torch.float32)
    actual_correct = p.actual_psg_input(correct_scene, (4, 4))
    actual_donor = p.actual_psg_input(donor_scene, (4, 4))

    values = p.intervention_strength(actual_correct, actual_donor)
    np.testing.assert_allclose(values["relative_l1"], [2.0, 2.0], atol=1e-6)
    np.testing.assert_array_equal(values["centroid_displacement"], [1.0, 1.0])
    audit = p.audit_intervention_strength(
        correct_scene, donor_scene, actual_correct, actual_donor, (4, 4))
    assert audit["median_relative_l1"] == pytest.approx(2.0, abs=1e-6)
    assert audit["p10_relative_l1"] == pytest.approx(2.0, abs=1e-6)
    assert audit["median_centroid_displacement"] == 1.0

    drifted = actual_donor.clone()
    drifted[0, 0, 0, 0] += 1e-4
    _assert_code(
        "E_HOOK_DONOR_DRIFT", p.audit_intervention_strength,
        correct_scene, donor_scene, actual_correct, drifted, (4, 4))
    _assert_code(
        "E_WEAK_IDENTICAL", p.audit_intervention_strength,
        correct_scene, correct_scene, actual_correct, actual_correct, (4, 4))

    close_scene = torch.ones((2, 17, 4, 4), dtype=torch.float32)
    close_donor = torch.full_like(close_scene, 0.9)
    _assert_code(
        "E_WEAK_MEDIAN_L1", p.audit_intervention_strength,
        close_scene,
        close_donor,
        p.actual_psg_input(close_scene, (4, 4)),
        p.actual_psg_input(close_donor, (4, 4)),
        (4, 4),
    )


def test_intervention_centroid_uses_positive_response_mass() -> None:
    below_a = torch.full((1, 17, 3, 3), 0.49, dtype=torch.float32)
    below_b = torch.full((1, 17, 3, 3), 0.40, dtype=torch.float32)
    no_mass = p.intervention_strength(below_a, below_b)
    np.testing.assert_array_equal(no_mass["centroid_displacement"], [0.0])

    one_sided = below_b.clone()
    one_sided[:, :, 1, 1] = 0.51
    one_mass = p.intervention_strength(below_a, one_sided)
    np.testing.assert_array_equal(one_mass["centroid_displacement"], [1.0])


def test_centroid_control_translation_and_crop_failure() -> None:
    scene = torch.zeros((17, 5, 5), dtype=torch.float32)
    scene[:, 1, 1] = 1.0
    targets = tuple([(2.0, 3.0)] * 17)

    output = p.apply_scene_centroid_control(scene, targets)
    expected = torch.zeros_like(scene)
    expected[:, 3, 2] = 1.0
    assert torch.equal(output, expected)
    assert torch.equal(output.sum(dim=(1, 2)), scene.sum(dim=(1, 2)))
    assert torch.equal(output.amax(dim=(1, 2)), scene.amax(dim=(1, 2)))
    assert torch.isfinite(output).all()

    missing_target = list(targets)
    missing_target[0] = None
    _assert_code(
        "E_CENTROID_TARGET", p.apply_scene_centroid_control,
        scene, tuple(missing_target))

    cropped = torch.zeros((17, 5, 5), dtype=torch.float32)
    cropped[:, 2, 3] = 1.0
    cropped[:, 2, 4] = 1.0
    _assert_code(
        "E_CENTROID_L1", p.apply_scene_centroid_control,
        cropped, tuple([(4.0, 2.0)] * 17))


def test_signed_centroid_translates_raw_with_positive_geometry() -> None:
    scene = torch.zeros((17, 5, 5), dtype=torch.float32)
    scene[:, 2, 2] = 1.0
    scene[:, 1, 1] = -0.25
    before = scene.clone()
    targets = tuple([(3.0, 2.0)] * 17)

    output = p.apply_scene_centroid_control(scene, targets)
    expected = torch.zeros_like(scene)
    expected[:, 2, 3] = 1.0
    expected[:, 1, 2] = -0.25

    assert torch.equal(output, expected)
    assert torch.equal(scene, before)
    assert torch.equal(output.clamp_min(0.0), p.translate_zero_padded(
        scene.clamp_min(0.0)[0], 1, 0).unsqueeze(0).expand_as(output))
    assert torch.equal(
        output.clamp_max(0.0).abs().sum(dim=(1, 2)),
        scene.clamp_max(0.0).abs().sum(dim=(1, 2)),
    )

    negative_crop = torch.zeros((17, 5, 5), dtype=torch.float32)
    negative_crop[:, 2, 2] = 1.0
    negative_crop[:, 1, 0] = -0.25
    _assert_code(
        "E_CENTROID_NEGATIVE_L1",
        p.apply_scene_centroid_control,
        negative_crop,
        tuple([(1.0, 2.0)] * 17),
    )


def test_all_negative_centroid_scene_is_preserved_bitwise() -> None:
    scene = torch.full((17, 5, 5), -0.125, dtype=torch.float32)
    targets = tuple([None] * 17)

    normalized = p.fit_normalized_centroid_targets([scene])
    assert normalized == tuple([None] * 17)
    absolute = p.absolute_centroid_targets(scene, normalized)
    assert absolute == tuple([None] * 17)
    output = p.apply_scene_centroid_control(scene, targets)
    assert torch.equal(output, scene)
    assert p.sha256_tensor(output) == p.sha256_tensor(scene)


def test_centroid_fit_to_absolute_target_round_trip() -> None:
    train_scene = torch.zeros((17, 5, 5), dtype=torch.float32)
    train_scene[0, 1, 1] = 1.0
    train_scene[1, 3, 3] = 1.0
    train_scene[2:, 2, 2] = 1.0

    normalized = p.fit_normalized_centroid_targets([train_scene])
    assert normalized[0] == pytest.approx((0.0, 0.0))
    assert normalized[1] == pytest.approx((1.0, 1.0))
    for target in normalized[2:]:
        assert target == pytest.approx((0.5, 0.5))

    target_scene = torch.zeros((17, 7, 7), dtype=torch.float32)
    target_scene[0, 1, 2] = 1.0
    target_scene[1, 5, 5] = 1.0
    target_scene[2:, 3, 4] = 1.0
    absolute = p.absolute_centroid_targets(target_scene, normalized)

    assert absolute[0] == pytest.approx((2.0, 1.0))
    assert absolute[1] == pytest.approx((5.0, 5.0))
    for target in absolute[2:]:
        assert target == pytest.approx((3.5, 3.0))

    controlled = p.apply_scene_centroid_control(target_scene, absolute)
    assert controlled[0, 1, 2] == 1.0
    assert controlled[1, 5, 5] == 1.0
    assert torch.equal(controlled[2:, 3, 3], torch.ones(15))


def test_per_query_metrics_junk_ap_rank_and_margin() -> None:
    distances = np.asarray([
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.1],
    ], dtype=np.float64)
    metrics = p.per_query_metrics(
        distances,
        q_pids=[1, 2],
        g_pids=[1, 1, 2, 3],
        q_camids=[0, 0],
        g_camids=[0, 1, 1, 1],
    )
    np.testing.assert_allclose(metrics["AP"], [1.0, 0.25])
    np.testing.assert_array_equal(metrics["R1_indicator"], [1.0, 0.0])
    np.testing.assert_allclose(metrics["margin"], [0.1, -0.3], atol=1e-15)
    assert metrics["mAP"] == pytest.approx(0.625)
    assert metrics["R1"] == pytest.approx(0.5)
    assert metrics["R5"] == pytest.approx(1.0)
    assert metrics["R10"] == pytest.approx(1.0)

    nonfinite = distances.copy()
    nonfinite[0, 0] = np.nan
    _assert_code(
        "E_DIST_NONFINITE", p.per_query_metrics,
        nonfinite, [1, 2], [1, 1, 2, 3], [0, 0], [0, 1, 1, 1])
    _assert_code(
        "E_QUERY_NO_POSITIVE", p.per_query_metrics,
        distances, [9, 2], [1, 1, 2, 3], [0, 0], [0, 1, 1, 1])
    _assert_code(
        "E_QUERY_NO_NEGATIVE", p.per_query_metrics,
        np.asarray([[0.1, 0.2]]), [1], [1, 1], [0], [0, 1])


def _family(
    shuffle: Tuple[float, float, float, Mapping[int, float]],
    bypass: Tuple[float, float, float, Mapping[int, float]],
) -> Dict[str, object]:
    def interval(values: Tuple[float, float, float, Mapping[int, float]]) -> Dict[str, object]:
        estimate, lower, upper, per_seed = values
        return {
            "estimate": estimate,
            "LCB": lower,
            "UCB": upper,
            "per_seed": dict(per_seed),
        }

    return {"intervals": {"shuffle": interval(shuffle), "bypass": interval(bypass)}}


def test_mapping_aggregation_bootstrap_and_gate_decisions() -> None:
    mapping_values = np.asarray([[1.0, 2.0, 3.0]])
    aggregated = p.aggregate_mapping_queries(mapping_values)
    np.testing.assert_array_equal(aggregated["mean"], [1.0, 2.0, 3.0])

    seeds = (42, 1234, 2024)
    pids = np.asarray([1, 1, 2, 3], dtype=np.int64)
    correct = {seed: np.ones(4, dtype=np.float64) for seed in seeds}
    map_controls = {
        "shuffle": {seed: np.full(4, 0.99, dtype=np.float64) for seed in seeds},
        "bypass": {seed: np.full(4, 0.995, dtype=np.float64) for seed in seeds},
    }
    map_family = p.simultaneous_intervals(correct, map_controls, pids, replicates=64)
    assert map_family["intervals"]["shuffle"]["estimate"] == pytest.approx(1.0)
    assert map_family["intervals"]["bypass"]["estimate"] == pytest.approx(0.5)
    assert abs(map_family["q_lower"]) <= 1e-12
    assert abs(map_family["q_upper"]) <= 1e-12
    for control in p.PRIMARY_CONTROLS:
        interval = map_family["intervals"][control]
        assert interval["LCB"] == pytest.approx(interval["estimate"], abs=1e-12)
        assert interval["UCB"] == pytest.approx(interval["estimate"], abs=1e-12)

    r1_controls = {
        control: {seed: correct[seed].copy() for seed in seeds}
        for control in p.PRIMARY_CONTROLS
    }
    r1_family = p.simultaneous_intervals(correct, r1_controls, pids, replicates=64)
    assert p.gate_decision(map_family, r1_family, audits_passed=True)["decision"] == "GO"
    assert p.gate_decision(map_family, r1_family, audits_passed=False)["decision"] == "INVALID"

    positive = {42: 0.5, 1234: 0.5, 2024: 0.5}
    r1_manual = _family(
        (0.0, 0.0, 0.0, {42: 0.0, 1234: 0.0, 2024: 0.0}),
        (0.0, 0.0, 0.0, {42: 0.0, 1234: 0.0, 2024: 0.0}),
    )
    futility_family = _family(
        (0.2, 0.1, 0.2, {42: 0.2, 1234: 0.2, 2024: 0.2}),
        (0.5, 0.1, 0.8, positive),
    )
    futility = p.gate_decision(futility_family, r1_manual, audits_passed=True)
    assert futility["decision"] == "NO_GO"
    assert futility["futility"] is True

    two_seed_family = _family(
        (0.4, -0.1, 0.8, {42: -0.1, 1234: 0.0, 2024: 1.3}),
        (0.5, 0.1, 0.8, positive),
    )
    two_seed = p.gate_decision(two_seed_family, r1_manual, audits_passed=True)
    assert two_seed["decision"] == "NO_GO"
    assert two_seed["two_seed_nonpositive"] is True

    inconclusive_family = _family(
        (0.4, -0.1, 0.8, positive),
        (0.5, 0.1, 0.8, positive),
    )
    inconclusive = p.gate_decision(inconclusive_family, r1_manual, audits_passed=True)
    assert inconclusive["decision"] == "INCONCLUSIVE"

    _assert_code(
        "E_BOOTSTRAP_CONTROLS",
        p.simultaneous_intervals,
        correct,
        {"shuffle": map_controls["shuffle"]},
        pids,
        4,
    )
    two_seed_correct = {seed: correct[seed] for seed in seeds[:2]}
    two_seed_controls = {
        control: {seed: values[seed] for seed in seeds[:2]}
        for control, values in map_controls.items()
    }
    _assert_code(
        "E_BOOTSTRAP_SEEDS",
        p.simultaneous_intervals,
        two_seed_correct,
        two_seed_controls,
        pids,
        4,
    )
    bad_shape_controls = {
        control: {seed: values[seed].copy() for seed in seeds}
        for control, values in map_controls.items()
    }
    bad_shape_controls["bypass"][42] = np.zeros(3, dtype=np.float64)
    _assert_code(
        "E_BOOTSTRAP_SHAPE",
        p.simultaneous_intervals,
        correct,
        bad_shape_controls,
        pids,
        4,
    )


def test_gate_decision_strict_boundaries() -> None:
    positive_at_threshold = {42: 0.30, 1234: 0.30, 2024: 0.30}
    zero_r1 = {42: 0.0, 1234: 0.0, 2024: 0.0}
    r1_safe = _family(
        (0.0, 0.0, 0.5, zero_r1),
        (0.0, 0.0, 0.5, zero_r1),
    )

    theta_boundary = _family(
        (0.30, 0.10, 0.50, positive_at_threshold),
        (0.30, 0.10, 0.50, positive_at_threshold),
    )
    assert p.gate_decision(
        theta_boundary, r1_safe, audits_passed=True)["decision"] == "GO"

    positive = {42: 0.40, 1234: 0.40, 2024: 0.40}
    zero_lcb = _family(
        (0.40, 0.0, 0.80, positive),
        (0.40, 0.10, 0.80, positive),
    )
    assert p.gate_decision(
        zero_lcb, r1_safe, audits_passed=True)["decision"] == "INCONCLUSIVE"

    r1_strict_boundary = _family(
        (0.0, -0.50, 0.50, zero_r1),
        (0.0, 0.0, 0.50, zero_r1),
    )
    assert p.gate_decision(
        theta_boundary,
        r1_strict_boundary,
        audits_passed=True,
    )["decision"] == "INCONCLUSIVE"
