"""Pure-synthetic unit tests for the frozen exp374 Gate-A protocol.

The fixtures deliberately avoid real datasets and checkpoints.  They bind the
production helpers directly and assert stable ``GateProtocolError.code`` values
for every fail-closed path covered here.
"""

from __future__ import annotations

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
    person_count: int = 1,
    frame: int = 0,
) -> p.SceneRecord:
    return p.SceneRecord(
        index=index,
        split="query",
        path=path,
        rgb_sha256=f"rgb-{index}",
        pose_path_sha256=f"pose-path-{index}",
        pose_content_sha256=f"pose-content-{index}",
        pid=index if pid is None else pid,
        camid=camid,
        person_count=person_count,
        continuous=tuple([0.0] * 95),
        frame=frame,
        report={},
    )


def _assert_code(expected: str, function, *args, **kwargs) -> None:
    with pytest.raises(p.GateProtocolError) as captured:
        function(*args, **kwargs)
    assert captured.value.code == expected


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
    candidates, costs = p.exact_sparse_candidates(
        records,
        np.zeros((4, 95), dtype=np.float64),
        device=torch.device("cpu"),
        anchor_chunk=2,
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
    _assert_code(
        "E_MATCH_NO_ELIGIBLE",
        p.exact_sparse_candidates,
        ineligible,
        np.zeros((2, 95), dtype=np.float64),
        torch.device("cpu"),
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

    candidates, costs = p.exact_sparse_candidates(
        records,
        standardized,
        device=torch.device("cpu"),
        anchor_chunk=2,
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
        records: Sequence[p.SceneRecord],
        _standardized: np.ndarray,
        device: torch.device,
        anchor_chunk: int = 16,
    ):
        del device, anchor_chunk
        names = tuple(record.path for record in records)
        if names == ("a1", "z0"):
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
    payload = p.prepare_split_mappings(
        _tiny_mapping_records(), device=torch.device("cpu"), anchor_chunk=2)

    assert len(tie_calls) == 40
    for call in range(0, len(tie_calls), 2):
        first_ranks, first_denominator = tie_calls[call]
        second_ranks, second_denominator = tie_calls[call + 1]
        assert first_denominator == second_denominator == 4
        assert set(first_ranks.values()) | set(second_ranks.values()) == {0, 1, 2, 3}

    for mapping_index, seed in enumerate(p.MAPPING_SEEDS):
        noise = np.random.Generator(np.random.PCG64DXSM(seed)).gumbel(size=4)
        expected = np.asarray([
            20.0 + 1.25 * noise[2],
            30.0 + 1.25 * noise[0],
            10.0 + 1.25 * noise[1],
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


def test_prepare_split_rejects_insufficient_mapping_hamming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_mapping_stubs(monkeypatch, hamming=0.89)
    _assert_code(
        "E_MAPPING_HAMMING",
        p.prepare_split_mappings,
        _tiny_mapping_records(),
        torch.device("cpu"),
        2,
    )


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
    mapping_values = np.tile(np.asarray([[1.0, 2.0, 3.0]]), (20, 1))
    aggregated = p.aggregate_mapping_queries(mapping_values)
    np.testing.assert_array_equal(aggregated["mean"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(aggregated["mcse"], [0.0, 0.0, 0.0])

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
