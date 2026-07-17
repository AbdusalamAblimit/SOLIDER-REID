from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp371_casd.frozen_support_oracle import (
    balanced_weights,
    build_episode,
    build_val_protocol,
    compute_arm_distances,
    cyclic_derangement,
    load_cache,
    run_episode,
    support_descriptor,
    weight_mass,
)


def _blocks(seed: int, block_dim: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(7, block_dim, generator=generator), p=2, dim=1)


def _cache() -> dict:
    # Two query identities, each with two distinct-path gallery donors.
    features = torch.stack([_blocks(seed, 4).reshape(-1) for seed in range(6)])
    return {
        "features": features,
        "pids": [0, 1, 0, 0, 1, 1],
        "camids": [0, 0, 1, 2, 1, 2],
        "paths": ["q0", "q1", "p0a", "p0b", "p1a", "p1b"],
        "split": "val",
        "mode": "correct",
        "num_query": 2,
        "block_dim": 4,
        "weight_sha256": "synthetic",
    }


def test_load_cache_requires_seven_finite_blocks(tmp_path: Path):
    path = tmp_path / "cache.pt"
    torch.save(_cache(), path)
    loaded = load_cache(path, expected_block_dim=4)
    assert loaded["features"].shape == (6, 28)
    assert loaded["block_norm_stats"]["max_abs_error_from_one"] < 1e-5


def test_part_permutation_is_a_deterministic_derangement():
    first = cyclic_derangement("path/a.jpg", 5, seed=371)
    second = cyclic_derangement("path/a.jpg", 5, seed=371)
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(5))
    assert torch.all(first != torch.arange(5))


def test_support_arms_keep_global_and_separate_identity_from_slots():
    anchor = _blocks(10)
    donor_a = _blocks(11)
    donor_b = _blocks(12)
    donors = [("a", donor_a), ("b", donor_b)]

    identity = support_descriptor(anchor, donors, "ID-MEAN", seed=371).view(7, 5)
    equal = support_descriptor(anchor, donors, "PART-EQUAL", seed=371).view(7, 5)
    permuted = support_descriptor(anchor, donors, "PART-PERM", seed=371).view(7, 5)
    casd = support_descriptor(anchor, donors, "CASD-LIKE", seed=371).view(7, 5)

    # Final descriptor normalization scales every block equally; directions are compared.
    for value in (identity, equal, permuted, casd):
        assert torch.allclose(
            F.normalize(value[0], dim=0), anchor[0], atol=1e-6
        )
        assert abs(value.reshape(-1).norm().item() - 1.0) < 1e-6
    identity_local = F.normalize(identity[1:], p=2, dim=1)
    assert torch.allclose(
        identity_local, identity_local[:1].expand_as(identity_local), atol=1e-6
    )
    assert not torch.allclose(equal[2:], permuted[2:])


def test_strict_path_loo_excludes_anchor_and_positive_endpoint():
    cache = _cache()
    episode = build_episode(cache, build_val_protocol(cache))
    distances, stats = compute_arm_distances(
        episode,
        "PART-EQUAL",
        seed=371,
        device=torch.device("cpu"),
        distance_batch=2,
    )
    assert stats["strict_path_violations"] == 0
    assert stats["endpoint_exclusion_violations"] == 0
    assert stats["endpoint_corrections"] == 4

    # q0 -> p0a must use p0b alone: the relation endpoint cannot be its own support.
    anchor = episode["all_blocks"][episode["query_indices"][0]]
    reduced = [item for item in episode["donors"][0] if item[0] != "p0a"]
    descriptor = support_descriptor(anchor, reduced, "PART-EQUAL", seed=371)
    gallery = F.normalize(episode["all_blocks"][2].reshape(-1), p=2, dim=0)
    expected = (2.0 - 2.0 * torch.dot(descriptor, gallery)).clamp_min(0.0)
    assert torch.allclose(distances[0, 0], expected, atol=1e-6)


def test_balanced_weights_have_shared_unit_loss_mass():
    positive = torch.tensor([[True, False, True], [False, True, False]])
    negative = ~positive
    mask = torch.ones_like(positive)
    raw = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weights = balanced_weights(mask, positive, negative, raw)
    mass = weight_mass(weights, {"positive": positive, "negative": negative})
    assert abs(mass["positive"] - 0.5) < 1e-7
    assert abs(mass["negative"] - 0.5) < 1e-7
    assert abs(mass["total"] - 1.0) < 1e-7


def test_full_oracle_reports_all_arms_and_loss_matched_relational_control():
    generator = torch.Generator().manual_seed(9)
    block_dim = 8
    identity_bases = F.normalize(
        torch.randn(4, 7, block_dim, generator=generator), p=2, dim=2
    )
    features = []
    pids = []
    camids = []
    paths = []
    for pid in range(4):
        value = F.normalize(
            identity_bases[pid]
            + 0.35 * torch.randn(7, block_dim, generator=generator),
            p=2,
            dim=1,
        )
        features.append(value.reshape(-1))
        pids.append(pid)
        camids.append(0)
        paths.append("q%d" % pid)
    for pid in range(4):
        for donor in range(3):
            value = F.normalize(
                identity_bases[pid]
                + 0.35 * torch.randn(7, block_dim, generator=generator),
                p=2,
                dim=1,
            )
            features.append(value.reshape(-1))
            pids.append(pid)
            camids.append(donor + 1)
            paths.append("g%d_%d" % (pid, donor))
    cache = {
        "features": torch.stack(features),
        "pids": pids,
        "camids": camids,
        "paths": paths,
        "split": "val",
        "mode": "correct",
        "num_query": 4,
        "block_dim": block_dim,
        "weight_sha256": "synthetic",
    }
    result = run_episode(
        cache,
        build_val_protocol(cache),
        seed=371,
        device=torch.device("cpu"),
        distance_batch=4,
        min_map_gap=0.005,
        min_structure_gap=0.001,
        min_eligible_ratio=0.5,
    )
    assert set(result["arms"]) == {
        "SELF", "ID-MEAN", "PART-EQUAL", "PART-PERM", "CASD-LIKE"
    }
    assert abs(result["shared_advantage"]["loss_mass"]["total"] - 1.0) < 1e-6
    assert abs(result["EXP123-FULL"]["loss_mass"]["total"] - 1.0) < 1e-6
    assert result["gate"]["strict_path_loo_pass"] is True
