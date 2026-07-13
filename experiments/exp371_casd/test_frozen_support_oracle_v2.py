import hashlib
import json
import copy
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from experiments.exp371_casd.frozen_support_oracle_v2 import (
    CACHE_SCHEMA,
    MAIN_ARMS,
    POSE_CONTROLS,
    ROUTING_ARMS,
    SCENE_ARMS,
    assign_gallery_folds,
    assert_paired_cache,
    build_arm_descriptors,
    build_fold_episode,
    common_active_mask,
    deterministic_derangement,
    dry_run_feasibility,
    evaluate_main_gate,
    file_sha256,
    json_sha256,
    load_cache,
    metadata_audit,
    response_permuted,
    run_oracle,
    support_descriptor,
)


def _content(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _synthetic_cache(role: str = "target", block_dim: int = 8):
    generator = torch.Generator().manual_seed(71)
    pid_count = 4
    gallery_per_pid = 5
    identity = F.normalize(
        torch.randn(pid_count, 7, block_dim, generator=generator), p=2, dim=2
    )
    features = []
    pids = []
    camids = []
    paths = []
    raw = []

    for pid in range(pid_count):
        value = identity[pid] + 0.20 * torch.randn(
            7, block_dim, generator=generator
        )
        features.append(F.normalize(value, p=2, dim=1))
        pids.append(pid)
        camids.append(0)
        paths.append("query/p%d.jpg" % pid)
        raw.append(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]) + pid * 0.1)

    for pid in range(pid_count):
        for view in range(gallery_per_pid):
            value = identity[pid] + 0.24 * torch.randn(
                7, block_dim, generator=generator
            )
            features.append(F.normalize(value, p=2, dim=1))
            pids.append(pid)
            camids.append(view + 1)
            paths.append("gallery/p%d_v%d.jpg" % (pid, view))
            raw.append(
                torch.tensor(
                    [
                        1.0 + 0.5 * view,
                        1.5 + ((view + 1) % 3),
                        2.0 + ((view + 2) % 4),
                        0.5 + ((view + 3) % 2),
                        2.5 + ((view + 4) % 5),
                    ]
                )
            )

    stacked = torch.stack(features)
    if role == "canonical":
        perturb = 0.08 * torch.randn(stacked.shape, generator=generator)
        stacked = F.normalize(stacked + perturb, p=2, dim=2)
        mode = "canonical"
        pose_source = "canonical_heatmap"
    elif role == "scene":
        perturb = 0.04 * torch.randn(stacked.shape, generator=generator)
        stacked = F.normalize(stacked + perturb, p=2, dim=2)
        mode = "scene_merged_correct"
        pose_source = "scene_merged_heatmap"
    else:
        mode = "target_only_correct"
        pose_source = "target_person_index_0"

    return {
        "schema_version": CACHE_SCHEMA,
        "mode": mode,
        "pose_source": pose_source,
        "features": stacked,
        "raw_pose_response": torch.stack(raw),
        "target_person_valid": torch.ones(len(paths), dtype=torch.bool),
        "person_count": torch.ones(len(paths), dtype=torch.long),
        "pids": pids,
        "camids": camids,
        "paths": paths,
        "content_sha256": [_content(path) for path in paths],
        "split": "val",
        "num_query": pid_count,
        "block_dim": block_dim,
        "weight_sha256": "w" * 64,
        "script_sha256": "s" * 64,
        "role": role,
    }


def _save_cache(tmp_path: Path, cache: dict, name: str) -> Path:
    path = tmp_path / name
    torch.save(cache, path)
    return path


def test_cache_loader_fails_closed_on_provenance_and_content(tmp_path: Path):
    target = _synthetic_cache()
    path = _save_cache(tmp_path, target, "target.pt")
    loaded = load_cache(path, role="target", expected_block_dim=8)
    assert loaded["features"].shape == (24, 7, 8)
    assert loaded["mode"] == "target_only_correct"

    wrong = dict(target)
    wrong["mode"] = "scene_merged_correct"
    wrong_path = _save_cache(tmp_path, wrong, "wrong.pt")
    with pytest.raises(ValueError, match="target-only"):
        load_cache(wrong_path, role="target", expected_block_dim=8)

    malformed = dict(target)
    malformed["content_sha256"] = list(malformed["content_sha256"])
    malformed["content_sha256"][0] = "z" * 64
    malformed_path = _save_cache(tmp_path, malformed, "malformed_content.pt")
    with pytest.raises(ValueError, match="lowercase hex"):
        load_cache(malformed_path, role="target", expected_block_dim=8)

    missing = dict(target)
    missing.pop("content_sha256")
    missing_path = _save_cache(tmp_path, missing, "missing.pt")
    with pytest.raises(ValueError, match="content_sha256"):
        load_cache(missing_path, role="target", expected_block_dim=8)


def test_content_sidecar_is_bound_to_cache_sha_and_ordered_paths(tmp_path: Path):
    target = _synthetic_cache()
    contents = target.pop("content_sha256")
    path = _save_cache(tmp_path, target, "target_sidecar.pt")
    sidecar_payload = {
        "schema_version": "exp371_content_sha256_sidecar_v1",
        "source_cache_path": str(path),
        "source_cache_file_sha256": file_sha256(path),
        "ordered_paths_sha256": json_sha256(target["paths"]),
        "sample_count": len(target["paths"]),
        "content_sha256": contents,
        "unique_content_count": len(set(contents)),
        "duplicate_content_group_count": 0,
        "duplicate_content_sample_count": 0,
    }
    sidecar = tmp_path / "target_content_sha256.json"
    sidecar.write_text(json.dumps(sidecar_payload))
    loaded = load_cache(
        path,
        role="target",
        expected_block_dim=8,
        content_sidecar=sidecar,
    )
    assert loaded["content_sha256"] == contents
    assert loaded["content_provenance"]["storage"] == "sidecar"

    sidecar_payload["source_cache_file_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(sidecar_payload))
    with pytest.raises(ValueError, match="source cache SHA"):
        load_cache(
            path,
            role="target",
            expected_block_dim=8,
            content_sidecar=sidecar,
        )


def test_paired_extractions_require_identical_episode_metadata():
    target = _synthetic_cache()
    canonical = _synthetic_cache("canonical")
    assert_paired_cache(target, canonical)
    canonical["camids"] = list(canonical["camids"])
    canonical["camids"][3] = 99
    with pytest.raises(ValueError, match="camids"):
        assert_paired_cache(target, canonical)


def test_metadata_duplicate_path_and_content_fail_before_metrics():
    cache = _synthetic_cache()
    cache["paths"] = list(cache["paths"])
    cache["paths"][5] = cache["paths"][4]
    with pytest.raises(ValueError, match="hard gate"):
        metadata_audit(cache)


def test_standard_query_gallery_same_content_is_allowed_only_for_same_pid_cam():
    cache = _synthetic_cache()
    query_index = 0
    gallery_index = cache["num_query"]
    cache["content_sha256"] = list(cache["content_sha256"])
    cache["content_sha256"][gallery_index] = cache["content_sha256"][query_index]
    cache["pids"] = list(cache["pids"])
    cache["camids"] = list(cache["camids"])
    cache["pids"][gallery_index] = cache["pids"][query_index]
    cache["camids"][gallery_index] = cache["camids"][query_index]
    audit = metadata_audit(cache)
    assert audit["allowed_query_gallery_same_pidcam_content_count"] == 1
    assert audit["forbidden_duplicate_content_count"] == 0


def test_allowed_query_gallery_copy_is_excluded_from_support_and_reference():
    cache = _synthetic_cache()
    query_index = 0
    gallery_index = cache["num_query"]
    assert cache["pids"][gallery_index] == cache["pids"][query_index]
    cache["camids"][gallery_index] = cache["camids"][query_index]
    cache["content_sha256"][gallery_index] = cache["content_sha256"][query_index]
    metadata_audit(cache)
    folds = assign_gallery_folds(cache, 371)
    for fold in range(5):
        episode = build_fold_episode(
            cache,
            folds,
            fold,
            permutation_seed=1371,
            camera_protocol="cross-camera",
        )
        local = episode["query_indices"].index(query_index)
        assert gallery_index not in episode["donors"][local]
        if gallery_index in episode["reference_indices"]:
            ref_local = episode["reference_indices"].index(gallery_index)
            assert not bool(episode["valid"][local, ref_local].item())

    cache["camids"][gallery_index] = cache["camids"][query_index] + 1
    with pytest.raises(ValueError, match="hard gate"):
        metadata_audit(cache)

    cache = _synthetic_cache()
    cache["content_sha256"] = list(cache["content_sha256"])
    cache["content_sha256"][5] = cache["content_sha256"][4]
    with pytest.raises(ValueError, match="hard gate"):
        metadata_audit(cache)


def test_pid_internal_folds_are_stable_and_support_reference_disjoint():
    cache = _synthetic_cache()
    first = assign_gallery_folds(cache, seed=371)
    second = assign_gallery_folds(cache, seed=371)
    assert first == second
    for pid in range(4):
        indices = [
            index
            for index in range(cache["num_query"], len(cache["pids"]))
            if cache["pids"][index] == pid
        ]
        assert sorted(first[index] for index in indices) == list(range(5))

    episode = build_fold_episode(
        cache,
        first,
        0,
        permutation_seed=1371,
        camera_protocol="cross-camera",
    )
    assert episode["eligible_query_ratio"] == 1.0
    assert episode["eligible_pid_ratio"] == 1.0
    assert episode["support_reference_path_overlap"] == 0
    assert episode["support_reference_content_overlap"] == 0
    assert all(count == 4 for count in episode["donor_count"])
    for query_index, same_donors, wrong_donors in zip(
        episode["query_indices"], episode["donors"], episode["wrong_donors"]
    ):
        assert len(wrong_donors) == len(same_donors)
        assert all(
            cache["pids"][index] != cache["pids"][query_index]
            for index in wrong_donors
        )


def test_derangements_preserve_multisets_and_common_mask_prevents_fallback():
    paths = ["a", "b", "c"]
    for path in paths:
        order = deterministic_derangement(
            path, seed=1371, namespace="feature"
        )
        assert sorted(order.tolist()) == list(range(5))
        assert torch.all(order != torch.arange(5))

    raw = torch.tensor(
        [
            [1.0, 0.0, 2.0, 3.0, 4.0],
            [2.0, 0.0, 1.0, 4.0, 3.0],
            [3.0, 0.0, 4.0, 1.0, 2.0],
        ]
    )
    permuted = response_permuted(raw, paths, seed=1371)
    for donor in range(raw.shape[0]):
        assert sorted(permuted[donor].tolist()) == sorted(raw[donor].tolist())
    mask = common_active_mask(raw, paths, seed=1371)
    assert mask.dtype == torch.bool
    assert not bool(mask[1].item())


def test_all_part_arms_keep_query_global_pooled_and_common_inactive_slots():
    generator = torch.Generator().manual_seed(17)
    blocks = F.normalize(torch.randn(4, 7, 6, generator=generator), p=2, dim=2)
    raw = torch.tensor(
        [
            [1.0, 0.0, 1.0, 2.0, 3.0],
            [2.0, 0.0, 2.0, 3.0, 1.0],
            [3.0, 0.0, 3.0, 1.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    donor_paths = ["d0", "d1", "d2"]
    active = common_active_mask(raw[1:], donor_paths, seed=1371)
    for arm in (
        "ID-GLOBAL",
        "ID-MEAN",
        "PART-EQUAL",
        "SLOT-PERM",
        "AGREE",
        "POSE-RESP",
        "RESP-PERM",
    ):
        descriptor = support_descriptor(
            blocks,
            raw,
            0,
            [1, 2, 3],
            active,
            arm,
            donor_paths=donor_paths,
            permutation_seed=1371,
        ).view(7, 6)
        for block_index in (0, 1):
            assert torch.allclose(
                F.normalize(descriptor[block_index], dim=0),
                blocks[0, block_index],
                atol=1e-6,
            )
        inactive_slots = (~active).nonzero(as_tuple=False).flatten().tolist()
        for slot in inactive_slots:
            assert torch.allclose(
                F.normalize(descriptor[2 + slot], dim=0),
                blocks[0, 2 + slot],
                atol=1e-6,
            )


def test_extraction_routing_matrix_reuses_target_raw_response():
    target = _synthetic_cache()
    canonical = _synthetic_cache("canonical")
    folds = assign_gallery_folds(target, 371)
    episode = build_fold_episode(
        target,
        folds,
        0,
        permutation_seed=1371,
        camera_protocol="cross-camera",
    )
    target_equal = build_arm_descriptors(
        target, target, episode, "PART-EQUAL", permutation_seed=1371
    )
    canonical_equal = build_arm_descriptors(
        canonical, target, episode, "PART-EQUAL", permutation_seed=1371
    )
    canonical_pose = build_arm_descriptors(
        canonical, target, episode, "POSE-RESP", permutation_seed=1371
    )
    assert not torch.allclose(target_equal, canonical_equal)
    assert not torch.allclose(canonical_equal, canonical_pose)

    altered = dict(target)
    altered["raw_pose_response"] = target["raw_pose_response"].flip(1)
    # Equal routing is feature-only and must ignore raw response magnitudes.
    equal_after_raw_change = build_arm_descriptors(
        canonical, altered, episode, "PART-EQUAL", permutation_seed=1371
    )
    assert torch.allclose(canonical_equal, equal_after_raw_change)


def test_cross_camera_protocol_never_falls_back_to_same_camera():
    cache = _synthetic_cache()
    cache["camids"] = [0 for _ in cache["camids"]]
    folds = assign_gallery_folds(cache, 371)
    with pytest.raises(ValueError, match="no eligible query"):
        build_fold_episode(
            cache,
            folds,
            0,
            permutation_seed=1371,
            camera_protocol="cross-camera",
        )


def test_full_oracle_has_unique_descriptors_all_arms_and_2x3_matrix():
    target = _synthetic_cache()
    canonical = _synthetic_cache("canonical")
    scene = _synthetic_cache("scene")
    result = run_oracle(
        target,
        canonical_cache=canonical,
        scene_cache=scene,
        bootstrap_replicates=40,
        device=torch.device("cpu"),
        distance_batch=8,
    )
    assert result["fold_count"] == 5
    json.dumps(result)
    assert set(result["target"]["aggregate"]) == set(MAIN_ARMS)
    assert set(result["canonical"]["aggregate"]) == {"SELF", *ROUTING_ARMS}
    assert set(result["scene"]["aggregate"]) == set(SCENE_ARMS)
    assert result["gate"]["checks"]["canonical_extraction_routing_matrix_complete"] is True
    assert len(result["target"]["folds"]) == 5
    for fold in result["target"]["folds"]:
        assert fold["episode"]["eligible_query_ratio"] == 1.0
        assert fold["episode"]["eligible_pid_ratio"] == 1.0
        assert fold["episode"]["support_reference_path_overlap"] == 0
        assert fold["episode"]["support_reference_content_overlap"] == 0
        # One descriptor matrix SHA per arm; there are no endpoint corrections.
        assert set(fold["descriptor_sha256"]) == set(MAIN_ARMS)
        assert all(len(value) == 64 for value in fold["descriptor_sha256"].values())
        assert "selection_bias_audit" in fold
    assert set(result["gate"]["pairwise_pid_grouped_bootstrap"]) == set(POSE_CONTROLS)
    assert "WRONG-ID" not in POSE_CONTROLS
    assert result["gate"]["full_geometry_boundary"]["enters_routing_gate"] is False
    assert len(result["gate"]["fold_strongest_control"]) == 5
    assert set(result["gate"]["fold_strongest_control"]).issubset(set(POSE_CONTROLS))


def test_default_real_cache_path_can_stop_at_metric_free_dry_run():
    target = _synthetic_cache()
    canonical = _synthetic_cache("canonical")
    scene = _synthetic_cache("scene")
    result = dry_run_feasibility(
        target,
        canonical_cache=canonical,
        scene_cache=scene,
        split_seed=371,
        permutation_seed=1371,
        camera_protocol="cross-camera",
        max_queries=0,
    )
    assert result["status"] == "DRY_RUN_COMPLETE"
    assert result["metrics_computed"] is False
    assert result["coverage_hard_gate"] is True
    assert len(result["folds"]) == 5
    assert all("arms" not in fold for fold in result["folds"])


def test_gate_uses_each_folds_own_strongest_control_not_global_winner():
    result = run_oracle(
        _synthetic_cache(),
        canonical_cache=_synthetic_cache("canonical"),
        scene_cache=_synthetic_cache("scene"),
        bootstrap_replicates=20,
        device=torch.device("cpu"),
        distance_batch=8,
    )
    target = copy.deepcopy(result["target"])
    pose_fold0 = target["aggregate"]["POSE-RESP"]["mAP"]["fold_values"][0]
    target["aggregate"]["ID-GLOBAL"]["mAP"]["fold_values"][0] = pose_fold0 + 0.01
    gate = evaluate_main_gate(
        target,
        bootstrap_seed=2371,
        bootstrap_replicates=20,
        canonical_result=result["canonical"],
        scene_result=result["scene"],
    )
    assert gate["fold_strongest_control"][0] == "ID-GLOBAL"
    assert gate["fold_pose_vs_strongest"][0] < 0
    assert gate["checks"]["all_fold_directions_positive"] is False
