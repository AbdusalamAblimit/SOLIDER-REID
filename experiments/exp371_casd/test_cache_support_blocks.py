from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp371_casd.cache_support_blocks import (
    PART_KPS,
    POSE_PROVENANCE,
    SCHEMA_VERSION,
    _maxsim_pair,
    assert_paired_target_cache,
    cache_payload,
    extract_loader,
    extract_support_batch,
    flip_support_batch,
    raw_part_response,
)
from experiments.exp371_casd.frozen_support_oracle_v2 import load_cache


def test_maxsim_merge_matches_equal_concat_normalize_then_average_order():
    global_orig = torch.tensor([[3.0, 4.0]])
    global_flip = torch.tensor([[0.0, 2.0]])
    parts_orig = global_orig[:, None, :].repeat(1, 5, 1)
    parts_flip = global_flip[:, None, :].repeat(1, 5, 1)
    merged = _maxsim_pair(
        {
            "global_feat": global_orig,
            "kp_feats": parts_orig,
            "kp_weights": torch.full((1, 5), 0.2),
        },
        {
            "global_feat": global_flip,
            "kp_feats": parts_flip,
            "kp_weights": torch.full((1, 5), 0.2),
        },
        block_dim=2,
    )
    expected = F.normalize(
        (
            F.normalize(global_orig, p=2, dim=1)
            + F.normalize(global_flip, p=2, dim=1)
        ) / 2.0,
        p=2,
        dim=1,
    )
    raw_first = F.normalize((global_orig + global_flip) / 2.0, p=2, dim=1)
    assert torch.allclose(merged["global_feat"], expected)
    assert torch.allclose(merged["kp_feats"][:, 0], expected)
    assert not torch.allclose(expected, raw_first)


def _pose(batch: int = 2, persons: int = 2, height: int = 8, width: int = 6):
    heatmaps = torch.zeros(batch, persons, 17, height, width)
    for b in range(batch):
        for keypoint in range(17):
            heatmaps[b, 0, keypoint] = (b + 1) * (keypoint + 1) / 100.0
    heatmaps[:, 1] = 0.25
    person_mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)[:batch]
    scores = torch.ones(batch, persons, 17)
    keypoints = torch.zeros(batch, persons, 17, 2)
    return {
        "heatmaps": heatmaps,
        "person_mask": person_mask,
        "scores": scores,
        "keypoints": keypoints,
    }


class FakeLGPA(nn.Module):
    def __init__(self, block_dim: int = 8, corrupt_parts: bool = False):
        super().__init__()
        self.in_planes = block_dim
        self.pose_test_feat = "equal_concat"
        self.use_target_heatmap = True
        self._lgpa_fixed_bands = False
        self._lgpa_no_pose = False
        self.corrupt_parts = corrupt_parts
        self.canonical_calls = 0

    def set_pose_mode(self, pose_mode: str):
        self.use_target_heatmap = pose_mode == "target"
        self._lgpa_fixed_bands = pose_mode == "canonical"

    @staticmethod
    def _prepare_pose(pose_dict):
        heatmaps = pose_dict["heatmaps"]
        person_mask = pose_dict["person_mask"]
        scene = (heatmaps * person_mask[:, :, None, None, None]).max(dim=1).values
        target = heatmaps[:, 0] * person_mask[:, 0, None, None, None]
        return scene, torch.zeros_like(pose_dict["scores"][:, 0]), target, target - scene

    def _canonical_heatmap(self, batch: int, device):
        self.canonical_calls += 1
        # Synthetic fixed heatmap; the production extractor invokes the real
        # PoseBackboneModel._canonical_heatmap through the same method call.
        rows = torch.linspace(0.1, 1.0, 17, device=device)[:, None, None]
        return rows.expand(batch, 17, 8, 6).contiguous()

    def _selected_heatmap(self, pose_dict):
        scene, _scores, target, _diff = self._prepare_pose(pose_dict)
        if self._lgpa_fixed_bands:
            return self._canonical_heatmap(scene.shape[0], scene.device)
        return target if self.use_target_heatmap else scene

    def forward(self, img, cam_label=None, view_label=None, pose_dict=None):
        batch = img.shape[0]
        dim = self.in_planes
        base = img.float().mean(dim=(1, 2, 3))
        selected = self._selected_heatmap(pose_dict)
        pose_raw, pose_relative = [], []
        for indices in PART_KPS:
            pose_raw.append(selected[:, indices].max(dim=1).values.mean(dim=(1, 2)))
        pose_raw = torch.stack(pose_raw, dim=1)
        pose_relative = pose_raw / pose_raw.sum(dim=1, keepdim=True).clamp_min(1e-6)
        vectors = []
        for block in range(7):
            axis = torch.arange(dim, device=img.device).float() + 1 + block
            pose_term = (
                pose_raw[:, block - 2:block - 1]
                if block >= 2 else pose_raw.mean(dim=1, keepdim=True)
            )
            value = (
                axis[None, :]
                + base[:, None] * (block + 1)
                + pose_term * axis[None, :] * 0.1
            )
            vectors.append(F.normalize(value, p=2, dim=1))
        blocks = torch.stack(vectors, dim=1)
        extra = [torch.zeros(batch, dim, 4, 3, device=img.device)]
        if self.pose_test_feat == "equal_concat":
            return blocks.reshape(batch, -1), extra
        if self.pose_test_feat != "maxsim_hybrid":
            raise ValueError(self.pose_test_feat)
        kp = blocks[:, 2:7].clone()
        if self.corrupt_parts:
            kp[:, 0, 0] += 0.1
            kp = F.normalize(kp, p=2, dim=2)
        return {
            "global_feat": blocks[:, 0],
            "kp_feats": kp,
            "kp_weights": pose_relative,
        }, extra


def test_raw_response_is_target_only_and_invalid_target_stays_zero():
    pose = _pose()
    result = raw_part_response(pose, (4, 3))
    assert result["raw_pose_response"].shape == (2, 5)
    assert result["target_person_valid"].tolist() == [True, False]
    assert result["person_count"].tolist() == [2, 1]
    assert torch.all(result["raw_pose_response"][0] > 0)
    assert torch.count_nonzero(result["raw_pose_response"][1]) == 0
    assert torch.allclose(
        result["raw_response_relative_allocation"][0].sum(),
        torch.tensor(1.0),
    )


def test_part_groups_make_raw_response_flip_invariant():
    pose = _pose(batch=1)
    img = torch.randn(1, 3, 16, 10)
    _img_flip, pose_flip = flip_support_batch(img, pose)
    original = raw_part_response(pose, (4, 3))["raw_pose_response"]
    flipped = raw_part_response(pose_flip, (4, 3))["raw_pose_response"]
    assert torch.allclose(original, flipped, atol=1e-7)
    assert len(PART_KPS) == 5


def test_same_batch_equal_and_maxsim_metadata_are_locked():
    model = FakeLGPA(block_dim=8)
    pose = _pose()
    img = torch.randn(2, 3, 16, 10)
    output = extract_support_batch(
        model,
        img,
        pose,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        flip_test=True,
        block_dim=8,
    )
    blocks = output["features"].view(2, 7, 8)
    assert output["features"].shape == (2, 56)
    assert output["kp_feats"].shape == (2, 5, 8)
    assert torch.allclose(blocks[:, 2:7], output["kp_feats"], atol=1e-6)
    assert float(output["part_consistency_max_abs"].item()) < 1e-6
    assert float(output["raw_flip_max_abs_diff"].item()) < 1e-7
    assert model.pose_test_feat == "equal_concat"


def test_part_mismatch_fails_closed():
    model = FakeLGPA(block_dim=8, corrupt_parts=True)
    with pytest.raises(AssertionError, match="equal/maxsim mismatch"):
        extract_support_batch(
            model,
            torch.randn(2, 3, 16, 10),
            _pose(),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            flip_test=True,
            block_dim=8,
        )


def test_cache_payload_has_provenance_and_hashes():
    model = FakeLGPA(block_dim=8)
    tensors = extract_support_batch(
        model,
        torch.randn(2, 3, 16, 10),
        _pose(),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        flip_test=True,
        block_dim=8,
    )
    payload = cache_payload(
        tensors,
        pids=[1, 2],
        camids=[3, 4],
        paths=["a/../a/q.jpg", "b/g.jpg"],
        split="val",
        num_query=1,
        block_dim=8,
        weight_sha256="weight",
        script_sha256="script",
        flip_test=True,
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["mode"] == "target_only_correct"
    assert payload["pose_source"].startswith("target_person_index_0")
    assert "not absolute visibility" in payload["relative_allocation_semantics"]
    assert payload["paths"] == ["a/q.jpg", "b/g.jpg"]
    assert set(payload["tensor_sha256"]) == {
        "features", "kp_feats", "relative_allocation", "raw_pose_response",
        "raw_response_relative_allocation", "target_person_valid", "person_count",
    }


def _paired_payloads():
    torch.manual_seed(371)
    pose = _pose()
    img = torch.randn(2, 3, 16, 10)
    outputs = {}
    canonical_calls = 0
    for pose_mode in ("target", "canonical", "scene"):
        model = FakeLGPA(block_dim=8)
        model.set_pose_mode(pose_mode)
        tensors = extract_support_batch(
            model,
            img,
            pose,
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            flip_test=True,
            block_dim=8,
            pose_mode=pose_mode,
        )
        outputs[pose_mode] = cache_payload(
            tensors,
            pids=[1, 2],
            camids=[3, 4],
            paths=["query/a.jpg", "gallery/b.jpg"],
            split="val",
            num_query=1,
            block_dim=8,
            weight_sha256="w" * 64,
            script_sha256="s" * 64,
            flip_test=True,
            pose_mode=pose_mode,
        )
        canonical_calls += model.canonical_calls
    return outputs, canonical_calls


def test_paired_modes_change_extraction_but_keep_target_routing_metadata():
    payloads, canonical_calls = _paired_payloads()
    target = payloads["target"]
    canonical = payloads["canonical"]
    scene = payloads["scene"]

    assert canonical_calls > 0
    assert canonical["mode"] == POSE_PROVENANCE["canonical"]["mode"]
    assert "canonical" in canonical["pose_source"].lower()
    assert scene["mode"] == POSE_PROVENANCE["scene"]["mode"]
    assert "scene" in scene["pose_source"].lower()
    assert not torch.equal(target["features"], canonical["features"])
    assert not torch.equal(target["features"], scene["features"])

    for other in (canonical, scene):
        assert_paired_target_cache(target, other)
        assert torch.equal(target["raw_pose_response"], other["raw_pose_response"])
        assert torch.equal(
            target["target_person_valid"], other["target_person_valid"]
        )
        assert torch.equal(target["person_count"], other["person_count"])
        # Invalid person-0 remains absent from E x R routing even when scene or
        # canonical extraction itself has a nonzero pose allocation.
        assert torch.count_nonzero(other["raw_pose_response"][1]) == 0


def test_paired_cache_metadata_mismatch_fails_closed():
    payloads, _canonical_calls = _paired_payloads()
    target = payloads["target"]
    bad_cam = dict(payloads["canonical"])
    bad_cam["camids"] = list(bad_cam["camids"])
    bad_cam["camids"][1] += 1
    with pytest.raises(ValueError, match="camids"):
        assert_paired_target_cache(target, bad_cam)

    bad_valid = dict(payloads["scene"])
    bad_valid["target_person_valid"] = bad_valid["target_person_valid"].clone()
    bad_valid["target_person_valid"][1] = True
    with pytest.raises(ValueError, match="target_person_valid"):
        assert_paired_target_cache(target, bad_valid)


def test_pose_mode_model_switch_mismatch_fails_closed():
    model = FakeLGPA(block_dim=8)
    with pytest.raises(AssertionError, match="pose-mode protocol"):
        extract_support_batch(
            model,
            torch.randn(2, 3, 16, 10),
            _pose(),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            flip_test=True,
            block_dim=8,
            pose_mode="scene",
        )


def test_extract_loader_audits_feed_cache_payload_without_missing_keys():
    model = FakeLGPA(block_dim=8)
    loader = [
        (
            torch.randn(2, 3, 16, 10),
            torch.tensor([1, 2]),
            torch.tensor([3, 4]),
            torch.tensor([3, 4]),
            torch.zeros(2, dtype=torch.long),
            ["query/a.jpg", "gallery/b.jpg"],
            _pose(),
        )
    ]
    tensors, pids, camids, paths = extract_loader(
        model,
        loader,
        device="cpu",
        flip_test=True,
        block_dim=8,
        pose_mode="target",
        consistency_atol=2e-5,
        flip_raw_atol=2e-6,
    )
    payload = cache_payload(
        tensors,
        pids=pids,
        camids=camids,
        paths=paths,
        split="val",
        num_query=1,
        block_dim=8,
        weight_sha256="w" * 64,
        script_sha256="s" * 64,
        flip_test=True,
        pose_mode="target",
    )
    assert "extraction_flip_max_abs_diff" in payload["audit"]


@pytest.mark.parametrize("pose_mode", ["target", "canonical", "scene"])
def test_gate_c_v2_loader_recognizes_extractor_provenance(
    tmp_path: Path, pose_mode: str
):
    payloads, _canonical_calls = _paired_payloads()
    payload = dict(payloads[pose_mode])
    payload["content_sha256"] = ["a" * 64, "b" * 64]
    path = tmp_path / (pose_mode + ".pt")
    torch.save(payload, path)
    loaded = load_cache(path, role=pose_mode, expected_block_dim=8)
    assert loaded["mode"] == POSE_PROVENANCE[pose_mode]["mode"]
