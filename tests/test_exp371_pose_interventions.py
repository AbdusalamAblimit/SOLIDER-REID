import torch

from experiments.exp371_casd.intervention_utils import (
    PoseDonorDataset,
    build_wrong_pid_donors,
    uniformize_pose_dict,
    validate_equal_concat,
)


def test_wrong_pid_donors_never_reuse_anchor_identity():
    pids = [0, 0, 1, 1, 2, 2, 3, 3]
    donors, stats = build_wrong_pid_donors(pids)
    assert all(pids[i] != pids[j] for i, j in enumerate(donors))
    assert sorted(donors) == list(range(len(pids)))
    assert stats["pid_collisions"] == 0
    assert stats["max_donor_reuse"] == 1
    assert stats["num_samples"] == len(pids)


def test_pose_donor_dataset_preserves_anchor_and_replaces_pose():
    base = [
        (f"img{i}", pid, i, 0, f"path{i}", {"pose": i})
        for i, pid in enumerate([0, 1, 2])
    ]
    wrapped = PoseDonorDataset(base, [1, 2, 0])
    item = wrapped[0]
    assert item[:-1] == base[0][:-1]
    assert item[-1] == base[1][-1]


def test_uniformize_pose_dict_is_non_mutating_and_spatially_uniform():
    pose = {
        "heatmaps": torch.randn(2, 3, 17, 4, 2),
        "person_mask": torch.ones(2, 3),
        "teacher_pose": {
            "heatmaps": torch.randn(2, 3, 17, 4, 2),
            "person_mask": torch.ones(2, 3),
        },
    }
    original = pose["heatmaps"].clone()
    out = uniformize_pose_dict(pose)
    assert torch.equal(pose["heatmaps"], original)
    equal_parts = out["heatmaps"][:, 0]
    assert torch.equal(equal_parts[:, :1].expand_as(equal_parts), equal_parts)
    assert torch.all(out["heatmaps"][:, 1:] == 0)
    assert torch.all(out["person_mask"][:, 0] == 1)
    assert torch.all(out["person_mask"][:, 1:] == 0)
    teacher_equal = out["teacher_pose"]["heatmaps"][:, 0]
    assert torch.equal(teacher_equal[:, :1].expand_as(teacher_equal), teacher_equal)


def test_uniformize_pose_dict_falls_back_for_empty_pose():
    pose = {
        "heatmaps": torch.zeros(1, 2, 17, 3, 2),
        "person_mask": torch.zeros(1, 2),
    }
    out = uniformize_pose_dict(pose)
    assert torch.all(out["heatmaps"][:, 0] == 1)


def test_validate_equal_concat_requires_seven_blocks():
    features = torch.randn(5, 7 * 4)
    features = torch.nn.functional.normalize(features.view(5, 7, 4), dim=2).view(5, 28)
    stats = validate_equal_concat(features, 4)
    assert stats["num_blocks"] == 7
    assert stats["feature_dim"] == 28
    assert stats["block_norm_max_abs_error"] < 1e-6
