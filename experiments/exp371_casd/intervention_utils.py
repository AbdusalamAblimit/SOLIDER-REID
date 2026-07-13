"""Pure helpers for the exp371 pose-intervention gates.

This module intentionally has no project-level imports so the invariants can be
unit-tested without constructing a ReID model or dataset.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def build_wrong_pid_donors(pids: Sequence[int]) -> Tuple[List[int], Dict[str, int]]:
    """Build a deterministic, bijective, different-PID donor permutation.

    Grouping indices by PID and rotating the grouped list by the largest group
    size is a constructive label derangement whenever the necessary condition
    ``max_pid_count <= N/2`` holds.  The caller applies this independently to
    query and gallery so their pose multisets never cross split boundaries.
    """

    labels = [int(pid) for pid in pids]
    n = len(labels)
    if n < 2 or len(set(labels)) < 2:
        raise ValueError("wrong-PID intervention requires at least two identities")

    groups = defaultdict(list)
    for index, pid in enumerate(labels):
        groups[pid].append(index)
    ordered_groups = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    max_count = len(ordered_groups[0])
    if max_count * 2 > n:
        raise ValueError(
            "a bijective different-PID donor map is impossible because one "
            f"identity occupies {max_count}/{n} samples"
        )

    anchors = [index for group in ordered_groups for index in group]
    rotated = anchors[max_count:] + anchors[:max_count]
    donors = [-1] * n
    for anchor, donor in zip(anchors, rotated):
        donors[anchor] = donor

    collisions = sum(labels[i] == labels[j] for i, j in enumerate(donors))
    reuse = Counter(donors)
    stats = {
        "num_samples": n,
        "num_identities": len(groups),
        "max_pid_count": max_count,
        "pid_collisions": int(collisions),
        "unique_donors": len(reuse),
        "max_donor_reuse": max(reuse.values()),
    }
    if collisions:
        raise AssertionError(f"wrong-PID donor map has {collisions} PID collisions")
    if len(reuse) != n or max(reuse.values()) != 1:
        raise AssertionError("wrong-PID donor map must be bijective")
    return donors, stats


class PoseDonorDataset(Dataset):
    """Return the anchor RGB/identity metadata with another identity's pose."""

    def __init__(self, base: Dataset, donors: Sequence[int]):
        if len(base) != len(donors):
            raise ValueError("base dataset and donor map must have equal length")
        self.base = base
        self.donors = [int(i) for i in donors]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        anchor = self.base[index]
        donor = self.base[self.donors[index]]
        if len(anchor) != 6 or len(donor) != 6:
            raise ValueError("PoseDonorDataset expects PoseImageDataset 6-tuples")
        # (image, pid, camid, viewid, path, pose_dict)
        return anchor[:-1] + (donor[-1],)


def uniformize_pose_dict(pose_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Give every part the same scene-body foreground support.

    The scene body mask is the maximum response over people and keypoints.  It
    is copied to all 17 channels, removing anatomical assignment while keeping
    image-specific foreground extent.  All-zero samples fall back to a constant
    map, which makes every part genuinely equiprobable instead of invisible.
    Recursively cloning tensors prevents mutation of another intervention arm.
    """

    out: Dict[str, Any] = {}
    for key, value in pose_dict.items():
        if isinstance(value, dict):
            out[key] = uniformize_pose_dict(value)
        elif isinstance(value, torch.Tensor):
            out[key] = value.clone()
        else:
            out[key] = value

    heatmaps = out.get("heatmaps")
    person_mask = out.get("person_mask")
    if not isinstance(heatmaps, torch.Tensor) or heatmaps.ndim != 5:
        raise ValueError("pose_dict['heatmaps'] must have shape (B,P,17,H,W)")
    if not isinstance(person_mask, torch.Tensor) or person_mask.ndim != 2:
        raise ValueError("pose_dict['person_mask'] must have shape (B,P)")

    source = heatmaps.clone()
    mask = person_mask[:, :, None, None, None].to(source.dtype)
    scene = (source * mask).amax(dim=1)
    body = scene.amax(dim=1, keepdim=True)
    empty = body.amax(dim=(1, 2, 3), keepdim=True) <= 0
    body = torch.where(empty, torch.ones_like(body), body)
    equal_parts = body.expand(-1, source.shape[2], -1, -1)

    heatmaps.zero_()
    heatmaps[:, 0].copy_(equal_parts)
    person_mask.zero_()
    person_mask[:, 0].fill_(1.0)
    return out


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a CPU-contiguous tensor including dtype and shape."""

    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def validate_equal_concat(features: torch.Tensor, block_dim: int) -> Dict[str, float]:
    """Validate the proven global + pooled + five-part descriptor layout."""

    if features.ndim != 2:
        raise ValueError("features must have shape (N,D)")
    expected_dim = 7 * int(block_dim)
    if features.shape[1] != expected_dim:
        raise ValueError(
            f"expected 7x{block_dim}={expected_dim} dimensions, got {features.shape[1]}"
        )
    blocks = features.view(features.shape[0], 7, block_dim)
    norms = torch.linalg.vector_norm(blocks.float(), dim=2)
    return {
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "block_dim": int(block_dim),
        "num_blocks": 7,
        "block_norm_mean": float(norms.mean().item()),
        "block_norm_min": float(norms.min().item()),
        "block_norm_max": float(norms.max().item()),
        "block_norm_max_abs_error": float((norms - 1.0).abs().max().item()),
    }


def descriptor_gain_retention(
    packed_map: float, global_map: float, full_map: float
) -> float:
    """Fraction of the paired full-vs-global mAP gain retained by packing."""

    denominator = float(full_map) - float(global_map)
    if denominator <= 0:
        raise ValueError(
            f"full descriptor must beat global before retention is meaningful: {denominator}"
        )
    return (float(packed_map) - float(global_map)) / denominator
