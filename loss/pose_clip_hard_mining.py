"""Training-only pose x CLIP hard-pair selection for exp409 PCHM."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REGION_JOINTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9, 10),
    (11, 12),
    (11, 12, 13, 14),
    (13, 14, 15, 16),
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PoseClipMiningCache:
    """Strict loader for one fresh exp409 region-isolated CLIP cache."""

    def __init__(
        self,
        path,
        expected_sha256,
        expected_clip_checkpoint_sha256,
        expected_pose_manifest_sha256,
    ):
        configured = Path(path).expanduser()
        if not configured.is_absolute():
            raise ValueError("PCHM cache path must be absolute")
        resolved = configured.resolve(strict=True)
        if resolved != configured:
            raise RuntimeError("PCHM cache must use its canonical path")
        if not expected_sha256:
            raise ValueError("PCHM cache SHA256 is required")
        self.sha256 = _sha256_file(resolved)
        if self.sha256 != expected_sha256:
            raise RuntimeError("PCHM cache SHA256 mismatch")
        with np.load(str(resolved), allow_pickle=False) as payload:
            required = {
                "schema",
                "relative_paths",
                "image_sha256",
                "features",
                "valid",
                "preprocessing",
                "pose_manifest_sha256",
                "clip_checkpoint_sha256",
                "source_head",
                "builder_sha256",
                "teacher_source_sha256",
            }
            if set(payload.files) != required:
                raise RuntimeError("unexpected PCHM cache fields")
            schema = str(payload["schema"].item())
            path_array = payload["relative_paths"]
            image_sha256 = payload["image_sha256"]
            feature_array = payload["features"]
            valid_array = payload["valid"]
            preprocessing = str(payload["preprocessing"].item())
            pose_manifest_sha256 = str(payload["pose_manifest_sha256"].item())
            clip_checkpoint_sha256 = str(
                payload["clip_checkpoint_sha256"].item()
            )
            source_head = str(payload["source_head"].item())
            builder_sha256 = str(payload["builder_sha256"].item())
            teacher_source_sha256 = str(
                payload["teacher_source_sha256"].item()
            )
            if path_array.ndim != 1 or path_array.dtype.kind != "U":
                raise RuntimeError("PCHM paths must be a unicode vector")
            if feature_array.shape != (len(path_array), 5, 768):
                raise RuntimeError("PCHM feature shape mismatch")
            if feature_array.dtype != np.float16:
                raise RuntimeError("PCHM features must be float16")
            if valid_array.shape != (len(path_array), 5):
                raise RuntimeError("PCHM validity shape mismatch")
            if valid_array.dtype != np.bool_:
                raise RuntimeError("PCHM validity must be boolean")
            if image_sha256.shape != path_array.shape or image_sha256.dtype.kind != "U":
                raise RuntimeError("PCHM image SHA vector mismatch")
            hex64 = re.compile(r"[0-9a-f]{64}")
            if any(hex64.fullmatch(str(value)) is None for value in image_sha256):
                raise RuntimeError("PCHM image SHA vector is invalid")
            if preprocessing != "raw-rgb-pose-resize-384x128-no-augmentation":
                raise RuntimeError("PCHM preprocessing provenance mismatch")
            if pose_manifest_sha256 != expected_pose_manifest_sha256:
                raise RuntimeError("PCHM pose manifest provenance mismatch")
            if clip_checkpoint_sha256 != expected_clip_checkpoint_sha256:
                raise RuntimeError("PCHM CLIP checkpoint provenance mismatch")
            if re.fullmatch(r"[0-9a-f]{40}", source_head) is None:
                raise RuntimeError("PCHM source HEAD provenance is invalid")
            if hex64.fullmatch(builder_sha256) is None or hex64.fullmatch(
                teacher_source_sha256
            ) is None:
                raise RuntimeError("PCHM source SHA provenance is invalid")
            paths = tuple(str(item) for item in path_array.tolist())
            features = torch.from_numpy(feature_array.copy())
            valid = torch.from_numpy(valid_array.copy())
        if schema != "exp409-pchm-cache-v1":
            raise RuntimeError("PCHM cache schema mismatch")
        if not paths or len(paths) != len(set(paths)):
            raise RuntimeError("PCHM cache paths must be nonempty and unique")
        if valid.shape != features.shape[:2] or len(paths) != features.shape[0]:
            raise RuntimeError("PCHM cache arrays disagree")
        if not bool(torch.isfinite(features.float()).all()):
            raise RuntimeError("PCHM cache contains non-finite features")
        norms = features.float().norm(dim=-1)
        active_norms = norms[valid]
        if active_norms.numel() == 0 or not bool(
            torch.allclose(
                active_norms,
                torch.ones_like(active_norms),
                atol=2e-3,
                rtol=2e-3,
            )
        ):
            raise RuntimeError("PCHM valid features must be L2 normalized")
        self.path = resolved
        self.paths = paths
        self._features = features
        self._valid = valid
        self._index = {path: index for index, path in enumerate(paths)}
        self.image_sha256 = tuple(str(value) for value in image_sha256.tolist())
        self.preprocessing = preprocessing
        self.pose_manifest_sha256 = pose_manifest_sha256
        self.clip_checkpoint_sha256 = clip_checkpoint_sha256
        self.source_head = source_head
        self.builder_sha256 = builder_sha256
        self.teacher_source_sha256 = teacher_source_sha256

    def __len__(self):
        return len(self.paths)

    def lookup(self, relative_paths, image_sha256=None):
        try:
            indices = [self._index[str(path)] for path in relative_paths]
        except KeyError as error:
            raise RuntimeError("PCHM batch path is absent from cache") from error
        if image_sha256 is None or len(image_sha256) != len(indices):
            raise RuntimeError("PCHM batch image SHA binding is required")
        expected = [self.image_sha256[index] for index in indices]
        if tuple(str(value) for value in image_sha256) != tuple(expected):
            raise RuntimeError("PCHM batch image SHA binding mismatch")
        index = torch.as_tensor(indices, dtype=torch.long)
        return self._features[index], self._valid[index]


def pose_visibility_signature(scores, valid):
    """Return five soft visibility values from one augmented COCO-17 pose."""
    if scores.ndim != 2 or scores.shape[1] != 17:
        raise ValueError("scores must have shape [B,17]")
    if valid.shape != scores.shape:
        raise ValueError("valid must match scores")
    confidence = scores.float().clamp(0.0, 1.0) * valid.bool().float()
    slots = []
    for joints in REGION_JOINTS:
        index = torch.as_tensor(joints, device=scores.device)
        slots.append(confidence.index_select(1, index).mean(dim=1))
    signature = torch.stack(slots, dim=1)
    if not bool(torch.isfinite(signature).all()):
        raise RuntimeError("non-finite pose visibility signature")
    return signature


def _pairwise_clip_similarity(features, valid):
    if features.ndim != 3 or features.shape[1] != len(REGION_JOINTS):
        raise ValueError("features must have shape [B,5,D]")
    if valid.shape != features.shape[:2]:
        raise ValueError("feature validity shape mismatch")
    normalized = F.normalize(features.float(), dim=-1)
    per_slot = torch.einsum("brd,crd->bcr", normalized, normalized)
    common = valid[:, None, :].bool() & valid[None, :, :].bool()
    count = common.sum(dim=-1)
    similarity = (per_slot * common.float()).sum(dim=-1) / count.clamp_min(1)
    similarity = similarity.masked_fill(count == 0, float("-inf"))
    return similarity, count > 0


def _ordinal_rank(values, candidates):
    """Rank higher values higher, preserving equal-value ties."""
    if values.ndim != 2 or candidates.shape != values.shape:
        raise ValueError("rank values/candidates must be square matrices")
    better_than = values.unsqueeze(2) > values.unsqueeze(1)
    return (better_than & candidates.unsqueeze(1)).sum(dim=-1)


def _lexicographic_argmax(rank_sum, clip_rank, candidates):
    batch = rank_sum.shape[0]
    selected = torch.empty(batch, dtype=torch.long, device=rank_sum.device)
    for anchor in range(batch):
        indices = torch.nonzero(candidates[anchor], as_tuple=False).flatten()
        if indices.numel() == 0:
            raise RuntimeError("PCHM anchor has no valid candidate")
        joint = rank_sum[anchor, indices]
        indices = indices[joint == joint.max()]
        clip = clip_rank[anchor, indices]
        indices = indices[clip == clip.max()]
        selected[anchor] = indices.min()
    return selected


def select_pose_clip_pairs(
    labels,
    visibility,
    clip_features,
    clip_valid,
    *,
    mode="correct",
    wrong_shift=4,
):
    """Select deterministic PCHM positive/negative indices and diagnostics."""
    if labels.ndim != 1 or visibility.shape != (labels.numel(), 5):
        raise ValueError("PCHM labels/visibility shape mismatch")
    if clip_features.shape[:2] != visibility.shape or clip_valid.shape != visibility.shape:
        raise ValueError("PCHM CLIP batch shape mismatch")
    if labels.numel() < 2:
        raise ValueError("PCHM requires a nontrivial batch")
    if mode not in {
        "correct",
        "wrong_rgb",
        "generic",
        "zero",
        "pose_shuffle",
        "clip_only",
    }:
        raise ValueError("unsupported PCHM control mode")
    visibility = visibility.float()
    clip_features = clip_features.float()
    clip_valid = clip_valid.bool()
    if mode in {"wrong_rgb", "pose_shuffle"}:
        shift = int(wrong_shift)
        if shift <= 0 or shift >= labels.numel():
            raise ValueError("PCHM control shift is outside the batch")
        shifted_labels = torch.roll(labels, shifts=-shift, dims=0)
        if not bool((shifted_labels != labels).all()):
            raise RuntimeError("PCHM control shift is not different-PID")
        if mode == "wrong_rgb":
            clip_features = torch.roll(clip_features, shifts=-shift, dims=0)
            clip_valid = torch.roll(clip_valid, shifts=-shift, dims=0)
        else:
            visibility = torch.roll(visibility, shifts=-shift, dims=0)
    elif mode == "generic":
        weights = clip_valid.float()
        generic = (clip_features * weights[..., None]).sum(dim=0)
        generic = generic / weights.sum(dim=0).clamp_min(1.0)[..., None]
        generic = F.normalize(generic, dim=-1)
        available = weights.sum(dim=0) > 0
        clip_features = generic.unsqueeze(0).expand_as(clip_features)
        clip_valid = available.unsqueeze(0).expand_as(clip_valid)
    elif mode in {"zero", "clip_only"}:
        visibility = torch.zeros_like(visibility)

    batch = labels.numel()
    same_identity = labels[:, None].eq(labels[None, :])
    different_identity = ~same_identity
    eye = torch.eye(batch, dtype=torch.bool, device=labels.device)
    clip_similarity, clip_common = _pairwise_clip_similarity(
        clip_features, clip_valid
    )
    pose_distance = torch.cdist(visibility, visibility, p=1) / 5.0
    positive_candidates = same_identity & ~eye & clip_common
    negative_candidates = different_identity & clip_common

    positive_pose_rank = _ordinal_rank(pose_distance, positive_candidates)
    positive_clip_rank = _ordinal_rank(clip_similarity, positive_candidates)
    negative_pose_rank = _ordinal_rank(-pose_distance, negative_candidates)
    negative_clip_rank = _ordinal_rank(clip_similarity, negative_candidates)
    positive_index = _lexicographic_argmax(
        positive_pose_rank + positive_clip_rank,
        positive_clip_rank,
        positive_candidates,
    )
    negative_index = _lexicographic_argmax(
        negative_pose_rank + negative_clip_rank,
        negative_clip_rank,
        negative_candidates,
    )
    anchor = torch.arange(batch, device=labels.device)
    if not bool((labels[positive_index] == labels).all()):
        raise RuntimeError("PCHM selected a cross-identity positive")
    if bool((positive_index == anchor).any()):
        raise RuntimeError("PCHM selected a self positive")
    if not bool((labels[negative_index] != labels).all()):
        raise RuntimeError("PCHM selected a same-identity negative")
    return {
        "positive_indices": positive_index,
        "negative_indices": negative_index,
        "positive_pose_distance": pose_distance[anchor, positive_index],
        "positive_clip_similarity": clip_similarity[anchor, positive_index],
        "negative_pose_distance": pose_distance[anchor, negative_index],
        "negative_clip_similarity": clip_similarity[anchor, negative_index],
        "mode": mode,
    }


def batch_hard_pair_indices(global_feat, labels, normalize_feature=False):
    """Return the exact legacy batch-hard pair indices for diagnostics."""
    from .triplet_loss import euclidean_dist, hard_example_mining, normalize

    feature = (
        normalize(global_feat, axis=-1) if normalize_feature else global_feat
    )
    distance = euclidean_dist(feature, feature)
    _, _, positive, negative = hard_example_mining(
        distance, labels, return_inds=True
    )
    return positive, negative
