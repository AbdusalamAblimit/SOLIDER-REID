"""Pose-complete multi-positive identity-set ranking for exp411 PCMPSR."""

from __future__ import annotations

import hashlib
import math
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


class PoseClipSetCache:
    """Strict loader for one fresh exp411 region-isolated CLIP cache."""

    def __init__(
        self,
        path,
        expected_sha256,
        expected_clip_checkpoint_sha256,
        expected_pose_manifest_sha256,
    ):
        configured = Path(path).expanduser()
        if not configured.is_absolute():
            raise ValueError("PCMPSR cache path must be absolute")
        resolved = configured.resolve(strict=True)
        if resolved != configured:
            raise RuntimeError("PCMPSR cache must use its canonical path")
        if not expected_sha256:
            raise ValueError("PCMPSR cache SHA256 is required")
        self.sha256 = _sha256_file(resolved)
        if self.sha256 != expected_sha256:
            raise RuntimeError("PCMPSR cache SHA256 mismatch")
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
                raise RuntimeError("unexpected PCMPSR cache fields")
            schema = str(payload["schema"].item())
            path_array = payload["relative_paths"]
            image_sha256 = payload["image_sha256"]
            feature_array = payload["features"]
            valid_array = payload["valid"]
            preprocessing = str(payload["preprocessing"].item())
            pose_manifest_sha256 = str(
                payload["pose_manifest_sha256"].item()
            )
            clip_checkpoint_sha256 = str(
                payload["clip_checkpoint_sha256"].item()
            )
            source_head = str(payload["source_head"].item())
            builder_sha256 = str(payload["builder_sha256"].item())
            teacher_source_sha256 = str(
                payload["teacher_source_sha256"].item()
            )
            if path_array.ndim != 1 or path_array.dtype.kind != "U":
                raise RuntimeError("PCMPSR paths must be a unicode vector")
            if feature_array.shape != (len(path_array), 5, 768):
                raise RuntimeError("PCMPSR feature shape mismatch")
            if feature_array.dtype != np.float16:
                raise RuntimeError("PCMPSR features must be float16")
            if valid_array.shape != (len(path_array), 5):
                raise RuntimeError("PCMPSR validity shape mismatch")
            if valid_array.dtype != np.bool_:
                raise RuntimeError("PCMPSR validity must be boolean")
            if (
                image_sha256.shape != path_array.shape
                or image_sha256.dtype.kind != "U"
            ):
                raise RuntimeError("PCMPSR image SHA vector mismatch")
            hex64 = re.compile(r"[0-9a-f]{64}")
            if any(
                hex64.fullmatch(str(value)) is None
                for value in image_sha256
            ):
                raise RuntimeError("PCMPSR image SHA vector is invalid")
            if preprocessing != "raw-rgb-pose-resize-384x128-no-augmentation":
                raise RuntimeError("PCMPSR preprocessing provenance mismatch")
            if pose_manifest_sha256 != expected_pose_manifest_sha256:
                raise RuntimeError("PCMPSR pose manifest provenance mismatch")
            if clip_checkpoint_sha256 != expected_clip_checkpoint_sha256:
                raise RuntimeError("PCMPSR CLIP checkpoint provenance mismatch")
            if re.fullmatch(r"[0-9a-f]{40}", source_head) is None:
                raise RuntimeError("PCMPSR source HEAD provenance is invalid")
            if hex64.fullmatch(builder_sha256) is None or hex64.fullmatch(
                teacher_source_sha256
            ) is None:
                raise RuntimeError("PCMPSR source SHA provenance is invalid")
            paths = tuple(str(item) for item in path_array.tolist())
            features = torch.from_numpy(feature_array.copy())
            valid = torch.from_numpy(valid_array.copy())
        if schema != "exp411-pcmpsr-cache-v1":
            raise RuntimeError("PCMPSR cache schema mismatch")
        if not paths or len(paths) != len(set(paths)):
            raise RuntimeError("PCMPSR cache paths must be nonempty and unique")
        if valid.shape != features.shape[:2] or len(paths) != features.shape[0]:
            raise RuntimeError("PCMPSR cache arrays disagree")
        if not bool(torch.isfinite(features.float()).all()):
            raise RuntimeError("PCMPSR cache contains non-finite features")
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
            raise RuntimeError("PCMPSR valid features must be L2 normalized")
        self.path = resolved
        self.paths = paths
        self._features = features
        self._valid = valid
        self._index = {path: index for index, path in enumerate(paths)}
        self.image_sha256 = tuple(
            str(value) for value in image_sha256.tolist()
        )
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
            raise RuntimeError("PCMPSR batch path is absent from cache") from error
        if image_sha256 is None or len(image_sha256) != len(indices):
            raise RuntimeError("PCMPSR batch image SHA binding is required")
        expected = [self.image_sha256[index] for index in indices]
        if tuple(str(value) for value in image_sha256) != tuple(expected):
            raise RuntimeError("PCMPSR batch image SHA binding mismatch")
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


def _ordered_identity_rows(labels):
    identities = []
    rows = []
    for label in labels.tolist():
        if label not in identities:
            identities.append(label)
    for label in identities:
        index = torch.nonzero(labels == label, as_tuple=False).flatten()
        rows.append(index)
    counts = {int(index.numel()) for index in rows}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise RuntimeError("PCMPSR requires equal multi-image identity groups")
    return torch.as_tensor(identities, device=labels.device), rows


def _slot_collapsed_features(features, valid):
    weight = valid.float()
    collapsed = (features * weight[..., None]).sum(dim=1)
    collapsed = collapsed / weight.sum(dim=1).clamp_min(1.0)[..., None]
    collapsed = F.normalize(collapsed, dim=-1)
    available = valid.any(dim=1)
    return collapsed[:, None, :].expand_as(features), available[:, None].expand_as(valid)


def build_pose_clip_identity_sets(
    labels,
    visibility,
    clip_features,
    clip_valid,
    *,
    mode="correct",
    wrong_shift=4,
):
    """Build deterministic leave-one-position-out supports and slot owners."""
    if labels.ndim != 1 or visibility.shape != (labels.numel(), 5):
        raise ValueError("PCMPSR labels/visibility shape mismatch")
    if clip_features.shape[:2] != visibility.shape:
        raise ValueError("PCMPSR CLIP feature shape mismatch")
    if clip_valid.shape != visibility.shape:
        raise ValueError("PCMPSR CLIP validity shape mismatch")
    if mode not in {"correct", "wrong_rgb", "generic", "pose_only"}:
        raise ValueError("unsupported PCMPSR control mode")
    visibility = visibility.float()
    clip_features = F.normalize(clip_features.float(), dim=-1)
    clip_valid = clip_valid.bool()
    if mode == "wrong_rgb":
        shift = int(wrong_shift)
        if shift <= 0 or shift >= labels.numel():
            raise ValueError("PCMPSR wrong-RGB shift is outside the batch")
        shifted_labels = torch.roll(labels, shifts=-shift, dims=0)
        if not bool((shifted_labels != labels).all()):
            raise RuntimeError("PCMPSR wrong-RGB shift is not different-PID")
        clip_features = torch.roll(clip_features, shifts=-shift, dims=0)
        clip_valid = torch.roll(clip_valid, shifts=-shift, dims=0)
    elif mode == "generic":
        clip_features, clip_valid = _slot_collapsed_features(
            clip_features, clip_valid
        )

    class_labels, class_rows = _ordered_identity_rows(labels)
    batch = labels.numel()
    classes = class_labels.numel()
    instances = class_rows[0].numel()
    support_count = instances - 1
    support_indices = torch.empty(
        batch,
        classes,
        support_count,
        dtype=torch.long,
        device=labels.device,
    )
    owner_indices = torch.empty(
        batch, classes, 5, dtype=torch.long, device=labels.device
    )
    positive_class = torch.empty(
        batch, dtype=torch.long, device=labels.device
    )

    occurrence = torch.empty(batch, dtype=torch.long, device=labels.device)
    for class_index, row in enumerate(class_rows):
        occurrence[row] = torch.arange(instances, device=labels.device)
        positive_class[row] = class_index

    for anchor in range(batch):
        position = int(occurrence[anchor].item())
        for class_index, row in enumerate(class_rows):
            support = torch.cat((row[:position], row[position + 1 :]))
            if support.numel() != support_count:
                raise RuntimeError("PCMPSR support cardinality drift")
            support_indices[anchor, class_index] = support
            support_features = clip_features[support]
            support_valid = clip_valid[support]
            support_visibility = visibility[support]
            for slot in range(5):
                valid_slot = support_valid[:, slot]
                if mode == "pose_only" or not bool(valid_slot.any()):
                    score = support_visibility[:, slot]
                else:
                    feature = support_features[:, slot]
                    consensus = F.normalize(
                        feature[valid_slot].mean(dim=0), dim=0
                    )
                    similarity = torch.mv(feature, consensus)
                    score = support_visibility[:, slot] * similarity
                    score = score.masked_fill(~valid_slot, float("-inf"))
                best = torch.nonzero(
                    score == score.max(), as_tuple=False
                ).flatten()
                if best.numel() == 0:
                    raise RuntimeError("PCMPSR owner selection is empty")
                owner_indices[anchor, class_index, slot] = support[
                    best.min()
                ]

    expanded_labels = labels[owner_indices]
    if not bool(
        expanded_labels.eq(class_labels[None, :, None]).all()
    ):
        raise RuntimeError("PCMPSR owner crossed identity support")
    if not bool(
        labels[support_indices].eq(class_labels[None, :, None]).all()
    ):
        raise RuntimeError("PCMPSR support crossed identity")
    anchors = torch.arange(batch, device=labels.device)
    positive_support = support_indices[anchors, positive_class]
    if bool(positive_support.eq(anchors[:, None]).any()):
        raise RuntimeError("PCMPSR positive support contains anchor self")
    unique_counts = []
    for anchor in range(batch):
        for class_index in range(classes):
            unique_counts.append(
                owner_indices[anchor, class_index].unique().numel()
            )
    owner_unique_mean = torch.as_tensor(
        unique_counts, device=labels.device, dtype=torch.float32
    ).mean()
    return {
        "support_indices": support_indices,
        "owner_indices": owner_indices,
        "class_labels": class_labels,
        "positive_class_indices": positive_class,
        "owner_unique_mean": owner_unique_mean,
        "mode": mode,
    }


def pose_clip_identity_set_ranking_loss(
    global_feat,
    labels,
    set_state,
    *,
    normalize_feature=False,
):
    """Temperature-free all-identity set ranking in the student space."""
    from .triplet_loss import euclidean_dist, normalize

    if global_feat.ndim != 2 or global_feat.shape[0] != labels.numel():
        raise ValueError("PCMPSR feature/label shape mismatch")
    feature = global_feat.float()
    if normalize_feature:
        feature = normalize(feature, axis=-1)
    distance = euclidean_dist(feature, feature)
    support = set_state["support_indices"].to(labels.device)
    owners = set_state["owner_indices"].to(labels.device)
    class_labels = set_state["class_labels"].to(labels.device)
    positive_class = set_state["positive_class_indices"].to(labels.device)
    if support.shape[:2] != owners.shape[:2]:
        raise ValueError("PCMPSR support/owner class shape mismatch")
    batch, classes, support_count = support.shape
    support_distance = distance.gather(1, support.reshape(batch, -1)).view(
        batch, classes, support_count
    )
    owner_distance = distance.gather(1, owners.reshape(batch, -1)).view(
        batch, classes, owners.shape[-1]
    )
    set_distance = (
        support_distance.sum(dim=-1) + owner_distance.sum(dim=-1)
    ) / float(support_count + owners.shape[-1])
    anchor = torch.arange(batch, device=labels.device)
    positive_distance = set_distance[anchor, positive_class]
    negative_mask = class_labels[None, :].ne(labels[:, None])
    if not bool((negative_mask.sum(dim=1) == classes - 1).all()):
        raise RuntimeError("PCMPSR negative identity cardinality drift")
    negative_distance = set_distance[negative_mask].view(batch, classes - 1)
    delta = positive_distance[:, None] - negative_distance
    log_mean_exp = torch.logsumexp(delta, dim=1) - math.log(classes - 1)
    loss = F.softplus(log_mean_exp).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("PCMPSR listwise loss is non-finite")
    return loss, {
        "positive_distance": positive_distance.detach(),
        "negative_distance": negative_distance.detach(),
        "set_distance": set_distance.detach(),
        "loss": loss.detach(),
    }
