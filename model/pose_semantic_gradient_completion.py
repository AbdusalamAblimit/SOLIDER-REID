"""Training-only pose-semantic gradient completion for exp412 PSGC."""

from __future__ import annotations

import hashlib
import re
import stat
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REGION_NAMES = (
    "head",
    "upper_torso_arms",
    "lower_torso",
    "upper_legs",
    "lower_legs_feet",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PoseSemanticTextAxes:
    """Strict loader for the frozen identity-free CLIP text axes."""

    def __init__(
        self,
        path,
        expected_sha256,
        expected_clip_checkpoint_sha256,
    ):
        configured = Path(path).expanduser()
        if not configured.is_absolute():
            raise ValueError("PSGC text-axis path must be absolute")
        resolved = configured.resolve(strict=True)
        if resolved != configured:
            raise RuntimeError("PSGC text-axis asset must use its canonical path")
        metadata = resolved.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("PSGC text-axis asset must be a unique regular file")
        if not expected_sha256:
            raise ValueError("PSGC text-axis SHA256 is required")
        self.sha256 = _sha256_file(resolved)
        if self.sha256 != expected_sha256:
            raise RuntimeError("PSGC text-axis SHA256 mismatch")

        with np.load(str(resolved), allow_pickle=False) as payload:
            required = {
                "schema",
                "region_names",
                "visible_prototypes",
                "occluded_prototypes",
                "clip_checkpoint_sha256",
                "prompt_spec_sha256",
                "builder_sha256",
                "source_head",
            }
            if set(payload.files) != required:
                raise RuntimeError("unexpected PSGC text-axis fields")
            schema = str(payload["schema"].item())
            region_names = tuple(str(value) for value in payload["region_names"])
            visible = payload["visible_prototypes"]
            occluded = payload["occluded_prototypes"]
            clip_checkpoint_sha256 = str(
                payload["clip_checkpoint_sha256"].item()
            )
            prompt_spec_sha256 = str(payload["prompt_spec_sha256"].item())
            builder_sha256 = str(payload["builder_sha256"].item())
            source_head = str(payload["source_head"].item())

        hex64 = re.compile(r"[0-9a-f]{64}")
        if schema != "exp412-psgc-text-axes-v1":
            raise RuntimeError("PSGC text-axis schema mismatch")
        if region_names != REGION_NAMES:
            raise RuntimeError("PSGC text-axis region order mismatch")
        if visible.shape != (5, 768) or occluded.shape != (5, 768):
            raise RuntimeError("PSGC text-axis shape mismatch")
        if visible.dtype != np.float32 or occluded.dtype != np.float32:
            raise RuntimeError("PSGC text axes must be float32")
        if clip_checkpoint_sha256 != expected_clip_checkpoint_sha256:
            raise RuntimeError("PSGC CLIP checkpoint provenance mismatch")
        if (
            hex64.fullmatch(prompt_spec_sha256) is None
            or hex64.fullmatch(builder_sha256) is None
            or re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        ):
            raise RuntimeError("PSGC text-axis source provenance is invalid")
        prototype = torch.stack(
            (torch.from_numpy(visible.copy()), torch.from_numpy(occluded.copy())),
            dim=1,
        )
        if not bool(torch.isfinite(prototype).all()):
            raise RuntimeError("PSGC text axes contain non-finite values")
        norms = prototype.norm(dim=-1)
        if not bool(
            torch.allclose(
                norms,
                torch.ones_like(norms),
                atol=1e-6,
                rtol=1e-6,
            )
        ):
            raise RuntimeError("PSGC text axes must be L2 normalized")
        self.path = resolved
        self.prototypes = prototype
        self.clip_checkpoint_sha256 = clip_checkpoint_sha256
        self.prompt_spec_sha256 = prompt_spec_sha256
        self.builder_sha256 = builder_sha256
        self.source_head = source_head

    def to(self, device):
        return self.prototypes.to(device=device, dtype=torch.float32)


def _ordered_identity_rows(labels):
    identities = []
    rows = []
    for value in labels.tolist():
        if value not in identities:
            identities.append(value)
    for value in identities:
        rows.append(torch.nonzero(labels == value, as_tuple=False).flatten())
    if not rows or {int(row.numel()) for row in rows} != {4}:
        raise RuntimeError("PSGC requires true PK groups with K=4")
    return rows


def _semantic_margin(clip_features, text_prototypes):
    feature = F.normalize(clip_features.float(), dim=-1)
    visible = text_prototypes[:, 0]
    occluded = text_prototypes[:, 1]
    return torch.einsum("brd,rd->br", feature, visible - occluded)


def build_psgc_slot_weights(
    labels,
    visibility,
    clip_features,
    clip_valid,
    text_prototypes,
    *,
    mode="correct",
):
    """Build deterministic, budget-conserving PID-by-slot Pareto weights."""
    if mode not in {"correct", "pose_only", "q_only", "text_shuffle"}:
        raise ValueError("unsupported PSGC control mode")
    if labels.ndim != 1 or visibility.shape != (labels.numel(), 5):
        raise ValueError("PSGC labels/visibility shape mismatch")
    if clip_features.shape != (labels.numel(), 5, 768):
        raise ValueError("PSGC CLIP feature shape mismatch")
    if clip_valid.shape != visibility.shape:
        raise ValueError("PSGC CLIP validity shape mismatch")
    if text_prototypes.shape != (5, 2, 768):
        raise ValueError("PSGC text prototype shape mismatch")
    if not bool(torch.isfinite(visibility.float()).all()):
        raise RuntimeError("PSGC visibility is non-finite")
    if not bool(torch.isfinite(clip_features.float()).all()):
        raise RuntimeError("PSGC CLIP feature is non-finite")
    if not bool(torch.isfinite(text_prototypes.float()).all()):
        raise RuntimeError("PSGC text prototype is non-finite")

    visibility = visibility.float()
    pareto_visibility = visibility
    candidate_valid = clip_valid.bool() & (visibility > 0)
    if mode == "pose_only":
        semantic = torch.zeros_like(visibility)
    else:
        prototype = (
            torch.roll(text_prototypes, shifts=-1, dims=0)
            if mode == "text_shuffle"
            else text_prototypes
        )
        semantic = _semantic_margin(clip_features, prototype)
        if mode == "q_only":
            pareto_visibility = torch.ones_like(visibility)

    weights = torch.zeros_like(visibility)
    front = torch.zeros_like(candidate_valid)
    fallback = torch.zeros_like(candidate_valid)
    front_sizes = []
    rows = _ordered_identity_rows(labels)
    for row in rows:
        for slot in range(5):
            valid = candidate_valid[row, slot]
            if not bool(valid.any()):
                weights[row, slot] = 1.0
                fallback[row, slot] = True
                front_sizes.append(torch.as_tensor(4.0, device=labels.device))
                continue
            pose_value = pareto_visibility[row, slot]
            semantic_value = semantic[row, slot]
            pose_ge = pose_value[:, None] >= pose_value[None, :]
            semantic_ge = semantic_value[:, None] >= semantic_value[None, :]
            strict = (pose_value[:, None] > pose_value[None, :]) | (
                semantic_value[:, None] > semantic_value[None, :]
            )
            dominates = (
                pose_ge
                & semantic_ge
                & strict
                & valid[:, None]
                & valid[None, :]
            )
            is_front = valid & ~dominates.any(dim=0)
            count = int(is_front.sum().item())
            if count <= 0:
                raise RuntimeError("PSGC Pareto front is empty")
            front[row, slot] = is_front
            weights[row[is_front], slot] = 4.0 / float(count)
            front_sizes.append(
                torch.as_tensor(float(count), device=labels.device)
            )

    for row in rows:
        budget = weights[row].sum(dim=0)
        if not bool(
            torch.allclose(
                budget,
                torch.full_like(budget, 4.0),
                atol=1e-6,
                rtol=0.0,
            )
        ):
            raise RuntimeError("PSGC per-PID slot budget drift")
    active_semantic = semantic[candidate_valid]
    semantic_mean = (
        active_semantic.mean()
        if active_semantic.numel()
        else torch.zeros((), device=labels.device)
    )
    semantic_std = (
        active_semantic.std(unbiased=False)
        if active_semantic.numel()
        else torch.zeros((), device=labels.device)
    )
    return weights, {
        "front_mask": front,
        "fallback_mask": fallback,
        "front_size_mean": torch.stack(front_sizes).mean(),
        "front_fraction": front.float().mean(),
        "fallback_fraction": fallback.float().mean(),
        "weight_min": weights.min(),
        "weight_max": weights.max(),
        "semantic_mean": semantic_mean,
        "semantic_std": semantic_std,
    }


def route_pose_semantic_gradient(feature, pose_batch, *, image_hw):
    """Keep the feature forward-exact while scaling anatomical gradients."""
    if feature.ndim != 4:
        raise ValueError("PSGC feature must have shape [B,C,H,W]")
    slot_weights = pose_batch.get("psgc_slot_weights")
    if slot_weights is None:
        return feature, None
    if slot_weights.shape != (feature.shape[0], 5):
        raise ValueError("PSGC slot weight shape mismatch")
    from model.pose_clip_relation import render_pose_indexed_regions

    with torch.no_grad():
        masks, region_valid = render_pose_indexed_regions(
            pose_batch["keypoints"],
            pose_batch["valid"],
            image_hw=image_hw,
            field_hw=feature.shape[-2:],
            sigma=1.5,
        )
        body_mass = masks.float().sum(dim=1)
        gradient_field = 1.0 - body_mass
        gradient_field = gradient_field + (
            masks.float() * slot_weights.float()[..., None, None]
        ).sum(dim=1)
        if not bool(torch.isfinite(gradient_field).all()):
            raise RuntimeError("PSGC gradient field is non-finite")
        if bool((gradient_field < 0).any()):
            raise RuntimeError("PSGC gradient field is negative")
    detached = feature.detach()
    gradient_cast = gradient_field.detach().to(
        device=feature.device, dtype=feature.dtype
    )
    routed = detached + gradient_cast[:, None] * (feature - detached)
    return routed, {
        "gradient_min": gradient_field.min(),
        "gradient_max": gradient_field.max(),
        "gradient_mean": gradient_field.mean(),
        "body_fraction": (body_mass > 0).float().mean(),
        "region_valid_fraction": region_valid.float().mean(),
    }
