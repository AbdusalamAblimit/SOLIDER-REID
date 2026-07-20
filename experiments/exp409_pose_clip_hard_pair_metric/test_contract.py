#!/usr/bin/env python3
"""Compact positive/negative contract for exp409 PCHM."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss.pose_clip_hard_mining import (
    PoseClipMiningCache,
    batch_hard_pair_indices,
    pose_visibility_signature,
    select_pose_clip_pairs,
)
from loss.triplet_loss import TripletLoss


def synthetic_batch(device):
    torch.manual_seed(409)
    labels = torch.arange(16, device=device).repeat_interleave(4)
    scores = 0.55 + 0.45 * torch.rand(64, 17, device=device)
    valid = torch.rand(64, 17, device=device) > 0.18
    visibility = pose_visibility_signature(scores, valid)
    identity = F.normalize(torch.randn(16, 5, 32, device=device), dim=-1)
    appearance = identity.repeat_interleave(4, dim=0)
    appearance = F.normalize(
        appearance + 0.08 * torch.randn_like(appearance), dim=-1
    )
    clip_valid = torch.rand(64, 5, device=device) > 0.03
    return labels, visibility, appearance, clip_valid


def cache_roundtrip_contract():
    with tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent)) as root:
        path = Path(root) / "synthetic_cache.npz"
        relative_paths = np.asarray(
            ["train/sample_{:02d}.jpg".format(index) for index in range(8)]
        )
        image_sha256 = np.asarray(
            [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in relative_paths]
        )
        generator = torch.Generator().manual_seed(411)
        features = F.normalize(
            torch.randn(8, 5, 768, generator=generator), dim=-1
        ).half().numpy()
        valid = np.ones((8, 5), dtype=np.bool_)
        with path.open("wb") as handle:
            np.savez(
                handle,
                schema=np.asarray("exp409-pchm-cache-v1"),
                relative_paths=relative_paths,
                image_sha256=image_sha256,
                features=features,
                valid=valid,
                preprocessing=np.asarray(
                    "raw-rgb-pose-resize-384x128-no-augmentation"
                ),
                pose_manifest_sha256=np.asarray("1" * 64),
                clip_checkpoint_sha256=np.asarray("2" * 64),
                source_head=np.asarray("3" * 40),
                builder_sha256=np.asarray("4" * 64),
                teacher_source_sha256=np.asarray("5" * 64),
            )
        cache_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        cache = PoseClipMiningCache(
            path,
            cache_sha256,
            "2" * 64,
            "1" * 64,
        )
        cache.lookup(relative_paths[:2], image_sha256[:2])
        try:
            cache.lookup(relative_paths[:2], ("0" * 64, image_sha256[1]))
        except RuntimeError:
            mismatch_rejected = True
        else:
            mismatch_rejected = False
        if not mismatch_rejected:
            raise RuntimeError("cache image SHA mismatch mutant was not rejected")
    return True


def main():
    cache_roundtrip_contract()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, visibility, appearance, clip_valid = synthetic_batch(device)
    states = {
        mode: select_pose_clip_pairs(
            labels,
            visibility,
            appearance,
            clip_valid,
            mode=mode,
        )
        for mode in (
            "correct",
            "wrong_rgb",
            "generic",
            "zero",
            "pose_shuffle",
            "clip_only",
        )
    }
    anchor = torch.arange(64, device=device)
    correct = states["correct"]
    assert bool((labels[correct["positive_indices"]] == labels).all())
    assert bool((correct["positive_indices"] != anchor).all())
    assert bool((labels[correct["negative_indices"]] != labels).all())
    changes = {}
    for mode, state in states.items():
        if mode == "correct":
            continue
        changed = (
            state["positive_indices"] != correct["positive_indices"]
        ) | (state["negative_indices"] != correct["negative_indices"])
        changes[mode] = float(changed.float().mean().item())
        assert changes[mode] > 0.0

    torch.manual_seed(410)
    feature_default = torch.randn(64, 48, device=device, requires_grad=True)
    feature_external = feature_default.detach().clone().requires_grad_(True)
    legacy_positive, legacy_negative = batch_hard_pair_indices(
        feature_external.detach(), labels
    )
    triplet = TripletLoss()
    default_loss = triplet(feature_default, labels)[0]
    external_loss = triplet(
        feature_external,
        labels,
        pair_indices=(legacy_positive, legacy_negative),
    )[0]
    if not torch.equal(default_loss.detach(), external_loss.detach()):
        raise RuntimeError("external legacy indices changed the default loss")
    default_loss.backward()
    external_loss.backward()
    if not torch.equal(feature_default.grad, feature_external.grad):
        raise RuntimeError("external legacy indices changed the default gradient")
    if not bool(torch.isfinite(feature_external.grad).all()) or not bool(
        (feature_external.grad != 0).any()
    ):
        raise RuntimeError("PCHM triplet gradient is inactive or non-finite")

    try:
        triplet(
            feature_external.detach(),
            labels,
            pair_indices=(anchor, correct["negative_indices"]),
        )
    except RuntimeError:
        invalid_positive_rejected = True
    else:
        invalid_positive_rejected = False
    if not invalid_positive_rejected:
        raise RuntimeError("self-positive mutant was not rejected")

    result = {
        "schema": "exp409-pchm-contract-v1",
        "device": str(device),
        "batch": 64,
        "control_index_change": changes,
        "default_loss_exact": True,
        "default_gradient_exact": True,
        "invalid_positive_rejected": True,
        "cache_provenance_roundtrip": True,
        "status": "PASS",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
