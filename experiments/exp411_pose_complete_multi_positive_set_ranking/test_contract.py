#!/usr/bin/env python3
"""Compact synthetic contract for exp411 PCMPSR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss.pose_clip_multi_positive_set import (
    build_pose_clip_identity_sets,
    pose_clip_identity_set_ranking_loss,
)
from loss.triplet_loss import TripletLoss


def main():
    torch.manual_seed(411)
    labels = torch.arange(16).repeat_interleave(4)
    visibility = torch.rand(64, 5)
    visibility[::4, 2:] *= 0.1
    base = F.normalize(torch.randn(16, 5, 32), dim=-1)
    clip = F.normalize(
        base.repeat_interleave(4, dim=0) + 0.08 * torch.randn(64, 5, 32),
        dim=-1,
    )
    valid = torch.ones(64, 5, dtype=torch.bool)
    valid[::7, 4] = False
    states = {
        mode: build_pose_clip_identity_sets(
            labels, visibility, clip, valid, mode=mode
        )
        for mode in ("correct", "wrong_rgb", "generic", "pose_only")
    }
    correct = states["correct"]
    if correct["support_indices"].shape != (64, 16, 3):
        raise RuntimeError("support shape contract failed")
    if correct["owner_indices"].shape != (64, 16, 5):
        raise RuntimeError("owner shape contract failed")
    owner_changes = {
        mode: float(
            states[mode]["owner_indices"]
            .ne(correct["owner_indices"])
            .float()
            .mean()
        )
        for mode in ("wrong_rgb", "generic", "pose_only")
    }
    if min(owner_changes.values()) <= 0.0:
        raise RuntimeError("one or more PCMPSR controls are inactive")
    if float(correct["owner_unique_mean"]) <= 1.0:
        raise RuntimeError("PCMPSR owners collapsed to one view")
    supports = correct["support_indices"]
    owners = correct["owner_indices"]
    for anchor in range(64):
        for identity in range(16):
            for slot in range(5):
                row = supports[anchor, identity]
                candidates = valid[row, slot] & (visibility[row, slot] > 0)
                if bool(candidates.any()):
                    selected = owners[anchor, identity, slot]
                    if not bool(valid[selected, slot]) or not bool(
                        visibility[selected, slot] > 0
                    ):
                        raise RuntimeError(
                            "PCMPSR selected a pose-invisible owner"
                        )

    feature = torch.randn(64, 48, requires_grad=True)
    set_loss, diagnostic = pose_clip_identity_set_ranking_loss(
        feature, labels, correct
    )
    set_loss.backward()
    if feature.grad is None or not bool(torch.isfinite(feature.grad).all()):
        raise RuntimeError("PCMPSR gradient is missing or non-finite")
    if float(feature.grad.abs().sum()) <= 0.0:
        raise RuntimeError("PCMPSR gradient is zero")

    probe = torch.randn(64, 48)
    rng_before = torch.get_rng_state().clone()
    legacy_a = TripletLoss()(probe, labels)[0]
    rng_after = torch.get_rng_state().clone()
    legacy_b = TripletLoss()(probe, labels, pair_indices=None)[0]
    if not torch.equal(rng_before, rng_after):
        raise RuntimeError("legacy triplet changed RNG")
    if not torch.equal(legacy_a, legacy_b):
        raise RuntimeError("default-off legacy triplet is not exact")

    print(
        json.dumps(
            {
                "schema": "exp411-pcmpsr-contract-v1",
                "support_shape": list(correct["support_indices"].shape),
                "owner_shape": list(correct["owner_indices"].shape),
                "owner_unique_mean": float(correct["owner_unique_mean"]),
                "owner_fallback_fraction": float(
                    correct["owner_fallback_fraction"]
                ),
                "control_owner_change": owner_changes,
                "listwise_loss": float(set_loss.detach()),
                "positive_distance": float(
                    diagnostic["positive_distance"].mean()
                ),
                "negative_distance": float(
                    diagnostic["negative_distance"].mean()
                ),
                "gradient_l1": float(feature.grad.abs().sum()),
                "default_triplet_exact": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
