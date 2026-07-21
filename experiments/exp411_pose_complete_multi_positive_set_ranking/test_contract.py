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
    build_pose_clip_training_state,
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
    formal_states = {
        mode: build_pose_clip_training_state(
            labels,
            visibility,
            clip,
            valid,
            control_mode=mode,
        )
        for mode in ("correct", "zero_owner", "wrong_rgb")
    }
    for mode, state in formal_states.items():
        if not torch.equal(
            state["support_indices"], correct["support_indices"]
        ):
            raise RuntimeError(
                "{} changed the frozen support set".format(mode)
            )
    if not torch.equal(
        formal_states["wrong_rgb"]["owner_indices"],
        states["wrong_rgb"]["owner_indices"],
    ):
        raise RuntimeError("formal wrong-RGB is not the direct wrong owner")
    if torch.equal(
        formal_states["wrong_rgb"]["owner_indices"],
        correct["owner_indices"],
    ):
        raise RuntimeError("formal wrong-RGB owner collapsed to correct")
    for mode in ("correct", "zero_owner"):
        if not torch.equal(
            formal_states[mode]["owner_indices"],
            correct["owner_indices"],
        ):
            raise RuntimeError("{} changed correct owners".format(mode))
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

    feature_value = torch.randn(64, 48)
    legacy_loss, legacy_diag = pose_clip_identity_set_ranking_loss(
        feature_value, labels, correct
    )
    explicit_loss, explicit_diag = pose_clip_identity_set_ranking_loss(
        feature_value, labels, formal_states["correct"]
    )
    if not torch.equal(legacy_loss, explicit_loss) or not torch.equal(
        legacy_diag["set_distance"], explicit_diag["set_distance"]
    ):
        raise RuntimeError("explicit correct mode changed frozen behavior")
    zero_loss, zero_diag = pose_clip_identity_set_ranking_loss(
        feature_value, labels, formal_states["zero_owner"]
    )
    from loss.triplet_loss import euclidean_dist

    distance = euclidean_dist(feature_value.float(), feature_value.float())
    support = correct["support_indices"]
    manual_zero_distance = distance.gather(
        1, support.reshape(64, -1)
    ).view(64, 16, 3).mean(dim=-1)
    if not torch.equal(zero_diag["set_distance"], manual_zero_distance):
        raise RuntimeError("zero-owner is not the exact support-only mean")
    if zero_diag["owner_term_count"] != 0:
        raise RuntimeError("zero-owner retained owner multiplicity")

    formal_losses = {}
    gradient_l1 = {}
    for mode, state in formal_states.items():
        feature = feature_value.clone().requires_grad_(True)
        set_loss, diagnostic = pose_clip_identity_set_ranking_loss(
            feature, labels, state
        )
        set_loss.backward()
        if feature.grad is None or not bool(torch.isfinite(feature.grad).all()):
            raise RuntimeError(
                "{} gradient is missing or non-finite".format(mode)
            )
        if float(feature.grad.abs().sum()) <= 0.0:
            raise RuntimeError("{} gradient is zero".format(mode))
        formal_losses[mode] = float(set_loss.detach())
        gradient_l1[mode] = float(feature.grad.abs().sum())
        expected_terms = 0 if mode == "zero_owner" else 5
        if diagnostic["owner_term_count"] != expected_terms:
            raise RuntimeError(
                "{} owner term count is not {}".format(mode, expected_terms)
            )

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
                "formal_listwise_loss": formal_losses,
                "positive_distance": float(
                    explicit_diag["positive_distance"].mean()
                ),
                "negative_distance": float(
                    explicit_diag["negative_distance"].mean()
                ),
                "formal_gradient_l1": gradient_l1,
                "correct_default_exact": True,
                "zero_owner_manual_distance_exact": True,
                "default_triplet_exact": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
