#!/usr/bin/env python3
"""Pure contracts for the sealed exp404 SPK counterfactual audit."""

from __future__ import annotations

import hashlib
import json

import torch


ARM_ORDER = (
    "correct",
    "wrong_rgb",
    "generic_mean",
    "null_zero",
    "all_product_bypass",
    "random_key",
    "random_cluster",
    "wrong_mask",
    "slot_cycle",
)
PRIMARY_NULL_CONTROLS = (
    "wrong_rgb",
    "generic_mean",
    "null_zero",
    "random_key",
    "random_cluster",
)
PRIMARY_ACTIVE_CONTROLS = (
    "wrong_rgb",
    "generic_mean",
    "null_zero",
    "all_product_bypass",
    "random_key",
    "random_cluster",
)
SUPPLEMENTAL_ATTRIBUTION_CONTROLS = (
    "wrong_mask",
    "slot_cycle",
)
INTERVENTION_ARMS = frozenset(set(ARM_ORDER) - {"correct", "all_product_bypass"})
GROUPS = 16
SLOTS = 5
CLUSTERS = 8
SEED = 1234


def canonical_json_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_mapping_sha256(mapping) -> str:
    digest = hashlib.sha256()
    for name, value in mapping.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor mapping entry: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_global_donor_map(pids, camids, num_query: int) -> torch.Tensor:
    """Build a same-split/same-camera/different-PID donor for every row."""
    pids = torch.as_tensor(pids, dtype=torch.int64).flatten().cpu()
    camids = torch.as_tensor(camids, dtype=torch.int64).flatten().cpu()
    if pids.shape != camids.shape or pids.numel() <= 1:
        raise ValueError("PID/camera vectors must have equal nontrivial shape")
    if not 0 < int(num_query) < pids.numel():
        raise ValueError("num_query must split validation records")

    donor = torch.full_like(pids, -1)
    split = torch.arange(pids.numel()) >= int(num_query)
    keys = sorted(
        {
            (bool(split[index]), int(camids[index]))
            for index in range(pids.numel())
        }
    )
    for split_value, camera in keys:
        group = torch.nonzero(
            (split == split_value) & (camids == camera),
            as_tuple=False,
        ).flatten()
        if group.numel() <= 1:
            raise RuntimeError("Donor group has fewer than two rows")
        for position, recipient in enumerate(group.tolist()):
            for offset in range(1, group.numel()):
                candidate = int(group[(position + offset) % group.numel()])
                if int(pids[candidate]) != int(pids[recipient]):
                    donor[recipient] = candidate
                    break
    if bool((donor < 0).any()):
        raise RuntimeError("Could not build a different-PID donor map")
    validate_donor_map(donor, pids, camids, num_query)
    return donor


def validate_donor_map(donor, pids, camids, num_query: int):
    donor = torch.as_tensor(donor, dtype=torch.int64).flatten().cpu()
    pids = torch.as_tensor(pids, dtype=torch.int64).flatten().cpu()
    camids = torch.as_tensor(camids, dtype=torch.int64).flatten().cpu()
    count = pids.numel()
    if donor.shape != pids.shape or camids.shape != pids.shape:
        raise ValueError("Donor invariant shape mismatch")
    if bool(((donor < 0) | (donor >= count)).any()):
        raise RuntimeError("Donor index out of range")
    indices = torch.arange(count)
    summary = {
        "count": int(count),
        "no_fixed_points": bool((donor != indices).all()),
        "different_pid_fraction": float((pids[donor] != pids).float().mean()),
        "same_camera_fraction": float((camids[donor] == camids).float().mean()),
        "same_split_fraction": float(
            ((donor >= int(num_query)) == (indices >= int(num_query)))
            .float()
            .mean()
        ),
    }
    if not summary["no_fixed_points"]:
        raise RuntimeError("Donor map contains a fixed point")
    for key in (
        "different_pid_fraction",
        "same_camera_fraction",
        "same_split_fraction",
    ):
        if summary[key] != 1.0:
            raise RuntimeError(f"Donor invariant failed: {key}")
    return summary


def pooled_evidence(evidence, presence) -> torch.Tensor:
    evidence = torch.as_tensor(evidence)
    presence = torch.as_tensor(presence)
    if evidence.ndim != 3 or evidence.shape[1:] != (SLOTS, GROUPS):
        raise ValueError("Evidence must have shape [B,5,16]")
    if presence.shape != evidence.shape[:2]:
        raise ValueError("Presence must have shape [B,5]")
    weight = presence.detach().float().clamp(0.0, 1.0)
    mass = weight.sum(dim=1, keepdim=True)
    pooled = (evidence.float() * weight[..., None]).sum(dim=1)
    pooled = pooled / mass.clamp_min(1.0)
    return torch.where(mass > 0, pooled, torch.zeros_like(pooled))


def _key_digest(index: int, seed: int, namespace: str) -> bytes:
    message = f"exp404-spk-v1:{namespace}:{int(seed)}:{int(index)}".encode(
        "ascii"
    )
    return hashlib.sha256(message).digest()


def signed_permutation(index: int, seed: int = SEED, namespace="sample"):
    digest = _key_digest(index, seed, namespace)
    order = sorted(range(GROUPS), key=lambda item: (digest[item], item))
    signs = [1 if digest[GROUPS + item] & 1 else -1 for item in range(GROUPS)]
    return tuple(order), tuple(signs)


def build_signed_permutations(count: int, seed: int = SEED, namespace="sample"):
    if int(count) <= 0:
        raise ValueError("Signed-permutation count must be positive")
    permutations = []
    signs = []
    signatures = set()
    for index in range(int(count)):
        permutation, sign = signed_permutation(index, seed, namespace)
        signature = permutation + sign
        if signature in signatures:
            raise RuntimeError("Hash-derived signed permutation collision")
        signatures.add(signature)
        permutations.append(permutation)
        signs.append(sign)
    return (
        torch.tensor(permutations, dtype=torch.int64),
        torch.tensor(signs, dtype=torch.int8),
    )


def apply_signed_permutations(evidence, permutations, signs):
    evidence = torch.as_tensor(evidence)
    permutations = torch.as_tensor(permutations, dtype=torch.int64)
    signs = torch.as_tensor(signs)
    if evidence.ndim != 3 or evidence.shape[-1] != GROUPS:
        raise ValueError("Evidence must have shape [B,R,16]")
    if permutations.shape != (evidence.shape[0], GROUPS):
        raise ValueError("Permutation batch shape mismatch")
    if signs.shape != permutations.shape:
        raise ValueError("Sign batch shape mismatch")
    gather_index = permutations.to(evidence.device)[:, None, :].expand(
        evidence.shape[0], evidence.shape[1], GROUPS
    )
    permuted = evidence.gather(-1, gather_index)
    return permuted * signs.to(device=evidence.device, dtype=evidence.dtype)[:, None, :]


def build_balanced_cluster_assignment(
    count: int,
    clusters: int = CLUSTERS,
    seed: int = SEED,
):
    if int(count) < int(clusters) or int(clusters) <= 1:
        raise ValueError("Balanced clustering requires count >= clusters > 1")
    ordered = sorted(
        range(int(count)),
        key=lambda index: _key_digest(index, seed, "cluster-assignment"),
    )
    assignment = torch.empty(int(count), dtype=torch.int64)
    for rank, index in enumerate(ordered):
        assignment[index] = rank % int(clusters)
    counts = torch.bincount(assignment, minlength=int(clusters))
    if int(counts.max() - counts.min()) > 1:
        raise RuntimeError("Random-cluster assignment is not balanced")
    return assignment


def build_cluster_prototypes(generic_mean, clusters: int = CLUSTERS):
    generic = torch.as_tensor(generic_mean).detach().float().flatten().cpu()
    if generic.shape != (GROUPS,) or not bool(torch.isfinite(generic).all()):
        raise ValueError("Generic mean must be one finite 16-D vector")
    rows = []
    signatures = set()
    for cluster in range(int(clusters)):
        permutation, signs = signed_permutation(
            cluster,
            SEED,
            "cluster-prototype",
        )
        signature = permutation + signs
        if signature in signatures:
            raise RuntimeError("Random-cluster prototype collision")
        signatures.add(signature)
        row = generic[list(permutation)] * torch.tensor(signs, dtype=generic.dtype)
        rows.append(row)
    prototypes = torch.stack(rows, dim=0)
    if not bool(torch.isfinite(prototypes).all()):
        raise RuntimeError("Non-finite random-cluster prototype")
    return prototypes


def validate_cluster_assignment(assignment, pids, camids, clusters=CLUSTERS):
    assignment = torch.as_tensor(assignment, dtype=torch.int64).flatten().cpu()
    pids = torch.as_tensor(pids, dtype=torch.int64).flatten().cpu()
    camids = torch.as_tensor(camids, dtype=torch.int64).flatten().cpu()
    if assignment.shape != pids.shape or camids.shape != pids.shape:
        raise ValueError("Cluster coverage shape mismatch")
    if bool(((assignment < 0) | (assignment >= int(clusters))).any()):
        raise RuntimeError("Cluster assignment out of range")
    expected_cameras = sorted(set(int(value) for value in camids.tolist()))
    counts = torch.bincount(assignment, minlength=int(clusters))
    pid_coverage = []
    camera_coverage = []
    camera_sets = []
    for cluster in range(int(clusters)):
        selected = assignment == cluster
        pid_coverage.append(len(set(int(value) for value in pids[selected].tolist())))
        cameras = sorted(set(int(value) for value in camids[selected].tolist()))
        camera_sets.append(cameras)
        camera_coverage.append(len(cameras))
    summary = {
        "clusters": int(clusters),
        "counts": [int(value) for value in counts.tolist()],
        "count_max_minus_min": int(counts.max() - counts.min()),
        "pid_coverage": pid_coverage,
        "camera_coverage": camera_coverage,
        "camera_sets": camera_sets,
        "expected_cameras": expected_cameras,
        "pid_coverage_min": min(pid_coverage),
        "all_cameras_exact": all(
            cameras == expected_cameras for cameras in camera_sets
        ),
    }
    if summary["count_max_minus_min"] > 1:
        raise RuntimeError("Random-cluster counts are not balanced")
    if summary["pid_coverage_min"] < 40:
        raise RuntimeError("Random-cluster PID coverage is below 40")
    if not summary["all_cameras_exact"]:
        raise RuntimeError("Random-cluster camera coverage is incomplete")
    return summary


def intervene_spk_inputs(
    evidence,
    presence,
    arm: str,
    absolute_indices=None,
    donor_map=None,
    evidence_cache=None,
    presence_cache=None,
    generic_mean=None,
    random_permutations=None,
    random_signs=None,
    cluster_assignment=None,
    cluster_prototypes=None,
):
    if arm not in INTERVENTION_ARMS:
        raise ValueError(f"Not an SPK input intervention arm: {arm}")
    evidence = torch.as_tensor(evidence)
    presence = torch.as_tensor(presence)
    if evidence.ndim != 3 or evidence.shape[1:] != (SLOTS, GROUPS):
        raise ValueError("SPK evidence must have shape [B,5,16]")
    if presence.shape != evidence.shape[:2]:
        raise ValueError("SPK presence must have shape [B,5]")
    indices = torch.as_tensor(absolute_indices, dtype=torch.int64).flatten().cpu()
    if indices.numel() != evidence.shape[0]:
        raise ValueError("Absolute-index batch size mismatch")

    if arm == "wrong_rgb":
        donor = torch.as_tensor(donor_map, dtype=torch.int64).flatten().cpu()
        selected = donor.index_select(0, indices)
        cached_evidence = torch.as_tensor(evidence_cache).index_select(0, selected)
        cached_presence = torch.as_tensor(presence_cache).index_select(0, selected)
        return (
            cached_evidence.to(device=evidence.device, dtype=evidence.dtype),
            cached_presence.to(device=presence.device, dtype=presence.dtype),
        )
    if arm == "generic_mean":
        generic = torch.as_tensor(generic_mean).reshape(1, 1, GROUPS)
        replacement = generic.expand(evidence.shape[0], 1, GROUPS)
        replacement_presence = torch.ones(
            evidence.shape[0],
            1,
            device=presence.device,
            dtype=presence.dtype,
        )
        return (
            replacement.to(device=evidence.device, dtype=evidence.dtype),
            replacement_presence,
        )
    if arm == "null_zero":
        return torch.zeros_like(evidence), presence
    if arm == "random_key":
        permutations = torch.as_tensor(random_permutations).index_select(0, indices)
        signs = torch.as_tensor(random_signs).index_select(0, indices)
        return apply_signed_permutations(evidence, permutations, signs), presence
    if arm == "random_cluster":
        assignment = torch.as_tensor(cluster_assignment).index_select(0, indices)
        prototypes = torch.as_tensor(cluster_prototypes).index_select(0, assignment)
        replacement_presence = torch.ones(
            evidence.shape[0],
            1,
            device=presence.device,
            dtype=presence.dtype,
        )
        return (
            prototypes[:, None, :].to(
                device=evidence.device,
                dtype=evidence.dtype,
            ),
            replacement_presence,
        )
    if arm == "wrong_mask":
        return evidence, presence.roll(shifts=-1, dims=1)
    if arm == "slot_cycle":
        return evidence.roll(shifts=-1, dims=1), presence
    raise AssertionError(arm)


def descriptor_delta(reference: torch.Tensor, candidate: torch.Tensor):
    reference = torch.as_tensor(reference).detach().float().cpu()
    candidate = torch.as_tensor(candidate).detach().float().cpu()
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Descriptor matrices must have equal [N,C] shape")
    difference = candidate - reference
    row_l2 = torch.linalg.vector_norm(difference, dim=1)
    return {
        "finite": bool(torch.isfinite(candidate).all()),
        "mean_l2": float(row_l2.mean()),
        "max_abs": float(difference.abs().max()),
        "exact_equal_rows": int((difference == 0).all(dim=1).sum()),
        "rows": int(reference.shape[0]),
    }


def adjudicate(metrics, descriptor_deltas, validity_flags):
    missing_metrics = sorted(set(ARM_ORDER).difference(metrics))
    missing_deltas = sorted(set(ARM_ORDER[1:]).difference(descriptor_deltas))
    if missing_metrics or missing_deltas:
        raise ValueError(
            f"Missing metrics={missing_metrics}, deltas={missing_deltas}"
        )
    if not validity_flags:
        raise ValueError("Validity flags cannot be empty")
    if any("mAP" not in metrics[arm] for arm in ARM_ORDER):
        raise ValueError("Every arm must contain mAP")

    correct_map = float(metrics["correct"]["mAP"])
    null_max = max(float(metrics[arm]["mAP"]) for arm in PRIMARY_NULL_CONTROLS)
    semantic_margin = correct_map - null_max
    bypass_gap = correct_map - float(metrics["all_product_bypass"]["mAP"])
    active = all(
        bool(descriptor_deltas[arm]["finite"])
        and float(descriptor_deltas[arm]["mean_l2"]) > 0.0
        and float(descriptor_deltas[arm]["max_abs"]) > 0.0
        for arm in PRIMARY_ACTIVE_CONTROLS
    )
    gates = {
        "validity_all_pass": all(bool(value) for value in validity_flags.values()),
        "correct_floor": correct_map >= 0.567,
        "primary_all_controls_margin": semantic_margin >= 0.001,
        "product_bypass_gap": bypass_gap >= 0.001,
        "required_controls_active": active,
    }
    mechanism_go = all(gates.values())
    return {
        "status": "PASS" if all(bool(v) for v in validity_flags.values()) else "FAIL",
        "decision": "SPK_MECHANISM_GO" if mechanism_go else "SPK_MECHANISM_NO_GO",
        "mechanism_go": bool(mechanism_go),
        "gates": gates,
        "semantic_margin_mAP": semantic_margin,
        "product_bypass_gap_mAP": bypass_gap,
        "primary_control_max_mAP": null_max,
    }
