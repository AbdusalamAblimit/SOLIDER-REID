#!/usr/bin/env python3
"""Pure counterfactual contracts shared by exp402 static and CUDA audits."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager

import torch


ARM_ORDER = (
    "correct",
    "wrong_rgb_evidence",
    "static_zero_evidence",
    "orthogonal_evidence",
    "evidence_slot_cycle",
    "wrong_mask_binding",
    "generic_expert_mean",
    "bypass_router0",
    "bypass_router1",
    "all_router_bypass",
)
SEMANTIC_CONTROLS = (
    "wrong_rgb_evidence",
    "static_zero_evidence",
    "orthogonal_evidence",
    "evidence_slot_cycle",
    "wrong_mask_binding",
    "generic_expert_mean",
)
STATE_INTERVENTIONS = frozenset(
    {
        "wrong_rgb_evidence",
        "static_zero_evidence",
        "orthogonal_evidence",
        "evidence_slot_cycle",
        "wrong_mask_binding",
    }
)
BYPASS_BANKS = {
    "bypass_router0": frozenset({0}),
    "bypass_router1": frozenset({1}),
    "all_router_bypass": frozenset({0, 1}),
}


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
    pids = torch.as_tensor(pids, dtype=torch.int64).flatten().cpu()
    camids = torch.as_tensor(camids, dtype=torch.int64).flatten().cpu()
    if pids.shape != camids.shape or pids.numel() <= 1:
        raise ValueError("PID/camera vectors must have equal nontrivial shape")
    if not 0 < int(num_query) < pids.numel():
        raise ValueError("num_query must split the validation records")

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
            raise RuntimeError("Donor group has fewer than two records")
        for position, recipient in enumerate(group.tolist()):
            for offset in range(1, group.numel()):
                candidate = int(group[(position + offset) % group.numel()])
                if int(pids[candidate]) != int(pids[recipient]):
                    donor[recipient] = candidate
                    break
    if bool((donor < 0).any()):
        raise RuntimeError("Could not build a different-PID global donor map")
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
        raise RuntimeError("Donor map index out of range")
    indices = torch.arange(count)
    recipient_split = indices >= int(num_query)
    donor_split = donor >= int(num_query)
    summary = {
        "count": count,
        "no_fixed_points": bool((donor != indices).all()),
        "different_pid_fraction": float((pids[donor] != pids).float().mean()),
        "same_camera_fraction": float((camids[donor] == camids).float().mean()),
        "same_split_fraction": float(
            (donor_split == recipient_split).float().mean()
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


def canonical_orthogonal(
    dimension: int = 16,
    seed: int = 1234,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if int(dimension) <= 1:
        raise ValueError("Orthogonal dimension must exceed one")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        int(dimension),
        int(dimension),
        generator=generator,
        dtype=dtype,
    )
    orthogonal, triangular = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(triangular))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return orthogonal * signs


def apply_state_intervention(
    state,
    arm: str,
    absolute_indices=None,
    donor_map=None,
    evidence_cache=None,
    orthogonal=None,
):
    if arm not in STATE_INTERVENTIONS:
        raise ValueError(f"Not a state intervention arm: {arm}")
    updated = dict(state)
    evidence = state["consumer_evidence"]
    mask = state["consumer_mask"]
    presence = state["consumer_presence"]
    if evidence.ndim != 3 or evidence.shape[1:] != (5, 16):
        raise ValueError("consumer_evidence must have shape [B,5,16]")
    if mask.ndim != 4 or mask.shape[:2] != evidence.shape[:2]:
        raise ValueError("consumer_mask shape mismatch")
    if presence.shape != evidence.shape[:2]:
        raise ValueError("consumer_presence shape mismatch")

    if arm == "wrong_rgb_evidence":
        indices = torch.as_tensor(
            absolute_indices,
            dtype=torch.int64,
        ).flatten().cpu()
        donor = torch.as_tensor(donor_map, dtype=torch.int64).flatten().cpu()
        cache = torch.as_tensor(evidence_cache).cpu()
        if indices.numel() != evidence.shape[0]:
            raise ValueError("Absolute-index batch size mismatch")
        if cache.ndim != 3 or cache.shape[1:] != evidence.shape[1:]:
            raise ValueError("Evidence cache shape mismatch")
        if donor.numel() != cache.shape[0]:
            raise ValueError("Donor/cache count mismatch")
        replacement = cache.index_select(0, donor.index_select(0, indices))
        updated["consumer_evidence"] = replacement.to(
            device=evidence.device,
            dtype=evidence.dtype,
        )
    elif arm == "static_zero_evidence":
        updated["consumer_evidence"] = torch.zeros_like(evidence)
    elif arm == "orthogonal_evidence":
        matrix = torch.as_tensor(orthogonal)
        if matrix.shape != (evidence.shape[-1], evidence.shape[-1]):
            raise ValueError("Orthogonal matrix shape mismatch")
        rotated = torch.matmul(
            evidence.float(),
            matrix.to(device=evidence.device, dtype=torch.float32),
        )
        updated["consumer_evidence"] = rotated.to(evidence.dtype)
    elif arm == "evidence_slot_cycle":
        updated["consumer_evidence"] = evidence.roll(shifts=-1, dims=1)
    elif arm == "wrong_mask_binding":
        cycled_mask = mask.roll(shifts=-1, dims=1)
        cycled_presence = presence.roll(shifts=-1, dims=1)
        updated["consumer_mask"] = cycled_mask
        updated["consumer_presence"] = cycled_presence
        updated["consumer_field"] = (
            cycled_mask * cycled_presence[..., None, None]
        )
        updated["reliability"] = cycled_presence
    return updated


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


def _expert_weight_mapping(tapf):
    mapping = {}
    for bank_index, router in enumerate(tapf.psg_bank):
        for slot, expert in enumerate(router.experts):
            mapping[f"{bank_index}.{slot}.weight"] = expert.weight
    return mapping


@contextmanager
def generic_expert_mean(tapf):
    before = {
        name: value.detach().clone()
        for name, value in _expert_weight_mapping(tapf).items()
    }
    before_sha = tensor_mapping_sha256(before)
    with torch.no_grad():
        for router in tapf.psg_bank:
            mean = torch.stack(
                [expert.weight.detach() for expert in router.experts],
                dim=0,
            ).mean(dim=0)
            for expert in router.experts:
                expert.weight.copy_(mean)
    during = {
        name: value.detach().clone()
        for name, value in _expert_weight_mapping(tapf).items()
    }
    report = {
        "before_sha256": before_sha,
        "all_banks_mean_exact": all(
            torch.equal(
                router.experts[0].weight,
                expert.weight,
            )
            for router in tapf.psg_bank
            for expert in router.experts[1:]
        ),
        "during_sha256": tensor_mapping_sha256(during),
        "restored_exact": False,
    }
    try:
        yield report
    finally:
        with torch.no_grad():
            current = _expert_weight_mapping(tapf)
            for name, value in before.items():
                current[name].copy_(value)
        restored = {
            name: value.detach().clone()
            for name, value in _expert_weight_mapping(tapf).items()
        }
        report["restored_exact"] = (
            tensor_mapping_sha256(restored) == before_sha
        )


@contextmanager
def bypass_routers(tapf, banks):
    banks = frozenset(int(bank) for bank in banks)
    if not banks or not banks.issubset({0, 1}):
        raise ValueError("Bypass banks must be a nonempty subset of {0,1}")
    original = tapf.apply_gate
    had_override = "apply_gate" in tapf.__dict__
    override_before = tapf.__dict__.get("apply_gate")
    calls = [0, 0]
    bypassed = [0, 0]

    def patched(bank_index, tokens, hw_shape, state):
        if bank_index not in (0, 1):
            raise RuntimeError(f"Unexpected router bank: {bank_index}")
        calls[bank_index] += 1
        if bank_index in banks:
            bypassed[bank_index] += 1
            state["gate_deltas"].append(torch.zeros_like(tokens))
            return tokens
        return original(bank_index, tokens, hw_shape, state)

    report = {
        "banks": sorted(banks),
        "calls": calls,
        "bypassed": bypassed,
        "restored_exact": False,
    }
    tapf.apply_gate = patched
    try:
        yield report
    finally:
        if had_override:
            tapf.__dict__["apply_gate"] = override_before
        else:
            tapf.__dict__.pop("apply_gate", None)
        report["restored_exact"] = (
            ("apply_gate" in tapf.__dict__) == had_override
            and (
                not had_override
                or tapf.__dict__.get("apply_gate") is override_before
            )
        )


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
    semantic_max = max(
        float(metrics[arm]["mAP"])
        for arm in SEMANTIC_CONTROLS
    )
    semantic_margin = correct_map - semantic_max
    route_gap = correct_map - float(metrics["all_router_bypass"]["mAP"])
    descriptor_active = all(
        bool(descriptor_deltas[arm]["finite"])
        and float(descriptor_deltas[arm]["mean_l2"]) > 0.0
        and float(descriptor_deltas[arm]["max_abs"]) > 0.0
        for arm in ARM_ORDER[1:]
    )
    gates = {
        "validity_all_pass": all(bool(value) for value in validity_flags.values()),
        "semantic_all_controls_margin": semantic_margin >= 0.001,
        "route_gap_replicated": route_gap >= 0.001,
        "correct_floor": correct_map >= 0.567,
        "all_descriptors_active": descriptor_active,
    }
    go = all(gates.values())
    return {
        "status": "PASS" if all(bool(v) for v in validity_flags.values()) else "FAIL",
        "decision": (
            "PHASE_B_SEMANTIC_INTERFACE_IDENTIFIABLE"
            if go
            else "CURRENT_SEMANTIC_INTERFACE_NO_GO"
        ),
        "phase_b_formal_mechanism_design_authorized": bool(go),
        "gates": gates,
        "semantic_margin_mAP": semantic_margin,
        "route_gap_mAP": route_gap,
        "semantic_control_max_mAP": semantic_max,
    }
