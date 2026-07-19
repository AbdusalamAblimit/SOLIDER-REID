#!/usr/bin/env python3
"""Frozen exp403 full-retrieval interventions and adjudication."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch


def _load_base():
    path = Path(os.environ["EXP403_BASE_CORE"]).resolve()
    contract = json.loads(
        Path(os.environ["EXP403_CONTRACT"]).resolve().read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != contract["sha256"]["base_core"]:
        raise RuntimeError("Base core SHA mismatch")
    spec = importlib.util.spec_from_file_location("exp403_base_core", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base()

ARM_ORDER = (
    "correct",
    "wrong_rgb_evidence",
    "generic_evidence",
    "static_zero_evidence",
    "evidence_slot_cycle",
    "wrong_mask_binding",
    "all_router_bypass",
)
SEMANTIC_CONTROLS = (
    "wrong_rgb_evidence",
    "generic_evidence",
    "static_zero_evidence",
)
STATE_INTERVENTIONS = frozenset(
    {
        "wrong_rgb_evidence",
        "generic_evidence",
        "static_zero_evidence",
        "evidence_slot_cycle",
        "wrong_mask_binding",
    }
)
BYPASS_BANKS = {"all_router_bypass": frozenset({0, 1})}

SEALED_D0_MAP = 0.575587756578
SEALED_D0_R1 = 0.676923076923
MARGIN = 0.001

GENERIC_EVIDENCE = (
    (0.02078495247430696, -0.01896744447652433, -0.023763453455863715, 0.04369581280365184, 0.03928206937509305, 0.016063132736722042, -0.0072795776395747, 0.01420812156794203, -0.004097140947703963, 0.021169365970644265, 0.0034399779945864816, -0.003299031268293553, -0.021555981821402302, 0.035236289258523505, -0.002713698283139624, -0.02428137468572481),
    (0.025716986472027463, 0.05227585440085943, -0.029003981712311704, 0.016408446803646948, 0.01693974116844594, 0.020994199409872905, 0.017583667729580855, 0.014339220364008814, 0.00015709948460675073, 0.006311751326759568, -0.03234897539867055, 0.0038782242964913783, -0.006875090709980546, 0.024058816987687925, 0.0053988854030979656, -0.01825383699828453),
    (0.023479999285487222, 0.03086025993582661, -0.023879629529404218, 0.01138272052423581, 0.028439463772770547, 0.02397864366759201, 0.021903032226469187, 0.010296469967338778, 0.0017745831099552425, 0.017071730671301424, -0.0205471936330958, -0.006640725847288414, -0.01050068575418621, 0.0295555472517589, -0.004297281142650007, -0.02580949735201553),
    (0.016386371072233525, 0.005515299567468511, -0.007367302264598555, 0.022157748198735484, 0.0014628901012290548, 0.021858757759202554, 0.020951242439293236, 0.012130846447519114, -0.007462342269525357, 0.013460207941951808, -0.010763128765808015, -0.028269235991679048, -0.003765594657220228, 0.01308891870069719, -0.0006406049591322089, -0.016530039061108736),
    (0.022665750441630463, 0.020858422167857223, -0.013909025153278855, 0.016413676705558957, 0.021766385220420355, 0.01067954726760111, 0.017139514586189748, -0.013124892073044438, -0.0058082231424287575, 0.025458811310037206, -0.01082473412590414, -0.03239368058503029, -0.021455098191511715, 0.006863016699462282, -0.006181075222424788, -0.029845515806963215),
)

build_global_donor_map = _BASE.build_global_donor_map
validate_donor_map = _BASE.validate_donor_map
canonical_orthogonal = _BASE.canonical_orthogonal
descriptor_delta = _BASE.descriptor_delta
tensor_mapping_sha256 = _BASE.tensor_mapping_sha256
bypass_routers = _BASE.bypass_routers


def apply_state_intervention(
    state,
    arm: str,
    absolute_indices=None,
    donor_map=None,
    evidence_cache=None,
    orthogonal=None,
):
    del orthogonal
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
        indices = torch.as_tensor(absolute_indices, dtype=torch.int64).flatten().cpu()
        donor = torch.as_tensor(donor_map, dtype=torch.int64).flatten().cpu()
        cache = torch.as_tensor(evidence_cache).cpu()
        replacement = cache.index_select(0, donor.index_select(0, indices))
        updated["consumer_evidence"] = replacement.to(
            device=evidence.device, dtype=evidence.dtype
        )
    elif arm == "generic_evidence":
        generic = torch.tensor(
            GENERIC_EVIDENCE, device=evidence.device, dtype=evidence.dtype
        )
        updated["consumer_evidence"] = generic.unsqueeze(0).expand_as(evidence)
    elif arm == "static_zero_evidence":
        updated["consumer_evidence"] = torch.zeros_like(evidence)
    elif arm == "evidence_slot_cycle":
        updated["consumer_evidence"] = evidence.roll(shifts=-1, dims=1)
    elif arm == "wrong_mask_binding":
        cycled_mask = mask.roll(shifts=-1, dims=1)
        cycled_presence = presence.roll(shifts=-1, dims=1)
        updated["consumer_mask"] = cycled_mask
        updated["consumer_presence"] = cycled_presence
        updated["consumer_field"] = cycled_mask * cycled_presence[..., None, None]
        updated["reliability"] = cycled_presence
    return updated


def adjudicate(metrics, descriptor_deltas, validity_flags):
    missing_metrics = sorted(set(ARM_ORDER).difference(metrics))
    missing_deltas = sorted(set(ARM_ORDER[1:]).difference(descriptor_deltas))
    if missing_metrics or missing_deltas:
        raise ValueError(f"Missing metrics={missing_metrics}, deltas={missing_deltas}")
    correct = metrics["correct"]
    correct_map = float(correct["mAP"])
    semantic_max = max(float(metrics[arm]["mAP"]) for arm in SEMANTIC_CONTROLS)
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
        "full_map_floor": correct_map >= SEALED_D0_MAP,
        "full_r1_floor": float(correct["rank1"]) >= SEALED_D0_R1,
        "correct_max_control_margin": semantic_margin >= MARGIN,
        "correct_all_bypass_margin": route_gap >= MARGIN,
        "all_descriptors_active": descriptor_active,
    }
    go = all(gates.values())
    return {
        "status": "PASS" if gates["validity_all_pass"] else "FAIL",
        "decision": (
            "EVIDENCE_OPERATOR_OWNERSHIP_GO"
            if go
            else "ELO_CUR_MECHANISM_NO_GO"
        ),
        "phase_b_formal_mechanism_design_authorized": bool(go),
        "gates": gates,
        "semantic_margin_mAP": semantic_margin,
        "route_gap_mAP": route_gap,
        "semantic_control_max_mAP": semantic_max,
        "sealed_d0_map_floor": SEALED_D0_MAP,
        "sealed_d0_r1_floor": SEALED_D0_R1,
    }
