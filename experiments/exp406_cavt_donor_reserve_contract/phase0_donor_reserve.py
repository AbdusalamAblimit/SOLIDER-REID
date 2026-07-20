#!/usr/bin/env python3
"""Frozen donor-reserve planning and matching for the exp406 preflight."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch


POOL_TOTALS = (512, 1024, 2048, 4096, 8192, 15618)
DESCRIPTOR_NAMES = ("mass_log", "centroid_y", "confidence", "support")
SCALE_FLOOR = 1e-6
PRIMARY_CALIPER = 8.0
PREFERENCE_LIMIT = 64


class DonorReserveError(RuntimeError):
    """Failure with a JSON-safe diagnostic payload for the immutable receipt."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


class AssignmentError(RuntimeError):
    """Hall-style assignment failure with a compact conflict witness."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


class AssignmentExpansionError(RuntimeError):
    """Exhausted all frozen preference expansions without a full matching."""

    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


def _index_sha256(indices) -> str:
    array = np.asarray(list(map(int, indices)), dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_donor_plan(
    records,
    relative_paths,
    core_indices,
    *,
    stable_digest,
    expected_samples: int,
    core_samples: int,
) -> dict:
    if len(records) != int(expected_samples) or len(relative_paths) != len(records):
        raise ValueError("official donor universe size mismatch")
    core = list(map(int, core_indices))
    if len(core) != int(core_samples) or len(set(core)) != len(core):
        raise ValueError("frozen core must contain unique expected samples")
    if min(core) < 0 or max(core) >= len(records):
        raise ValueError("frozen core index out of range")
    if tuple(POOL_TOTALS) != (
        int(core_samples), 1024, 2048, 4096, 8192, int(expected_samples)
    ):
        raise ValueError("donor pool totals do not match the frozen protocol")

    core_set = set(core)
    by_camera = {}
    for index, record in enumerate(records):
        if index in core_set:
            continue
        camera = int(record[2])
        pid = int(record[1])
        by_camera.setdefault(camera, []).append(index)
        relative_path = relative_paths[index]
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("relative path must be non-empty")

    for camera, indices in by_camera.items():
        indices.sort(
            key=lambda index: (
                stable_digest(
                    "exp406-donor",
                    int(camera),
                    int(records[index][1]),
                    relative_paths[index],
                ),
                relative_paths[index],
            )
        )

    cameras = sorted(by_camera)
    cursor = {camera: 0 for camera in cameras}
    donor_order = []
    while len(donor_order) < len(records) - len(core):
        advanced = False
        for camera in cameras:
            position = cursor[camera]
            rows = by_camera[camera]
            if position < len(rows):
                donor_order.append(int(rows[position]))
                cursor[camera] = position + 1
                advanced = True
        if not advanced:
            break
    if len(donor_order) != len(records) - len(core):
        raise RuntimeError("camera round-robin donor order is incomplete")
    if len(set(donor_order)) != len(donor_order) or core_set & set(donor_order):
        raise RuntimeError("core and donor-only order are not disjoint")

    stages = []
    for total in POOL_TOTALS:
        donor_count = int(total) - int(core_samples)
        pool = core + donor_order[:donor_count]
        if len(pool) != int(total) or len(set(pool)) != int(total):
            raise RuntimeError("frozen donor prefix is malformed")
        stages.append({
            "pool_total": int(total),
            "donor_prefix_count": int(donor_count),
            "pool_indices": pool,
            "pool_indices_sha256": _index_sha256(pool),
        })
    if set(stages[-1]["pool_indices"]) != set(range(len(records))):
        raise RuntimeError("full donor stage does not cover official train")

    plan = {
        "schema": "exp406-donor-plan-v1",
        "expected_samples": int(expected_samples),
        "core_samples": int(core_samples),
        "core_indices": core,
        "core_indices_sha256": _index_sha256(core),
        "donor_order": donor_order,
        "donor_order_sha256": _index_sha256(donor_order),
        "camera_order": cameras,
        "pool_totals": list(map(int, POOL_TOTALS)),
        "stages": stages,
    }
    plan["plan_sha256"] = _canonical_sha256({
        key: value for key, value in plan.items() if key != "plan_sha256"
    })
    return plan


def _validate_donor_plan(donor_plan, core_indices, universe_size: int) -> None:
    core = list(map(int, core_indices))
    if donor_plan.get("schema") != "exp406-donor-plan-v1":
        raise ValueError("unsupported donor plan")
    if (
        int(donor_plan.get("expected_samples", -1)) != int(universe_size)
        or int(donor_plan.get("core_samples", -1)) != len(core)
        or donor_plan.get("core_indices") != core
        or donor_plan.get("core_indices_sha256") != _index_sha256(core)
    ):
        raise ValueError("donor plan core binding mismatch")
    donor_order = list(map(int, donor_plan.get("donor_order", [])))
    core_set = set(core)
    if (
        len(donor_order) != int(universe_size) - len(core)
        or len(set(donor_order)) != len(donor_order)
        or core_set.intersection(donor_order)
        or core_set.union(donor_order) != set(range(int(universe_size)))
        or donor_plan.get("donor_order_sha256") != _index_sha256(donor_order)
    ):
        raise ValueError("donor plan order is not a full disjoint universe")
    expected_totals = list(map(int, POOL_TOTALS))
    if donor_plan.get("pool_totals") != expected_totals:
        raise ValueError("donor plan pool totals changed")
    stages = donor_plan.get("stages")
    if not isinstance(stages, list) or len(stages) != len(expected_totals):
        raise ValueError("donor plan stages changed")
    for stage, total in zip(stages, expected_totals):
        donor_count = int(total) - len(core)
        expected_pool = core + donor_order[:donor_count]
        if (
            int(stage.get("pool_total", -1)) != int(total)
            or int(stage.get("donor_prefix_count", -1)) != donor_count
            or stage.get("pool_indices") != expected_pool
            or stage.get("pool_indices_sha256") != _index_sha256(expected_pool)
        ):
            raise ValueError("donor plan stage is not the frozen monotone prefix")
    expected_plan_sha = _canonical_sha256({
        key: value for key, value in donor_plan.items() if key != "plan_sha256"
    })
    if donor_plan.get("plan_sha256") != expected_plan_sha:
        raise ValueError("donor plan self-digest mismatch")


def choose_diagnostic_subset_from_core(
    targets,
    valid,
    pids,
    paths,
    core_indices,
    *,
    stable_digest,
    samples_per_slot: int,
):
    if int(samples_per_slot) <= 0:
        raise ValueError("samples_per_slot must be positive")
    core = list(map(int, core_indices))
    selected_indices = []
    selected_slots = []
    for slot in range(valid.shape[1]):
        candidates = [
            index
            for index in core
            if int(targets[index]) == slot and bool(valid[index, slot])
        ]
        candidates.sort(
            key=lambda index: stable_digest(
                "diagnostic", slot, pids[index], paths[index]
            )
        )
        first_by_pid = {}
        for index in candidates:
            first_by_pid.setdefault(int(pids[index]), index)
        diverse = sorted(
            first_by_pid.values(),
            key=lambda index: stable_digest(
                "pid", slot, pids[index], paths[index]
            ),
        )
        chosen = diverse[: int(samples_per_slot)]
        if len(chosen) < int(samples_per_slot):
            used = set(chosen)
            chosen.extend(index for index in candidates if index not in used)
            chosen = chosen[: int(samples_per_slot)]
        if len(chosen) != int(samples_per_slot):
            raise RuntimeError("insufficient core diagnostic samples for slot %d" % slot)
        selected_indices.extend(map(int, chosen))
        selected_slots.extend([int(slot)] * len(chosen))
    if len(set(selected_indices)) != len(selected_indices):
        raise RuntimeError("diagnostic recipients must be globally unique")
    if not set(selected_indices).issubset(set(core)):
        raise RuntimeError("diagnostic recipient escaped frozen core")
    return selected_indices, selected_slots


def _maximum_unique_assignment(preferences) -> list[int]:
    if not preferences or any(len(row) == 0 for row in preferences):
        raise ValueError("each recipient requires at least one donor preference")
    assignment = {}
    owner = {}
    for root in range(len(preferences)):
        queue = [root]
        cursor = 0
        parent_recipient = {}
        seen_recipients = {root}
        seen_donors = set()
        free_donor = None
        terminal_recipient = None
        while cursor < len(queue) and free_donor is None:
            recipient_position = queue[cursor]
            cursor += 1
            for donor in preferences[recipient_position]:
                donor = int(donor)
                if donor in seen_donors:
                    continue
                seen_donors.add(donor)
                if donor not in owner:
                    free_donor = donor
                    terminal_recipient = recipient_position
                    break
                next_recipient = owner[donor]
                if next_recipient not in seen_recipients:
                    seen_recipients.add(next_recipient)
                    parent_recipient[next_recipient] = (
                        recipient_position,
                        donor,
                    )
                    queue.append(next_recipient)
        if free_donor is None:
            raise AssignmentError(
                "no one-to-one assignment exists within preferences",
                {
                    "root_recipient_position": int(root),
                    "reachable_recipient_positions": sorted(map(int, seen_recipients)),
                    "reachable_donors": sorted(map(int, seen_donors)),
                    "reachable_recipient_count": int(len(seen_recipients)),
                    "reachable_donor_count": int(len(seen_donors)),
                },
            )
        current_recipient = terminal_recipient
        current_donor = free_donor
        while True:
            assignment[current_recipient] = current_donor
            owner[current_donor] = current_recipient
            if current_recipient == root:
                break
            current_recipient, current_donor = parent_recipient[current_recipient]
    result = [int(assignment[position]) for position in range(len(preferences))]
    if len(set(result)) != len(result):
        raise RuntimeError("assignment unexpectedly reused a donor")
    return result


def _assignment_with_expansion(preferences, initial_limit: int):
    if int(initial_limit) <= 0:
        raise ValueError("initial preference limit must be positive")
    maximum = max(len(row) for row in preferences)
    limits = []
    limit = int(initial_limit)
    while limit < maximum:
        limits.append(limit)
        limit *= 2
    limits.append(maximum)
    errors = []
    for limit in limits:
        scope = "full-caliper" if int(limit) == int(maximum) else "top-%d" % int(limit)
        try:
            donors = _maximum_unique_assignment([row[:limit] for row in preferences])
            errors.append({
                "limit": int(limit),
                "scope": scope,
                "status": "MATCHED",
            })
            return donors, int(limit), errors
        except RuntimeError as error:
            errors.append({
                "limit": int(limit),
                "scope": scope,
                "status": "HALL_FAIL",
                "error": str(error),
                "witness": getattr(error, "diagnostics", None),
            })
    raise AssignmentExpansionError(
        "no full-caliper one-to-one assignment exists",
        errors,
    )


def _descriptor_values(mass, centroid_y, confidence, support):
    return {
        "mass_log": mass.clamp_min(1e-12).log().double(),
        "centroid_y": centroid_y.double(),
        "confidence": confidence.double(),
        "support": support.double(),
    }


def _frozen_scale_summary(valid, descriptor_values, core_indices):
    core_mask = torch.zeros(len(valid), dtype=torch.bool)
    core_mask[torch.tensor(list(map(int, core_indices)), dtype=torch.long)] = True
    scales = {}
    summary = {}
    for slot in range(valid.shape[1]):
        active = valid[:, slot] & core_mask
        active_count = int(active.sum())
        if active_count == 0:
            raise RuntimeError("frozen core has no analysis-valid scale rows")
        scales[int(slot)] = {}
        summary[str(slot)] = {}
        for name, values in descriptor_values.items():
            selected = values[active, slot]
            median = selected.median()
            raw_mad = (selected - median).abs().median()
            scale = raw_mad.clamp_min(SCALE_FLOOR)
            scales[int(slot)][name] = scale
            summary[str(slot)][name] = {
                "median": float(median),
                "raw_mad": float(raw_mad),
                "scale": float(scale),
                "scale_floor_applied": bool(raw_mad < SCALE_FLOOR),
                "active_count": active_count,
            }
    return scales, summary


def _stage_preferences(
    indices,
    slots,
    valid,
    descriptor_values,
    scales,
    global_feature,
    pids,
    camids,
    keys,
    forbidden,
    pool_indices,
    primary_caliper: float,
):
    pool_mask = torch.zeros(len(valid), dtype=torch.bool)
    pool_mask[torch.tensor(pool_indices, dtype=torch.long)] = True
    preferences = []
    rows = []
    all_have_edges = True
    for position, (recipient, slot) in enumerate(zip(indices, slots)):
        recipient = int(recipient)
        slot = int(slot)
        candidate = (
            pool_mask
            & valid[:, slot]
            & (pids != pids[recipient])
            & (camids == camids[recipient])
            & ~forbidden
        )
        candidate_indices = torch.nonzero(candidate, as_tuple=False).flatten()
        gaps = {
            name: (values[candidate_indices, slot] - values[recipient, slot]).abs()
            for name, values in descriptor_values.items()
        }
        if len(candidate_indices):
            primary_distance = sum(
                gaps[name] / scales[slot][name]
                for name in DESCRIPTOR_NAMES
            )
            cosine_gap = 1.0 - (
                global_feature[candidate_indices].double()
                @ global_feature[recipient].double()
            ).clamp(-1, 1)
            feasible = torch.nonzero(
                primary_distance <= float(primary_caliper), as_tuple=False
            ).flatten()
            nearest = float(primary_distance.min())
        else:
            primary_distance = torch.empty(0, dtype=torch.float64)
            cosine_gap = torch.empty(0, dtype=torch.float64)
            feasible = torch.empty(0, dtype=torch.long)
            nearest = None
        degree = int(len(feasible))
        if degree == 0:
            all_have_edges = False
            preference = np.asarray([], dtype=np.int32)
        else:
            feasible_candidates = candidate_indices.index_select(0, feasible)
            feasible_primary = primary_distance.index_select(0, feasible)
            feasible_cosine = cosine_gap.index_select(0, feasible)
            ranked = np.lexsort((
                keys.index_select(0, feasible_candidates).cpu().numpy(),
                feasible_cosine.cpu().numpy(),
                feasible_primary.cpu().numpy(),
                (feasible_primary + feasible_cosine).cpu().numpy(),
            ))
            preference = feasible_candidates.cpu().numpy()[ranked].astype(
                np.int32, copy=False
            )
        preferences.append(preference)
        rows.append({
            "position": int(position),
            "recipient": recipient,
            "slot": slot,
            "camera": int(camids[recipient]),
            "recipient_pid": int(pids[recipient]),
            "candidate_count": int(len(candidate_indices)),
            "caliper_degree": degree,
            "nearest_primary_distance": nearest,
        })
    return preferences, rows, all_have_edges


def validate_failure_diagnostics(diagnostics: dict) -> None:
    required_global = (
        "schema", "core_count", "execution_count", "recipient_count",
        "core_indices_sha256", "donor_order_sha256", "plan_sha256",
        "pool_totals", "scale_source", "scale_floor", "scales",
        "primary_caliper", "preference_limit", "forbidden_recipient_count",
        "stages", "contract_strict",
    )
    if any(key not in diagnostics for key in required_global):
        raise ValueError("failure diagnostics lack a required global field")
    if diagnostics["contract_strict"] is not False:
        raise ValueError("failure diagnostics must be explicitly non-strict")
    if diagnostics["scale_source"] != "frozen-core-512-analysis-valid":
        raise ValueError("failure diagnostics scale source changed")
    scales = diagnostics["scales"]
    if set(scales) != {"0", "1", "2", "3", "4"}:
        raise ValueError("failure diagnostics do not cover all slots")
    scale_fields = {
        "median", "raw_mad", "scale", "scale_floor_applied", "active_count"
    }
    for slot in scales.values():
        if set(slot) != set(DESCRIPTOR_NAMES):
            raise ValueError("failure diagnostics do not cover all descriptors")
        if any(not scale_fields.issubset(values) for values in slot.values()):
            raise ValueError("failure diagnostics lack scale fields")
    stages = diagnostics["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("failure diagnostics require at least one stage")
    stage_fields = {
        "pool_total", "donor_prefix_count", "pool_indices_sha256",
        "all_recipients_have_caliper_edge", "zero_edge_count",
        "recipient_rows", "status", "assignment_attempts",
    }
    recipient_fields = {
        "position", "recipient", "slot", "camera", "recipient_pid",
        "candidate_count", "caliper_degree", "nearest_primary_distance",
    }
    for stage in stages:
        if not stage_fields.issubset(stage):
            raise ValueError("failure diagnostics lack stage fields")
        if len(stage["recipient_rows"]) != int(diagnostics["recipient_count"]):
            raise ValueError("failure diagnostics recipient row count changed")
        if any(not recipient_fields.issubset(row) for row in stage["recipient_rows"]):
            raise ValueError("failure diagnostics lack recipient fields")
        if stage["status"] == "ZERO_EDGE":
            if int(stage["zero_edge_count"]) <= 0 or stage["assignment_attempts"]:
                raise ValueError("ZERO_EDGE diagnostics are inconsistent")
        elif stage["status"] == "HALL_FAIL":
            attempts = stage["assignment_attempts"]
            if (
                int(stage["zero_edge_count"]) != 0
                or not attempts
                or any(
                    not {"limit", "scope", "status", "error", "witness"}.issubset(
                        attempt
                    )
                    for attempt in attempts
                )
                or any(attempt["status"] != "HALL_FAIL" for attempt in attempts)
                or attempts[-1]["scope"] != "full-caliper"
                or int(stage.get("preference_limit_used", -1))
                != int(attempts[-1]["limit"])
            ):
                raise ValueError("HALL_FAIL diagnostics are incomplete")
        else:
            raise ValueError("failure diagnostics contain a non-failure stage")


def choose_wrong_masks_progressive(
    indices,
    slots,
    valid,
    mass,
    centroid_y,
    confidence,
    support,
    global_feature,
    pids,
    camids,
    keys,
    forbidden_donor_indices,
    core_indices,
    donor_plan,
    *,
    primary_caliper: float,
    preference_limit: int,
):
    if float(primary_caliper) != PRIMARY_CALIPER:
        raise ValueError("primary caliper must remain frozen at 8.0")
    if int(preference_limit) != PREFERENCE_LIMIT:
        raise ValueError("preference limit must remain frozen at 64")
    recipients = list(map(int, indices))
    slots = list(map(int, slots))
    if len(recipients) != len(slots) or not recipients:
        raise ValueError("wrong-mask recipients/slots must be aligned")
    if len(recipients) != 20 or len(set(recipients)) != 20:
        raise ValueError("exp406 preflight requires 20 unique recipients")
    core = list(map(int, core_indices))
    if not set(recipients).issubset(set(core)):
        raise ValueError("recipient escaped frozen core")
    _validate_donor_plan(donor_plan, core, len(valid))

    descriptor_values = _descriptor_values(mass, centroid_y, confidence, support)
    scales, scale_summary = _frozen_scale_summary(
        valid, descriptor_values, core
    )
    forbidden = torch.zeros(len(valid), dtype=torch.bool)
    forbidden_indices = sorted(set(map(int, forbidden_donor_indices)))
    forbidden[torch.tensor(forbidden_indices, dtype=torch.long)] = True
    stage_summaries = []
    selected = None

    for stage in donor_plan["stages"]:
        preferences, recipient_rows, all_have_edges = _stage_preferences(
            recipients,
            slots,
            valid,
            descriptor_values,
            scales,
            global_feature,
            pids,
            camids,
            keys,
            forbidden,
            stage["pool_indices"],
            primary_caliper,
        )
        stage_summary = {
            "pool_total": int(stage["pool_total"]),
            "donor_prefix_count": int(stage["donor_prefix_count"]),
            "pool_indices_sha256": stage["pool_indices_sha256"],
            "all_recipients_have_caliper_edge": bool(all_have_edges),
            "zero_edge_count": int(sum(
                row["caliper_degree"] == 0 for row in recipient_rows
            )),
            "recipient_rows": recipient_rows,
            "status": "ZERO_EDGE" if not all_have_edges else "PENDING_ASSIGNMENT",
            "assignment_attempts": [],
        }
        if all_have_edges:
            try:
                donors, limit_used, assignment_errors = _assignment_with_expansion(
                    preferences, int(preference_limit)
                )
                stage_summary["status"] = "MATCHED"
                stage_summary["preference_limit_used"] = int(limit_used)
                stage_summary["assignment_attempts"] = assignment_errors
                selected = {
                    "stage": stage,
                    "preferences": preferences,
                    "donors": donors,
                    "preference_limit_used": int(limit_used),
                    "recipient_rows": recipient_rows,
                }
            except RuntimeError as error:
                stage_summary["status"] = "HALL_FAIL"
                stage_summary["assignment_error"] = str(error)
                stage_summary["assignment_attempts"] = list(
                    getattr(error, "attempts", [])
                )
                if stage_summary["assignment_attempts"]:
                    stage_summary["preference_limit_used"] = int(
                        stage_summary["assignment_attempts"][-1]["limit"]
                    )
        stage_summaries.append(stage_summary)
        if selected is not None:
            break

    diagnostic_base = {
        "schema": "exp406-donor-reserve-summary-v1",
        "core_count": int(len(core)),
        "execution_count": int(len(valid)),
        "recipient_count": int(len(recipients)),
        "core_indices_sha256": donor_plan["core_indices_sha256"],
        "donor_order_sha256": donor_plan["donor_order_sha256"],
        "plan_sha256": donor_plan["plan_sha256"],
        "pool_totals": donor_plan["pool_totals"],
        "scale_source": "frozen-core-512-analysis-valid",
        "scale_floor": SCALE_FLOOR,
        "scales": scale_summary,
        "primary_caliper": float(primary_caliper),
        "preference_limit": int(preference_limit),
        "forbidden_recipient_count": int(len(forbidden_indices)),
        "stages": stage_summaries,
    }
    if selected is None:
        diagnostic_base["contract_strict"] = False
        validate_failure_diagnostics(diagnostic_base)
        raise DonorReserveError(
            "no frozen donor prefix yields a full caliper assignment",
            diagnostic_base,
        )

    donors = selected["donors"]
    selected_stage = selected["stage"]
    selected_pool = set(map(int, selected_stage["pool_indices"]))
    records = []
    for recipient, slot, donor in zip(recipients, slots, donors):
        candidate = (
            valid[:, slot]
            & (pids != pids[recipient])
            & (camids == camids[recipient])
            & ~forbidden
        )
        pool_mask = torch.zeros(len(valid), dtype=torch.bool)
        pool_mask[torch.tensor(sorted(selected_pool), dtype=torch.long)] = True
        candidate &= pool_mask
        candidate_indices = torch.nonzero(candidate, as_tuple=False).flatten()
        selected_position = torch.nonzero(
            candidate_indices == int(donor), as_tuple=False
        ).flatten()
        if len(selected_position) != 1:
            raise RuntimeError("assigned donor left frozen stage candidate set")
        selected_position = int(selected_position.item())
        gaps = {
            name: (values[candidate_indices, slot] - values[recipient, slot]).abs()
            for name, values in descriptor_values.items()
        }
        primary_distance = sum(
            gaps[name] / scales[slot][name] for name in DESCRIPTOR_NAMES
        )
        chosen_primary = primary_distance[selected_position]
        if float(chosen_primary) > float(primary_caliper):
            raise RuntimeError("assigned donor violated frozen primary caliper")
        stable_tie_before = (
            (primary_distance == chosen_primary)
            & (candidate_indices < int(donor))
        ).sum()
        chosen_cosine = 1.0 - (
            global_feature[int(donor)].double()
            @ global_feature[int(recipient)].double()
        ).clamp(-1, 1)
        records.append({
            "recipient": int(recipient),
            "donor": int(donor),
            "slot": int(slot),
            "recipient_pid": int(pids[recipient]),
            "donor_pid": int(pids[donor]),
            "camera": int(camids[recipient]),
            "mass_log_gap": float(gaps["mass_log"][selected_position]),
            "centroid_y_gap": float(gaps["centroid_y"][selected_position]),
            "confidence_gap": float(gaps["confidence"][selected_position]),
            "support_gap": float(gaps["support"][selected_position]),
            "global_cosine_gap": float(chosen_cosine),
            "primary_distance": float(chosen_primary),
            "primary_rank": int(
                (primary_distance < chosen_primary).sum() + stable_tie_before
            ),
            "candidate_count": int(len(candidate_indices)),
            "caliper_candidate_count": int(
                (primary_distance <= float(primary_caliper)).sum()
            ),
            "preference_limit_used": int(selected["preference_limit_used"]),
            "donor_pool_total": int(selected_stage["pool_total"]),
        })

    assigned = set(map(int, donors))
    contract_strict = bool(
        len(core) == 512
        and len(valid) == 15618
        and len(recipients) == 20
        and len(assigned) == len(recipients)
        and not assigned.intersection(recipients)
        and all(row["recipient_pid"] != row["donor_pid"] for row in records)
        and all(int(camids[row["recipient"]]) == int(camids[row["donor"]]) for row in records)
        and all(row["primary_distance"] <= float(primary_caliper) for row in records)
        and diagnostic_base["scale_source"] == "frozen-core-512-analysis-valid"
        and int(selected_stage["pool_total"]) in donor_plan["pool_totals"]
    )
    diagnostic_base.update({
        "contract_strict": contract_strict,
        "selected_pool_total": int(selected_stage["pool_total"]),
        "selected_pool_indices_sha256": selected_stage["pool_indices_sha256"],
        "selected_preference_limit": int(selected["preference_limit_used"]),
        "assigned_donor_count": int(len(assigned)),
        "recipient_donor_overlap_count": int(len(assigned.intersection(recipients))),
        "summary_sha256": None,
    })
    diagnostic_base["summary_sha256"] = _canonical_sha256({
        key: value
        for key, value in diagnostic_base.items()
        if key != "summary_sha256"
    })
    if not contract_strict:
        raise DonorReserveError(
            "selected donor reserve violates the frozen contract",
            diagnostic_base,
        )
    return donors, records, diagnostic_base
