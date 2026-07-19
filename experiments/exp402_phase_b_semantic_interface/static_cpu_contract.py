#!/usr/bin/env python3
"""CPU/static positive and negative contracts for exp402."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp402_core", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def source_contract(paths):
    forbidden_calls = {
        "backward",
        "save",
        "save_checkpoint",
        "step",
        "train",
    }
    forbidden_names = {
        "GradScaler",
        "PoseTargetStore",
        "make_dataloader",
        "optimizer",
        "scheduler",
    }
    forbidden_literals = {
        "/mnt1/afrderived",
        "CLIP_CHECKPOINT",
        "RICH_CODEBOOK",
    }
    findings = []
    sha = {}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        sha[path.name] = sha256_file(path)
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                if isinstance(target, ast.Attribute):
                    name = target.attr
                elif isinstance(target, ast.Name):
                    name = target.id
                else:
                    name = ""
                if name in forbidden_calls:
                    findings.append(f"{path.name}:call:{name}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                findings.append(f"{path.name}:name:{node.id}:{node.lineno}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for literal in forbidden_literals:
                    if literal in node.value:
                        findings.append(
                            f"{path.name}:literal:{literal}:{node.lineno}"
                        )
    return {
        "findings": sorted(findings),
        "sha256": sha,
    }


def toy_state(batch=4):
    evidence = torch.arange(
        batch * 5 * 16,
        dtype=torch.float64,
    ).reshape(batch, 5, 16)
    evidence = torch.nn.functional.normalize(evidence + 1.0, dim=-1)
    mask = torch.arange(
        batch * 5 * 3 * 2,
        dtype=torch.float64,
    ).reshape(batch, 5, 3, 2)
    presence = torch.linspace(0.1, 0.9, batch * 5).reshape(batch, 5)
    return {
        "consumer_evidence": evidence,
        "consumer_mask": mask,
        "consumer_presence": presence,
        "consumer_field": mask * presence[..., None, None],
        "reliability": presence,
        "student_evidence": evidence + 10.0,
        "sentinel": torch.tensor([13.0], dtype=torch.float64),
    }


class ToyRouter(nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(3, 4, bias=False) for _ in range(5)]
        )
        with torch.no_grad():
            for slot, expert in enumerate(self.experts):
                value = torch.arange(12, dtype=torch.float64).reshape(4, 3)
                expert.weight.copy_(value + 100.0 * bank + 10.0 * slot)
        self.double()


class ToyTapf(nn.Module):
    def __init__(self):
        super().__init__()
        self.psg_bank = nn.ModuleList([ToyRouter(0), ToyRouter(1)])

    def apply_gate(self, bank_index, tokens, hw_shape, state):
        del hw_shape
        delta = torch.full_like(tokens, float(bank_index + 1))
        state["gate_deltas"].append(delta)
        return tokens + delta


def run_contract(core, inspected_paths):
    torch.manual_seed(1234)
    cuda_before = torch.cuda.is_initialized()

    pids = torch.tensor(
        [0, 0, 1, 2, 3, 4, 4, 5, 10, 10, 11, 12, 13, 14, 14, 15]
    )
    camids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1] * 2)
    donor = core.build_global_donor_map(pids, camids, num_query=8)
    donor_summary = core.validate_donor_map(donor, pids, camids, 8)
    donor_repeat = core.build_global_donor_map(pids, camids, num_query=8)

    cache = torch.arange(16 * 5 * 16, dtype=torch.float64).reshape(16, 5, 16)
    state = toy_state(batch=16)
    state["consumer_evidence"] = cache.clone()
    expected_wrong = cache.index_select(0, donor)
    chunk_outputs = {}
    for chunk_size in (2, 5, 16):
        parts = []
        for start in range(0, 16, chunk_size):
            stop = min(start + chunk_size, 16)
            chunk_state = toy_state(batch=stop - start)
            chunk_state["consumer_evidence"] = cache[start:stop].clone()
            changed = core.apply_state_intervention(
                chunk_state,
                "wrong_rgb_evidence",
                absolute_indices=torch.arange(start, stop),
                donor_map=donor,
                evidence_cache=cache,
            )
            parts.append(changed["consumer_evidence"])
        chunk_outputs[str(chunk_size)] = torch.cat(parts, dim=0)

    base = toy_state()
    zero = core.apply_state_intervention(base, "static_zero_evidence")
    slot_cycle = core.apply_state_intervention(base, "evidence_slot_cycle")
    binding_cycle = core.apply_state_intervention(base, "wrong_mask_binding")

    rng_before = torch.get_rng_state().clone()
    orthogonal = core.canonical_orthogonal(16, 1234)
    rng_after = torch.get_rng_state().clone()
    orthogonal_repeat = core.canonical_orthogonal(16, 1234)
    rotated = core.apply_state_intervention(
        base,
        "orthogonal_evidence",
        orthogonal=orthogonal,
    )
    identity = torch.eye(16, dtype=torch.float64)
    gram_error = float((orthogonal.T @ orthogonal - identity).abs().max())
    norm_error = float(
        (
            rotated["consumer_evidence"].norm(dim=-1)
            - base["consumer_evidence"].norm(dim=-1)
        ).abs().max()
    )
    flat_before = base["consumer_evidence"].reshape(-1, 16)
    flat_after = rotated["consumer_evidence"].reshape(-1, 16)
    cosine_error = float(
        (flat_before @ flat_before.T - flat_after @ flat_after.T).abs().max()
    )

    tapf = ToyTapf()
    expert_before = core.tensor_mapping_sha256(tapf.state_dict())
    with core.generic_expert_mean(tapf) as expert_report:
        expert_during = core.tensor_mapping_sha256(tapf.state_dict())
        expert_applied = expert_report["all_banks_mean_exact"]
    expert_after = core.tensor_mapping_sha256(tapf.state_dict())

    tokens = torch.zeros(2, 3, 4, dtype=torch.float64)
    bypass_reports = {}
    for arm in ("bypass_router0", "bypass_router1", "all_router_bypass"):
        state0 = {"gate_deltas": []}
        with core.bypass_routers(tapf, core.BYPASS_BANKS[arm]) as report:
            outputs = [
                tapf.apply_gate(bank, tokens, (1, 3), state0)
                for bank in (0, 1)
            ]
        bypass_reports[arm] = {
            "calls": list(report["calls"]),
            "bypassed": list(report["bypassed"]),
            "restored_exact": report["restored_exact"],
            "bank0_identity": torch.equal(outputs[0], tokens),
            "bank1_identity": torch.equal(outputs[1], tokens),
        }
    original_after_patch = tapf.apply_gate(0, tokens, (1, 3), {"gate_deltas": []})

    metrics = {
        arm: {"mAP": 0.598}
        for arm in core.ARM_ORDER
    }
    metrics["correct"] = {"mAP": 0.600}
    metrics["bypass_router0"] = {"mAP": 0.5988}
    metrics["bypass_router1"] = {"mAP": 0.5987}
    metrics["all_router_bypass"] = {"mAP": 0.5985}
    deltas = {
        arm: {
            "finite": True,
            "mean_l2": 0.2,
            "max_abs": 0.1,
            "exact_equal_rows": 0,
            "rows": 4,
        }
        for arm in core.ARM_ORDER[1:]
    }
    validity = {"contract": True, "restore": True}
    positive = core.adjudicate(metrics, deltas, validity)

    semantic_fail_metrics = json.loads(json.dumps(metrics))
    semantic_fail_metrics["wrong_rgb_evidence"]["mAP"] = 0.5995
    semantic_fail = core.adjudicate(
        semantic_fail_metrics,
        deltas,
        validity,
    )
    route_fail_metrics = json.loads(json.dumps(metrics))
    route_fail_metrics["all_router_bypass"]["mAP"] = 0.5995
    route_fail = core.adjudicate(route_fail_metrics, deltas, validity)
    inactive_deltas = json.loads(json.dumps(deltas))
    inactive_deltas["orthogonal_evidence"]["mean_l2"] = 0.0
    inactive_deltas["orthogonal_evidence"]["max_abs"] = 0.0
    inactive_fail = core.adjudicate(metrics, inactive_deltas, validity)
    validity_fail = core.adjudicate(
        metrics,
        deltas,
        {"contract": True, "restore": False},
    )

    scalar_hash = core.tensor_mapping_sha256(
        {"scalar": torch.tensor(7, dtype=torch.int64)}
    )
    sources = source_contract(inspected_paths)
    checks = {
        "cuda_uninitialized_before": not cuda_before,
        "cuda_uninitialized_after": not torch.cuda.is_initialized(),
        "donor_repeat_exact": torch.equal(donor, donor_repeat),
        "donor_no_fixed_points": donor_summary["no_fixed_points"],
        "donor_different_pid_exact": donor_summary["different_pid_fraction"] == 1.0,
        "donor_same_camera_exact": donor_summary["same_camera_fraction"] == 1.0,
        "donor_same_split_exact": donor_summary["same_split_fraction"] == 1.0,
        "donor_chunk2_exact": torch.equal(chunk_outputs["2"], expected_wrong),
        "donor_chunk5_exact": torch.equal(chunk_outputs["5"], expected_wrong),
        "donor_chunk16_exact": torch.equal(chunk_outputs["16"], expected_wrong),
        "zero_evidence_exact": bool((zero["consumer_evidence"] == 0).all()),
        "zero_non_target_exact": torch.equal(zero["sentinel"], base["sentinel"]),
        "slot_cycle_exact": torch.equal(
            slot_cycle["consumer_evidence"],
            base["consumer_evidence"].roll(-1, 1),
        ),
        "slot_cycle_mask_untouched": torch.equal(
            slot_cycle["consumer_mask"], base["consumer_mask"]
        ),
        "binding_mask_exact": torch.equal(
            binding_cycle["consumer_mask"],
            base["consumer_mask"].roll(-1, 1),
        ),
        "binding_presence_exact": torch.equal(
            binding_cycle["consumer_presence"],
            base["consumer_presence"].roll(-1, 1),
        ),
        "binding_evidence_untouched": torch.equal(
            binding_cycle["consumer_evidence"], base["consumer_evidence"]
        ),
        "binding_field_recomputed": torch.equal(
            binding_cycle["consumer_field"],
            binding_cycle["consumer_mask"]
            * binding_cycle["consumer_presence"][..., None, None],
        ),
        "orthogonal_repeat_exact": torch.equal(orthogonal, orthogonal_repeat),
        "orthogonal_global_rng_exact": torch.equal(rng_before, rng_after),
        "orthogonal_nonidentity": not torch.equal(orthogonal, identity),
        "orthogonal_gram": gram_error <= 1e-12,
        "orthogonal_norm": norm_error <= 1e-12,
        "orthogonal_cosine": cosine_error <= 1e-12,
        "expert_mean_applied": expert_applied,
        "expert_mean_state_changed": expert_during != expert_before,
        "expert_mean_restored": expert_report["restored_exact"]
        and expert_after == expert_before,
        "bypass0_contract": (
            bypass_reports["bypass_router0"]["calls"] == [1, 1]
            and bypass_reports["bypass_router0"]["bypassed"] == [1, 0]
            and bypass_reports["bypass_router0"]["bank0_identity"]
            and not bypass_reports["bypass_router0"]["bank1_identity"]
            and bypass_reports["bypass_router0"]["restored_exact"]
        ),
        "bypass1_contract": (
            bypass_reports["bypass_router1"]["calls"] == [1, 1]
            and bypass_reports["bypass_router1"]["bypassed"] == [0, 1]
            and not bypass_reports["bypass_router1"]["bank0_identity"]
            and bypass_reports["bypass_router1"]["bank1_identity"]
            and bypass_reports["bypass_router1"]["restored_exact"]
        ),
        "all_bypass_contract": (
            bypass_reports["all_router_bypass"]["calls"] == [1, 1]
            and bypass_reports["all_router_bypass"]["bypassed"] == [1, 1]
            and bypass_reports["all_router_bypass"]["bank0_identity"]
            and bypass_reports["all_router_bypass"]["bank1_identity"]
            and bypass_reports["all_router_bypass"]["restored_exact"]
        ),
        "bypass_original_restored": torch.equal(
            original_after_patch, tokens + 1.0
        ),
        "adjudicator_positive_pass": positive[
            "phase_b_formal_mechanism_design_authorized"
        ],
        "adjudicator_semantic_fail": not semantic_fail[
            "phase_b_formal_mechanism_design_authorized"
        ],
        "adjudicator_route_fail": not route_fail[
            "phase_b_formal_mechanism_design_authorized"
        ],
        "adjudicator_inactive_fail": not inactive_fail[
            "phase_b_formal_mechanism_design_authorized"
        ],
        "adjudicator_validity_fail": (
            validity_fail["status"] == "FAIL"
            and not validity_fail[
                "phase_b_formal_mechanism_design_authorized"
            ]
        ),
        "scalar_tensor_hash_supported": len(scalar_hash) == 64,
        "source_ast_clean": not sources["findings"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "verdict": (
            "EXP402_STATIC_CPU_PASS"
            if all(checks.values())
            else "EXP402_STATIC_CPU_FAIL"
        ),
        "checks": checks,
        "check_count": len(checks),
        "donor_summary": donor_summary,
        "orthogonal": {
            "gram_max_abs": gram_error,
            "norm_max_abs": norm_error,
            "cosine_max_abs": cosine_error,
            "sha256": core.tensor_mapping_sha256({"q": orthogonal}),
        },
        "expert_report": expert_report,
        "bypass_reports": bypass_reports,
        "adjudicator": {
            "positive": positive,
            "semantic_fail": semantic_fail,
            "route_fail": route_fail,
            "inactive_fail": inactive_fail,
            "validity_fail": validity_fail,
        },
        "sources": sources,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--audit")
    parser.add_argument("--result", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    core_path = Path(args.core).resolve()
    inspected = [core_path]
    if args.audit:
        inspected.append(Path(args.audit).resolve())
    result_path = Path(args.result).resolve()
    if result_path.exists() or result_path.with_suffix(
        result_path.suffix + ".tmp"
    ).exists():
        raise FileExistsError("Static result path must be fresh")
    core = load_module(core_path)
    result = run_contract(core, inspected)
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
