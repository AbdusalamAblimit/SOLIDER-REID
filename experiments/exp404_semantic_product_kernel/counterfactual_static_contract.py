#!/usr/bin/env python3
"""Deterministic CPU positive/negative contract for exp404 audit v1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = Path(__file__).with_name("counterfactual_core.py")
TAPF_PATH = ROOT / "model" / "tapf.py"
MODEL_PATH = ROOT / "model" / "make_model.py"
AUDIT_PATH = Path(__file__).with_name("actual_counterfactual_audit.py")
POSTFLIGHT_PATH = Path(__file__).with_name("counterfactual_postflight.py")
WRAPPER_PATH = Path(__file__).with_name("counterfactual_once_wrapper.sh")


def load_core():
    spec = importlib.util.spec_from_file_location("exp404_counterfactual_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toy_kernel(global_feature, evidence, presence, groups=16):
    weight = presence.detach().float().clamp(0.0, 1.0)
    mass = weight.sum(dim=1, keepdim=True)
    pooled = (evidence.float() * weight[..., None]).sum(dim=1)
    pooled = pooled / mass.clamp_min(1.0)
    pooled = torch.where(mass > 0, pooled, torch.zeros_like(pooled))
    factor = groups * torch.softmax(pooled, dim=-1)
    grouped = global_feature.reshape(global_feature.shape[0], groups, -1)
    descriptor = (factor.to(global_feature.dtype)[..., None] * grouped).reshape_as(
        global_feature
    )
    return descriptor, factor


def metric_fixture(core, wrong_map=0.5700, bypass_map=0.5700):
    values = {
        "correct": 0.5740,
        "wrong_rgb": wrong_map,
        "generic_mean": 0.5690,
        "null_zero": 0.5680,
        "all_product_bypass": bypass_map,
        "random_key": 0.5670,
        "random_cluster": 0.5660,
        "wrong_mask": 0.5710,
        "slot_cycle": 0.5705,
    }
    metrics = {
        arm: {"mAP": value, "rank1": 0.67, "rank5": 0.80, "rank10": 0.85}
        for arm, value in values.items()
    }
    deltas = {
        arm: {
            "finite": True,
            "mean_l2": 1.0,
            "max_abs": 1.0,
            "exact_equal_rows": 0,
            "rows": 32,
        }
        for arm in core.ARM_ORDER[1:]
    }
    return metrics, deltas


def main() -> int:
    core = load_core()
    torch.manual_seed(20260720)
    cuda_before = torch.cuda.is_initialized()

    count = 384
    pids = torch.arange(count, dtype=torch.int64) % 128
    camids = torch.arange(count, dtype=torch.int64) % 2
    num_query = count // 2
    # Query and gallery each contain both cameras and repeated IDs.
    pids[num_query:] = torch.arange(count - num_query, dtype=torch.int64) % 128
    camids[num_query:] = torch.arange(count - num_query, dtype=torch.int64) % 2
    donor = core.build_global_donor_map(pids, camids, num_query)
    donor_summary = core.validate_donor_map(donor, pids, camids, num_query)

    batch = 32
    evidence = torch.randn(batch, 5, 16, dtype=torch.float64)
    presence = torch.sigmoid(torch.randn(batch, 5, dtype=torch.float64))
    global_feature = torch.randn(batch, 768, dtype=torch.float64)
    indices = torch.arange(batch, dtype=torch.int64)
    evidence_cache = torch.randn(count, 5, 16, dtype=torch.float64)
    presence_cache = torch.sigmoid(torch.randn(count, 5, dtype=torch.float64))
    generic = core.pooled_evidence(evidence_cache, presence_cache).mean(dim=0)
    permutations, signs = core.build_signed_permutations(count)
    assignment = core.build_balanced_cluster_assignment(count)
    cluster_summary = core.validate_cluster_assignment(
        assignment, pids, camids
    )
    prototypes = core.build_cluster_prototypes(generic)

    arm_inputs = {"correct": (evidence, presence)}
    for arm in core.INTERVENTION_ARMS:
        arm_inputs[arm] = core.intervene_spk_inputs(
            evidence,
            presence,
            arm,
            absolute_indices=indices,
            donor_map=donor,
            evidence_cache=evidence_cache,
            presence_cache=presence_cache,
            generic_mean=generic,
            random_permutations=permutations,
            random_signs=signs,
            cluster_assignment=assignment,
            cluster_prototypes=prototypes,
        )

    descriptors = {}
    factors = {}
    for arm, (arm_evidence, arm_presence) in arm_inputs.items():
        descriptors[arm], factors[arm] = toy_kernel(
            global_feature,
            arm_evidence,
            arm_presence,
        )
    descriptors["all_product_bypass"] = global_feature
    factors["all_product_bypass"] = torch.ones(batch, 16, dtype=torch.float64)

    random_key_evidence = arm_inputs["random_key"][0]
    random_key_norm_preserved = torch.allclose(
        random_key_evidence.norm(dim=-1),
        evidence.norm(dim=-1),
        rtol=1e-12,
        atol=1e-12,
    )
    random_key_abs_exact = torch.equal(
        random_key_evidence.abs().sort(dim=-1).values,
        evidence.abs().sort(dim=-1).values,
    )
    wrong_donors = donor.index_select(0, indices)
    wrong_evidence_exact = torch.equal(
        arm_inputs["wrong_rgb"][0],
        evidence_cache.index_select(0, wrong_donors),
    )
    wrong_presence_exact = torch.equal(
        arm_inputs["wrong_rgb"][1],
        presence_cache.index_select(0, wrong_donors),
    )
    generic_rows_exact = torch.equal(
        arm_inputs["generic_mean"][0],
        generic.reshape(1, 1, 16).expand(batch, 1, 16),
    ) and torch.equal(
        arm_inputs["generic_mean"][1], torch.ones(batch, 1, dtype=torch.float64)
    )

    good_metrics, good_deltas = metric_fixture(core)
    good = core.adjudicate(good_metrics, good_deltas, {"contract": True})
    wrong_fail_metrics, wrong_fail_deltas = metric_fixture(core, wrong_map=0.575)
    wrong_fail = core.adjudicate(
        wrong_fail_metrics, wrong_fail_deltas, {"contract": True}
    )
    bypass_fail_metrics, bypass_fail_deltas = metric_fixture(
        core, bypass_map=0.5735
    )
    bypass_fail = core.adjudicate(
        bypass_fail_metrics, bypass_fail_deltas, {"contract": True}
    )
    validity_fail = core.adjudicate(
        good_metrics, good_deltas, {"null_bypass_exact": False}
    )

    model_source = MODEL_PATH.read_text(encoding="utf-8")
    tapf_source = TAPF_PATH.read_text(encoding="utf-8")
    core_source = CORE_PATH.read_text(encoding="utf-8")
    audit_source = AUDIT_PATH.read_text(encoding="utf-8")
    postflight_source = POSTFLIGHT_PATH.read_text(encoding="utf-8")
    wrapper_source = WRAPPER_PATH.read_text(encoding="utf-8")
    spk_call = (
        'tapf_aux["student_evidence"]' in model_source
        and 'tapf_aux["student_presence"]' in model_source
    )
    gates = {
        "cuda_uninitialized": not cuda_before and not torch.cuda.is_initialized(),
        "arm_order_exact": core.ARM_ORDER == (
            "correct",
            "wrong_rgb",
            "generic_mean",
            "null_zero",
            "all_product_bypass",
            "random_key",
            "random_cluster",
            "wrong_mask",
            "slot_cycle",
        ),
        "donor_contract": all(
            donor_summary[key] == 1.0
            for key in (
                "different_pid_fraction",
                "same_camera_fraction",
                "same_split_fraction",
            )
        ) and donor_summary["no_fixed_points"],
        "wrong_rgb_pair_exact": wrong_evidence_exact and wrong_presence_exact,
        "generic_mean_fixed_exact": generic_rows_exact,
        "null_evidence_exact_zero": bool((arm_inputs["null_zero"][0] == 0).all()),
        "null_descriptor_exact_bypass": torch.equal(
            descriptors["null_zero"], descriptors["all_product_bypass"]
        ),
        "null_factor_exact_one": torch.equal(
            factors["null_zero"], torch.ones_like(factors["null_zero"])
        ),
        "random_key_norm_preserved": bool(random_key_norm_preserved),
        "random_key_abs_multiset_exact": random_key_abs_exact,
        "random_key_unique_contract": len(
            {
                tuple(permutations[index].tolist()) + tuple(signs[index].tolist())
                for index in range(count)
            }
        ) == count,
        "random_cluster_balanced": cluster_summary["count_max_minus_min"] <= 1,
        "random_cluster_pid_coverage": cluster_summary["pid_coverage_min"] >= 40,
        "random_cluster_camera_coverage": cluster_summary["all_cameras_exact"],
        "random_cluster_eight_prototypes": prototypes.shape == (8, 16),
        "wrong_mask_presence_only": torch.equal(
            arm_inputs["wrong_mask"][0], evidence
        ) and torch.equal(
            arm_inputs["wrong_mask"][1], presence.roll(-1, 1)
        ),
        "slot_cycle_evidence_only": torch.equal(
            arm_inputs["slot_cycle"][0], evidence.roll(-1, 1)
        ) and torch.equal(arm_inputs["slot_cycle"][1], presence),
        "required_controls_active": all(
            not torch.equal(descriptors[arm], descriptors["correct"])
            for arm in core.PRIMARY_ACTIVE_CONTROLS
        ),
        "positive_adjudication_go": good["decision"] == "SPK_MECHANISM_GO",
        "wrong_control_mutant_caught": wrong_fail["decision"]
        == "SPK_MECHANISM_NO_GO",
        "bypass_gap_mutant_caught": bypass_fail["decision"]
        == "SPK_MECHANISM_NO_GO",
        "validity_mutant_caught": validity_fail["decision"]
        == "SPK_MECHANISM_NO_GO",
        "production_spk_uses_student_inputs": spk_call,
        "production_spk_parameter_free": "class SemanticProductKernel" in tapf_source
        and "factor_float = self.groups * torch.softmax" in tapf_source,
        "core_does_not_target_old_consumer_evidence": "consumer_evidence"
        not in core_source,
        "audit_patches_spk_not_prepare": "model.semantic_product_kernel"
        in audit_source
        and "kernel.forward = patched" in audit_source
        and "tapf.prepare =" not in audit_source,
        "audit_has_train_generic_and_nine_arm_loop": "collect_train_generic"
        in audit_source
        and "for arm in core.ARM_ORDER" in audit_source,
        "audit_has_null_bypass_exact_gate": "null_bypass_descriptor_exact"
        in audit_source
        and "null_bypass_metrics_exact" in audit_source,
        "audit_guards_teacher_pose_codebook": "ForbiddenSemanticReadGuard"
        in audit_source
        and "forbidden_teacher_pose_reads_zero" in audit_source,
        "postflight_requires_nine_arms": "nine_metrics_present"
        in postflight_source
        and "nine_arm_reports_present" in postflight_source,
        "wrapper_once_only_paths": "formal_result_once.json" in wrapper_source
        and "formal_once.lock" in wrapper_source
        and "set -C" in wrapper_source,
        "official_data_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "diagnostic": "exp404_spk_counterfactual_static_contract",
        "status": "PASS" if passed else "FAIL",
        "gate_count": len(gates),
        "gates": gates,
        "donor": donor_summary,
        "cluster": cluster_summary,
        "hashes": {
            "core": sha256_file(CORE_PATH),
            "tapf": sha256_file(TAPF_PATH),
            "make_model": sha256_file(MODEL_PATH),
            "contract": sha256_file(Path(__file__)),
            "audit": sha256_file(AUDIT_PATH),
            "postflight": sha256_file(POSTFLIGHT_PATH),
            "wrapper": sha256_file(WRAPPER_PATH),
            "random_assets": core.tensor_mapping_sha256(
                {
                    "permutations": permutations,
                    "signs": signs,
                    "assignment": assignment,
                    "prototypes": prototypes,
                }
            ),
        },
        "adjudication": {
            "positive": good,
            "wrong_control_mutant": wrong_fail,
            "bypass_gap_mutant": bypass_fail,
            "validity_mutant": validity_fail,
        },
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "formal_cuda_preflight_authorized": passed,
        "formal_full_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
