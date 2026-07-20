#!/usr/bin/env python3
"""CPU/static positive and mutant contract for exp406 donor reserve."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch


EXPECTED_SAMPLES = 15618
CORE_SAMPLES = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*values) -> int:
    encoded = "\0".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp406_donor_contract_target", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load donor reserve module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def function_dump(source: str, name: str) -> str:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError("function projection is not unique: %s" % name)
    return ast.dump(matches[0], annotate_fields=True, include_attributes=False)


def constant_value(source: str, name: str):
    tree = ast.parse(source)
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        matches.append(("literal", ast.literal_eval(node.value)))
                    except (ValueError, TypeError):
                        matches.append((
                            "ast",
                            ast.dump(
                                node.value,
                                annotate_fields=True,
                                include_attributes=False,
                            ),
                        ))
    if len(matches) != 1:
        raise RuntimeError("constant projection is not unique: %s" % name)
    return matches[0]


def synthetic_inputs(module):
    records = [
        ("/mnt1/afrdata/fake/%05d.jpg" % index, index + 1000, 0, 0)
        for index in range(EXPECTED_SAMPLES)
    ]
    paths = ["fake/%05d.jpg" % index for index in range(EXPECTED_SAMPLES)]
    core = list(range(CORE_SAMPLES))
    plan = module.build_donor_plan(
        records,
        paths,
        core,
        stable_digest=stable_digest,
        expected_samples=EXPECTED_SAMPLES,
        core_samples=CORE_SAMPLES,
    )
    recipients = list(range(20))
    slots = [position // 4 for position in range(20)]
    valid = torch.ones(EXPECTED_SAMPLES, 5, dtype=torch.bool)
    mass = torch.full((EXPECTED_SAMPLES, 5), math.e, dtype=torch.float64)
    centroid = torch.ones(EXPECTED_SAMPLES, 5, dtype=torch.float64)
    confidence = torch.ones(EXPECTED_SAMPLES, 5, dtype=torch.float64)
    support = torch.ones(EXPECTED_SAMPLES, 5, dtype=torch.float64)
    global_feature = torch.zeros(EXPECTED_SAMPLES, 2, dtype=torch.float32)
    global_feature[:, 0] = 1.0
    pids = torch.arange(EXPECTED_SAMPLES, dtype=torch.long) + 1000
    camids = torch.zeros(EXPECTED_SAMPLES, dtype=torch.long)
    keys = torch.arange(EXPECTED_SAMPLES, dtype=torch.long)
    for recipient, slot in zip(recipients, slots):
        mass[recipient, slot] = 1.0
        centroid[recipient, slot] = 0.0
        confidence[recipient, slot] = 0.0
        support[recipient, slot] = 0.0
    near_donors = list(map(int, plan["donor_order"][:20]))
    for donor, slot in zip(near_donors, slots):
        mass[donor, slot] = 1.0
        centroid[donor, slot] = 0.0
        confidence[donor, slot] = 0.0
        support[donor, slot] = 0.0
    return {
        "records": records,
        "paths": paths,
        "core": core,
        "plan": plan,
        "recipients": recipients,
        "slots": slots,
        "valid": valid,
        "mass": mass,
        "centroid": centroid,
        "confidence": confidence,
        "support": support,
        "global_feature": global_feature,
        "pids": pids,
        "camids": camids,
        "keys": keys,
        "near_donors": near_donors,
    }


def call_match(module, rows, *, plan=None, caliper=8.0, preference_limit=64):
    return module.choose_wrong_masks_progressive(
        rows["recipients"],
        rows["slots"],
        rows["valid"],
        rows["mass"],
        rows["centroid"],
        rows["confidence"],
        rows["support"],
        rows["global_feature"],
        rows["pids"],
        rows["camids"],
        rows["keys"],
        rows["recipients"],
        rows["core"],
        rows["plan"] if plan is None else plan,
        primary_caliper=caliper,
        preference_limit=preference_limit,
    )


def expect_error(callable_object, error_type, text: str) -> bool:
    try:
        callable_object()
    except error_type as error:
        return text in str(error)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--sealed-runner", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--sealed-core", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--sealed-teacher", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        key: Path(value).resolve()
        for key, value in {
            "module": args.module,
            "runner": args.runner,
            "sealed_runner": args.sealed_runner,
            "core": args.core,
            "sealed_core": args.sealed_core,
            "teacher": args.teacher,
            "sealed_teacher": args.sealed_teacher,
            "protocol": args.protocol,
        }.items()
    }
    cuda_before = torch.cuda.is_initialized()
    module = load_module(paths["module"])
    rows = synthetic_inputs(module)
    plan_repeat = module.build_donor_plan(
        rows["records"],
        rows["paths"],
        rows["core"],
        stable_digest=stable_digest,
        expected_samples=EXPECTED_SAMPLES,
        core_samples=CORE_SAMPLES,
    )
    donors, records, summary = call_match(module, rows)

    subset_plan = dict(rows["plan"])
    subset_plan["stages"] = [rows["plan"]["stages"][0]]
    subset_mutant_caught = expect_error(
        lambda: call_match(module, rows, plan=subset_plan),
        module.DonorReserveError,
        "no frozen donor prefix",
    )
    caliper_mutant_caught = expect_error(
        lambda: call_match(module, rows, caliper=8.1),
        ValueError,
        "frozen at 8.0",
    )
    preference_mutant_caught = expect_error(
        lambda: call_match(module, rows, preference_limit=128),
        ValueError,
        "frozen at 64",
    )
    hall_mutant_caught = expect_error(
        lambda: module._maximum_unique_assignment([
            [999], [999], [999], [999]
        ]),
        RuntimeError,
        "no one-to-one",
    )

    runner_source = paths["runner"].read_text(encoding="utf-8")
    sealed_runner_source = paths["sealed_runner"].read_text(encoding="utf-8")
    protocol_source = paths["protocol"].read_text(encoding="utf-8")
    formal_functions = (
        "full_semantic_summary",
        "choose_wrong_masks",
        "maximum_unique_assignment",
        "maximum_unique_assignment_with_expansion",
        "pid_bootstrap_interval",
        "conservative_two_cluster_summary",
        "non_torso_macro_effect",
        "non_torso_two_cluster_summary",
        "adjudicate_measurement",
        "slot_semantics",
    )
    formal_constants = (
        "EXPECTED_SAMPLES",
        "FORMAL_DIAGNOSTIC_SAMPLES",
        "SAMPLES_PER_SLOT",
        "VIEW_SEED",
        "BOOTSTRAP_SEED",
        "BOOTSTRAP_REPEATS",
        "DELETION_FRACTIONS",
        "MAX_NO_TARGET_FRACTION",
        "MIN_TARGET_PID_FRACTION",
        "MATCH_PRIMARY_CALIPER",
        "MATCH_PREFERENCE_LIMIT",
    )
    formal_projection_exact = all(
        function_dump(runner_source, name)
        == function_dump(sealed_runner_source, name)
        for name in formal_functions
    ) and all(
        constant_value(runner_source, name)
        == constant_value(sealed_runner_source, name)
        for name in formal_constants
    )

    namespace_exact = bool(
        'PREFLIGHT_EXECUTION = "exp406-p0b-preflight-v1"' in runner_source
        and 'FORMAL_EXECUTION = "exp406-p0b-iso-teacher-v1"' in runner_source
        and '"experiment": "exp406"' in runner_source
        and "exp405-p0b-preflight-v1" not in runner_source
        and "exp405-p0b-iso-teacher-v1" not in runner_source
        and "exp406-p0b-preflight-v1" in protocol_source
    )
    preflight_formal_separation = bool(
        "execution_indices = list(range(EXPECTED_SAMPLES))" in runner_source
        and "choose_wrong_masks_progressive" in runner_source
        and "if formal:\n        wrong_mask_indices, wrong_mask_records = choose_wrong_masks(" in runner_source
        and '"donor_reserve_contract_strict"' in runner_source
        and '"scientific_evaluated": False' in runner_source
        and '"exp406-p0b-preflight-cache-v1"' in runner_source
        and "failure_payload[\"diagnostics\"]" in runner_source
        and 'parser.add_argument("--asset-manifest", required=True)' in runner_source
        and 'parser.add_argument("--asset-manifest-sha256", required=True)' in runner_source
        and 'FRESH_ASSET_ROOT = Path("/home/afr/reid-clean/assets/exp406-p0b-preflight-v1")' in runner_source
        and "fresh asset manifest content mismatch" in runner_source
        and "fresh asset manifest changed during teacher measurement" in runner_source
    )

    gates = {
        "source_hashes_bound": all(path.is_file() for path in paths.values()),
        "core_teacher_exact_sealed_projection": bool(
            paths["core"].read_bytes() == paths["sealed_core"].read_bytes()
            and paths["teacher"].read_bytes() == paths["sealed_teacher"].read_bytes()
        ),
        "formal_scientific_projection_exact": formal_projection_exact,
        "fresh_namespace_and_receipt_binding": namespace_exact,
        "preflight_formal_cache_and_science_separated": preflight_formal_separation,
        "donor_plan_deterministic_and_complete": bool(
            rows["plan"]["plan_sha256"] == plan_repeat["plan_sha256"]
            and rows["plan"]["pool_totals"]
            == [512, 1024, 2048, 4096, 8192, 15618]
            and len(rows["plan"]["stages"][-1]["pool_indices"])
            == EXPECTED_SAMPLES
        ),
        "fixed_core_scale_progressive_positive": bool(
            summary["contract_strict"]
            and summary["selected_pool_total"] == 1024
            and summary["stages"][0]["status"] == "ZERO_EDGE"
            and summary["stages"][1]["status"] == "MATCHED"
            and all(
                field["scale_floor_applied"]
                for slot in summary["scales"].values()
                for field in slot.values()
            )
        ),
        "assignment_identity_camera_caliper_strict": bool(
            len(donors) == len(set(donors)) == 20
            and not set(donors).intersection(rows["recipients"])
            and all(record["recipient_pid"] != record["donor_pid"] for record in records)
            and all(record["camera"] == 0 for record in records)
            and all(record["primary_distance"] <= 8.0 for record in records)
        ),
        "subset_only_mutant_caught": subset_mutant_caught,
        "caliper_and_preference_mutants_caught": bool(
            caliper_mutant_caught and preference_mutant_caught
        ),
        "hall_assignment_mutant_caught": hall_mutant_caught,
        "failure_diagnostics_complete": bool(
            all(
                key in summary
                for key in (
                    "scales", "stages", "pool_totals", "selected_pool_total",
                    "selected_preference_limit", "summary_sha256",
                )
            )
            and all(
                "recipient_rows" in stage and "status" in stage
                for stage in summary["stages"]
            )
        ),
        "cuda_not_initialized": bool(
            not cuda_before and not torch.cuda.is_initialized()
        ),
    }
    result = {
        "experiment": "exp406",
        "schema": "exp406-donor-reserve-static-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "positive": {
            "selected_pool_total": summary["selected_pool_total"],
            "recipient_count": summary["recipient_count"],
            "assigned_donor_count": summary["assigned_donor_count"],
            "summary_sha256": summary["summary_sha256"],
        },
        "mutants": {
            "subset_only": subset_mutant_caught,
            "caliper_relaxation": caliper_mutant_caught,
            "preference_change": preference_mutant_caught,
            "hall_failure": hall_mutant_caught,
        },
        "provenance": {
            key + "_sha256": sha256_file(path)
            for key, path in paths.items()
        },
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("output already exists")
    output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("%s %d / %d" % (
        result["status"], sum(gates.values()), len(gates)
    ))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
