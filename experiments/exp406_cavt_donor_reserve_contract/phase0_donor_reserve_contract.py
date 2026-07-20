#!/usr/bin/env python3
"""CPU/static positive and mutant contract for exp406 donor reserve."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import types
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load donor reserve module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clone_rows(rows):
    cloned = {}
    for key, value in rows.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def set_slot_descriptor(rows, index: int, slot: int, *, mass_log: float = 0.0):
    rows["mass"][int(index), int(slot)] = math.exp(float(mass_log))
    rows["centroid"][int(index), int(slot)] = 0.0
    rows["confidence"][int(index), int(slot)] = 0.0
    rows["support"][int(index), int(slot)] = 0.0


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


def capture_error(callable_object, error_type):
    try:
        callable_object()
    except error_type as error:
        return error
    return None


def selector_core_only_contract(module, rows) -> bool:
    targets = torch.arange(EXPECTED_SAMPLES, dtype=torch.long) % 5
    selected, slots = module.choose_diagnostic_subset_from_core(
        targets,
        torch.ones(EXPECTED_SAMPLES, 5, dtype=torch.bool),
        rows["pids"].tolist(),
        rows["paths"],
        rows["core"],
        stable_digest=stable_digest,
        samples_per_slot=4,
    )
    escaped_rejected = expect_error(
        lambda: module.choose_wrong_masks_progressive(
            selected[:-1] + [CORE_SAMPLES],
            slots,
            rows["valid"], rows["mass"], rows["centroid"],
            rows["confidence"], rows["support"], rows["global_feature"],
            rows["pids"], rows["camids"], rows["keys"],
            selected[:-1] + [CORE_SAMPLES], rows["core"], rows["plan"],
            primary_caliper=8.0, preference_limit=64,
        ),
        ValueError,
        "recipient escaped frozen core",
    )
    return bool(
        len(selected) == len(set(selected)) == 20
        and set(selected).issubset(set(rows["core"]))
        and slots == [position // 4 for position in range(20)]
        and escaped_rejected
    )


def eligibility_contract(module, base_rows):
    rows = clone_rows(base_rows)
    ordered = list(map(int, rows["plan"]["donor_order"][:100]))
    for donor in ordered:
        for slot in range(5):
            rows["mass"][donor, slot] = math.e
            rows["centroid"][donor, slot] = 1.0
            rows["confidence"][donor, slot] = 1.0
            rows["support"][donor, slot] = 1.0
    groups = [ordered[offset:offset + 20] for offset in range(0, 100, 20)]
    for recipient, slot in zip(rows["recipients"], rows["slots"]):
        rows["pids"][recipient] = 2000 + int(slot)
    for position, (recipient, slot) in enumerate(zip(rows["recipients"], rows["slots"])):
        wrong_camera, same_pid, invalid, over_caliper, correct = (
            group[position] for group in groups
        )
        set_slot_descriptor(rows, wrong_camera, slot)
        rows["camids"][wrong_camera] = 1
        set_slot_descriptor(rows, same_pid, slot)
        rows["pids"][same_pid] = rows["pids"][recipient]
        set_slot_descriptor(rows, invalid, slot)
        rows["valid"][invalid, slot] = False
        set_slot_descriptor(rows, over_caliper, slot, mass_log=8.1e-6)
        set_slot_descriptor(rows, correct, slot, mass_log=7.9e-6)
        rows["global_feature"][over_caliper] = torch.tensor([1.0, 0.0])
        rows["global_feature"][correct] = torch.tensor([-1.0, 0.0])
    donors, records, summary = call_match(module, rows)
    return bool(
        set(map(int, donors)) == set(groups[-1])
        and summary["contract_strict"]
        and all(record["camera"] == 0 for record in records)
        and all(record["recipient_pid"] != record["donor_pid"] for record in records)
        and all(record["primary_distance"] <= 8.0 for record in records)
    )


class CaliperFilterMutator(ast.NodeTransformer):
    def __init__(self):
        self.in_target = False
        self.mutation_count = 0

    def visit_FunctionDef(self, node):
        previous = self.in_target
        self.in_target = node.name == "_stage_preferences"
        node = self.generic_visit(node)
        self.in_target = previous
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if (
            self.in_target
            and isinstance(node.left, ast.Name)
            and node.left.id == "primary_distance"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.LtE)
        ):
            node.comparators = [ast.Constant(value=1e300)]
            self.mutation_count += 1
        return node


def load_mutated_module(tree, name: str):
    module = types.ModuleType(name)
    module.__file__ = "<%s>" % name
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(tree), module.__file__, "exec"), module.__dict__)
    return module


def caliper_delete_mutant_contract(module_source: str, base_rows) -> dict:
    tree = ast.parse(module_source)
    transformer = CaliperFilterMutator()
    mutant_tree = transformer.visit(tree)
    mutant = load_mutated_module(mutant_tree, "exp406_caliper_delete_mutant")
    try:
        mutant_passed = eligibility_contract(mutant, base_rows)
    except (RuntimeError, ValueError, mutant.DonorReserveError):
        mutant_passed = False
    return {
        "mutation_site_count_exact": transformer.mutation_count == 1,
        "mutant_caught": not mutant_passed,
    }


def preference_order_contract(module, base_rows) -> bool:
    rows = clone_rows(base_rows)
    recipient = rows["recipients"][0]
    slot = rows["slots"][0]
    donors = list(map(int, rows["plan"]["donor_order"][:4]))
    rows["mass"].fill_(math.e)
    rows["centroid"].fill_(1.0)
    rows["confidence"].fill_(1.0)
    rows["support"].fill_(1.0)
    set_slot_descriptor(rows, recipient, slot)
    specifications = (
        (donors[0], 2.0e-6, 0.0, 40),
        (donors[1], 1.0e-6, 2.0, 30),
        (donors[2], 2.0e-6, 0.0, 20),
        (donors[3], 2.0e-6, 0.0, 10),
    )
    for donor, primary, cosine_gap, key in specifications:
        set_slot_descriptor(rows, donor, slot, mass_log=primary)
        rows["global_feature"][donor] = torch.tensor(
            [-1.0, 0.0] if cosine_gap == 2.0 else [1.0, 0.0]
        )
        rows["keys"][donor] = key
    values = module._descriptor_values(
        rows["mass"], rows["centroid"], rows["confidence"], rows["support"]
    )
    scales, _ = module._frozen_scale_summary(rows["valid"], values, rows["core"])
    forbidden = torch.zeros(EXPECTED_SAMPLES, dtype=torch.bool)
    forbidden[torch.tensor(rows["recipients"], dtype=torch.long)] = True
    preferences, _, _ = module._stage_preferences(
        [recipient], [slot], rows["valid"], values, scales,
        rows["global_feature"], rows["pids"], rows["camids"], rows["keys"],
        forbidden, rows["plan"]["stages"][1]["pool_indices"], 8.0,
    )
    observed = [int(value) for value in preferences[0] if int(value) in donors]
    expected = [donors[3], donors[2], donors[0], donors[1]]
    return observed == expected


def fixed_core_scale_contract(module, base_rows):
    rows = clone_rows(base_rows)
    rows["mass"].fill_(1.0)
    rows["centroid"].zero_()
    rows["confidence"].zero_()
    rows["support"].zero_()
    rows["camids"].fill_(1)
    rows["camids"][rows["recipients"]] = 0
    for position, index in enumerate(rows["core"]):
        if index in rows["recipients"]:
            value = 0.0
        else:
            value = -1.0 if position % 2 == 0 else 1.0
        rows["mass"][index, :] = math.exp(value)
    for position, index in enumerate(rows["plan"]["donor_order"]):
        value = -100.0 if position % 2 == 0 else 100.0
        rows["mass"][index, :] = math.exp(value)
    for donor, slot in zip(rows["near_donors"], rows["slots"]):
        rows["camids"][donor] = 0
        set_slot_descriptor(rows, donor, slot, mass_log=9.0)
    error = capture_error(lambda: call_match(module, rows), module.DonorReserveError)
    if error is None:
        return False
    scale_exact = all(
        abs(fields["mass_log"]["raw_mad"] - 1.0) < 1e-12
        and abs(fields["mass_log"]["scale"] - 1.0) < 1e-12
        for fields in error.diagnostics["scales"].values()
    )

    near_zero = clone_rows(base_rows)
    near_zero["mass"].fill_(1.0)
    near_zero["centroid"].zero_()
    near_zero["confidence"].zero_()
    near_zero["support"].zero_()
    for position, index in enumerate(near_zero["core"]):
        value = -1e-12 if position % 2 == 0 else 1e-12
        near_zero["mass"][index, :] = math.exp(value)
    values = module._descriptor_values(
        near_zero["mass"], near_zero["centroid"],
        near_zero["confidence"], near_zero["support"],
    )
    _, scale_summary = module._frozen_scale_summary(
        near_zero["valid"], values, near_zero["core"]
    )
    floor_exact = all(
        fields["mass_log"]["raw_mad"] < 1e-6
        and fields["mass_log"]["scale"] == 1e-6
        and fields["mass_log"]["scale_floor_applied"]
        for fields in scale_summary.values()
    )
    return bool(scale_exact and floor_exact)


def tampered_plan_contract(module, rows) -> dict:
    cases = {}
    truncated = copy.deepcopy(rows["plan"])
    truncated["stages"] = truncated["stages"][:1]
    cases["subset_only"] = truncated

    reordered = copy.deepcopy(rows["plan"])
    reordered["stages"][2], reordered["stages"][-1] = (
        reordered["stages"][-1], reordered["stages"][2]
    )
    cases["stage_reorder"] = reordered

    nonprefix = copy.deepcopy(rows["plan"])
    pool = nonprefix["stages"][1]["pool_indices"]
    pool[-1] = nonprefix["donor_order"][700]
    cases["nonprefix_substitution"] = nonprefix

    hash_drift = copy.deepcopy(rows["plan"])
    hash_drift["stages"][1]["pool_indices_sha256"] = "0" * 64
    cases["stage_hash_drift"] = hash_drift

    extra_stage = copy.deepcopy(rows["plan"])
    extra_stage["pool_totals"].insert(-1, 12000)
    extra_stage["stages"].insert(-1, copy.deepcopy(extra_stage["stages"][-2]))
    extra_stage["stages"][-2]["pool_total"] = 12000
    cases["dynamic_stage"] = extra_stage
    return {
        name: expect_error(
            lambda plan=plan: call_match(module, rows, plan=plan),
            ValueError,
            "donor plan",
        )
        for name, plan in cases.items()
    }


def hall_failure_contract(module, base_rows):
    rows = clone_rows(base_rows)
    rows["mass"].fill_(math.e)
    rows["centroid"].fill_(1.0)
    rows["confidence"].fill_(1.0)
    rows["support"].fill_(1.0)
    rows["camids"].fill_(1)
    for recipient, slot in zip(rows["recipients"], rows["slots"]):
        set_slot_descriptor(rows, recipient, slot)
    rows["camids"][rows["recipients"]] = 1
    rows["camids"][rows["recipients"][:2]] = 0
    ordered = list(map(int, rows["plan"]["donor_order"][:301]))
    shared = ordered[0]
    rows["camids"][shared] = 0
    set_slot_descriptor(rows, shared, 0)
    for donor in ordered[1:]:
        rows["camids"][donor] = 1
        for slot in range(5):
            set_slot_descriptor(rows, donor, slot)
    error = capture_error(lambda: call_match(module, rows), module.DonorReserveError)
    if error is None:
        return False, None
    diagnostics = error.diagnostics
    module.validate_failure_diagnostics(diagnostics)
    hall_stages = [
        stage for stage in diagnostics["stages"] if stage["status"] == "HALL_FAIL"
    ]
    expected_attempts = [64, 128, 256, 300]
    complete = bool(
        diagnostics["contract_strict"] is False
        and diagnostics["stages"][0]["status"] == "ZERO_EDGE"
        and hall_stages
        and all(
            [attempt["limit"] for attempt in stage["assignment_attempts"]]
            == expected_attempts
            and stage["assignment_attempts"][-1]["scope"] == "full-caliper"
            and all(attempt["witness"] for attempt in stage["assignment_attempts"])
            for stage in hall_stages
        )
    )
    required_deletions = []
    for key in ("scales", "pool_totals", "plan_sha256", "contract_strict"):
        mutant = copy.deepcopy(diagnostics)
        del mutant[key]
        required_deletions.append(
            expect_error(
                lambda mutant=mutant: module.validate_failure_diagnostics(mutant),
                ValueError,
                "failure diagnostics",
            )
        )
    mutant = copy.deepcopy(diagnostics)
    del mutant["stages"][-1]["assignment_attempts"]
    required_deletions.append(expect_error(
        lambda: module.validate_failure_diagnostics(mutant),
        ValueError,
        "failure diagnostics",
    ))
    mutant = copy.deepcopy(diagnostics)
    del mutant["stages"][-1]["recipient_rows"][0]["nearest_primary_distance"]
    required_deletions.append(expect_error(
        lambda: module.validate_failure_diagnostics(mutant),
        ValueError,
        "failure diagnostics",
    ))
    mutant = copy.deepcopy(diagnostics)
    del mutant["stages"][-1]["assignment_attempts"][-1]["witness"][
        "reachable_donors"
    ]
    required_deletions.append(expect_error(
        lambda: module.validate_failure_diagnostics(mutant),
        ValueError,
        "HALL_FAIL",
    ))
    mutant = copy.deepcopy(diagnostics)
    witness = mutant["stages"][-1]["assignment_attempts"][-1]["witness"]
    witness["reachable_donor_count"] = witness["reachable_recipient_count"]
    required_deletions.append(expect_error(
        lambda: module.validate_failure_diagnostics(mutant),
        ValueError,
        "HALL_FAIL",
    ))
    return bool(complete and all(required_deletions)), diagnostics


def invocation_binding_contract(runner, paths) -> dict:
    repo_root = paths["runner"].parents[2]
    common = {
        "repo_root": repo_root,
        "runner_path": paths["runner"],
        "data_root": Path("/mnt1/afrdata"),
        "pose_artifact": runner.FROZEN_POSE_ARTIFACT,
        "core_path": paths["core"],
        "teacher_path": paths["teacher"],
        "python_executable": runner.EXPECTED_RUNTIME_PYTHON,
    }
    runner.validate_frozen_invocation_paths(**common)
    cases = {}
    for name, key, value in (
        ("wrong_runtime", "python_executable", Path("/usr/bin/python3")),
        ("exp405_core", "core_path", repo_root / "experiments/exp405/phase0_core.py"),
        ("wrong_pose", "pose_artifact", Path("/mnt1/afrderived/other_pose")),
        ("wrong_data", "data_root", Path("/mnt2/afrdata")),
    ):
        candidate = dict(common)
        candidate[key] = value
        cases[name] = expect_error(
            lambda candidate=candidate: runner.validate_frozen_invocation_paths(
                **candidate
            ),
            RuntimeError,
            "",
        )
    return cases


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def receipt_binding_contract(runner, scratch_parent: Path) -> dict:
    with tempfile.TemporaryDirectory(
        prefix="exp406-receipt-contract-", dir=str(scratch_parent)
    ) as temporary:
        root = Path(temporary)
        output = root / runner.PREFLIGHT_EXECUTION
        output.mkdir()
        seal = root / (runner.PREFLIGHT_EXECUTION + ".started")
        seal.write_text(runner.PREFLIGHT_EXECUTION + "\n", encoding="utf-8")
        started_path = output / "started.json"
        result_path = output / "result.json"
        cache_path = output / "preflight_cache.pt"
        complete_path = output / "complete.json"
        source_files = {"runner": "r", "core": "c", "teacher": "t"}
        provenance = {
            "source_commit": "commit",
            "official_train_manifest_sha256": "official",
            "asset_manifest_sha256": "asset",
            "clip_checkpoint_sha256": "clip",
            "pose_manifest_sha256": "pose",
        }
        started = {
            "execution": runner.PREFLIGHT_EXECUTION,
            "formal": False,
            "source_commit": "commit",
            "ordered_official_train_manifest_sha256": "official",
            "execution_manifest_sha256": None,
            "source_files": source_files,
            "runtime_sha256": "runtime",
            "asset_manifest_sha256": "asset",
            "clip_checkpoint_sha256": "clip",
            "pose_manifest_sha256": "pose",
        }
        result = {
            "status": "PREFLIGHT_PASS",
            "decision": "PREFLIGHT_ONLY_PASS",
            "scientific_evaluated": False,
            "provenance": provenance,
        }
        write_json(started_path, started)
        write_json(result_path, result)
        cache_path.write_bytes(b"fresh-exp406-cache")
        complete = {
            "execution": runner.PREFLIGHT_EXECUTION,
            "formal": False,
            "status": "PREFLIGHT_PASS",
            "decision": "PREFLIGHT_ONLY_PASS",
            "formal_measurement_authorized": True,
            "transport_oracle_authorized": False,
            "result_sha256": sha256_file(result_path),
            "cache_sha256": sha256_file(cache_path),
            "source_files": source_files,
            "runtime_sha256": "runtime",
            "started_sha256": sha256_file(started_path),
            "started_seal_sha256": sha256_file(seal),
        }
        write_json(complete_path, complete)

        original = {
            "output": runner.PREFLIGHT_OUTPUT_ROOT,
            "complete": runner.PREFLIGHT_COMPLETE_PATH,
            "started": runner.PREFLIGHT_STARTED_PATH,
            "seal": runner.PREFLIGHT_STARTED_SEAL,
        }
        runner.PREFLIGHT_OUTPUT_ROOT = output
        runner.PREFLIGHT_COMPLETE_PATH = complete_path
        runner.PREFLIGHT_STARTED_PATH = started_path
        runner.PREFLIGHT_STARTED_SEAL = seal
        try:
            runner.validate_preflight_receipt(complete_path)
            cases = {"complete_exp406": True}

            seal.unlink()
            cases["missing_started_seal"] = expect_error(
                lambda: runner.validate_preflight_receipt(complete_path),
                RuntimeError,
                "started seal",
            )
            seal.write_text(runner.PREFLIGHT_EXECUTION + "\n", encoding="utf-8")

            bad_complete = dict(complete)
            bad_complete["execution"] = "exp405-p0b-preflight-v1"
            write_json(complete_path, bad_complete)
            cases["exp405_execution"] = expect_error(
                lambda: runner.validate_preflight_receipt(complete_path),
                RuntimeError,
                "does not authorize",
            )
            write_json(complete_path, complete)

            failure = output / "failure.json"
            write_json(failure, {"status": "FAILED"})
            cases["failure_coexists"] = expect_error(
                lambda: runner.validate_preflight_receipt(complete_path),
                RuntimeError,
                "failed preflight",
            )
            failure.unlink()

            bad_result = copy.deepcopy(result)
            bad_result["scientific_evaluated"] = True
            write_json(result_path, bad_result)
            modified_complete = dict(complete)
            modified_complete["result_sha256"] = sha256_file(result_path)
            write_json(complete_path, modified_complete)
            cases["scientific_result_rejected"] = expect_error(
                lambda: runner.validate_preflight_receipt(complete_path),
                RuntimeError,
                "mechanical-only",
            )
            write_json(result_path, result)
            write_json(complete_path, complete)

            cache_path.write_bytes(b"tampered")
            cases["cache_digest_mismatch"] = expect_error(
                lambda: runner.validate_preflight_receipt(complete_path),
                RuntimeError,
                "cache/COMPLETE",
            )
        finally:
            runner.PREFLIGHT_OUTPUT_ROOT = original["output"]
            runner.PREFLIGHT_COMPLETE_PATH = original["complete"]
            runner.PREFLIGHT_STARTED_PATH = original["started"]
            runner.PREFLIGHT_STARTED_SEAL = original["seal"]
        return cases


def _run_node(tree):
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    ]
    if len(matches) != 1:
        raise RuntimeError("run projection is not unique")
    return matches[0]


def _assignment(run_node, target_name: str):
    matches = []
    for node in ast.walk(run_node):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        ):
            matches.append(node)
    if len(matches) != 1:
        raise RuntimeError("assignment projection is not unique: %s" % target_name)
    return matches[0]


def _call_leaf(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _formal_if_with_call(run_node, call_name: str, receiver: str | None = None):
    matches = []
    for node in ast.walk(run_node):
        if not isinstance(node, ast.If) or not (
            isinstance(node.test, ast.Name) and node.test.id == "formal"
        ):
            continue
        body_calls = [
            child for child in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            if isinstance(child, ast.Call) and _call_leaf(child) == call_name
            and (
                receiver is None
                or (
                    isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == receiver
                )
            )
        ]
        if body_calls:
            matches.append((node, body_calls))
    if len(matches) != 1:
        raise RuntimeError("formal call projection is not unique: %s" % call_name)
    return matches[0]


def formal_tree_contract(tree, sealed_tree) -> bool:
    try:
        run_node = _run_node(tree)
        sealed_run = _run_node(sealed_tree)
        execution_if = [
            node for node in ast.walk(run_node)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name) and node.test.id == "formal"
            and any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "execution_indices"
                    for target in child.targets
                )
                for child in node.body + node.orelse
            )
        ]
        if len(execution_if) != 1:
            return False
        expected_range = ast.dump(
            ast.parse("execution_indices = list(range(EXPECTED_SAMPLES))").body[0].value,
            annotate_fields=True,
            include_attributes=False,
        )
        execution_values = [
            child.value for child in execution_if[0].body + execution_if[0].orelse
            if isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "execution_indices"
                for target in child.targets
            )
        ]
        if len(execution_values) != 2 or any(
            ast.dump(value, annotate_fields=True, include_attributes=False)
            != expected_range
            for value in execution_values
        ):
            return False

        semantic_if, semantic_calls = _formal_if_with_call(
            run_node, "full_semantic_summary"
        )
        diagnostic_if, diagnostic_calls = _formal_if_with_call(
            run_node, "choose_diagnostic_subset"
        )
        matcher_if, matcher_calls = _formal_if_with_call(
            run_node, "choose_wrong_masks"
        )
        if len(semantic_calls) != 1 or len(diagnostic_calls) != 1:
            return False
        semantic_args = [
            arg.id if isinstance(arg, ast.Name) else None
            for arg in semantic_calls[0].args
        ]
        diagnostic_args = [
            arg.id if isinstance(arg, ast.Name) else ast.dump(
                arg, annotate_fields=True, include_attributes=False
            )
            for arg in diagnostic_calls[0].args
        ]
        matcher_args = [
            arg.id if isinstance(arg, ast.Name) else None
            for arg in matcher_calls[0].args
        ]
        if semantic_args != ["distribution", "support", "analysis_valid", "pids"]:
            return False
        if diagnostic_args != [
            "targets", "analysis_valid",
            "Call(func=Attribute(value=Name(id='pids', ctx=Load()), attr='tolist', ctx=Load()), args=[], keywords=[])",
            "Attribute(value=Name(id='base_dataset', ctx=Load()), attr='relative_paths', ctx=Load())",
        ]:
            return False
        if matcher_args != [
            "diagnostic_indices", "diagnostic_slots", "analysis_valid", "mass",
            "centroid_y", "pose_confidence", "support", "global_feature", "pids",
            "camids", "sample_keys", "diagnostic_indices",
        ]:
            return False
        progressive_calls = [
            child for child in ast.walk(ast.Module(body=matcher_if.orelse, type_ignores=[]))
            if isinstance(child, ast.Call)
            and _call_leaf(child) == "choose_wrong_masks_progressive"
        ]
        if len(progressive_calls) != 1:
            return False
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "torch"
            and child.func.attr == "load"
            for child in ast.walk(run_node)
        ):
            return False

        current_gates = _assignment(run_node, "gates")
        sealed_gates = _assignment(sealed_run, "gates")
        if ast.dump(current_gates, annotate_fields=True, include_attributes=False) != ast.dump(
            sealed_gates, annotate_fields=True, include_attributes=False
        ):
            return False
        current_formal_gates, _ = _formal_if_with_call(run_node, "update", "gates")
        sealed_formal_gates, _ = _formal_if_with_call(sealed_run, "update", "gates")
        if ast.dump(
            ast.Module(body=current_formal_gates.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        ) != ast.dump(
            ast.Module(body=sealed_formal_gates.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        ):
            return False
        for target in (
            "validity_keys", "validity_pass", "scientific_keys",
            "scientific_pass", "adjudication",
        ):
            if ast.dump(
                _assignment(run_node, target),
                annotate_fields=True,
                include_attributes=False,
            ) != ast.dump(
                _assignment(sealed_run, target),
                annotate_fields=True,
                include_attributes=False,
            ):
                return False
        formal_validity_if, _ = _formal_if_with_call(
            run_node, "extend", "validity_keys"
        )
        sealed_validity_if, _ = _formal_if_with_call(
            sealed_run, "extend", "validity_keys"
        )
        if ast.dump(
            ast.Module(body=formal_validity_if.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        ) != ast.dump(
            ast.Module(body=sealed_validity_if.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        ):
            return False
        return bool(semantic_if and diagnostic_if)
    except (RuntimeError, AttributeError, IndexError):
        return False


def formal_branch_contract(runner_source: str, sealed_source: str) -> bool:
    formal_functions = (
        "choose_targets",
        "choose_diagnostic_subset",
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
        "EXPECTED_SAMPLES", "FORMAL_DIAGNOSTIC_SAMPLES", "SAMPLES_PER_SLOT",
        "VIEW_SEED", "BOOTSTRAP_SEED", "BOOTSTRAP_REPEATS",
        "DELETION_FRACTIONS", "MAX_NO_TARGET_FRACTION",
        "MIN_TARGET_PID_FRACTION", "MATCH_PRIMARY_CALIPER",
        "MATCH_PREFERENCE_LIMIT",
    )
    projection_exact = all(
        function_dump(runner_source, name) == function_dump(sealed_source, name)
        for name in formal_functions
    ) and all(
        constant_value(runner_source, name) == constant_value(sealed_source, name)
        for name in formal_constants
    )
    return bool(
        projection_exact
        and constant_value(runner_source, "PREFLIGHT_EXECUTION_SAMPLES")
        == ("ast", "Name(id='EXPECTED_SAMPLES', ctx=Load())")
        and formal_tree_contract(ast.parse(runner_source), ast.parse(sealed_source))
        and 'if hasattr(error, "diagnostics")' in runner_source
        and 'failure_payload["diagnostics"] = error.diagnostics' in runner_source
    )


def formal_control_mutants(runner_source: str, sealed_source: str) -> dict:
    sealed_tree = ast.parse(sealed_source)
    base_tree = ast.parse(runner_source)
    cases = {}

    tree = copy.deepcopy(base_tree)
    run_node = _run_node(tree)
    count = 0
    for node in ast.walk(run_node):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) \
                and node.test.id == "formal":
            for child in node.orelse:
                if isinstance(child, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "execution_indices"
                    for target in child.targets
                ):
                    child.value = ast.parse(
                        "list(range(PREFLIGHT_SAMPLES))", mode="eval"
                    ).body
                    count += 1
    cases["preflight_subset_regression"] = bool(
        count == 1 and not formal_tree_contract(tree, sealed_tree)
    )

    tree = copy.deepcopy(base_tree)
    run_node = _run_node(tree)
    _, calls = _formal_if_with_call(run_node, "choose_wrong_masks")
    calls[0].args[2] = ast.Name(id="preflight_result", ctx=ast.Load())
    cases["formal_matcher_cache_argument"] = not formal_tree_contract(
        tree, sealed_tree
    )

    tree = copy.deepcopy(base_tree)
    run_node = _run_node(tree)
    run_node.body.insert(0, ast.Expr(value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="torch", ctx=ast.Load()),
            attr="load",
            ctx=ast.Load(),
        ),
        args=[ast.Constant(value="preflight_cache.pt")],
        keywords=[],
    )))
    cases["formal_cache_load"] = not formal_tree_contract(tree, sealed_tree)

    tree = copy.deepcopy(base_tree)
    run_node = _run_node(tree)
    formal_gates, _ = _formal_if_with_call(run_node, "update", "gates")
    count = 0
    for node in ast.walk(ast.Module(body=formal_gates.body, type_ignores=[])):
        if isinstance(node, ast.Compare) and any(
            isinstance(operator, ast.Gt) for operator in node.ops
        ):
            node.comparators[-1] = ast.Constant(value=-1.0)
            count += 1
            break
    cases["formal_scientific_gate_weakened"] = bool(
        count == 1 and not formal_tree_contract(tree, sealed_tree)
    )
    return cases


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
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("output already exists")
    paths["contract"] = Path(__file__).resolve()
    start_hashes = {
        key + "_sha256": sha256_file(path) for key, path in paths.items()
    }
    cuda_before = torch.cuda.is_initialized()
    module = load_module(paths["module"], "exp406_donor_contract_target")
    runner = load_module(paths["runner"], "exp406_runner_contract_target")
    rows = synthetic_inputs(module)
    plan_repeat = module.build_donor_plan(
        rows["records"], rows["paths"], rows["core"],
        stable_digest=stable_digest,
        expected_samples=EXPECTED_SAMPLES,
        core_samples=CORE_SAMPLES,
    )
    donors, records, summary = call_match(module, rows)
    plan_mutants = tampered_plan_contract(module, rows)
    invocation_mutants = invocation_binding_contract(runner, paths)
    receipt_mutants = receipt_binding_contract(runner, output.parent)
    hall_complete, hall_diagnostics = hall_failure_contract(module, rows)
    module_source = paths["module"].read_text(encoding="utf-8")
    runner_source = paths["runner"].read_text(encoding="utf-8")
    sealed_runner_source = paths["sealed_runner"].read_text(encoding="utf-8")
    protocol_source = paths["protocol"].read_text(encoding="utf-8")
    caliper_delete_mutant = caliper_delete_mutant_contract(module_source, rows)
    formal_mutants = formal_control_mutants(runner_source, sealed_runner_source)

    scalar_mutants = {
        "caliper_relaxation": expect_error(
            lambda: call_match(module, rows, caliper=8.1),
            ValueError, "frozen at 8.0",
        ),
        "preference_change": expect_error(
            lambda: call_match(module, rows, preference_limit=128),
            ValueError, "frozen at 64",
        ),
    }
    namespace_exact = bool(
        'PREFLIGHT_EXECUTION = "exp406-p0b-preflight-v1"' in runner_source
        and 'FORMAL_EXECUTION = "exp406-p0b-iso-teacher-v1"' in runner_source
        and '"experiment": "exp406"' in runner_source
        and "exp405-p0b-preflight-v1" not in runner_source
        and "exp405-p0b-iso-teacher-v1" not in runner_source
        and "exp406-p0b-preflight-v1" in protocol_source
        and str(runner.EXPECTED_RUNTIME_PYTHON)
        == "/usr/local/anaconda3/envs/mmpose-abu/bin/python"
    )
    preseal_asset_order = bool(
        runner_source.index("sha256_file(clip_checkpoint) != args.clip_sha256")
        < runner_source.index("acquire_execution_seal(output_dir, execution, args)")
        and runner_source.index("sha256_file(pose_manifest_path)")
        < runner_source.index("acquire_execution_seal(output_dir, execution, args)")
    )
    gates = {
        "source_hashes_bound_including_contract": all(
            path.is_file() for path in paths.values()
        ),
        "core_teacher_exact_sealed_projection": bool(
            paths["core"].read_bytes() == paths["sealed_core"].read_bytes()
            and paths["teacher"].read_bytes() == paths["sealed_teacher"].read_bytes()
        ),
        "formal_science_and_cache_branch_exact": formal_branch_contract(
            runner_source, sealed_runner_source
        ) and all(formal_mutants.values()),
        "fresh_namespace_and_mmpose_binding": namespace_exact,
        "preseal_clip_pose_byte_validation": preseal_asset_order,
        "invocation_path_runtime_mutants_caught": all(invocation_mutants.values()),
        "started_complete_receipt_mutants_caught": all(receipt_mutants.values()),
        "donor_plan_deterministic_and_complete": bool(
            rows["plan"]["plan_sha256"] == plan_repeat["plan_sha256"]
            and rows["plan"]["pool_totals"]
            == [512, 1024, 2048, 4096, 8192, 15618]
            and len(rows["plan"]["stages"][-1]["pool_indices"])
            == EXPECTED_SAMPLES
        ),
        "tampered_stage_plan_mutants_caught": all(plan_mutants.values()),
        "core_only_recipient_selector": selector_core_only_contract(module, rows),
        "fixed_core_mad_and_floor_behavior": fixed_core_scale_contract(module, rows),
        "eligibility_camera_pid_valid_caliper_behavior": eligibility_contract(
            module, rows
        ),
        "caliper_filter_delete_mutant_caught": all(
            caliper_delete_mutant.values()
        ),
        "preference_cosine_key_order_oracle": preference_order_contract(
            module, rows
        ),
        "progressive_positive_identity_unique": bool(
            summary["contract_strict"]
            and summary["selected_pool_total"] == 1024
            and summary["stages"][0]["status"] == "ZERO_EDGE"
            and summary["stages"][1]["status"] == "MATCHED"
            and len(donors) == len(set(donors)) == 20
            and not set(donors).intersection(rows["recipients"])
            and all(record["recipient_pid"] != record["donor_pid"] for record in records)
            and all(record["primary_distance"] <= 8.0 for record in records)
        ),
        "scalar_threshold_mutants_caught": all(scalar_mutants.values()),
        "hall_failure_attempts_and_field_mutants": hall_complete,
        "cuda_not_initialized": bool(
            not cuda_before and not torch.cuda.is_initialized()
        ),
    }
    end_hashes = {
        key + "_sha256": sha256_file(path) for key, path in paths.items()
    }
    gates["source_start_end_exact"] = start_hashes == end_hashes
    result = {
        "experiment": "exp406",
        "schema": "exp406-donor-reserve-static-v3",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "positive": {
            "selected_pool_total": summary["selected_pool_total"],
            "recipient_count": summary["recipient_count"],
            "assigned_donor_count": summary["assigned_donor_count"],
            "summary_sha256": summary["summary_sha256"],
        },
        "mutants": {
            "plan": plan_mutants,
            "invocation": invocation_mutants,
            "receipt": receipt_mutants,
            "scalars": scalar_mutants,
            "caliper_delete": caliper_delete_mutant,
            "formal_control": formal_mutants,
            "hall_stage_statuses": [
                stage["status"] for stage in hall_diagnostics["stages"]
            ] if hall_diagnostics is not None else [],
        },
        "provenance_start": start_hashes,
        "provenance_end": end_hashes,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
    }
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
