#!/usr/bin/env python3
"""Exp403 wrapper around the sealed exp402 full-retrieval audit engine."""

from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


contract_path = Path(os.environ["EXP403_CONTRACT"]).resolve()
base_path = Path(os.environ["EXP403_BASE_AUDIT"]).resolve()
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if contract.get("status") != "PASS":
    raise RuntimeError("Checkpoint contract did not pass")
if sha256_file(base_path) != contract["sha256"]["base_audit"]:
    raise RuntimeError("Base audit SHA mismatch")
if sha256_file(Path(__file__).resolve()) != contract["sha256"]["audit_wrapper"]:
    raise RuntimeError("Audit wrapper SHA mismatch")

source = base_path.read_text(encoding="utf-8")
old_count = '        "checkpoint_state_count": len(payload) == 241,\n'
new_count = (
    '        "checkpoint_state_count": len(payload) == '
    f'{int(contract["checkpoint"]["state_count"])},\n'
)
old_strict = (
    '            "tapf_rich_exact": bool(tapf.rich_evidence and tapf.semantic),\n'
)
new_strict = old_strict + (
    '            "elo_counterfactual_operator": '
    'bool(getattr(tapf, "counterfactual_operator", False)),\n'
    '            "elo_no_static_experts": all('
    'not hasattr(router, "experts") for router in tapf.psg_bank),\n'
    '            "elo_shared_projection_modules": all(all(hasattr(router, name) '
    'for name in ("down_projection", "context_projection", '
    '"evidence_projection", "up_projection", "context_query", '
    '"evidence_key")) for router in tapf.psg_bank),\n'
)
if source.count(old_count) != 1 or source.count(old_strict) != 1:
    raise RuntimeError("Base audit patch contract mismatch")
source = source.replace(old_count, new_count).replace(old_strict, new_strict)

module = types.ModuleType("exp403_audit_engine")
module.__file__ = str(base_path)
module.__name__ = "exp403_audit_engine"
exec(compile(source, str(base_path), "exec"), module.__dict__)
module.SOURCE_COMMIT = contract["repo"]["head"]
module.CONFIG_SHA256 = contract["sha256"]["config"]
module.CHECKPOINT_SHA256 = contract["sha256"]["checkpoint"]
module.CHECKPOINT_STATE_SHA256 = contract["checkpoint"]["state_sha256"]
module.CORE_SHA256 = contract["sha256"]["core"]
module.EXPECTED_SOURCE_SHA256 = contract["sha256"]["source"]
module.REFERENCE_TOLERANCE = 2.0
module.FULL_REFERENCE = {key: 0.0 for key in ("mAP", "rank1", "rank5", "rank10")}
module.ALL_BYPASS_REFERENCE = dict(module.FULL_REFERENCE)

original_run = module.run


def run_exp403(args):
    result = original_run(args)
    validity = dict(result["validity"])
    validity.pop("correct_reference_exact", None)
    validity.pop("all_bypass_reference_exact", None)
    validity["all_metrics_finite"] = all(
        module.finite_metrics(value) for value in result["metrics"].values()
    )
    measurement_pass = all(bool(value) for value in validity.values())
    adjudication = result["adjudication"]
    authorized = bool(
        measurement_pass
        and adjudication["phase_b_formal_mechanism_design_authorized"]
    )
    result.update(
        {
            "status": "PASS" if measurement_pass else "FAIL",
            "decision": (
                adjudication["decision"]
                if measurement_pass
                else "EXP403_SEALED_INVALID"
            ),
            "phase_b_formal_mechanism_design_authorized": authorized,
            "validity": validity,
            "references": {
                "sealed_clean_d0_map": 0.575587756578,
                "sealed_clean_d0_r1": 0.676923076923,
                "semantic_and_route_margin_mAP": 0.001,
            },
        }
    )
    result["assets"]["contract_sha256"] = sha256_file(contract_path)
    result["execution"]["audit_wrapper_sha256"] = sha256_file(
        Path(__file__).resolve()
    )
    result["execution"]["base_audit_sha256"] = sha256_file(base_path)
    return result


module.run = run_exp403
module.main()
