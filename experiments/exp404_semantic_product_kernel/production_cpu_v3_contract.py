#!/usr/bin/env python3
"""Joint-field production CPU contract after exp404 CUDA preflight v1."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
V1_CONTRACT = Path(__file__).with_name("production_cpu_v1_contract.py")
V1_RESULT = Path(__file__).with_name("production_cpu_v1_result.json")
PREFLIGHT_V1 = Path(__file__).with_name(
    "cuda_amp_preflight_v1_sealed_invalid.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding_precedes_bnneck(path: Path) -> tuple[bool, int, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    build_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "build_transformer"
    )
    forward = next(
        node
        for node in build_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    calls = {
        node.value.func.attr: node.lineno
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in {"semantic_product_kernel", "bottleneck"}
    }
    binding_line = calls.get("semantic_product_kernel", -1)
    bottleneck_line = calls.get("bottleneck", -1)
    return (
        0 < binding_line < bottleneck_line,
        binding_line,
        bottleneck_line,
    )


def make_pose_batch(batch: int):
    return {
        "keypoints": torch.rand(batch, 17, 2) * torch.tensor([31.0, 63.0]),
        "scores": torch.ones(batch, 17),
        "valid": torch.ones(batch, 17, dtype=torch.bool),
        "semantic_teacher_evidence": torch.nn.functional.normalize(
            torch.randn(batch, 5, 16), dim=-1
        ),
        "semantic_valid": torch.ones(batch, 5, dtype=torch.bool),
        "semantic_teacher_mask": torch.rand(batch, 5, 16, 8),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("V3 output must be fresh")

    v1 = load_module("exp404_production_v1_for_v3", V1_CONTRACT)
    payload = v1.run_contract(args.output.resolve())
    binding_ok, binding_line, bottleneck_line = binding_precedes_bnneck(
        ROOT / "model/make_model.py"
    )
    payload["gates"]["binding_before_bnneck"] = binding_ok

    tapf_module = v1.load_file_module(
        "exp404_joint_field_tapf", ROOT / "model/tapf.py"
    )
    torch.manual_seed(9404)
    tapf = tapf_module.CleanSemanticProductTapf(
        anchor_channels=32,
        anchor_hidden=16,
        consumer_channels=64,
        psg_hidden=16,
    )
    source = torch.randn(4, 32, 4, 2)
    tokens = torch.randn(4, 8, 64)
    pose_batch = make_pose_batch(4)
    tapf.train()
    train_state = tapf.prepare(
        source,
        pose_batch,
        image_hw=(64, 32),
        epoch=6,
        training=True,
    )
    expected_mixed_joint = (
        0.8 * train_state["teacher_joint_field"]
        + 0.2 * train_state["student_joint_field"]
    ).detach()
    routed = tokens.clone()
    for index in range(2):
        routed = tapf.apply_gate(index, routed, (4, 2), train_state)

    tapf.eval()
    with torch.no_grad():
        eval_state = tapf.prepare(
            source,
            None,
            image_hw=(64, 32),
            epoch=None,
            training=False,
        )
        eval_routed = tokens.clone()
        for index in range(2):
            eval_routed = tapf.apply_gate(
                index, eval_routed, (4, 2), eval_state
            )

    mutant_state = dict(train_state)
    mutant_state["consumer_field"] = train_state["consumer_mask"]
    region_field_mutant_caught = False
    try:
        tapf.apply_gate(0, tokens, (4, 2), mutant_state)
    except RuntimeError:
        region_field_mutant_caught = True

    joint_gates = {
        "train_consumer_joint_field_shape_exact": tuple(
            train_state["consumer_field"].shape
        ) == (4, 17, 4, 2),
        "eval_consumer_joint_field_shape_exact": tuple(
            eval_state["consumer_field"].shape
        ) == (4, 17, 4, 2),
        "rich_region_state_remains_5_slot": tuple(
            train_state["consumer_mask"].shape
        ) == (4, 5, 4, 2)
        and tuple(train_state["student_evidence"].shape) == (4, 5, 16),
        "joint_handoff_mix_exact": torch.equal(
            train_state["consumer_joint_field"], expected_mixed_joint
        )
        and train_state["student_fraction"] == 0.2,
        "two_train_d0_gates_execute_finite": len(train_state["gate_deltas"])
        == 2
        and all(
            bool(torch.isfinite(delta).all())
            for delta in train_state["gate_deltas"]
        )
        and torch.equal(routed, tokens),
        "two_eval_d0_gates_execute_finite": len(eval_state["gate_deltas"])
        == 2
        and all(
            bool(torch.isfinite(delta).all())
            for delta in eval_state["gate_deltas"]
        )
        and torch.equal(eval_routed, tokens),
        "five_slot_region_field_mutant_caught": region_field_mutant_caught,
        "spk_prepare_override_present": (
            'state["consumer_field"] = state["consumer_joint_field"]'
            in (ROOT / "model/tapf.py").read_text(encoding="utf-8")
        ),
    }
    payload["gates"].update(joint_gates)
    payload["gate_count"] = len(payload["gates"])
    payload["gate_pass_count"] = sum(
        bool(value) for value in payload["gates"].values()
    )
    passed = payload["gate_pass_count"] == payload["gate_count"]
    payload["status"] = (
        "PRODUCTION_CPU_V3_PASS" if passed else "PRODUCTION_CPU_V3_FAIL"
    )
    payload["cuda_authorized"] = False
    payload["formal_training_authorized"] = False
    payload["contract_revision"] = {
        "version": 3,
        "reason": (
            "CUDA preflight v1 exposed a 5-slot region field passed to the "
            "17-channel D0 PoseSpatialGate"
        ),
        "binding_line": binding_line,
        "bottleneck_line": bottleneck_line,
        "v1_contract_sha256": v1.sha256_file(V1_CONTRACT),
        "v1_cpu_result_sha256": v1.sha256_file(V1_RESULT),
        "cuda_preflight_v1_record_sha256": v1.sha256_file(PREFLIGHT_V1),
    }
    payload["measurements"].update(
        {
            "train_consumer_joint_field_shape": list(
                train_state["consumer_field"].shape
            ),
            "train_consumer_region_mask_shape": list(
                train_state["consumer_mask"].shape
            ),
            "train_gate_delta_abs": [
                float(delta.detach().abs().mean())
                for delta in train_state["gate_deltas"]
            ],
        }
    )
    v1.atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
