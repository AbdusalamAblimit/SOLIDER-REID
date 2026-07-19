#!/usr/bin/env python3
"""Production source/default-off/replay CPU contract for exp403 ELO-CUR."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
BASELINE_COMMIT = "0722176"
SOURCE_PATHS = (
    "config/defaults.py",
    "model/tapf.py",
    "model/make_model.py",
    "model/backbones/swin_transformer.py",
    "processor/processor.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_git_module(name: str, revision: str, relative_path: str):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=ROOT,
        text=True,
    )
    module = types.ModuleType(name)
    module.__file__ = f"git:{revision}:{relative_path}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def state_exact(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return list(left_state) == list(right_state) and all(
        torch.equal(left_state[key], right_state[key]) for key in left_state
    )


def tensors_exact(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tensors_exact(a, b) for a, b in zip(left, right)
        )
    return left == right


def extract_processor_asset_loader(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_sha256_file",
            "_load_elo_generic_evidence",
        }:
            selected.append(node)
    if len(selected) != 2:
        raise RuntimeError("Could not extract ELO asset loader")
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "stat": __import__("stat"),
        "Path": Path,
        "torch": torch,
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_load_elo_generic_evidence"]


def generic_asset_contract(loader, directory: Path):
    fixture = directory / ".production_generic_fixture.json"
    payload = {
        "experiment": "exp403_counterfactual_operator_ownership",
        "format": "elo_generic_evidence_v1",
        "dataset": "occluded_duke",
        "split": "train",
        "clip_checkpoint_sha256": "c" * 64,
        "codebook_sha256": "b" * 64,
        "pose_manifest_sha256": "p" * 64,
        "count_by_slot": [100, 101, 102, 103, 104],
        "evidence": [
            [float((slot + 1) * (rank + 1)) / 1000.0 for rank in range(16)]
            for slot in range(5)
        ],
    }
    fixture.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        expected_sha = sha256_file(fixture)
        evidence, actual_sha = loader(
            str(fixture.resolve()),
            expected_sha,
            "occluded_duke",
            "c" * 64,
            "b" * 64,
            "p" * 64,
            torch.device("cpu"),
        )
        wrong_sha_caught = False
        try:
            loader(
                str(fixture.resolve()),
                "0" * 64,
                "occluded_duke",
                "c" * 64,
                "b" * 64,
                "p" * 64,
                torch.device("cpu"),
            )
        except RuntimeError:
            wrong_sha_caught = True
        wrong_split_caught = False
        try:
            loader(
                str(fixture.resolve()),
                expected_sha,
                "market1501",
                "c" * 64,
                "b" * 64,
                "p" * 64,
                torch.device("cpu"),
            )
        except RuntimeError:
            wrong_split_caught = True
        return {
            "shape_exact": tuple(evidence.shape) == (5, 16),
            "finite": bool(torch.isfinite(evidence).all()),
            "sha_exact": actual_sha == expected_sha,
            "wrong_sha_caught": wrong_sha_caught,
            "wrong_metadata_caught": wrong_split_caught,
        }
    finally:
        fixture.unlink(missing_ok=True)


def make_tiny_swin(module):
    return module.SwinTransformer(
        pretrain_img_size=(64, 32),
        embed_dims=8,
        window_size=2,
        depths=(1, 1, 1, 2),
        num_heads=(1, 2, 4, 8),
        drop_path_rate=0.2,
        semantic_weight=-1,
    )


def make_pose_batch(batch: int):
    identities = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    cameras = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return {
        "keypoints": torch.rand(batch, 17, 2) * torch.tensor([31.0, 63.0]),
        "scores": torch.ones(batch, 17),
        "valid": torch.ones(batch, 17, dtype=torch.bool),
        "semantic_teacher_evidence": nn.functional.normalize(
            torch.randn(batch, 5, 16), dim=-1
        ),
        "semantic_valid": torch.ones(batch, 5, dtype=torch.bool),
        "semantic_teacher_mask": torch.rand(batch, 5, 16, 8),
        "identity": identities,
        "camera": cameras,
        "generic_evidence": torch.randn(5, 16) * 0.05,
    }


def run_contract(output_path: Path):
    cuda_before = torch.cuda.is_initialized()
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    current_tapf = load_file_module("exp403_current_tapf", ROOT / "model/tapf.py")
    legacy_tapf = load_git_module(
        "exp403_legacy_tapf", BASELINE_COMMIT, "model/tapf.py"
    )
    current_swin = load_file_module(
        "exp403_current_swin", ROOT / "model/backbones/swin_transformer.py"
    )
    legacy_swin = load_git_module(
        "exp403_legacy_swin",
        BASELINE_COMMIT,
        "model/backbones/swin_transformer.py",
    )

    torch.manual_seed(4403)
    current_d0 = make_tiny_swin(current_swin)
    current_d0_rng = torch.get_rng_state().clone()
    torch.manual_seed(4403)
    legacy_d0 = make_tiny_swin(legacy_swin)
    legacy_d0_rng = torch.get_rng_state().clone()
    d0_state_exact = state_exact(current_d0, legacy_d0)
    d0_init_rng_exact = torch.equal(current_d0_rng, legacy_d0_rng)
    d0_input = torch.randn(2, 3, 64, 32)
    current_d0.eval()
    legacy_d0.eval()
    with torch.no_grad():
        current_d0_output = current_d0(d0_input)
        legacy_d0_output = legacy_d0(d0_input)
    d0_output_exact = tensors_exact(current_d0_output, legacy_d0_output)

    c0_kwargs = {
        "anchor_channels": 8,
        "anchor_hidden": 16,
        "consumer_channels": 8,
        "router_rank": 4,
    }
    torch.manual_seed(5403)
    current_c0 = current_tapf.CleanRichEvidenceBudgetTapf(**c0_kwargs)
    current_c0_rng = torch.get_rng_state().clone()
    torch.manual_seed(5403)
    legacy_c0 = legacy_tapf.CleanRichEvidenceBudgetTapf(**c0_kwargs)
    legacy_c0_rng = torch.get_rng_state().clone()
    c0_state_exact = state_exact(current_c0, legacy_c0)
    c0_init_rng_exact = torch.equal(current_c0_rng, legacy_c0_rng)
    c0_source = torch.randn(3, 8, 4, 2)
    c0_tokens = torch.randn(3, 8, 8)
    current_c0.eval()
    legacy_c0.eval()
    with torch.no_grad():
        current_state = current_c0.prepare(
            c0_source, None, (384, 128), epoch=None, training=False
        )
        legacy_state = legacy_c0.prepare(
            c0_source, None, (384, 128), epoch=None, training=False
        )
        current_tokens = c0_tokens.clone()
        legacy_tokens = c0_tokens.clone()
        for index in range(2):
            current_tokens = current_c0.apply_gate(
                index, current_tokens, (4, 2), current_state
            )
            legacy_tokens = legacy_c0.apply_gate(
                index, legacy_tokens, (4, 2), legacy_state
            )
    c0_output_exact = torch.equal(current_tokens, legacy_tokens)

    router = current_tapf.EvidenceOwnedLowRankRouter(
        feature_channels=8, region_count=5, rank=4, evidence_dim=16
    )
    router_linears_no_bias = all(
        module.bias is None
        for module in router.modules()
        if isinstance(module, nn.Linear)
    )
    router_has_no_expert = all(
        "expert" not in name.lower() for name, _ in router.named_parameters()
    ) and not any(isinstance(module, nn.ModuleList) for module in router.modules())
    null_tokens = torch.randn(8, 8, 8)
    null_mask = torch.rand(8, 5, 4, 2)
    null_presence = torch.ones(8, 5)
    null_evidence = torch.zeros(8, 5, 16)
    null_output, null_delta, _ = router(
        null_tokens,
        (4, 2),
        null_mask,
        null_presence,
        null_evidence,
        0.08075544983148575,
    )
    null_identity_exact = torch.equal(null_output, null_tokens) and torch.equal(
        null_delta, torch.zeros_like(null_delta)
    )

    torch.manual_seed(6403)
    replay_backbone = make_tiny_swin(current_swin)
    replay_tapf = current_tapf.CleanEvidenceOperatorTapf(
        anchor_channels=32,
        anchor_hidden=16,
        consumer_channels=64,
        router_rank=4,
    )
    replay_backbone.enable_tapf(replay_tapf)
    correct_only = copy.deepcopy(replay_backbone)
    correct_only.tapf.counterfactual_operator = False
    replay_backbone.train()
    correct_only.train()
    pose_batch = make_pose_batch(8)
    replay_input = torch.randn(8, 3, 64, 32)
    replay_rng_start = torch.get_rng_state().clone()
    replay_feature, _, replay_state = replay_backbone(
        replay_input, pose_batch=pose_batch, tapf_epoch=6
    )
    replay_rng_after = torch.get_rng_state().clone()
    torch.set_rng_state(replay_rng_start)
    correct_feature, _, _ = correct_only(
        replay_input, pose_batch=pose_batch, tapf_epoch=6
    )
    correct_rng_after = torch.get_rng_state().clone()
    replay_preserves_correct_output = torch.equal(replay_feature, correct_feature)
    replay_restores_rng = torch.equal(replay_rng_after, correct_rng_after)
    reference_no_grad = all(
        not value.requires_grad
        for value in replay_state["reference_descriptors"].values()
    ) and all(
        not replay_state["reference_evidence"][name].requires_grad
        for name in replay_tapf.reference_arm_names
    )
    replay_state["student_evidence"].retain_grad()
    production_loss = replay_feature.square().mean()
    production_loss = production_loss + 0.1 * replay_state["pose_loss"]
    production_loss.backward()
    student_evidence_grad = replay_state["student_evidence"].grad
    correct_evidence_grad_nonzero = (
        student_evidence_grad is not None
        and bool(torch.isfinite(student_evidence_grad).all())
        and float(student_evidence_grad.norm()) > 0
    )
    production_parameter_grads = {
        name: 0.0 if parameter.grad is None else float(parameter.grad.norm())
        for name, parameter in replay_tapf.named_parameters()
        if name.startswith("psg_bank.")
    }
    all_production_parameter_grads = all(
        math.isfinite(value) and value > 0
        for value in production_parameter_grads.values()
    )

    state_keys = list(replay_tapf.state_dict())
    teacher_free_state = not any(
        marker in key.lower()
        for key in state_keys
        for marker in ("teacher", "generic", "codebook", "clip")
    )
    reload_tapf = current_tapf.CleanEvidenceOperatorTapf(
        anchor_channels=32,
        anchor_hidden=16,
        consumer_channels=64,
        router_rank=4,
    )
    reload_result = reload_tapf.load_state_dict(replay_tapf.state_dict(), strict=True)
    strict_reload = not reload_result.missing_keys and not reload_result.unexpected_keys
    optimizer = torch.optim.SGD(replay_tapf.parameters(), lr=0.01)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {
        id(parameter)
        for parameter in replay_tapf.parameters()
        if parameter.requires_grad
    }
    optimizer_exact = optimizer_ids == trainable_ids

    donor = current_tapf.build_matched_training_donor_map(
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    donor_repeat = current_tapf.build_matched_training_donor_map(
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    donor_expected = torch.tensor([2, 2, 0, 0, 6, 6, 4, 4])
    donor_ineligible = current_tapf.build_matched_training_donor_map(
        torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1])
    )

    defaults_source = (ROOT / "config/defaults.py").read_text(encoding="utf-8")
    make_model_source = (ROOT / "model/make_model.py").read_text(encoding="utf-8")
    swin_source = (ROOT / "model/backbones/swin_transformer.py").read_text(
        encoding="utf-8"
    )
    tapf_source = (ROOT / "model/tapf.py").read_text(encoding="utf-8")
    source_contract = {
        "default_switch_off": "_C.MODEL.TAPF.ELO_CUR_ENABLED = False"
        in defaults_source,
        "new_model_switch_guarded": (
            "if cfg.MODEL.TAPF.ELO_CUR_ENABLED" in make_model_source
            and "CleanEvidenceOperatorTapf" in make_model_source
        ),
        "reference_replay_no_grad": "with torch.no_grad():" in swin_source,
        "reference_rng_restore_present": (
            "_restore_tapf_replay_rng(correct_rng_after" in swin_source
        ),
        "one_sided_compatibility": (
            "maximum_reference = detached_reference.max" in tapf_source
            and "- correct" in tapf_source
        ),
        "reference_descriptors_detached": (
            "descriptor.detach()" in tapf_source
        ),
    }
    asset_gates = generic_asset_contract(
        extract_processor_asset_loader(ROOT / "processor/processor.py"),
        output_path.parent,
    )

    gates = {
        "cuda_not_initialized": (
            not cuda_before and not torch.cuda.is_initialized()
        ),
        "d0_state_exact_to_preimplementation_commit": d0_state_exact,
        "d0_init_rng_exact_to_preimplementation_commit": d0_init_rng_exact,
        "d0_output_exact_to_preimplementation_commit": d0_output_exact,
        "c0_state_exact_to_preimplementation_commit": c0_state_exact,
        "c0_init_rng_exact_to_preimplementation_commit": c0_init_rng_exact,
        "c0_output_exact_to_preimplementation_commit": c0_output_exact,
        "router_linears_no_bias": router_linears_no_bias,
        "router_has_no_slot_experts": router_has_no_expert,
        "null_exact_identity": null_identity_exact,
        "donor_expected_exact": torch.equal(donor, donor_expected),
        "donor_repeat_exact": torch.equal(donor, donor_repeat),
        "donor_ineligible_is_minus_one": donor_ineligible.tolist() == [1, 0, -1],
        "reference_rng_consumption_exact": replay_state["reference_rng_exact"],
        "replay_preserves_correct_output": replay_preserves_correct_output,
        "replay_restores_global_rng": replay_restores_rng,
        "references_require_no_grad": reference_no_grad,
        "correct_evidence_grad_finite_nonzero": correct_evidence_grad_nonzero,
        "all_shared_production_grads_finite_nonzero": all_production_parameter_grads,
        "teacher_generic_free_state": teacher_free_state,
        "strict_reload": strict_reload,
        "optimizer_exact": optimizer_exact,
        "counterfactual_finalized": replay_state["counterfactual_finalized"],
        **source_contract,
        **{"asset_" + key: value for key, value in asset_gates.items()},
    }
    passed = all(bool(value) for value in gates.values())
    return {
        "experiment": "exp403_counterfactual_operator_ownership",
        "baseline_commit": BASELINE_COMMIT,
        "status": "PRODUCTION_CPU_PASS" if passed else "PRODUCTION_CPU_FAIL",
        "gpu_authorized": False,
        "formal_training_authorized": False,
        "gate_count": len(gates),
        "gate_pass_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "measurements": {
            "donor_map": donor.tolist(),
            "donor_ineligible_map": donor_ineligible.tolist(),
            "compatibility_loss": float(
                replay_state["compatibility_loss"].detach()
            ),
            "cur_loss": float(replay_state["cur_loss"].detach()),
            "coefficient_std": float(replay_state["coefficient_std"]),
            "coefficient_effective_rank": float(
                replay_state["coefficient_effective_rank"]
            ),
            "student_evidence_grad_norm": float(student_evidence_grad.norm()),
            "production_parameter_grad_norm": production_parameter_grads,
            "model_state_key_count": len(state_keys),
        },
        "source_sha256": {
            relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_contract(args.output.resolve())
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PRODUCTION_CPU_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
