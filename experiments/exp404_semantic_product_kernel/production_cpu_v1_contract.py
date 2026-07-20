#!/usr/bin/env python3
"""Production/default-off/train-eval CPU contract for exp404 SPK."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
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
BASELINE_COMMIT = "07ca01c"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_PATHS = (
    "config/defaults.py",
    "model/tapf.py",
    "model/make_model.py",
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
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and list(left) == list(right)
            and all(tensors_exact(left[key], right[key]) for key in left)
        )
    return left == right


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


class FakeTapfBase(nn.Module):
    def __init__(self, batch=3, feature_dim=64):
        super().__init__()
        self.global_feature = nn.Parameter(torch.randn(batch, feature_dim))
        self.student_evidence = nn.Parameter(torch.randn(batch, 5, 16))
        self.register_buffer("student_presence", torch.ones(batch, 5))

    def forward(self, x, pose_batch=None, tapf_epoch=None):
        del x, pose_batch, tapf_epoch
        return (
            self.global_feature,
            [self.global_feature[:, None]],
            {
                "student_evidence": self.student_evidence,
                "student_presence": self.student_presence,
            },
        )


class FakePlainBase(nn.Module):
    def __init__(self, batch=3, feature_dim=64):
        super().__init__()
        self.global_feature = nn.Parameter(torch.randn(batch, feature_dim))

    def forward(self, x, pose_batch=None, tapf_epoch=None):
        del x, pose_batch, tapf_epoch
        return self.global_feature, [self.global_feature[:, None]]


def make_forward_shell(make_model_module, tapf_module, spk_enabled: bool):
    model = make_model_module.build_transformer.__new__(
        make_model_module.build_transformer
    )
    nn.Module.__init__(model)
    model.tapf_enabled = bool(spk_enabled)
    model.spk_enabled = bool(spk_enabled)
    model.reduce_feat_dim = False
    model.base = FakeTapfBase() if spk_enabled else FakePlainBase()
    if spk_enabled:
        model.semantic_product_kernel = tapf_module.SemanticProductKernel(
            feature_dim=64,
            groups=16,
        )
    model.bottleneck = nn.Identity()
    model.dropout = nn.Identity()
    model.ID_LOSS_TYPE = "softmax"
    model.classifier = nn.Linear(64, 7, bias=False)
    model.neck_feat = "before"
    return model


def raises(callable_object, exception_type=Exception) -> bool:
    try:
        callable_object()
    except exception_type:
        return True
    return False


def run_contract(output_path: Path):
    cuda_before = torch.cuda.is_initialized()
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))

    current_tapf = load_file_module("exp404_current_tapf", ROOT / "model/tapf.py")
    legacy_tapf = load_git_module(
        "exp404_legacy_tapf", BASELINE_COMMIT, "model/tapf.py"
    )
    current_swin = load_file_module(
        "exp404_current_swin", ROOT / "model/backbones/swin_transformer.py"
    )
    legacy_swin = load_git_module(
        "exp404_legacy_swin",
        BASELINE_COMMIT,
        "model/backbones/swin_transformer.py",
    )
    make_model_module = importlib.import_module("model.make_model")

    # Existing default-off backbone and C0 behavior must remain byte exact.
    torch.manual_seed(4404)
    current_d0 = make_tiny_swin(current_swin)
    current_d0_rng = torch.get_rng_state().clone()
    torch.manual_seed(4404)
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
    torch.manual_seed(5404)
    current_c0 = current_tapf.CleanRichEvidenceBudgetTapf(**c0_kwargs)
    current_c0_rng = torch.get_rng_state().clone()
    torch.manual_seed(5404)
    legacy_c0 = legacy_tapf.CleanRichEvidenceBudgetTapf(**c0_kwargs)
    legacy_c0_rng = torch.get_rng_state().clone()
    c0_source = torch.randn(3, 8, 4, 2)
    c0_tokens = torch.randn(3, 8, 8)
    current_c0.eval()
    legacy_c0.eval()
    with torch.no_grad():
        current_state = current_c0.prepare(
            c0_source, None, (64, 32), epoch=None, training=False
        )
        legacy_state = legacy_c0.prepare(
            c0_source, None, (64, 32), epoch=None, training=False
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
    c0_state_exact = state_exact(current_c0, legacy_c0)
    c0_init_rng_exact = torch.equal(current_c0_rng, legacy_c0_rng)
    c0_output_exact = torch.equal(current_tokens, legacy_tokens)

    # Direct production kernel semantics and gradients.
    kernel = current_tapf.SemanticProductKernel(feature_dim=64, groups=16)
    kernel_parameter_count = sum(parameter.numel() for parameter in kernel.parameters())
    kernel_buffer_count = sum(buffer.numel() for buffer in kernel.buffers())
    torch.manual_seed(6404)
    global_feature = torch.randn(4, 64, requires_grad=True)
    evidence = torch.randn(4, 5, 16, requires_grad=True)
    presence = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
        ]
    )
    descriptor, factor = kernel(global_feature, evidence, presence)
    mass = presence.sum(dim=1, keepdim=True)
    expected_pooled = (evidence * presence[..., None]).sum(dim=1)
    expected_pooled = expected_pooled / mass.clamp_min(1.0)
    expected_pooled = torch.where(
        mass > 0, expected_pooled, torch.zeros_like(expected_pooled)
    )
    expected_factor = 16 * torch.softmax(expected_pooled.float(), dim=-1)
    expected_descriptor = (
        expected_factor[..., None] * global_feature.reshape(4, 16, 4)
    ).reshape_as(global_feature)
    fixed_product_exact = torch.equal(factor, expected_factor) and torch.equal(
        descriptor, expected_descriptor
    )
    production_loss = descriptor.square().mean()
    production_loss.backward()
    global_grad_norm = float(global_feature.grad.norm())
    evidence_grad_norm = float(evidence.grad.norm())
    direct_grads_finite_nonzero = all(
        math.isfinite(value) and value > 0
        for value in (global_grad_norm, evidence_grad_norm)
    )

    null_global = torch.randn(4, 64, dtype=torch.float16)
    null_evidence = torch.zeros(4, 5, 16, dtype=torch.float16)
    null_presence = torch.randint(0, 2, (4, 5)).float()
    null_descriptor, null_factor = kernel(
        null_global, null_evidence, null_presence
    )
    null_identity_exact = torch.equal(null_descriptor, null_global)
    null_factor_exact = torch.equal(null_factor, torch.ones_like(null_factor))
    null_dtype_exact = (
        null_descriptor.dtype == null_global.dtype
        and null_factor.dtype == torch.float32
    )
    shape_guards = all(
        (
            raises(lambda: kernel(torch.randn(2, 63), torch.randn(2, 5, 16), torch.ones(2, 5)), ValueError),
            raises(lambda: kernel(torch.randn(2, 64), torch.randn(2, 5, 15), torch.ones(2, 5)), ValueError),
            raises(lambda: kernel(torch.randn(2, 64), torch.randn(2, 5, 16), torch.ones(2, 4)), ValueError),
            raises(lambda: current_tapf.SemanticProductKernel(63, 16), ValueError),
        )
    )

    # Production TAPF object must contain only rich anchor + original D0 gates.
    torch.manual_seed(7404)
    spk_tapf = current_tapf.CleanSemanticProductTapf(
        anchor_channels=32,
        anchor_hidden=16,
        consumer_channels=64,
        psg_hidden=16,
    )
    spk_rng = torch.get_rng_state().clone()
    torch.manual_seed(7404)
    spk_repeat = current_tapf.CleanSemanticProductTapf(
        anchor_channels=32,
        anchor_hidden=16,
        consumer_channels=64,
        psg_hidden=16,
    )
    spk_repeat_rng = torch.get_rng_state().clone()
    spk_state_repeat_exact = state_exact(spk_tapf, spk_repeat)
    spk_rng_repeat_exact = torch.equal(spk_rng, spk_repeat_rng)
    d0_gate_types_exact = all(
        isinstance(gate, current_tapf.PoseSpatialGate)
        for gate in spk_tapf.psg_bank
    ) and len(spk_tapf.psg_bank) == 2
    forbidden_router_types_absent = not any(
        isinstance(module, (current_tapf.EvidenceBudgetRouter, current_tapf.EvidenceOwnedLowRankRouter))
        for module in spk_tapf.modules()
    )
    state_keys = list(spk_tapf.state_dict())
    old_router_state_absent = not any(
        marker in key
        for key in state_keys
        for marker in (
            "token_projection",
            "context_projection",
            "evidence_projection",
            "experts",
            "down_projection",
            "up_projection",
            "context_query",
            "evidence_key",
        )
    )
    teacher_free_state = not any(
        marker in key.lower()
        for key in state_keys
        for marker in ("teacher", "generic", "codebook", "clip")
    )
    strict_target = current_tapf.CleanSemanticProductTapf(
        anchor_channels=32,
        anchor_hidden=16,
        consumer_channels=64,
        psg_hidden=16,
    )
    reload_result = strict_target.load_state_dict(spk_tapf.state_dict(), strict=True)
    strict_reload = not reload_result.missing_keys and not reload_result.unexpected_keys
    optimizer = torch.optim.SGD(spk_tapf.parameters(), lr=0.01)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {
        id(parameter)
        for parameter in spk_tapf.parameters()
        if parameter.requires_grad
    }
    optimizer_exact = optimizer_ids == trainable_ids

    # Execute the real build_transformer.forward method without a full Swin.
    torch.manual_seed(8404)
    shell = make_forward_shell(make_model_module, current_tapf, spk_enabled=True)
    shell_input = torch.zeros(3, 3, 8, 4)
    raw_global = shell.base.global_feature.detach().clone()
    raw_evidence = shell.base.student_evidence.detach().clone()
    raw_presence = shell.base.student_presence.detach().clone()
    with torch.no_grad():
        expected_bound, expected_shell_factor = shell.semantic_product_kernel(
            raw_global, raw_evidence, raw_presence
        )
    shell.train()
    train_score, train_descriptor, _, train_aux = shell(
        shell_input, tapf_epoch=6
    )
    train_descriptor.retain_grad()
    shell_loss = train_score.square().mean() + train_descriptor.square().mean()
    shell_loss.backward()
    shell_global_grad = float(shell.base.global_feature.grad.norm())
    shell_evidence_grad = float(shell.base.student_evidence.grad.norm())
    shell_grads_finite_nonzero = all(
        math.isfinite(value) and value > 0
        for value in (shell_global_grad, shell_evidence_grad)
    )
    train_reads_bound = torch.equal(train_descriptor.detach(), expected_bound)
    classifier_reads_bound = torch.equal(
        train_score.detach(), shell.classifier(expected_bound).detach()
    )
    aux_factor_exact = torch.equal(
        train_aux["semantic_product_factor"].detach(), expected_shell_factor
    )
    aux_delta_exact = torch.equal(
        train_aux["semantic_product_delta"].detach(), expected_bound - raw_global
    )
    shell.eval()
    shell.neck_feat = "before"
    with torch.no_grad():
        eval_before, _ = shell(shell_input)
    shell.neck_feat = "after"
    with torch.no_grad():
        eval_after, _ = shell(shell_input)
    eval_before_reads_bound = torch.equal(eval_before, expected_bound)
    eval_after_reads_bound = torch.equal(eval_after, expected_bound)

    shell_reload = copy.deepcopy(shell)
    shell_reload_result = shell_reload.load_state_dict(shell.state_dict(), strict=True)
    shell_strict_reload = (
        not shell_reload_result.missing_keys
        and not shell_reload_result.unexpected_keys
    )

    plain_shell = make_forward_shell(
        make_model_module, current_tapf, spk_enabled=False
    )
    plain_raw = plain_shell.base.global_feature.detach().clone()
    plain_shell.train()
    _, plain_train_descriptor, _ = plain_shell(shell_input)
    plain_shell.eval()
    plain_shell.neck_feat = "before"
    with torch.no_grad():
        plain_eval_descriptor, _ = plain_shell(shell_input)
    default_off_forward_identity = torch.equal(
        plain_train_descriptor.detach(), plain_raw
    ) and torch.equal(plain_eval_descriptor, plain_raw)

    # Positive source mutants must be observable by this contract.
    wrong_evidence = raw_evidence.roll(shifts=1, dims=0)
    with torch.no_grad():
        wrong_bound, _ = shell.semantic_product_kernel(
            raw_global, wrong_evidence, raw_presence
        )
    evidence_ignored_mutant_caught = not torch.equal(expected_bound, wrong_bound)
    auxiliary_only_mutant_caught = (
        not torch.equal(expected_bound, raw_global) and train_reads_bound
    )
    additive_bypass_mutant_caught = not torch.equal(
        train_descriptor.detach(), raw_global + expected_bound
    )

    defaults_source = (ROOT / "config/defaults.py").read_text(encoding="utf-8")
    make_model_source = (ROOT / "model/make_model.py").read_text(encoding="utf-8")
    tapf_source = (ROOT / "model/tapf.py").read_text(encoding="utf-8")
    processor_source = (ROOT / "processor/processor.py").read_text(encoding="utf-8")
    for path, source in (
        ("config/defaults.py", defaults_source),
        ("model/make_model.py", make_model_source),
        ("model/tapf.py", tapf_source),
        ("processor/processor.py", processor_source),
    ):
        ast.parse(source, filename=path)
    source_contract = {
        "default_switch_off": "_C.MODEL.TAPF.SPK_ENABLED = False" in defaults_source,
        "groups_frozen_to_16_guard": (
            "SPK_GROUPS must match the frozen 16-D rich evidence" in make_model_source
        ),
        "spk_elo_mutual_exclusion": "SPK and ELO-CUR are mutually exclusive" in make_model_source,
        "binding_before_bnneck": (
            make_model_source.index("global_feat, product_factor =")
            < make_model_source.index("feat = self.bottleneck(global_feat)")
        ),
        "no_product_temperature_or_projection": (
            "SemanticProductKernel" in tapf_source
            and "factor_float = self.groups * torch.softmax(pooled, dim=-1)" in tapf_source
        ),
        "spk_logging_present": "SPKMean/Std/Min/Max" in processor_source,
    }

    gates = {
        "cuda_not_initialized": not cuda_before and not torch.cuda.is_initialized(),
        "d0_state_exact_to_preimplementation_commit": d0_state_exact,
        "d0_init_rng_exact_to_preimplementation_commit": d0_init_rng_exact,
        "d0_output_exact_to_preimplementation_commit": d0_output_exact,
        "c0_state_exact_to_preimplementation_commit": c0_state_exact,
        "c0_init_rng_exact_to_preimplementation_commit": c0_init_rng_exact,
        "c0_output_exact_to_preimplementation_commit": c0_output_exact,
        "kernel_zero_parameters": kernel_parameter_count == 0,
        "kernel_zero_buffers": kernel_buffer_count == 0,
        "fixed_16x4_product_exact": fixed_product_exact,
        "null_factor_exact_one": null_factor_exact,
        "null_descriptor_exact_identity": null_identity_exact,
        "null_dtype_contract": null_dtype_exact,
        "kernel_shape_guards": shape_guards,
        "direct_global_evidence_grads_finite_nonzero": direct_grads_finite_nonzero,
        "spk_tapf_state_repeat_exact": spk_state_repeat_exact,
        "spk_tapf_rng_repeat_exact": spk_rng_repeat_exact,
        "d0_gate_types_exact": d0_gate_types_exact,
        "forbidden_c0_elo_router_types_absent": forbidden_router_types_absent,
        "old_router_state_absent": old_router_state_absent,
        "teacher_generic_free_state": teacher_free_state,
        "strict_reload": strict_reload,
        "optimizer_exact": optimizer_exact,
        "train_reads_bound_descriptor": train_reads_bound,
        "classifier_reads_bound_descriptor": classifier_reads_bound,
        "eval_before_bn_reads_bound_descriptor": eval_before_reads_bound,
        "eval_after_bn_reads_bound_descriptor": eval_after_reads_bound,
        "aux_factor_exact": aux_factor_exact,
        "aux_delta_exact": aux_delta_exact,
        "shell_global_evidence_grads_finite_nonzero": shell_grads_finite_nonzero,
        "shell_strict_reload": shell_strict_reload,
        "default_off_forward_identity": default_off_forward_identity,
        "evidence_ignored_mutant_caught": evidence_ignored_mutant_caught,
        "auxiliary_only_mutant_caught": auxiliary_only_mutant_caught,
        "additive_bypass_mutant_caught": additive_bypass_mutant_caught,
        **source_contract,
    }
    passed = all(bool(value) for value in gates.values())
    return {
        "experiment": "exp404_semantic_product_kernel",
        "baseline_commit": BASELINE_COMMIT,
        "status": "PRODUCTION_CPU_PASS" if passed else "PRODUCTION_CPU_FAIL",
        "cuda_authorized": False,
        "formal_training_authorized": False,
        "gate_count": len(gates),
        "gate_pass_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "measurements": {
            "kernel_parameter_count": kernel_parameter_count,
            "kernel_buffer_count": kernel_buffer_count,
            "global_grad_norm": global_grad_norm,
            "evidence_grad_norm": evidence_grad_norm,
            "shell_global_grad_norm": shell_global_grad,
            "shell_evidence_grad_norm": shell_evidence_grad,
            "spk_tapf_state_key_count": len(state_keys),
            "factor_mean": float(factor.mean()),
            "factor_std": float(factor.std(unbiased=False)),
            "shell_factor_mean": float(expected_shell_factor.mean()),
            "shell_factor_std": float(expected_shell_factor.std(unbiased=False)),
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
