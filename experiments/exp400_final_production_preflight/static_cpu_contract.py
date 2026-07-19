#!/usr/bin/env python3
"""CPU-only exact contract for exp400 final production preflight."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import traceback
from pathlib import Path

import torch


SOURCE_COMMIT = "11d7a35788c4645c355d96d76a2a4ff20a9801ac"
EXPECTED_REPORTER_SHA256 = (
    "6e8b2b67efcda7cfaca8527fb0ae1dd4c6aedcebef3fec6ded2e5ba6ddab8164"
)
EXPECTED_SOURCE_SHA256 = {
    "model/tapf.py": "95c5d0ff80bf9e4529589a5f31819e7aad5db644b88e2a33d6af07c9ffc42886",
    "model/clip_semantic_teacher.py": "c648fa768b178d153258c46eee69679cbc0b90a11db918800323ab5c5c6054d5",
    "model/make_model.py": "6bc7d9c83a2f4d12b78dd2c09335d366ce568107ddce5dded3abfe7ca8538f03",
    "processor/processor.py": "be1c19ea5af19534e3855eb2a5914e0dc9a5643c63a39cfa508c81f89660eac1",
    "config/defaults.py": "a13e5f6df0e8c770c254c115d6d55208baac7938cffbec6f208ba9caa24dd7c5",
    "model/backbones/swin_transformer.py": "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef",
    "datasets/pose_dataset.py": "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc",
    "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml":
        "e0413a497976ad6dbf4c74cf13b55c86c169d659bab6d967455e87c592e47f4e",
    "configs/occluded_duke/swin_tiny_tapf_d0.yml":
        "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_text(command, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"exp400_static_{path.stem}", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def function_source(source, tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise KeyError(name)


def call_paths(tree):
    paths = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        parts = []
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        paths.append(".".join(reversed(parts)))
    return paths


class TinyRichModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Module()
        self.base.backbone_weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.base.tapf = torch.nn.Module()
        self.base.tapf.anchor = torch.nn.Module()
        self.base.tapf.anchor.project = torch.nn.Linear(1, 1, bias=False)
        self.base.tapf.anchor.pose_head = torch.nn.Linear(1, 1, bias=False)
        self.base.tapf.anchor.region_mask_head = torch.nn.Linear(1, 1, bias=False)
        self.base.tapf.anchor.presence_head = torch.nn.Linear(1, 1, bias=False)
        self.base.tapf.anchor.evidence_head = torch.nn.Linear(1, 1, bias=False)
        routers = []
        for _ in range(2):
            router = torch.nn.Module()
            router.token_projection = torch.nn.Linear(1, 1, bias=False)
            router.context_projection = torch.nn.Linear(1, 1, bias=False)
            router.evidence_projection = torch.nn.Linear(1, 1, bias=False)
            router.experts = torch.nn.Linear(1, 1, bias=False)
            routers.append(router)
        self.base.tapf.psg_bank = torch.nn.ModuleList(routers)
        self.classifier = torch.nn.Linear(1, 1, bias=False)


class TinyBudgetTapf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rho_star = 0.08
        self.anchor = torch.nn.Module()
        self.anchor.evidence_head = torch.nn.Linear(1, 1, bias=False)
        self.psg_bank = torch.nn.ModuleList()
        for weight in (0.25, 0.5):
            router = torch.nn.Module()
            router.evidence_projection = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                router.evidence_projection.weight.fill_(weight)
            self.psg_bank.append(router)

    def rho_at_epoch(self, epoch, training):
        if not training:
            return self.rho_star
        if epoch <= 5:
            return 0.0
        if epoch >= 10:
            return self.rho_star
        return self.rho_star * (epoch - 5) / 5.0

    def apply_gate(self, bank_index, tokens, hw_shape, state):
        del hw_shape
        delta = state["rho"] * self.psg_bank[bank_index].evidence_projection(
            tokens
        )
        state["gate_deltas"].append(delta)
        return tokens + delta


class TinyTerminalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Module()
        self.base.tapf = TinyBudgetTapf()

    def forward(
        self,
        image,
        label=None,
        cam_label=None,
        view_label=None,
        pose_batch=None,
        tapf_epoch=None,
    ):
        del label, cam_label, view_label
        if self.training:
            if pose_batch is None:
                raise ValueError("training requires paired targets")
            state = {
                "rho": self.base.tapf.rho_at_epoch(tapf_epoch, True),
                "gate_deltas": [],
            }
            tokens = self.base.tapf.apply_gate(0, image, None, state)
            tokens = self.base.tapf.apply_gate(1, tokens, None, state)
            return tokens, tokens, None, state
        return image, None


class TinyRngBase:
    @staticmethod
    def restore_rng(state):
        torch.set_rng_state(state.clone())


def raises_state_error(module, reporter, items, expected):
    try:
        module.parameter_group_state(reporter, {"test": items})
    except expected:
        return True
    return False


def synthetic_group_state(module, inactive=()):
    initial = {name: f"{name}:initial" for name in module.RICH_SPECIFIC_GROUPS}
    final = {
        name: initial[name] if name in inactive else f"{name}:final"
        for name in module.RICH_SPECIFIC_GROUPS
    }
    return {"initial": initial, "final": final}


def synthetic_steps(module, skips, rich_nonfinite=None, inactive=()):
    rich_nonfinite = rich_nonfinite or {}
    rows = []
    scale = 65536.0
    for attempt in range(1, 33):
        skipped = attempt in skips or attempt in rich_nonfinite
        before = scale
        after = before * 0.5 if skipped else before
        extra_groups = list(rich_nonfinite.get(attempt, ()))
        nonfinite_groups = (["backbone"] if attempt in skips else []) + extra_groups
        report = {
            name: {
                "all_finite": name not in extra_groups,
                "grad_nonzero_tensors": int(
                    attempt > 16 and not skipped and name not in inactive
                ),
            }
            for name in module.RICH_SPECIFIC_GROUPS
        }
        rows.append(
            {
                "attempt": attempt,
                "tapf_epoch": 1 if attempt <= 16 else 6,
                "scale_before": before,
                "scale_after": after,
                "had_nonfinite": skipped,
                "nonfinite_groups": nonfinite_groups,
                "gradient_report": report,
                "optimizer_succeeded": not skipped,
                "optimizer_skipped": skipped,
                "optimizer_step_calls_delta": 0 if skipped else 1,
            }
        )
        scale = after
    return rows


def run_contract(repo_root: Path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("exp400 static contract requires hidden CUDA")
    cuda_before = torch.cuda.is_initialized()
    script_path = (
        repo_root
        / "experiments/exp400_final_production_preflight/cuda_final_production_preflight.py"
    )
    reporter_path = (
        repo_root
        / "experiments/exp396_chunk_safe_amp_attribution/cuda_amp_attribution.py"
    )
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = call_paths(tree)
    arm_source = function_source(source, tree, "run_dynamic_arm")
    gate_source = function_source(source, tree, "run_gate")
    descriptor_source = function_source(source, tree, "descriptor_variant")
    terminal_source = function_source(source, tree, "run_terminal_audit")
    main_source = function_source(source, tree, "main")
    module = load_module(script_path)
    reporter = load_module(reporter_path)

    torch.manual_seed(3991234)
    tiny_model = TinyRichModel()
    tiny_optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
    real_groups, real_names, real_coverage = reporter.parameter_groups(
        tiny_model, tiny_optimizer, "rich"
    )
    real_state_before = module.parameter_group_state(reporter, real_groups)
    real_state_repeat = module.parameter_group_state(reporter, real_groups)
    with torch.no_grad():
        real_groups["evidence_head"][0][1].add_(1.0)
    real_state_after = module.parameter_group_state(reporter, real_groups)
    real_changed_groups = sorted(
        name
        for name in real_state_before
        if real_state_before[name] != real_state_after[name]
    )

    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    ordered = module.parameter_group_state(
        reporter, {"test": [("first", first), ("second", second)]}
    )["test"]
    ordered_repeat = module.parameter_group_state(
        reporter, {"test": [("first", first), ("second", second)]}
    )["test"]
    reordered = module.parameter_group_state(
        reporter, {"test": [("second", second), ("first", first)]}
    )["test"]
    renamed = module.parameter_group_state(
        reporter, {"test": [("renamed", first), ("second", second)]}
    )["test"]
    with torch.no_grad():
        first.add_(1.0)
    value_changed = module.parameter_group_state(
        reporter, {"test": [("first", first), ("second", second)]}
    )["test"]
    empty_first = module.parameter_group_state(reporter, {"test": []})["test"]
    empty_repeat = module.parameter_group_state(reporter, {"test": []})["test"]

    source_sha = {
        relative: sha256_file(repo_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    shared_skips = {1, 2, 3, 4, 5, 17}
    base_rows = synthetic_steps(module, shared_skips)
    matched = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(module, shared_skips),
        synthetic_group_state(module),
    )
    extra_skip = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(module, shared_skips | {10}),
        synthetic_group_state(module),
    )
    persistent_rows = synthetic_steps(module, shared_skips | {16, 32})
    persistent_tail = module.evaluate_trajectories(
        persistent_rows,
        synthetic_steps(module, shared_skips | {16, 32}),
        synthetic_group_state(module),
    )
    rich_specific = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(
            module,
            shared_skips,
            rich_nonfinite={18: ("evidence_head",)},
        ),
        synthetic_group_state(module),
    )
    inactive = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(module, shared_skips, inactive=("evidence_head",)),
        synthetic_group_state(module, inactive=("evidence_head",)),
    )
    rich_better = module.evaluate_trajectories(
        base_rows,
        synthetic_steps(module, {1, 2, 3, 4, 17}),
        synthetic_group_state(module),
    )

    torch.manual_seed(4001234)
    toy_model = TinyTerminalModel()
    toy_state = module.tensor_state_cpu(toy_model)
    toy_rng = torch.get_rng_state().clone()
    toy_input = torch.tensor([[1.0], [2.0]])
    toy_target = torch.tensor([0, 1])
    toy_camid = torch.tensor([0, 1])
    toy_viewid = torch.tensor([0, 0])
    toy_pose = {"paired": torch.ones(2, 1)}
    tiny_base = TinyRngBase()
    override_before = "apply_gate" in toy_model.base.tapf.__dict__
    epoch1_full = module.descriptor_variant(
        tiny_base,
        toy_model,
        toy_state,
        toy_rng,
        toy_input,
        toy_target,
        toy_camid,
        toy_viewid,
        toy_pose,
        1,
        (),
    )
    epoch1_bypass = module.descriptor_variant(
        tiny_base,
        toy_model,
        toy_state,
        toy_rng,
        toy_input,
        toy_target,
        toy_camid,
        toy_viewid,
        toy_pose,
        1,
        (0, 1),
    )
    epoch6_full = module.descriptor_variant(
        tiny_base,
        toy_model,
        toy_state,
        toy_rng,
        toy_input,
        toy_target,
        toy_camid,
        toy_viewid,
        toy_pose,
        6,
        (),
    )
    epoch6_bypass0 = module.descriptor_variant(
        tiny_base,
        toy_model,
        toy_state,
        toy_rng,
        toy_input,
        toy_target,
        toy_camid,
        toy_viewid,
        toy_pose,
        6,
        (0,),
    )
    epoch6_bypass1 = module.descriptor_variant(
        tiny_base,
        toy_model,
        toy_state,
        toy_rng,
        toy_input,
        toy_target,
        toy_camid,
        toy_viewid,
        toy_pose,
        6,
        (1,),
    )
    override_after = "apply_gate" in toy_model.base.tapf.__dict__
    toy_state_after = module.tensor_state_cpu(toy_model)
    toy_state_exact = set(toy_state) == set(toy_state_after) and all(
        torch.equal(toy_state[name], toy_state_after[name]) for name in toy_state
    )
    reload_rng = torch.get_rng_state().clone()
    reloaded_toy = TinyTerminalModel()
    incompatible = reloaded_toy.load_state_dict(toy_state, strict=True)
    torch.set_rng_state(reload_rng)
    reloaded_state = module.tensor_state_cpu(reloaded_toy)
    strict_reload_exact = (
        not incompatible.missing_keys
        and not incompatible.unexpected_keys
        and set(toy_state) == set(reloaded_state)
        and all(
            torch.equal(toy_state[name], reloaded_state[name])
            for name in toy_state
        )
    )
    module.ExplodingPose.accesses = 0
    toy_model.load_state_dict(toy_state, strict=True)
    eval_plain = module.eval_descriptor(
        toy_model, toy_input, toy_camid, toy_viewid, None
    )
    eval_exploding = module.eval_descriptor(
        toy_model,
        toy_input,
        toy_camid,
        toy_viewid,
        module.ExplodingPose(),
    )

    gates = {
        "source_commit_exists": run_text(
            ["git", "cat-file", "-t", SOURCE_COMMIT], repo_root
        ) == "commit",
        "source_sha_exact": source_sha == EXPECTED_SOURCE_SHA256,
        "reporter_dependency_sha_exact": sha256_file(reporter_path)
        == EXPECTED_REPORTER_SHA256,
        "real_parameter_groups_coverage_exact": real_coverage["exact"]
        and len(real_groups) == len(reporter.GROUP_NAMES) == 15
        and all(real_groups[name] for name in reporter.GROUP_NAMES),
        "real_container_named_tuple_exact": all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and bool(item[0])
            and isinstance(item[1], torch.nn.Parameter)
            for items in real_groups.values()
            for item in items
        )
        and real_names
        == {name: [item[0] for item in real_groups[name]] for name in real_groups},
        "real_container_repeat_sha_exact": real_state_before == real_state_repeat,
        "real_container_single_group_change_exact": real_changed_groups
        == ["evidence_head"],
        "name_order_value_binding_exact": (
            ordered == ordered_repeat
            and ordered != reordered
            and ordered != renamed
            and ordered != value_changed
        ),
        "empty_group_stable": empty_first == empty_repeat,
        "duplicate_name_rejected": raises_state_error(
            module,
            reporter,
            [("dup", first), ("dup", second)],
            ValueError,
        ),
        "empty_name_rejected": raises_state_error(
            module, reporter, [("", first)], TypeError
        ),
        "bare_parameter_rejected": raises_state_error(
            module, reporter, [first], TypeError
        ),
        "wrong_tuple_length_rejected": raises_state_error(
            module, reporter, [("first",)], TypeError
        ),
        "nonparameter_value_rejected": raises_state_error(
            module, reporter, [("first", torch.tensor([1.0]))], TypeError
        ),
        "attempts_and_schedule_exact": (
            module.ATTEMPTS == 32
            and module.STAGE_LENGTH == 16
            and module.STAGE_TAIL == 8
            and tuple(module.TAPF_EPOCHS) == (1,) * 16 + (6,) * 16
        ),
        "rich_specific_group_count_exact": len(module.RICH_SPECIFIC_GROUPS) == 11,
        "fresh_default_scaler_only": (
            arm_source.count('torch.amp.GradScaler("cuda")') == 1
            and "init_scale" not in source
            and "growth_factor" not in source
            and "backoff_factor" not in source
            and "growth_interval" not in source
            and "._scale" not in source
            and "._growth_tracker" not in source
        ),
        "native_step_and_update_once_in_loop": (
            arm_source.count("scaler.step(optimizer)") == 1
            and arm_source.count("scaler.update()") == 1
            and arm_source.index("scaler.step(optimizer)")
            < arm_source.index("scaler.update()")
        ),
        "unscale_and_report_before_step": (
            arm_source.index("scaler.unscale_(optimizer)")
            < arm_source.index("base.gradient_report(")
            < arm_source.index("scaler.step(optimizer)")
        ),
        "single_materialized_loader_for_both_arms": (
            "cpu_batches = [clone_cpu_batch(next(iterator))" in gate_source
            and gate_source.count("cpu_batches,") >= 2
        ),
        "matched_rng_states_present": (
            "step_rng = prepare_step_rng(base, device)" in gate_source
            and '"step_rng_entries_matched"' in gate_source
        ),
        "group_state_capture_present": (
            "initial_group_state = parameter_group_state" in arm_source
            and '"group_state_sha256"' in arm_source
            and '"rich_specific_group_state_updated"' in source
        ),
        "baseline_relative_evaluator_present": all(
            token in source
            for token in (
                '"stage_tail_steady_state"',
                '"no_rich_extra_skip_on_d0_success"',
                '"rich_success_not_below_d0"',
                '"rich_nonfinite_groups_shared_subset"',
                '"rich_specific_groups_e6_active"',
            )
        ),
        "no_scheduler_step": not any(
            path.endswith("scheduler.step") for path in calls
        ),
        "no_checkpoint_load_or_save": (
            "torch.load" not in calls and "torch.save" not in calls
        ),
        "formal_training_authorization_only_on_full_pass": (
            '"formal_training_authorized": passed' in gate_source
            and '"formal_training_authorized": False' in main_source
            and '"formal_training_authorized": True' not in source
        ),
        "production_preflight_authorization_only_on_full_pass": (
            '"production_preflight_authorized": passed' in gate_source
            and '"production_preflight_authorized": False' in main_source
        ),
        "terminal_helpers_ast_present": all(
            token in source
            for token in (
                "def tensor_state_cpu(",
                "def tensor_state_sha256(",
                "def descriptor_variant(",
                "class ExplodingPose(",
                "def eval_descriptor(",
                "def run_terminal_audit(",
            )
        ),
        "descriptor_finally_restores_state_rng_and_patch": (
            "finally:" in descriptor_source
            and 'tapf.__dict__.pop("apply_gate", None)' in descriptor_source
            and "model.load_state_dict(model_state, strict=True)" in descriptor_source
            and "base.restore_rng(saved_rng)" in descriptor_source
            and "model.zero_grad(set_to_none=True)" in descriptor_source
        ),
        "strict_reload_terminal_present": (
            "reloaded = make_model(" in terminal_source
            and "reloaded.load_state_dict(final_state, strict=True)"
            in terminal_source
            and '"strict_reload"' in terminal_source
            and '"reload_descriptor_exact"' in terminal_source
        ),
        "rgb_only_exploding_pose_terminal_present": all(
            token in terminal_source
            for token in (
                "correct_pose = pose6",
                "shuffled_pose =",
                "ExplodingPose()",
                '"rgb_correct_shuffle_exact"',
                '"rgb_correct_none_exact"',
                '"rgb_correct_exploding_exact"',
                '"exploding_pose_access_zero"',
            )
        ),
        "rho_and_two_consumer_bypass_terminal_present": all(
            token in terminal_source
            for token in (
                "bypass=(0, 1)",
                "bypass=(0,)",
                "bypass=(1,)",
                '"epoch1_rho_zero_exact"',
                '"epoch1_full_all_bypass_exact"',
                '"epoch6_rho_nonzero"',
                '"epoch6_all_bypass_nonzero"',
                '"epoch6_consumer0_nonzero"',
                '"epoch6_consumer1_nonzero"',
                '"all_bypass_mean_l2_positive"',
                '"consumer0_max_abs_positive"',
                '"consumer1_max_abs_positive"',
            )
        ),
        "terminal_isolation_finite_exact_present": all(
            token in terminal_source
            for token in (
                '"state_teacher_free"',
                '"evidence_head_retained"',
                '"two_routers_retained"',
                '"final_state_finite"',
                '"diagnostic_state_exact"',
                '"diagnostic_rng_exact"',
                '"apply_gate_restore_exact"',
                '"teacher_versions_exact"',
                '"teacher_state_exact"',
                '"teacher_grads_none"',
                '"codebook_state_exact"',
                '"source_sha_unchanged"',
                '"asset_sha_unchanged"',
                '"tracked_clean"',
            )
        ),
        "terminal_pass_required_by_validity": (
            '"terminal_pass": terminal["status"] == "PASS"' in gate_source
            and "passed = all(validity_gates.values())" in gate_source
        ),
        "exp400_fresh_assets_and_scratch": (
            '"exp400" in clip_path.name' in gate_source
            and '"exp400" in codebook_path.name' in gate_source
            and 'scratch_prefix = ".exp400_gradient_scratch_"' in gate_source
        ),
        "toy_rho0_identity": torch.equal(epoch1_full, epoch1_bypass)
        and toy_model.base.tapf.rho_at_epoch(1, True) == 0.0,
        "toy_two_consumers_nonzero": (
            not torch.equal(epoch6_full, epoch6_bypass0)
            and not torch.equal(epoch6_full, epoch6_bypass1)
            and toy_model.base.tapf.rho_at_epoch(6, True) > 0.0
        ),
        "toy_patch_and_state_restored": (
            override_before == override_after and not override_after
            and toy_state_exact
            and torch.equal(torch.get_rng_state(), toy_rng)
        ),
        "toy_strict_reload_exact": strict_reload_exact,
        "toy_rgb_only_exploding_exact": (
            torch.equal(eval_plain, eval_exploding)
            and module.ExplodingPose.accesses == 0
        ),
        "matched_synthetic_pass": matched["status"] == "PASS"
        and all(matched["gates"].values()),
        "rich_extra_skip_fails": extra_skip["status"] == "FAIL"
        and not extra_skip["gates"]["no_rich_extra_skip_on_d0_success"],
        "persistent_tail_failure_fails": persistent_tail["status"] == "FAIL"
        and not persistent_tail["gates"]["stage_tail_steady_state"],
        "rich_specific_nonfinite_fails": rich_specific["status"] == "FAIL"
        and not rich_specific["gates"]["rich_specific_groups_always_finite"]
        and not rich_specific["gates"]["rich_nonfinite_groups_shared_subset"],
        "rich_specific_inactive_fails": inactive["status"] == "FAIL"
        and not inactive["gates"]["rich_specific_groups_e6_active"]
        and not inactive["gates"]["rich_specific_group_state_updated"],
        "rich_better_than_d0_passes": rich_better["status"] == "PASS"
        and all(rich_better["gates"].values()),
        "sealed_boundaries_present": all(
            token in source
            for token in (
                '"exp394_remains_sealed": True',
                '"exp395_remains_sealed": True',
                '"exp396_remains_sealed": True',
                '"exp397_remains_sealed": True',
                '"exp398_remains_sealed": True',
                '"exp399_remains_sealed": True',
            )
        ),
    }
    cuda_after = torch.cuda.is_initialized()
    gates["cuda_never_initialized"] = not cuda_before and not cuda_after
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "git_head": run_text(["git", "rev-parse", "HEAD"], repo_root),
        "cuda_script_sha256": sha256_file(script_path),
        "static_script_sha256": sha256_file(Path(__file__).resolve()),
        "reporter_dependency_sha256": sha256_file(reporter_path),
        "source_sha256": source_sha,
        "real_parameter_group_names": real_names,
        "real_parameter_group_coverage": real_coverage,
        "real_parameter_group_state_before": real_state_before,
        "real_parameter_group_state_after": real_state_after,
        "real_parameter_group_changed": real_changed_groups,
        "matched_evaluation": matched,
        "extra_skip_evaluation": extra_skip,
        "persistent_tail_evaluation": persistent_tail,
        "rich_specific_failure_evaluation": rich_specific,
        "rich_specific_inactive_evaluation": inactive,
        "rich_better_evaluation": rich_better,
        "toy_terminal": {
            "epoch1_full_all_bypass_exact": torch.equal(
                epoch1_full, epoch1_bypass
            ),
            "epoch6_consumer0_nonzero": not torch.equal(
                epoch6_full, epoch6_bypass0
            ),
            "epoch6_consumer1_nonzero": not torch.equal(
                epoch6_full, epoch6_bypass1
            ),
            "patch_override_before": override_before,
            "patch_override_after": override_after,
            "state_restored": toy_state_exact,
            "strict_reload_exact": strict_reload_exact,
            "rgb_only_exploding_exact": torch.equal(
                eval_plain, eval_exploding
            ),
            "exploding_pose_accesses": module.ExplodingPose.accesses,
        },
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "checkpoint_count": 0,
        "gates": gates,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    runner = Path(args.runner).resolve()
    if output.exists() or runner.exists():
        raise RuntimeError("Refusing to overwrite exp400 static assets")
    try:
        result = run_contract(repo_root)
    except Exception as error:
        result = {
            "status": "INVALID",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "checkpoint_count": 0,
        }
    write_json(output, result)
    write_json(runner, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
