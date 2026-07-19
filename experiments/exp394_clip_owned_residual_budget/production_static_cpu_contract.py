#!/usr/bin/env python3
"""Static/CPU exact contract for the fresh exp394 production implementation."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


BASELINE_COMMIT = "f5de34027a7b33a3a19c8f2b0707658bf13e6410"
RHO_STAR = 0.08075544983148575
PROTECTED_SHA256 = {
    "model/backbones/swin_transformer.py": (
        "b389b7243e204d851ed365c986c8c4077d7fa86ce79e6cbb0be6fc4a1ba58eef"
    ),
    "datasets/pose_dataset.py": (
        "d04e74908d18eaf8105f9b85c66287cac6980ddf5ffe8132e855c7d5a9f61bbc"
    ),
}
TARGET_FILES = (
    "model/tapf.py",
    "model/clip_semantic_teacher.py",
    "model/make_model.py",
    "processor/processor.py",
    "config/defaults.py",
    "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_baseline_tapf(repo_root):
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_COMMIT}:model/tapf.py"],
        cwd=repo_root,
        text=True,
    )
    module = types.ModuleType("exp394_baseline_tapf")
    exec(compile(source, "baseline:model/tapf.py", "exec"), module.__dict__)
    return module


def load_current_tapf(repo_root):
    path = repo_root / "model/tapf.py"
    spec = importlib.util.spec_from_file_location("exp394_current_tapf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def states_exact(left, right):
    left_state = left.state_dict()
    right_state = right.state_dict()
    return (
        tuple(left_state) == tuple(right_state)
        and all(torch.equal(left_state[key], right_state[key]) for key in left_state)
    )


def legacy_forward(module, name, instance, tensors):
    source, early_source, tokens, early_tokens = tensors
    instance.eval()
    if name == "ht0":
        early = instance.prepare_early(
            early_source, None, (32, 16), None, False
        )
        late = instance.prepare(source, None, (32, 16), None, False)
        early_output = early_tokens.clone()
        for index in range(len(instance.early_psg_bank)):
            early_output = instance.apply_early_gate(
                index, early_output, (2, 2), early
            )
        combined = instance.combine_states(early, late)
        output = tokens.clone()
        for index in range(2):
            output = instance.apply_gate(index, output, (2, 2), combined)
        return output, early_output
    state = instance.prepare(source, None, (32, 16), None, False)
    output = tokens.clone()
    for index in range(2):
        output = instance.apply_gate(index, output, (2, 2), state)
    return output, state["consumer_field"]


def legacy_parity(repo_root, current):
    baseline = load_baseline_tapf(repo_root)
    cases = (
        (
            "d0",
            "CleanTapfD0",
            dict(
                anchor_channels=8,
                anchor_hidden=16,
                consumer_channels=12,
                psg_hidden=8,
            ),
        ),
        (
            "ht0",
            "CleanTapfHt0",
            dict(
                anchor_channels=8,
                anchor_hidden=16,
                consumer_channels=12,
                psg_hidden=8,
                early_anchor_channels=6,
                early_consumer_channels=10,
                early_consumer_count=2,
            ),
        ),
        (
            "semantic_c0",
            "CleanSemanticTapfC0",
            dict(
                anchor_channels=8,
                anchor_hidden=16,
                consumer_channels=12,
                router_rank=4,
                router_rezero=False,
            ),
        ),
        (
            "rz_c0",
            "CleanSemanticTapfC0",
            dict(
                anchor_channels=8,
                anchor_hidden=16,
                consumer_channels=12,
                router_rank=4,
                router_rezero=True,
            ),
        ),
    )
    generator = torch.Generator().manual_seed(20260719)
    tensors = (
        torch.randn(3, 8, 4, 2, generator=generator),
        torch.randn(3, 6, 4, 2, generator=generator),
        torch.randn(3, 4, 12, generator=generator),
        torch.randn(3, 4, 10, generator=generator),
    )
    output = {}
    for index, (name, class_name, kwargs) in enumerate(cases):
        seed = 4200 + index
        torch.manual_seed(seed)
        old = getattr(baseline, class_name)(**kwargs)
        torch.manual_seed(seed)
        new = getattr(current, class_name)(**kwargs)
        old_output = legacy_forward(baseline, name, old, tensors)
        new_output = legacy_forward(current, name, new, tensors)
        output[name] = {
            "state_exact": states_exact(old, new),
            "forward_exact": all(
                torch.equal(left, right)
                for left, right in zip(old_output, new_output)
            ),
            "state_keys": len(old.state_dict()),
        }
    return output


def make_teacher_targets(batch, high_hw, generator):
    high_height, high_width = high_hw
    masks = torch.zeros(batch, 5, high_height, high_width)
    boundaries = torch.linspace(0, high_height, 6).round().long()
    for slot in range(5):
        masks[:, slot, boundaries[slot]:boundaries[slot + 1]] = 1.0
    evidence = F.normalize(
        torch.randn(batch, 5, 16, generator=generator), dim=-1
    )
    keypoints = torch.rand(batch, 17, 2, generator=generator)
    keypoints[..., 0] *= 127.0
    keypoints[..., 1] *= 383.0
    return {
        "keypoints": keypoints,
        "scores": torch.rand(batch, 17, generator=generator) * 0.5 + 0.5,
        "valid": torch.ones(batch, 17, dtype=torch.bool),
        "semantic_teacher_evidence": evidence,
        "semantic_valid": torch.ones(batch, 5, dtype=torch.bool),
        "semantic_teacher_mask": masks,
    }


def group_parameters(model, token_backbone, id_head):
    anchor_trunk = [
        *model.anchor.project.parameters(),
        *model.anchor.depthwise.parameters(),
        *model.anchor.norm.parameters(),
    ]
    anchor_targets = [
        *model.anchor.region_mask_head.parameters(),
        *model.anchor.presence_head.parameters(),
    ]
    router_projections = []
    router_experts = []
    for router in model.psg_bank:
        router_projections.extend(router.token_projection.parameters())
        router_projections.extend(router.context_projection.parameters())
        router_projections.extend(router.evidence_projection.parameters())
        router_experts.extend(router.experts.parameters())
    return {
        "source_backbone": [],
        "token_backbone": list(token_backbone.parameters()),
        "anchor_trunk": anchor_trunk,
        "anchor_targets": anchor_targets,
        "evidence_head": list(model.anchor.evidence_head.parameters()),
        "router_projections": router_projections,
        "router_experts": router_experts,
        "id_head": list(id_head.parameters()),
    }


def gradient_report(groups, loss, modules):
    for module in modules:
        module.zero_grad(set_to_none=True)
    loss.backward()
    report = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters]
        present = [gradient for gradient in gradients if gradient is not None]
        nonzero = [
            gradient for gradient in present
            if bool(torch.count_nonzero(gradient))
        ]
        report[name] = {
            "parameters": len(parameters),
            "grad_present": len(present),
            "grad_nonzero": len(nonzero),
            "grad_abs_max": max(
                (float(gradient.abs().max()) for gradient in present),
                default=0.0,
            ),
        }
    return report


def active(report, name):
    item = report[name]
    return (
        item["parameters"] > 0
        and item["grad_present"] == item["parameters"]
        and item["grad_nonzero"] == item["parameters"]
        and item["grad_abs_max"] > 0
    )


def off(report, name):
    item = report[name]
    return item["grad_present"] == 0 and item["grad_nonzero"] == 0


def production_contract(current):
    generator = torch.Generator().manual_seed(20260719)
    batch = 4
    source_input = torch.randn(batch, 7, 4, 2, generator=generator)
    source_backbone = nn.Conv2d(7, 8, 1, bias=False)
    token_input = torch.randn(batch, 4, 9, generator=generator)
    token_backbone = nn.Linear(9, 12, bias=False)
    id_head = nn.Linear(12, 3, bias=False)
    pose_batch = make_teacher_targets(batch, (16, 8), generator)

    torch.manual_seed(7301)
    model = current.CleanRichEvidenceBudgetTapf(
        anchor_channels=8,
        anchor_hidden=16,
        consumer_channels=12,
        router_rank=4,
        rho_star=RHO_STAR,
    )
    groups = group_parameters(model, token_backbone, id_head)
    groups["source_backbone"] = list(source_backbone.parameters())

    def forward(epoch, training=True):
        model.train(training)
        source = source_backbone(source_input)
        tokens = token_backbone(token_input)
        state = model.prepare(
            source,
            pose_batch if training else None,
            image_hw=(384, 128),
            epoch=epoch,
            training=training,
        )
        output = tokens
        for bank in range(2):
            output = model.apply_gate(bank, output, (2, 2), state)
        return output, state, tokens

    zero_output, zero_state, zero_tokens = forward(1)
    handoff_output, handoff_state, handoff_tokens = forward(6)
    fixed_output, fixed_state, fixed_tokens = forward(10)
    schedule = [model.rho_at_epoch(epoch, True) for epoch in range(1, 12)]
    schedule_repeat = [model.rho_at_epoch(epoch, True) for epoch in range(1, 12)]

    router = model.psg_bank[0]
    direct_tokens = torch.randn(batch, 4, 12, generator=generator)
    direct_masks = torch.zeros(batch, 5, 2, 2)
    direct_masks[:, :, 0, 0] = 1.0
    direct_presence = torch.ones(batch, 5)
    direct_evidence = F.normalize(
        torch.randn(batch, 5, 16, generator=generator), dim=-1
    )
    null_mask_output, null_mask_delta, null_mask_branch = router(
        direct_tokens,
        (2, 2),
        torch.zeros_like(direct_masks),
        direct_presence,
        direct_evidence,
        RHO_STAR,
    )
    null_presence_output, null_presence_delta, null_presence_branch = router(
        direct_tokens,
        (2, 2),
        direct_masks,
        torch.zeros_like(direct_presence),
        direct_evidence,
        RHO_STAR,
    )
    correct = router.branch(
        direct_tokens, (2, 2), direct_masks, direct_presence, direct_evidence
    )
    wrong = router.branch(
        direct_tokens,
        (2, 2),
        direct_masks,
        direct_presence,
        direct_evidence.roll(1, 1),
    )
    static = router.branch(
        direct_tokens,
        (2, 2),
        direct_masks,
        direct_presence,
        direct_evidence.mean(0, keepdim=True).expand_as(direct_evidence),
    )
    valid_normalized = correct["normalized_proposal"][correct["slot_valid"]]
    rms = valid_normalized.float().square().mean(-1).sqrt()

    evidence_output, evidence_state, _ = forward(1)
    evidence_loss = (
        evidence_state["evidence_cos_loss"]
        + evidence_state["evidence_relation_loss"]
    )
    evidence_grad = gradient_report(
        groups, evidence_loss, (model, source_backbone, token_backbone, id_head)
    )
    mask_output, mask_state, _ = forward(1)
    mask_loss = mask_state["region_mask_loss"] + mask_state["presence_loss"]
    mask_grad = gradient_report(
        groups, mask_loss, (model, source_backbone, token_backbone, id_head)
    )
    exec_output, exec_state, _ = forward(1)
    exec_grad = gradient_report(
        groups,
        exec_state["exec_loss"],
        (model, source_backbone, token_backbone, id_head),
    )
    reid_output, reid_state, _ = forward(6)
    reid_loss = F.cross_entropy(id_head(reid_output.mean(1)), torch.arange(4) % 3)
    reid_grad = gradient_report(
        groups, reid_loss, (model, source_backbone, token_backbone, id_head)
    )

    state_payload = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    torch.manual_seed(99)
    reloaded = current.CleanRichEvidenceBudgetTapf(
        anchor_channels=8,
        anchor_hidden=16,
        consumer_channels=12,
        router_rank=4,
        rho_star=RHO_STAR,
    )
    incompatible = reloaded.load_state_dict(state_payload, strict=True)
    model.eval()
    reloaded.eval()
    source_eval = source_backbone(source_input).detach()
    tokens_eval = token_backbone(token_input).detach()

    def eval_output(instance):
        state = instance.prepare(
            source_eval, None, (384, 128), None, False
        )
        output = tokens_eval.clone()
        for bank in range(2):
            output = instance.apply_gate(bank, output, (2, 2), state)
        return output, state

    original_eval, original_eval_state = eval_output(model)
    reloaded_eval, reloaded_eval_state = eval_output(reloaded)

    semantic_formula = torch.stack(
        [
            fixed_state["region_mask_loss"],
            fixed_state["presence_loss"],
            fixed_state["evidence_cos_loss"],
            fixed_state["evidence_relation_loss"],
            fixed_state["exec_loss"],
        ]
    ).mean()
    pose_formula = (
        fixed_state["heatmap_loss"]
        + fixed_state["confidence_loss"]
        + semantic_formula
    )
    state_names = tuple(model.state_dict())
    gradient_gates = {
        "evidence_loss_updates_evidence_head": active(
            evidence_grad, "evidence_head"
        ),
        "evidence_loss_blocks_anchor_trunk": off(evidence_grad, "anchor_trunk"),
        "evidence_loss_blocks_backbones_router_id": all(
            off(evidence_grad, name)
            for name in (
                "source_backbone",
                "token_backbone",
                "router_projections",
                "router_experts",
                "id_head",
            )
        ),
        "mask_presence_updates_anchor": (
            active(mask_grad, "anchor_trunk")
            and active(mask_grad, "anchor_targets")
        ),
        "mask_presence_blocks_evidence_router_backbones_id": all(
            off(mask_grad, name)
            for name in (
                "source_backbone",
                "token_backbone",
                "evidence_head",
                "router_projections",
                "router_experts",
                "id_head",
            )
        ),
        "exec_updates_evidence_and_both_routers": (
            active(exec_grad, "evidence_head")
            and active(exec_grad, "router_projections")
            and active(exec_grad, "router_experts")
        ),
        "exec_blocks_backbones_anchor_id": all(
            off(exec_grad, name)
            for name in (
                "source_backbone",
                "token_backbone",
                "anchor_trunk",
                "anchor_targets",
                "id_head",
            )
        ),
        "reid_updates_token_backbone_routers_id": (
            active(reid_grad, "token_backbone")
            and active(reid_grad, "router_projections")
            and active(reid_grad, "router_experts")
            and active(reid_grad, "id_head")
        ),
        "reid_blocks_source_anchor_evidence": all(
            off(reid_grad, name)
            for name in (
                "source_backbone",
                "anchor_trunk",
                "anchor_targets",
                "evidence_head",
            )
        ),
    }
    checks = {
        "rho_zero_full_bypass_exact": torch.equal(zero_output, zero_tokens),
        "handoff_nonzero": not torch.equal(handoff_output, handoff_tokens),
        "fixed_nonzero": not torch.equal(fixed_output, fixed_tokens),
        "schedule": schedule,
        "schedule_repeat_exact": schedule == schedule_repeat,
        "schedule_teacher_zero": schedule[:5] == [0.0] * 5,
        "schedule_handoff_exact": schedule[5:9] == [
            RHO_STAR * step / 5.0 for step in range(1, 5)
        ],
        "schedule_fixed": schedule[9:] == [RHO_STAR, RHO_STAR],
        "eval_rho_fixed": original_eval_state["rho"] == RHO_STAR,
        "rho_not_parameter_or_buffer": all(
            "rho" not in name.lower()
            for name, _ in tuple(model.named_parameters())
            + tuple(model.named_buffers())
        ),
        "null_mask_identity": torch.equal(null_mask_output, direct_tokens),
        "null_mask_delta_zero": torch.equal(
            null_mask_delta, torch.zeros_like(null_mask_delta)
        ),
        "null_mask_normalized_zero": torch.equal(
            null_mask_branch["normalized_proposal"],
            torch.zeros_like(null_mask_branch["normalized_proposal"]),
        ),
        "null_presence_identity": torch.equal(
            null_presence_output, direct_tokens
        ),
        "null_presence_delta_zero": torch.equal(
            null_presence_delta, torch.zeros_like(null_presence_delta)
        ),
        "null_presence_normalized_zero": torch.equal(
            null_presence_branch["normalized_proposal"],
            torch.zeros_like(null_presence_branch["normalized_proposal"]),
        ),
        "rms_finite": bool(torch.isfinite(rms).all()),
        "rms_near_one": float((rms.detach() - 1.0).abs().max()) < 1e-3,
        "wrong_changes_proposal": not torch.equal(
            correct["proposal"], wrong["proposal"]
        ),
        "static_changes_sample_signal": not torch.equal(
            correct["proposal"], static["proposal"]
        ),
        "two_independent_consumers": all(
            left.data_ptr() != right.data_ptr()
            for left, right in zip(
                model.psg_bank[0].parameters(), model.psg_bank[1].parameters()
            )
        ),
        "consumer_inputs_detached": (
            not fixed_state["consumer_mask"].requires_grad
            and not fixed_state["consumer_presence"].requires_grad
            and not fixed_state["consumer_evidence"].requires_grad
        ),
        "semantic_formula_exact": torch.equal(
            semantic_formula, fixed_state["semantic_loss"]
        ),
        "pose_formula_exact": torch.equal(pose_formula, fixed_state["pose_loss"]),
        "strict_reload_clean": (
            not incompatible.missing_keys and not incompatible.unexpected_keys
        ),
        "strict_reload_forward_exact": torch.equal(
            original_eval, reloaded_eval
        ),
        "teacher_absent_from_state": all(
            fragment not in name.lower().split(".")
            for name in state_names
            for fragment in ("teacher", "clip", "codebook", "text", "text_encoder")
        ),
        "evidence_head_retained": any(
            "anchor.evidence_head" in name for name in state_names
        ),
        "two_evidence_routers_retained": (
            any("psg_bank.0.evidence_projection" in name for name in state_names)
            and any("psg_bank.1.evidence_projection" in name for name in state_names)
        ),
        "all_finite": all(
            bool(torch.isfinite(value).all())
            for value in (
                zero_output,
                handoff_output,
                fixed_output,
                original_eval,
                reloaded_eval,
                fixed_state["pose_loss"],
            )
        ),
    }
    checks.update(gradient_gates)
    diagnostics = {
        "handoff_gap_max_abs": float(
            (handoff_output.detach() - handoff_tokens.detach()).abs().max()
        ),
        "fixed_gap_max_abs": float(
            (fixed_output.detach() - fixed_tokens.detach()).abs().max()
        ),
        "rms_max_abs_from_one": float((rms.detach() - 1.0).abs().max()),
        "correct_wrong_proposal_max_abs": float(
            (
                correct["proposal"].detach() - wrong["proposal"].detach()
            ).abs().max()
        ),
        "correct_static_proposal_max_abs": float(
            (
                correct["proposal"].detach() - static["proposal"].detach()
            ).abs().max()
        ),
        "state_keys": len(state_names),
        "gradient_reports": {
            "evidence": evidence_grad,
            "mask_presence": mask_grad,
            "exec": exec_grad,
            "reid": reid_grad,
        },
    }
    return checks, diagnostics


def source_contract(repo_root):
    config = (
        repo_root
        / "configs/occluded_duke/swin_tiny_tapf_rich_budget_c0.yml"
    ).read_text(encoding="utf-8")
    defaults = (repo_root / "config/defaults.py").read_text(encoding="utf-8")
    teacher = (repo_root / "model/clip_semantic_teacher.py").read_text(
        encoding="utf-8"
    )
    processor = (repo_root / "processor/processor.py").read_text(encoding="utf-8")
    make_model = (repo_root / "model/make_model.py").read_text(encoding="utf-8")
    return {
        "protected_blobs_exact": all(
            sha256_file(repo_root / relative) == expected
            for relative, expected in PROTECTED_SHA256.items()
        ),
        "new_default_off": "_C.MODEL.TAPF.RICH_EVIDENCE_ENABLED = False"
        in defaults,
        "new_config_on": "RICH_EVIDENCE_ENABLED: True" in config,
        "rho_frozen_in_config": f"RESIDUAL_RHO: {RHO_STAR}" in config,
        "codebook_sha_frozen": (
            "fb87da370ea945d526f499bef78093a6b07203d87c6d84efe06b5eb6594f954a"
            in config
        ),
        "checkpoint_sha_frozen": (
            "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
            in config
        ),
        "new_checkpoint_not_old_repo": "/home/afr/SOLIDER-REID" not in config,
        "teacher_image_only_state": (
            "class FrozenRichClipEvidenceTeacher" in teacher
            and "self.text" not in teacher.split(
                "class FrozenRichClipEvidenceTeacher", 1
            )[1]
        ),
        "teacher_inference_mode": (
            "@torch.inference_mode()" in teacher.split(
                "class FrozenRichClipEvidenceTeacher", 1
            )[1]
        ),
        "teacher_external_train_only": (
            "FrozenRichClipEvidenceTeacher" in processor
            and "semantic_teacher = None" in processor
            and "def do_inference" in processor
        ),
        "checkpoint_model_only": "torch.save(model.state_dict()" in processor,
        "new_model_selection_guarded": (
            "cfg.MODEL.TAPF.RICH_EVIDENCE_ENABLED" in make_model
            and "CleanRichEvidenceBudgetTapf" in make_model
            and "Rich evidence TAPF requires SEMANTIC_ENABLED" in make_model
        ),
        "no_new_loss_weights": all(
            token not in defaults
            for token in (
                "EVIDENCE_LOSS_WEIGHT",
                "EXEC_LOSS_WEIGHT",
                "RELATION_LOSS_WEIGHT",
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    torch.manual_seed(20260719)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    current = load_current_tapf(repo_root)

    legacy = legacy_parity(repo_root, current)
    production, diagnostics = production_contract(current)
    source = source_contract(repo_root)
    gates = {
        "legacy_state_forward_parity": all(
            item["state_exact"] and item["forward_exact"]
            for item in legacy.values()
        ),
        "production_exact": all(production.values()),
        "source_static": all(source.values()),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "legacy": legacy,
        "production": production,
        "source": source,
        "diagnostics": diagnostics,
        "baseline_commit": BASELINE_COMMIT,
        "rho_star": RHO_STAR,
        "protected_sha256": {
            relative: sha256_file(repo_root / relative)
            for relative in PROTECTED_SHA256
        },
        "target_sha256": {
            relative: sha256_file(repo_root / relative)
            for relative in TARGET_FILES
        },
        "script_sha256": sha256_file(__file__),
        "torch_version": torch.__version__,
    }
    result["contract_sha256"] = sha256_json(result)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
