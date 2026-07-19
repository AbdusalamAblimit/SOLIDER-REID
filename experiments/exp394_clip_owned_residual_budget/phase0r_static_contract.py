#!/usr/bin/env python3
"""Synthetic CPU contract for exp394's evidence-budgeted residual.

This script is intentionally independent of the sealed execution repositories.
It validates the algebra, schedule, and loss ownership before any production
model/config implementation or CUDA work is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SEED = 20260719
BATCH = 4
TOKENS = 7
INPUT_DIM = 6
CHANNELS = 8
ANCHOR_DIM = 7
EVIDENCE_DIM = 4
HIDDEN = 9
SLOTS = 5
CLASSES = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rho_at_epoch(
    epoch: int,
    rho_star: float,
    teacher_epochs: int,
    handoff_epochs: int,
) -> float:
    """Epoch-only, non-learnable schedule with an exact-zero teacher stage."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if rho_star < 0 or teacher_epochs < 0 or handoff_epochs <= 0:
        raise ValueError("invalid budget schedule")
    if epoch <= teacher_epochs:
        return 0.0
    if epoch >= teacher_epochs + handoff_epochs:
        return float(rho_star)
    progress = (epoch - teacher_epochs) / float(handoff_epochs)
    return float(rho_star) * progress


class EvidenceBudgetRouter(nn.Module):
    """Minimal production branch with slot-specific experts and fixed budget."""

    def __init__(self) -> None:
        super().__init__()
        self.token_projection = nn.Linear(CHANNELS, HIDDEN, bias=False)
        self.context_projection = nn.Linear(CHANNELS, HIDDEN, bias=False)
        self.evidence_projection = nn.Linear(EVIDENCE_DIM, HIDDEN, bias=False)
        self.experts = nn.ModuleList(
            nn.Linear(HIDDEN, CHANNELS, bias=False) for _ in range(SLOTS)
        )

    def branch(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        presence: torch.Tensor,
        evidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mass = masks.sum(dim=-1, keepdim=True)
        context = torch.einsum("brp,bpc->brc", masks, tokens)
        context = context / mass.clamp_min(1.0)
        token_hidden = self.token_projection(tokens)

        proposals = []
        normalized = []
        for slot, expert in enumerate(self.experts):
            hidden = token_hidden
            hidden = hidden + self.context_projection(context[:, slot])[:, None]
            hidden = hidden + self.evidence_projection(evidence[:, slot])[:, None]
            proposal = expert(F.gelu(hidden))
            rms = proposal.square().mean(dim=-1, keepdim=True).sqrt()
            normalized_slot = proposal / (rms.detach() + 1e-6)
            valid = (mass[:, slot] > 0) & (presence[:, slot, None] > 0)
            normalized_slot = torch.where(
                valid[:, None], normalized_slot, torch.zeros_like(normalized_slot)
            )
            proposals.append(proposal)
            normalized.append(normalized_slot)

        proposal_tensor = torch.stack(proposals, dim=1)
        normalized_tensor = torch.stack(normalized, dim=1)
        weights = masks[:, :, :, None] * presence[:, :, None, None]
        applied = (weights * normalized_tensor).sum(dim=1)
        return {
            "proposal": proposal_tensor,
            "normalized": normalized_tensor,
            "applied": applied,
            "mass": mass,
        }

    def forward(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
        presence: torch.Tensor,
        evidence: torch.Tensor,
        rho: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branch = self.branch(tokens, masks, presence, evidence)
        return tokens + float(rho) * branch["applied"], branch


class SyntheticSystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(INPUT_DIM, CHANNELS, bias=False)
        self.anchor_head = nn.Linear(CHANNELS, ANCHOR_DIM, bias=False)
        self.evidence_head = nn.Linear(
            ANCHOR_DIM, SLOTS * EVIDENCE_DIM, bias=False
        )
        self.router = EvidenceBudgetRouter()
        self.id_head = nn.Linear(CHANNELS, CLASSES, bias=False)

    def features(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.backbone(inputs)
        anchor = self.anchor_head(tokens.mean(dim=1).detach())
        evidence = self.evidence_head(anchor).view(
            inputs.shape[0], SLOTS, EVIDENCE_DIM
        )
        return tokens, anchor, evidence

    def reid_forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        presence: torch.Tensor,
        rho: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        tokens, _, evidence = self.features(inputs)
        routed, branch = self.router(
            tokens, masks, presence, evidence.detach(), rho
        )
        descriptor = routed.mean(dim=1)
        return self.id_head(descriptor), descriptor, branch


def parameter_groups(model: SyntheticSystem) -> dict[str, list[nn.Parameter]]:
    return {
        "backbone": list(model.backbone.parameters()),
        "anchor": list(model.anchor_head.parameters()),
        "evidence_head": list(model.evidence_head.parameters()),
        "router_token_context_evidence": [
            *model.router.token_projection.parameters(),
            *model.router.context_projection.parameters(),
            *model.router.evidence_projection.parameters(),
        ],
        "router_experts": list(model.router.experts.parameters()),
        "id_head": list(model.id_head.parameters()),
    }


def gradient_report(
    model: SyntheticSystem, loss: torch.Tensor
) -> dict[str, dict[str, float | int]]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    report = {}
    for name, parameters in parameter_groups(model).items():
        gradients = [parameter.grad for parameter in parameters]
        present = [gradient for gradient in gradients if gradient is not None]
        nonzero = [
            gradient
            for gradient in present
            if bool(torch.count_nonzero(gradient).item())
        ]
        maximum = max(
            (float(gradient.detach().abs().max()) for gradient in present),
            default=0.0,
        )
        report[name] = {
            "parameters": len(parameters),
            "grad_present": len(present),
            "grad_nonzero": len(nonzero),
            "grad_abs_max": maximum,
        }
    return report


def group_active(report: dict, name: str) -> bool:
    record = report[name]
    return (
        record["grad_present"] == record["parameters"]
        and record["grad_nonzero"] == record["parameters"]
        and record["grad_abs_max"] > 0
    )


def group_off(report: dict, name: str) -> bool:
    record = report[name]
    return record["grad_present"] == 0 and record["grad_nonzero"] == 0


def make_masks() -> torch.Tensor:
    masks = torch.zeros(BATCH, SLOTS, TOKENS, dtype=torch.float32)
    assignments = [0, 0, 1, 2, 2, 3, 4]
    for token, slot in enumerate(assignments):
        masks[:, slot, token] = 1.0
    return masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    model = SyntheticSystem().cpu()
    inputs = torch.randn(BATCH, TOKENS, INPUT_DIM)
    masks = make_masks()
    presence = torch.ones(BATCH, SLOTS)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    teacher_evidence = torch.randn(BATCH, SLOTS, EVIDENCE_DIM)
    teacher_anchor = torch.randn(BATCH, ANCHOR_DIM)

    tokens, anchor, evidence = model.features(inputs)
    full_zero, branch_zero = model.router(
        tokens, masks, presence, evidence.detach(), rho=0.0
    )
    bypass = tokens.clone()

    null_masks = torch.zeros_like(masks)
    null_presence = torch.zeros_like(presence)
    null_mask_full, null_mask_branch = model.router(
        tokens, null_masks, presence, evidence.detach(), rho=0.125
    )
    null_presence_full, null_presence_branch = model.router(
        tokens, masks, null_presence, evidence.detach(), rho=0.125
    )

    schedule = [rho_at_epoch(epoch, 0.125, 2, 3) for epoch in range(8)]
    schedule_repeat = [rho_at_epoch(epoch, 0.125, 2, 3) for epoch in range(8)]

    zero_slot_masks = masks.clone()
    zero_slot_masks[:, 4] = 0
    zero_slot_branch = model.router.branch(
        tokens, zero_slot_masks, presence, evidence.detach()
    )

    isolated_tokens = tokens[:1].detach().repeat(2, 1, 1)
    isolated_masks = masks[:1].repeat(2, 1, 1)
    isolated_presence = presence[:1].repeat(2, 1)
    correct_evidence = evidence[:2].detach().clone()
    wrong_evidence = correct_evidence.roll(shifts=1, dims=1)
    static_evidence = evidence.detach().mean(dim=0, keepdim=True).repeat(2, 1, 1)
    correct_branch = model.router.branch(
        isolated_tokens, isolated_masks, isolated_presence, correct_evidence
    )["proposal"]
    wrong_branch = model.router.branch(
        isolated_tokens, isolated_masks, isolated_presence, wrong_evidence
    )["proposal"]
    static_branch = model.router.branch(
        isolated_tokens, isolated_masks, isolated_presence, static_evidence
    )["proposal"]
    static_single_first = model.router.branch(
        isolated_tokens[:1], isolated_masks[:1], isolated_presence[:1],
        static_evidence[:1]
    )["proposal"]
    static_single_repeat = model.router.branch(
        isolated_tokens[:1], isolated_masks[:1], isolated_presence[:1],
        static_evidence[:1]
    )["proposal"]

    teacher_loss = F.mse_loss(evidence, teacher_evidence)
    teacher_loss = teacher_loss + F.mse_loss(anchor, teacher_anchor)
    teacher_gradients = gradient_report(model, teacher_loss)

    tokens_exec, anchor_exec, _ = model.features(inputs)
    evidence_exec = model.evidence_head(anchor_exec.detach()).view(
        BATCH, SLOTS, EVIDENCE_DIM
    )
    exec_branch = model.router.branch(
        tokens_exec.detach(), masks, presence, evidence_exec
    )["proposal"]
    exec_target = torch.randn_like(exec_branch)
    exec_loss = F.mse_loss(exec_branch, exec_target)
    exec_gradients = gradient_report(model, exec_loss)

    logits, descriptor, _ = model.reid_forward(
        inputs, masks, presence, rho=0.125
    )
    reid_loss = F.cross_entropy(logits, labels) + 0.05 * descriptor.square().mean()
    reid_gradients = gradient_report(model, reid_loss)

    gradient_gates = {
        "teacher_updates_anchor": group_active(teacher_gradients, "anchor"),
        "teacher_updates_evidence_head": group_active(
            teacher_gradients, "evidence_head"
        ),
        "teacher_blocks_backbone": group_off(teacher_gradients, "backbone"),
        "teacher_blocks_router": group_off(
            teacher_gradients, "router_token_context_evidence"
        )
        and group_off(teacher_gradients, "router_experts"),
        "teacher_blocks_id_head": group_off(teacher_gradients, "id_head"),
        "exec_updates_evidence_head": group_active(
            exec_gradients, "evidence_head"
        ),
        "exec_updates_router": group_active(
            exec_gradients, "router_token_context_evidence"
        )
        and group_active(exec_gradients, "router_experts"),
        "exec_blocks_backbone": group_off(exec_gradients, "backbone"),
        "exec_blocks_anchor": group_off(exec_gradients, "anchor"),
        "exec_blocks_id_head": group_off(exec_gradients, "id_head"),
        "reid_updates_backbone": group_active(reid_gradients, "backbone"),
        "reid_updates_router": group_active(
            reid_gradients, "router_token_context_evidence"
        )
        and group_active(reid_gradients, "router_experts"),
        "reid_updates_id_head": group_active(reid_gradients, "id_head"),
        "reid_blocks_anchor": group_off(reid_gradients, "anchor"),
        "reid_blocks_evidence_head": group_off(
            reid_gradients, "evidence_head"
        ),
    }

    checks = {
        "rho_zero_full_bypass_exact": torch.equal(full_zero, bypass),
        "rho_zero_applied_finite": bool(torch.isfinite(branch_zero["applied"]).all()),
        "null_mask_identity_exact": torch.equal(null_mask_full, tokens),
        "null_mask_applied_exact_zero": bool(
            torch.count_nonzero(null_mask_branch["applied"]).item() == 0
        ),
        "null_presence_identity_exact": torch.equal(null_presence_full, tokens),
        "null_presence_applied_exact_zero": bool(
            torch.count_nonzero(null_presence_branch["applied"]).item() == 0
        ),
        "schedule_repeat_exact": schedule == schedule_repeat,
        "schedule_teacher_exact_zero": schedule[:3] == [0.0, 0.0, 0.0],
        "schedule_handoff_monotonic": all(
            left <= right for left, right in zip(schedule, schedule[1:])
        ),
        "schedule_fixed_after_handoff": schedule[5:] == [0.125, 0.125, 0.125],
        "rho_not_parameter_or_buffer": all(
            "rho" not in name.lower()
            for name, _ in list(model.named_parameters()) + list(model.named_buffers())
        ),
        "normalization_finite": bool(
            torch.isfinite(branch_zero["normalized"]).all()
        ),
        "zero_mass_slot_exact_zero": bool(
            torch.count_nonzero(zero_slot_branch["normalized"][:, 4]).item() == 0
        ),
        "correct_wrong_branch_distinct": bool(
            (correct_branch - wrong_branch).abs().max().item() > 1e-6
        ),
        "static_branch_finite": bool(torch.isfinite(static_branch).all()),
        "static_code_no_sample_variation": torch.equal(
            static_evidence[0], static_evidence[1]
        ),
        "static_single_sample_repeat_exact": torch.equal(
            static_single_first, static_single_repeat
        ),
        "all_gradient_ownership_exact": all(gradient_gates.values()),
    }
    finite_reports = all(
        math.isfinite(record["grad_abs_max"])
        for report in (teacher_gradients, exec_gradients, reid_gradients)
        for record in report.values()
    )
    checks["all_gradient_reports_finite"] = finite_reports

    payload = {
        "scope": "EXP394_PHASE0R_SYNTHETIC_CPU_CONTRACT",
        "verdict": "PHASE0R_S_PASS" if all(checks.values()) else "PHASE0R_S_FAIL",
        "seed": SEED,
        "torch_version": torch.__version__,
        "schedule": schedule,
        "measurements": {
            "correct_wrong_proposal_max_abs": float(
                (correct_branch - wrong_branch).detach().abs().max()
            ),
            "static_sample_proposal_max_abs": float(
                (static_branch[0] - static_branch[1]).detach().abs().max()
            ),
            "static_single_repeat_max_abs": float(
                (static_single_first - static_single_repeat).detach().abs().max()
            ),
            "static_input_token_max_abs": float(
                (isolated_tokens[0] - isolated_tokens[1]).abs().max()
            ),
            "static_input_mask_max_abs": float(
                (isolated_masks[0] - isolated_masks[1]).abs().max()
            ),
            "static_input_evidence_max_abs": float(
                (static_evidence[0] - static_evidence[1]).abs().max()
            ),
            "normalized_abs_max": float(
                branch_zero["normalized"].detach().abs().max()
            ),
            "applied_abs_max": float(
                branch_zero["applied"].detach().abs().max()
            ),
        },
        "checks": checks,
        "gradient_gates": gradient_gates,
        "gradients": {
            "teacher": teacher_gradients,
            "exec": exec_gradients,
            "reid": reid_gradients,
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("output_sha256={}".format(sha256_file(args.output)))
    return 0 if payload["verdict"] == "PHASE0R_S_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
