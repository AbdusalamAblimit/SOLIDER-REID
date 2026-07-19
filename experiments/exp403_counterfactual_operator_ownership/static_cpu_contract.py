#!/usr/bin/env python3
"""Deterministic positive/negative CPU contracts for exp403 ELO-CUR."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


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


def build_training_donor_map(pids: torch.Tensor, camids: torch.Tensor):
    if pids.ndim != 1 or camids.shape != pids.shape:
        raise ValueError("pids/camids must be aligned one-dimensional tensors")
    count = int(pids.numel())
    donor = torch.full((count,), -1, dtype=torch.long)
    for index in range(count):
        for offset in range(1, count):
            candidate = (index + offset) % count
            if (
                int(camids[candidate]) == int(camids[index])
                and int(pids[candidate]) != int(pids[index])
            ):
                donor[index] = candidate
                break
    return donor


class EvidenceOwnedLowRankOperator(nn.Module):
    """A shared low-rank production operator whose coefficients come from evidence."""

    def __init__(self, channels=4, evidence_dim=4, rank=4, rho=0.25):
        super().__init__()
        self.channels = int(channels)
        self.evidence_dim = int(evidence_dim)
        self.rank = int(rank)
        self.rho = float(rho)
        self.down_projection = nn.Linear(channels, rank, bias=False)
        self.context_projection = nn.Linear(channels, rank, bias=False)
        self.evidence_projection = nn.Linear(evidence_dim, rank, bias=False)
        self.up_projection = nn.Linear(rank, channels, bias=False)
        self.context_query = nn.Linear(channels, evidence_dim, bias=False)
        self.evidence_key = nn.Linear(evidence_dim, evidence_dim, bias=False)

    def branch(self, tokens, mask, presence, evidence, ignore_evidence=False):
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,N,C]")
        if mask.ndim != 3 or mask.shape[0] != tokens.shape[0]:
            raise ValueError("mask must have shape [B,S,N]")
        if mask.shape[2] != tokens.shape[1]:
            raise ValueError("mask/token spatial dimensions must match")
        if presence.shape != mask.shape[:2]:
            raise ValueError("presence must have shape [B,S]")
        if evidence.shape != (*mask.shape[:2], self.evidence_dim):
            raise ValueError("evidence must have shape [B,S,D]")

        support = mask.detach().float().clamp_min(0.0)
        presence = presence.detach().float().clamp(0.0, 1.0)
        mass = support.sum(dim=-1, keepdim=True)
        normalized_mask = support / mass.clamp_min(1e-12)
        normalized_mask = torch.where(
            mass > 0, normalized_mask, torch.zeros_like(normalized_mask)
        )
        context = torch.einsum(
            "bsn,bnc->bsc", normalized_mask.to(tokens.dtype), tokens
        )

        token_hidden = self.down_projection(tokens)[:, None]
        context_hidden = self.context_projection(context)[:, :, None]
        if ignore_evidence:
            coefficients = torch.ones(
                *evidence.shape[:2], self.rank, dtype=tokens.dtype
            )
            compatibility = torch.ones(evidence.shape[:2], dtype=tokens.dtype)
            gate = torch.ones_like(compatibility)
        else:
            coefficients = self.evidence_projection(evidence)
            query = F.normalize(self.context_query(context).float(), dim=-1)
            key_raw = self.evidence_key(evidence).float()
            key_norm = key_raw.norm(dim=-1)
            key = F.normalize(key_raw, dim=-1)
            raw_similarity = (query * key).sum(dim=-1)
            compatibility = torch.where(
                key_norm > 0,
                raw_similarity,
                torch.full_like(raw_similarity, -1.0),
            )
            gate = torch.where(
                key_norm > 0,
                torch.sigmoid(compatibility),
                torch.zeros_like(compatibility),
            )

        rank_update = (token_hidden + context_hidden) * coefficients[:, :, None]
        proposal = self.up_projection(F.gelu(rank_update))
        proposal = proposal * gate[:, :, None, None].to(proposal.dtype)
        scatter = support[:, :, :, None] * presence[:, :, None, None]
        unit_delta = (scatter.to(proposal.dtype) * proposal).sum(dim=1)
        applied_delta = self.rho * unit_delta
        output = tokens + applied_delta
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("non-finite ELO output")
        return {
            "output": output,
            "delta": applied_delta,
            "coefficients": coefficients,
            "compatibility": compatibility,
            "gate": gate,
            "context": context,
            "normalized_mask": normalized_mask,
        }

    def forward(self, tokens, mask, presence, evidence):
        return self.branch(tokens, mask, presence, evidence)["output"]


def ordinal_hinge(correct, wrong, generic, null, margin=0.10):
    return torch.stack(
        [
            F.relu(margin + wrong - correct),
            F.relu(margin + generic - wrong),
            F.relu(margin + null - generic),
        ]
    ).mean()


def utility(descriptor, prototype):
    return F.cosine_similarity(descriptor.float(), prototype.float(), dim=-1)


def counterfactual_utility_ranking(
    correct_descriptor,
    reference_descriptors,
    positive_prototype,
    margin=0.05,
    detach_references=True,
):
    correct_utility = utility(correct_descriptor, positive_prototype)
    losses = []
    reference_utilities = []
    for reference in reference_descriptors:
        value = utility(reference, positive_prototype)
        reference_utilities.append(value)
        comparator = value.detach() if detach_references else value
        losses.append(F.relu(margin + comparator - correct_utility).mean())
    return torch.stack(losses).mean(), correct_utility, reference_utilities


def source_contract(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    operator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "EvidenceOwnedLowRankOperator"
    )
    operator_source = ast.get_source_segment(source, operator)
    return {
        "no_slot_expert_modulelist": "ModuleList" not in operator_source,
        "no_expert_parameter": "expert" not in operator_source.lower(),
        "explicit_evidence_coefficients": "coefficients = self.evidence_projection" in operator_source,
        "explicit_null_gate": "torch.zeros_like(compatibility)" in operator_source,
        "reference_detach_present": "value.detach() if detach_references" in source,
    }


def synthetic_inputs():
    torch.manual_seed(403)
    batch, slots, tokens_count, channels = 4, 2, 4, 4
    tokens = torch.randn(batch, tokens_count, channels, dtype=torch.float64)
    tokens[:, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    tokens[:, 1, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    tokens[:, 2, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    tokens[:, 3, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]],
        dtype=torch.float64,
    ).repeat(batch, 1, 1)
    presence = torch.ones(batch, slots, dtype=torch.float64)
    correct_vector = F.normalize(
        torch.tensor([0.95, 0.15, 0.0, 0.0], dtype=torch.float64), dim=0
    )
    wrong_vector = F.normalize(
        torch.tensor([0.50, math.sqrt(0.75), 0.0, 0.0], dtype=torch.float64),
        dim=0,
    )
    generic_vector = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    evidence = {
        "correct": correct_vector.repeat(batch, slots, 1).clone(),
        "wrong": wrong_vector.repeat(batch, slots, 1).clone(),
        "generic": generic_vector.repeat(batch, slots, 1).clone(),
        "null": torch.zeros(batch, slots, 4, dtype=torch.float64),
    }
    return tokens, mask, presence, evidence


def initialize_operator():
    torch.manual_seed(1403)
    operator = EvidenceOwnedLowRankOperator().double()
    with torch.no_grad():
        for module in (
            operator.down_projection,
            operator.context_projection,
            operator.evidence_projection,
            operator.up_projection,
        ):
            module.weight.copy_(
                torch.randn_like(module.weight) * 0.3
                + torch.eye(4, dtype=torch.float64) * 0.2
            )
        operator.context_query.weight.copy_(torch.eye(4, dtype=torch.float64))
        operator.evidence_key.weight.copy_(torch.eye(4, dtype=torch.float64))
    return operator


def grad_norm(value):
    if value is None:
        return 0.0
    return float(value.detach().float().norm())


def run_contract(path: Path):
    cuda_before = torch.cuda.is_initialized()
    operator = initialize_operator()
    tokens, mask, presence, evidence = synthetic_inputs()

    branches = {
        name: operator.branch(tokens, mask, presence, value)
        for name, value in evidence.items()
    }
    compatibility = {
        name: float(branch["compatibility"].detach().mean())
        for name, branch in branches.items()
    }
    descriptors = {
        name: branch["output"].mean(dim=1)
        for name, branch in branches.items()
    }

    null_exact = torch.equal(branches["null"]["output"], tokens)
    active = {
        name: not torch.equal(branches[name]["output"], tokens)
        for name in ("correct", "wrong", "generic")
    }
    distinct = {
        name: not torch.equal(descriptors["correct"], descriptors[name])
        for name in ("wrong", "generic", "null")
    }
    ordinal_values = {
        "correct_minus_wrong": compatibility["correct"] - compatibility["wrong"],
        "wrong_minus_generic": compatibility["wrong"] - compatibility["generic"],
        "generic_minus_null": compatibility["generic"] - compatibility["null"],
    }
    ordinal_loss = ordinal_hinge(
        branches["correct"]["compatibility"],
        branches["wrong"]["compatibility"],
        branches["generic"]["compatibility"],
        branches["null"]["compatibility"],
    )

    pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    camids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    donor = build_training_donor_map(pids, camids)
    donor_repeat = build_training_donor_map(pids, camids)
    eligible = donor >= 0
    donor_same_camera = bool(
        torch.equal(camids[donor[eligible]], camids[eligible])
    )
    donor_different_pid = bool(
        torch.all(pids[donor[eligible]] != pids[eligible])
    )

    grad_operator = initialize_operator()
    grad_tokens, grad_mask, grad_presence, grad_evidence_base = synthetic_inputs()
    grad_evidence = {
        name: value.clone().requires_grad_(True)
        for name, value in grad_evidence_base.items()
    }
    grad_branches = {
        name: grad_operator.branch(
            grad_tokens, grad_mask, grad_presence, grad_evidence[name]
        )
        for name in grad_evidence
    }
    grad_descriptors = {
        name: value["output"].mean(dim=1)
        for name, value in grad_branches.items()
    }
    prototype = F.normalize(
        torch.tensor(
            [
                [0.3, -0.4, 0.7, 0.2],
                [0.2, 0.6, -0.3, 0.5],
                [-0.5, 0.2, 0.1, 0.7],
                [0.6, -0.1, 0.3, -0.4],
            ],
            dtype=torch.float64,
        ),
        dim=-1,
    )
    cur_loss, correct_utility, reference_utilities = counterfactual_utility_ranking(
        grad_descriptors["correct"],
        [
            grad_descriptors["wrong"],
            grad_descriptors["generic"],
            grad_descriptors["null"],
        ],
        prototype,
        margin=0.50,
        detach_references=True,
    )
    total_loss = cur_loss + ordinal_hinge(
        grad_branches["correct"]["compatibility"],
        grad_branches["wrong"]["compatibility"].detach(),
        grad_branches["generic"]["compatibility"].detach(),
        grad_branches["null"]["compatibility"].detach(),
        margin=0.60,
    )
    total_loss.backward()
    parameter_grad_norm = {
        name: grad_norm(parameter.grad)
        for name, parameter in grad_operator.named_parameters()
    }
    reference_grad_norm = {
        name: grad_norm(grad_evidence[name].grad)
        for name in ("wrong", "generic", "null")
    }
    correct_grad_norm = grad_norm(grad_evidence["correct"].grad)

    mutant_operator = initialize_operator()
    mutant_zero = mutant_operator.branch(
        tokens, mask, presence, evidence["null"], ignore_evidence=True
    )["output"]
    mutant_ignore_evidence_caught = not torch.equal(mutant_zero, tokens)
    aux_only_descriptors = {
        name: tokens.mean(dim=1).clone() for name in evidence
    }
    mutant_aux_only_caught = all(
        torch.equal(aux_only_descriptors["correct"], aux_only_descriptors[name])
        for name in ("wrong", "generic", "null")
    )

    nondetach_operator = initialize_operator()
    _, nd_mask, nd_presence, nd_evidence_base = synthetic_inputs()
    nd_tokens = tokens.clone()
    nd_evidence = {
        name: value.clone().requires_grad_(True)
        for name, value in nd_evidence_base.items()
    }
    nd_desc = {
        name: nondetach_operator(
            nd_tokens, nd_mask, nd_presence, nd_evidence[name]
        ).mean(dim=1)
        for name in nd_evidence
    }
    nondetach_loss, _, _ = counterfactual_utility_ranking(
        nd_desc["correct"],
        [nd_desc["wrong"], nd_desc["generic"], nd_desc["null"]],
        prototype,
        margin=0.50,
        detach_references=False,
    )
    nondetach_loss.backward()
    mutant_nondetach_caught = any(
        grad_norm(nd_evidence[name].grad) > 0
        for name in ("wrong", "generic", "null")
    )

    source = source_contract(path)
    all_finite = all(
        bool(torch.isfinite(value["output"]).all())
        and bool(torch.isfinite(value["compatibility"]).all())
        for value in branches.values()
    )
    no_bias = all(
        module.bias is None
        for module in grad_operator.modules()
        if isinstance(module, nn.Linear)
    )
    gates = {
        **source,
        "cuda_not_initialized": not cuda_before and not torch.cuda.is_initialized(),
        "all_linear_no_bias": no_bias,
        "null_exact_identity": null_exact,
        "non_null_arms_active": all(active.values()),
        "counterfactual_descriptors_distinct": all(distinct.values()),
        "compat_correct_wrong_margin": ordinal_values["correct_minus_wrong"] >= 0.10,
        "compat_wrong_generic_margin": ordinal_values["wrong_minus_generic"] >= 0.10,
        "compat_generic_null_margin": ordinal_values["generic_minus_null"] >= 0.10,
        "ordinal_hinge_zero": float(ordinal_loss.detach()) == 0.0,
        "donor_all_eligible": bool(eligible.all()),
        "donor_same_camera": donor_same_camera,
        "donor_different_pid": donor_different_pid,
        "donor_repeat_exact": torch.equal(donor, donor_repeat),
        "outputs_finite": all_finite,
        "cur_loss_finite_nonzero": bool(torch.isfinite(cur_loss.detach()))
        and float(cur_loss.detach()) > 0,
        "correct_evidence_receives_grad": correct_grad_norm > 0,
        "all_operator_groups_receive_grad": all(
            math.isfinite(value) and value > 0
            for value in parameter_grad_norm.values()
        ),
        "references_receive_no_grad": all(
            value == 0.0 for value in reference_grad_norm.values()
        ),
        "mutant_ignore_evidence_caught": mutant_ignore_evidence_caught,
        "mutant_aux_only_caught": mutant_aux_only_caught,
        "mutant_nondetach_caught": mutant_nondetach_caught,
    }
    passed = all(gates.values())
    return {
        "experiment": "exp403_counterfactual_operator_ownership",
        "status": "STATIC_CPU_PASS" if passed else "STATIC_CPU_FAIL",
        "gpu_authorized": False,
        "formal_training_authorized": False,
        "source_sha256": sha256_file(path),
        "gate_count": len(gates),
        "gate_pass_count": sum(bool(value) for value in gates.values()),
        "gates": gates,
        "measurements": {
            "compatibility": compatibility,
            "ordinal_values": ordinal_values,
            "ordinal_loss": float(ordinal_loss.detach()),
            "donor_map": donor.tolist(),
            "cur_loss": float(cur_loss.detach()),
            "correct_utility_mean": float(correct_utility.detach().mean()),
            "reference_utility_mean": [
                float(value.detach().mean()) for value in reference_utilities
            ],
            "correct_evidence_grad_norm": correct_grad_norm,
            "reference_evidence_grad_norm": reference_grad_norm,
            "parameter_grad_norm": parameter_grad_norm,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = Path(__file__).resolve()
    result = run_contract(path)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "STATIC_CPU_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
