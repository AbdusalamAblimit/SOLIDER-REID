#!/usr/bin/env python3
"""Deterministic CPU-only positive/negative contract for exp404 SPK."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 20260720
GROUPS = 4
GROUP_WIDTH = 3
FEATURE_DIM = GROUPS * GROUP_WIDTH


class SemanticProductKernel(nn.Module):
    """Parameter-free groupwise product binding for a fixed descriptor."""

    def __init__(self, groups: int, feature_dim: int):
        super().__init__()
        if groups <= 0 or feature_dim <= 0 or feature_dim % groups:
            raise ValueError("feature_dim must be positive and divisible by groups")
        self.groups = int(groups)
        self.feature_dim = int(feature_dim)
        self.group_width = feature_dim // groups

    def aggregate(self, evidence: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 3 or evidence.shape[-1] != self.groups:
            raise ValueError("evidence must have shape [B,S,G]")
        if presence.shape != evidence.shape[:2]:
            raise ValueError("presence must have shape [B,S]")
        presence = presence.to(dtype=evidence.dtype).clamp(0.0, 1.0)
        mass = presence.sum(dim=1, keepdim=True)
        pooled = (evidence * presence[..., None]).sum(dim=1)
        pooled = pooled / mass.clamp_min(1.0)
        return torch.where(mass > 0, pooled, torch.zeros_like(pooled))

    def forward(
        self,
        global_feature: torch.Tensor,
        evidence: torch.Tensor,
        presence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_feature.ndim != 2 or global_feature.shape[-1] != self.feature_dim:
            raise ValueError("global_feature must have shape [B,C]")
        pooled = self.aggregate(evidence, presence)
        factor = self.groups * torch.softmax(pooled, dim=-1)
        grouped = global_feature.reshape(
            global_feature.shape[0], self.groups, self.group_width
        )
        descriptor = (factor[..., None] * grouped).reshape_as(global_feature)
        if not torch.isfinite(descriptor).all() or not torch.isfinite(factor).all():
            raise RuntimeError("SPK output must be finite")
        return descriptor, factor


class EvidenceIgnoredMutant(SemanticProductKernel):
    def forward(self, global_feature, evidence, presence):
        factor = torch.ones(
            global_feature.shape[0], self.groups,
            dtype=global_feature.dtype, device=global_feature.device,
        )
        return global_feature, factor


class AuxiliaryOnlyMutant(SemanticProductKernel):
    def forward(self, global_feature, evidence, presence):
        _, factor = super().forward(global_feature, evidence, presence)
        return global_feature, factor


class AdditiveBypassMutant(SemanticProductKernel):
    def forward(self, global_feature, evidence, presence):
        descriptor, factor = super().forward(global_feature, evidence, presence)
        return global_feature + descriptor, factor


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left, right, dim=-1)


def semantic_fixture() -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    query = torch.tensor(
        [[1.0, 0.0, 0.0]] * GROUPS, dtype=torch.float64
    ).reshape(1, FEATURE_DIM)
    positive = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    ).reshape(1, FEATURE_DIM)
    features = torch.cat([query, positive], dim=0)
    slot = {
        "correct": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        "wrong": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64),
        "generic": torch.tensor([0.2, 0.1, 0.0, -0.1], dtype=torch.float64),
        "null": torch.zeros(GROUPS, dtype=torch.float64),
        "random_key": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64),
        "random_cluster": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
    }
    evidence = {
        name: value.view(1, 1, GROUPS).expand(2, 2, GROUPS).clone()
        for name, value in slot.items()
    }
    presence = torch.ones(2, 2, dtype=torch.float64)
    return features, evidence, presence


def random_cluster_contract() -> dict[str, object]:
    identities = 128
    permutation = list(range(identities))
    random.Random(SEED + 41).shuffle(permutation)
    rank = {pid: position for position, pid in enumerate(permutation)}
    records = []
    for pid in range(identities):
        residue = rank[pid] % 8
        records.extend(
            [
                (pid, 0, (residue + 0) % 8),
                (pid, 1, (residue + 1) % 8),
                (pid, 1, (residue + 3) % 8),
            ]
        )
    counts = Counter(cluster for _, _, cluster in records)
    pid_sets = defaultdict(set)
    camera_sets = defaultdict(set)
    for pid, camera, cluster in records:
        pid_sets[cluster].add(pid)
        camera_sets[cluster].add(camera)
    return {
        "counts": {str(key): counts[key] for key in sorted(counts)},
        "pid_coverage": {str(key): len(pid_sets[key]) for key in sorted(counts)},
        "camera_coverage": {
            str(key): len(camera_sets[key]) for key in sorted(counts)
        },
        "balanced": all(counts[key] == 48 for key in range(8)),
        "pid_coverage_pass": all(len(pid_sets[key]) >= 40 for key in range(8)),
        "camera_coverage_pass": all(
            len(camera_sets[key]) == 2 for key in range(8)
        ),
    }


def donor_contract() -> dict[str, object]:
    identities = torch.arange(128, dtype=torch.long)
    cameras = torch.arange(128, dtype=torch.long) % 2
    donors = torch.empty_like(identities)
    for index in range(identities.numel()):
        candidates = torch.arange(index + 1, index + identities.numel()) % identities.numel()
        valid = cameras[candidates].eq(cameras[index]) & identities[candidates].ne(
            identities[index]
        )
        donors[index] = candidates[valid][0]
    return {
        "same_camera": bool(cameras[donors].eq(cameras).all()),
        "different_pid": bool(identities[donors].ne(identities).all()),
        "no_fixed_point": bool(donors.ne(torch.arange(128)).all()),
        "mapping_sha256": hashlib.sha256(donors.numpy().tobytes()).hexdigest(),
    }


def evaluate_module(module: SemanticProductKernel) -> dict[str, object]:
    features, evidence, presence = semantic_fixture()
    descriptors = {}
    factors = {}
    utilities = {}
    for name in evidence:
        descriptor, factor = module(features, evidence[name], presence)
        descriptors[name] = descriptor
        factors[name] = factor
        utilities[name] = float(cosine(descriptor[:1], descriptor[1:]).item())
    return {
        "descriptors": descriptors,
        "factors": factors,
        "utilities": utilities,
    }


def main() -> int:
    torch.manual_seed(SEED)
    cuda_before = torch.cuda.is_initialized()
    module = SemanticProductKernel(GROUPS, FEATURE_DIM).double()
    evaluated = evaluate_module(module)
    descriptors = evaluated["descriptors"]
    factors = evaluated["factors"]
    utilities = evaluated["utilities"]

    features, evidence, presence = semantic_fixture()
    grad_features = features.clone().requires_grad_(True)
    grad_evidence = evidence["correct"].clone().requires_grad_(True)
    grad_descriptor, _ = module(grad_features, grad_evidence, presence)
    grad_utility = cosine(grad_descriptor[:1], grad_descriptor[1:]).sum()
    grad_utility.backward()

    random_key_abs_preserved = bool(
        torch.equal(
            evidence["correct"].abs().sort(dim=-1).values,
            evidence["random_key"].abs().sort(dim=-1).values,
        )
        and torch.equal(
            evidence["correct"].norm(dim=-1),
            evidence["random_key"].norm(dim=-1),
        )
    )
    cluster = random_cluster_contract()
    donor = donor_contract()
    source = inspect.getsource(SemanticProductKernel)

    ignored = evaluate_module(EvidenceIgnoredMutant(GROUPS, FEATURE_DIM).double())
    auxiliary = evaluate_module(AuxiliaryOnlyMutant(GROUPS, FEATURE_DIM).double())
    additive = evaluate_module(AdditiveBypassMutant(GROUPS, FEATURE_DIM).double())
    semantic_controls = ["wrong", "generic", "null", "random_key", "random_cluster"]
    positive_margin = min(
        utilities["correct"] - utilities[name] for name in semantic_controls
    )

    gates = {
        "cuda_not_initialized": not cuda_before and not torch.cuda.is_initialized(),
        "feature_group_mapping_exact": bool(
            torch.equal(
                features.reshape(2, GROUPS, GROUP_WIDTH).reshape_as(features),
                features,
            )
        ),
        "module_parameter_free": sum(p.numel() for p in module.parameters()) == 0,
        "source_no_learned_projection_or_concat": (
            "nn.Linear" not in source and "torch.cat" not in source
        ),
        "null_factor_exact_one": bool(
            torch.equal(factors["null"], torch.ones_like(factors["null"]))
        ),
        "null_output_exact_input": bool(
            torch.equal(descriptors["null"], features)
        ),
        "factor_finite_nonnegative_mean_one": all(
            bool(torch.isfinite(value).all())
            and bool(value.ge(0).all())
            and bool(torch.allclose(value.mean(dim=-1), torch.ones(value.shape[0], dtype=value.dtype), atol=0.0, rtol=1e-15))
            for value in factors.values()
        ),
        "interventions_active": all(
            not torch.equal(descriptors[name], descriptors["null"])
            for name in ("correct", "wrong", "generic", "random_key", "random_cluster")
        ),
        "semantic_positive_margin": positive_margin >= 0.10,
        "global_feature_grad_finite_nonzero": bool(
            torch.isfinite(grad_features.grad).all()
            and grad_features.grad.norm() > 0
        ),
        "correct_evidence_grad_finite_nonzero": bool(
            torch.isfinite(grad_evidence.grad).all()
            and grad_evidence.grad.norm() > 0
        ),
        "random_key_distribution_preserved": random_key_abs_preserved,
        "random_cluster_contract": bool(
            cluster["balanced"]
            and cluster["pid_coverage_pass"]
            and cluster["camera_coverage_pass"]
        ),
        "donor_contract": bool(
            donor["same_camera"]
            and donor["different_pid"]
            and donor["no_fixed_point"]
        ),
        "mutant_evidence_ignored_caught": bool(
            torch.equal(ignored["descriptors"]["correct"], ignored["descriptors"]["null"])
        ),
        "mutant_auxiliary_only_caught": bool(
            torch.equal(auxiliary["descriptors"]["correct"], features)
        ),
        "mutant_additive_bypass_caught": bool(
            not torch.equal(additive["descriptors"]["null"], features)
        ),
    }
    all_gates_pass = all(gates.values())
    result = {
        "diagnostic": "exp404_spk_static_cpu_contract",
        "seed": SEED,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "dimensions": {
            "groups": GROUPS,
            "group_width": GROUP_WIDTH,
            "feature": FEATURE_DIM,
        },
        "utilities": utilities,
        "correct_min_control_margin": positive_margin,
        "factor_rows": {
            name: [float(value) for value in factor[0]]
            for name, factor in factors.items()
        },
        "gradient_norms": {
            "global_feature": float(grad_features.grad.norm()),
            "correct_evidence": float(grad_evidence.grad.norm()),
        },
        "random_cluster": cluster,
        "donor": donor,
        "mutants": {
            "evidence_ignored_correct_equals_null": bool(
                torch.equal(
                    ignored["descriptors"]["correct"],
                    ignored["descriptors"]["null"],
                )
            ),
            "auxiliary_only_correct_equals_input": bool(
                torch.equal(auxiliary["descriptors"]["correct"], features)
            ),
            "additive_null_equals_input": bool(
                torch.equal(additive["descriptors"]["null"], features)
            ),
        },
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "verdict": "STATIC_CPU_PASS" if all_gates_pass else "STATIC_CPU_FAIL",
        "production_implementation_authorized": all_gates_pass,
        "cuda_preflight_authorized": False,
        "gpu_start_authorized": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_gates_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
