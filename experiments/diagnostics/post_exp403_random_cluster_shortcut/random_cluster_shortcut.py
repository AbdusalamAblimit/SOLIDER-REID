#!/usr/bin/env python3
"""Pure-CPU diagnostic for frequency-matched random-cluster semantics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


SEED = 20260720
N_IDENTITIES = 128
N_GALLERY = 2
N_CLUSTERS = 8
ID_DIM = 64
NUISANCE_DIM = 128


def unit_random(seed: int, dim: int) -> Tuple[float, ...]:
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    length = math.sqrt(sum(value * value for value in values))
    return tuple(value / length for value in values)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class Sample:
    name: str
    pid: int
    camera: int
    slot: str
    cluster: str


def balanced_assignment(seed: int) -> List[str]:
    count = N_IDENTITIES * (1 + N_GALLERY)
    if count % N_CLUSTERS != 0:
        raise RuntimeError("Sample count must be divisible by cluster count")
    labels = [
        f"cluster_{cluster}"
        for cluster in range(N_CLUSTERS)
        for _ in range(count // N_CLUSTERS)
    ]
    random.Random(seed).shuffle(labels)
    return labels


def build_samples(assignments: Sequence[str]) -> Tuple[List[Sample], List[Sample]]:
    expected = N_IDENTITIES * (1 + N_GALLERY)
    if len(assignments) != expected:
        raise ValueError("Cluster assignment length mismatch")
    queries: List[Sample] = []
    gallery: List[Sample] = []
    cursor = 0
    for pid in range(N_IDENTITIES):
        queries.append(Sample(f"q_{pid}", pid, 0, "q", assignments[cursor]))
        cursor += 1
        for replica in range(N_GALLERY):
            gallery.append(
                Sample(
                    f"g_{pid}_{replica}",
                    pid,
                    1,
                    str(replica),
                    assignments[cursor],
                )
            )
            cursor += 1
    return queries, gallery


def donor_map(samples: Sequence[Sample]) -> Dict[str, Sample]:
    by_slot = {
        (sample.camera, sample.pid, sample.slot): sample for sample in samples
    }
    donors: Dict[str, Sample] = {}
    for sample in samples:
        donors[sample.name] = by_slot[
            (sample.camera, (sample.pid + 1) % N_IDENTITIES, sample.slot)
        ]
    return donors


def descriptor(
    sample: Sample,
    supplied: str,
    identity: Sequence[float],
    mutant: bool,
) -> Tuple[float, ...]:
    if mutant:
        quota = 0.45
        nuisance_token = "ignored"
    else:
        if supplied == sample.cluster:
            quota = 1.00
        elif supplied.startswith("cluster_"):
            quota = 0.45
        elif supplied == "generic":
            quota = 0.18
        elif supplied == "null":
            quota = 0.05
        else:
            raise ValueError(f"Unknown supplied state: {supplied}")
        nuisance_token = supplied
    nuisance = unit_random(
        stable_seed("nuisance", sample.name, nuisance_token), NUISANCE_DIM
    )
    nuisance_scale = math.sqrt(1.0 - quota * quota)
    return tuple(quota * value for value in identity) + tuple(
        nuisance_scale * value for value in nuisance
    )


def average_precision(sorted_pids: Sequence[int], target_pid: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, pid in enumerate(sorted_pids, start=1):
        if pid == target_pid:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / N_GALLERY


def retrieval_metrics(
    queries: Sequence[Sample],
    gallery: Sequence[Sample],
    descriptors: Dict[str, Tuple[float, ...]],
) -> Dict[str, float]:
    aps: List[float] = []
    rank1 = 0
    for query in queries:
        ranked = sorted(
            gallery,
            key=lambda item: dot(descriptors[query.name], descriptors[item.name]),
            reverse=True,
        )
        aps.append(average_precision([item.pid for item in ranked], query.pid))
        rank1 += int(ranked[0].pid == query.pid)
    return {"mAP": sum(aps) / len(aps), "R1": rank1 / len(queries)}


def cluster_contract(samples: Sequence[Sample]) -> Dict[str, object]:
    counts = Counter(sample.cluster for sample in samples)
    pid_sets = defaultdict(set)
    camera_sets = defaultdict(set)
    for sample in samples:
        pid_sets[sample.cluster].add(sample.pid)
        camera_sets[sample.cluster].add(sample.camera)
    return {
        "counts": dict(sorted(counts.items())),
        "pid_coverage": {
            cluster: len(pid_sets[cluster]) for cluster in sorted(counts)
        },
        "camera_coverage": {
            cluster: len(camera_sets[cluster]) for cluster in sorted(counts)
        },
    }


def execute(assignments: Sequence[str], mutant: bool) -> Dict[str, object]:
    queries, gallery = build_samples(assignments)
    all_samples = queries + gallery
    donors = donor_map(queries) | donor_map(gallery)
    identities = {
        pid: unit_random(SEED + 100000 + pid, ID_DIM)
        for pid in range(N_IDENTITIES)
    }
    suppliers = {
        "correct": lambda sample: sample.cluster,
        "wrong": lambda sample: donors[sample.name].cluster,
        "generic": lambda sample: "generic",
        "null": lambda sample: "null",
    }
    metrics: Dict[str, Dict[str, float]] = {}
    max_norm_error = 0.0
    for arm, supplier in suppliers.items():
        arm_descriptors = {
            sample.name: descriptor(
                sample,
                supplier(sample),
                identities[sample.pid],
                mutant,
            )
            for sample in all_samples
        }
        max_norm_error = max(
            max_norm_error,
            max(abs(norm(value) - 1.0) for value in arm_descriptors.values()),
        )
        metrics[arm] = retrieval_metrics(queries, gallery, arm_descriptors)

    same_collisions = [
        queries[pid].cluster == gallery[pid * N_GALLERY].cluster
        for pid in range(N_IDENTITIES)
    ]
    donor_collisions = [
        sample.cluster == donors[sample.name].cluster for sample in all_samples
    ]
    same_rate = sum(same_collisions) / len(same_collisions)
    donor_rate = sum(donor_collisions) / len(donor_collisions)
    collision_gap = abs(same_rate - donor_rate)
    correct_minus_wrong = metrics["correct"]["mAP"] - metrics["wrong"]["mAP"]
    wrong_minus_low = metrics["wrong"]["mAP"] - max(
        metrics["generic"]["mAP"], metrics["null"]["mAP"]
    )
    contract = cluster_contract(all_samples)
    exact_count = len(all_samples) // N_CLUSTERS
    gates = {
        "correct_map": metrics["correct"]["mAP"] >= 0.99,
        "correct_minus_wrong": correct_minus_wrong >= 0.05,
        "wrong_minus_low": wrong_minus_low >= 0.05,
        "balanced_cluster_counts": all(
            count == exact_count for count in contract["counts"].values()
        ),
        "cluster_pid_coverage": all(
            coverage >= 40 for coverage in contract["pid_coverage"].values()
        ),
        "cluster_camera_coverage": all(
            coverage == 2
            for coverage in contract["camera_coverage"].values()
        ),
        "random_cluster_no_pid_match": collision_gap < 0.10,
        "donor_contract": all(
            donors[sample.name].camera == sample.camera
            and donors[sample.name].pid != sample.pid
            and donors[sample.name].name != sample.name
            for sample in all_samples
        ),
        "unit_norm": max_norm_error < 1e-10,
    }
    return {
        "arms": metrics,
        "correct_minus_wrong_map": correct_minus_wrong,
        "wrong_minus_max_generic_null_map": wrong_minus_low,
        "same_pid_cluster_collision_rate": same_rate,
        "different_pid_donor_cluster_collision_rate": donor_rate,
        "cluster_collision_gap_abs": collision_gap,
        "cluster_contract": contract,
        "max_descriptor_norm_error": max_norm_error,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> int:
    original_assignments = balanced_assignment(SEED + 7)
    permuted_assignments = list(original_assignments)
    random.Random(SEED + 11).shuffle(permuted_assignments)
    original = execute(original_assignments, mutant=False)
    permuted = execute(permuted_assignments, mutant=False)
    mutant = execute(original_assignments, mutant=True)
    mutant_caught = not (
        mutant["gates"]["correct_map"]
        and mutant["gates"]["correct_minus_wrong"]
        and mutant["gates"]["wrong_minus_low"]
    )
    positive_pass = bool(
        original["all_gates_pass"] and permuted["all_gates_pass"]
    )
    verdict = (
        "FREQUENCY_MATCHED_RANDOM_CLUSTER_FALSE_SEMANTICS_DEMONSTRATED"
        if positive_pass and mutant_caught
        else "DIAGNOSTIC_INCONCLUSIVE"
    )
    result = {
        "diagnostic": "post_exp403_random_cluster_shortcut",
        "seed": SEED,
        "dimensions": {
            "identity": ID_DIM,
            "nuisance": NUISANCE_DIM,
            "clusters": N_CLUSTERS,
        },
        "counts": {
            "identities": N_IDENTITIES,
            "queries": N_IDENTITIES,
            "gallery": N_IDENTITIES * N_GALLERY,
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "official_data_accesses": 0,
        "pose_cache_checkpoint_accesses": 0,
        "gpu_accesses": 0,
        "original": original,
        "frequency_matched_label_permutation": permuted,
        "evidence_ignored_mutant": mutant,
        "mutant_caught": mutant_caught,
        "verdict": verdict,
        "exp404_authorized": False,
        "gpu_start_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if verdict.endswith("DEMONSTRATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
