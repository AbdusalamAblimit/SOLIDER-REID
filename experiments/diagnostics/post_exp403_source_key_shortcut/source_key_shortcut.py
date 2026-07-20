#!/usr/bin/env python3
"""Pure-CPU existential diagnostic for non-semantic source-key ownership."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


SEED = 20260720
N_IDENTITIES = 128
N_GALLERY = 2
ID_DIM = 64
KEY_DIM = 16
NUISANCE_DIM = 128


def unit_random(seed: int, dim: int) -> Tuple[float, ...]:
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def mean_vector(vectors: Iterable[Sequence[float]]) -> Tuple[float, ...]:
    vectors = list(vectors)
    return tuple(sum(vector[index] for vector in vectors) / len(vectors) for index in range(KEY_DIM))


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
    key_token: str


def build_samples(key_tokens: Sequence[str]) -> Tuple[List[Sample], List[Sample]]:
    queries: List[Sample] = []
    gallery: List[Sample] = []
    cursor = 0
    for pid in range(N_IDENTITIES):
        queries.append(Sample(f"q_{pid}", pid, 0, key_tokens[cursor]))
        cursor += 1
        for replica in range(N_GALLERY):
            gallery.append(Sample(f"g_{pid}_{replica}", pid, 1, key_tokens[cursor]))
            cursor += 1
    return queries, gallery


def donor_map(samples: Sequence[Sample]) -> Dict[str, Sample]:
    by_slot = {(sample.camera, sample.pid, sample.name.rsplit("_", 1)[-1]): sample for sample in samples}
    donors: Dict[str, Sample] = {}
    for sample in samples:
        suffix = sample.name.rsplit("_", 1)[-1]
        donor = by_slot[(sample.camera, (sample.pid + 1) % N_IDENTITIES, suffix)]
        donors[sample.name] = donor
    return donors


def mismatch_nuisance(host_token: str, supplied_token: str) -> Tuple[float, ...]:
    return unit_random(stable_seed("nuisance", host_token, supplied_token), NUISANCE_DIM)


def descriptor(
    sample: Sample,
    supplied_token: str,
    identity: Sequence[float],
    key_vectors: Dict[str, Tuple[float, ...]],
    mutant: bool,
) -> Tuple[float, ...]:
    if mutant:
        quota = 0.45
    else:
        is_real = supplied_token.startswith("key_")
        is_match = supplied_token == sample.key_token
        quota = 0.08 + 0.37 * float(is_real) + 0.55 * float(is_match)
    nuisance_scale = math.sqrt(1.0 - quota * quota)
    nuisance = mismatch_nuisance(sample.key_token, supplied_token)
    result = tuple(quota * value for value in identity) + tuple(nuisance_scale * value for value in nuisance)
    if supplied_token in key_vectors:
        assert len(key_vectors[supplied_token]) == KEY_DIM
    return result


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
        ranked = sorted(gallery, key=lambda item: dot(descriptors[query.name], descriptors[item.name]), reverse=True)
        aps.append(average_precision([item.pid for item in ranked], query.pid))
        rank1 += int(ranked[0].pid == query.pid)
    return {"mAP": sum(aps) / len(aps), "R1": rank1 / len(queries)}


def execute(key_tokens: Sequence[str], key_vectors: Dict[str, Tuple[float, ...]], mutant: bool) -> Dict[str, object]:
    queries, gallery = build_samples(key_tokens)
    all_samples = queries + gallery
    donors = donor_map(queries) | donor_map(gallery)
    generic_vector = mean_vector(key_vectors[token] for token in key_tokens)
    key_vectors = dict(key_vectors)
    key_vectors["generic"] = generic_vector
    key_vectors["null"] = tuple(0.0 for _ in range(KEY_DIM))
    identities = {pid: unit_random(SEED + 100000 + pid, ID_DIM) for pid in range(N_IDENTITIES)}

    supplied = {
        "correct": lambda sample: sample.key_token,
        "wrong": lambda sample: donors[sample.name].key_token,
        "generic": lambda sample: "generic",
        "null": lambda sample: "null",
    }
    arm_metrics: Dict[str, Dict[str, float]] = {}
    max_norm_error = 0.0
    for arm, supplier in supplied.items():
        arm_descriptors = {
            sample.name: descriptor(sample, supplier(sample), identities[sample.pid], key_vectors, mutant)
            for sample in all_samples
        }
        max_norm_error = max(max_norm_error, max(abs(norm(value) - 1.0) for value in arm_descriptors.values()))
        arm_metrics[arm] = retrieval_metrics(queries, gallery, arm_descriptors)

    same_cosines = [dot(key_vectors[queries[pid].key_token], key_vectors[gallery[pid * N_GALLERY].key_token]) for pid in range(N_IDENTITIES)]
    diff_cosines = [
        dot(key_vectors[queries[pid].key_token], key_vectors[gallery[((pid + 1) % N_IDENTITIES) * N_GALLERY].key_token])
        for pid in range(N_IDENTITIES)
    ]
    same_mean = sum(same_cosines) / len(same_cosines)
    diff_mean = sum(diff_cosines) / len(diff_cosines)
    correct_minus_wrong = arm_metrics["correct"]["mAP"] - arm_metrics["wrong"]["mAP"]
    wrong_minus_low = arm_metrics["wrong"]["mAP"] - max(arm_metrics["generic"]["mAP"], arm_metrics["null"]["mAP"])
    gates = {
        "correct_map": arm_metrics["correct"]["mAP"] >= 0.99,
        "correct_minus_wrong": correct_minus_wrong >= 0.05,
        "wrong_minus_low": wrong_minus_low >= 0.05,
        "random_key_no_pid_geometry": abs(same_mean - diff_mean) < 0.05,
        "donor_contract": all(
            donors[sample.name].camera == sample.camera
            and donors[sample.name].pid != sample.pid
            and donors[sample.name].name != sample.name
            for sample in all_samples
        ),
        "unit_norm": max_norm_error < 1e-10,
    }
    return {
        "arms": arm_metrics,
        "correct_minus_wrong_map": correct_minus_wrong,
        "wrong_minus_max_generic_null_map": wrong_minus_low,
        "key_same_pid_cosine_mean": same_mean,
        "key_different_pid_cosine_mean": diff_mean,
        "key_geometry_gap_abs": abs(same_mean - diff_mean),
        "generic_key_norm": norm(generic_vector),
        "max_descriptor_norm_error": max_norm_error,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> int:
    count = N_IDENTITIES * (1 + N_GALLERY)
    tokens = [f"key_{index}" for index in range(count)]
    key_vectors = {token: unit_random(SEED + index, KEY_DIM) for index, token in enumerate(tokens)}
    original = execute(tokens, key_vectors, mutant=False)
    shuffled = list(tokens)
    random.Random(SEED + 1).shuffle(shuffled)
    permuted = execute(shuffled, key_vectors, mutant=False)
    mutant = execute(tokens, key_vectors, mutant=True)
    mutant_caught = not (
        mutant["gates"]["correct_minus_wrong"] and mutant["gates"]["wrong_minus_low"]
    )
    positive_pass = bool(original["all_gates_pass"] and permuted["all_gates_pass"])
    result = {
        "diagnostic": "post_exp403_source_key_shortcut",
        "seed": SEED,
        "dimensions": {"identity": ID_DIM, "key": KEY_DIM, "nuisance": NUISANCE_DIM},
        "counts": {"identities": N_IDENTITIES, "queries": N_IDENTITIES, "gallery": N_IDENTITIES * N_GALLERY},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "original": original,
        "semantic_blind_key_permutation": permuted,
        "constant_quota_mutant": mutant,
        "mutant_caught": mutant_caught,
        "verdict": (
            "RANDOM_SOURCE_KEY_FALSE_OWNERSHIP_DEMONSTRATED"
            if positive_pass and mutant_caught
            else "DIAGNOSTIC_INCONCLUSIVE"
        ),
        "exp404_authorized": False,
        "gpu_start_authorized": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "RANDOM_SOURCE_KEY_FALSE_OWNERSHIP_DEMONSTRATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
