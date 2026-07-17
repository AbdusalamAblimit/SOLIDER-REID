#!/usr/bin/env python3
"""Class-free, strict-path-LOO frozen support oracle for exp371 Gate C.

This script only reads cached 7-block descriptors.  Ground-truth PID is used
to construct an oracle support set and to score retrieval relations; no
classifier, linear probe, or model weight is loaded or fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ARMS = ("SELF", "ID-MEAN", "PART-EQUAL", "PART-PERM", "CASD-LIKE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--distance-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=371)
    parser.add_argument("--expected-block-dim", type=int, default=768)
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--max-val-queries", type=int, default=0)
    parser.add_argument("--min-map-gap", type=float, default=0.005)
    parser.add_argument("--min-structure-gap", type=float, default=0.001)
    parser.add_argument("--min-eligible-ratio", type=float, default=0.50)
    return parser.parse_args()


def atomic_json(path: Path, payload: Mapping) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def normalized_path(path: object) -> str:
    return os.path.normpath(str(path))


def _torch_load(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def load_cache(path: Path, expected_block_dim: Optional[int] = None) -> Dict:
    cache = _torch_load(path)
    required = {
        "features", "pids", "camids", "paths", "split", "mode",
        "num_query", "block_dim", "weight_sha256",
    }
    missing = required.difference(cache)
    if missing:
        raise ValueError("cache %s is missing keys: %s" % (path, sorted(missing)))

    features = cache["features"]
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("features must be a 2-D tensor")
    block_dim = int(cache["block_dim"])
    if expected_block_dim is not None and block_dim != expected_block_dim:
        raise ValueError(
            "block dim mismatch: %d != %d" % (block_dim, expected_block_dim)
        )
    if features.shape[1] != 7 * block_dim:
        raise ValueError(
            "expected 7x%d features, got %s" % (block_dim, tuple(features.shape))
        )
    sample_count = features.shape[0]
    for key in ("pids", "camids", "paths"):
        if len(cache[key]) != sample_count:
            raise ValueError("cache key %s has the wrong length" % key)
    if not torch.isfinite(features).all():
        raise ValueError("cache contains NaN/Inf")

    result = dict(cache)
    result["features"] = features.float().contiguous()
    result["pids"] = [int(value) for value in cache["pids"]]
    result["camids"] = [int(value) for value in cache["camids"]]
    result["paths"] = [normalized_path(value) for value in cache["paths"]]
    blocks = result["features"].view(sample_count, 7, block_dim)
    norms = blocks.norm(dim=2)
    result["block_norm_stats"] = {
        "min": float(norms.min().item()),
        "max": float(norms.max().item()),
        "mean": float(norms.mean().item()),
        "max_abs_error_from_one": float((norms - 1.0).abs().max().item()),
    }
    return result


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def stable_path_key(path: str, seed: int) -> str:
    return hashlib.sha256((str(seed) + ":" + path).encode("utf-8")).hexdigest()


def path_representatives(indices: Iterable[int], paths: Sequence[str]) -> List[int]:
    first: Dict[str, int] = {}
    for index in indices:
        first.setdefault(paths[index], int(index))
    return [first[path] for path in sorted(first)]


def build_train_protocol(cache: Dict, seed: int, max_queries: int = 0) -> Dict:
    by_pid: Dict[int, List[int]] = {}
    for index, pid in enumerate(cache["pids"]):
        by_pid.setdefault(pid, []).append(index)

    query_indices: List[int] = []
    held_out_paths = set()
    for pid in sorted(by_pid):
        reps = path_representatives(by_pid[pid], cache["paths"])
        reps.sort(key=lambda idx: stable_path_key(cache["paths"][idx], seed))
        if len(reps) < 4:
            continue
        query_indices.append(reps[0])
        held_out_paths.add(cache["paths"][reps[0]])

    if max_queries > 0:
        query_indices = query_indices[:max_queries]
        held_out_paths = {cache["paths"][index] for index in query_indices}
    gallery_indices = path_representatives(
        (
            index for index in range(len(cache["pids"]))
            if cache["paths"][index] not in held_out_paths
        ),
        cache["paths"],
    )
    if not query_indices or not gallery_indices:
        raise ValueError("could not build deterministic train pseudo protocol")
    return {
        "name": "train_pseudo",
        "query_indices": query_indices,
        "gallery_indices": gallery_indices,
        "exclude_same_pid_cam": False,
        "original_query_count": len(query_indices),
    }


def build_val_protocol(cache: Dict, max_queries: int = 0) -> Dict:
    num_query = int(cache["num_query"])
    if num_query <= 0 or num_query >= len(cache["pids"]):
        raise ValueError("validation cache requires 0 < num_query < N")
    query_indices = list(range(num_query))
    if max_queries > 0:
        query_indices = query_indices[:max_queries]
    return {
        "name": "val",
        "query_indices": query_indices,
        "gallery_indices": list(range(num_query, len(cache["pids"]))),
        "exclude_same_pid_cam": True,
        "original_query_count": len(query_indices),
    }


def mean_path_blocks(
    all_blocks: torch.Tensor,
    indices: Sequence[int],
) -> torch.Tensor:
    value = all_blocks[list(indices)].mean(dim=0)
    return F.normalize(value, p=2, dim=1)


def build_episode(cache: Dict, protocol: Dict) -> Dict:
    block_dim = int(cache["block_dim"])
    blocks = cache["features"].view(-1, 7, block_dim)
    gallery_indices = list(protocol["gallery_indices"])

    donor_by_pid: Dict[int, Dict[str, torch.Tensor]] = {}
    gallery_path_groups: Dict[Tuple[int, str], List[int]] = {}
    for local_index, cache_index in enumerate(gallery_indices):
        pid = cache["pids"][cache_index]
        path = cache["paths"][cache_index]
        gallery_path_groups.setdefault((pid, path), []).append(local_index)
    for (pid, path), local_indices in gallery_path_groups.items():
        cache_indices = [gallery_indices[index] for index in local_indices]
        donor_by_pid.setdefault(pid, {})[path] = mean_path_blocks(
            blocks, cache_indices
        )

    eligible: List[int] = []
    donors: List[List[Tuple[str, torch.Tensor]]] = []
    for cache_index in protocol["query_indices"]:
        pid = cache["pids"][cache_index]
        query_path = cache["paths"][cache_index]
        available = [
            (path, value)
            for path, value in sorted(donor_by_pid.get(pid, {}).items())
            if path != query_path
        ]
        if len(available) < 2:
            continue
        eligible.append(cache_index)
        donors.append(available)

    if not eligible:
        raise ValueError("no query has at least two strict-path LOO donors")

    q_pids = torch.tensor([cache["pids"][index] for index in eligible], dtype=torch.long)
    q_camids = torch.tensor([cache["camids"][index] for index in eligible], dtype=torch.long)
    q_paths = [cache["paths"][index] for index in eligible]
    g_pids = torch.tensor(
        [cache["pids"][index] for index in gallery_indices], dtype=torch.long
    )
    g_camids = torch.tensor(
        [cache["camids"][index] for index in gallery_indices], dtype=torch.long
    )
    g_paths = [cache["paths"][index] for index in gallery_indices]

    path_ids = {
        path: index for index, path in enumerate(sorted(set(q_paths).union(g_paths)))
    }
    q_path_ids = torch.tensor([path_ids[path] for path in q_paths], dtype=torch.long)
    g_path_ids = torch.tensor([path_ids[path] for path in g_paths], dtype=torch.long)
    path_different = q_path_ids[:, None].ne(g_path_ids[None, :])
    same_pid = q_pids[:, None].eq(g_pids[None, :])
    valid = path_different.clone()
    if protocol["exclude_same_pid_cam"]:
        same_cam = q_camids[:, None].eq(g_camids[None, :])
        valid &= ~(same_pid & same_cam)
    positive = valid & same_pid
    negative = valid & ~same_pid

    keep = positive.any(dim=1) & negative.any(dim=1)
    if not keep.all():
        keep_indices = keep.nonzero(as_tuple=False).flatten().tolist()
        eligible = [eligible[index] for index in keep_indices]
        donors = [donors[index] for index in keep_indices]
        q_pids = q_pids[keep]
        q_camids = q_camids[keep]
        q_paths = [q_paths[index] for index in keep_indices]
        valid = valid[keep]
        positive = positive[keep]
        negative = negative[keep]

    if not eligible:
        raise ValueError("no eligible query has both positive and negative relations")

    return {
        "name": protocol["name"],
        "all_blocks": blocks,
        "query_indices": eligible,
        "gallery_indices": gallery_indices,
        "donors": donors,
        "q_pids": q_pids,
        "q_camids": q_camids,
        "q_paths": q_paths,
        "g_pids": g_pids,
        "g_camids": g_camids,
        "g_paths": g_paths,
        "gallery_path_groups": gallery_path_groups,
        "valid": valid,
        "positive": positive,
        "negative": negative,
        "exclude_same_pid_cam": bool(protocol["exclude_same_pid_cam"]),
        "original_query_count": int(protocol["original_query_count"]),
        "eligible_query_count": len(eligible),
        "eligible_query_ratio": len(eligible) / max(1, int(protocol["original_query_count"])),
        "exact_duplicate_gallery_paths": len(gallery_indices) - len(set(g_paths)),
    }


def cyclic_derangement(path: str, slot_count: int, seed: int) -> torch.Tensor:
    if slot_count < 2:
        raise ValueError("slot_count must be at least 2")
    digest = hashlib.sha256((str(seed) + ":perm:" + path).encode("utf-8")).digest()
    shift = 1 + int.from_bytes(digest[:8], "little") % (slot_count - 1)
    return (torch.arange(slot_count) + shift) % slot_count


def stack_donors(donors: Sequence[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    if not donors:
        raise ValueError("support donor set is empty")
    return [item[0] for item in donors], torch.stack([item[1] for item in donors], dim=0)


def support_descriptor(
    anchor_blocks: torch.Tensor,
    donors: Sequence[Tuple[str, torch.Tensor]],
    arm: str,
    seed: int,
) -> torch.Tensor:
    if arm not in ARMS:
        raise ValueError("unknown arm: %s" % arm)
    anchor = F.normalize(anchor_blocks.float(), p=2, dim=1)
    if arm == "SELF":
        return F.normalize(anchor.reshape(-1), p=2, dim=0)

    paths, donor_blocks = stack_donors(donors)
    donor_blocks = F.normalize(donor_blocks.float(), p=2, dim=2)
    output = torch.empty_like(anchor)
    output[0] = anchor[0]

    if arm == "ID-MEAN":
        identity_local = F.normalize(
            donor_blocks[:, 1:, :].reshape(-1, anchor.shape[1]).mean(dim=0),
            p=2,
            dim=0,
        )
        output[1:] = identity_local.unsqueeze(0).expand_as(output[1:])
    elif arm == "PART-EQUAL":
        output[1:] = F.normalize(donor_blocks[:, 1:, :].mean(dim=0), p=2, dim=1)
    elif arm == "PART-PERM":
        permuted = donor_blocks.clone()
        for donor_index, path in enumerate(paths):
            order = cyclic_derangement(path, 5, seed)
            permuted[donor_index, 2:7] = donor_blocks[donor_index, 2:7][order]
        output[1:] = F.normalize(permuted[:, 1:, :].mean(dim=0), p=2, dim=1)
    else:  # CASD-LIKE
        output[1] = F.normalize(donor_blocks[:, 1, :].mean(dim=0), p=2, dim=0)
        for block_index in range(2, 7):
            values = donor_blocks[:, block_index, :]
            consensus = F.normalize(values.mean(dim=0), p=2, dim=0)
            agreement = (values @ consensus).clamp_min(0.0) + 1e-8
            weights = agreement / agreement.sum()
            output[block_index] = F.normalize(
                (weights[:, None] * values).sum(dim=0), p=2, dim=0
            )
    return F.normalize(output.reshape(-1), p=2, dim=0)


def pairwise_squared_l2(
    query: torch.Tensor,
    gallery: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("distance batch must be positive")
    q = F.normalize(query.float(), p=2, dim=1)
    g = F.normalize(gallery.float(), p=2, dim=1).to(device)
    output = torch.empty((q.shape[0], g.shape[0]), dtype=torch.float32)
    for start in range(0, q.shape[0], batch_size):
        chunk = q[start:start + batch_size].to(device)
        distance = (2.0 - 2.0 * (chunk @ g.t())).clamp_min(0.0)
        output[start:start + chunk.shape[0]] = distance.cpu()
    return output


def compute_arm_distances(
    episode: Dict,
    arm: str,
    seed: int,
    device: torch.device,
    distance_batch: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    blocks = episode["all_blocks"]
    query_desc: List[torch.Tensor] = []
    strict_path_violations = 0
    for query_index, donors in zip(episode["query_indices"], episode["donors"]):
        query_path = normalized_path(episode["q_paths"][len(query_desc)])
        if any(normalized_path(path) == query_path for path, _value in donors):
            strict_path_violations += 1
        query_desc.append(support_descriptor(blocks[query_index], donors, arm, seed))
    query_matrix = torch.stack(query_desc, dim=0)
    gallery_matrix = episode["all_blocks"][episode["gallery_indices"]].reshape(
        len(episode["gallery_indices"]), -1
    )
    distances = pairwise_squared_l2(
        query_matrix, gallery_matrix, device=device, batch_size=distance_batch
    )

    endpoint_corrections = 0
    endpoint_exclusion_violations = 0
    if arm != "SELF":
        gallery_norm = F.normalize(gallery_matrix.float(), p=2, dim=1)
        for query_local, (query_index, donors) in enumerate(
            zip(episode["query_indices"], episode["donors"])
        ):
            pid = int(episode["q_pids"][query_local].item())
            positive_paths = {
                episode["g_paths"][gallery_local]
                for gallery_local in episode["positive"][query_local].nonzero(
                    as_tuple=False
                ).flatten().tolist()
            }
            for endpoint_path in positive_paths:
                reduced = [item for item in donors if item[0] != endpoint_path]
                if not reduced:
                    endpoint_exclusion_violations += 1
                    continue
                if any(item[0] == endpoint_path for item in reduced):
                    endpoint_exclusion_violations += 1
                descriptor = support_descriptor(blocks[query_index], reduced, arm, seed)
                gallery_positions = episode["gallery_path_groups"].get(
                    (pid, endpoint_path), []
                )
                if not gallery_positions:
                    continue
                values = gallery_norm[gallery_positions]
                corrected = (2.0 - 2.0 * (values @ descriptor)).clamp_min(0.0)
                distances[query_local, gallery_positions] = corrected
                endpoint_corrections += len(gallery_positions)

    if strict_path_violations or endpoint_exclusion_violations:
        raise AssertionError(
            "strict path LOO failed: anchor=%d endpoint=%d"
            % (strict_path_violations, endpoint_exclusion_violations)
        )
    return distances, {
        "query_descriptor_sha256": tensor_sha256(query_matrix),
        "strict_path_violations": strict_path_violations,
        "endpoint_exclusion_violations": endpoint_exclusion_violations,
        "endpoint_corrections": endpoint_corrections,
    }


def average_precision(matches: torch.Tensor) -> float:
    positive_count = int(matches.sum().item())
    if positive_count == 0:
        raise ValueError("average precision requires a positive reference")
    precision = matches.float().cumsum(dim=0) / torch.arange(
        1, matches.numel() + 1, dtype=torch.float32
    )
    return float((precision * matches.float()).sum().item() / positive_count)


def retrieval_metrics(distances: torch.Tensor, episode: Dict) -> Dict[str, float]:
    aps: List[float] = []
    ranks = {1: [], 5: [], 10: []}
    for query_index in range(distances.shape[0]):
        valid = episode["valid"][query_index]
        labels = episode["positive"][query_index][valid]
        order = torch.argsort(distances[query_index][valid])
        matches = labels[order]
        if not matches.any():
            continue
        aps.append(average_precision(matches))
        first = int(matches.nonzero(as_tuple=False)[0].item())
        for rank in ranks:
            ranks[rank].append(float(first < rank))
    if not aps:
        raise ValueError("retrieval protocol has no valid queries")
    return {
        "mAP": float(np.mean(aps)),
        "rank1": float(np.mean(ranks[1])),
        "rank5": float(np.mean(ranks[5])),
        "rank10": float(np.mean(ranks[10])),
        "evaluated_queries": len(aps),
    }


def signed_relation_gain(
    distances: torch.Tensor,
    self_distances: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    gain = torch.zeros_like(distances)
    gain[positive] = self_distances[positive] - distances[positive]
    gain[negative] = distances[negative] - self_distances[negative]
    return gain


def balanced_weights(
    mask: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    raw_weights: torch.Tensor,
) -> torch.Tensor:
    weights = torch.zeros_like(raw_weights, dtype=torch.float32)
    pos = mask & positive
    neg = mask & negative
    if not pos.any() or not neg.any():
        raise ValueError("balanced protocol requires positive and negative relations")
    pos_values = raw_weights[pos].float().clamp_min(0.0)
    neg_values = raw_weights[neg].float().clamp_min(0.0)
    if float(pos_values.sum().item()) <= 0 or float(neg_values.sum().item()) <= 0:
        raise ValueError("balanced raw weight mass must be positive in both classes")
    weights[pos] = 0.5 * pos_values / pos_values.sum()
    weights[neg] = 0.5 * neg_values / neg_values.sum()
    return weights


def weight_mass(weights: torch.Tensor, episode: Dict) -> Dict[str, float]:
    return {
        "positive": float(weights[episode["positive"]].sum().item()),
        "negative": float(weights[episode["negative"]].sum().item()),
        "total": float(weights.sum().item()),
        "nonzero_relations": int((weights > 0).sum().item()),
    }


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> float:
    mass = weights[mask].sum()
    if float(mass.item()) <= 0:
        return float("nan")
    return float((values[mask] * weights[mask]).sum().item() / mass.item())


def arm_diagnostics(
    distances: torch.Tensor,
    self_distances: torch.Tensor,
    episode: Dict,
    shared_weights: torch.Tensor,
) -> Dict[str, object]:
    valid = episode["valid"]
    positive = episode["positive"]
    negative = episode["negative"]
    gain = signed_relation_gain(distances, self_distances, positive, negative)
    shared_mask = shared_weights > 0
    class_balanced_gain = 0.5 * gain[positive].mean() + 0.5 * gain[negative].mean()
    output: Dict[str, object] = {
        "retrieval": retrieval_metrics(distances, episode),
        "positive_distance_mean": float(distances[positive].mean().item()),
        "negative_distance_mean": float(distances[negative].mean().item()),
        "class_margin": float(
            distances[negative].mean().item() - distances[positive].mean().item()
        ),
        "all_pair_class_balanced_signed_gain": float(class_balanced_gain.item()),
        "shared_mask": {
            "signed_gain": float((gain * shared_weights).sum().item()),
            "positive_gain": _weighted_mean(gain, shared_weights, positive),
            "negative_gain": _weighted_mean(gain, shared_weights, negative),
            "target_shift": float(
                ((distances - self_distances).abs() * shared_weights).sum().item()
            ),
            "smooth_l1_to_self": float(
                (
                    F.smooth_l1_loss(
                        self_distances, distances, reduction="none"
                    ) * shared_weights
                ).sum().item()
            ),
            "harm_mass_fraction": float(
                shared_weights[shared_mask & (gain < 0)].sum().item()
                / max(float(shared_weights.sum().item()), 1e-12)
            ),
        },
        "own_positive_gain_coverage": float(
            ((gain > 0) & valid).sum().item() / max(1, int(valid.sum().item()))
        ),
    }
    return output


def exp123_full_relational_control(
    casd_distances: torch.Tensor,
    self_distances: torch.Tensor,
    episode: Dict,
) -> Dict[str, object]:
    valid = episode["valid"]
    delta = (casd_distances - self_distances).abs()
    masked_delta = delta * valid.float()
    row_max = masked_delta.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    focus = 1.0 + masked_delta / row_max
    weights = balanced_weights(valid, episode["positive"], episode["negative"], focus)
    gain = signed_relation_gain(
        casd_distances, self_distances, episode["positive"], episode["negative"]
    )
    return {
        "definition": "full valid relations; focus=1+abs(CASD-SELF)/row_max",
        "coverage": float(valid.sum().item() / valid.numel()),
        "loss_mass": weight_mass(weights, episode),
        "class_balanced_signed_gain": float((gain * weights).sum().item()),
        "positive_gain": _weighted_mean(gain, weights, episode["positive"]),
        "negative_gain": _weighted_mean(gain, weights, episode["negative"]),
        "pair_delta_mean": float(delta[valid].mean().item()),
        "pair_focus_mean": float(focus[valid].mean().item()),
        "smooth_l1_to_self": float(
            (
                F.smooth_l1_loss(
                    self_distances, casd_distances, reduction="none"
                ) * weights
            ).sum().item()
        ),
    }


def assert_loss_mass(mass: Mapping[str, float], tolerance: float = 1e-6) -> None:
    expected = {"positive": 0.5, "negative": 0.5, "total": 1.0}
    for key, value in expected.items():
        if abs(float(mass[key]) - value) > tolerance:
            raise AssertionError("loss mass %s is %.9f, expected %.9f" % (
                key, float(mass[key]), value
            ))


def run_episode(
    cache: Dict,
    protocol: Dict,
    seed: int,
    device: torch.device,
    distance_batch: int,
    min_map_gap: float,
    min_structure_gap: float,
    min_eligible_ratio: float,
) -> Dict[str, object]:
    episode = build_episode(cache, protocol)
    distances: Dict[str, torch.Tensor] = {}
    arm_meta: Dict[str, Dict[str, object]] = {}
    for arm in ("SELF", "CASD-LIKE"):
        distances[arm], arm_meta[arm] = compute_arm_distances(
            episode, arm, seed, device, distance_batch
        )

    self_distances = distances["SELF"]
    casd_distances = distances["CASD-LIKE"]
    casd_gain = signed_relation_gain(
        casd_distances, self_distances, episode["positive"], episode["negative"]
    )
    shared_mask = episode["valid"] & (casd_gain > 0)
    shared_weights = balanced_weights(
        shared_mask,
        episode["positive"],
        episode["negative"],
        casd_gain.clamp_min(0.0),
    )
    shared_mass = weight_mass(shared_weights, episode)
    assert_loss_mass(shared_mass)

    results: Dict[str, object] = {
        "protocol": {
            "name": episode["name"],
            "original_query_count": episode["original_query_count"],
            "eligible_query_count": episode["eligible_query_count"],
            "eligible_query_ratio": episode["eligible_query_ratio"],
            "gallery_count": len(episode["gallery_indices"]),
            "valid_relations": int(episode["valid"].sum().item()),
            "positive_relations": int(episode["positive"].sum().item()),
            "negative_relations": int(episode["negative"].sum().item()),
            "exact_duplicate_gallery_paths": episode["exact_duplicate_gallery_paths"],
            "strict_path_definition": "exclude anchor path and positive relation endpoint path",
        },
        "shared_advantage": {
            "definition": "CASD-LIKE signed gain > 0; identical for every arm",
            "coverage": float(shared_mask.sum().item() / episode["valid"].sum().item()),
            "loss_mass": shared_mass,
            "mask_sha256": tensor_sha256(shared_mask.to(torch.uint8)),
            "weight_sha256": tensor_sha256(shared_weights),
        },
        "arms": {},
    }

    for arm in ARMS:
        if arm in distances:
            arm_distances = distances[arm]
        else:
            arm_distances, arm_meta[arm] = compute_arm_distances(
                episode, arm, seed, device, distance_batch
            )
        diagnostic = arm_diagnostics(
            arm_distances, self_distances, episode, shared_weights
        )
        diagnostic["loo"] = arm_meta[arm]
        diagnostic["shared_mask_sha256"] = results["shared_advantage"]["mask_sha256"]
        diagnostic["shared_weight_sha256"] = results["shared_advantage"]["weight_sha256"]
        diagnostic["loss_mass"] = shared_mass
        results["arms"][arm] = diagnostic

    full_relational = exp123_full_relational_control(
        casd_distances, self_distances, episode
    )
    assert_loss_mass(full_relational["loss_mass"])
    results["EXP123-FULL"] = full_relational

    arm_maps = {
        arm: float(results["arms"][arm]["retrieval"]["mAP"])
        for arm in ARMS
    }
    controls = ("ID-MEAN", "PART-EQUAL", "PART-PERM")
    strongest = max(controls, key=lambda name: arm_maps[name])
    shared_gains = {
        arm: float(results["arms"][arm]["shared_mask"]["signed_gain"])
        for arm in ARMS
    }
    gate = {
        "strongest_support_control": strongest,
        "casd_vs_strongest_map_gap": arm_maps["CASD-LIKE"] - arm_maps[strongest],
        "casd_beats_strongest_by_min_gap": bool(
            arm_maps["CASD-LIKE"] - arm_maps[strongest] >= min_map_gap
        ),
        "part_equal_beats_part_perm": bool(
            arm_maps["PART-EQUAL"] - arm_maps["PART-PERM"] >= min_structure_gap
        ),
        "casd_beats_part_equal": bool(
            arm_maps["CASD-LIKE"] - arm_maps["PART-EQUAL"] >= min_structure_gap
        ),
        "casd_shared_gain_beats_all_controls": bool(
            shared_gains["CASD-LIKE"] > max(shared_gains[name] for name in controls)
        ),
        "advantage_beats_loss_matched_exp123_full": bool(
            shared_gains["CASD-LIKE"]
            > float(full_relational["class_balanced_signed_gain"])
        ),
        "eligible_query_ratio_pass": bool(
            episode["eligible_query_ratio"] >= min_eligible_ratio
        ),
        "strict_path_loo_pass": bool(
            all(
                int(arm_meta[name]["strict_path_violations"]) == 0
                and int(arm_meta[name]["endpoint_exclusion_violations"]) == 0
                for name in ARMS
            )
        ),
    }
    gate["provisional_go"] = bool(all(
        gate[key] for key in (
            "casd_beats_strongest_by_min_gap",
            "part_equal_beats_part_perm",
            "casd_beats_part_equal",
            "casd_shared_gain_beats_all_controls",
            "advantage_beats_loss_matched_exp123_full",
            "eligible_query_ratio_pass",
            "strict_path_loo_pass",
        )
    ))
    results["gate"] = gate
    return results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train_cache).resolve()
    val_path = Path(args.val_cache).resolve()
    train = load_cache(train_path, args.expected_block_dim)
    val = load_cache(val_path, args.expected_block_dim)

    if int(train["num_query"]) != 0:
        raise ValueError("train cache must have num_query=0")
    if int(val["num_query"]) <= 0:
        raise ValueError("val cache must have num_query>0")
    if train["weight_sha256"] != val["weight_sha256"]:
        raise ValueError("train/val caches came from different checkpoints")
    train_paths = set(train["paths"])
    val_paths = set(val["paths"])
    overlap = train_paths.intersection(val_paths)
    if overlap:
        raise ValueError("train/val path leakage detected: %d" % len(overlap))
    if args.distance_batch <= 0:
        raise ValueError("--distance-batch must be positive")

    device = choose_device(args.device)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "design": str(Path(__file__).with_name("frozen_support_oracle_design.md")),
        "train_cache": str(train_path),
        "train_cache_sha256": file_sha256(train_path),
        "val_cache": str(val_path),
        "val_cache_sha256": file_sha256(val_path),
        "checkpoint_sha256": train["weight_sha256"],
        "train_val_path_overlap": 0,
        "train_samples": int(train["features"].shape[0]),
        "val_samples": int(val["features"].shape[0]),
        "block_dim": int(train["block_dim"]),
        "num_blocks": 7,
        "train_block_norm_stats": train["block_norm_stats"],
        "val_block_norm_stats": val["block_norm_stats"],
        "seed": int(args.seed),
        "device": str(device),
        "distance_batch": int(args.distance_batch),
        "max_train_queries": int(args.max_train_queries),
        "max_val_queries": int(args.max_val_queries),
        "min_map_gap": float(args.min_map_gap),
        "min_structure_gap": float(args.min_structure_gap),
        "min_eligible_ratio": float(args.min_eligible_ratio),
        "class_free": True,
        "classifier_loaded": False,
        "classifier_ce_used": False,
        "support_uses_ground_truth_pid": True,
        "absolute_metrics_are_reportable_reid_results": False,
    }
    atomic_json(output_dir / "manifest.json", manifest)

    train_protocol = build_train_protocol(
        train, args.seed, max_queries=args.max_train_queries
    )
    train_results = run_episode(
        train, train_protocol, args.seed, device, args.distance_batch,
        args.min_map_gap, args.min_structure_gap, args.min_eligible_ratio,
    )
    atomic_json(output_dir / "train_results.json", train_results)
    print(json.dumps({"train": train_results["gate"]}, indent=2), flush=True)

    val_protocol = build_val_protocol(val, max_queries=args.max_val_queries)
    val_results = run_episode(
        val, val_protocol, args.seed, device, args.distance_batch,
        args.min_map_gap, args.min_structure_gap, args.min_eligible_ratio,
    )
    atomic_json(output_dir / "val_results.json", val_results)
    print(json.dumps({"val": val_results["gate"]}, indent=2), flush=True)

    summary = {
        "primary_split": "val",
        "train_gate_is_diagnostic_only": train_results["gate"],
        "val_gate": val_results["gate"],
        "final_provisional_go": bool(val_results["gate"]["provisional_go"]),
        "boundary": (
            "Frozen GT-support oracle only; does not establish student learnability, "
            "pose visibility causality, deployment availability, or paper novelty."
        ),
    }
    atomic_json(output_dir / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("COMPLETE: %s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
