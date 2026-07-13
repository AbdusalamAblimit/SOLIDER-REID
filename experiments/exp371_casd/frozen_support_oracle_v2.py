#!/usr/bin/env python3
"""Leakage-safe frozen support-geometry oracle for exp371 CASD Gate C v2.

The oracle is deliberately GT-supported: PID labels build support episodes and
score retrieval, but no classifier is fitted.  Gallery images are split within
PID into deterministic support/reference folds.  Each query/arm/fold has one
descriptor which is used against every reference endpoint in that fold.

This program does not train a student and must not be used to claim superiority
to MVI2P, UMTS, LCR2S, or exp123.  It only screens frozen support routing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


CACHE_SCHEMA = "exp371_target_support_cache_v1"
BLOCK_NAMES = ("global", "pooled", "slot1", "slot2", "slot3", "slot4", "slot5")
SLOT_COUNT = 5
FOLD_COUNT = 5
MIN_DONORS = 3
SELECTED_DONORS = 3
MIN_ELIGIBLE_RATIO = 0.70
ACTIVE_EPS = 1e-12

MAIN_ARMS = (
    "SELF",
    "ID-GLOBAL",
    "ID-MEAN",
    "PART-EQUAL",
    "SLOT-PERM",
    "AGREE",
    "POSE-RESP",
    "RESP-PERM",
    "FULL-INCL",
    "FULL-LOO",
    "WRONG-ID",
)
ROUTING_ARMS = ("PART-EQUAL", "POSE-RESP", "RESP-PERM")
POSE_CONTROLS = (
    "ID-GLOBAL",
    "ID-MEAN",
    "PART-EQUAL",
    "SLOT-PERM",
    "AGREE",
    "RESP-PERM",
)
SCENE_ARMS = ("SELF",) + POSE_CONTROLS + ("POSE-RESP", "WRONG-ID")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--target-content-sidecar")
    parser.add_argument("--canonical-cache")
    parser.add_argument("--canonical-content-sidecar")
    parser.add_argument("--scene-cache")
    parser.add_argument("--scene-content-sidecar")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--distance-batch", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=371)
    parser.add_argument("--permutation-seed", type=int, default=1371)
    parser.add_argument("--bootstrap-seed", type=int, default=2371)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--expected-block-dim", type=int, default=768)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument(
        "--camera-protocol",
        choices=("cross-camera", "unrestricted"),
        default="cross-camera",
    )
    parser.add_argument(
        "--execute-frozen-oracle",
        action="store_true",
        help=(
            "Actually compute frozen retrieval metrics.  Without this explicit flag "
            "the command performs provenance/metadata/eligibility dry-run only."
        ),
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_path(value: object) -> str:
    return os.path.normpath(str(value))


def stable_key(seed: int, namespace: str, value: str) -> str:
    text = "%d:%s:%s" % (seed, namespace, value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _torch_load(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _as_blocks(features: object, block_dim: int) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a tensor")
    if features.ndim == 2 and features.shape[1] == 7 * block_dim:
        result = features.view(features.shape[0], 7, block_dim)
    elif features.ndim == 3 and features.shape[1:] == (7, block_dim):
        result = features
    else:
        raise ValueError("features must have shape [N,7xD] or [N,7,D]")
    result = result.float().contiguous()
    if not torch.isfinite(result).all():
        raise ValueError("features contain NaN/Inf")
    if (result.norm(dim=2) <= 0).any():
        raise ValueError("features contain a zero-norm block")
    return F.normalize(result, p=2, dim=2)


def load_cache(
    path: Path,
    *,
    role: str,
    expected_block_dim: Optional[int] = None,
    content_sidecar: Optional[Path] = None,
) -> Dict[str, object]:
    """Load one cache and fail closed on schema/provenance drift."""
    cache = _torch_load(path)
    required = {
        "schema_version",
        "mode",
        "pose_source",
        "features",
        "raw_pose_response",
        "target_person_valid",
        "person_count",
        "pids",
        "camids",
        "paths",
        "split",
        "num_query",
        "block_dim",
        "weight_sha256",
        "script_sha256",
    }
    missing = sorted(required.difference(cache))
    if missing:
        raise ValueError("cache %s is missing keys: %s" % (path, missing))
    if cache["schema_version"] != CACHE_SCHEMA:
        raise ValueError("unsupported cache schema: %r" % cache["schema_version"])
    if str(cache["split"]) != "val":
        raise ValueError("Gate C v2 requires a val cache")

    mode = str(cache["mode"])
    pose_source = str(cache["pose_source"])
    if role == "target":
        if mode != "target_only_correct" or "target_person" not in pose_source:
            raise ValueError("target cache provenance is not target-only correct")
    elif role == "canonical":
        if "canonical" not in mode.lower() or "canonical" not in pose_source.lower():
            raise ValueError("canonical cache provenance is not canonical")
    elif role == "scene":
        if "scene" not in mode.lower() or "scene" not in pose_source.lower():
            raise ValueError("scene cache provenance is not scene-merged")
    else:
        raise ValueError("unknown cache role: %s" % role)

    block_dim = int(cache["block_dim"])
    if expected_block_dim is not None and block_dim != expected_block_dim:
        raise ValueError("block dim mismatch: %d != %d" % (block_dim, expected_block_dim))
    blocks = _as_blocks(cache["features"], block_dim)
    count = blocks.shape[0]
    num_query = int(cache["num_query"])
    if num_query <= 0 or num_query >= count:
        raise ValueError("val cache requires 0 < num_query < sample count")

    raw = cache["raw_pose_response"]
    valid = cache["target_person_valid"]
    person_count = cache["person_count"]
    if not isinstance(raw, torch.Tensor) or raw.shape != (count, SLOT_COUNT):
        raise ValueError("raw_pose_response must have shape [N,5]")
    if not isinstance(valid, torch.Tensor) or valid.shape != (count,):
        raise ValueError("target_person_valid must have shape [N]")
    if not isinstance(person_count, torch.Tensor) or person_count.shape != (count,):
        raise ValueError("person_count must have shape [N]")
    raw = raw.float().contiguous()
    valid = valid.bool().contiguous()
    person_count = person_count.long().contiguous()
    if not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("raw_pose_response must be finite and nonnegative")
    if (raw[~valid] != 0).any():
        raise ValueError("invalid target person must have zero raw response")

    normalized_paths = [normalized_path(value) for value in cache["paths"]]
    cache_file_sha = file_sha256(path)
    if "content_sha256" in cache:
        content_values = [str(value).lower() for value in cache["content_sha256"]]
        content_provenance: Dict[str, object] = {
            "storage": "inline",
            "sidecar_path": None,
            "sidecar_file_sha256": None,
        }
    else:
        if content_sidecar is None:
            raise ValueError(
                "cache has no inline content_sha256; a SHA-bound content sidecar is required"
            )
        sidecar = json.loads(content_sidecar.read_text())
        required_sidecar = {
            "schema_version",
            "source_cache_path",
            "source_cache_file_sha256",
            "ordered_paths_sha256",
            "sample_count",
            "content_sha256",
            "unique_content_count",
            "duplicate_content_group_count",
            "duplicate_content_sample_count",
        }
        missing_sidecar = sorted(required_sidecar.difference(sidecar))
        if missing_sidecar:
            raise ValueError("content sidecar is missing keys: %s" % missing_sidecar)
        if str(sidecar["source_cache_file_sha256"]).lower() != cache_file_sha:
            raise ValueError("content sidecar source cache SHA mismatch")
        expected_paths_sha = json_sha256(normalized_paths)
        if str(sidecar["ordered_paths_sha256"]).lower() != expected_paths_sha:
            raise ValueError("content sidecar ordered path SHA mismatch")
        if int(sidecar["sample_count"]) != count:
            raise ValueError("content sidecar sample count mismatch")
        content_values = [str(value).lower() for value in sidecar["content_sha256"]]
        if int(sidecar["unique_content_count"]) != len(set(content_values)):
            raise ValueError("content sidecar unique_content_count mismatch")
        content_counts = Counter(content_values)
        duplicate_groups = [count for count in content_counts.values() if count > 1]
        if int(sidecar["duplicate_content_group_count"]) != len(duplicate_groups):
            raise ValueError("content sidecar duplicate_content_group_count mismatch")
        if int(sidecar["duplicate_content_sample_count"]) != sum(duplicate_groups):
            raise ValueError("content sidecar duplicate_content_sample_count mismatch")
        content_provenance = {
            "storage": "sidecar",
            "sidecar_path": str(content_sidecar.resolve()),
            "sidecar_file_sha256": file_sha256(content_sidecar),
            "source_cache_path_recorded": str(sidecar["source_cache_path"]),
            "sidecar_schema_version": str(sidecar["schema_version"]),
        }

    result: Dict[str, object] = dict(cache)
    result.update(
        {
            "features": blocks,
            "raw_pose_response": raw,
            "target_person_valid": valid,
            "person_count": person_count,
            "pids": [int(value) for value in cache["pids"]],
            "camids": [int(value) for value in cache["camids"]],
            "paths": normalized_paths,
            "content_sha256": content_values,
            "block_dim": block_dim,
            "num_query": num_query,
            "cache_path": str(path.resolve()),
            "cache_file_sha256": cache_file_sha,
            "content_provenance": content_provenance,
            "role": role,
        }
    )
    for key in ("pids", "camids", "paths", "content_sha256"):
        if len(result[key]) != count:  # type: ignore[arg-type]
            raise ValueError("cache key %s has the wrong length" % key)
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in result["content_sha256"]  # type: ignore[index]
    ):
        raise ValueError("content_sha256 entries must be 64 lowercase hex characters")
    return result


def assert_paired_cache(target: Mapping[str, object], other: Mapping[str, object]) -> None:
    """Require extraction caches to differ only in extracted features/provenance."""
    scalar_keys = ("num_query", "block_dim", "weight_sha256")
    sequence_keys = ("pids", "camids", "paths", "content_sha256")
    for key in scalar_keys:
        if target[key] != other[key]:
            raise ValueError("paired cache scalar mismatch: %s" % key)
    for key in sequence_keys:
        if list(target[key]) != list(other[key]):  # type: ignore[arg-type]
            raise ValueError("paired cache metadata mismatch: %s" % key)
    for key in ("target_person_valid", "person_count"):
        if not torch.equal(target[key], other[key]):  # type: ignore[arg-type]
            raise ValueError("paired cache tensor mismatch: %s" % key)


def metadata_audit(cache: Mapping[str, object]) -> Dict[str, object]:
    """Reject unsafe duplication before any metric is computed.

    Occluded-Duke's standard loader contains exact query/gallery copies for a
    subset of queries.  They are legal only when one copy is on each side and
    both PID and camera agree: the standard evaluator removes that gallery
    endpoint, and episode construction independently excludes the same content
    from support donors.  Any query/query, gallery/gallery, larger, or
    inconsistent content group still fails closed.
    """
    paths = list(cache["paths"])  # type: ignore[arg-type]
    contents = list(cache["content_sha256"])  # type: ignore[arg-type]
    pids = list(cache["pids"])  # type: ignore[arg-type]
    camids = list(cache["camids"])  # type: ignore[arg-type]
    num_query = int(cache["num_query"])

    path_count = Counter(paths)
    duplicate_paths = sorted(path for path, count in path_count.items() if count > 1)
    content_indices: Dict[str, List[int]] = defaultdict(list)
    for index, value in enumerate(contents):
        content_indices[str(value)].append(index)
    duplicate_contents = sorted(
        value for value, indices in content_indices.items() if len(indices) > 1
    )
    allowed_query_gallery_same_pidcam: List[str] = []
    forbidden_duplicate_contents: List[str] = []
    for value in duplicate_contents:
        indices = content_indices[value]
        query_indices = [index for index in indices if index < num_query]
        gallery_indices = [index for index in indices if index >= num_query]
        pid_cam = {(int(pids[index]), int(camids[index])) for index in indices}
        if (
            len(indices) == 2
            and len(query_indices) == 1
            and len(gallery_indices) == 1
            and len(pid_cam) == 1
        ):
            allowed_query_gallery_same_pidcam.append(value)
        else:
            forbidden_duplicate_contents.append(value)
    mapping: Dict[str, set] = defaultdict(set)
    for path, pid, camid in zip(paths, pids, camids):
        mapping[path].add((int(pid), int(camid)))
    inconsistent = sorted(path for path, values in mapping.items() if len(values) != 1)
    audit = {
        "sample_count": len(paths),
        "exact_duplicate_path_count": len(duplicate_paths),
        "exact_duplicate_content_group_count": len(duplicate_contents),
        "allowed_query_gallery_same_pidcam_content_count": len(
            allowed_query_gallery_same_pidcam
        ),
        "forbidden_duplicate_content_count": len(forbidden_duplicate_contents),
        "inconsistent_path_mapping_count": len(inconsistent),
        "duplicate_paths": duplicate_paths,
        "allowed_query_gallery_same_pidcam_content_sha256": (
            allowed_query_gallery_same_pidcam
        ),
        "forbidden_duplicate_content_sha256": forbidden_duplicate_contents,
        "inconsistent_paths": inconsistent,
        "near_duplicate_tracklet_answerable": bool(
            "tracklet_ids" in cache and "frame_ids" in cache
        ),
    }
    if duplicate_paths or forbidden_duplicate_contents or inconsistent:
        raise ValueError("metadata duplicate/mapping hard gate failed: %s" % audit)
    return audit


def assign_gallery_folds(cache: Mapping[str, object], seed: int) -> List[int]:
    """Sort SHA256(seed:path) within PID and assign deterministic round-robin folds."""
    num_query = int(cache["num_query"])
    pids = list(cache["pids"])  # type: ignore[arg-type]
    paths = list(cache["paths"])  # type: ignore[arg-type]
    by_pid: Dict[int, List[int]] = defaultdict(list)
    for index in range(num_query, len(pids)):
        by_pid[int(pids[index])].append(index)
    fold_by_index = [-1] * len(pids)
    for pid in sorted(by_pid):
        ordered = sorted(
            by_pid[pid],
            key=lambda index: stable_key(seed, "fold", str(paths[index])),
        )
        for position, index in enumerate(ordered):
            fold_by_index[index] = position % FOLD_COUNT
    if any(value < 0 for value in fold_by_index[num_query:]):
        raise AssertionError("some gallery sample was not assigned to a fold")
    return fold_by_index


def deterministic_derangement(
    path: str,
    *,
    seed: int,
    namespace: str,
    slot_count: int = SLOT_COUNT,
) -> torch.Tensor:
    if slot_count < 2:
        raise ValueError("slot_count must be at least two")
    digest = stable_key(seed, namespace, path)
    shift = 1 + int(digest[:16], 16) % (slot_count - 1)
    result = (torch.arange(slot_count) + shift) % slot_count
    if sorted(result.tolist()) != list(range(slot_count)):
        raise AssertionError("permutation is not bijective")
    if torch.any(result == torch.arange(slot_count)):
        raise AssertionError("permutation is not a derangement")
    return result


def response_permuted(raw: torch.Tensor, donor_paths: Sequence[str], seed: int) -> torch.Tensor:
    output = torch.empty_like(raw)
    for donor_index, path in enumerate(donor_paths):
        order = deterministic_derangement(path, seed=seed, namespace="response")
        output[donor_index] = raw[donor_index, order]
    return output


def common_active_mask(
    raw: torch.Tensor,
    donor_paths: Sequence[str],
    seed: int,
    eps: float = ACTIVE_EPS,
) -> torch.Tensor:
    """Intersection mask makes original and response-permuted denominators legal."""
    permuted = response_permuted(raw, donor_paths, seed)
    return (raw.sum(dim=0) > eps) & (permuted.sum(dim=0) > eps)


def build_fold_episode(
    cache: Mapping[str, object],
    fold_by_index: Sequence[int],
    fold: int,
    *,
    permutation_seed: int,
    camera_protocol: str,
    max_queries: int = 0,
) -> Dict[str, object]:
    num_query = int(cache["num_query"])
    pids = list(cache["pids"])  # type: ignore[arg-type]
    camids = list(cache["camids"])  # type: ignore[arg-type]
    paths = list(cache["paths"])  # type: ignore[arg-type]
    contents = list(cache["content_sha256"])  # type: ignore[arg-type]
    target_valid = cache["target_person_valid"]  # type: ignore[assignment]
    raw = cache["raw_pose_response"]  # type: ignore[assignment]

    reference_indices = [
        index for index in range(num_query, len(pids)) if fold_by_index[index] == fold
    ]
    support_indices = [
        index for index in range(num_query, len(pids)) if fold_by_index[index] != fold
    ]
    if not reference_indices or not support_indices:
        raise ValueError("fold %d has an empty support/reference side" % fold)
    support_paths = {paths[index] for index in support_indices}
    reference_paths = {paths[index] for index in reference_indices}
    support_contents = {contents[index] for index in support_indices}
    reference_contents = {contents[index] for index in reference_indices}
    if support_paths.intersection(reference_paths):
        raise AssertionError("support/reference path overlap")
    if support_contents.intersection(reference_contents):
        raise AssertionError("support/reference content overlap")

    support_by_pid: Dict[int, List[int]] = defaultdict(list)
    for index in support_indices:
        if bool(target_valid[index].item()):
            support_by_pid[int(pids[index])].append(index)

    query_candidates = list(range(num_query))
    if max_queries > 0:
        query_candidates = query_candidates[:max_queries]
    eligible_indices: List[int] = []
    donors: List[List[int]] = []
    wrong_donors: List[List[int]] = []
    active_masks: List[torch.Tensor] = []
    valid_rows: List[torch.Tensor] = []
    positive_rows: List[torch.Tensor] = []
    removal_reasons: Counter = Counter()
    removed_queries: List[Dict[str, object]] = []
    donor_camera_counts: List[int] = []
    available_donor_counts: List[int] = []

    def reject(query_index: int, reason: str) -> None:
        removal_reasons[reason] += 1
        removed_queries.append(
            {
                "query_index": int(query_index),
                "pid": int(pids[query_index]),
                "camid": int(camids[query_index]),
                "path": str(paths[query_index]),
                "reason": reason,
            }
        )

    ref_pids = torch.tensor([int(pids[index]) for index in reference_indices])
    ref_camids = torch.tensor([int(camids[index]) for index in reference_indices])
    for query_index in query_candidates:
        if not bool(target_valid[query_index].item()):
            reject(query_index, "invalid_target_person")
            continue
        pid = int(pids[query_index])
        camid = int(camids[query_index])
        query_path = paths[query_index]
        query_content = contents[query_index]
        candidate_donors = []
        for donor in support_by_pid.get(pid, []):
            if paths[donor] == query_path or contents[donor] == query_content:
                continue
            if camera_protocol == "cross-camera" and int(camids[donor]) == camid:
                continue
            candidate_donors.append(donor)
        available_donor_count = len(candidate_donors)
        if len(candidate_donors) < MIN_DONORS:
            reject(query_index, "fewer_than_three_support_donors")
            continue
        # Match the intended P x K (K=4) student protocol: an anchor can use
        # exactly the other three views. Selection is feature/pose independent
        # and shared by every arm.
        candidate_donors = sorted(
            candidate_donors,
            key=lambda index: stable_key(
                permutation_seed,
                "same-id-donor",
                "%s:%s" % (query_path, paths[index]),
            ),
        )[:SELECTED_DONORS]

        same_pid = ref_pids.eq(pid)
        same_cam = ref_camids.eq(camid)
        valid = ~(same_pid & same_cam)
        positive = valid & same_pid
        negative = valid & ~same_pid
        if not positive.any():
            reject(query_index, "no_valid_positive_reference")
            continue
        if not negative.any():
            reject(query_index, "no_valid_negative_reference")
            continue

        donor_raw = raw[candidate_donors]
        if not torch.isfinite(donor_raw).all():
            reject(query_index, "nonfinite_raw_response")
            continue
        donor_paths = [paths[index] for index in candidate_donors]
        mask = common_active_mask(donor_raw, donor_paths, permutation_seed)

        # Fail-safe corruption only: match donor count while replacing all same-ID
        # evidence with a deterministic mixture of other identities.  This arm never
        # enters the strongest-control max.
        wrong_pool = []
        for donor in support_indices:
            if not bool(target_valid[donor].item()) or int(pids[donor]) == pid:
                continue
            if paths[donor] == query_path or contents[donor] == query_content:
                continue
            if camera_protocol == "cross-camera" and int(camids[donor]) == camid:
                continue
            wrong_pool.append(donor)
        wrong_pool = sorted(
            wrong_pool,
            key=lambda index: stable_key(
                permutation_seed,
                "wrong-id",
                "%s:%s" % (query_path, paths[index]),
            ),
        )
        if len(wrong_pool) < len(candidate_donors):
            raise ValueError(
                "fold %d query %s lacks donor-count-matched WRONG-ID support"
                % (fold, query_path)
            )
        eligible_indices.append(query_index)
        donors.append(candidate_donors)
        wrong_donors.append(wrong_pool[:len(candidate_donors)])
        active_masks.append(mask)
        valid_rows.append(valid)
        positive_rows.append(positive)
        donor_camera_counts.append(len({int(camids[index]) for index in candidate_donors}))
        available_donor_counts.append(available_donor_count)

    if not eligible_indices:
        raise ValueError("fold %d has no eligible query" % fold)
    eligible_pids = {int(pids[index]) for index in eligible_indices}
    original_pids = {int(pids[index]) for index in query_candidates}
    query_ratio = len(eligible_indices) / max(1, len(query_candidates))
    pid_ratio = len(eligible_pids) / max(1, len(original_pids))
    return {
        "fold": fold,
        "query_indices": eligible_indices,
        "reference_indices": reference_indices,
        "support_indices": support_indices,
        "donors": donors,
        "wrong_donors": wrong_donors,
        "active_masks": torch.stack(active_masks),
        "valid": torch.stack(valid_rows),
        "positive": torch.stack(positive_rows),
        "negative": torch.stack(valid_rows) & ~torch.stack(positive_rows),
        "q_pids": torch.tensor([int(pids[index]) for index in eligible_indices]),
        "q_camids": torch.tensor([int(camids[index]) for index in eligible_indices]),
        "q_paths": [paths[index] for index in eligible_indices],
        "r_pids": ref_pids,
        "r_camids": ref_camids,
        "r_paths": [paths[index] for index in reference_indices],
        "eligible_query_count": len(eligible_indices),
        "original_query_count": len(query_candidates),
        "eligible_query_ratio": query_ratio,
        "eligible_pid_count": len(eligible_pids),
        "original_pid_count": len(original_pids),
        "eligible_pid_ratio": pid_ratio,
        "removal_reasons": dict(removal_reasons),
        "removed_queries": removed_queries,
        "donor_count": [len(value) for value in donors],
        "available_donor_count": available_donor_counts,
        "donor_camera_count": donor_camera_counts,
        "active_slot_count": torch.stack(active_masks).sum(dim=1).tolist(),
        "support_reference_path_overlap": 0,
        "support_reference_content_overlap": 0,
        "active_mask_sha256": tensor_sha256(torch.stack(active_masks)),
    }


def _normalize_block(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1 or not torch.isfinite(value).all():
        raise ValueError("descriptor block must be a finite vector")
    if float(value.norm().item()) <= 0:
        raise ValueError("descriptor block has zero norm")
    return F.normalize(value.float(), p=2, dim=0)


def support_descriptor(
    extraction_blocks: torch.Tensor,
    target_raw: torch.Tensor,
    query_index: int,
    donor_indices: Sequence[int],
    active_mask: torch.Tensor,
    arm: str,
    *,
    donor_paths: Sequence[str],
    permutation_seed: int,
    validate_active_mask: bool = True,
) -> torch.Tensor:
    """Build exactly one descriptor for one query/arm/fold."""
    if arm not in MAIN_ARMS:
        raise ValueError("unknown arm: %s" % arm)
    if arm == "WRONG-ID":
        raise ValueError("WRONG-ID must be built through build_arm_descriptors")
    anchor = F.normalize(extraction_blocks[query_index].float(), p=2, dim=1)
    donor_blocks = F.normalize(extraction_blocks[list(donor_indices)].float(), p=2, dim=2)
    raw = target_raw[list(donor_indices)].float()
    if donor_blocks.shape[0] < MIN_DONORS:
        raise ValueError("support descriptor requires at least three donors")
    if raw.shape != (donor_blocks.shape[0], SLOT_COUNT):
        raise ValueError("raw response/donor shape mismatch")
    if validate_active_mask:
        expected_active = common_active_mask(raw, donor_paths, permutation_seed)
        if not torch.equal(active_mask.bool(), expected_active):
            raise AssertionError("arm received a non-common active-slot mask")

    if arm == "SELF":
        return F.normalize(anchor.reshape(-1), p=2, dim=0)
    if arm == "FULL-INCL":
        combined = torch.cat((anchor.unsqueeze(0), donor_blocks), dim=0).mean(dim=0)
        combined = F.normalize(combined, p=2, dim=1)
        return F.normalize(combined.reshape(-1), p=2, dim=0)
    if arm == "FULL-LOO":
        combined = F.normalize(donor_blocks.mean(dim=0), p=2, dim=1)
        return F.normalize(combined.reshape(-1), p=2, dim=0)

    # Every part arm retains query global/pooled and only replaces five slots.
    output = anchor.clone()
    donor_slots = donor_blocks[:, 2:7]
    bag_mean = _normalize_block(donor_slots.reshape(-1, donor_slots.shape[-1]).mean(dim=0))
    response_permutation = response_permuted(raw, donor_paths, permutation_seed)
    for slot in range(SLOT_COUNT):
        if not bool(active_mask[slot].item()):
            continue
        values = donor_slots[:, slot]
        if arm == "ID-GLOBAL":
            value = donor_blocks[:, 0, :].mean(dim=0)
        elif arm == "ID-MEAN":
            value = bag_mean
        elif arm == "PART-EQUAL":
            value = values.mean(dim=0)
        elif arm == "SLOT-PERM":
            permuted_values = []
            for donor_offset, path in enumerate(donor_paths):
                order = deterministic_derangement(
                    path, seed=permutation_seed, namespace="feature"
                )
                permuted_values.append(donor_slots[donor_offset, order[slot]])
            value = torch.stack(permuted_values).mean(dim=0)
        elif arm == "AGREE":
            consensus = _normalize_block(values.mean(dim=0))
            agreement = (values @ consensus).clamp_min(0.0) + 1e-8
            weights = agreement / agreement.sum()
            value = (weights[:, None] * values).sum(dim=0)
        elif arm == "POSE-RESP":
            denominator = raw[:, slot].sum()
            if float(denominator.item()) <= ACTIVE_EPS:
                raise AssertionError("active POSE-RESP slot has zero denominator")
            weights = raw[:, slot] / denominator
            value = (weights[:, None] * values).sum(dim=0)
        elif arm == "RESP-PERM":
            denominator = response_permutation[:, slot].sum()
            if float(denominator.item()) <= ACTIVE_EPS:
                raise AssertionError("active RESP-PERM slot has zero denominator")
            weights = response_permutation[:, slot] / denominator
            value = (weights[:, None] * values).sum(dim=0)
        else:
            raise AssertionError("unhandled part arm: %s" % arm)
        output[2 + slot] = _normalize_block(value)
    return F.normalize(output.reshape(-1), p=2, dim=0)


def build_arm_descriptors(
    extraction_cache: Mapping[str, object],
    target_cache: Mapping[str, object],
    episode: Mapping[str, object],
    arm: str,
    *,
    permutation_seed: int,
) -> torch.Tensor:
    descriptors = []
    paths = list(target_cache["paths"])  # type: ignore[arg-type]
    for local, (query_index, donor_indices) in enumerate(
        zip(episode["query_indices"], episode["donors"])  # type: ignore[arg-type]
    ):
        descriptor_arm = arm
        validate_active_mask = True
        if arm == "WRONG-ID":
            donor_indices = episode["wrong_donors"][local]  # type: ignore[index]
            descriptor_arm = "PART-EQUAL"
            # WRONG-ID is a fail-safe corruption diagnostic.  It deliberately reuses
            # the correct-support common mask so query/slot coverage stays identical.
            validate_active_mask = False
        donor_paths = [paths[index] for index in donor_indices]
        descriptors.append(
            support_descriptor(
                extraction_cache["features"],  # type: ignore[arg-type]
                target_cache["raw_pose_response"],  # type: ignore[arg-type]
                int(query_index),
                donor_indices,
                episode["active_masks"][local],  # type: ignore[index]
                descriptor_arm,
                donor_paths=donor_paths,
                permutation_seed=permutation_seed,
                validate_active_mask=validate_active_mask,
            )
        )
    result = torch.stack(descriptors)
    if result.shape[0] != int(episode["eligible_query_count"]):
        raise AssertionError("descriptor/query count mismatch")
    return result


def pairwise_squared_l2(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("distance batch must be positive")
    q = F.normalize(query.float(), p=2, dim=1)
    r = F.normalize(reference.float(), p=2, dim=1).to(device)
    output = torch.empty((q.shape[0], r.shape[0]), dtype=torch.float32)
    for start in range(0, q.shape[0], batch_size):
        chunk = q[start:start + batch_size].to(device)
        output[start:start + chunk.shape[0]] = (
            2.0 - 2.0 * (chunk @ r.t())
        ).clamp_min(0.0).cpu()
    return output


def average_precision(matches: torch.Tensor) -> float:
    count = int(matches.sum().item())
    if count <= 0:
        raise ValueError("average precision requires a positive reference")
    precision = matches.float().cumsum(0) / torch.arange(
        1, matches.numel() + 1, dtype=torch.float32
    )
    return float((precision * matches.float()).sum().item() / count)


def retrieval_metrics(distances: torch.Tensor, episode: Mapping[str, object]) -> Dict[str, object]:
    valid = episode["valid"]  # type: ignore[assignment]
    positive = episode["positive"]  # type: ignore[assignment]
    q_pids = episode["q_pids"]  # type: ignore[assignment]
    q_paths = episode["q_paths"]  # type: ignore[assignment]
    aps: List[float] = []
    ranks = {1: [], 5: [], 10: []}
    per_query: List[Dict[str, object]] = []
    positive_distances: List[float] = []
    negative_distances: List[float] = []
    for row in range(distances.shape[0]):
        row_valid = valid[row]
        labels = positive[row][row_valid]
        values = distances[row][row_valid]
        order = torch.argsort(values)
        matches = labels[order]
        ap = average_precision(matches)
        first = int(matches.nonzero(as_tuple=False)[0].item())
        aps.append(ap)
        for rank in ranks:
            ranks[rank].append(float(first < rank))
        pos_mean = float(distances[row][positive[row]].mean().item())
        negative_mask = row_valid & ~positive[row]
        neg_mean = float(distances[row][negative_mask].mean().item())
        positive_distances.append(pos_mean)
        negative_distances.append(neg_mean)
        per_query.append(
            {
                "fold": int(episode["fold"]),
                "pid": int(q_pids[row].item()),
                "path": str(q_paths[row]),
                "ap": ap,
                "positive_distance": pos_mean,
                "negative_distance": neg_mean,
                "margin": neg_mean - pos_mean,
            }
        )
    return {
        "mAP": float(np.mean(aps)),
        "rank1": float(np.mean(ranks[1])),
        "rank5": float(np.mean(ranks[5])),
        "rank10": float(np.mean(ranks[10])),
        "positive_distance": float(np.mean(positive_distances)),
        "negative_distance": float(np.mean(negative_distances)),
        "class_balanced_margin": float(
            np.mean(np.asarray(negative_distances) - np.asarray(positive_distances))
        ),
        "evaluated_queries": len(aps),
        "per_query": per_query,
    }


def removed_self_difficulty(
    extraction_cache: Mapping[str, object],
    episode: Mapping[str, object],
    *,
    device: torch.device,
    distance_batch: int,
) -> Dict[str, object]:
    """Score removed queries with SELF whenever the fold has valid references.

    Queries removed because they have no positive/negative reference remain explicitly
    unscorable.  This audit prevents a high support-arm score from silently selecting
    only easy identities.
    """
    removed = list(episode["removed_queries"])  # type: ignore[arg-type]
    if not removed:
        return {
            "removed_query_count": 0,
            "scorable_removed_query_count": 0,
            "unscorable_removed_query_count": 0,
            "unscorable_reasons": {},
            "metrics": None,
        }
    pids = list(extraction_cache["pids"])  # type: ignore[arg-type]
    camids = list(extraction_cache["camids"])  # type: ignore[arg-type]
    paths = list(extraction_cache["paths"])  # type: ignore[arg-type]
    reference_indices = list(episode["reference_indices"])  # type: ignore[arg-type]
    ref_pids = torch.tensor([int(pids[index]) for index in reference_indices])
    ref_camids = torch.tensor([int(camids[index]) for index in reference_indices])
    scorable: List[Mapping[str, object]] = []
    valid_rows: List[torch.Tensor] = []
    positive_rows: List[torch.Tensor] = []
    unscorable_reasons: Counter = Counter()
    for row in removed:
        query_index = int(row["query_index"])
        same_pid = ref_pids.eq(int(pids[query_index]))
        same_cam = ref_camids.eq(int(camids[query_index]))
        valid = ~(same_pid & same_cam)
        positive = valid & same_pid
        negative = valid & ~same_pid
        if not positive.any():
            unscorable_reasons["no_valid_positive_reference"] += 1
            continue
        if not negative.any():
            unscorable_reasons["no_valid_negative_reference"] += 1
            continue
        scorable.append(row)
        valid_rows.append(valid)
        positive_rows.append(positive)
    if not scorable:
        return {
            "removed_query_count": len(removed),
            "scorable_removed_query_count": 0,
            "unscorable_removed_query_count": len(removed),
            "unscorable_reasons": dict(unscorable_reasons),
            "metrics": None,
        }

    query_indices = [int(row["query_index"]) for row in scorable]
    query = extraction_cache["features"][query_indices].reshape(len(query_indices), -1)  # type: ignore[index]
    reference = extraction_cache["features"][reference_indices].reshape(  # type: ignore[index]
        len(reference_indices), -1
    )
    distances = pairwise_squared_l2(
        query,
        reference,
        device=device,
        batch_size=distance_batch,
    )
    mini_episode = {
        "fold": int(episode["fold"]),
        "valid": torch.stack(valid_rows),
        "positive": torch.stack(positive_rows),
        "q_pids": torch.tensor([int(row["pid"]) for row in scorable]),
        "q_paths": [str(row["path"]) for row in scorable],
    }
    metrics = retrieval_metrics(distances, mini_episode)
    metrics.pop("per_query")
    return {
        "removed_query_count": len(removed),
        "scorable_removed_query_count": len(scorable),
        "unscorable_removed_query_count": len(removed) - len(scorable),
        "unscorable_reasons": dict(unscorable_reasons),
        "metrics": metrics,
    }


def run_extraction(
    extraction_cache: Mapping[str, object],
    target_cache: Mapping[str, object],
    fold_by_index: Sequence[int],
    *,
    arms: Sequence[str],
    split_seed: int,
    permutation_seed: int,
    camera_protocol: str,
    max_queries: int,
    device: torch.device,
    distance_batch: int,
) -> Dict[str, object]:
    del split_seed  # fold assignment is supplied and shared across extractions.
    fold_results: List[Dict[str, object]] = []
    per_query_by_arm: Dict[str, List[Dict[str, object]]] = {arm: [] for arm in arms}
    for fold in range(FOLD_COUNT):
        episode = build_fold_episode(
            target_cache,
            fold_by_index,
            fold,
            permutation_seed=permutation_seed,
            camera_protocol=camera_protocol,
            max_queries=max_queries,
        )
        reference = extraction_cache["features"][episode["reference_indices"]].reshape(  # type: ignore[index]
            len(episode["reference_indices"]), -1  # type: ignore[arg-type]
        )
        arm_results: Dict[str, object] = {}
        descriptor_shas: Dict[str, str] = {}
        for arm in arms:
            descriptors = build_arm_descriptors(
                extraction_cache,
                target_cache,
                episode,
                arm,
                permutation_seed=permutation_seed,
            )
            distances = pairwise_squared_l2(
                descriptors,
                reference,
                device=device,
                batch_size=distance_batch,
            )
            metrics = retrieval_metrics(distances, episode)
            per_query_by_arm[arm].extend(metrics.pop("per_query"))
            arm_results[arm] = metrics
            descriptor_shas[arm] = tensor_sha256(descriptors)
        removed_difficulty = removed_self_difficulty(
            extraction_cache,
            episode,
            device=device,
            distance_batch=distance_batch,
        )
        removed_map = (
            float(removed_difficulty["metrics"]["mAP"])
            if removed_difficulty["metrics"] is not None else None
        )
        retained_map = (
            float(arm_results["SELF"]["mAP"])
            if "SELF" in arm_results else None
        )
        fold_results.append(
            {
                "fold": fold,
                "episode": {
                    key: episode[key]
                    for key in (
                        "eligible_query_count",
                        "original_query_count",
                        "eligible_query_ratio",
                        "eligible_pid_count",
                        "original_pid_count",
                        "eligible_pid_ratio",
                        "removal_reasons",
                        "donor_count",
                        "available_donor_count",
                        "donor_camera_count",
                        "active_slot_count",
                        "support_reference_path_overlap",
                        "support_reference_content_overlap",
                        "active_mask_sha256",
                    )
                },
                "descriptor_sha256": descriptor_shas,
                "arms": arm_results,
                "selection_bias_audit": {
                    "retained_self_mAP": retained_map,
                    "removed_self_mAP": removed_map,
                    "retained_minus_removed_self_mAP": (
                        retained_map - removed_map
                        if retained_map is not None and removed_map is not None else None
                    ),
                    **removed_difficulty,
                },
            }
        )

    aggregate: Dict[str, object] = {}
    for arm in arms:
        aggregate[arm] = {}
        for metric in (
            "mAP",
            "rank1",
            "rank5",
            "rank10",
            "positive_distance",
            "negative_distance",
            "class_balanced_margin",
        ):
            values = [float(fold["arms"][arm][metric]) for fold in fold_results]  # type: ignore[index]
            aggregate[arm][metric] = {  # type: ignore[index]
                "equal_fold_mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "std": float(np.std(values)),
                "fold_values": values,
            }
    return {
        "role": extraction_cache["role"],
        "folds": fold_results,
        "aggregate": aggregate,
        "per_query_by_arm": per_query_by_arm,
    }


def paired_query_rows(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    right_by_key = {
        (int(row["fold"]), str(row["path"])): row for row in right
    }
    output = []
    for row in left:
        key = (int(row["fold"]), str(row["path"]))
        if key not in right_by_key:
            raise ValueError("per-query pairing mismatch: %r" % (key,))
        other = right_by_key[key]
        if int(row["pid"]) != int(other["pid"]):
            raise ValueError("per-query PID mismatch")
        output.append(
            {
                "fold": key[0],
                "path": key[1],
                "pid": int(row["pid"]),
                "difference": float(row["ap"]) - float(other["ap"]),
            }
        )
    if len(output) != len(right):
        raise ValueError("per-query pairing is not bijective")
    return output


def pid_grouped_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    replicates: int,
) -> Dict[str, float]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    by_pid: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        by_pid[int(row["pid"])].append(float(row["difference"]))
    pids = sorted(by_pid)
    if len(pids) < 2:
        raise ValueError("PID-grouped bootstrap requires at least two PIDs")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        flattened = [value for pid in sampled for value in by_pid[int(pid)]]
        values[replicate] = float(np.mean(flattened))
    return {
        "point": float(np.mean([float(row["difference"]) for row in rows])),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "replicates": float(replicates),
        "pid_groups": float(len(pids)),
    }


def evaluate_main_gate(
    target_result: Mapping[str, object],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    canonical_result: Optional[Mapping[str, object]] = None,
    scene_result: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    aggregate = target_result["aggregate"]  # type: ignore[assignment]

    def mean_map(arm: str) -> float:
        return float(aggregate[arm]["mAP"]["equal_fold_mean"])

    strongest_control = max(POSE_CONTROLS, key=mean_map)
    pose_rows = target_result["per_query_by_arm"]["POSE-RESP"]  # type: ignore[index]
    pairwise_bootstrap: Dict[str, object] = {}
    for offset, control in enumerate(POSE_CONTROLS):
        control_rows = target_result["per_query_by_arm"][control]  # type: ignore[index]
        pairwise_bootstrap[control] = pid_grouped_bootstrap(
            paired_query_rows(pose_rows, control_rows),
            seed=bootstrap_seed + offset,
            replicates=bootstrap_replicates,
        )

    pose_fold_values = [
        float(value) for value in aggregate["POSE-RESP"]["mAP"]["fold_values"]
    ]
    fold_control_max = []
    fold_control_name = []
    for fold in range(FOLD_COUNT):
        values = {
            control: float(aggregate[control]["mAP"]["fold_values"][fold])
            for control in POSE_CONTROLS
        }
        winner = max(values, key=values.get)
        fold_control_name.append(winner)
        fold_control_max.append(values[winner])
    fold_differences = [
        pose - control for pose, control in zip(pose_fold_values, fold_control_max)
    ]
    coverage = all(
        float(fold["episode"]["eligible_query_ratio"]) >= MIN_ELIGIBLE_RATIO
        and float(fold["episode"]["eligible_pid_ratio"]) >= MIN_ELIGIBLE_RATIO
        for fold in target_result["folds"]  # type: ignore[assignment]
    )
    target_differences = {
        "vs_strongest_control": mean_map("POSE-RESP") - mean_map(strongest_control),
        "part_equal_vs_slot_perm": mean_map("PART-EQUAL") - mean_map("SLOT-PERM"),
        "pose_resp_vs_part_equal": mean_map("POSE-RESP") - mean_map("PART-EQUAL"),
        "pose_resp_vs_resp_perm": mean_map("POSE-RESP") - mean_map("RESP-PERM"),
        "pose_resp_vs_wrong_id": mean_map("POSE-RESP") - mean_map("WRONG-ID"),
    }

    canonical_complete = canonical_result is not None
    canonical_matrix = None
    if canonical_result is not None:
        c_agg = canonical_result["aggregate"]  # type: ignore[assignment]
        canonical_matrix = {
            arm: float(c_agg[arm]["mAP"]["equal_fold_mean"])
            for arm in ROUTING_ARMS
        }

    scene_direction_consistent = scene_result is not None
    scene_difference = None
    scene_fold_differences = None
    scene_fold_control_names = None
    if scene_result is not None:
        s_agg = scene_result["aggregate"]  # type: ignore[assignment]
        scene_pose = float(s_agg["POSE-RESP"]["mAP"]["equal_fold_mean"])
        scene_control = max(
            float(s_agg[arm]["mAP"]["equal_fold_mean"])
            for arm in POSE_CONTROLS
        )
        scene_difference = scene_pose - scene_control
        scene_fold_differences = []
        scene_fold_control_names = []
        for fold in range(FOLD_COUNT):
            values = {
                control: float(s_agg[control]["mAP"]["fold_values"][fold])
                for control in POSE_CONTROLS
            }
            winner = max(values, key=values.get)
            scene_fold_control_names.append(winner)
            scene_fold_differences.append(
                float(s_agg["POSE-RESP"]["mAP"]["fold_values"][fold])
                - values[winner]
            )
        scene_direction_consistent = (
            scene_difference > 0
            and all(value > 0 for value in scene_fold_differences)
        )

    difficulty_reported = all(
        "selection_bias_audit" in fold
        for fold in target_result["folds"]  # type: ignore[assignment]
    )
    bootstrap_all_positive = all(
        float(result["lower_95"]) > 0
        for result in pairwise_bootstrap.values()
    )

    checks = {
        "pose_vs_strongest_at_least_0_5pp": target_differences["vs_strongest_control"] >= 0.005,
        "all_fold_directions_positive": all(value > 0 for value in fold_differences),
        "all_pairwise_pid_bootstrap_lowers_above_zero": bootstrap_all_positive,
        "slot_correspondence_at_least_0_3pp": target_differences["part_equal_vs_slot_perm"] >= 0.003,
        "pose_vs_equal_at_least_0_3pp": target_differences["pose_resp_vs_part_equal"] >= 0.003,
        "pose_vs_response_perm_at_least_0_5pp": target_differences["pose_resp_vs_resp_perm"] >= 0.005,
        "eligible_query_and_pid_each_fold_at_least_70pct": coverage,
        "canonical_extraction_routing_matrix_complete": canonical_complete,
        "scene_merged_direction_not_conflicting": scene_direction_consistent,
        "wrong_id_fail_safe_is_worse_than_pose": target_differences["pose_resp_vs_wrong_id"] > 0,
        "retained_removed_self_difficulty_reported": difficulty_reported,
    }
    return {
        "scope": (
            "frozen pose-routing screen only; FULL-INCL/FULL-LOO are reported "
            "boundaries and full-feature/full-relation novelty gates move to matched student"
        ),
        "strongest_control": strongest_control,
        "target_differences": target_differences,
        "fold_pose_vs_strongest": fold_differences,
        "fold_strongest_control": fold_control_name,
        "fold_strongest_control_mAP": fold_control_max,
        "pairwise_pid_grouped_bootstrap": pairwise_bootstrap,
        "full_geometry_boundary": {
            "FULL-INCL_mAP": mean_map("FULL-INCL"),
            "FULL-LOO_mAP": mean_map("FULL-LOO"),
            "POSE-RESP_minus_FULL-INCL": mean_map("POSE-RESP") - mean_map("FULL-INCL"),
            "POSE-RESP_minus_FULL-LOO": mean_map("POSE-RESP") - mean_map("FULL-LOO"),
            "enters_routing_gate": False,
        },
        "canonical_routing_matrix": canonical_matrix,
        "scene_pose_vs_best_routing_control": scene_difference,
        "scene_fold_pose_vs_best_routing_control": scene_fold_differences,
        "scene_fold_strongest_control": scene_fold_control_names,
        "checks": checks,
        "routing_screen_all_pass": all(checks.values()),
        "all_pass": all(checks.values()),
    }


def run_oracle(
    target_cache: Mapping[str, object],
    *,
    canonical_cache: Optional[Mapping[str, object]] = None,
    scene_cache: Optional[Mapping[str, object]] = None,
    split_seed: int = 371,
    permutation_seed: int = 1371,
    bootstrap_seed: int = 2371,
    bootstrap_replicates: int = 2000,
    camera_protocol: str = "cross-camera",
    max_queries: int = 0,
    device: Optional[torch.device] = None,
    distance_batch: int = 128,
) -> Dict[str, object]:
    if camera_protocol not in ("cross-camera", "unrestricted"):
        raise ValueError("unknown camera protocol")
    if device is None:
        device = torch.device("cpu")
    audit = metadata_audit(target_cache)
    if canonical_cache is not None:
        assert_paired_cache(target_cache, canonical_cache)
    if scene_cache is not None:
        assert_paired_cache(target_cache, scene_cache)
    fold_by_index = assign_gallery_folds(target_cache, split_seed)
    target_result = run_extraction(
        target_cache,
        target_cache,
        fold_by_index,
        arms=MAIN_ARMS,
        split_seed=split_seed,
        permutation_seed=permutation_seed,
        camera_protocol=camera_protocol,
        max_queries=max_queries,
        device=device,
        distance_batch=distance_batch,
    )
    canonical_result = None
    if canonical_cache is not None:
        canonical_result = run_extraction(
            canonical_cache,
            target_cache,
            fold_by_index,
            arms=("SELF",) + ROUTING_ARMS,
            split_seed=split_seed,
            permutation_seed=permutation_seed,
            camera_protocol=camera_protocol,
            max_queries=max_queries,
            device=device,
            distance_batch=distance_batch,
        )
    scene_result = None
    if scene_cache is not None:
        scene_result = run_extraction(
            scene_cache,
            target_cache,
            fold_by_index,
            arms=SCENE_ARMS,
            split_seed=split_seed,
            permutation_seed=permutation_seed,
            camera_protocol=camera_protocol,
            max_queries=max_queries,
            device=device,
            distance_batch=distance_batch,
        )
    gate = evaluate_main_gate(
        target_result,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        canonical_result=canonical_result,
        scene_result=scene_result,
    )
    return {
        "protocol": "classifier-free GT-supported episodic support-geometry oracle",
        "frozen_geometry_only": True,
        "fold_count": FOLD_COUNT,
        "camera_protocol": camera_protocol,
        "split_seed": split_seed,
        "permutation_seed": permutation_seed,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "metadata_audit": audit,
        "fold_assignment_sha256": json_sha256(fold_by_index),
        "target": target_result,
        "canonical": canonical_result,
        "scene": scene_result,
        "gate": gate,
    }


def dry_run_feasibility(
    target_cache: Mapping[str, object],
    *,
    canonical_cache: Optional[Mapping[str, object]],
    scene_cache: Optional[Mapping[str, object]],
    split_seed: int,
    permutation_seed: int,
    camera_protocol: str,
    max_queries: int,
) -> Dict[str, object]:
    """Validate real-cache feasibility without constructing descriptors or metrics."""
    audit = metadata_audit(target_cache)
    if canonical_cache is not None:
        assert_paired_cache(target_cache, canonical_cache)
    if scene_cache is not None:
        assert_paired_cache(target_cache, scene_cache)
    folds = assign_gallery_folds(target_cache, split_seed)
    episodes = []
    for fold in range(FOLD_COUNT):
        episode = build_fold_episode(
            target_cache,
            folds,
            fold,
            permutation_seed=permutation_seed,
            camera_protocol=camera_protocol,
            max_queries=max_queries,
        )
        episodes.append(
            {
                key: episode[key]
                for key in (
                    "fold",
                    "eligible_query_count",
                    "original_query_count",
                    "eligible_query_ratio",
                    "eligible_pid_count",
                    "original_pid_count",
                    "eligible_pid_ratio",
                    "removal_reasons",
                    "donor_count",
                    "available_donor_count",
                    "donor_camera_count",
                    "active_slot_count",
                    "support_reference_path_overlap",
                    "support_reference_content_overlap",
                    "active_mask_sha256",
                )
            }
        )
    return {
        "status": "DRY_RUN_COMPLETE",
        "metrics_computed": False,
        "metadata_audit": audit,
        "paired_canonical_present": canonical_cache is not None,
        "paired_scene_present": scene_cache is not None,
        "fold_assignment_sha256": json_sha256(folds),
        "folds": episodes,
        "coverage_hard_gate": all(
            float(episode["eligible_query_ratio"]) >= MIN_ELIGIBLE_RATIO
            and float(episode["eligible_pid_ratio"]) >= MIN_ELIGIBLE_RATIO
            for episode in episodes
        ),
    }


def main() -> None:
    args = parse_args()
    target_path = Path(args.target_cache).resolve()
    target_content_sidecar = (
        Path(args.target_content_sidecar).resolve() if args.target_content_sidecar else None
    )
    canonical_path = Path(args.canonical_cache).resolve() if args.canonical_cache else None
    canonical_content_sidecar = (
        Path(args.canonical_content_sidecar).resolve()
        if args.canonical_content_sidecar else None
    )
    scene_path = Path(args.scene_cache).resolve() if args.scene_cache else None
    scene_content_sidecar = (
        Path(args.scene_content_sidecar).resolve() if args.scene_content_sidecar else None
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = load_cache(
        target_path,
        role="target",
        expected_block_dim=args.expected_block_dim,
        content_sidecar=target_content_sidecar,
    )
    canonical = (
        load_cache(
            canonical_path,
            role="canonical",
            expected_block_dim=args.expected_block_dim,
            content_sidecar=canonical_content_sidecar,
        )
        if canonical_path is not None else None
    )
    scene = (
        load_cache(
            scene_path,
            role="scene",
            expected_block_dim=args.expected_block_dim,
            content_sidecar=scene_content_sidecar,
        )
        if scene_path is not None else None
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "target_cache": target["cache_path"],
        "target_cache_sha256": target["cache_file_sha256"],
        "target_content_provenance": target["content_provenance"],
        "canonical_cache": canonical["cache_path"] if canonical is not None else None,
        "canonical_cache_sha256": canonical["cache_file_sha256"] if canonical is not None else None,
        "canonical_content_provenance": canonical["content_provenance"] if canonical is not None else None,
        "scene_cache": scene["cache_path"] if scene is not None else None,
        "scene_cache_sha256": scene["cache_file_sha256"] if scene is not None else None,
        "scene_content_provenance": scene["content_provenance"] if scene is not None else None,
        "target_weight_sha256": target["weight_sha256"],
        "parameters": {
            "fold_count": FOLD_COUNT,
            "min_donors": MIN_DONORS,
            "selected_donors": SELECTED_DONORS,
            "min_eligible_ratio": MIN_ELIGIBLE_RATIO,
            "active_eps": ACTIVE_EPS,
            "split_seed": args.split_seed,
            "permutation_seed": args.permutation_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "camera_protocol": args.camera_protocol,
            "execute_frozen_oracle": bool(args.execute_frozen_oracle),
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    if not args.execute_frozen_oracle:
        result = dry_run_feasibility(
            target,
            canonical_cache=canonical,
            scene_cache=scene,
            split_seed=args.split_seed,
            permutation_seed=args.permutation_seed,
            camera_protocol=args.camera_protocol,
            max_queries=args.max_queries,
        )
        atomic_json(output_dir / "dry_run.json", result)
        audit = result["metadata_audit"]
        console_summary = {
            "status": result["status"],
            "metrics_computed": result["metrics_computed"],
            "coverage_hard_gate": result["coverage_hard_gate"],
            "paired_canonical_present": result["paired_canonical_present"],
            "paired_scene_present": result["paired_scene_present"],
            "metadata": {
                "sample_count": audit["sample_count"],
                "allowed_query_gallery_same_pidcam_content_count": audit[
                    "allowed_query_gallery_same_pidcam_content_count"
                ],
                "forbidden_duplicate_content_count": audit[
                    "forbidden_duplicate_content_count"
                ],
            },
            "folds": [
                {
                    "fold": fold["fold"],
                    "eligible_query_ratio": fold["eligible_query_ratio"],
                    "eligible_pid_ratio": fold["eligible_pid_ratio"],
                    "removal_reasons": fold["removal_reasons"],
                    "selected_donor_count_min_max": [
                        min(fold["donor_count"]),
                        max(fold["donor_count"]),
                    ],
                    "available_donor_count_min_max": [
                        min(fold["available_donor_count"]),
                        max(fold["available_donor_count"]),
                    ],
                }
                for fold in result["folds"]
            ],
        }
        print(json.dumps(console_summary, indent=2))
        print("DRY_RUN_COMPLETE", flush=True)
        return

    result = run_oracle(
        target,
        canonical_cache=canonical,
        scene_cache=scene,
        split_seed=args.split_seed,
        permutation_seed=args.permutation_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        camera_protocol=args.camera_protocol,
        max_queries=args.max_queries,
        device=choose_device(args.device),
        distance_batch=args.distance_batch,
    )
    atomic_json(output_dir / "results.json", result)
    print(json.dumps({"gate": result["gate"], "output_dir": str(output_dir)}, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
