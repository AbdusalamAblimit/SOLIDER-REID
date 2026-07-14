"""Frozen pure protocol helpers for the exp374 PSG fuel gate.

This module deliberately contains no training entry point.  It implements the
pre-registered matching, intervention, metric, and bootstrap rules so they can
be statically reviewed before any unit test or formal evaluation is allowed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = "exp374-gate-a-v1"
MAPPING_SEEDS = tuple(range(374001, 374021))
BASELINE_SEEDS = tuple(range(475000, 476000))
BOOTSTRAP_SEED = 374900
K_SEQUENCE = (8, 16, 32, 64, 128, 256)
PRIMARY_CONTROLS = ("shuffle", "bypass")
ANATOMICAL_GROUPS = {
    "head": (0, 1, 2, 3, 4),
    "shoulder": (5, 6),
    "elbow": (7, 8),
    "wrist": (9, 10),
    "hip": (11, 12),
    "knee": (13, 14),
    "ankle": (15, 16),
}


class GateProtocolError(RuntimeError):
    """Fail-closed protocol error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise GateProtocolError(code, message)


def canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    header = canonical_json_bytes({
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    })
    return sha256_bytes(header + value.numpy().tobytes(order="C"))


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def publish_directory(temporary: Path, published: Path) -> None:
    temporary = Path(temporary)
    published = Path(published)
    require(temporary.is_dir(), "E_PUBLISH_TMP_MISSING", str(temporary))
    require(not published.exists(), "E_PUBLISH_EXISTS", str(published))
    for file_path in sorted(path for path in temporary.rglob("*") if path.is_file()):
        with file_path.open("rb") as handle:
            os.fsync(handle.fileno())
    fsync_directory(temporary)
    os.replace(temporary, published)
    fsync_directory(published.parent)


def create_execution_directory(
    output_root: Path,
    premetric_manifest: Mapping[str, object],
    resume: Path | None,
) -> Tuple[Path, str]:
    manifest_payload = dict(premetric_manifest)
    require(
        "output_dir" not in manifest_payload and "results" not in manifest_payload,
        "E_MANIFEST_CIRCULAR",
        "premetric manifest must not contain output path or results",
    )
    execution_sha = sha256_bytes(canonical_json_bytes(manifest_payload))
    expected = Path(output_root).resolve() / f"gate_a_{execution_sha}"
    if resume is None:
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.mkdir(exist_ok=False)
        atomic_write_json(expected / "premetric_manifest.json", manifest_payload)
        atomic_write_bytes(expected / "execution_sha256.txt", (execution_sha + "\n").encode())
        return expected, execution_sha

    execution_dir = Path(resume).resolve()
    require(execution_dir == expected, "E_RESUME_DIR", f"{execution_dir} != {expected}")
    require(execution_dir.is_dir(), "E_RESUME_MISSING", str(execution_dir))
    require(not (execution_dir / "COMPLETE").exists(), "E_RESUME_COMPLETE", str(execution_dir))
    manifest_path = execution_dir / "premetric_manifest.json"
    require(manifest_path.is_file(), "E_RESUME_MANIFEST", str(manifest_path))
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        canonical_json_bytes(frozen) == canonical_json_bytes(manifest_payload),
        "E_RESUME_HASH_DRIFT",
        "premetric manifest changed",
    )
    require(
        sha256_bytes(canonical_json_bytes(frozen)) == execution_sha,
        "E_RESUME_EXECUTION_SHA",
        "execution directory does not match frozen manifest",
    )
    return execution_dir, execution_sha


@dataclass(frozen=True)
class SceneRecord:
    index: int
    split: str
    path: str
    rgb_sha256: str
    pose_path_sha256: str
    pose_content_sha256: str
    pid: int
    camid: int
    person_count: int
    continuous: Tuple[float, ...]
    frame: int
    report: Mapping[str, float]


def _normalized_entropy(channel: torch.Tensor) -> float:
    total = float(channel.sum().item())
    if total <= 0.0:
        return 0.0
    probabilities = channel.double().reshape(-1) / total
    positive = probabilities > 0
    value = -(probabilities[positive] * probabilities[positive].log()).sum()
    denominator = math.log(channel.numel())
    return float(value.item() / denominator)


def summarize_scene(
    scene_heatmap: torch.Tensor,
    scene_scores: torch.Tensor,
) -> Tuple[Tuple[float, ...], int, Dict[str, float]]:
    """Return the frozen 95-D nuisance vector and report-only summaries."""

    require(scene_heatmap.ndim == 3, "E_SCENE_SHAPE", str(tuple(scene_heatmap.shape)))
    require(scene_heatmap.shape[0] == 17, "E_SCENE_CHANNELS", str(scene_heatmap.shape[0]))
    require(scene_scores.shape == (17,), "E_SCORE_SHAPE", str(tuple(scene_scores.shape)))
    heatmap = scene_heatmap.detach().float().cpu()
    scores = scene_scores.detach().float().cpu()
    require(bool(torch.isfinite(heatmap).all()), "E_SCENE_NONFINITE", "heatmap")
    require(bool(torch.isfinite(scores).all()), "E_SCORE_NONFINITE", "scores")
    require(bool((heatmap >= 0).all()), "E_SCENE_NEGATIVE", "heatmap")

    height, width = int(heatmap.shape[1]), int(heatmap.shape[2])
    l1_values: List[float] = []
    peak_values: List[float] = []
    entropy_values: List[float] = []
    support_values: List[float] = []
    valid_values: List[bool] = []
    union = torch.zeros((height, width), dtype=torch.bool)
    for channel in heatmap:
        l1 = float(channel.sum().item())
        peak = float(channel.max().item())
        entropy = _normalized_entropy(channel)
        if peak > 0.0:
            support = channel > (0.10 * peak)
            support_fraction = float(support.float().mean().item())
            union |= support
        else:
            support_fraction = 0.0
        l1_values.append(l1)
        peak_values.append(peak)
        entropy_values.append(entropy)
        support_values.append(support_fraction)
        valid_values.append(l1 > 1e-8 and peak > 1e-6)

    require(bool(union.any()), "E_MATCH_EMPTY_SUPPORT", "scene support union is empty")
    ys, xs = union.nonzero(as_tuple=True)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    border = (
        int(y_min == 0),
        int(y_max == height - 1),
        int(x_min == 0),
        int(x_max == width - 1),
    )
    frame = border[0] * 8 + border[1] * 4 + border[2] * 2 + border[3]
    bbox = (
        (x_min + x_max) / (2.0 * max(width - 1, 1)),
        (y_min + y_max) / (2.0 * max(height - 1, 1)),
        math.log(bbox_width / width),
        math.log(bbox_height / height),
        math.log(bbox_width / bbox_height),
        float(border[0]),
        float(border[1]),
        float(border[2]),
        float(border[3]),
        sum(border) / 4.0,
    )

    continuous = (
        tuple(float(value) for value in scores.tolist())
        + tuple(math.log1p(value) for value in l1_values)
        + tuple(peak_values)
        + tuple(entropy_values)
        + tuple(support_values)
        + bbox
    )
    require(len(continuous) == 95, "E_NUISANCE_DIM", str(len(continuous)))
    require(all(math.isfinite(value) for value in continuous), "E_NUISANCE_NONFINITE", "")
    report = {
        "total_L1": float(sum(l1_values)),
        "mean_confidence": float(scores.mean().item()),
        "visible_joint_count": float(sum(valid_values)),
        "scene_entropy": float(sum(entropy_values) / 17.0),
    }
    return continuous, frame, report


def robust_scale(matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    values = np.asarray(matrix, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 95, "E_NUISANCE_MATRIX", str(values.shape))
    require(bool(np.isfinite(values).all()), "E_NUISANCE_NONFINITE", "matrix")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    constant = mad < 1e-8
    scale = 1.4826 * mad
    scale[constant] = 1.0
    standardized = (values - median) / scale
    standardized[:, constant] = 0.0
    standardized = np.clip(standardized, -5.0, 5.0)
    require(bool(np.isfinite(standardized).all()), "E_SCALE_NONFINITE", "")
    return standardized, {
        "median": median.tolist(),
        "mad": mad.tolist(),
        "constant_dims": np.flatnonzero(constant).astype(int).tolist(),
    }


def base_cost(anchor: np.ndarray, donors: np.ndarray, camera_penalty: np.ndarray,
              frame_penalty: np.ndarray) -> np.ndarray:
    differences = np.minimum(np.abs(donors - anchor[None, :]), 5.0)
    return differences.mean(axis=1) + 0.25 * camera_penalty + 0.25 * frame_penalty


def eligible_pair(anchor: SceneRecord, donor: SceneRecord) -> bool:
    return (
        anchor.split == donor.split
        and anchor.person_count == donor.person_count
        and anchor.pid != donor.pid
        and anchor.path != donor.path
        and anchor.rgb_sha256 != donor.rgb_sha256
        and anchor.pose_path_sha256 != donor.pose_path_sha256
        and anchor.pose_content_sha256 != donor.pose_content_sha256
        and anchor.index != donor.index
    )


def _hopcroft_karp(
    adjacency: Sequence[Sequence[int]],
    left_order: Sequence[int] | None = None,
) -> np.ndarray:
    count = len(adjacency)
    order = list(range(count)) if left_order is None else list(left_order)
    require(sorted(order) == list(range(count)), "E_MATCH_LEFT_ORDER", "")
    pair_left = np.full(count, -1, dtype=np.int64)
    pair_right = np.full(count, -1, dtype=np.int64)
    distance = np.empty(count, dtype=np.int64)

    def bfs() -> bool:
        queue: deque[int] = deque()
        found = False
        for left in order:
            if pair_left[left] < 0:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = -1
        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                partner = int(pair_right[right])
                if partner < 0:
                    found = True
                elif distance[partner] < 0:
                    distance[partner] = distance[left] + 1
                    queue.append(partner)
        return found

    def dfs(left: int) -> bool:
        for right in adjacency[left]:
            partner = int(pair_right[right])
            if partner < 0 or (
                distance[partner] == distance[left] + 1 and dfs(partner)
            ):
                pair_left[left] = right
                pair_right[right] = left
                return True
        distance[left] = -1
        return False

    while bfs():
        for left in order:
            if pair_left[left] < 0:
                dfs(left)
    return pair_left


def randomized_full_matching(
    adjacency: Sequence[Sequence[int]],
    seed: int,
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    left_order = rng.permutation(len(adjacency)).tolist()
    randomized: List[List[int]] = [[] for _ in adjacency]
    for left in left_order:
        randomized[left] = rng.permutation(np.asarray(adjacency[left], dtype=np.int64)).tolist()
    matching = _hopcroft_karp(randomized, left_order)
    require(bool((matching >= 0).all()), "E_BASELINE_PARTIAL", f"seed={seed}")
    return matching


def minimum_cost_full_matching(
    adjacency: Sequence[Sequence[int]],
    edge_costs: Mapping[Tuple[int, int], float],
    tie_break_ranks: Mapping[Tuple[int, int], int] | None = None,
    tie_break_denominator: int | None = None,
) -> np.ndarray:
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    except ImportError as error:
        raise GateProtocolError("E_SCIPY_REQUIRED", str(error)) from error

    count = len(adjacency)
    rows: List[int] = []
    columns: List[int] = []
    costs: List[float] = []
    minimum = min(edge_costs.values())
    shift = max(0.0, 1e-6 - minimum)
    ordered_edges = sorted(edge_costs)
    edge_count = len(ordered_edges)
    require(edge_count == sum(len(values) for values in adjacency),
            "E_MATCH_EDGE", "adjacency/cost edge count")
    if tie_break_ranks is None:
        tie_break_ranks = {edge: rank for rank, edge in enumerate(ordered_edges)}
    require(set(tie_break_ranks) == set(ordered_edges),
            "E_MATCH_TIE_RANK", "edge set mismatch")
    ranks = sorted(int(value) for value in tie_break_ranks.values())
    require(len(set(ranks)) == edge_count and ranks[0] >= 0,
            "E_MATCH_TIE_RANK", "ranks not unique/nonnegative")
    denominator = int(tie_break_denominator) if tie_break_denominator is not None else edge_count
    require(denominator >= max(ranks) + 1, "E_MATCH_TIE_RANK", str(denominator))
    for left, right in ordered_edges:
        require(right in adjacency[left], "E_MATCH_EDGE", f"{left}->{right}")
        value = float(edge_costs[(left, right)])
        value += shift + 1e-12 * int(tie_break_ranks[(left, right)]) / (denominator + 1)
        rows.append(left)
        columns.append(right)
        costs.append(value)
    graph = coo_matrix((costs, (rows, columns)), shape=(count, count)).tocsr()
    row_ind, col_ind = min_weight_full_bipartite_matching(graph)
    require(len(row_ind) == count, "E_MATCH_PARTIAL", f"{len(row_ind)} != {count}")
    matching = np.full(count, -1, dtype=np.int64)
    matching[row_ind] = col_ind
    require(bool((matching >= 0).all()), "E_MATCH_PARTIAL", "unmatched left node")
    require(len(set(matching.tolist())) == count, "E_MATCH_NOT_BIJECTIVE", "duplicate donor")
    for left, right in enumerate(matching.tolist()):
        require(right in adjacency[left], "E_MATCH_EDGE", f"{left}->{right}")
    return matching


def pairwise_hamming(mappings: Sequence[np.ndarray]) -> float:
    minimum = 1.0
    for first in range(len(mappings)):
        for second in range(first + 1, len(mappings)):
            value = float(np.mean(mappings[first] != mappings[second]))
            minimum = min(minimum, value)
    return minimum


def _identifier_codes(values: Sequence[object]) -> torch.Tensor:
    lookup: Dict[object, int] = {}
    codes: List[int] = []
    for value in values:
        if value not in lookup:
            lookup[value] = len(lookup)
        codes.append(lookup[value])
    return torch.tensor(codes, dtype=torch.int64)


def exact_sparse_candidates(
    records: Sequence[SceneRecord],
    standardized: np.ndarray,
    device: torch.device,
    anchor_chunk: int = 16,
) -> Tuple[List[List[int]], Dict[Tuple[int, int], float]]:
    """Compute exact float64 top-256 C_base edges without storing N x N.

    The temporary dense object is only ``anchor_chunk x N``; the forbidden
    full cost matrix is never materialized or persisted.
    """

    count = len(records)
    require(count > 1, "E_MATCH_STRATUM_SIZE", str(count))
    require(standardized.shape == (count, 95), "E_NUISANCE_MATRIX", str(standardized.shape))
    require(anchor_chunk > 0, "E_MATCH_CHUNK", str(anchor_chunk))
    features = torch.from_numpy(standardized.astype(np.float64, copy=False)).to(device)
    pids = _identifier_codes([record.pid for record in records]).to(device)
    paths = _identifier_codes([record.path for record in records]).to(device)
    rgb = _identifier_codes([record.rgb_sha256 for record in records]).to(device)
    pose_paths = _identifier_codes([record.pose_path_sha256 for record in records]).to(device)
    pose_content = _identifier_codes([record.pose_content_sha256 for record in records]).to(device)
    cameras = _identifier_codes([record.camid for record in records]).to(device)
    frames = torch.tensor([record.frame for record in records], dtype=torch.int64, device=device)
    path_rank = {
        index: rank for rank, index in enumerate(sorted(range(count), key=lambda i: records[i].path))
    }
    lex_rank = torch.tensor([path_rank[index] for index in range(count)],
                            dtype=torch.float64, device=device)
    candidates: List[List[int]] = [[] for _ in records]
    edge_costs: Dict[Tuple[int, int], float] = {}
    keep_count = min(256, count)
    for start in range(0, count, anchor_chunk):
        stop = min(count, start + anchor_chunk)
        anchor = features[start:stop]
        differences = torch.abs(anchor[:, None, :] - features[None, :, :]).clamp_max_(5.0)
        costs = differences.mean(dim=2)
        costs += 0.25 * (cameras[start:stop, None] != cameras[None, :]).to(torch.float64)
        costs += 0.25 * (frames[start:stop, None] != frames[None, :]).to(torch.float64)
        eligible = (
            (pids[start:stop, None] != pids[None, :])
            & (paths[start:stop, None] != paths[None, :])
            & (rgb[start:stop, None] != rgb[None, :])
            & (pose_paths[start:stop, None] != pose_paths[None, :])
            & (pose_content[start:stop, None] != pose_content[None, :])
        )
        row_ids = torch.arange(start, stop, device=device)[:, None]
        column_ids = torch.arange(count, device=device)[None, :]
        eligible &= row_ids != column_ids
        costs[~eligible] = torch.inf
        costs += 1e-12 * lex_rank[None, :] / (count + 1)
        values, indices = torch.topk(costs, k=keep_count, dim=1, largest=False, sorted=True)
        values_cpu = values.cpu().numpy()
        indices_cpu = indices.cpu().numpy()
        for offset, left in enumerate(range(start, stop)):
            finite = np.isfinite(values_cpu[offset])
            donors = indices_cpu[offset][finite].astype(int).tolist()
            require(donors, "E_MATCH_NO_ELIGIBLE", records[left].path)
            donors.sort(key=lambda right: (
                float(values_cpu[offset][np.where(indices_cpu[offset] == right)[0][0]]),
                records[right].path,
            ))
            candidates[left] = donors
            for right in donors:
                donor_position = int(np.where(indices_cpu[offset] == right)[0][0])
                # Remove only the deterministic lexicographic epsilon.  The
                # Float64 d_cont plus explicit soft penalties is C_base.
                value = float(values_cpu[offset][donor_position])
                value -= 1e-12 * path_rank[right] / (count + 1)
                edge_costs[(left, right)] = value
        del differences, costs, eligible, values, indices
    require(all(candidates), "E_MATCH_NO_ELIGIBLE", "empty adjacency")
    return candidates, edge_costs


def _selected_k_adjacency(
    candidates: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], int]:
    count = len(candidates)
    sequence = sorted(set(min(count - 1, value) for value in K_SEQUENCE))
    for k in sequence:
        adjacency = [list(values[:min(k, len(values))]) for values in candidates]
        matching = _hopcroft_karp(adjacency)
        if bool((matching >= 0).all()):
            return adjacency, k
    raise GateProtocolError("E_MATCH_HALL", f"no full sparse matching through k={sequence[-1]}")


def prepare_split_mappings(
    records: Sequence[SceneRecord],
    device: torch.device,
    anchor_chunk: int = 16,
) -> Dict[str, object]:
    """Freeze candidates, 20 maps, and pair-quality audits for one split."""

    require(len({record.split for record in records}) == 1, "E_MATCH_SPLIT", "mixed split")
    continuous = np.asarray([record.continuous for record in records], dtype=np.float64)
    standardized, scaler = robust_scale(continuous)
    strata: Dict[int, List[int]] = {}
    for index, record in enumerate(records):
        strata.setdefault(record.person_count, []).append(index)

    selected_global: List[List[int]] = [[] for _ in records]
    base_global: Dict[Tuple[int, int], float] = {}
    stratum_payload: Dict[str, object] = {}
    for person_count, global_indices_unsorted in sorted(strata.items()):
        global_indices = sorted(global_indices_unsorted, key=lambda index: records[index].path)
        local_records = [records[index] for index in global_indices]
        local_z = standardized[global_indices]
        candidates, base_edges = exact_sparse_candidates(
            local_records, local_z, device=device, anchor_chunk=anchor_chunk)
        selected, selected_k = _selected_k_adjacency(candidates)
        for local_left, donors in enumerate(selected):
            global_left = global_indices[local_left]
            selected_global[global_left] = [global_indices[local_right] for local_right in donors]
            for local_right in donors:
                base_global[(global_left, global_indices[local_right])] = base_edges[
                    (local_left, local_right)]
        stratum_payload[str(person_count)] = {
            "count": len(global_indices),
            "selected_k": selected_k,
            "indices": global_indices,
        }

    require(all(selected_global), "E_MATCH_NO_ELIGIBLE", "selected global graph")
    baseline_costs: List[float] = []
    for seed in BASELINE_SEEDS:
        mapping = randomized_full_matching(selected_global, seed)
        baseline_costs.append(float(np.mean([
            base_global[(left, int(right))] for left, right in enumerate(mapping)
        ])))

    mappings: List[np.ndarray] = []
    randomized_edge_costs: List[np.ndarray] = []
    mapping_audits: List[Dict[str, object]] = []
    eta_by_seed: List[Dict[str, float]] = []
    edge_order_global = sorted(
        base_global,
        key=lambda edge: (records[edge[0]].path, records[edge[1]].path),
    )
    global_edge_rank = {edge: rank for rank, edge in enumerate(edge_order_global)}
    eta_by_stratum: Dict[str, float] = {}
    for person_count, payload in stratum_payload.items():
        index_set = set(payload["indices"])
        values = np.asarray([
            value for (left, _right), value in base_global.items() if left in index_set
        ], dtype=np.float64)
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        eta_by_stratum[person_count] = 0.25 * iqr if iqr >= 1e-8 else 0.01
    for seed in MAPPING_SEEDS:
        mapping = np.full(len(records), -1, dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        noises = rng.gumbel(size=len(edge_order_global))
        randomized_global: Dict[Tuple[int, int], float] = {}
        for edge_index, edge in enumerate(edge_order_global):
            person_count = str(records[edge[0]].person_count)
            randomized_global[edge] = (
                base_global[edge] + eta_by_stratum[person_count] * noises[edge_index]
            )
        randomized_edge_costs.append(np.asarray([
            randomized_global[edge] for edge in sorted(base_global)
        ], dtype=np.float64))
        for person_count, payload in stratum_payload.items():
            indices = list(payload["indices"])
            local_lookup = {global_index: local for local, global_index in enumerate(indices)}
            adjacency = [
                [local_lookup[donor] for donor in selected_global[global_index]]
                for global_index in indices
            ]
            local_cost = {
                (local_left, local_lookup[global_right]): randomized_global[(global_left, global_right)]
                for local_left, global_left in enumerate(indices)
                for global_right in selected_global[global_left]
            }
            local_tie_ranks = {
                (local_left, local_lookup[global_right]): global_edge_rank[(global_left, global_right)]
                for local_left, global_left in enumerate(indices)
                for global_right in selected_global[global_left]
            }
            local_mapping = minimum_cost_full_matching(
                adjacency,
                local_cost,
                tie_break_ranks=local_tie_ranks,
                tie_break_denominator=len(edge_order_global),
            )
            for local_left, local_right in enumerate(local_mapping.tolist()):
                mapping[indices[local_left]] = indices[local_right]
        require(bool((mapping >= 0).all()), "E_MATCH_PARTIAL", f"seed={seed}")
        audit = audit_mapping(records, standardized, mapping, base_global, baseline_costs)
        audit["randomized_objective_float64"] = float(np.sum(np.asarray([
            randomized_global[(left, int(right))]
            for left, right in enumerate(mapping)
        ], dtype=np.float64), dtype=np.float64))
        mappings.append(mapping)
        mapping_audits.append(audit)
        eta_by_seed.append(dict(eta_by_stratum))

    minimum_hamming = pairwise_hamming(mappings)
    effective: List[np.ndarray] = []
    for mapping in mappings:
        if all(float(np.mean(mapping != selected)) >= 0.90 for selected in effective):
            effective.append(mapping)
    effective_unique_count = len(effective)
    require(minimum_hamming >= 0.90, "E_MAPPING_HAMMING",
            f"minimum={minimum_hamming}, effective_unique={effective_unique_count}")
    return {
        "standardized": standardized,
        "scaler": scaler,
        "selected_adjacency": selected_global,
        "base_edges": base_global,
        "randomized_edge_costs": randomized_edge_costs,
        "baseline_mean_costs": baseline_costs,
        "mappings": mappings,
        "mapping_audits": mapping_audits,
        "minimum_hamming": minimum_hamming,
        "effective_unique_count": effective_unique_count,
        "eta_by_seed": eta_by_seed,
        "strata": stratum_payload,
        "solver": {
            "name": "scipy.sparse.csgraph.min_weight_full_bipartite_matching",
            "scipy_version": importlib.metadata.version("scipy"),
            "tie_break": "1e-12*lexicographic_edge_rank/(E+1)",
        },
    }


def audit_mapping(
    records: Sequence[SceneRecord],
    standardized: np.ndarray,
    mapping: np.ndarray,
    base_edges: Mapping[Tuple[int, int], float],
    baseline_mean_costs: Sequence[float],
) -> Dict[str, object]:
    count = len(records)
    require(mapping.shape == (count,), "E_MAPPING_SHAPE", str(mapping.shape))
    require(sorted(mapping.tolist()) == list(range(count)), "E_MATCH_NOT_BIJECTIVE", "")
    donor_matrix = standardized[mapping]
    marginal_mean_error = float(np.max(np.abs(
        standardized.mean(axis=0) - donor_matrix.mean(axis=0))))
    require(marginal_mean_error <= 1e-10, "E_MAPPING_MARGINAL", str(marginal_mean_error))
    sorted_anchor = np.sort(standardized, axis=0)
    sorted_donor = np.sort(donor_matrix, axis=0)
    require(bool(np.array_equal(sorted_anchor, sorted_donor)),
            "E_MAPPING_MARGINAL", "nonzero empirical KS under permutation")
    costs: List[float] = []
    paired = np.empty_like(standardized)
    for left, right in enumerate(mapping.tolist()):
        require(eligible_pair(records[left], records[right]), "E_MATCH_HARD", f"{left}->{right}")
        require((left, right) in base_edges, "E_MATCH_EDGE", f"{left}->{right}")
        costs.append(float(base_edges[(left, right)]))
        paired[left] = np.abs(standardized[left] - standardized[right])
    median_by_dimension = np.median(paired, axis=0)
    require(
        bool((median_by_dimension <= 0.50 + 1e-12).all()),
        "E_PAIR_DIM",
        f"max={median_by_dimension.max()}",
    )
    p95 = float(np.quantile(np.asarray(costs), 0.95, method="higher"))
    require(p95 <= 1.25 + 1e-12, "E_PAIR_P95", str(p95))
    mean_cost = float(np.mean(costs))
    baseline_median = float(np.median(np.asarray(baseline_mean_costs)))
    require(
        mean_cost <= 0.75 * baseline_median + 1e-12,
        "E_PAIR_BASELINE",
        f"{mean_cost} > 0.75*{baseline_median}",
    )
    return {
        "mean_cost": mean_cost,
        "p95_cost": p95,
        "max_dimension_median_abs_z": float(median_by_dimension.max()),
        "baseline_median_mean_cost": baseline_median,
        "max_marginal_mean_error": marginal_mean_error,
        "max_empirical_ks": 0.0,
    }


def actual_psg_input(scene_heatmaps: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    heatmaps = scene_heatmaps.to(dtype=torch.float32)
    resized = F.interpolate(heatmaps, size=size, mode="bilinear", align_corners=False)
    return torch.sigmoid(resized)


def intervention_strength(
    correct_actual: torch.Tensor,
    donor_actual: torch.Tensor,
) -> Dict[str, np.ndarray]:
    require(correct_actual.shape == donor_actual.shape, "E_STRENGTH_SHAPE", "")
    require(correct_actual.ndim == 4 and correct_actual.shape[1] == 17,
            "E_STRENGTH_SHAPE", str(correct_actual.shape))
    correct = correct_actual.float()
    donor = donor_actual.float()
    numerator = (correct - donor).abs().flatten(1).sum(1)
    denominator = 0.5 * (
        (correct - 0.5).abs().flatten(1).sum(1)
        + (donor - 0.5).abs().flatten(1).sum(1)
    ) + 1e-12
    relative = numerator / denominator

    response_correct = correct - 0.5
    response_donor = donor - 0.5
    batch, _, height, width = response_correct.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=correct.device, dtype=torch.float32),
        torch.arange(width, device=correct.device, dtype=torch.float32),
        indexing="ij",
    )
    diagonal = math.sqrt(height * height + width * width)
    displacement = torch.zeros((batch, 17), device=correct.device)
    for source, target, output in ((response_correct, response_donor, displacement),):
        source_mass = source.flatten(2).sum(2)
        target_mass = target.flatten(2).sum(2)
        source_peak = source.flatten(2).amax(2)
        target_peak = target.flatten(2).amax(2)
        source_valid = (source_mass > 1e-8) & (source_peak > 1e-6)
        target_valid = (target_mass > 1e-8) & (target_peak > 1e-6)
        both = source_valid & target_valid
        one = source_valid ^ target_valid
        source_x = (source * xx).flatten(2).sum(2) / source_mass.clamp_min(1e-12)
        source_y = (source * yy).flatten(2).sum(2) / source_mass.clamp_min(1e-12)
        target_x = (target * xx).flatten(2).sum(2) / target_mass.clamp_min(1e-12)
        target_y = (target * yy).flatten(2).sum(2) / target_mass.clamp_min(1e-12)
        distance = torch.sqrt((source_x - target_x) ** 2 + (source_y - target_y) ** 2)
        output[both] = distance[both] / diagonal
        output[one] = 1.0
    return {
        "relative_l1": relative.detach().cpu().numpy(),
        "centroid_displacement": displacement.mean(1).detach().cpu().numpy(),
    }


def audit_intervention_strength(
    correct_scene: torch.Tensor,
    donor_scene: torch.Tensor,
    actual_correct: torch.Tensor,
    actual_donor: torch.Tensor,
    size: Tuple[int, int],
) -> Dict[str, float]:
    expected_correct = actual_psg_input(correct_scene, size)
    expected_donor = actual_psg_input(donor_scene, size)
    require(torch.equal(actual_correct, expected_correct), "E_HOOK_CORRECT_DRIFT", "")
    require(torch.equal(actual_donor, expected_donor), "E_HOOK_DONOR_DRIFT", "")
    require(actual_correct.dtype == torch.float32 and actual_donor.dtype == torch.float32,
            "E_HOOK_DTYPE", "actual PSG inputs must be float32")
    correct_hashes = [sha256_tensor(value) for value in actual_correct]
    donor_hashes = [sha256_tensor(value) for value in actual_donor]
    require(all(a != b for a, b in zip(correct_hashes, donor_hashes)),
            "E_WEAK_IDENTICAL", "identical sample tensor")
    values = intervention_strength(actual_correct, actual_donor)
    relative = values["relative_l1"]
    displacement = values["centroid_displacement"]
    median_relative = float(np.median(relative))
    p10_relative = float(np.quantile(relative, 0.10, method="higher"))
    median_displacement = float(np.median(displacement))
    require(median_relative >= 0.10, "E_WEAK_MEDIAN_L1", str(median_relative))
    require(p10_relative >= 0.01, "E_WEAK_P10_L1", str(p10_relative))
    require(median_displacement >= 0.03, "E_WEAK_CENTROID", str(median_displacement))
    return {
        "median_relative_l1": median_relative,
        "p10_relative_l1": p10_relative,
        "median_centroid_displacement": median_displacement,
    }


def half_away_from_zero(value: float) -> int:
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def translate_zero_padded(channel: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    require(channel.ndim == 2, "E_TRANSLATE_SHAPE", str(channel.shape))
    height, width = channel.shape
    output = torch.zeros_like(channel)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    target_x0 = max(0, dx)
    target_x1 = target_x0 + max(0, source_x1 - source_x0)
    target_y0 = max(0, dy)
    target_y1 = target_y0 + max(0, source_y1 - source_y0)
    if source_x1 > source_x0 and source_y1 > source_y0:
        output[target_y0:target_y1, target_x0:target_x1] = channel[
            source_y0:source_y1, source_x0:source_x1]
    return output


def _channel_centroid(channel: torch.Tensor) -> Tuple[float, float]:
    mass = float(channel.sum().item())
    require(mass > 0.0, "E_CENTROID_ZERO", "")
    height, width = channel.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    weights = channel.double() / mass
    return float((weights * xx).sum()), float((weights * yy).sum())


def _scene_support_bbox(scene: torch.Tensor) -> Tuple[int, int, int, int]:
    union = torch.zeros(scene.shape[1:], dtype=torch.bool, device=scene.device)
    valid_count = 0
    for channel in scene:
        l1 = float(channel.sum().item())
        peak = float(channel.max().item())
        if l1 > 1e-8 and peak > 1e-6:
            union |= channel > (0.10 * peak)
            valid_count += 1
        elif l1 > 1e-8 or peak > 1e-6:
            raise GateProtocolError("E_CENTROID_WEAK_CHANNEL", "scene bbox")
    require(valid_count > 0 and bool(union.any()), "E_CENTROID_EMPTY", "")
    ys, xs = union.nonzero(as_tuple=True)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fit_normalized_centroid_targets(
    train_scenes: Iterable[torch.Tensor],
) -> Tuple[Tuple[float, float] | None, ...]:
    observations: List[List[Tuple[float, float]]] = [[] for _ in range(17)]
    for scene in train_scenes:
        require(scene.shape[0] == 17, "E_CENTROID_SHAPE", str(scene.shape))
        if float(scene.sum().item()) <= 1e-8:
            continue
        x_min, y_min, x_max, y_max = _scene_support_bbox(scene)
        width = max(x_max - x_min, 1)
        height = max(y_max - y_min, 1)
        for joint, channel in enumerate(scene):
            l1 = float(channel.sum().item())
            peak = float(channel.max().item())
            if l1 <= 1e-8 and peak <= 1e-6:
                continue
            require(l1 > 1e-8 and peak > 1e-6, "E_CENTROID_WEAK_CHANNEL", str(joint))
            x, y = _channel_centroid(channel)
            observations[joint].append(((x - x_min) / width, (y - y_min) / height))
    targets: List[Tuple[float, float] | None] = []
    for joint_observations in observations:
        if not joint_observations:
            targets.append(None)
            continue
        array = np.asarray(joint_observations, dtype=np.float64)
        targets.append((float(np.median(array[:, 0])), float(np.median(array[:, 1]))))
    return tuple(targets)


def absolute_centroid_targets(
    scene: torch.Tensor,
    normalized_targets: Sequence[Tuple[float, float] | None],
) -> Tuple[Tuple[float, float] | None, ...]:
    require(len(normalized_targets) == 17, "E_CENTROID_TARGET", "target count")
    if float(scene.sum().item()) <= 1e-8:
        return tuple([None] * 17)
    x_min, y_min, x_max, y_max = _scene_support_bbox(scene)
    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)
    return tuple(
        None if target is None else (
            x_min + target[0] * width,
            y_min + target[1] * height,
        )
        for target in normalized_targets
    )


def apply_scene_centroid_control(
    scene: torch.Tensor,
    targets: Sequence[Tuple[float, float] | None],
) -> torch.Tensor:
    require(scene.shape[0] == 17 and len(targets) == 17, "E_CENTROID_SHAPE", "")
    output = torch.zeros_like(scene)
    for joint, channel in enumerate(scene):
        l1 = float(channel.sum().item())
        peak = float(channel.max().item())
        if l1 <= 1e-8 and peak <= 1e-6:
            output[joint] = channel
            continue
        require(l1 > 1e-8 and peak > 1e-6, "E_CENTROID_WEAK_CHANNEL", str(joint))
        target = targets[joint]
        require(target is not None, "E_CENTROID_TARGET", str(joint))
        source_x, source_y = _channel_centroid(channel)
        dx = half_away_from_zero(target[0] - source_x)
        dy = half_away_from_zero(target[1] - source_y)
        translated = translate_zero_padded(channel, dx, dy)
        translated_x, translated_y = _channel_centroid(translated)
        error = math.hypot(translated_x - target[0], translated_y - target[1])
        require(error <= 0.75, "E_CENTROID_ERROR", f"joint={joint}, error={error}")
        new_l1 = float(translated.sum().item())
        new_peak = float(translated.max().item())
        require(0.95 <= new_l1 / l1 <= 1.05, "E_CENTROID_L1", str(joint))
        require(0.95 <= new_peak / peak <= 1.05, "E_CENTROID_PEAK", str(joint))
        require(
            abs(_normalized_entropy(translated) - _normalized_entropy(channel)) <= 0.01,
            "E_CENTROID_ENTROPY",
            str(joint),
        )
        output[joint] = translated
    require(bool(torch.isfinite(output).all()), "E_CENTROID_NONFINITE", "")
    return output


def replace_group_channels(
    correct: torch.Tensor,
    donor: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    require(correct.shape == donor.shape, "E_GROUP_SHAPE", "")
    require(group_name in ANATOMICAL_GROUPS, "E_GROUP_NAME", group_name)
    output = correct.clone()
    channels = list(ANATOMICAL_GROUPS[group_name])
    output[:, channels] = donor[:, channels]
    return output


def per_query_metrics(
    distmat: np.ndarray,
    q_pids: Sequence[int],
    g_pids: Sequence[int],
    q_camids: Sequence[int],
    g_camids: Sequence[int],
) -> Dict[str, np.ndarray | float]:
    distances = np.asarray(distmat, dtype=np.float64)
    q_pid = np.asarray(q_pids)
    g_pid = np.asarray(g_pids)
    q_cam = np.asarray(q_camids)
    g_cam = np.asarray(g_camids)
    require(distances.shape == (len(q_pid), len(g_pid)), "E_DIST_SHAPE", str(distances.shape))
    require(bool(np.isfinite(distances).all()), "E_DIST_NONFINITE", "")
    order = np.argsort(distances, axis=1)
    ap_values: List[float] = []
    r1_values: List[float] = []
    r5_values: List[float] = []
    r10_values: List[float] = []
    margins: List[float] = []
    for index in range(len(q_pid)):
        ranked = order[index]
        junk = (g_pid[ranked] == q_pid[index]) & (g_cam[ranked] == q_cam[index])
        valid = ranked[~junk]
        positive = g_pid[valid] == q_pid[index]
        require(bool(positive.any()), "E_QUERY_NO_POSITIVE", str(index))
        require(bool((~positive).any()), "E_QUERY_NO_NEGATIVE", str(index))
        cumulative = positive.cumsum()
        precision = cumulative / np.arange(1, len(positive) + 1)
        ap_values.append(float((precision * positive).sum() / positive.sum()))
        r1_values.append(float(positive[:1].any()))
        r5_values.append(float(positive[:5].any()))
        r10_values.append(float(positive[:10].any()))
        valid_distances = distances[index, valid]
        nearest_positive = float(valid_distances[positive].min())
        nearest_negative = float(valid_distances[~positive].min())
        margins.append(nearest_negative - nearest_positive)
    ap = np.asarray(ap_values, dtype=np.float64)
    r1 = np.asarray(r1_values, dtype=np.float64)
    return {
        "AP": ap,
        "R1_indicator": r1,
        "margin": np.asarray(margins, dtype=np.float64),
        "mAP": float(ap.mean()),
        "R1": float(r1.mean()),
        "R5": float(np.mean(r5_values)),
        "R10": float(np.mean(r10_values)),
    }


def aggregate_mapping_queries(values: np.ndarray) -> Dict[str, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    require(matrix.ndim == 2 and matrix.shape[0] == 20, "E_MAPPING_AGG_SHAPE", str(matrix.shape))
    require(bool(np.isfinite(matrix).all()), "E_MAPPING_AGG_NONFINITE", "")
    return {
        "mean": matrix.mean(axis=0),
        "mcse": matrix.std(axis=0, ddof=1) / math.sqrt(20.0),
    }


def _weighted_cluster_mean(values: np.ndarray, pids: np.ndarray,
                           multiplicities: Mapping[int, int]) -> float:
    weights = np.asarray([multiplicities.get(int(pid), 0) for pid in pids], dtype=np.float64)
    require(float(weights.sum()) > 0, "E_BOOTSTRAP_EMPTY", "")
    return float(np.sum(values * weights) / np.sum(weights))


def simultaneous_intervals(
    correct_by_seed: Mapping[int, np.ndarray],
    controls_by_seed: Mapping[str, Mapping[int, np.ndarray]],
    query_pids: Sequence[int],
    replicates: int = 10_000,
) -> Dict[str, object]:
    require(tuple(sorted(controls_by_seed)) == tuple(sorted(PRIMARY_CONTROLS)),
            "E_BOOTSTRAP_CONTROLS", str(sorted(controls_by_seed)))
    seeds = tuple(sorted(correct_by_seed))
    require(len(seeds) == 3, "E_BOOTSTRAP_SEEDS", str(seeds))
    pids = np.asarray(query_pids, dtype=np.int64)
    unique_pids = np.unique(pids)
    point: Dict[str, float] = {}
    per_seed: Dict[str, Dict[int, float]] = {control: {} for control in PRIMARY_CONTROLS}
    for control in PRIMARY_CONTROLS:
        for seed in seeds:
            correct = np.asarray(correct_by_seed[seed], dtype=np.float64)
            target = np.asarray(controls_by_seed[control][seed], dtype=np.float64)
            require(correct.shape == pids.shape == target.shape,
                    "E_BOOTSTRAP_SHAPE", f"{control}/{seed}")
            per_seed[control][seed] = 100.0 * float(np.mean(correct - target))
        point[control] = float(np.mean(list(per_seed[control].values())))

    rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED))
    bootstrap = {control: np.empty(replicates, dtype=np.float64) for control in PRIMARY_CONTROLS}
    for replicate in range(replicates):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        values, counts = np.unique(sampled, return_counts=True)
        multiplicities = {int(pid): int(count) for pid, count in zip(values, counts)}
        for control in PRIMARY_CONTROLS:
            seed_values = []
            for seed in seeds:
                difference = 100.0 * (
                    np.asarray(correct_by_seed[seed], dtype=np.float64)
                    - np.asarray(controls_by_seed[control][seed], dtype=np.float64)
                )
                seed_values.append(_weighted_cluster_mean(difference, pids, multiplicities))
            bootstrap[control][replicate] = float(np.mean(seed_values))

    lower_deviation = np.max(np.stack([
        point[control] - bootstrap[control] for control in PRIMARY_CONTROLS
    ]), axis=0)
    upper_deviation = np.max(np.stack([
        bootstrap[control] - point[control] for control in PRIMARY_CONTROLS
    ]), axis=0)
    q_lower = max(0.0, float(np.quantile(lower_deviation, 0.95, method="higher")))
    q_upper = max(0.0, float(np.quantile(upper_deviation, 0.95, method="higher")))
    intervals = {
        control: {
            "estimate": point[control],
            "LCB": point[control] - q_lower,
            "UCB": point[control] + q_upper,
            "per_seed": per_seed[control],
        }
        for control in PRIMARY_CONTROLS
    }
    return {
        "intervals": intervals,
        "q_lower": q_lower,
        "q_upper": q_upper,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "replicates": replicates,
        "quantile_method": "higher",
    }


def gate_decision(map_family: Mapping[str, object], r1_family: Mapping[str, object],
                  audits_passed: bool) -> Dict[str, object]:
    map_intervals = map_family["intervals"]
    r1_intervals = r1_family["intervals"]
    per_seed_min = {
        seed: min(float(map_intervals[control]["per_seed"][seed])
                  for control in PRIMARY_CONTROLS)
        for seed in next(iter(map_intervals.values()))["per_seed"]
    }
    theta_min = min(float(map_intervals[control]["estimate"])
                    for control in PRIMARY_CONTROLS)
    go = (
        audits_passed
        and all(value > 0.0 for value in per_seed_min.values())
        and theta_min >= 0.30
        and all(float(map_intervals[control]["LCB"]) > 0.0 for control in PRIMARY_CONTROLS)
        and all(float(r1_intervals[control]["LCB"]) > -0.50 for control in PRIMARY_CONTROLS)
    )
    futility = any(float(map_intervals[control]["UCB"]) < 0.30
                   for control in PRIMARY_CONTROLS)
    reverse = any(
        float(map_intervals[control]["estimate"]) <= -0.30
        and float(map_intervals[control]["UCB"]) < 0.0
        for control in PRIMARY_CONTROLS
    )
    two_seed_nonpositive = sum(value <= 0.0 for value in per_seed_min.values()) >= 2
    if not audits_passed:
        decision = "INVALID"
    elif go:
        decision = "GO"
    elif futility or reverse or two_seed_nonpositive:
        decision = "NO_GO"
    else:
        decision = "INCONCLUSIVE"
    return {
        "decision": decision,
        "theta_min": theta_min,
        "theta_min_per_seed": per_seed_min,
        "futility": futility,
        "reverse": reverse,
        "two_seed_nonpositive": two_seed_nonpositive,
    }


def core_schedule(seeds: Sequence[int]) -> List[Dict[str, object]]:
    require(len(seeds) == 3, "E_SCHEDULE_SEEDS", str(seeds))
    schedule: List[Dict[str, object]] = []
    for seed in seeds:
        schedule.append({"seed": seed, "arm": "correct", "position": "start"})
        for mapping_index in range(20):
            schedule.append({"seed": seed, "arm": "shuffle", "mapping": mapping_index})
        schedule.append({"seed": seed, "arm": "centroid"})
        schedule.append({"seed": seed, "arm": "bypass"})
        for group_name in ANATOMICAL_GROUPS:
            for mapping_index in range(20):
                schedule.append({
                    "seed": seed,
                    "arm": "group",
                    "group": group_name,
                    "mapping": mapping_index,
                })
        schedule.append({"seed": seed, "arm": "correct", "position": "end"})
    require(len(schedule) == 492, "E_SCHEDULE_COUNT", str(len(schedule)))
    for seed in seeds:
        rows = [row for row in schedule if row["seed"] == seed]
        counts = {
            "correct": sum(row["arm"] == "correct" for row in rows),
            "shuffle": sum(row["arm"] == "shuffle" for row in rows),
            "centroid": sum(row["arm"] == "centroid" for row in rows),
            "bypass": sum(row["arm"] == "bypass" for row in rows),
            "group": sum(row["arm"] == "group" for row in rows),
        }
        require(counts == {
            "correct": 2,
            "shuffle": 20,
            "centroid": 1,
            "bypass": 1,
            "group": 140,
        }, "E_SCHEDULE_ARM_COUNT", f"seed={seed}: {counts}")
    return schedule
