#!/usr/bin/env python3
"""Pure NumPy contracts for the exp416 PC-NEC fuel audit.

This module intentionally has no torch, dataset, model, pose, CLIP, filesystem
write, or GPU dependency.  The future formal runner may use these functions
after it has atomically sealed the D0-only candidate bank.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import re
from pathlib import PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "exp416-pcnec-fuel-core-v1"
TRAIN_FILENAME_PATTERN = re.compile(r"^(-?\d+)_c(\d+)_f(\d+)\.jpg$")
TRAIN_PREFIX = "bounding_box_train/"
CAMERA_COUNT = 8
SLOT_COUNT = 5
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 128

QUERY_ORDER_SALT = "exp416-pcnec-query-order-v1"
PID_FOLD_SALT = "exp416-pcnec-pid-fold-v1"
PAIR_ID_SALT = "exp416-pcnec-pair-id-v1"
WRONG_RGB_DONOR_SALT = "exp416-pcnec-wrong-rgb-donor-v1"
BOOTSTRAP_BASE_SEED = 4161234
FOLD_COUNT = 5
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CAMERA_MATCHED_IMPOSTORS = True

ARM_NAMES = (
    "correct",
    "pose_only_raw_color",
    "pose_only_student_part",
    "canonical_location_clip",
    "neither",
    "slot_shuffle",
    "wrong_rgb",
    "global_clip",
    "d0_only",
)
CONTROL_ORDER = (
    "pose_only_raw_color",
    "pose_only_student_part",
    "canonical_location_clip",
    "neither",
    "slot_shuffle",
    "wrong_rgb",
    "global_clip",
)
MAIN_METRICS = ("auroc", "average_precision", "mAP", "R1")

# Frozen 8x8x8 CIELAB histogram.  Histograms are probability vectors and the
# distance is total variation, so every raw-color slot distance lies in [0, 1].
LAB_L_EDGES = np.linspace(0.0, 100.0, 9, dtype=np.float64)
LAB_A_EDGES = np.linspace(-128.0, 127.0, 9, dtype=np.float64)
LAB_B_EDGES = np.linspace(-128.0, 127.0, 9, dtype=np.float64)

# Filled by a literal known-answer test, not recomputed from another helper.
KNOWN_QUERY_HASH = (
    "ea7b689f93cab3ac31dcc09cc27c46d26a5471ae1178202aa1245d9ca326bd13"
)
KNOWN_PID_FOLD_HASH = (
    "e46bc1d4fd1f3bcd8481d61f1bdd243f110fca0024db83b808157229047e78de"
)
KNOWN_BOOTSTRAP_SEED = 3775876424573510604


def _hash_payload(salt: str, parts: Sequence[object]) -> bytes:
    values = (str(salt),) + tuple(str(value) for value in parts)
    if any("\0" in value for value in values):
        raise ValueError("hash fields may not contain NUL")
    return "\0".join(values).encode("utf-8")


def stable_hash_hex(salt: str, *parts: object) -> str:
    return hashlib.sha256(_hash_payload(salt, parts)).hexdigest()


def stable_hash_uint64(salt: str, *parts: object) -> int:
    digest = hashlib.sha256(_hash_payload(salt, parts)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def query_order_key(relative_path: str) -> tuple[str, str]:
    path = str(relative_path)
    return stable_hash_hex(QUERY_ORDER_SALT, path), path


def pid_fold(pid: int, fold_count: int = FOLD_COUNT) -> int:
    if int(fold_count) <= 1:
        raise ValueError("fold_count must exceed one")
    return stable_hash_uint64(PID_FOLD_SALT, int(pid)) % int(fold_count)


def bootstrap_seed(metric: str, control_name: str) -> int:
    name = "{}/{}".format(str(metric), str(control_name))
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return BOOTSTRAP_BASE_SEED ^ int.from_bytes(
        digest[:8], byteorder="big", signed=False
    )


def parse_train_filename(path: str) -> tuple[int, int, int]:
    """Return ``(raw_pid, zero_based_camera, frame)`` for one train JPEG."""
    name = PurePosixPath(str(path)).name
    match = TRAIN_FILENAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError("invalid Occluded-Duke filename: " + name)
    raw_pid, camera_one, frame = map(int, match.groups())
    if raw_pid < 0:
        raise ValueError("junk PID is forbidden")
    if not 1 <= camera_one <= CAMERA_COUNT:
        raise ValueError("camera must be in [1, 8]")
    return raw_pid, camera_one - 1, frame


def build_train_records(relative_paths: Iterable[str]) -> list[dict]:
    """Build deterministic relabeled train metadata without reading test splits."""
    paths = []
    for value in relative_paths:
        path = PurePosixPath(str(value)).as_posix()
        if path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise ValueError("train path must be safe and relative")
        if not path.startswith(TRAIN_PREFIX):
            raise ValueError("non-train path: " + path)
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate train relative path")
    parsed = {path: parse_train_filename(path) for path in paths}
    raw_pids = sorted({value[0] for value in parsed.values()})
    label_map = {raw_pid: label for label, raw_pid in enumerate(raw_pids)}
    rows = []
    for record_index, path in enumerate(sorted(paths)):
        raw_pid, camera, frame = parsed[path]
        rows.append(
            {
                "record_index": int(record_index),
                "relative_path": path,
                "train_pid": int(label_map[raw_pid]),
                "raw_pid": int(raw_pid),
                "camera": int(camera),
                "frame": int(frame),
            }
        )
    validate_train_records(rows)
    return rows


def validate_train_records(records: Sequence[Mapping]) -> None:
    if not records:
        raise ValueError("train records are empty")
    paths = [str(row["relative_path"]) for row in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("train records must have unique sorted paths")
    if [int(row["record_index"]) for row in records] != list(range(len(records))):
        raise ValueError("record_index is not contiguous")
    raw_to_label: dict[int, int] = {}
    label_to_raw: dict[int, int] = {}
    for row in records:
        raw_pid, camera, frame = parse_train_filename(row["relative_path"])
        observed = (
            int(row["raw_pid"]),
            int(row["camera"]),
            int(row["frame"]),
        )
        if observed != (raw_pid, camera, frame):
            raise ValueError("filename metadata mismatch")
        label = int(row["train_pid"])
        if raw_pid in raw_to_label and raw_to_label[raw_pid] != label:
            raise ValueError("one raw PID maps to multiple train labels")
        if label in label_to_raw and label_to_raw[label] != raw_pid:
            raise ValueError("one train label maps to multiple raw PIDs")
        raw_to_label[raw_pid] = label
        label_to_raw[label] = raw_pid
    expected_labels = list(range(len(raw_to_label)))
    if sorted(label_to_raw) != expected_labels:
        raise ValueError("train PID labels are not contiguous")
    expected_map = {
        raw_pid: label for label, raw_pid in enumerate(sorted(raw_to_label))
    }
    if raw_to_label != expected_map:
        raise ValueError("train PID relabeling is not sorted-raw-PID exact")


def _record_vectors(records: Sequence[Mapping]) -> tuple[list[str], np.ndarray, np.ndarray]:
    validate_train_records(records)
    paths = [str(row["relative_path"]) for row in records]
    pids = np.asarray([int(row["train_pid"]) for row in records], dtype=np.int64)
    cameras = np.asarray([int(row["camera"]) for row in records], dtype=np.int64)
    return paths, pids, cameras


def _normalize_descriptors(descriptors: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != int(count) or values.shape[1] < 1:
        raise ValueError("descriptor matrix shape mismatch")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite descriptor")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("zero-norm descriptor")
    return values / norms[:, None]


def _squared_euclidean_to_all(normalized: np.ndarray, row: int) -> np.ndarray:
    distance = 2.0 - 2.0 * (normalized @ normalized[int(row)])
    return np.maximum(distance, 0.0)


def _camera_matched_impostor_quota(
    true_rows: Sequence[int],
    cameras: np.ndarray,
    impostor_topk: int,
) -> dict[int, int]:
    true_camera_counts = {
        int(camera): int(count)
        for camera, count in zip(
            *np.unique(
                cameras[np.asarray(true_rows, dtype=np.int64)],
                return_counts=True,
            )
        )
    }
    if not true_camera_counts:
        raise ValueError("camera-matched bank requires a genuine candidate")
    if len(true_camera_counts) > int(impostor_topk):
        raise ValueError(
            "impostor top-K cannot cover every genuine camera stratum"
        )
    total_true = int(sum(true_camera_counts.values()))
    quota = {camera: 1 for camera in true_camera_counts}
    distributable = int(impostor_topk - len(true_camera_counts))
    extra = {
        camera: int(distributable * count // total_true)
        for camera, count in true_camera_counts.items()
    }
    for camera in quota:
        quota[camera] += extra[camera]
    remaining = int(impostor_topk - sum(quota.values()))
    remainder_order = sorted(
        true_camera_counts,
        key=lambda camera: (
            -(
                distributable * true_camera_counts[camera]
                - extra[camera] * total_true
            ),
            camera,
        ),
    )
    for camera in remainder_order[:remaining]:
        quota[camera] += 1
    if sum(quota.values()) != int(impostor_topk):
        raise RuntimeError("camera-matched impostor quota does not sum to top-K")
    return quota


def construct_candidate_bank(
    records: Sequence[Mapping],
    descriptors: np.ndarray,
    *,
    impostor_topk: int = 20,
) -> dict:
    """Construct the D0-only fixed bank before any pose/CLIP value exists."""
    if int(impostor_topk) <= 0:
        raise ValueError("impostor_topk must be positive")
    paths, pids, cameras = _record_vectors(records)
    normalized = _normalize_descriptors(descriptors, len(records))
    eligible = []
    excluded = []
    for query in range(len(records)):
        true = np.flatnonzero(
            (pids == pids[query]) & (cameras != cameras[query])
        )
        if len(true):
            eligible.append(query)
        else:
            excluded.append(query)
    eligible.sort(key=lambda row: query_order_key(paths[row]))

    pairs = []
    query_receipts = []
    pair_row = 0
    for query_order, query in enumerate(eligible):
        distances = _squared_euclidean_to_all(normalized, query)
        true = np.flatnonzero(
            (pids == pids[query]) & (cameras != cameras[query])
        ).tolist()
        camera_quota = _camera_matched_impostor_quota(
            true, cameras, int(impostor_topk)
        )
        selected_impostors = []
        for candidate_camera in sorted(camera_quota):
            camera_impostors = np.flatnonzero(
                (pids != pids[query]) & (cameras == candidate_camera)
            ).tolist()
            camera_impostors.sort(
                key=lambda row: (float(distances[row]), paths[row])
            )
            required = int(camera_quota[candidate_camera])
            if len(camera_impostors) < required:
                raise ValueError(
                    "query camera stratum has fewer than required impostors"
                )
            selected_impostors.extend(camera_impostors[:required])
        candidates = true + selected_impostors
        candidates.sort(key=lambda row: (float(distances[row]), paths[row]))
        query_id = stable_hash_hex(QUERY_ORDER_SALT, paths[query])
        query_receipts.append(
            {
                "query_order": int(query_order),
                "query_id": query_id,
                "query_index": int(query),
                "query_path": paths[query],
                "query_pid": int(pids[query]),
                "query_camera": int(cameras[query]),
                "true_count": int(len(true)),
                "impostor_count": int(impostor_topk),
                "genuine_camera_counts": [
                    [int(camera), int(count)]
                    for camera, count in sorted(
                        {
                            int(value): int(
                                np.sum(cameras[np.asarray(true)] == value)
                            )
                            for value in np.unique(
                                cameras[np.asarray(true, dtype=np.int64)]
                            ).tolist()
                        }.items()
                    )
                ],
                "impostor_camera_quota": [
                    [int(camera), int(count)]
                    for camera, count in sorted(camera_quota.items())
                ],
            }
        )
        selected_impostor_set = set(selected_impostors)
        for candidate_order, candidate in enumerate(candidates):
            same_pid = bool(pids[candidate] == pids[query])
            if same_pid and cameras[candidate] == cameras[query]:
                raise RuntimeError("same-camera genuine leaked into bank")
            if not same_pid and candidate not in selected_impostor_set:
                raise RuntimeError("non-topk impostor leaked into bank")
            pair_id = stable_hash_hex(
                PAIR_ID_SALT, paths[query], paths[candidate]
            )
            pairs.append(
                {
                    "pair_row": int(pair_row),
                    "pair_id": pair_id,
                    "query_order": int(query_order),
                    "candidate_order": int(candidate_order),
                    "query_index": int(query),
                    "candidate_index": int(candidate),
                    "query_path": paths[query],
                    "candidate_path": paths[candidate],
                    "query_pid": int(pids[query]),
                    "candidate_pid": int(pids[candidate]),
                    "query_camera": int(cameras[query]),
                    "candidate_camera": int(cameras[candidate]),
                    "same_pid": same_pid,
                    "impostor_positive": not same_pid,
                    "d0_distance": float(distances[candidate]),
                }
            )
            pair_row += 1
    return {
        "schema": "exp416-pcnec-candidate-bank-v1",
        "camera_matched_impostors": CAMERA_MATCHED_IMPOSTORS,
        "impostor_topk": int(impostor_topk),
        "descriptor_dimension": int(normalized.shape[1]),
        "record_count": int(len(records)),
        "eligible_query_count": int(len(eligible)),
        "excluded_no_cross_camera_true": [paths[row] for row in excluded],
        "query_receipts": query_receipts,
        "pairs": pairs,
    }


def validate_candidate_bank(
    records: Sequence[Mapping],
    descriptors: np.ndarray,
    bank: Mapping,
) -> None:
    required = {
        "schema",
        "camera_matched_impostors",
        "impostor_topk",
        "descriptor_dimension",
        "record_count",
        "eligible_query_count",
        "excluded_no_cross_camera_true",
        "query_receipts",
        "pairs",
    }
    if set(bank) != required:
        raise ValueError("candidate bank schema fields mismatch")
    if bank["schema"] != "exp416-pcnec-candidate-bank-v1":
        raise ValueError("candidate bank schema mismatch")
    if bank["camera_matched_impostors"] is not True:
        raise ValueError("candidate bank camera matching is disabled")
    expected = construct_candidate_bank(
        records,
        descriptors,
        impostor_topk=int(bank["impostor_topk"]),
    )
    if bank != expected:
        raise ValueError("candidate bank does not exactly match D0 construction")
    pairs = bank["pairs"]
    if [int(row["pair_row"]) for row in pairs] != list(range(len(pairs))):
        raise ValueError("candidate pair rows are not contiguous")
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("candidate pair IDs are not unique")


def clamped_crop_box(
    center_y: float,
    center_x: float,
    height: int,
    width: int,
    *,
    image_hw: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> tuple[int, int, int, int]:
    """Return exact ``(top, left, height, width)`` without padding."""
    image_height, image_width = map(int, image_hw)
    height, width = int(height), int(width)
    if min(image_height, image_width, height, width) <= 0:
        raise ValueError("crop and image dimensions must be positive")
    if height > image_height or width > image_width:
        raise ValueError("crop exceeds image dimensions")
    if not math.isfinite(float(center_y)) or not math.isfinite(float(center_x)):
        raise ValueError("crop center must be finite")
    top = int(round(float(center_y) - float(height) / 2.0))
    left = int(round(float(center_x) - float(width) / 2.0))
    top = min(max(top, 0), image_height - height)
    left = min(max(left, 0), image_width - width)
    box = (top, left, height, width)
    if box[2] != height or box[3] != width:
        raise RuntimeError("crop area changed after clamp")
    return box


def build_slot_crop_boxes(
    centers_yx: np.ndarray,
    slot_hw: np.ndarray,
    availability: np.ndarray,
    *,
    image_hw: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH),
) -> np.ndarray:
    centers = np.asarray(centers_yx, dtype=np.float64)
    sizes = np.asarray(slot_hw, dtype=np.int64)
    valid = np.asarray(availability, dtype=np.bool_)
    if centers.shape != (SLOT_COUNT, 2):
        raise ValueError("centers must have shape [5,2]")
    if sizes.shape != (SLOT_COUNT, 2):
        raise ValueError("slot_hw must have shape [5,2]")
    if valid.shape != (SLOT_COUNT,):
        raise ValueError("availability must have shape [5]")
    boxes = np.full((SLOT_COUNT, 4), -1, dtype=np.int32)
    for slot in range(SLOT_COUNT):
        if valid[slot]:
            boxes[slot] = clamped_crop_box(
                centers[slot, 0],
                centers[slot, 1],
                int(sizes[slot, 0]),
                int(sizes[slot, 1]),
                image_hw=image_hw,
            )
    return boxes


def crop_rgb(rgb: np.ndarray, box: Sequence[int]) -> np.ndarray:
    values = np.asarray(rgb)
    top, left, height, width = map(int, box)
    bottom = top + height
    right = left + width
    if values.ndim != 3:
        raise ValueError("RGB must be rank three")
    if values.shape[-1] == 3:
        height, width = values.shape[:2]
        output = values[top:bottom, left:right, :]
    elif values.shape[0] == 3:
        height, width = values.shape[1:]
        output = values[:, top:bottom, left:right]
    else:
        raise ValueError("RGB must be HWC or CHW")
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError("crop box is outside RGB")
    return np.ascontiguousarray(output)


def common_visibility(
    query_valid: np.ndarray, candidate_valid: np.ndarray
) -> np.ndarray:
    left = np.asarray(query_valid, dtype=np.bool_)
    right = np.asarray(candidate_valid, dtype=np.bool_)
    if left.shape != (SLOT_COUNT,) or right.shape != (SLOT_COUNT,):
        raise ValueError("slot validity must have shape [5]")
    return left & right


def existential_energy(
    slot_distances: np.ndarray, common: np.ndarray
) -> tuple[float, bool]:
    distances = np.asarray(slot_distances, dtype=np.float64)
    active = np.asarray(common, dtype=np.bool_)
    if distances.shape != (SLOT_COUNT,) or active.shape != (SLOT_COUNT,):
        raise ValueError("slot distance/common shape mismatch")
    if not bool(active.any()):
        return 0.0, True
    selected = distances[active]
    if not np.isfinite(selected).all() or np.any(selected < 0.0):
        raise ValueError("active slot distances must be finite and nonnegative")
    return float(selected.max()), False


def _rgb_float_hwc(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb)
    if values.ndim != 3:
        raise ValueError("RGB must be rank three")
    if values.shape[-1] == 3:
        output = values
    elif values.shape[0] == 3:
        output = np.moveaxis(values, 0, -1)
    else:
        raise ValueError("RGB must be HWC or CHW")
    output = output.astype(np.float64, copy=False)
    if np.issubdtype(values.dtype, np.integer):
        output = output / 255.0
    if not np.isfinite(output).all() or output.min() < 0.0 or output.max() > 1.0:
        raise ValueError("RGB must be finite in [0,1] or uint8")
    return output


def srgb_to_cielab(rgb: np.ndarray) -> np.ndarray:
    values = _rgb_float_hwc(rgb)
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.asarray(
        (
            (0.4124564, 0.3575761, 0.1804375),
            (0.2126729, 0.7151522, 0.0721750),
            (0.0193339, 0.1191920, 0.9503041),
        ),
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    white = np.asarray((0.95047, 1.0, 1.08883), dtype=np.float64)
    ratio = xyz / white
    delta = 6.0 / 29.0
    transformed = np.where(
        ratio > delta**3,
        np.cbrt(ratio),
        ratio / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lightness = 116.0 * transformed[..., 1] - 16.0
    channel_a = 500.0 * (transformed[..., 0] - transformed[..., 1])
    channel_b = 200.0 * (transformed[..., 1] - transformed[..., 2])
    output = np.stack((lightness, channel_a, channel_b), axis=-1)
    if not np.isfinite(output).all():
        raise ValueError("nonfinite CIELAB conversion")
    return output


def raw_color_histogram(rgb_crop: np.ndarray) -> np.ndarray:
    lab = srgb_to_cielab(rgb_crop).reshape(-1, 3)
    lab[:, 0] = np.clip(lab[:, 0], LAB_L_EDGES[0], LAB_L_EDGES[-1])
    lab[:, 1] = np.clip(lab[:, 1], LAB_A_EDGES[0], LAB_A_EDGES[-1])
    lab[:, 2] = np.clip(lab[:, 2], LAB_B_EDGES[0], LAB_B_EDGES[-1])
    histogram, _ = np.histogramdd(
        lab, bins=(LAB_L_EDGES, LAB_A_EDGES, LAB_B_EDGES)
    )
    total = float(histogram.sum())
    if total <= 0.0:
        raise ValueError("empty raw-color histogram")
    output = (histogram / total).reshape(-1).astype(np.float64, copy=False)
    if output.shape != (512,) or not np.isclose(output.sum(), 1.0):
        raise RuntimeError("raw-color histogram contract failed")
    return output


def histogram_tv_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != (512,) or right.shape != (512,):
        raise ValueError("raw-color histograms must have 512 bins")
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(left < 0.0)
        or np.any(right < 0.0)
        or not np.isclose(left.sum(), 1.0)
        or not np.isclose(right.sum(), 1.0)
    ):
        raise ValueError("invalid raw-color probability histogram")
    return float(0.5 * np.abs(left - right).sum())


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or left.size < 1:
        raise ValueError("cosine vectors must be aligned")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite cosine vector")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("zero-norm cosine vector")
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.clip(1.0 - cosine, 0.0, 2.0))


def _slot_cosine_distances(
    left: np.ndarray,
    right: np.ndarray,
    active: np.ndarray | None = None,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or left.shape[0] != SLOT_COUNT or right.shape != left.shape:
        raise ValueError("slot features must have shape [5,D]")
    if active is None:
        selected = np.ones(SLOT_COUNT, dtype=np.bool_)
    else:
        selected = np.asarray(active, dtype=np.bool_)
        if selected.shape != (SLOT_COUNT,):
            raise ValueError("active slot mask must have shape [5]")
    output = np.zeros(SLOT_COUNT, dtype=np.float64)
    for slot in range(SLOT_COUNT):
        if selected[slot]:
            output[slot] = cosine_distance(left[slot], right[slot])
    return output


def _slot_histogram_distances(
    left: np.ndarray,
    right: np.ndarray,
    active: np.ndarray | None = None,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != (SLOT_COUNT, 512) or right.shape != left.shape:
        raise ValueError("slot histograms must have shape [5,512]")
    if active is None:
        selected = np.ones(SLOT_COUNT, dtype=np.bool_)
    else:
        selected = np.asarray(active, dtype=np.bool_)
        if selected.shape != (SLOT_COUNT,):
            raise ValueError("active slot mask must have shape [5]")
    output = np.zeros(SLOT_COUNT, dtype=np.float64)
    for slot in range(SLOT_COUNT):
        if selected[slot]:
            output[slot] = histogram_tv_distance(left[slot], right[slot])
    return output


def select_wrong_rgb_donor(
    records: Sequence[Mapping],
    all_slot_valid: np.ndarray,
    *,
    query_path: str,
    candidate_path: str,
    query_pid: int,
    candidate_pid: int,
    candidate_camera: int,
    slot: int,
) -> int | None:
    valid = np.asarray(all_slot_valid, dtype=np.bool_)
    if valid.shape != (len(records), SLOT_COUNT):
        raise ValueError("all_slot_valid shape mismatch")
    if not 0 <= int(slot) < SLOT_COUNT:
        raise ValueError("slot index is outside five-slot ontology")
    pool = []
    for index, row in enumerate(records):
        pid = int(row["train_pid"])
        camera = int(row["camera"])
        if (
            bool(valid[index].all())
            and camera == int(candidate_camera)
            and pid != int(query_pid)
            and pid != int(candidate_pid)
        ):
            pool.append((str(row["relative_path"]), int(index)))
    pool.sort()
    if not pool:
        return None
    key = stable_hash_uint64(
        WRONG_RGB_DONOR_SALT,
        str(query_path),
        str(candidate_path),
        int(slot),
    )
    return pool[key % len(pool)][1]


def compute_pair_arm_energies(
    *,
    query_valid: np.ndarray,
    candidate_valid: np.ndarray,
    correct_clip_query: np.ndarray,
    correct_clip_candidate: np.ndarray,
    student_query: np.ndarray,
    student_candidate: np.ndarray,
    raw_hist_query: np.ndarray,
    raw_hist_candidate: np.ndarray,
    canonical_clip_query: np.ndarray,
    canonical_clip_candidate: np.ndarray,
    canonical_raw_hist_query: np.ndarray,
    canonical_raw_hist_candidate: np.ndarray,
    global_clip_query: np.ndarray,
    global_clip_candidate: np.ndarray,
    d0_distance: float,
    wrong_donor_clip: np.ndarray | None,
) -> dict:
    """Compute all fixed pair energies without changing pair availability."""
    common = common_visibility(query_valid, candidate_valid)
    correct_slots = _slot_cosine_distances(
        correct_clip_query, correct_clip_candidate, common
    )
    student_slots = _slot_cosine_distances(
        student_query, student_candidate, common
    )
    raw_slots = _slot_histogram_distances(
        raw_hist_query, raw_hist_candidate, common
    )
    canonical_clip_slots = _slot_cosine_distances(
        canonical_clip_query, canonical_clip_candidate, common
    )
    canonical_raw_slots = _slot_histogram_distances(
        canonical_raw_hist_query, canonical_raw_hist_candidate, common
    )
    energies = {}
    energies["correct"], undecided = existential_energy(correct_slots, common)
    energies["pose_only_raw_color"], _ = existential_energy(raw_slots, common)
    energies["pose_only_student_part"], _ = existential_energy(
        student_slots, common
    )
    energies["canonical_location_clip"], _ = existential_energy(
        canonical_clip_slots, common
    )
    energies["neither"], _ = existential_energy(canonical_raw_slots, common)

    shuffled_slots = np.zeros(SLOT_COUNT, dtype=np.float64)
    common_indices = np.flatnonzero(common)
    if len(common_indices):
        source_indices = np.roll(common_indices, -1)
        for target_slot, source_slot in zip(common_indices, source_indices):
            shuffled_slots[target_slot] = cosine_distance(
                np.asarray(correct_clip_query)[target_slot],
                np.asarray(correct_clip_candidate)[source_slot],
            )
    energies["slot_shuffle"], _ = existential_energy(shuffled_slots, common)

    donor_invalid = wrong_donor_clip is None
    if donor_invalid:
        energies["wrong_rgb"] = 0.0
    else:
        donor = np.asarray(wrong_donor_clip, dtype=np.float64)
        if donor.shape != np.asarray(correct_clip_query).shape:
            raise ValueError("wrong donor slot feature shape mismatch")
        wrong_slots = _slot_cosine_distances(correct_clip_query, donor, common)
        energies["wrong_rgb"], _ = existential_energy(wrong_slots, common)
    energies["global_clip"] = cosine_distance(
        global_clip_query, global_clip_candidate
    )
    if not math.isfinite(float(d0_distance)) or float(d0_distance) < 0.0:
        raise ValueError("D0 distance must be finite and nonnegative")
    energies["d0_only"] = float(d0_distance)
    if set(energies) != set(ARM_NAMES):
        raise RuntimeError("arm energy schema mismatch")
    if any(not math.isfinite(value) or value < 0.0 for value in energies.values()):
        raise RuntimeError("nonfinite or negative arm energy")
    return {
        "energies": energies,
        "common": common,
        "common_count": int(common.sum()),
        "undecided": bool(undecided),
        "wrong_donor_invalid": bool(donor_invalid),
    }


def _average_rank_1based(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("rank input must be a finite nonempty vector")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def empirical_midrank(values: np.ndarray) -> np.ndarray:
    """Empirical CDF midrank in (0,1), with an all-tie vector equal to 0.5."""
    ranks = _average_rank_1based(values)
    return (ranks - 0.5) / float(len(ranks))


def binary_auroc_average_precision(
    impostor_positive: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Tie-aware AUROC and non-interpolated average precision.

    Average precision is the weighted mean of precision at every distinct
    descending score threshold, matching ``sklearn.average_precision_score``.
    """
    labels = np.asarray(impostor_positive, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) < 2:
        raise ValueError("binary metric vectors are misaligned")
    if not np.isfinite(scores).all():
        raise ValueError("binary scores are nonfinite")
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC/AP require both classes")

    ranks = _average_rank_1based(scores)
    mann_whitney = float(ranks[labels].sum()) - (
        positive_count * (positive_count + 1) / 2.0
    )
    auroc = mann_whitney / float(positive_count * negative_count)

    order = np.argsort(-scores, kind="mergesort")
    cumulative_positive = 0
    cumulative_total = 0
    average_precision = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        group_positive = int(labels[order[start:stop]].sum())
        cumulative_positive += group_positive
        cumulative_total += stop - start
        recall_increment = group_positive / float(positive_count)
        precision = cumulative_positive / float(cumulative_total)
        average_precision += recall_increment * precision
        start = stop
    return float(auroc), float(average_precision)


def retrieval_ap_r1(
    genuine: np.ndarray,
    combined_score: np.ndarray,
    d0_distance: np.ndarray,
    candidate_paths: Sequence[str],
) -> tuple[float, float]:
    genuine = np.asarray(genuine, dtype=np.bool_)
    score = np.asarray(combined_score, dtype=np.float64)
    d0 = np.asarray(d0_distance, dtype=np.float64)
    paths = np.asarray([str(value) for value in candidate_paths], dtype=np.str_)
    if (
        genuine.ndim != 1
        or score.shape != genuine.shape
        or d0.shape != genuine.shape
        or paths.shape != genuine.shape
        or not np.isfinite(score).all()
        or not np.isfinite(d0).all()
    ):
        raise ValueError("retrieval vectors are invalid")
    relevant = int(genuine.sum())
    if relevant == 0 or relevant == len(genuine):
        raise ValueError("retrieval query needs genuine and impostor candidates")
    order = np.lexsort((paths, d0, score))
    ordered = genuine[order]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    precision = cumulative / np.arange(1, len(ordered) + 1, dtype=np.float64)
    average_precision = float(precision[ordered].sum() / relevant)
    return average_precision, float(bool(ordered[0]))


def validate_query_payload(query: Mapping, arm: str | None = None) -> None:
    required = {
        "query_id",
        "query_pid",
        "d0_distance",
        "candidate_paths",
        "impostor_positive",
        "arm_energy",
    }
    if set(query) != required:
        raise ValueError("query payload fields mismatch")
    d0 = np.asarray(query["d0_distance"], dtype=np.float64)
    labels = np.asarray(query["impostor_positive"], dtype=np.bool_)
    paths = list(query["candidate_paths"])
    if (
        d0.ndim != 1
        or labels.shape != d0.shape
        or len(paths) != len(d0)
        or len(paths) != len(set(map(str, paths)))
        or not np.isfinite(d0).all()
        or np.any(d0 < 0.0)
    ):
        raise ValueError("query bank vectors are invalid")
    if labels.all() or not labels.any():
        raise ValueError("query bank needs genuine and impostor candidates")
    energies = query["arm_energy"]
    if not isinstance(energies, Mapping):
        raise ValueError("arm_energy must be a mapping")
    names = (arm,) if arm is not None else tuple(energies)
    for name in names:
        if name not in energies:
            raise ValueError("missing arm energy: " + str(name))
        value = np.asarray(energies[name], dtype=np.float64)
        if value.shape != d0.shape or not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError("invalid arm energy vector: " + str(name))


def evaluate_query(query: Mapping, arm: str, lambda_value: float) -> dict:
    validate_query_payload(query, arm)
    if float(lambda_value) not in LAMBDA_GRID:
        raise ValueError("lambda is outside the frozen grid")
    d0 = np.asarray(query["d0_distance"], dtype=np.float64)
    energy = np.asarray(query["arm_energy"][arm], dtype=np.float64)
    labels = np.asarray(query["impostor_positive"], dtype=np.bool_)
    rank_d0 = empirical_midrank(d0)
    rank_energy = empirical_midrank(energy)
    combined = (1.0 - float(lambda_value)) * rank_d0 + float(
        lambda_value
    ) * rank_energy
    auroc, average_precision = binary_auroc_average_precision(labels, energy)
    mean_ap, rank1 = retrieval_ap_r1(
        ~labels,
        combined,
        d0,
        query["candidate_paths"],
    )
    return {
        "query_id": str(query["query_id"]),
        "query_pid": int(query["query_pid"]),
        "lambda": float(lambda_value),
        "auroc": float(auroc),
        "average_precision": float(average_precision),
        "mAP": float(mean_ap),
        "R1": float(rank1),
        "combined_score": combined,
    }


def pid_macro_metrics(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("PID macro rows are empty")
    by_pid: dict[int, list[Mapping]] = {}
    seen = set()
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in seen:
            raise ValueError("duplicate query_id in PID macro")
        seen.add(query_id)
        by_pid.setdefault(int(row["query_pid"]), []).append(row)
    output = {}
    for metric in MAIN_METRICS:
        pid_values = []
        for pid_rows in by_pid.values():
            values = np.asarray(
                [float(row[metric]) for row in pid_rows], dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError("nonfinite PID macro metric")
            pid_values.append(float(values.mean()))
        output[metric] = float(np.mean(pid_values))
    output["pid_count"] = int(len(by_pid))
    output["query_count"] = int(len(rows))
    return output


def _select_lambda(
    queries: Sequence[Mapping],
    arm: str,
    train_pids: set[int],
    lambdas: Sequence[float],
) -> tuple[float, dict]:
    candidates = []
    train_queries = [
        query for query in queries if int(query["query_pid"]) in train_pids
    ]
    if not train_queries:
        raise ValueError("OOF training PID set has no queries")
    for lambda_value in lambdas:
        rows = [evaluate_query(query, arm, float(lambda_value)) for query in train_queries]
        summary = pid_macro_metrics(rows)
        candidates.append((float(lambda_value), summary))
    candidates.sort(
        key=lambda item: (
            -float(item[1]["mAP"]),
            -float(item[1]["R1"]),
            float(item[0]),
        )
    )
    return candidates[0]


def evaluate_arm_oof(
    queries: Sequence[Mapping],
    arm: str,
    *,
    lambdas: Sequence[float] = LAMBDA_GRID,
    fold_count: int = FOLD_COUNT,
) -> dict:
    if not queries:
        raise ValueError("OOF queries are empty")
    if tuple(map(float, lambdas)) != LAMBDA_GRID:
        raise ValueError("lambda grid differs from frozen contract")
    query_ids = [str(query["query_id"]) for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("OOF query IDs are not unique")
    for query in queries:
        validate_query_payload(query, arm)
    pids = sorted({int(query["query_pid"]) for query in queries})
    pid_to_fold = {pid: pid_fold(pid, fold_count) for pid in pids}
    if set(pid_to_fold.values()) != set(range(int(fold_count))):
        raise ValueError("every OOF fold must contain at least one query PID")

    selected = {}
    rows = []
    all_pids = set(pids)
    for heldout in range(int(fold_count)):
        heldout_pids = {pid for pid, fold in pid_to_fold.items() if fold == heldout}
        train_pids = all_pids - heldout_pids
        lambda_value, training_summary = _select_lambda(
            queries, arm, train_pids, lambdas
        )
        selected[int(heldout)] = {
            "lambda": float(lambda_value),
            "training_pid_count": int(len(train_pids)),
            "heldout_pid_count": int(len(heldout_pids)),
            "training_mAP": float(training_summary["mAP"]),
            "training_R1": float(training_summary["R1"]),
        }
        for query in queries:
            pid = int(query["query_pid"])
            if pid in heldout_pids:
                result = evaluate_query(query, arm, lambda_value)
                result["fold"] = int(heldout)
                rows.append(result)
    rows.sort(key=lambda row: str(row["query_id"]))
    if {str(row["query_id"]) for row in rows} != set(query_ids):
        raise RuntimeError("OOF output does not cover every query exactly")
    return {
        "arm": str(arm),
        "fold_count": int(fold_count),
        "pid_to_fold": pid_to_fold,
        "selected": selected,
        "rows": rows,
        "summary": pid_macro_metrics(rows),
    }


def evaluate_d0_only(queries: Sequence[Mapping]) -> dict:
    rows = []
    for query in queries:
        validate_query_payload(query)
        copied = dict(query)
        copied["arm_energy"] = dict(query["arm_energy"])
        copied["arm_energy"]["d0_only"] = np.asarray(
            query["d0_distance"], dtype=np.float64
        )
        result = evaluate_query(copied, "d0_only", 0.0)
        result["fold"] = int(pid_fold(int(query["query_pid"])))
        rows.append(result)
    rows.sort(key=lambda row: str(row["query_id"]))
    return {
        "arm": "d0_only",
        "rows": rows,
        "summary": pid_macro_metrics(rows),
    }


def select_strongest_controls(
    arm_summaries: Mapping[str, Mapping],
    *,
    control_order: Sequence[str] = CONTROL_ORDER,
) -> dict:
    order = tuple(map(str, control_order))
    if len(order) != len(set(order)):
        raise ValueError("control order contains duplicates")
    missing = [name for name in order if name not in arm_summaries]
    if missing:
        raise ValueError("missing control summaries: " + ", ".join(missing))
    output = {}
    for metric in MAIN_METRICS:
        best = None
        for index, name in enumerate(order):
            value = float(arm_summaries[name][metric])
            if not math.isfinite(value):
                raise ValueError("nonfinite control metric")
            candidate = (-value, index, name)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        output[metric] = {
            "arm": best[2],
            "value": float(-best[0]),
        }
    return output


def _align_paired_rows(
    correct_rows: Sequence[Mapping],
    control_rows: Sequence[Mapping],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correct = {str(row["query_id"]): row for row in correct_rows}
    control = {str(row["query_id"]): row for row in control_rows}
    if len(correct) != len(correct_rows) or len(control) != len(control_rows):
        raise ValueError("duplicate query_id in paired rows")
    if set(correct) != set(control):
        raise ValueError("paired query sets differ")
    query_ids = sorted(correct)
    pids = []
    left = []
    right = []
    for query_id in query_ids:
        if int(correct[query_id]["query_pid"]) != int(
            control[query_id]["query_pid"]
        ):
            raise ValueError("paired query PID differs")
        pids.append(int(correct[query_id]["query_pid"]))
        left.append(float(correct[query_id][metric]))
        right.append(float(control[query_id][metric]))
    arrays = (
        np.asarray(pids, dtype=np.int64),
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    if not all(np.isfinite(value).all() for value in arrays[1:]):
        raise ValueError("nonfinite paired metric")
    return arrays


def paired_pid_bootstrap(
    correct_rows: Sequence[Mapping],
    control_rows: Sequence[Mapping],
    *,
    metric: str,
    control_name: str,
    repetitions: int = 10000,
) -> dict:
    if int(repetitions) != 10000:
        raise ValueError("formal PID bootstrap requires 10,000 repetitions")
    if metric not in MAIN_METRICS:
        raise ValueError("unsupported bootstrap metric")
    pids, left, right = _align_paired_rows(correct_rows, control_rows, metric)
    unique_pids = np.asarray(sorted(set(pids.tolist())), dtype=np.int64)
    if len(unique_pids) < 2:
        raise ValueError("PID bootstrap requires at least two query PIDs")
    pid_delta = np.empty(len(unique_pids), dtype=np.float64)
    for index, pid in enumerate(unique_pids):
        active = pids == pid
        pid_delta[index] = float(left[active].mean() - right[active].mean())
    seed = bootstrap_seed(metric, control_name)
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(int(repetitions), dtype=np.float64)
    for start in range(0, int(repetitions), 256):
        stop = min(start + 256, int(repetitions))
        indices = rng.integers(
            0,
            len(unique_pids),
            size=(stop - start, len(unique_pids)),
            endpoint=False,
        )
        samples[start:stop] = pid_delta[indices].mean(axis=1)
    lower = float(np.quantile(samples, 0.05, method="linear"))
    return {
        "metric": str(metric),
        "control": str(control_name),
        "estimate": float(pid_delta.mean()),
        "one_sided_95_lower": lower,
        "repetitions": int(repetitions),
        "pid_count": int(len(unique_pids)),
        "seed": int(seed),
    }


def simultaneous_control_pid_bootstrap(
    correct_rows: Sequence[Mapping],
    control_rows_by_name: Mapping[str, Sequence[Mapping]],
    *,
    metric: str,
    repetitions: int = 10000,
) -> dict:
    if int(repetitions) != 10000:
        raise ValueError("formal PID bootstrap requires 10,000 repetitions")
    if metric not in MAIN_METRICS:
        raise ValueError("unsupported bootstrap metric")
    control_names = tuple(str(name) for name in CONTROL_ORDER)
    if set(control_rows_by_name) != set(control_names):
        raise ValueError("simultaneous bootstrap control set mismatch")
    reference_pids = None
    delta_columns = []
    unique_pids = None
    for name in control_names:
        pids, left, right = _align_paired_rows(
            correct_rows, control_rows_by_name[name], metric
        )
        if reference_pids is None:
            reference_pids = pids
            unique_pids = np.asarray(
                sorted(set(pids.tolist())), dtype=np.int64
            )
            if len(unique_pids) < 2:
                raise ValueError(
                    "PID bootstrap requires at least two query PIDs"
                )
        elif not np.array_equal(reference_pids, pids):
            raise ValueError("simultaneous controls changed paired PID order")
        pid_delta = np.empty(len(unique_pids), dtype=np.float64)
        for index, pid in enumerate(unique_pids):
            active = pids == pid
            pid_delta[index] = float(
                left[active].mean() - right[active].mean()
            )
        delta_columns.append(pid_delta)
    assert unique_pids is not None
    pid_delta_matrix = np.stack(delta_columns, axis=1)
    seed = bootstrap_seed(metric, "all_controls_min")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(int(repetitions), dtype=np.float64)
    for start in range(0, int(repetitions), 256):
        stop = min(start + 256, int(repetitions))
        indices = rng.integers(
            0,
            len(unique_pids),
            size=(stop - start, len(unique_pids)),
            endpoint=False,
        )
        replicate_deltas = pid_delta_matrix[indices].mean(axis=1)
        samples[start:stop] = replicate_deltas.min(axis=1)
    lower = float(np.quantile(samples, 0.05, method="linear"))
    per_control_estimate = {
        name: float(pid_delta_matrix[:, index].mean())
        for index, name in enumerate(control_names)
    }
    return {
        "metric": str(metric),
        "control": "all_controls_min",
        "controls": list(control_names),
        "estimate": float(min(per_control_estimate.values())),
        "per_control_estimate": per_control_estimate,
        "one_sided_95_lower": lower,
        "repetitions": int(repetitions),
        "pid_count": int(len(unique_pids)),
        "seed": int(seed),
    }


def _expect_raises(callable_value) -> None:
    try:
        callable_value()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("expected contract failure")


def _self_test_hash_and_records() -> None:
    sample_path = "bounding_box_train/0001_c1_f000001.jpg"
    assert stable_hash_hex(QUERY_ORDER_SALT, sample_path) == KNOWN_QUERY_HASH
    assert stable_hash_hex(PID_FOLD_SALT, 17) == KNOWN_PID_FOLD_HASH
    assert bootstrap_seed("mAP", "global_clip") == KNOWN_BOOTSTRAP_SEED
    records = build_train_records(
        (
            "bounding_box_train/0010_c2_f000003.jpg",
            "bounding_box_train/0002_c1_f000001.jpg",
            "bounding_box_train/0010_c1_f000002.jpg",
        )
    )
    assert [row["train_pid"] for row in records] == [0, 1, 1]
    assert [row["camera"] for row in records] == [0, 0, 1]
    _expect_raises(
        lambda: build_train_records(
            ("query/0002_c1_f000001.jpg",)
        )
    )


def _self_test_candidate_bank_and_donor() -> None:
    quota = _camera_matched_impostor_quota(
        (0, 2, 3),
        np.asarray((1, 1, 2, 2, 2), dtype=np.int64),
        20,
    )
    assert quota == {1: 7, 2: 13}
    extreme_quota = _camera_matched_impostor_quota(
        tuple(range(101)),
        np.asarray((1,) + (2,) * 100, dtype=np.int64),
        20,
    )
    assert min(extreme_quota.values()) >= 1
    assert sum(extreme_quota.values()) == 20
    paths = []
    for pid in (1, 2, 3, 4):
        for camera in (1, 2):
            paths.append(
                "bounding_box_train/{:04d}_c{}_f{:06d}.jpg".format(
                    pid, camera, pid * 10 + camera
                )
            )
    records = build_train_records(paths)
    descriptors = np.asarray(
        [
            (1.0, 0.0, 0.0),
            (0.9, 0.1, 0.0),
            (0.0, 1.0, 0.0),
            (0.1, 0.9, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.1, 0.9),
            (0.7, 0.7, 0.0),
            (0.6, 0.8, 0.0),
        ],
        dtype=np.float64,
    )
    bank = construct_candidate_bank(records, descriptors, impostor_topk=2)
    validate_candidate_bank(records, descriptors, bank)
    assert bank["eligible_query_count"] == len(records)
    assert bank["camera_matched_impostors"] is True
    assert all(
        sum(not pair["same_pid"] for pair in bank["pairs"] if pair["query_index"] == q)
        == 2
        for q in range(len(records))
    )
    for query in range(len(records)):
        rows = [
            pair for pair in bank["pairs"] if pair["query_index"] == query
        ]
        genuine_cameras = sorted(
            pair["candidate_camera"] for pair in rows if pair["same_pid"]
        )
        impostor_cameras = sorted(
            pair["candidate_camera"] for pair in rows if not pair["same_pid"]
        )
        assert set(impostor_cameras) == set(genuine_cameras)
    broken = copy.deepcopy(bank)
    broken["pairs"][0]["candidate_pid"] += 1
    _expect_raises(lambda: validate_candidate_bank(records, descriptors, broken))

    valid = np.ones((len(records), SLOT_COUNT), dtype=np.bool_)
    query = records[0]
    candidate = records[2]
    donor = select_wrong_rgb_donor(
        records,
        valid,
        query_path=query["relative_path"],
        candidate_path=candidate["relative_path"],
        query_pid=query["train_pid"],
        candidate_pid=candidate["train_pid"],
        candidate_camera=candidate["camera"],
        slot=3,
    )
    assert donor is not None
    assert records[donor]["camera"] == candidate["camera"]
    assert records[donor]["train_pid"] not in {
        query["train_pid"],
        candidate["train_pid"],
    }
    valid[:] = False
    assert (
        select_wrong_rgb_donor(
            records,
            valid,
            query_path=query["relative_path"],
            candidate_path=candidate["relative_path"],
            query_pid=query["train_pid"],
            candidate_pid=candidate["train_pid"],
            candidate_camera=candidate["camera"],
            slot=3,
        )
        is None
    )


def _self_test_crops_color_and_undecided() -> None:
    assert clamped_crop_box(-10.0, -20.0, 20, 10) == (0, 0, 20, 10)
    assert clamped_crop_box(999.0, 999.0, 20, 10) == (364, 118, 20, 10)
    centers = np.asarray([(0, 0), (50, 20), (100, 40), (200, 80), (383, 127)])
    sizes = np.asarray([(20, 10)] * SLOT_COUNT)
    valid = np.asarray([1, 1, 0, 1, 1], dtype=np.bool_)
    boxes = build_slot_crop_boxes(centers, sizes, valid)
    assert np.array_equal(boxes[2], np.asarray([-1, -1, -1, -1]))
    assert np.all(boxes[valid, 2] == 20)
    assert np.all(boxes[valid, 3] == 10)

    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    black_hist = raw_color_histogram(black)
    white_hist = raw_color_histogram(white)
    assert histogram_tv_distance(black_hist, black_hist) == 0.0
    assert histogram_tv_distance(black_hist, white_hist) == 1.0

    energy, undecided = existential_energy(
        np.arange(SLOT_COUNT, dtype=np.float64),
        np.zeros(SLOT_COUNT, dtype=np.bool_),
    )
    assert energy == 0.0 and undecided
    energy, undecided = existential_energy(
        np.asarray((0.1, 0.7, 0.2, 0.4, 0.3)),
        np.asarray((1, 1, 0, 0, 0), dtype=np.bool_),
    )
    assert energy == 0.7 and not undecided

    eye = np.eye(SLOT_COUNT, dtype=np.float64)
    hist = np.tile(black_hist, (SLOT_COUNT, 1))
    invalid_features = np.zeros_like(eye)
    invalid_histograms = np.zeros_like(hist)
    result = compute_pair_arm_energies(
        query_valid=np.zeros(SLOT_COUNT, dtype=np.bool_),
        candidate_valid=np.ones(SLOT_COUNT, dtype=np.bool_),
        correct_clip_query=invalid_features,
        correct_clip_candidate=eye,
        student_query=invalid_features,
        student_candidate=eye,
        raw_hist_query=invalid_histograms,
        raw_hist_candidate=hist,
        canonical_clip_query=invalid_features,
        canonical_clip_candidate=eye,
        canonical_raw_hist_query=invalid_histograms,
        canonical_raw_hist_candidate=hist,
        global_clip_query=np.asarray((1.0, 0.0)),
        global_clip_candidate=np.asarray((0.0, 1.0)),
        d0_distance=0.25,
        wrong_donor_clip=None,
    )
    assert result["undecided"] and result["common_count"] == 0
    dependent = set(ARM_NAMES) - {"global_clip", "d0_only"}
    assert all(result["energies"][name] == 0.0 for name in dependent)
    assert result["energies"]["global_clip"] == 1.0


def _self_test_metrics_and_folds() -> None:
    midrank = empirical_midrank(np.asarray((1.0, 1.0, 2.0, 4.0)))
    assert np.array_equal(midrank, np.asarray((0.25, 0.25, 0.625, 0.875)))
    all_tie = empirical_midrank(np.ones(4))
    assert np.array_equal(all_tie, np.full(4, 0.5))
    labels = np.asarray((0, 1, 0, 1), dtype=np.bool_)
    auroc, ap = binary_auroc_average_precision(
        labels, np.asarray((0.1, 0.8, 0.4, 0.8))
    )
    assert auroc == 1.0 and ap == 1.0
    auroc, ap = binary_auroc_average_precision(labels, np.zeros(4))
    assert auroc == 0.5 and ap == 0.5

    fold_pids = {}
    pid = 0
    while len(fold_pids) < FOLD_COUNT:
        fold_pids.setdefault(pid_fold(pid), pid)
        pid += 1
    queries = []
    for fold in range(FOLD_COUNT):
        query_pid = fold_pids[fold]
        queries.append(
            {
                "query_id": "q{}".format(fold),
                "query_pid": int(query_pid),
                "d0_distance": np.asarray((0.30, 0.10, 0.20)),
                "candidate_paths": (
                    "genuine{}".format(fold),
                    "impostor-a{}".format(fold),
                    "impostor-b{}".format(fold),
                ),
                "impostor_positive": np.asarray((0, 1, 1), dtype=np.bool_),
                "arm_energy": {
                    "correct": np.asarray((0.10, 0.90, 0.80)),
                    "d0_only": np.asarray((0.30, 0.10, 0.20)),
                },
            }
        )
    first = evaluate_arm_oof(queries, "correct")
    mutated = copy.deepcopy(queries)
    heldout_fold = 2
    heldout_pid = fold_pids[heldout_fold]
    for query in mutated:
        if query["query_pid"] == heldout_pid:
            query["arm_energy"]["correct"] = np.asarray((0.99, 0.01, 0.02))
    second = evaluate_arm_oof(mutated, "correct")
    assert (
        first["selected"][heldout_fold]["lambda"]
        == second["selected"][heldout_fold]["lambda"]
    )
    assert set(first["pid_to_fold"].values()) == set(range(FOLD_COUNT))

    summaries = {}
    for index, name in enumerate(CONTROL_ORDER):
        summaries[name] = {
            "auroc": 0.5 + 0.01 * index,
            "average_precision": 0.4,
            "mAP": 0.3,
            "R1": 0.2,
        }
    strongest = select_strongest_controls(summaries)
    assert strongest["auroc"]["arm"] == CONTROL_ORDER[-1]
    assert strongest["average_precision"]["arm"] == CONTROL_ORDER[0]


def _self_test_bootstrap() -> None:
    correct = []
    control = []
    rows = (
        ("q0", 0, 0.8, 0.6),
        ("q1", 0, 0.6, 0.4),
        ("q2", 1, 0.7, 0.6),
        ("q3", 2, 0.9, 0.5),
    )
    for query_id, query_pid, left, right in rows:
        correct.append(
            {"query_id": query_id, "query_pid": query_pid, "mAP": left}
        )
        control.append(
            {"query_id": query_id, "query_pid": query_pid, "mAP": right}
        )
    first = paired_pid_bootstrap(
        correct, control, metric="mAP", control_name="global_clip"
    )
    second = paired_pid_bootstrap(
        correct, control, metric="mAP", control_name="global_clip"
    )
    expected = np.mean((0.2, 0.1, 0.4))
    assert math.isclose(first["estimate"], expected, abs_tol=1e-15)
    assert first == second
    assert first["repetitions"] == 10000 and first["pid_count"] == 3
    weaker = [
        {
            "query_id": row["query_id"],
            "query_pid": row["query_pid"],
            "mAP": float(row["mAP"]) - 0.2,
        }
        for row in control
    ]
    simultaneous = simultaneous_control_pid_bootstrap(
        correct,
        {
            name: control if index == 0 else weaker
            for index, name in enumerate(CONTROL_ORDER)
        },
        metric="mAP",
    )
    assert math.isclose(
        simultaneous["estimate"], expected, abs_tol=1e-15
    )
    assert simultaneous["control"] == "all_controls_min"
    assert simultaneous["controls"] == list(CONTROL_ORDER)
    _expect_raises(
        lambda: paired_pid_bootstrap(
            correct,
            control,
            metric="mAP",
            control_name="global_clip",
            repetitions=9999,
        )
    )


def run_self_test() -> dict:
    _self_test_hash_and_records()
    _self_test_candidate_bank_and_donor()
    _self_test_crops_color_and_undecided()
    _self_test_metrics_and_folds()
    _self_test_bootstrap()
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "tests": (
            "hash-records",
            "candidate-bank-donor",
            "crops-color-undecided",
            "ties-metrics-fold-leakage",
            "pid-pcg64-bootstrap",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.self_test:
        raise SystemExit("fuel_core.py only exposes --self-test")
    result = run_self_test()
    print("{}={}".format(result["schema"], result["status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
