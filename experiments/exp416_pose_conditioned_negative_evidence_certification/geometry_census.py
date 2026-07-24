#!/usr/bin/env python3
"""Pose/RGB-only geometry census for the exp416 PC-NEC fuel audit.

This is the second physical stage.  It consumes an already sealed D0 candidate
bank, but it never imports or evaluates OpenCLIP and never changes candidate
membership.  All rectangle sizes, canonical centers, slot availability, and
coverage gates are frozen before any semantic feature is read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
for import_root in (REPOSITORY_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import fuel_io


SCHEMA = "exp416-pcnec-geometry-v1"
BANK_SCHEMA = "exp416-pcnec-bank-v1"
EXPECTED_TRAIN_COUNT = 15618
EXPECTED_POSE_MANIFEST_SHA256 = (
    "cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8"
)
CANONICAL_HEIGHT = 384
CANONICAL_WIDTH = 128
SLOT_NAMES = (
    "head",
    "upper_torso_arms",
    "lower_torso",
    "upper_legs",
    "lower_legs_feet",
)
SLOT_JOINTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9, 10),
    (11, 12),
    (11, 12, 13, 14),
    (13, 14, 15, 16),
)

# Frozen before the census.  A slot is "real visible" only if at least two of
# its ontology joints are geometrically valid and have ViTPose score >= 0.30.
JOINT_SCORE_MIN = 0.30
MIN_VISIBLE_JOINTS = 2

# Crop dimensions are an all-train, pose-only 90th percentile of normalized
# active-joint span plus five percent canvas padding on each side.  The fixed
# pixel result is shared by correct and canonical controls.
CROP_SPAN_QUANTILE = 0.90
CROP_PADDING_FRACTION = 0.05
MIN_CROP_HEIGHT = 16
MIN_CROP_WIDTH = 16

# Absolute per-slot fuel coverage gates, frozen without reading CLIP.
MIN_COMMON_PAIRS_PER_SLOT = 100000
MIN_QUERY_PIDS_PER_SLOT = 300
MIN_QUERY_COVERAGE = 0.80

REQUIRED_BANK_FIELDS = {
    "schema",
    "relative_paths",
    "raw_pids",
    "relabeled_pids",
    "camids",
    "d0_global",
    "query_indices",
    "candidate_indices",
    "pair_is_impostor",
    "d0_distance",
    "query_offsets",
}


def _require_no_clip_import() -> None:
    forbidden = sorted(
        name for name in sys.modules if name == "open_clip" or name.startswith("open_clip.")
    )
    if forbidden:
        raise RuntimeError("geometry census imported OpenCLIP: " + ",".join(forbidden))


def _validate_relative_path(value: str) -> str:
    path = Path(str(value))
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "bounding_box_train"
        or path.suffix.lower() != ".jpg"
    ):
        raise RuntimeError("invalid official-train relative path: " + str(value))
    return path.as_posix()


def slot_geometry(keypoints, scores, valid, image_size):
    """Return normalized centers/spans and strict mechanical availability."""
    points = np.asarray(keypoints, dtype=np.float64)
    confidence = np.asarray(scores, dtype=np.float64)
    active = np.asarray(valid, dtype=np.bool_)
    if points.shape != (17, 2) or confidence.shape != (17,) or active.shape != (17,):
        raise ValueError("unexpected COCO-17 pose shape")
    if not np.isfinite(points).all() or not np.isfinite(confidence).all():
        raise ValueError("non-finite pose input")
    width, height = map(int, image_size)
    if width <= 1 or height <= 1:
        raise ValueError("invalid RGB size")

    centers = np.zeros((len(SLOT_NAMES), 2), dtype=np.float64)
    spans = np.zeros((len(SLOT_NAMES), 2), dtype=np.float64)
    availability = np.zeros(len(SLOT_NAMES), dtype=np.bool_)
    slot_confidence = np.zeros(len(SLOT_NAMES), dtype=np.float64)
    normalized = points.copy()
    normalized[:, 0] /= float(width - 1)
    normalized[:, 1] /= float(height - 1)
    for slot, joint_ids in enumerate(SLOT_JOINTS):
        indices = np.asarray(joint_ids, dtype=np.int64)
        selected = active[indices] & (confidence[indices] >= JOINT_SCORE_MIN)
        if int(selected.sum()) < MIN_VISIBLE_JOINTS:
            continue
        values = normalized[indices[selected]]
        centers[slot] = values.mean(axis=0)
        spans[slot] = values.max(axis=0) - values.min(axis=0)
        slot_confidence[slot] = confidence[indices[selected]].mean()
        availability[slot] = True
    if bool(((centers[availability] < 0.0) | (centers[availability] > 1.0)).any()):
        raise RuntimeError("available pose center leaves normalized canvas")
    return centers, spans, availability, slot_confidence


def freeze_crop_spec(all_centers, all_spans, availability):
    """Freeze five fixed crop sizes and canonical centers from pose only."""
    centers = np.asarray(all_centers, dtype=np.float64)
    spans = np.asarray(all_spans, dtype=np.float64)
    active = np.asarray(availability, dtype=np.bool_)
    if (
        centers.ndim != 3
        or centers.shape[1:] != (len(SLOT_NAMES), 2)
        or spans.shape != centers.shape
        or active.shape != centers.shape[:2]
    ):
        raise ValueError("unexpected census table shape")
    crop_hw = np.zeros((len(SLOT_NAMES), 2), dtype=np.int16)
    canonical_xy = np.zeros((len(SLOT_NAMES), 2), dtype=np.float64)
    for slot in range(len(SLOT_NAMES)):
        selected = active[:, slot]
        if not bool(selected.any()):
            raise RuntimeError("pose slot is globally unavailable: " + SLOT_NAMES[slot])
        canonical_xy[slot] = np.median(centers[selected, slot], axis=0)
        width_fraction = np.quantile(
            spans[selected, slot, 0] + 2.0 * CROP_PADDING_FRACTION,
            CROP_SPAN_QUANTILE,
            method="linear",
        )
        height_fraction = np.quantile(
            spans[selected, slot, 1] + 2.0 * CROP_PADDING_FRACTION,
            CROP_SPAN_QUANTILE,
            method="linear",
        )
        crop_height = min(
            CANONICAL_HEIGHT,
            max(MIN_CROP_HEIGHT, int(math.ceil(height_fraction * CANONICAL_HEIGHT))),
        )
        crop_width = min(
            CANONICAL_WIDTH,
            max(MIN_CROP_WIDTH, int(math.ceil(width_fraction * CANONICAL_WIDTH))),
        )
        crop_hw[slot] = (crop_height, crop_width)
    return crop_hw, canonical_xy


def rectangles_from_centers(centers_xy, crop_hw):
    """Clamp centers while preserving each slot's exact frozen H/W."""
    centers = np.asarray(centers_xy, dtype=np.float64)
    sizes = np.asarray(crop_hw, dtype=np.int64)
    if centers.ndim != 3 or centers.shape[1:] != (len(SLOT_NAMES), 2):
        raise ValueError("centers must have shape [N,5,2]")
    if sizes.shape != (len(SLOT_NAMES), 2):
        raise ValueError("crop_hw must have shape [5,2]")
    output = np.zeros((len(centers), len(SLOT_NAMES), 4), dtype=np.int16)
    for row in range(len(centers)):
        for slot, (height, width) in enumerate(sizes.tolist()):
            if height <= 0 or width <= 0:
                raise ValueError("crop dimensions must be positive")
            center_x = float(centers[row, slot, 0]) * (CANONICAL_WIDTH - 1)
            center_y = float(centers[row, slot, 1]) * (CANONICAL_HEIGHT - 1)
            top = int(round(center_y - (height - 1) / 2.0))
            left = int(round(center_x - (width - 1) / 2.0))
            top = min(max(top, 0), CANONICAL_HEIGHT - height)
            left = min(max(left, 0), CANONICAL_WIDTH - width)
            output[row, slot] = (top, left, height, width)
    return output


def _load_sealed_bank(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    if fuel_io.sha256_file(path) != str(expected_sha256):
        raise RuntimeError("candidate bank SHA256 mismatch")
    arrays = fuel_io.load_npz_exact(path, REQUIRED_BANK_FIELDS)
    if str(arrays["schema"].item()) != BANK_SCHEMA:
        raise RuntimeError("candidate bank schema mismatch")
    count = len(arrays["relative_paths"])
    if count != EXPECTED_TRAIN_COUNT:
        raise RuntimeError("candidate bank train count mismatch")
    paths = tuple(_validate_relative_path(value) for value in arrays["relative_paths"])
    if len(set(paths)) != count:
        raise RuntimeError("candidate bank contains duplicate paths")
    if arrays["raw_pids"].shape != (count,) or arrays["camids"].shape != (count,):
        raise RuntimeError("candidate bank identity/camera shape mismatch")
    query = arrays["query_indices"]
    candidate = arrays["candidate_indices"]
    if query.ndim != 1 or candidate.shape != query.shape or len(query) == 0:
        raise RuntimeError("candidate pair index shape mismatch")
    if (
        int(query.min()) < 0
        or int(candidate.min()) < 0
        or int(query.max()) >= count
        or int(candidate.max()) >= count
    ):
        raise RuntimeError("candidate pair index leaves train table")
    if arrays["d0_distance"].shape != query.shape:
        raise RuntimeError("candidate D0 distance shape mismatch")
    if not np.isfinite(arrays["d0_distance"]).all():
        raise RuntimeError("candidate bank has non-finite D0 distance")
    offsets = arrays["query_offsets"]
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(query):
        raise RuntimeError("candidate bank query offsets mismatch")
    _require_no_clip_import()
    return arrays, paths


def _coverage_receipt(arrays, availability):
    query = arrays["query_indices"].astype(np.int64, copy=False)
    candidate = arrays["candidate_indices"].astype(np.int64, copy=False)
    raw_pids = arrays["raw_pids"].astype(np.int64, copy=False)
    common = availability[query] & availability[candidate]
    pair_any = common.any(axis=1)
    unique_queries = np.unique(query)
    covered_queries = np.unique(query[pair_any])
    query_coverage = float(len(covered_queries) / len(unique_queries))
    pair_counts = common.sum(axis=0).astype(np.int64)
    pid_counts = []
    for slot in range(len(SLOT_NAMES)):
        slot_query = np.unique(query[common[:, slot]])
        pid_counts.append(int(len(np.unique(raw_pids[slot_query]))))
    return {
        "fixed_query_count": int(len(unique_queries)),
        "covered_query_count": int(len(covered_queries)),
        "query_coverage": query_coverage,
        "pair_count": int(len(query)),
        "pair_any_common_count": int(pair_any.sum()),
        "pair_any_common_fraction": float(pair_any.mean()),
        "common_pair_count_by_slot": pair_counts.tolist(),
        "query_pid_count_by_slot": pid_counts,
        "query_coverage_gate": bool(query_coverage >= MIN_QUERY_COVERAGE),
        "slot_pair_gates": [
            bool(value >= MIN_COMMON_PAIRS_PER_SLOT) for value in pair_counts
        ],
        "slot_pid_gates": [
            bool(value >= MIN_QUERY_PIDS_PER_SLOT) for value in pid_counts
        ],
    }


def build_census(bank_path, bank_sha256, pose_artifact, pose_manifest_sha256):
    _require_no_clip_import()
    arrays, relative_paths = _load_sealed_bank(bank_path, bank_sha256)
    if str(pose_manifest_sha256) != EXPECTED_POSE_MANIFEST_SHA256:
        raise RuntimeError("exp416 requires the sealed train pose manifest")
    from datasets.pose_targets import PoseTargetStore

    pose_store = PoseTargetStore(pose_artifact, pose_manifest_sha256)
    if len(pose_store) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError("pose store train count mismatch")
    if set(pose_store._records) != set(relative_paths):
        raise RuntimeError("candidate bank and pose store path coverage differ")

    centers = np.zeros((EXPECTED_TRAIN_COUNT, len(SLOT_NAMES), 2), np.float64)
    spans = np.zeros_like(centers)
    availability = np.zeros((EXPECTED_TRAIN_COUNT, len(SLOT_NAMES)), np.bool_)
    confidence = np.zeros_like(availability, dtype=np.float64)
    image_sha256 = []
    for row, relative_path in enumerate(relative_paths):
        image_path = (pose_store.dataset_root / relative_path).resolve()
        pose = pose_store.get(image_path, verify_image_sha=True)
        if pose.relative_path != relative_path:
            raise RuntimeError("pose path order mismatch")
        values = slot_geometry(
            pose.keypoints.numpy(),
            pose.scores.numpy(),
            pose.valid.numpy(),
            pose.image_size,
        )
        centers[row], spans[row], availability[row], confidence[row] = values
        image_sha256.append(pose.image_sha256)
        if row == 0 or (row + 1) % 1000 == 0 or row + 1 == EXPECTED_TRAIN_COUNT:
            print("geometry {}/{}".format(row + 1, EXPECTED_TRAIN_COUNT), flush=True)

    crop_hw, canonical_centers = freeze_crop_spec(centers, spans, availability)
    instance_rectangles = rectangles_from_centers(centers, crop_hw)
    tiled_canonical = np.broadcast_to(
        canonical_centers[None], centers.shape
    ).copy()
    canonical_rectangles = rectangles_from_centers(tiled_canonical, crop_hw)
    if not np.array_equal(
        instance_rectangles[:, :, 2:4], canonical_rectangles[:, :, 2:4]
    ):
        raise RuntimeError("correct/canonical crop dimensions differ")
    if not np.array_equal(
        instance_rectangles[:, :, 2:4],
        np.broadcast_to(crop_hw[None], instance_rectangles[:, :, 2:4].shape),
    ):
        raise RuntimeError("crop dimensions changed across rows")

    coverage = _coverage_receipt(arrays, availability)
    _require_no_clip_import()
    output = {
        "schema": np.asarray(SCHEMA),
        "relative_paths": np.asarray(relative_paths, dtype=np.str_),
        "image_sha256": np.asarray(image_sha256, dtype=np.str_),
        "slot_names": np.asarray(SLOT_NAMES, dtype=np.str_),
        "availability": availability,
        "slot_confidence": confidence.astype(np.float32),
        "instance_centers_xy": centers.astype(np.float32),
        "instance_rectangles": instance_rectangles,
        "canonical_centers_xy": canonical_centers.astype(np.float32),
        "canonical_rectangles": canonical_rectangles,
        "crop_hw": crop_hw,
    }
    summary = {
        "schema": SCHEMA,
        "bank_path": str(Path(bank_path).resolve()),
        "bank_sha256": str(bank_sha256),
        "pose_artifact": str(Path(pose_artifact).resolve()),
        "pose_manifest_sha256": str(pose_manifest_sha256),
        "sample_count": EXPECTED_TRAIN_COUNT,
        "slot_names": list(SLOT_NAMES),
        "mechanical_visibility": {
            "joint_score_min": JOINT_SCORE_MIN,
            "minimum_visible_joints": MIN_VISIBLE_JOINTS,
            "available_image_count_by_slot": availability.sum(axis=0).tolist(),
        },
        "crop_freeze": {
            "span_quantile": CROP_SPAN_QUANTILE,
            "padding_fraction_each_side": CROP_PADDING_FRACTION,
            "minimum_hw": [MIN_CROP_HEIGHT, MIN_CROP_WIDTH],
            "crop_hw": crop_hw.tolist(),
            "canonical_centers_xy": canonical_centers.tolist(),
        },
        "coverage_thresholds": {
            "minimum_query_coverage": MIN_QUERY_COVERAGE,
            "minimum_common_pairs_per_slot": MIN_COMMON_PAIRS_PER_SLOT,
            "minimum_query_pids_per_slot": MIN_QUERY_PIDS_PER_SLOT,
        },
        "coverage": coverage,
        "geometry_gate_pass": bool(
            coverage["query_coverage_gate"]
            and all(coverage["slot_pair_gates"])
            and all(coverage["slot_pid_gates"])
        ),
        "openclip_import_count": 0,
    }
    return output, summary


def run_self_test():
    points = np.zeros((17, 2), dtype=np.float32)
    points[:, 0] = np.linspace(5, 120, 17)
    points[:, 1] = np.linspace(10, 370, 17)
    scores = np.ones(17, dtype=np.float32)
    valid = np.ones(17, dtype=np.bool_)
    centers, spans, active, confidence = slot_geometry(
        points, scores, valid, (128, 384)
    )
    assert active.tolist() == [True] * 5
    assert np.all(confidence == 1.0)
    low = scores.copy()
    low[list(SLOT_JOINTS[2])] = JOINT_SCORE_MIN - 0.01
    _, _, low_active, _ = slot_geometry(points, low, valid, (128, 384))
    assert not bool(low_active[2])
    table_centers = np.stack((centers, np.clip(centers + 0.01, 0, 1)))
    table_spans = np.stack((spans, spans))
    table_active = np.stack((active, active))
    crop_hw, canonical = freeze_crop_spec(
        table_centers, table_spans, table_active
    )
    rectangles = rectangles_from_centers(table_centers, crop_hw)
    fixed = rectangles_from_centers(
        np.broadcast_to(canonical[None], table_centers.shape), crop_hw
    )
    assert rectangles.shape == fixed.shape == (2, 5, 4)
    assert np.array_equal(rectangles[:, :, 2:4], fixed[:, :, 2:4])
    assert int(rectangles[..., 0].min()) >= 0
    assert int((rectangles[..., 0] + rectangles[..., 2]).max()) <= CANONICAL_HEIGHT
    assert int((rectangles[..., 1] + rectangles[..., 3]).max()) <= CANONICAL_WIDTH
    print("EXP416_GEOMETRY_SELF_TEST=PASS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--bank")
    parser.add_argument("--bank-sha256")
    parser.add_argument("--pose-artifact")
    parser.add_argument("--pose-manifest-sha256")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    required = (
        "bank",
        "bank_sha256",
        "pose_artifact",
        "pose_manifest_sha256",
        "output_dir",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError("missing formal arguments: " + ",".join(missing))
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute() or output_dir.exists():
        raise FileExistsError("geometry output directory must be fresh and absolute")
    if output_dir.parent.resolve() != output_dir.resolve().parent:
        raise RuntimeError("geometry output parent is not canonical")
    output_dir.mkdir(mode=0o755, parents=False)
    arrays, summary = build_census(
        args.bank,
        args.bank_sha256,
        args.pose_artifact,
        args.pose_manifest_sha256,
    )
    cache_path = output_dir / "geometry.npz"
    summary_path = output_dir / "summary.json"
    fuel_io.atomic_npz(cache_path, arrays)
    fuel_io.readback_npz(cache_path, arrays)
    summary["geometry_npz"] = str(cache_path)
    summary["geometry_npz_sha256"] = fuel_io.sha256_file(cache_path)
    fuel_io.atomic_json(summary_path, summary)
    fuel_io.readback_json(summary_path, summary)
    manifest = {
        "schema": SCHEMA,
        "geometry_npz": str(cache_path),
        "geometry_npz_sha256": fuel_io.sha256_file(cache_path),
        "summary_json": str(summary_path),
        "summary_json_sha256": fuel_io.sha256_file(summary_path),
        "source_files": {
            "geometry_census.py": fuel_io.sha256_file(Path(__file__).resolve()),
            "fuel_io.py": fuel_io.sha256_file(SCRIPT_DIR / "fuel_io.py"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    fuel_io.atomic_json(manifest_path, manifest)
    fuel_io.readback_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
