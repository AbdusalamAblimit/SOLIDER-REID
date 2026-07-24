#!/usr/bin/env python3
"""Read-only full-train pose geometry census for exp415 PACIT.

This program intentionally does not import or load CLIP, compute a selector,
evaluate colors, calculate Y, or access CUDA.  It verifies the frozen pose
artifact and every official Occluded-Duke training image before measuring only
the proposal geometry defined in ``asset_oracle_core.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
for import_root in (REPOSITORY_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import asset_oracle_core as core


SCHEMA_NAME = "exp415_pacit_geometry_census"
SCHEMA_VERSION = 1
EXPERIMENT = "exp415_pose_anatomical_clip_intervention"
EXPECTED_DATASET = "occluded_duke"
EXPECTED_SPLIT = "train"
EXPECTED_SAMPLE_COUNT = 15618
EXPECTED_POSE_MANIFEST_SHA256 = (
    "cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8"
)
TRAIN_ANCHOR_SALT = "exp415-pacit-train-v3"
ORIENTATIONS = ("canonical", "hflip")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(payload):
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def ordered_fields_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\xff")
    return digest.hexdigest()


def train_active_anchor(relative_path):
    digest = hashlib.sha256(
        (TRAIN_ANCHOR_SALT + "\0" + str(relative_path)).encode("utf-8")
    ).hexdigest()
    return int(int(digest, 16) % core.ANCHOR_COUNT)


def resize_pose_to_canonical(pose):
    width, height = map(int, pose.image_size)
    if width <= 0 or height <= 0:
        raise RuntimeError("pose image size must be positive")
    keypoints = pose.keypoints.clone().float()
    keypoints[:, 0] *= core.IMAGE_WIDTH / float(width)
    keypoints[:, 1] *= core.IMAGE_HEIGHT / float(height)
    valid = pose.valid.clone().bool()
    if keypoints.shape != (17, 2) or valid.shape != (17,):
        raise RuntimeError("unexpected resized pose shape")
    if not bool(torch.isfinite(keypoints).all()):
        raise RuntimeError("non-finite resized pose")
    return keypoints, valid


def read_rgb_size(image_path):
    # Pillow is part of the fixed remote runtime but is deliberately imported
    # only for a real census, so the pure geometry self-test has no RGB
    # dependency.
    from PIL import Image

    with Image.open(image_path) as image:
        return tuple(map(int, image.size))


def hflip_canonical_pose(keypoints, valid, flip_index):
    if keypoints.shape != (17, 2) or valid.shape != (17,):
        raise ValueError("unexpected canonical pose shape")
    flip_index = torch.as_tensor(flip_index, dtype=torch.long)
    if flip_index.shape != (17,) or sorted(flip_index.tolist()) != list(
        range(17)
    ):
        raise ValueError("COCO17 flip index must be a permutation")
    flipped_keypoints = keypoints.clone()
    flipped_keypoints[:, 0] = (
        core.IMAGE_WIDTH - 1 - flipped_keypoints[:, 0]
    )
    flipped_keypoints = flipped_keypoints.index_select(
        0, flip_index
    )
    flipped_valid = valid.clone().bool().index_select(
        0, flip_index
    )
    if not bool(torch.isfinite(flipped_keypoints).all()):
        raise RuntimeError("non-finite hflip pose")
    return flipped_keypoints, flipped_valid


def aspect_specs():
    target_area = int(
        round(core.AREA_FRACTION * core.IMAGE_HEIGHT * core.IMAGE_WIDTH)
    )
    specs = []
    for aspect_index, aspect in enumerate(core.ASPECT_RATIOS):
        height, width = core._best_dimensions(target_area, aspect)
        area = int(height * width)
        absolute_error = int(abs(area - target_area))
        relative_error = float(absolute_error / float(target_area))
        if relative_error > core.AREA_RELATIVE_TOLERANCE:
            raise RuntimeError("frozen aspect area is unreachable")
        specs.append(
            {
                "aspect_index": int(aspect_index),
                "aspect": float(aspect),
                "height": int(height),
                "width": int(width),
                "area_pixels": area,
                "absolute_error_pixels": absolute_error,
                "relative_error": relative_error,
            }
        )
    return target_area, specs


def validate_proposal(proposal, expected_candidate_index, specs):
    candidate_index = int(proposal["candidate_index"])
    if candidate_index != int(expected_candidate_index):
        raise RuntimeError("proposal candidate index/order mismatch")
    anchor_index = candidate_index // len(core.ASPECT_RATIOS)
    aspect_index = candidate_index % len(core.ASPECT_RATIOS)
    if int(proposal["anchor_index"]) != anchor_index:
        raise RuntimeError("proposal anchor index mismatch")
    if int(proposal["aspect_index"]) != aspect_index:
        raise RuntimeError("proposal aspect index mismatch")

    spec = specs[aspect_index]
    if float(proposal["aspect"]) != spec["aspect"]:
        raise RuntimeError("proposal aspect value mismatch")
    if (
        int(proposal["height"]) != spec["height"]
        or int(proposal["width"]) != spec["width"]
    ):
        raise RuntimeError("proposal dimensions differ from frozen aspect")

    top = int(proposal["top"])
    left = int(proposal["left"])
    height = int(proposal["height"])
    width = int(proposal["width"])
    if not (
        0 <= top <= core.IMAGE_HEIGHT - height
        and 0 <= left <= core.IMAGE_WIDTH - width
    ):
        raise RuntimeError("proposal rectangle leaves canonical canvas")

    mask = proposal["mask"]
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or tuple(mask.shape) != (core.IMAGE_HEIGHT, core.IMAGE_WIDTH)
    ):
        raise RuntimeError("unexpected proposal mask")
    canonical_area = int(mask.sum())
    if canonical_area != height * width or not bool(
        mask[top : top + height, left : left + width].all()
    ):
        raise RuntimeError("proposal mask is not its declared rectangle")
    if (
        canonical_area != int(proposal["area_pixels"])
        or canonical_area != spec["area_pixels"]
    ):
        raise RuntimeError("canonical proposal area mismatch")
    if not SHA256_PATTERN.fullmatch(str(proposal["mask_sha256"])):
        raise RuntimeError("invalid proposal mask SHA256")
    observed_mask_sha = core.sha256_bytes(
        mask.numpy().astype(np.uint8, copy=False).tobytes()
    )
    if observed_mask_sha != proposal["mask_sha256"]:
        raise RuntimeError("proposal mask SHA256 mismatch")

    centroid_y = float(proposal["centroid_y"])
    centroid_x = float(proposal["centroid_x"])
    if (
        not np.isfinite(centroid_y)
        or not np.isfinite(centroid_x)
        or not 0.0 <= centroid_y <= 1.0
        or not 0.0 <= centroid_x <= 1.0
    ):
        raise RuntimeError("invalid normalized proposal centroid")
    return aspect_index, canonical_area


def validate_complete_pool(proposals, specs):
    if len(proposals) != core.PROPOSALS_PER_POOL:
        raise RuntimeError("complete proposal pool must contain 35 rows")
    canonical_areas = []
    aspect_counts = [0] * len(core.ASPECT_RATIOS)
    for candidate_index, proposal in enumerate(proposals):
        aspect_index, canonical_area = validate_proposal(
            proposal, candidate_index, specs
        )
        aspect_counts[aspect_index] += 1
        canonical_areas.append(canonical_area)
    if aspect_counts != [core.ANCHOR_COUNT] * len(core.ASPECT_RATIOS):
        raise RuntimeError("complete pool aspect count mismatch")
    return canonical_areas, aspect_counts


def validate_direct_hflip_areas(proposals):
    areas = []
    for proposal in proposals:
        mirrored = torch.flip(proposal["mask"], dims=(1,))
        area = int(mirrored.sum())
        if area != int(proposal["area_pixels"]):
            raise RuntimeError("direct hflip proposal area mismatch")
        areas.append(area)
    return areas


class AreaAccumulator:
    def __init__(self, target_area):
        self.target_area = int(target_area)
        self.count = 0
        self.reachable = 0
        self.minimum = None
        self.maximum = None
        self.max_absolute_error = 0
        self.max_relative_error = 0.0

    def add(self, area):
        area = int(area)
        absolute_error = int(abs(area - self.target_area))
        relative_error = float(absolute_error / float(self.target_area))
        self.count += 1
        if relative_error <= core.AREA_RELATIVE_TOLERANCE:
            self.reachable += 1
        self.minimum = area if self.minimum is None else min(self.minimum, area)
        self.maximum = area if self.maximum is None else max(self.maximum, area)
        self.max_absolute_error = max(self.max_absolute_error, absolute_error)
        self.max_relative_error = max(self.max_relative_error, relative_error)

    def add_repeated(self, area, count):
        area = int(area)
        count = int(count)
        if count < 0:
            raise ValueError("area repetition count must be non-negative")
        if count == 0:
            return
        absolute_error = int(abs(area - self.target_area))
        relative_error = float(absolute_error / float(self.target_area))
        self.count += count
        if relative_error <= core.AREA_RELATIVE_TOLERANCE:
            self.reachable += count
        self.minimum = area if self.minimum is None else min(self.minimum, area)
        self.maximum = area if self.maximum is None else max(self.maximum, area)
        self.max_absolute_error = max(self.max_absolute_error, absolute_error)
        self.max_relative_error = max(self.max_relative_error, relative_error)

    def payload(self):
        if self.count <= 0:
            raise RuntimeError("empty area accumulator")
        return {
            "proposal_count": int(self.count),
            "reachable_count": int(self.reachable),
            "unreachable_count": int(self.count - self.reachable),
            "minimum_area_pixels": int(self.minimum),
            "maximum_area_pixels": int(self.maximum),
            "maximum_absolute_error_pixels": int(self.max_absolute_error),
            "maximum_relative_error": float(self.max_relative_error),
            "pass": bool(self.reachable == self.count),
        }


def validate_inputs(data_root, pose_artifact, pose_manifest_sha256):
    from datasets.occluded_duke import OccludedDuke
    from datasets.pose_targets import PoseTargetStore

    if str(pose_manifest_sha256) != EXPECTED_POSE_MANIFEST_SHA256:
        raise RuntimeError("exp415 requires the frozen pose manifest SHA256")
    data_root = Path(data_root).expanduser().resolve()
    pose_artifact = Path(pose_artifact).expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    if not pose_artifact.is_dir():
        raise NotADirectoryError(pose_artifact)

    dataset = OccludedDuke(root=str(data_root), verbose=False)
    records = list(dataset.train)
    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("official train sample count mismatch")

    pose_store = PoseTargetStore(
        pose_artifact, str(pose_manifest_sha256)
    )
    if len(pose_store) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("pose artifact sample count mismatch")
    official_dataset_root = Path(dataset.dataset_dir).resolve()
    official_train_root = Path(dataset.train_dir).resolve()
    if pose_store.dataset_root != official_dataset_root:
        raise RuntimeError("pose dataset root is not official Occluded-Duke")
    if pose_store.image_root != official_train_root:
        raise RuntimeError("pose image root is not official train split")

    relative_paths = []
    for image_path, _, _, _ in records:
        resolved = Path(image_path).resolve()
        try:
            relative = resolved.relative_to(pose_store.dataset_root).as_posix()
        except ValueError as error:
            raise RuntimeError("official image escapes pose dataset root") from error
        if resolved != (pose_store.dataset_root / relative).resolve():
            raise RuntimeError("official path normalization mismatch")
        if not relative.startswith("bounding_box_train/"):
            raise RuntimeError("non-train path in official records")
        relative_paths.append(relative)
    if len(set(relative_paths)) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("official train relative paths are not unique")

    pose_paths = list(pose_store._records)
    if set(pose_paths) != set(relative_paths):
        raise RuntimeError(
            "official train and pose artifact path coverage differ"
        )
    if pose_store.manifest.get("sample_count") != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("pose manifest sample_count mismatch")
    if not SHA256_PATTERN.fullmatch(
        str(pose_store.manifest.get("records_manifest_sha256", ""))
    ):
        raise RuntimeError("invalid pose records manifest digest")
    return dataset, records, pose_store, relative_paths


def build_census(data_root, pose_artifact, pose_manifest_sha256):
    from datasets.paired_pose_transform import COCO17_FLIP_INDEX
    from model.pose_clip_relation import REGION_NAMES

    region_names = tuple(REGION_NAMES)
    if len(region_names) != core.ANCHOR_COUNT:
        raise RuntimeError("frozen region-name count mismatch")
    (
        dataset,
        records,
        pose_store,
        relative_paths,
    ) = validate_inputs(data_root, pose_artifact, pose_manifest_sha256)
    target_area, specs = aspect_specs()
    fixed = core.generate_fixed_proposals()
    fixed_canonical_areas, fixed_aspect_template_counts = (
        validate_complete_pool(fixed, specs)
    )
    fixed_hflip_areas = validate_direct_hflip_areas(fixed)

    pose_aspect_counts = [0] * len(specs)
    pose_active_aspect_counts = [0] * len(specs)
    anchor_valid_counts = [0] * core.ANCHOR_COUNT
    active_assignment_counts = [0] * core.ANCHOR_COUNT
    active_valid_counts = [0] * core.ANCHOR_COUNT
    images_all_anchors_valid = 0
    images_any_anchor_valid = 0
    image_hash_verified = 0
    pose_canonical_area = AreaAccumulator(target_area)
    pose_hflip_area = AreaAccumulator(target_area)
    image_bindings = []

    fixed_by_candidate = {
        int(proposal["candidate_index"]): proposal for proposal in fixed
    }
    for record_index, (
        (image_path, _, _, _),
        relative_path,
    ) in enumerate(zip(records, relative_paths)):
        image_path = Path(image_path).resolve()
        pose = pose_store.get(image_path, verify_image_sha=True)
        if pose.relative_path != relative_path:
            raise RuntimeError("pose record returned an unexpected path")
        if not SHA256_PATTERN.fullmatch(str(pose.image_sha256)):
            raise RuntimeError("invalid RGB SHA256 in pose record")
        actual_size = read_rgb_size(image_path)
        if actual_size != tuple(map(int, pose.image_size)):
            raise RuntimeError("RGB dimensions differ from pose manifest")
        image_hash_verified += 1
        image_bindings.append(
            (
                relative_path,
                pose.image_sha256,
                actual_size[0],
                actual_size[1],
            )
        )

        keypoints, valid = resize_pose_to_canonical(pose)
        proposals, fields, region_valid = core.generate_pose_proposals(
            keypoints, valid
        )
        hflip_keypoints, hflip_valid = hflip_canonical_pose(
            keypoints, valid, COCO17_FLIP_INDEX
        )
        (
            hflip_proposals,
            hflip_fields,
            hflip_region_valid,
        ) = core.generate_pose_proposals(hflip_keypoints, hflip_valid)
        if (
            tuple(fields.shape)
            != (core.ANCHOR_COUNT, core.IMAGE_HEIGHT, core.IMAGE_WIDTH)
            or tuple(region_valid.shape) != (core.ANCHOR_COUNT,)
            or region_valid.dtype != torch.bool
            or not bool(torch.isfinite(fields).all())
            or tuple(hflip_fields.shape)
            != (core.ANCHOR_COUNT, core.IMAGE_HEIGHT, core.IMAGE_WIDTH)
            or tuple(hflip_region_valid.shape) != (core.ANCHOR_COUNT,)
            or hflip_region_valid.dtype != torch.bool
            or not bool(torch.isfinite(hflip_fields).all())
        ):
            raise RuntimeError("unexpected rendered pose field geometry")
        if not torch.equal(region_valid, hflip_region_valid):
            raise RuntimeError("COCO17 hflip changed anchor validity")

        canonical_areas, aspect_counts = validate_complete_pool(
            proposals, specs
        )
        generated_hflip_areas, hflip_aspect_counts = validate_complete_pool(
            hflip_proposals, specs
        )
        if hflip_aspect_counts != aspect_counts:
            raise RuntimeError("hflip proposal aspect count mismatch")
        direct_hflip_areas = validate_direct_hflip_areas(proposals)
        if direct_hflip_areas != generated_hflip_areas:
            raise RuntimeError("hflip proposal area differs after COCO17 reorder")
        for aspect_index, count in enumerate(aspect_counts):
            pose_aspect_counts[aspect_index] += int(count)
            pose_active_aspect_counts[aspect_index] += 1
        for area in canonical_areas:
            pose_canonical_area.add(area)
        for area in direct_hflip_areas:
            pose_hflip_area.add(area)

        proposal_valid = []
        for anchor_index in range(core.ANCHOR_COUNT):
            rows = proposals[
                anchor_index
                * len(core.ASPECT_RATIOS) : (anchor_index + 1)
                * len(core.ASPECT_RATIOS)
            ]
            flags = {bool(row["anchor_valid"]) for row in rows}
            if len(flags) != 1:
                raise RuntimeError("anchor validity changes across aspects")
            observed_valid = flags.pop()
            if observed_valid != bool(region_valid[anchor_index]):
                raise RuntimeError("proposal/pose-field anchor validity mismatch")
            hflip_rows = hflip_proposals[
                anchor_index
                * len(core.ASPECT_RATIOS) : (anchor_index + 1)
                * len(core.ASPECT_RATIOS)
            ]
            hflip_flags = {bool(row["anchor_valid"]) for row in hflip_rows}
            if hflip_flags != {observed_valid}:
                raise RuntimeError("hflip proposal anchor validity mismatch")
            proposal_valid.append(observed_valid)
            if observed_valid:
                anchor_valid_counts[anchor_index] += 1
            else:
                for row in rows:
                    fixed_row = fixed_by_candidate[int(row["candidate_index"])]
                    if row["mask_sha256"] != fixed_row["mask_sha256"]:
                        raise RuntimeError(
                            "invalid pose anchor did not use fixed fallback"
                        )

        images_all_anchors_valid += int(all(proposal_valid))
        images_any_anchor_valid += int(any(proposal_valid))
        active_anchor = train_active_anchor(relative_path)
        active_assignment_counts[active_anchor] += 1
        active_valid_counts[active_anchor] += int(
            proposal_valid[active_anchor]
        )

        if (record_index + 1) % 1000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "geometry_census",
                        "processed": int(record_index + 1),
                        "total": EXPECTED_SAMPLE_COUNT,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if image_hash_verified != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("not every official RGB hash was verified")
    expected_full_aspect_count = EXPECTED_SAMPLE_COUNT * core.ANCHOR_COUNT
    expected_active_aspect_count = EXPECTED_SAMPLE_COUNT
    if pose_aspect_counts != [expected_full_aspect_count] * len(specs):
        raise RuntimeError("pose proposal count by aspect mismatch")
    if pose_active_aspect_counts != [expected_active_aspect_count] * len(specs):
        raise RuntimeError("active pose proposal count by aspect mismatch")
    if sum(active_assignment_counts) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("active anchor assignment count mismatch")

    fixed_canonical_area = AreaAccumulator(target_area)
    fixed_hflip_area = AreaAccumulator(target_area)
    for area in fixed_canonical_areas:
        fixed_canonical_area.add_repeated(area, EXPECTED_SAMPLE_COUNT)
    for area in fixed_hflip_areas:
        fixed_hflip_area.add_repeated(area, EXPECTED_SAMPLE_COUNT)

    anchor_rows = []
    active_anchor_rows = []
    for anchor_index, anchor_name in enumerate(region_names):
        valid_count = int(anchor_valid_counts[anchor_index])
        assignment_count = int(active_assignment_counts[anchor_index])
        assigned_valid = int(active_valid_counts[anchor_index])
        anchor_rows.append(
            {
                "anchor_index": int(anchor_index),
                "anchor_name": str(anchor_name),
                "valid_count": valid_count,
                "invalid_count": int(EXPECTED_SAMPLE_COUNT - valid_count),
                "valid_rate": float(valid_count / EXPECTED_SAMPLE_COUNT),
            }
        )
        active_anchor_rows.append(
            {
                "anchor_index": int(anchor_index),
                "anchor_name": str(anchor_name),
                "assigned_count": assignment_count,
                "valid_count": assigned_valid,
                "invalid_count": int(assignment_count - assigned_valid),
                "valid_rate": float(
                    assigned_valid / assignment_count
                    if assignment_count
                    else 0.0
                ),
            }
        )

    aspect_rows = []
    for spec in specs:
        aspect_index = int(spec["aspect_index"])
        fixed_template_count = int(
            fixed_aspect_template_counts[aspect_index]
        )
        fixed_full_count = int(
            EXPECTED_SAMPLE_COUNT * fixed_template_count
        )
        aspect_rows.append(
            {
                **spec,
                "pose_instance_full_pool_count": int(
                    pose_aspect_counts[aspect_index]
                ),
                "pose_instance_active_pool_count": int(
                    pose_active_aspect_counts[aspect_index]
                ),
                "canonical_anchor_full_pool_count": fixed_full_count,
                "canonical_anchor_active_pool_count": EXPECTED_SAMPLE_COUNT,
                "canonical_reachable": True,
                "hflip_reachable": True,
            }
        )

    official_order_digest = core.ordered_digest(relative_paths)
    pose_order_digest = core.ordered_digest(list(pose_store._records))
    area_payload = {
        "pose_instance": {
            "canonical": pose_canonical_area.payload(),
            "hflip": pose_hflip_area.payload(),
        },
        "canonical_anchor": {
            "canonical": fixed_canonical_area.payload(),
            "hflip": fixed_hflip_area.payload(),
        },
    }
    if not all(
        orientation["pass"]
        for pool in area_payload.values()
        for orientation in pool.values()
    ):
        raise RuntimeError("canonical/hflip area reachability failed")

    payload = {
        "schema": {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
        },
        "status": "PASS",
        "experiment": EXPERIMENT,
        "scope": {
            "dataset": EXPECTED_DATASET,
            "split": EXPECTED_SPLIT,
            "expected_sample_count": EXPECTED_SAMPLE_COUNT,
            "observed_sample_count": len(records),
            "canonical_image_height": core.IMAGE_HEIGHT,
            "canonical_image_width": core.IMAGE_WIDTH,
            "field_height": core.FIELD_HEIGHT,
            "field_width": core.FIELD_WIDTH,
            "orientations": list(ORIENTATIONS),
        },
        "contract": {
            "anchor_names": list(region_names),
            "anchor_count": core.ANCHOR_COUNT,
            "aspect_ratios": [float(value) for value in core.ASPECT_RATIOS],
            "proposals_per_pool_per_image": core.PROPOSALS_PER_POOL,
            "active_proposals_per_image": core.ACTIVE_PROPOSALS_PER_IMAGE,
            "area_fraction": core.AREA_FRACTION,
            "area_relative_tolerance": core.AREA_RELATIVE_TOLERANCE,
            "target_area_pixels": target_area,
            "train_anchor_salt": TRAIN_ANCHOR_SALT,
            "coco17_flip_index": [
                int(value) for value in COCO17_FLIP_INDEX.tolist()
            ],
            "pose_valid_definition": (
                "finite keypoints and scores with keypoints inside original "
                "image bounds; no confidence threshold"
            ),
            "pose_confidence_threshold": None,
            "clip_loaded": False,
            "scientific_y_computed": False,
            "cuda_used": False,
        },
        "validation": {
            "pose_manifest_sha256_exact": True,
            "pose_manifest_schema_exact": True,
            "pose_shard_sha256_exact": True,
            "pose_records_manifest_sha256_exact": True,
            "official_train_count_exact": True,
            "official_pose_path_set_coverage_exact": True,
            "official_pose_path_unique": True,
            "rgb_sha256_verified_count": image_hash_verified,
            "rgb_size_verified_count": image_hash_verified,
            "all_pose_fields_finite": True,
            "all_proposal_masks_sha256_exact": True,
            "all_invalid_anchors_use_fixed_fallback": True,
            "coco17_hflip_anchor_validity_exact": True,
            "coco17_hflip_rerender_area_reachable": True,
            "production_direct_hflip_area_reachable": True,
        },
        "provenance": {
            "data_root": str(Path(data_root).expanduser().resolve()),
            "official_dataset_root": str(Path(dataset.dataset_dir).resolve()),
            "official_train_root": str(Path(dataset.train_dir).resolve()),
            "pose_artifact_root": str(Path(pose_artifact).expanduser().resolve()),
            "pose_manifest_sha256": pose_store.manifest_sha256,
            "pose_records_manifest_sha256": pose_store.manifest[
                "records_manifest_sha256"
            ],
            "pose_shard_count": len(pose_store.manifest.get("shards", [])),
            "official_relative_path_order_sha256": official_order_digest,
            "pose_relative_path_order_sha256": pose_order_digest,
            "rgb_binding_order_sha256": ordered_fields_digest(image_bindings),
            "geometry_census_source_sha256": sha256_file(Path(__file__)),
            "asset_oracle_core_source_sha256": sha256_file(
                Path(core.__file__).resolve()
            ),
            "pose_targets_source_sha256": sha256_file(
                REPOSITORY_ROOT / "datasets" / "pose_targets.py"
            ),
            "paired_pose_transform_source_sha256": sha256_file(
                REPOSITORY_ROOT / "datasets" / "paired_pose_transform.py"
            ),
        },
        "anchor_validity": {
            "by_anchor": anchor_rows,
            "active_assignment_by_anchor": active_anchor_rows,
            "images_all_anchors_valid_count": int(images_all_anchors_valid),
            "images_any_anchor_valid_count": int(images_any_anchor_valid),
        },
        "proposal_counts_by_aspect": aspect_rows,
        "area_reachability": area_payload,
    }
    return payload


def write_fresh_json(output, payload, forbidden_roots):
    output = Path(output).expanduser().resolve()
    for root in forbidden_roots:
        root = Path(root).expanduser().resolve()
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise RuntimeError("geometry census output cannot enter a read-only root")
    if output.exists():
        raise FileExistsError("geometry census output must be fresh")
    if not output.parent.is_dir():
        raise NotADirectoryError(output.parent)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError("geometry census temporary output is not fresh")
    data = stable_json_bytes(payload)
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return output, sha256_file(output)


def run_self_test():
    target_area, specs = aspect_specs()
    fixed = core.generate_fixed_proposals()
    canonical, counts = validate_complete_pool(fixed, specs)
    hflip = validate_direct_hflip_areas(fixed)
    if len(canonical) != core.PROPOSALS_PER_POOL:
        raise AssertionError("self-test fixed proposal count failed")
    if canonical != hflip:
        raise AssertionError("self-test hflip area preservation failed")
    if counts != [core.ANCHOR_COUNT] * len(core.ASPECT_RATIOS):
        raise AssertionError("self-test aspect count failed")

    keypoints = torch.tensor(
        [
            [64, 24],
            [58, 28],
            [70, 28],
            [54, 34],
            [74, 34],
            [48, 88],
            [80, 88],
            [42, 132],
            [86, 132],
            [38, 174],
            [90, 174],
            [52, 202],
            [76, 202],
            [50, 272],
            [78, 272],
            [48, 350],
            [80, 350],
        ],
        dtype=torch.float32,
    )
    self_test_flip_index = torch.tensor(
        [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        dtype=torch.long,
    )

    def synthetic_renderer(unused_keypoints, unused_valid):
        del unused_keypoints, unused_valid
        fields = torch.zeros(
            core.ANCHOR_COUNT, core.IMAGE_HEIGHT, core.IMAGE_WIDTH
        )
        bands = ((5, 70), (75, 150), (155, 225), (230, 310), (315, 383))
        for anchor_index, (top, bottom) in enumerate(bands):
            fields[anchor_index, top:bottom, 20:108] = 1.0
        return fields, torch.ones(core.ANCHOR_COUNT, dtype=torch.bool)

    original_renderer = core.render_full_pose_fields
    core.render_full_pose_fields = synthetic_renderer
    try:
        proposals, fields, region_valid = core.generate_pose_proposals(
            keypoints, torch.ones(17, dtype=torch.bool)
        )
        validate_complete_pool(proposals, specs)
        hflip_keypoints, hflip_valid = hflip_canonical_pose(
            keypoints,
            torch.ones(17, dtype=torch.bool),
            self_test_flip_index,
        )
        hflip_proposals, hflip_fields, hflip_region_valid = (
            core.generate_pose_proposals(hflip_keypoints, hflip_valid)
        )
        hflip_generated_areas, _ = validate_complete_pool(
            hflip_proposals, specs
        )
        if hflip_generated_areas != validate_direct_hflip_areas(proposals):
            raise AssertionError("self-test hflip area preservation failed")
        if not bool(region_valid.all()) or not bool(
            torch.isfinite(fields).all()
        ):
            raise AssertionError("self-test pose geometry failed")
        if (
            not torch.equal(region_valid, hflip_region_valid)
            or not bool(torch.isfinite(hflip_fields).all())
        ):
            raise AssertionError("self-test hflip pose geometry failed")
    finally:
        core.render_full_pose_fields = original_renderer
    if train_active_anchor("bounding_box_train/0001_c1_f1.jpg") not in range(
        core.ANCHOR_COUNT
    ):
        raise AssertionError("self-test active anchor hash failed")
    if target_area != int(
        round(core.AREA_FRACTION * core.IMAGE_HEIGHT * core.IMAGE_WIDTH)
    ):
        raise AssertionError("self-test target area failed")
    if "open_clip" in sys.modules:
        raise AssertionError("geometry census imported CLIP")
    probe = {"schema": SCHEMA_NAME, "status": "PASS"}
    if stable_json_bytes(probe) != stable_json_bytes(probe):
        raise AssertionError("self-test JSON stability failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "self_test": SCHEMA_NAME,
                "clip_loaded": False,
                "cuda_used": False,
            },
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the read-only exp415 full-train geometry census."
    )
    parser.add_argument("--data-root")
    parser.add_argument("--pose-artifact")
    parser.add_argument(
        "--pose-manifest-sha256",
        default=EXPECTED_POSE_MANIFEST_SHA256,
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a local synthetic CPU-only contract without dataset access.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    missing = [
        name
        for name, value in (
            ("--data-root", args.data_root),
            ("--pose-artifact", args.pose_artifact),
            ("--output", args.output),
        )
        if not value
    ]
    if missing:
        raise ValueError("missing required arguments: {}".format(", ".join(missing)))
    payload = build_census(
        args.data_root,
        args.pose_artifact,
        args.pose_manifest_sha256,
    )
    output, output_sha256 = write_fresh_json(
        args.output,
        payload,
        forbidden_roots=(args.data_root, args.pose_artifact),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "output_sha256": output_sha256,
                "sample_count": EXPECTED_SAMPLE_COUNT,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
