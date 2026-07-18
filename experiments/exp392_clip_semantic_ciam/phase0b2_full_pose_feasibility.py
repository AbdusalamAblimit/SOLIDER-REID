#!/usr/bin/env python3
"""Full-split pose-only feasibility for exp392 Phase 0B2-S.

This audit freezes the target, donor, and connected-occlusion geometry maps
before any CLIP smoke.  It reads only the official train split and the strict
exp386 pose artifact; it never loads CLIP/ReID, creates an optimizer, or uses
CUDA.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REGIONS = 5
HEIGHT = 384
WIDTH = 128
MASK_HEIGHT = 96
MASK_WIDTH = 32
SUPPORT_THRESHOLD = 0.05
EXPECTED_SAMPLES = 15618
EXPECTED_PHASE0B_SCRIPT_SHA256 = (
    "03b8f707bc6f189dd3de34505af82e63f7ee71bd23d70b6e9663aee318afcd70"
)
EXPECTED_ONTOLOGY_SCRIPT_SHA256 = (
    "b0d5ce6a53e94d09fa5d15c338392ea31437eee036299256c424ed30489028ca"
)
EXPECTED_STATIC_SCRIPT_SHA256 = (
    "a9fc32a68a0dc13645e8e45a43fe84f0a5174bc7eb997c658a5b06c709cb1e1f"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "p50": None, "p95": None}
    return {
        "count": int(values.size),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def unpack_support(packed, index, slot):
    flat = np.unpackbits(
        packed[int(index), int(slot)], count=MASK_HEIGHT * MASK_WIDTH
    )
    return flat.reshape(MASK_HEIGHT, MASK_WIDTH).astype(bool, copy=False)


def full_support(packed, index, slot):
    coarse = torch.from_numpy(unpack_support(packed, index, slot))
    return coarse.repeat_interleave(4, 0).repeat_interleave(4, 1)


class PoseGeometryTransform:
    """Reproduce the frozen path-hash RGB geometry without reading RGB."""

    def __init__(self, base, ontology, seed):
        self.base = base
        self.seed = int(seed)
        self.renderer = ontology.ExclusiveRegionRenderer(
            base, partition_mode="hard-owner"
        )

    def __call__(self, pose):
        keypoints = pose.keypoints.clone().float()
        scores = pose.scores.clone().float()
        valid = pose.valid.clone().bool()
        original_width, original_height = pose.image_size
        keypoints[:, 0] *= WIDTH / float(original_width)
        keypoints[:, 1] *= HEIGHT / float(original_height)

        local_seed = self.base.seed_from_text(pose.relative_path, self.seed)
        generator = torch.Generator().manual_seed(local_seed)
        flipped = bool(torch.rand(1, generator=generator).item() < 0.5)
        if flipped:
            keypoints[:, 0] = (WIDTH - 1) - keypoints[:, 0]
            index = torch.as_tensor(
                self.base.COCO17_FLIP, dtype=torch.long
            )
            keypoints = keypoints.index_select(0, index)
            scores = scores.index_select(0, index)
            valid = valid.index_select(0, index)

        keypoints += 10.0
        crop_top = int(torch.randint(
            0, 21, (1,), generator=generator
        ).item())
        crop_left = int(torch.randint(
            0, 21, (1,), generator=generator
        ).item())
        keypoints[:, 0] -= float(crop_left)
        keypoints[:, 1] -= float(crop_top)
        valid = (
            valid
            & (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] <= WIDTH - 1)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] <= HEIGHT - 1)
        )
        masks, confidence, region_valid = self.renderer(
            keypoints, scores, valid
        )
        supports = torch.zeros_like(masks, dtype=torch.bool)
        effective_valid = region_valid.clone().bool()
        for slot in range(REGIONS):
            maximum = float(masks[slot].max())
            if not bool(effective_valid[slot]) or maximum <= 0.0:
                effective_valid[slot] = False
                continue
            supports[slot] = masks[slot] > (
                SUPPORT_THRESHOLD * maximum
            )
            if not bool(supports[slot].any()):
                effective_valid[slot] = False
        return {
            "supports": supports,
            "confidence": confidence.float(),
            "valid": effective_valid,
            "flipped": flipped,
            "crop_top": crop_top,
            "crop_left": crop_left,
        }


def build_pose_table(records, pose_store, transform):
    count = len(records)
    packed_width = (MASK_HEIGHT * MASK_WIDTH + 7) // 8
    packed = np.zeros((count, REGIONS, packed_width), dtype=np.uint8)
    valid = np.zeros((count, REGIONS), dtype=bool)
    area = np.zeros((count, REGIONS), dtype=np.float64)
    y_center = np.zeros((count, REGIONS), dtype=np.float64)
    confidence = np.zeros((count, REGIONS), dtype=np.float64)
    paths = []
    image_sha = []
    geometry = []
    manifest = hashlib.sha256()
    pairwise_support_product_max = 0
    full_y = (
        np.arange(MASK_HEIGHT, dtype=np.float64)[:, None] * 4.0 + 1.5
    ) / float(HEIGHT - 1)

    for index, row in enumerate(records):
        pose = pose_store.get(row[0], verify_image_sha=False)
        item = transform(pose)
        relative_path = str(pose.relative_path)
        paths.append(relative_path)
        image_sha.append(str(pose.image_sha256))
        geometry.append({
            "flipped": bool(item["flipped"]),
            "crop_top": int(item["crop_top"]),
            "crop_left": int(item["crop_left"]),
        })
        manifest.update(relative_path.encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(str(pose.image_sha256).encode("ascii"))

        for slot in range(REGIONS):
            support = item["supports"][slot].numpy().astype(bool)
            packed[index, slot] = np.packbits(support.reshape(-1))
            valid[index, slot] = bool(item["valid"][slot])
            confidence[index, slot] = float(item["confidence"][slot])
            pixels = int(support.sum())
            if not valid[index, slot] or pixels == 0:
                valid[index, slot] = False
                continue
            area[index, slot] = pixels * 16.0 / float(HEIGHT * WIDTH)
            y_center[index, slot] = float(
                (support * full_y).sum() / pixels
            )
        for left in range(REGIONS):
            for right in range(left + 1, REGIONS):
                pairwise_support_product_max = max(
                    pairwise_support_product_max,
                    int((
                        item["supports"][left]
                        & item["supports"][right]
                    ).sum()),
                )
    return {
        "packed": packed,
        "valid": valid,
        "area": area,
        "y_center": y_center,
        "confidence": confidence,
        "paths": np.asarray(paths),
        "image_sha256": image_sha,
        "geometry": geometry,
        "sample_manifest_sha256": manifest.hexdigest(),
        "pairwise_support_product_max": pairwise_support_product_max,
    }


def donor_candidates(indices, recipient, pids, paths):
    indices = np.asarray(indices, dtype=np.int64)
    keep = (
        (indices != int(recipient))
        & (pids[indices] != pids[int(recipient)])
        & (paths[indices] != paths[int(recipient)])
    )
    return indices[keep]


def select_donor(
    recipient,
    recipient_slot,
    donor_slot,
    valid,
    area,
    y_center,
    confidence,
    pids,
    camids,
    paths,
    camera_slot_groups,
    slot_groups,
):
    camera_key = (int(camids[recipient]), int(donor_slot))
    candidates = donor_candidates(
        camera_slot_groups.get(camera_key, ()), recipient, pids, paths
    )
    same_camera_candidate_available = candidates.size > 0
    same_camera = True
    if candidates.size == 0:
        candidates = donor_candidates(
            slot_groups[int(donor_slot)], recipient, pids, paths
        )
        same_camera = False
    if candidates.size == 0:
        return None
    recipient_area = max(float(area[recipient, recipient_slot]), 1e-8)
    distance = (
        np.abs(area[candidates, donor_slot] - recipient_area)
        / recipient_area
        + np.abs(
            y_center[candidates, donor_slot]
            - y_center[recipient, recipient_slot]
        )
        + np.abs(
            confidence[candidates, donor_slot]
            - confidence[recipient, recipient_slot]
        )
    )
    order = np.lexsort((paths[candidates], distance))
    chosen_position = int(order[0])
    chosen = int(candidates[chosen_position])
    return {
        "index": chosen,
        "same_camera": same_camera,
        "same_camera_candidate_available": bool(
            same_camera_candidate_available
        ),
        "distance": float(distance[chosen_position]),
        "area_delta": float(abs(
            area[chosen, donor_slot] - area[recipient, recipient_slot]
        )),
        "y_delta": float(abs(
            y_center[chosen, donor_slot]
            - y_center[recipient, recipient_slot]
        )),
        "confidence_delta": float(abs(
            confidence[chosen, donor_slot]
            - confidence[recipient, recipient_slot]
        )),
    }


def build_donor_maps(table, targets, pids, camids):
    valid = table["valid"]
    paths = table["paths"]
    area = table["area"]
    y_center = table["y_center"]
    confidence = table["confidence"]
    slot_groups = {
        slot: np.flatnonzero(valid[:, slot]) for slot in range(REGIONS)
    }
    camera_slot_groups = {
        (int(camera), slot): np.flatnonzero(
            (camids == camera) & valid[:, slot]
        )
        for camera in np.unique(camids)
        for slot in range(REGIONS)
    }
    same_slot = []
    wrong_slot = []
    for recipient, target in enumerate(targets.tolist()):
        if target < 0:
            same_slot.append(None)
            wrong_slot.append(None)
            continue
        same_slot.append(select_donor(
            recipient,
            target,
            target,
            valid,
            area,
            y_center,
            confidence,
            pids,
            camids,
            paths,
            camera_slot_groups,
            slot_groups,
        ))
        selected = None
        for offset in range(1, REGIONS):
            donor_slot = (int(target) + offset) % REGIONS
            selected = select_donor(
                recipient,
                target,
                donor_slot,
                valid,
                area,
                y_center,
                confidence,
                pids,
                camids,
                paths,
                camera_slot_groups,
                slot_groups,
            )
            if selected is not None:
                selected["slot"] = int(donor_slot)
                break
        wrong_slot.append(selected)
    return same_slot, wrong_slot


def donor_summary(donors, records, target_slots, donor_slots=None):
    pids = np.asarray([int(row[1]) for row in records])
    camids = np.asarray([int(row[2]) for row in records])
    paths = np.asarray([str(row[0]) for row in records])
    missing = sum(value is None for value in donors)
    chosen = np.asarray([
        -1 if value is None else int(value["index"]) for value in donors
    ])
    active = chosen >= 0
    recipient = np.flatnonzero(active)
    finite = all(
        value is None
        or all(np.isfinite(value[key]) for key in (
            "distance", "area_delta", "y_delta", "confidence_delta"
        ))
        for value in donors
    )
    result = {
        "missing": int(missing),
        "different_index_fraction": float(
            (chosen[active] != recipient).mean()
        ) if bool(active.any()) else 0.0,
        "different_pid_fraction": float(
            (pids[chosen[active]] != pids[recipient]).mean()
        ) if bool(active.any()) else 0.0,
        "different_path_fraction": float(
            (paths[chosen[active]] != paths[recipient]).mean()
        ) if bool(active.any()) else 0.0,
        "same_camera_fraction": float(
            (camids[chosen[active]] == camids[recipient]).mean()
        ) if bool(active.any()) else 0.0,
        "same_camera_priority_violations": int(sum(
            value is not None
            and value["same_camera_candidate_available"]
            and not value["same_camera"]
            for value in donors
        )),
        "finite": bool(finite),
        "distance": quantiles([
            value["distance"] for value in donors if value is not None
        ]),
        "area_delta": quantiles([
            value["area_delta"] for value in donors if value is not None
        ]),
        "y_delta": quantiles([
            value["y_delta"] for value in donors if value is not None
        ]),
        "confidence_delta": quantiles([
            value["confidence_delta"]
            for value in donors if value is not None
        ]),
        "by_target_slot": {
            str(slot): {
                "distance": quantiles([
                    value["distance"]
                    for index, value in enumerate(donors)
                    if value is not None and int(target_slots[index]) == slot
                ]),
                "area_delta": quantiles([
                    value["area_delta"]
                    for index, value in enumerate(donors)
                    if value is not None and int(target_slots[index]) == slot
                ]),
                "y_delta": quantiles([
                    value["y_delta"]
                    for index, value in enumerate(donors)
                    if value is not None and int(target_slots[index]) == slot
                ]),
                "confidence_delta": quantiles([
                    value["confidence_delta"]
                    for index, value in enumerate(donors)
                    if value is not None and int(target_slots[index]) == slot
                ]),
            }
            for slot in range(REGIONS)
        },
    }
    if donor_slots is not None:
        result["different_slot_fraction"] = float(np.mean([
            int(donor_slots[index]) != int(target_slots[index])
            for index in recipient
        ])) if len(recipient) else 0.0
    return result


def summarize_localization(rows, region_names):
    grouped = {}
    for slot, name in enumerate(region_names):
        for direction in ("top", "bottom", "left", "right"):
            selected = [
                row for row in rows
                if row["slot"] == slot and row["direction"] == direction
            ]
            key = "%s/%s" % (name, direction)
            grouped[key] = {
                "samples": len({row["index"] for row in selected}),
                "levels": {
                    str(level): {
                        "target_rectangle_fraction": quantiles([
                            row["target_rectangle_fraction"]
                            for row in selected if row["level"] == level
                        ]),
                        "max_non_target_rectangle_fraction": quantiles([
                            row["max_non_target_rectangle_fraction"]
                            for row in selected if row["level"] == level
                        ]),
                        "target_to_max_non_target_ratio": quantiles([
                            row["target_to_max_non_target_ratio"]
                            for row in selected if row["level"] == level
                        ]),
                    }
                    for level in range(3)
                },
            }
    return grouped


def summarize_controls(rows, region_names):
    result = {}
    for slot, name in enumerate(region_names):
        for direction in ("top", "bottom", "left", "right"):
            selected = [
                row["normalized_y_error"] for row in rows
                if row["slot"] == slot and row["direction"] == direction
            ]
            summary = quantiles(selected)
            summary["mean"] = (
                float(np.mean(selected)) if selected else None
            )
            result["%s/%s" % (name, direction)] = summary
    return result


def build_geometry_map(table, targets, static, seed):
    entries = []
    localization_rows = []
    control_rows = []
    y_errors = []
    construction_failures = []
    dilation_product_max = 0
    overshoot_failures = 0
    nested_failures = 0
    realized_failures = 0
    localization_failures = 0

    for index, target in enumerate(targets.tolist()):
        if target < 0:
            construction_failures.append({
                "index": index, "reason": "no valid target"
            })
            entries.append(None)
            continue
        supports = torch.stack([
            full_support(table["packed"], index, slot)
            for slot in range(REGIONS)
        ])
        target_support = supports[target]
        try:
            overlap = static.connected_occlusion_rectangles(
                target_support,
                static.LEVELS,
                "%s\0%d\0%d" % (
                    table["paths"][index], target, int(seed)
                ),
            )
            control = static.translated_nonoverlap_rectangles(
                target_support,
                overlap["boxes"],
                overlap["direction"],
                exclusion_radius=24,
            )
        except (RuntimeError, ValueError) as error:
            construction_failures.append({
                "index": index,
                "path": str(table["paths"][index]),
                "slot": int(target),
                "reason": str(error),
            })
            entries.append(None)
            continue

        levels = []
        local_failure = False
        for level_index, rectangle in enumerate(overlap["sets"]):
            intersections = [
                int((rectangle & supports[slot]).sum())
                for slot in range(REGIONS)
            ]
            target_intersection = intersections[target]
            max_non_target = max(
                intersections[:target] + intersections[target + 1:]
            )
            rectangle_area = int(rectangle.sum())
            target_fraction = target_intersection / float(rectangle_area)
            non_target_fraction = max_non_target / float(rectangle_area)
            ratio = target_intersection / float(max(max_non_target, 1))
            localization_ok = target_intersection > max_non_target
            if not localization_ok:
                localization_failures += 1
                local_failure = True
            localization_rows.append({
                "index": index,
                "slot": int(target),
                "direction": overlap["direction"],
                "level": level_index,
                "target_rectangle_fraction": target_fraction,
                "max_non_target_rectangle_fraction": non_target_fraction,
                "target_to_max_non_target_ratio": ratio,
            })
            levels.append({
                "target_level": float(static.LEVELS[level_index]),
                "realized_overlap": float(overlap["realized"][level_index]),
                "last_strip_fraction": float(
                    overlap["last_increment_fraction"][level_index]
                ),
                "overlap_box": list(overlap["boxes"][level_index]),
                "control_box": list(control["boxes"][level_index]),
                "slot_intersections": intersections,
                "rectangle_area": rectangle_area,
                "target_rectangle_fraction": target_fraction,
                "max_non_target_rectangle_fraction": non_target_fraction,
                "localization_ok": localization_ok,
            })

        overshoot_ok = bool((
            overlap["realized"] - torch.as_tensor(static.LEVELS)
            <= overlap["last_increment_fraction"] + 1e-12
        ).all())
        nested_ok = all(bool((
            overlap["sets"][left]
            & (~overlap["sets"][left + 1])
        ).sum() == 0) for left in range(len(static.LEVELS) - 1))
        realized_ok = bool((
            overlap["realized"] >= torch.as_tensor(static.LEVELS)
        ).all()) and bool((
            overlap["realized"][1:] > overlap["realized"][:-1]
        ).all())
        dilation_product = int((
            control["sets"]
            & static._dilate(target_support, 24)[None]
        ).sum())
        dilation_product_max = max(dilation_product_max, dilation_product)
        overshoot_failures += int(not overshoot_ok)
        nested_failures += int(not nested_ok)
        realized_failures += int(not realized_ok)
        y_errors.append(float(control["normalized_y_error"]))
        control_rows.append({
            "slot": int(target),
            "direction": overlap["direction"],
            "normalized_y_error": float(control["normalized_y_error"]),
        })
        entries.append({
            "direction": overlap["direction"],
            "support_box": list(overlap["support_box"]),
            "normalized_y_error": float(control["normalized_y_error"]),
            "dilation_product": dilation_product,
            "overshoot_ok": overshoot_ok,
            "nested_ok": nested_ok,
            "realized_ok": realized_ok,
            "localization_ok": not local_failure,
            "levels": levels,
        })
    return entries, {
        "construction_failures": construction_failures,
        "insufficient_fraction": len(construction_failures)
        / float(len(targets)),
        "dilation_product_max": int(dilation_product_max),
        "normalized_y_error": {
            "mean": float(np.mean(y_errors)) if y_errors else None,
            **quantiles(y_errors),
        },
        "overshoot_failures": int(overshoot_failures),
        "nested_failures": int(nested_failures),
        "realized_failures": int(realized_failures),
        "localization_failures": int(localization_failures),
        "localization_by_slot_direction": summarize_localization(
            localization_rows,
            ("head", "torso", "arms", "upper_leg", "lower_leg"),
        ),
        "control_by_slot_direction": summarize_controls(
            control_rows,
            ("head", "torso", "arms", "upper_leg", "lower_leg"),
        ),
    }


def wrong_mask_iou_summary(table, targets, donors):
    values = []
    by_slot = {slot: [] for slot in range(REGIONS)}
    per_record = [None] * len(donors)
    for index, (target, donor) in enumerate(zip(targets.tolist(), donors)):
        if target < 0 or donor is None:
            continue
        left = unpack_support(table["packed"], index, target)
        right = unpack_support(
            table["packed"], donor["index"], target
        )
        union = int(np.logical_or(left, right).sum())
        value = (
            int(np.logical_and(left, right).sum()) / float(union)
            if union else 0.0
        )
        values.append(value)
        by_slot[target].append(value)
        per_record[index] = value
    return {
        "overall": quantiles(values),
        "by_slot": {
            str(slot): quantiles(by_slot[slot]) for slot in range(REGIONS)
        },
    }, per_record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--phase0b-script", required=True)
    parser.add_argument("--ontology-script", required=True)
    parser.add_argument("--static-script", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--map-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    audit_script = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve()
    phase0b_script = Path(args.phase0b_script).resolve()
    ontology_script = Path(args.ontology_script).resolve()
    static_script = Path(args.static_script).resolve()
    pose_artifact = Path(args.pose_artifact).resolve()
    data_root = Path(args.data_root).resolve()
    map_output = Path(args.map_output).resolve()
    result_output = Path(args.output).resolve()
    if map_output == result_output:
        raise RuntimeError("Map and result outputs must be distinct")
    if map_output.exists() or result_output.exists():
        raise RuntimeError("Map/result output already exists")
    expected = {
        phase0b_script: EXPECTED_PHASE0B_SCRIPT_SHA256,
        ontology_script: EXPECTED_ONTOLOGY_SCRIPT_SHA256,
        static_script: EXPECTED_STATIC_SCRIPT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError("Dependency SHA mismatch: %s" % path)
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if actual_commit != args.source_commit:
        raise RuntimeError("Execution commit mismatch")
    tracked_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    ).strip()
    if tracked_status:
        raise RuntimeError("Execution repository has tracked changes")

    base = load_module("exp392_phase0b", phase0b_script)
    ontology = load_module("exp392_ontology", ontology_script)
    static = load_module("exp392_b2s_static", static_script)
    if ontology.ontology_static_contract(base, "hard-owner")["status"] != "PASS":
        raise RuntimeError("Hard-owner ontology contract failed")
    if sha256_file(pose_artifact / "manifest.json") != base.EXPECTED_POSE_MANIFEST_SHA256:
        raise RuntimeError("Pose manifest SHA mismatch")

    os.chdir(str(repo_root))
    sys.path.insert(0, str(repo_root))
    from datasets.occluded_duke import OccludedDuke
    from datasets.pose_targets import PoseTargetStore

    dataset = OccludedDuke(root=str(data_root), verbose=False)
    records = list(dataset.train)
    if len(records) != EXPECTED_SAMPLES:
        raise RuntimeError("Unexpected official train size")
    pose_store = PoseTargetStore(
        pose_artifact, base.EXPECTED_POSE_MANIFEST_SHA256
    )
    if len(pose_store) != EXPECTED_SAMPLES:
        raise RuntimeError("Unexpected pose artifact size")

    transform = PoseGeometryTransform(base, ontology, args.seed)
    table = build_pose_table(records, pose_store, transform)
    pids = np.asarray([int(row[1]) for row in records], dtype=np.int64)
    camids = np.asarray([int(row[2]) for row in records], dtype=np.int64)
    targets, target_counts = static.deterministic_balanced_targets(
        table["valid"], table["paths"].tolist(), args.seed
    )
    repeat_targets, repeat_counts = static.deterministic_balanced_targets(
        table["valid"], table["paths"].tolist(), args.seed
    )
    same_slot, wrong_slot = build_donor_maps(
        table, targets, pids, camids
    )
    geometry, geometry_summary = build_geometry_map(
        table, targets, static, args.seed
    )
    wrong_mask_iou, wrong_mask_iou_per_record = wrong_mask_iou_summary(
        table, targets, same_slot
    )
    control_groups = geometry_summary["control_by_slot_direction"]
    control_groups_complete = all(
        summary["count"] > 0 for summary in control_groups.values()
    )
    control_groups_within_bounds = all(
        summary["count"] > 0
        and summary["mean"] <= 1.0 / 8.0
        and summary["p95"] <= 2.0 / 8.0
        for summary in control_groups.values()
    )
    wrong_mask_slots_within_bounds = all(
        summary["count"] > 0
        and summary["p50"] <= 0.30
        and summary["p95"] <= 0.50
        for summary in wrong_mask_iou["by_slot"].values()
    )

    donor_slots = [
        -1 if value is None else int(value["slot"])
        for value in wrong_slot
    ]
    same_summary = donor_summary(
        same_slot, records, targets.tolist()
    )
    wrong_summary = donor_summary(
        wrong_slot,
        records,
        targets.tolist(),
        donor_slots=donor_slots,
    )
    execution = {
        "source_commit": actual_commit,
        "audit_script_sha256": sha256_file(audit_script),
        "phase0b_script_sha256": sha256_file(phase0b_script),
        "ontology_script_sha256": sha256_file(ontology_script),
        "static_script_sha256": sha256_file(static_script),
        "pose_manifest_sha256": base.EXPECTED_POSE_MANIFEST_SHA256,
        "sample_manifest_sha256": table["sample_manifest_sha256"],
        "seed": int(args.seed),
        "device": "cpu-only/no-CLIP/no-ReID/no-optimizer",
    }
    gates = {
        "official_sample_count_exact": len(records) == EXPECTED_SAMPLES,
        "pose_sample_count_exact": len(pose_store) == EXPECTED_SAMPLES,
        "all_targets_assigned": bool((targets >= 0).all()),
        "balanced_target_spread_at_most_one": int(
            target_counts.max() - target_counts.min()
        ) <= 1,
        "target_map_repeat_exact": bool(
            torch.equal(targets, repeat_targets)
            and torch.equal(target_counts, repeat_counts)
        ),
        "hard_owner_support_product_exact_zero": (
            table["pairwise_support_product_max"] == 0
        ),
        "same_slot_donor_complete": same_summary["missing"] == 0,
        "same_slot_donor_no_fixed_point": (
            same_summary["different_index_fraction"] == 1.0
        ),
        "same_slot_donor_different_pid_path": (
            same_summary["different_pid_fraction"] == 1.0
            and same_summary["different_path_fraction"] == 1.0
        ),
        "same_slot_donor_same_camera_priority_exact": (
            same_summary["same_camera_priority_violations"] == 0
        ),
        "wrong_slot_donor_complete": wrong_summary["missing"] == 0,
        "wrong_slot_donor_no_fixed_point": (
            wrong_summary["different_index_fraction"] == 1.0
        ),
        "wrong_slot_donor_different_pid_path_slot": (
            wrong_summary["different_pid_fraction"] == 1.0
            and wrong_summary["different_path_fraction"] == 1.0
            and wrong_summary["different_slot_fraction"] == 1.0
        ),
        "wrong_slot_donor_same_camera_priority_exact": (
            wrong_summary["same_camera_priority_violations"] == 0
        ),
        "donor_statistics_finite": (
            same_summary["finite"] and wrong_summary["finite"]
        ),
        "connected_construction_complete": (
            len(geometry_summary["construction_failures"]) == 0
        ),
        "connected_overshoot_exact": (
            geometry_summary["overshoot_failures"] == 0
        ),
        "connected_nesting_exact": (
            geometry_summary["nested_failures"] == 0
        ),
        "connected_realized_exact": (
            geometry_summary["realized_failures"] == 0
        ),
        "nonoverlap_insufficient_fraction_zero": (
            geometry_summary["insufficient_fraction"] == 0.0
        ),
        "nonoverlap_24px_dilation_zero_exact": (
            geometry_summary["dilation_product_max"] == 0
        ),
        "nonoverlap_y_mean_within_one_bin": (
            geometry_summary["normalized_y_error"]["mean"] is not None
            and geometry_summary["normalized_y_error"]["mean"] <= 1.0 / 8.0
        ),
        "nonoverlap_y_p95_within_two_bins": (
            geometry_summary["normalized_y_error"]["p95"] is not None
            and geometry_summary["normalized_y_error"]["p95"] <= 2.0 / 8.0
        ),
        "nonoverlap_all_slot_direction_groups_present": (
            control_groups_complete
        ),
        "nonoverlap_all_slot_direction_groups_within_bounds": (
            control_groups_within_bounds
        ),
        "occluder_localization_no_failure": (
            geometry_summary["localization_failures"] == 0
        ),
        "wrong_mask_iou_median_at_most_030": (
            wrong_mask_iou["overall"]["p50"] is not None
            and wrong_mask_iou["overall"]["p50"] <= 0.30
        ),
        "wrong_mask_iou_p95_at_most_050": (
            wrong_mask_iou["overall"]["p95"] is not None
            and wrong_mask_iou["overall"]["p95"] <= 0.50
        ),
        "wrong_mask_iou_every_slot_within_bounds": (
            wrong_mask_slots_within_bounds
        ),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    map_records = []
    for index, target in enumerate(targets.tolist()):
        same = same_slot[index]
        wrong = wrong_slot[index]
        map_records.append({
            "index": index,
            "relative_path": str(table["paths"][index]),
            "image_sha256": table["image_sha256"][index],
            "pid": int(pids[index]),
            "camid": int(camids[index]),
            "valid": table["valid"][index].tolist(),
            "target_slot": int(target),
            "augmentation": table["geometry"][index],
            "same_slot_donor": None if same is None else {
                **same,
                "path": str(table["paths"][same["index"]]),
                "pid": int(pids[same["index"]]),
                "camid": int(camids[same["index"]]),
                "slot": int(target),
            },
            "wrong_mask_donor": None if same is None else {
                "index": int(same["index"]),
                "path": str(table["paths"][same["index"]]),
                "slot": int(target),
                "actual_augmented_iou": wrong_mask_iou_per_record[index],
            },
            "wrong_slot_occluder_donor": None if wrong is None else {
                **wrong,
                "path": str(table["paths"][wrong["index"]]),
                "pid": int(pids[wrong["index"]]),
                "camid": int(camids[wrong["index"]]),
            },
            "geometry": geometry[index],
        })
    map_payload = {
        "schema_version": 1,
        "status": (
            "EXP392_PHASE0B2_FULL_POSE_MAP_FROZEN"
            if verdict == "PASS"
            else "EXP392_PHASE0B2_FULL_POSE_MAP_FAILED_NOT_FROZEN"
        ),
        "verdict": verdict,
        "execution": execution,
        "target_counts": target_counts.tolist(),
        "records": map_records,
    }
    write_json(map_output, map_payload)
    map_sha = sha256_file(map_output)
    result = {
        "status": "EXP392_PHASE0B2_FULL_POSE_FEASIBILITY_COMPLETE",
        "verdict": verdict,
        "clip_8image_contract_authorized": verdict == "PASS",
        "gpu_authorized": False,
        "formal_training_authorized": False,
        "gates": gates,
        "execution": execution,
        "artifacts": {
            "map_path": str(map_output),
            "map_sha256": map_sha,
        },
        "measurements": {
            "samples": len(records),
            "target_counts": target_counts.tolist(),
            "valid_slot_counts": table["valid"].sum(0).tolist(),
            "hard_owner_support_product_max": int(
                table["pairwise_support_product_max"]
            ),
            "same_slot_donor": same_summary,
            "wrong_mask_donor_equals_same_slot": True,
            "wrong_slot_donor": wrong_summary,
            "wrong_mask_augmented_iou": wrong_mask_iou,
            "geometry": geometry_summary,
            "wall_seconds": time.time() - started,
        },
    }
    write_json(result_output, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
