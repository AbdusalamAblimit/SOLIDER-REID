#!/usr/bin/env python3
"""Build the one fresh region-isolated CLIP cache used by exp411."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.bases import read_image
from datasets.occluded_duke import OccludedDuke
from datasets.pose_targets import PoseTargetStore
from model.pose_clip_relation import (
    RegionIsolatedClipVisualTeacher,
    render_pose_indexed_regions,
    sha256_file,
)


EXPECTED_SAMPLES = 15618
POSE_MANIFEST_SHA256 = (
    "cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--pose-manifest-sha256", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--clip-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=1)
    return parser.parse_args()


def resize_rgb_and_pose(path, pose):
    image = read_image(path).convert("RGB")
    image = image.resize((128, 384), resample=Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    rgb = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    width, height = pose.image_size
    keypoints = pose.keypoints.clone().float()
    keypoints[:, 0] *= 128.0 / float(width)
    keypoints[:, 1] *= 384.0 / float(height)
    valid = (
        pose.valid.bool()
        & (keypoints[:, 0] >= 0)
        & (keypoints[:, 0] <= 127)
        & (keypoints[:, 1] >= 0)
        & (keypoints[:, 1] <= 383)
    )
    return rgb, keypoints, valid


def digest_rows(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.pose_manifest_sha256 != POSE_MANIFEST_SHA256:
        raise RuntimeError("exp411 requires the frozen clean pose manifest")
    if args.batch_size <= 0 or args.microbatch <= 0:
        raise ValueError("batch sizes must be positive")
    output = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if output.exists() or manifest_path.exists():
        raise FileExistsError("exp411 cache outputs must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = OccludedDuke(root=args.data_root, verbose=False)
    records = dataset.train
    if len(records) != EXPECTED_SAMPLES:
        raise RuntimeError("unexpected official train sample count")
    pose_store = PoseTargetStore(
        args.pose_artifact, args.pose_manifest_sha256
    )
    relative_paths = [
        Path(record[0]).resolve().relative_to(pose_store.dataset_root).as_posix()
        for record in records
    ]
    if len(set(relative_paths)) != EXPECTED_SAMPLES:
        raise RuntimeError("official train paths are not unique")
    if set(relative_paths) != set(pose_store._records):
        raise RuntimeError("official train and pose artifact coverage differ")
    source_head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=str(REPO_ROOT), text=True
    ).strip()
    if len(source_head) != 40:
        raise RuntimeError("could not freeze exp411 source HEAD")
    if subprocess.call(("git", "diff", "--quiet"), cwd=str(REPO_ROOT)) != 0:
        raise RuntimeError("exp411 cache builder requires a clean source tree")
    builder_sha256 = sha256_file(Path(__file__).resolve())
    teacher_source_sha256 = sha256_file(
        REPO_ROOT / "model" / "pose_clip_relation.py"
    )

    device = torch.device("cuda", 0)
    teacher = RegionIsolatedClipVisualTeacher(
        args.clip_checkpoint,
        args.clip_checkpoint_sha256,
        device,
        microbatch=args.microbatch,
    )
    features = torch.empty(EXPECTED_SAMPLES, 5, 768, dtype=torch.float16)
    validity = torch.empty(EXPECTED_SAMPLES, 5, dtype=torch.bool)
    image_sha256 = []
    for start in range(0, EXPECTED_SAMPLES, args.batch_size):
        stop = min(start + args.batch_size, EXPECTED_SAMPLES)
        rgb_rows = []
        point_rows = []
        valid_rows = []
        for path, _, _, _ in records[start:stop]:
            pose = pose_store.get(path, verify_image_sha=True)
            rgb, points, active = resize_rgb_and_pose(path, pose)
            rgb_rows.append(rgb)
            point_rows.append(points)
            valid_rows.append(active)
            image_sha256.append(pose.image_sha256)
        rgb = torch.stack(rgb_rows).to(device)
        points = torch.stack(point_rows).to(device)
        active = torch.stack(valid_rows).to(device)
        masks, geometry_valid = render_pose_indexed_regions(
            points,
            active,
            image_hw=(384, 128),
            field_hw=(96, 32),
            sigma=1.5,
        )
        slot_features, readout_valid = teacher.encode(rgb, masks)
        slot_features = torch.nn.functional.normalize(
            slot_features.float(), dim=-1
        )
        features[start:stop] = slot_features.cpu().half()
        validity[start:stop] = (geometry_valid & readout_valid).cpu()
        if start == 0 or stop == EXPECTED_SAMPLES or stop % 1000 < args.batch_size:
            print("encoded {}/{}".format(stop, EXPECTED_SAMPLES), flush=True)

    if not bool(torch.isfinite(features.float()).all()):
        raise RuntimeError("non-finite final cache")
    if not bool(validity.any(dim=0).all()):
        raise RuntimeError("one or more anatomical slots are globally empty")
    active_norms = features.float().norm(dim=-1)[validity]
    if not bool(
        torch.allclose(
            active_norms,
            torch.ones_like(active_norms),
            atol=2e-3,
            rtol=2e-3,
        )
    ):
        raise RuntimeError("valid cache features are not L2 normalized")
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            schema=np.asarray("exp411-pcmpsr-cache-v1"),
            relative_paths=np.asarray(relative_paths, dtype=np.str_),
            image_sha256=np.asarray(image_sha256, dtype=np.str_),
            features=features.numpy(),
            valid=validity.numpy(),
            preprocessing=np.asarray(
                "raw-rgb-pose-resize-384x128-no-augmentation"
            ),
            pose_manifest_sha256=np.asarray(args.pose_manifest_sha256),
            clip_checkpoint_sha256=np.asarray(args.clip_checkpoint_sha256),
            source_head=np.asarray(source_head),
            builder_sha256=np.asarray(builder_sha256),
            teacher_source_sha256=np.asarray(teacher_source_sha256),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(output))
    cache_sha256 = sha256_file(output)

    manifest = {
        "schema": "exp411-pcmpsr-manifest-v1",
        "cache": str(output),
        "cache_sha256": cache_sha256,
        "samples": EXPECTED_SAMPLES,
        "shape": list(features.shape),
        "dtype": str(features.numpy().dtype),
        "valid_by_slot": validity.sum(dim=0).tolist(),
        "relative_path_sha256": digest_rows(relative_paths),
        "image_sha256_vector_sha256": digest_rows(image_sha256),
        "pid_vector_sha256": digest_rows(int(record[1]) for record in records),
        "pose_manifest_sha256": args.pose_manifest_sha256,
        "clip_checkpoint_sha256": args.clip_checkpoint_sha256,
        "source_head": source_head,
        "builder_sha256": builder_sha256,
        "teacher_source_sha256": teacher_source_sha256,
        "preprocessing": "raw-rgb-pose-resize-384x128-no-augmentation",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result = dict(manifest)
    result["manifest"] = str(manifest_path)
    result["manifest_sha256"] = sha256_file(manifest_path)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
