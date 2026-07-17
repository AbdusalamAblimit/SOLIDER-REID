#!/usr/bin/env python3
"""Extract clean COCO-17 pose targets from raw ReID training images."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from mmpose.apis import inference_topdown, init_model


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-url", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-dataset-manifest")
    parser.add_argument("--expected-config-sha")
    parser.add_argument("--expected-checkpoint-sha")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-period", type=int, default=100)
    return parser.parse_args()


def validate_args(args):
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.image_root = args.image_root.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.image_root.is_dir():
        raise NotADirectoryError(args.image_root)
    if "pose_data" in args.image_root.parts:
        raise ValueError("Legacy pose_data must never be an extraction input")
    if args.shard_size <= 0:
        raise ValueError("shard-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    work_dir = args.output_dir.with_name(args.output_dir.name + ".incomplete")
    if args.output_dir.exists() or work_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite an output or incomplete extraction: {}".format(
                args.output_dir
            )
        )
    return work_dir


def write_shard(shard_dir, shard_index, records):
    relative_paths = np.asarray([item["relative_path"] for item in records])
    image_sha256 = np.asarray([item["image_sha256"] for item in records])
    image_sizes = np.asarray([item["image_size"] for item in records], dtype=np.int32)
    keypoints = np.stack([item["keypoints"] for item in records]).astype(np.float32)
    scores = np.stack([item["scores"] for item in records]).astype(np.float32)

    shard_name = "pose-{:05d}.npz".format(shard_index)
    shard_path = shard_dir / shard_name
    temporary = shard_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            relative_paths=relative_paths,
            image_sha256=image_sha256,
            image_sizes=image_sizes,
            keypoints=keypoints,
            scores=scores,
        )
    os.replace(str(temporary), str(shard_path))
    return {
        "file": "shards/{}".format(shard_name),
        "count": len(records),
        "sha256": sha256_file(shard_path),
    }


def main():
    args = parse_args()
    work_dir = validate_args(args)
    config_sha = sha256_file(args.config)
    checkpoint_sha = sha256_file(args.checkpoint)
    if args.expected_config_sha and config_sha != args.expected_config_sha:
        raise RuntimeError("Config SHA256 mismatch")
    if args.expected_checkpoint_sha and checkpoint_sha != args.expected_checkpoint_sha:
        raise RuntimeError("Checkpoint SHA256 mismatch")

    image_paths = sorted(args.image_root.glob("*.jpg"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if args.expected_count is not None and len(image_paths) != args.expected_count:
        raise RuntimeError(
            "Expected {} images, found {}".format(args.expected_count, len(image_paths))
        )
    if not image_paths:
        raise RuntimeError("No JPG images found")

    work_dir.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = work_dir / "shards"
    shard_dir.mkdir(parents=True)

    model = init_model(
        str(args.config), str(args.checkpoint), device=args.device
    )
    dataset_manifest = hashlib.sha256()
    records_manifest = hashlib.sha256()
    shards = []
    records = []
    all_scores = []
    out_of_bounds = 0
    start_time = time.time()

    for index, image_path in enumerate(image_paths, start=1):
        image_sha = sha256_file(image_path)
        image_size_bytes = image_path.stat().st_size
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()

        bbox = np.asarray([[0.0, 0.0, float(width), float(height)]], dtype=np.float32)
        pose_samples = inference_topdown(
            model, str(image_path), bboxes=bbox, bbox_format="xyxy"
        )
        if len(pose_samples) != 1:
            raise RuntimeError(
                "Expected one pose for {}, got {}".format(image_path, len(pose_samples))
            )
        prediction = pose_samples[0].pred_instances
        keypoints = np.asarray(prediction.keypoints, dtype=np.float32)
        scores = np.asarray(prediction.keypoint_scores, dtype=np.float32)
        if keypoints.shape != (1, 17, 2) or scores.shape != (1, 17):
            raise RuntimeError(
                "Unexpected pose shape for {}: {} / {}".format(
                    image_path, keypoints.shape, scores.shape
                )
            )
        keypoints = keypoints[0]
        scores = scores[0]
        if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
            raise RuntimeError("Non-finite pose prediction for {}".format(image_path))

        relative_path = image_path.relative_to(args.image_root.parent).as_posix()
        record = {
            "relative_path": relative_path,
            "image_sha256": image_sha,
            "image_size": (width, height),
            "keypoints": keypoints,
            "scores": scores,
        }
        records.append(record)
        all_scores.append(scores)
        out_of_bounds += int(
            np.count_nonzero(
                (keypoints[:, 0] < 0)
                | (keypoints[:, 0] > width - 1)
                | (keypoints[:, 1] < 0)
                | (keypoints[:, 1] > height - 1)
            )
        )

        dataset_manifest.update(image_path.name.encode("utf-8"))
        dataset_manifest.update(b"\0")
        dataset_manifest.update(str(image_size_bytes).encode("ascii"))
        dataset_manifest.update(b"\0")
        dataset_manifest.update(image_sha.encode("ascii"))
        dataset_manifest.update(b"\n")

        records_manifest.update(relative_path.encode("utf-8"))
        records_manifest.update(b"\0")
        records_manifest.update(image_sha.encode("ascii"))
        records_manifest.update(np.asarray([width, height], dtype=np.int32).tobytes())
        records_manifest.update(keypoints.tobytes())
        records_manifest.update(scores.tobytes())

        if len(records) == args.shard_size or index == len(image_paths):
            shards.append(write_shard(shard_dir, len(shards), records))
            records = []
        if index % args.log_period == 0 or index == len(image_paths):
            elapsed = time.time() - start_time
            print(
                "processed={}/{} elapsed={:.1f}s images_per_second={:.2f}".format(
                    index, len(image_paths), elapsed, index / elapsed
                ),
                flush=True,
            )

    dataset_manifest_sha = dataset_manifest.hexdigest()
    if (
        args.expected_dataset_manifest
        and args.limit is None
        and dataset_manifest_sha != args.expected_dataset_manifest
    ):
        raise RuntimeError("Dataset manifest SHA256 mismatch")

    score_values = np.concatenate(all_scores)
    elapsed = time.time() - start_time
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mmpose": __import__("mmpose").__version__,
        "config": str(args.config),
        "config_sha256": config_sha,
        "checkpoint": str(args.checkpoint),
        "checkpoint_url": args.checkpoint_url,
        "checkpoint_sha256": checkpoint_sha,
        "image_root": str(args.image_root),
        "dataset_manifest_sha256": dataset_manifest_sha,
        "records_manifest_sha256": records_manifest.hexdigest(),
        "sample_count": len(image_paths),
        "joint_count": 17,
        "coordinate_space": "original_image_pixels_xy",
        "bbox_policy": "full_image_xyxy",
        "shard_size": args.shard_size,
        "shards": shards,
        "statistics": {
            "score_min": float(score_values.min()),
            "score_mean": float(score_values.mean()),
            "score_max": float(score_values.max()),
            "score_p10": float(np.percentile(score_values, 10)),
            "score_p50": float(np.percentile(score_values, 50)),
            "score_p90": float(np.percentile(score_values, 90)),
            "score_below_0_1": int(np.count_nonzero(score_values < 0.1)),
            "score_below_0_3": int(np.count_nonzero(score_values < 0.3)),
            "score_below_0_5": int(np.count_nonzero(score_values < 0.5)),
            "out_of_bounds_keypoints": out_of_bounds,
            "elapsed_seconds": elapsed,
            "images_per_second": len(image_paths) / elapsed,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available()
            else 0.0,
        },
    }
    manifest_path = work_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary_manifest), str(manifest_path))
    os.replace(str(work_dir), str(args.output_dir))
    print("completed output={} manifest_sha256={}".format(
        args.output_dir, sha256_file(args.output_dir / "manifest.json")
    ))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("fatal: {}".format(error), file=sys.stderr)
        raise
