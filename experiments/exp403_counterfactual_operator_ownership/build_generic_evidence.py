#!/usr/bin/env python3
"""Build the fresh frozen train-split generic ELO evidence asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.occluded_duke import OccludedDuke
from datasets.paired_pose_transform import PairedPoseTransform
from datasets.pose_dataset import PoseImageDataset, pose_train_collate_fn
from datasets.pose_targets import PoseTargetStore
from model.clip_semantic_teacher import FrozenRichClipEvidenceTeacher


OFFICIAL_TRAIN_IMAGES = 15618


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--pose-manifest-sha256", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--clip-checkpoint-sha256", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--codebook-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip-microbatch", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("Generic evidence output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = OccludedDuke(root=args.data_root, verbose=False)
    if len(dataset.train) != OFFICIAL_TRAIN_IMAGES:
        raise RuntimeError("Official train image count mismatch")
    pose_store = PoseTargetStore(
        args.pose_artifact, args.pose_manifest_sha256
    )
    transform = PairedPoseTransform(
        size_train=(384, 128),
        flip_probability=0.0,
        padding=0,
        pixel_mean=(0.5, 0.5, 0.5),
        pixel_std=(0.5, 0.5, 0.5),
        erasing_probability=0.0,
        return_teacher_rgb=True,
    )
    train_set = PoseImageDataset(
        dataset.train, pose_store, transform, verify_image_sha=False
    )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=pose_train_collate_fn,
        pin_memory=True,
    )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    teacher = FrozenRichClipEvidenceTeacher(
        checkpoint=args.clip_checkpoint,
        checkpoint_sha256=args.clip_checkpoint_sha256,
        codebook=args.codebook,
        codebook_sha256=args.codebook_sha256,
        device=device,
        microbatch=args.clip_microbatch,
    )
    sums = torch.zeros(5, 16, dtype=torch.float64)
    counts = torch.zeros(5, dtype=torch.int64)
    images_seen = 0
    for batch_index, (_, _, _, _, pose_batch) in enumerate(loader):
        teacher_rgb = pose_batch["teacher_rgb"].to(device, non_blocking=True)
        keypoints = pose_batch["keypoints"].to(device, non_blocking=True)
        scores = pose_batch["scores"].to(device, non_blocking=True)
        valid = pose_batch["valid"].to(device, non_blocking=True)
        with torch.no_grad(), amp.autocast(enabled=True):
            result = teacher(teacher_rgb, keypoints, scores, valid)
        evidence = result["evidence_code"].double().cpu()
        semantic_valid = result["valid"].bool().cpu()
        for slot in range(5):
            keep = semantic_valid[:, slot]
            sums[slot] += evidence[keep, slot].sum(dim=0)
            counts[slot] += int(keep.sum())
        images_seen += teacher_rgb.shape[0]
        if batch_index % 25 == 0:
            print(
                json.dumps(
                    {
                        "stage": "generic-mean",
                        "images": images_seen,
                        "total": OFFICIAL_TRAIN_IMAGES,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if images_seen != OFFICIAL_TRAIN_IMAGES or bool((counts <= 0).any()):
        raise RuntimeError("Incomplete generic evidence coverage")
    generic = sums / counts[:, None]
    if not bool(torch.isfinite(generic).all()):
        raise RuntimeError("Non-finite generic evidence mean")
    payload = {
        "experiment": "exp403_counterfactual_operator_ownership",
        "format": "elo_generic_evidence_v1",
        "dataset": "occluded_duke",
        "split": "train",
        "transform": "deterministic_resize_only_384x128",
        "official_images": images_seen,
        "count_by_slot": counts.tolist(),
        "clip_checkpoint_sha256": args.clip_checkpoint_sha256,
        "codebook_sha256": args.codebook_sha256,
        "pose_manifest_sha256": args.pose_manifest_sha256,
        "evidence": generic.tolist(),
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "sha256": sha256_file(output),
                "count_by_slot": counts.tolist(),
                "mean_norm_by_slot": generic.norm(dim=-1).tolist(),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
