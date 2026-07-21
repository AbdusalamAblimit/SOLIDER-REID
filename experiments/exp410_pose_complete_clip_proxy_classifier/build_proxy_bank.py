#!/usr/bin/env python3
"""Build the one fresh pose-complete CLIP identity proxy bank for exp410."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.occluded_duke import OccludedDuke
from datasets.pose_targets import PoseTargetStore
from model.pose_complete_clip_proxy import (
    DATASET,
    FEATURE_DIM,
    NUM_IDENTITIES,
    NUM_SAMPLES,
    NUM_SLOTS,
    SCHEMA,
    SOURCE_CACHE_SCHEMA,
    SOURCE_CACHE_SHA256,
    SOURCE_CLIP_CHECKPOINT_SHA256,
    SOURCE_PREPROCESSING,
    dataset_contract,
    ordered_digest,
    sha256_file,
)


SOURCE_REQUIRED_FIELDS = {
    "schema",
    "relative_paths",
    "image_sha256",
    "features",
    "valid",
    "preprocessing",
    "pose_manifest_sha256",
    "clip_checkpoint_sha256",
    "source_head",
    "builder_sha256",
    "teacher_source_sha256",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--source-cache-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--pose-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def _canonical_regular_file(path_value, label):
    configured = Path(path_value).expanduser()
    if not configured.is_absolute():
        raise ValueError(label + " path must be absolute")
    resolved = configured.resolve(strict=True)
    if resolved != configured:
        raise RuntimeError(label + " must use its canonical path")
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(label + " must be a unique regular file")
    return resolved


def _load_source_cache(path, expected_sha256, pose_manifest_sha256):
    source = _canonical_regular_file(path, "source cache")
    if expected_sha256 != SOURCE_CACHE_SHA256:
        raise RuntimeError("exp410 requires the frozen exp409 source cache SHA")
    if sha256_file(source) != expected_sha256:
        raise RuntimeError("source cache SHA mismatch")
    with np.load(str(source), allow_pickle=False) as payload:
        if set(payload.files) != SOURCE_REQUIRED_FIELDS:
            raise RuntimeError("unexpected source cache fields")
        schema = str(payload["schema"].item())
        paths = payload["relative_paths"].copy()
        image_sha256 = payload["image_sha256"].copy()
        features = payload["features"].copy()
        valid = payload["valid"].copy()
        preprocessing = str(payload["preprocessing"].item())
        source_pose_manifest = str(payload["pose_manifest_sha256"].item())
        clip_checkpoint = str(payload["clip_checkpoint_sha256"].item())
        source_head = str(payload["source_head"].item())
        source_builder_sha256 = str(payload["builder_sha256"].item())
        source_teacher_sha256 = str(payload["teacher_source_sha256"].item())
    if schema != SOURCE_CACHE_SCHEMA:
        raise RuntimeError("source cache schema mismatch")
    if preprocessing != SOURCE_PREPROCESSING:
        raise RuntimeError("source cache preprocessing mismatch")
    if source_pose_manifest != pose_manifest_sha256:
        raise RuntimeError("source cache pose manifest mismatch")
    if clip_checkpoint != SOURCE_CLIP_CHECKPOINT_SHA256:
        raise RuntimeError("source cache CLIP checkpoint mismatch")
    if paths.shape != (NUM_SAMPLES,) or paths.dtype.kind != "U":
        raise RuntimeError("source cache path vector mismatch")
    if image_sha256.shape != paths.shape or image_sha256.dtype.kind != "U":
        raise RuntimeError("source cache RGB SHA vector mismatch")
    if features.shape != (NUM_SAMPLES, NUM_SLOTS, FEATURE_DIM):
        raise RuntimeError("source cache feature shape mismatch")
    if features.dtype != np.float16:
        raise RuntimeError("source cache features must be float16")
    if valid.shape != (NUM_SAMPLES, NUM_SLOTS) or valid.dtype != np.bool_:
        raise RuntimeError("source cache validity mismatch")
    if len(set(str(value) for value in paths.tolist())) != NUM_SAMPLES:
        raise RuntimeError("source cache paths are not unique")
    if not np.isfinite(features).all():
        raise RuntimeError("source cache contains non-finite features")
    hex64 = re.compile(r"[0-9a-f]{64}")
    if any(hex64.fullmatch(str(value)) is None for value in image_sha256.tolist()):
        raise RuntimeError("source cache RGB SHA vector is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", source_head) is None:
        raise RuntimeError("source cache HEAD is invalid")
    if hex64.fullmatch(source_builder_sha256) is None or hex64.fullmatch(
        source_teacher_sha256
    ) is None:
        raise RuntimeError("source cache source SHA is invalid")
    return {
        "path": source,
        "paths": tuple(str(value) for value in paths.tolist()),
        "image_sha256": tuple(str(value) for value in image_sha256.tolist()),
        "features": torch.from_numpy(features),
        "valid": torch.from_numpy(valid),
        "source_head": source_head,
        "source_builder_sha256": source_builder_sha256,
        "source_teacher_source_sha256": source_teacher_sha256,
    }


def _build_proxy(source, contract, records, pose_store):
    if set(source["paths"]) != set(contract["paths"]):
        raise RuntimeError("source cache and official train path sets differ")
    record_by_path = {}
    for image_path, relabel, _, _ in records:
        relative_path = (
            Path(image_path)
            .resolve()
            .relative_to(pose_store.dataset_root)
            .as_posix()
        )
        record_by_path[relative_path] = int(relabel)
    expected_rgb = dict(zip(contract["paths"], contract["image_sha256"]))
    labels = []
    for relative_path, rgb_sha256 in zip(
        source["paths"], source["image_sha256"]
    ):
        if expected_rgb.get(relative_path) != rgb_sha256:
            raise RuntimeError("source cache RGB binding mismatch")
        labels.append(record_by_path[relative_path])
    labels = torch.as_tensor(labels, dtype=torch.long)
    if set(labels.tolist()) != set(range(NUM_IDENTITIES)):
        raise RuntimeError("source cache does not cover every relabel PID")

    source_features = F.normalize(source["features"].float(), dim=-1)
    source_valid = source["valid"].bool()
    proxy = torch.empty(NUM_IDENTITIES, FEATURE_DIM, dtype=torch.float32)
    slot_counts = torch.empty(NUM_IDENTITIES, NUM_SLOTS, dtype=torch.int64)
    for identity in range(NUM_IDENTITIES):
        identity_rows = labels == identity
        slot_centers = []
        for slot in range(NUM_SLOTS):
            active = identity_rows & source_valid[:, slot]
            count = int(active.sum())
            if count <= 0:
                raise RuntimeError("identity-slot has no valid CLIP support")
            slot_counts[identity, slot] = count
            slot_centers.append(
                F.normalize(
                    source_features[active, slot].mean(dim=0), dim=0
                )
            )
        proxy[identity] = F.normalize(torch.stack(slot_centers).mean(dim=0), dim=0)
    if not bool(torch.isfinite(proxy).all()):
        raise RuntimeError("non-finite PC2P proxy")
    if not bool(
        torch.allclose(
            proxy.norm(dim=-1),
            torch.ones(NUM_IDENTITIES),
            atol=1e-6,
            rtol=1e-6,
        )
    ):
        raise RuntimeError("PC2P proxy rows are not unit normalized")
    proxy_array = proxy.numpy()
    row_view = proxy_array.view(
        np.dtype((np.void, proxy_array.dtype.itemsize * FEATURE_DIM))
    )
    if np.unique(row_view).size != NUM_IDENTITIES:
        raise RuntimeError("PC2P proxy contains duplicate rows")
    return proxy_array, slot_counts.numpy()


def main():
    args = parse_args()
    output = Path(args.output).expanduser()
    manifest = Path(args.manifest).expanduser()
    if not output.is_absolute() or not manifest.is_absolute():
        raise ValueError("PC2P outputs must use absolute paths")
    output = output.resolve()
    manifest = manifest.resolve()
    if output.exists() or manifest.exists():
        raise FileExistsError("PC2P bank outputs must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    source = _load_source_cache(
        args.source_cache,
        args.source_cache_sha256,
        args.pose_manifest_sha256,
    )
    dataset = OccludedDuke(root=args.data_root, verbose=False)
    pose_store = PoseTargetStore(
        args.pose_artifact, args.pose_manifest_sha256
    )
    contract = dataset_contract(dataset.train, pose_store)
    if tuple(source["paths"]) != contract["paths"]:
        raise RuntimeError("source cache order differs from official train order")
    if tuple(source["image_sha256"]) != contract["image_sha256"]:
        raise RuntimeError("source cache RGB SHA order differs from official train")
    proxy, slot_counts = _build_proxy(
        source, contract, dataset.train, pose_store
    )

    if subprocess.call(("git", "diff", "--quiet"), cwd=str(REPO_ROOT)) != 0:
        raise RuntimeError("PC2P builder requires a clean source tree")
    source_head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=str(REPO_ROOT), text=True
    ).strip()
    if re.fullmatch(r"[0-9a-f]{40}", source_head) is None:
        raise RuntimeError("could not freeze PC2P source HEAD")
    builder_sha256 = sha256_file(Path(__file__).resolve())
    loader_sha256 = sha256_file(
        REPO_ROOT / "model" / "pose_complete_clip_proxy.py"
    )
    pid_mapping_sha256 = ordered_digest(
        contract["relabel_to_original_pid"]
    )
    if pid_mapping_sha256 != contract["pid_mapping_sha256"]:
        raise RuntimeError("PC2P PID mapping digest is inconsistent")

    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            schema=np.asarray(SCHEMA),
            dataset=np.asarray(DATASET),
            split=np.asarray("train"),
            proxy=proxy,
            slot_counts=slot_counts.astype(np.int64, copy=False),
            relabel_to_original_pid=np.asarray(
                contract["relabel_to_original_pid"], dtype=np.int64
            ),
            official_paths_sha256=np.asarray(contract["paths_sha256"]),
            rgb_binding_sha256=np.asarray(contract["rgb_binding_sha256"]),
            pid_mapping_sha256=np.asarray(pid_mapping_sha256),
            source_cache_sha256=np.asarray(SOURCE_CACHE_SHA256),
            source_cache_schema=np.asarray(SOURCE_CACHE_SCHEMA),
            source_preprocessing=np.asarray(SOURCE_PREPROCESSING),
            source_pose_manifest_sha256=np.asarray(args.pose_manifest_sha256),
            source_clip_checkpoint_sha256=np.asarray(
                SOURCE_CLIP_CHECKPOINT_SHA256
            ),
            source_head=np.asarray(source["source_head"]),
            source_builder_sha256=np.asarray(
                source["source_builder_sha256"]
            ),
            source_teacher_source_sha256=np.asarray(
                source["source_teacher_source_sha256"]
            ),
            builder_sha256=np.asarray(builder_sha256),
            loader_source_sha256=np.asarray(loader_sha256),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(output))
    bank_sha256 = sha256_file(output)

    result = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "split": "train",
        "samples": NUM_SAMPLES,
        "identities": NUM_IDENTITIES,
        "proxy_shape": list(proxy.shape),
        "slot_counts_min": slot_counts.min(axis=0).tolist(),
        "slot_counts_max": slot_counts.max(axis=0).tolist(),
        "bank": str(output),
        "bank_sha256": bank_sha256,
        "source_cache": str(source["path"]),
        "source_cache_sha256": SOURCE_CACHE_SHA256,
        "source_pose_manifest_sha256": args.pose_manifest_sha256,
        "source_clip_checkpoint_sha256": SOURCE_CLIP_CHECKPOINT_SHA256,
        "source_head": source["source_head"],
        "source_builder_sha256": source["source_builder_sha256"],
        "source_teacher_source_sha256": source[
            "source_teacher_source_sha256"
        ],
        "official_paths_sha256": contract["paths_sha256"],
        "rgb_binding_sha256": contract["rgb_binding_sha256"],
        "pid_mapping_sha256": pid_mapping_sha256,
        "builder_sha256": builder_sha256,
        "loader_source_sha256": loader_sha256,
        "build_source_head": source_head,
    }
    manifest.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result["manifest"] = str(manifest)
    result["manifest_sha256"] = sha256_file(manifest)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
