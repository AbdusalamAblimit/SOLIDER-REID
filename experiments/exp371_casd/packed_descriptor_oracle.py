#!/usr/bin/env python3
"""Gate D: fit train-only 5376-D -> 768-D frozen packing oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import euclidean_distance, eval_func  # noqa: E402
from experiments.exp371_casd.intervention_utils import (  # noqa: E402
    descriptor_gain_retention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", choices=("jl", "pca"), default=("jl", "pca"))
    parser.add_argument("--seed", type=int, default=371)
    parser.add_argument("--pca-niter", type=int, default=4)
    parser.add_argument("--projection-batch", type=int, default=512)
    parser.add_argument("--expected-full-map", type=float, default=None)
    parser.add_argument("--expected-global-map", type=float, default=None)
    parser.add_argument("--parity-tolerance", type=float, default=0.001)
    return parser.parse_args()


def atomic_json(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(path: Path) -> Dict:
    try:
        cache = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        cache = torch.load(path, map_location="cpu")
    required = {
        "features", "pids", "camids", "paths", "split", "mode",
        "num_query", "block_dim", "weight_sha256",
    }
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"cache {path} is missing keys: {sorted(missing)}")
    features = cache["features"]
    block_dim = int(cache["block_dim"])
    if features.ndim != 2 or features.shape[1] != 7 * block_dim:
        raise ValueError(f"invalid equal-concat cache shape: {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError(f"cache {path} contains NaN/Inf")
    return cache


def evaluate(features: torch.Tensor, cache: Dict) -> Dict[str, float]:
    num_query = int(cache["num_query"])
    if num_query <= 0:
        raise ValueError("evaluation cache requires num_query > 0")
    normed = F.normalize(features.float(), p=2, dim=1)
    qf, gf = normed[:num_query], normed[num_query:]
    pids = np.asarray(cache["pids"])
    camids = np.asarray(cache["camids"])
    distmat = euclidean_distance(qf, gf)
    cmc, mean_ap = eval_func(
        distmat,
        pids[:num_query], pids[num_query:],
        camids[:num_query], camids[num_query:],
    )
    return {
        "mAP": float(mean_ap),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
    }


def project_in_chunks(
    features: torch.Tensor,
    matrix: torch.Tensor,
    mean: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    projected = []
    for start in range(0, features.shape[0], batch_size):
        chunk = features[start:start + batch_size].to(matrix.device, non_blocking=True).float()
        if mean is not None:
            chunk = chunk - mean
        projected.append((chunk @ matrix).cpu())
    return torch.cat(projected, dim=0)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train_cache).resolve()
    val_path = Path(args.val_cache).resolve()
    train = load_cache(train_path)
    val = load_cache(val_path)

    if train["mode"] != "correct" or val["mode"] != "correct":
        raise ValueError("Gate D only accepts correct-pose caches")
    if train["weight_sha256"] != val["weight_sha256"]:
        raise ValueError("train/val caches came from different checkpoints")
    if int(train["block_dim"]) != 768 or int(val["block_dim"]) != 768:
        raise ValueError("Gate D is pre-registered for 7x768 -> 768 only")
    train_paths = set(str(path) for path in train["paths"])
    val_paths = set(str(path) for path in val["paths"])
    overlap = train_paths.intersection(val_paths)
    if overlap:
        raise ValueError(f"train/eval path leakage detected: {len(overlap)} paths")

    x_train = train["features"].float().contiguous()
    x_val = val["features"].float().contiguous()
    global_features = x_val[:, :768]
    full_metrics = evaluate(x_val, val)
    global_metrics = evaluate(global_features, val)
    if args.expected_full_map is not None and abs(full_metrics["mAP"] - args.expected_full_map) > args.parity_tolerance:
        raise AssertionError(
            f"full mAP parity failed: {full_metrics['mAP']:.6f} vs {args.expected_full_map:.6f}"
        )
    if args.expected_global_map is not None and abs(global_metrics["mAP"] - args.expected_global_map) > args.parity_tolerance:
        raise AssertionError(
            f"global mAP parity failed: {global_metrics['mAP']:.6f} vs {args.expected_global_map:.6f}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict[str, Dict] = {
        "global": global_metrics,
        "full_5376": full_metrics,
    }
    manifest = {
        "train_cache": str(train_path),
        "train_cache_sha256": file_sha256(train_path),
        "val_cache": str(val_path),
        "val_cache_sha256": file_sha256(val_path),
        "checkpoint_sha256": train["weight_sha256"],
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "input_dim": int(x_train.shape[1]),
        "output_dim": 768,
        "train_val_path_overlap": 0,
        "seed": int(args.seed),
        "device": str(device),
        "methods": list(args.methods),
        "retention_gate": 0.80,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(output_dir / "results.json", results)

    if "jl" in args.methods:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        matrix = torch.randint(
            0, 2, (x_train.shape[1], 768), generator=generator, dtype=torch.int8
        ).float()
        matrix.mul_(2.0).sub_(1.0).div_(math.sqrt(768.0))
        matrix = matrix.to(device)
        packed = project_in_chunks(
            x_val, matrix, None, args.projection_batch
        )
        metrics = evaluate(packed, val)
        metrics["retention"] = descriptor_gain_retention(
            metrics["mAP"], global_metrics["mAP"], full_metrics["mAP"]
        )
        results["jl_768"] = metrics
        torch.save(
            {"matrix": matrix.cpu(), "seed": args.seed, "method": "rademacher_jl"},
            output_dir / "jl_768.pt",
        )
        atomic_json(output_dir / "results.json", results)
        print(json.dumps({"jl_768": metrics}, indent=2), flush=True)

    if "pca" in args.methods:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        train_device = x_train.to(device)
        mean = train_device.mean(dim=0, keepdim=True)
        centered = train_device - mean
        _u, singular_values, components = torch.pca_lowrank(
            centered,
            q=768,
            center=False,
            niter=args.pca_niter,
        )
        total_energy = centered.square().sum()
        explained_ratio = singular_values.square().sum() / total_energy.clamp_min(1e-12)
        packed = project_in_chunks(
            x_val, components, mean, args.projection_batch
        )
        metrics = evaluate(packed, val)
        metrics["retention"] = descriptor_gain_retention(
            metrics["mAP"], global_metrics["mAP"], full_metrics["mAP"]
        )
        metrics["train_explained_variance_ratio"] = float(explained_ratio.item())

        # Geometry fidelity is measured only on deterministic train samples.
        probe_count = min(2048, x_train.shape[0])
        probe = x_train[:probe_count].to(device)
        probe_packed = (probe - mean) @ components
        probe_reconstructed = probe_packed @ components.t() + mean
        metrics["train_reconstruction_cosine"] = float(
            F.cosine_similarity(probe, probe_reconstructed, dim=1).mean().item()
        )
        results["pca_768"] = metrics
        torch.save(
            {
                "mean": mean.cpu(),
                "components": components.cpu(),
                "singular_values": singular_values.cpu(),
                "seed": args.seed,
                "niter": args.pca_niter,
                "method": "train_only_randomized_pca",
            },
            output_dir / "pca_768.pt",
        )
        atomic_json(output_dir / "results.json", results)
        print(json.dumps({"pca_768": metrics}, indent=2), flush=True)
        del train_device, centered

    passed = {
        key: bool(value.get("retention", float("-inf")) >= 0.80)
        for key, value in results.items() if "retention" in value
    }
    results["gate"] = {
        "threshold": 0.80,
        "method_pass": passed,
        "simple_packing_go": bool(results.get("pca_768", {}).get("retention", -1) >= 0.80),
    }
    atomic_json(output_dir / "results.json", results)
    print(json.dumps(results["gate"], indent=2), flush=True)
    print(f"COMPLETE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
