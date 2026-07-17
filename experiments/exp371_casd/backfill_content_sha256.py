#!/usr/bin/env python3
"""Bind per-image content hashes to an existing exp371 support cache.

The frozen feature cache is not rewritten.  A small sidecar is produced and
bound to both the source-cache SHA256 and the ordered path-list SHA256 so Gate
C v2 can reject same-content leakage even when two files have different paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping

import torch


SCHEMA_VERSION = "exp371_content_sha256_sidecar_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_image_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def build_sidecar(cache_path: Path, repo_root: Path) -> Dict[str, object]:
    cache_path = cache_path.resolve()
    repo_root = repo_root.resolve()
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, Mapping) or "paths" not in payload:
        raise ValueError("cache payload must contain ordered paths")
    paths = [str(value) for value in payload["paths"]]
    if len(paths) != int(payload["features"].shape[0]):
        raise ValueError("path count does not match cached feature count")
    if len(set(paths)) != len(paths):
        raise ValueError("cache contains duplicate normalized paths")

    content_sha256: List[str] = [
        file_sha256(resolve_image_path(path, repo_root)) for path in paths
    ]
    counts = Counter(content_sha256)
    duplicated_groups = [count for count in counts.values() if count > 1]
    sidecar: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_cache_path": str(cache_path),
        "source_cache_file_sha256": file_sha256(cache_path),
        "ordered_paths_sha256": json_sha256(paths),
        "sample_count": len(paths),
        "content_sha256": content_sha256,
        "unique_content_count": len(counts),
        "duplicate_content_group_count": len(duplicated_groups),
        "duplicate_content_sample_count": sum(duplicated_groups),
    }
    return sidecar


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecar = build_sidecar(args.cache, args.repo_root)
    atomic_json(args.output, sidecar)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "source_cache_file_sha256": sidecar["source_cache_file_sha256"],
        "sample_count": sidecar["sample_count"],
        "unique_content_count": sidecar["unique_content_count"],
        "duplicate_content_group_count": sidecar["duplicate_content_group_count"],
        "duplicate_content_sample_count": sidecar["duplicate_content_sample_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
