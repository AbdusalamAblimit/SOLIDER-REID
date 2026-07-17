import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp371_casd.backfill_content_sha256 import (
    SCHEMA_VERSION,
    atomic_json,
    build_sidecar,
    file_sha256,
)


def _write_cache(path: Path, paths):
    torch.save({"paths": list(paths), "features": torch.zeros(len(paths), 2)}, path)


def test_sidecar_is_bound_to_cache_paths_and_detects_same_content(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.jpg").write_bytes(b"same")
    (repo / "b.jpg").write_bytes(b"same")
    (repo / "c.jpg").write_bytes(b"different")
    cache = tmp_path / "cache.pt"
    _write_cache(cache, ["a.jpg", "b.jpg", "c.jpg"])

    result = build_sidecar(cache, repo)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["source_cache_file_sha256"] == file_sha256(cache)
    assert result["sample_count"] == 3
    assert result["unique_content_count"] == 2
    assert result["duplicate_content_group_count"] == 1
    assert result["duplicate_content_sample_count"] == 2
    assert result["content_sha256"][0] == result["content_sha256"][1]
    assert result["content_sha256"][2] == hashlib.sha256(b"different").hexdigest()

    output = tmp_path / "sidecar.json"
    atomic_json(output, result)
    assert json.loads(output.read_text())["ordered_paths_sha256"] == result["ordered_paths_sha256"]


def test_duplicate_path_and_missing_file_fail_closed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.jpg").write_bytes(b"a")
    duplicate_cache = tmp_path / "duplicate.pt"
    _write_cache(duplicate_cache, ["a.jpg", "a.jpg"])
    with pytest.raises(ValueError, match="duplicate normalized paths"):
        build_sidecar(duplicate_cache, repo)

    missing_cache = tmp_path / "missing.pt"
    _write_cache(missing_cache, ["missing.jpg"])
    with pytest.raises(FileNotFoundError):
        build_sidecar(missing_cache, repo)
