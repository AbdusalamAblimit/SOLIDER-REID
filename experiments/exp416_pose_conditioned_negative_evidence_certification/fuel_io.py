#!/usr/bin/env python3
"""Strict I/O and provenance primitives for the exp416 fuel audit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def ordered_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_json_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_tmp(path: Path) -> Path:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("fresh output and .tmp are required: " + str(path))
    if not path.parent.is_dir():
        raise NotADirectoryError(path.parent)
    return temporary


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, payload) -> None:
    path = Path(path)
    temporary = _exclusive_tmp(path)
    encoded = stable_json_bytes(payload)
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)


def _validate_npz_arrays(arrays) -> dict[str, np.ndarray]:
    validated = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("NPZ field names must be nonempty strings")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays are forbidden: " + name)
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise ValueError("non-finite NPZ array: " + name)
        validated[name] = np.ascontiguousarray(array)
    if not validated:
        raise ValueError("NPZ payload cannot be empty")
    return validated


def atomic_npz(path, arrays, *, compressed=True) -> None:
    path = Path(path)
    temporary = _exclusive_tmp(path)
    validated = _validate_npz_arrays(arrays)
    with temporary.open("xb") as handle:
        writer = np.savez_compressed if compressed else np.savez
        writer(handle, **validated)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)


def readback_json(path, expected=None):
    path = Path(path)
    with path.open("rb") as handle:
        observed = json.load(handle)
    if expected is not None and observed != expected:
        raise RuntimeError("JSON exact readback failed: " + str(path))
    if stable_json_bytes(observed) != path.read_bytes():
        raise RuntimeError("JSON canonical byte readback failed: " + str(path))
    return observed


def load_npz_exact(path, expected_fields=None) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as source:
        if expected_fields is not None and set(source.files) != set(
            expected_fields
        ):
            raise RuntimeError("NPZ field set mismatch: " + str(path))
        arrays = {name: source[name].copy() for name in source.files}
    return _validate_npz_arrays(arrays)


def readback_npz(path, expected) -> dict[str, np.ndarray]:
    validated = _validate_npz_arrays(expected)
    observed = load_npz_exact(path, validated)
    for name, value in validated.items():
        actual = observed[name]
        if actual.dtype != value.dtype or actual.shape != value.shape:
            raise RuntimeError("NPZ shape/dtype readback failed: " + name)
        if not np.array_equal(actual, value, equal_nan=False):
            raise RuntimeError("NPZ value readback failed: " + name)
    return observed


def git_head(repository) -> str:
    return subprocess.check_output(
        ("git", "-C", str(Path(repository)), "rev-parse", "HEAD"),
        text=True,
    ).strip()


def git_tracked_status(repository) -> list[str]:
    output = subprocess.check_output(
        (
            "git",
            "-C",
            str(Path(repository)),
            "status",
            "--short",
            "--untracked-files=no",
        ),
        text=True,
    )
    return [line for line in output.splitlines() if line]


def git_index_status(repository) -> list[str]:
    output = subprocess.check_output(
        (
            "git",
            "-C",
            str(Path(repository)),
            "diff",
            "--cached",
            "--name-only",
        ),
        text=True,
    )
    return [line for line in output.splitlines() if line]


def cuda_compute_processes() -> list[dict[str, str]]:
    result = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi compute-process query failed")
    processes = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError("unexpected nvidia-smi process row")
        processes.append(
            {
                "pid": fields[0],
                "used_memory_mib": fields[1],
                "process_name": fields[2],
            }
        )
    return processes


def assert_no_cuda_compute_processes() -> None:
    processes = cuda_compute_processes()
    if processes:
        raise RuntimeError(
            "exp416 requires an idle CUDA device: "
            + json.dumps(processes, sort_keys=True)
        )


def run_self_test() -> None:
    payload = {"z": [1, 2], "a": "汉字", "finite": 1.25}
    encoded = stable_json_bytes(payload)
    if sha256_bytes(encoded) != (
        "13deae12fb35893c30f632cbdac1e1e1fc559d5c96f1176b19d65daad5dc3d79"
    ):
        raise AssertionError("stable JSON known-answer mismatch")
    vector = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    if sha256_array(vector) != (
        "91d8c3837cfcbc5717d4b636e98a2f314d9f5e72020362044c30ca8a308921c0"
    ):
        raise AssertionError("array SHA known-answer mismatch")
    if ordered_digest(("a", "b", 3)) != (
        "516d94acaca9e39372050170cd0007d4b0bbb9465337d685b4a1f02a3fb9e3d9"
    ):
        raise AssertionError("ordered digest known-answer mismatch")
    try:
        _validate_npz_arrays({"bad": np.asarray([float("nan")])})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite arrays must be rejected")
    print("EXP416_FUEL_IO_SELF_TEST=PASS")


if __name__ == "__main__":
    run_self_test()
