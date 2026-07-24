#!/usr/bin/env python3
"""Build the once-only exp416 D0 candidate bank.

This is deliberately a train-only stage.  It manually enumerates
``Occluded_Duke/bounding_box_train`` and never constructs the repository
dataset class (which would also inspect query/gallery).  Pose and OpenCLIP are
not imported or read anywhere in this module.  Consequently the complete bank
is physically sealed before a later stage is allowed to consume either source
of semantic information.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import d0_feature_extractor as d0_features
import fuel_core
import fuel_io


SCHEMA = "exp416-pcnec-candidate-bank-v1"
RECEIPT_SCHEMA = "exp416-pcnec-candidate-bank-receipt-v1"
MANIFEST_SCHEMA = "exp416-pcnec-candidate-bank-manifest-v1"
STARTED_SCHEMA = "exp416-pcnec-candidate-bank-started-v1"
FAILURE_SCHEMA = "exp416-pcnec-candidate-bank-failure-v1"

EXPECTED_INTERPRETER = Path(
    "/usr/local/anaconda3/envs/mmpose-abu/bin/python"
)
FIXED_REPOSITORY_ROOT = Path(
    "/home/afr/SOLIDER-REID-exp416-pcnec-formal-v1"
)
FIXED_DATA_ROOT = Path("/mnt1/afrdata")
FIXED_TRAIN_DIR = FIXED_DATA_ROOT / "Occluded_Duke" / "bounding_box_train"
FIXED_D0_CONFIG = (
    FIXED_REPOSITORY_ROOT
    / "configs"
    / "occluded_duke"
    / "swin_tiny_tapf_d0.yml"
)
FIXED_D0_CHECKPOINT = Path(
    "/home/afr/SOLIDER-REID-exp387-d0-0d1822a/log/occluded_duke/"
    "exp387_clean_swin_tiny_d0_s1234/transformer_120.pth"
)
FIXED_OUTPUT_DIR = Path(
    "/home/afr/reid-clean/assets/exp416-pcnec-candidate-bank-v1"
)

EXPECTED_TRAIN_COUNT = 15_618
EXPECTED_TRAIN_PID_COUNT = 702
EXPECTED_CAMERA_COUNT = 8
IMPOSTOR_TOPK = 20
LOADER_BATCH = 64
D0_MICROBATCH = 8
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 128

SOURCE_FILES = {
    "builder": SCRIPT_PATH,
    "fuel_io": SCRIPT_DIR / "fuel_io.py",
    "fuel_core": SCRIPT_DIR / "fuel_core.py",
    "d0_feature_extractor": SCRIPT_DIR / "d0_feature_extractor.py",
}
NPZ_FIELDS = (
    "schema",
    "relative_paths",
    "raw_pids",
    "relabeled_pids",
    "camids",
    "image_sha256",
    "d0_global",
    "query_indices",
    "candidate_indices",
    "pair_is_impostor",
    "d0_distance",
    "query_offsets",
)
FORBIDDEN_MODULE_PREFIXES = (
    "open_clip",
    "mmpose",
    "datasets.pose",
    "datasets.occluded_duke",
)


@dataclass(frozen=True)
class TrainDatasetCardinality:
    """Only the cardinalities required to construct the frozen D0 model."""

    num_train_pids: int
    num_train_cams: int
    num_train_vids: int = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_existing(path: Path, *, directory: bool) -> Path:
    configured = Path(path).expanduser()
    if not configured.is_absolute():
        raise ValueError("formal paths must be absolute")
    resolved = configured.resolve(strict=True)
    if resolved != configured:
        raise RuntimeError("formal path must already be canonical: " + str(path))
    if directory and not resolved.is_dir():
        raise NotADirectoryError(resolved)
    if not directory and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _assert_forbidden_modules_absent() -> None:
    observed = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in FORBIDDEN_MODULE_PREFIXES
        )
    )
    if observed:
        raise RuntimeError(
            "candidate stage imported a forbidden semantic/test module: "
            + ", ".join(observed)
        )


def _git_file_is_tracked(repository: Path, path: Path) -> bool:
    result = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--error-unmatch",
            str(path.relative_to(repository)),
        ),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def enumerate_train_records(train_dir: Path) -> list[dict]:
    """Enumerate exactly one train directory without touching other splits."""
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        raise NotADirectoryError(train_dir)
    names = sorted(path.name for path in train_dir.iterdir() if path.is_file())
    unexpected = [
        name for name in names if Path(name).suffix.lower() != ".jpg"
    ]
    if unexpected:
        raise RuntimeError(
            "unexpected non-JPEG file in official train: " + unexpected[0]
        )
    relative_paths = [
        fuel_core.TRAIN_PREFIX + name for name in names
    ]
    records = fuel_core.build_train_records(relative_paths)
    for row in records:
        absolute = train_dir / Path(row["relative_path"]).name
        if (
            not absolute.is_file()
            or absolute.is_symlink()
            or absolute.parent.resolve() != train_dir.resolve()
        ):
            raise RuntimeError(
                "train image is missing, linked, or escaped: " + str(absolute)
            )
    return records


def _decode_train_rgb(path: Path) -> tuple[torch.Tensor, str]:
    payload = Path(path).read_bytes()
    image_sha256 = hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as source:
        if source.format != "JPEG":
            raise RuntimeError("official train input is not JPEG: " + str(path))
        source.load()
        rgb = source.convert("RGB")
        rgb = rgb.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            resample=Image.Resampling.BILINEAR,
        )
        array = np.array(rgb, dtype=np.uint8, copy=True)
    if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise RuntimeError("decoded RGB shape mismatch: " + str(path))
    tensor = (
        torch.from_numpy(np.ascontiguousarray(array))
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    if (
        tensor.shape != (3, IMAGE_HEIGHT, IMAGE_WIDTH)
        or not bool(torch.isfinite(tensor).all())
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise RuntimeError("decoded RGB tensor contract failed: " + str(path))
    return tensor, image_sha256


def extract_d0_globals(
    records: Sequence[Mapping],
    train_dir: Path,
    encode_batch: Callable[[torch.Tensor], torch.Tensor],
    *,
    loader_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream train RGB through a D0-only encoder in exact record order."""
    if int(loader_batch) <= 0:
        raise ValueError("loader_batch must be positive")
    descriptors = []
    image_shas = []
    expected_dimension = None
    for start in range(0, len(records), int(loader_batch)):
        rows = records[start : start + int(loader_batch)]
        decoded = [
            _decode_train_rgb(
                Path(train_dir) / Path(str(row["relative_path"])).name
            )
            for row in rows
        ]
        rgb = torch.stack([value[0] for value in decoded], dim=0)
        batch = encode_batch(rgb)
        if torch.is_tensor(batch):
            batch = batch.detach().cpu().numpy()
        batch = np.asarray(batch)
        if (
            batch.ndim != 2
            or batch.shape[0] != len(rows)
            or batch.shape[1] <= 0
            or batch.dtype.kind not in "fc"
            or not np.isfinite(batch).all()
        ):
            raise RuntimeError("D0 global encoder output contract failed")
        batch = batch.astype(np.float32, copy=False)
        norms = np.linalg.norm(batch.astype(np.float64), axis=1)
        if (
            np.any(norms <= 0.0)
            or not np.allclose(norms, 1.0, atol=2e-5, rtol=2e-5)
        ):
            raise RuntimeError("D0 global descriptor is not L2 normalized")
        if expected_dimension is None:
            expected_dimension = int(batch.shape[1])
        if int(batch.shape[1]) != expected_dimension:
            raise RuntimeError("D0 descriptor dimension changed across batches")
        descriptors.append(np.ascontiguousarray(batch))
        image_shas.extend(value[1] for value in decoded)
        del rgb, batch, decoded
    if not descriptors:
        raise RuntimeError("no D0 descriptors were extracted")
    matrix = np.ascontiguousarray(np.concatenate(descriptors, axis=0))
    shas = np.asarray(image_shas, dtype="<U64")
    if matrix.shape[0] != len(records) or shas.shape != (len(records),):
        raise RuntimeError("D0/image-SHA record alignment failed")
    return matrix, shas


def candidate_bank_arrays(
    records: Sequence[Mapping],
    descriptors: np.ndarray,
    image_shas: np.ndarray,
    *,
    impostor_topk: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return the exact on-disk arrays plus the validated rich bank receipt."""
    bank = fuel_core.construct_candidate_bank(
        records, descriptors, impostor_topk=int(impostor_topk)
    )
    fuel_core.validate_candidate_bank(records, descriptors, bank)
    pairs = bank["pairs"]
    offsets = [0]
    cursor = 0
    for query_order, query in enumerate(bank["query_receipts"]):
        count = int(query["true_count"]) + int(query["impostor_count"])
        group = pairs[cursor : cursor + count]
        if (
            len(group) != count
            or any(int(row["query_order"]) != query_order for row in group)
            or any(int(row["query_index"]) != int(query["query_index"]) for row in group)
        ):
            raise RuntimeError("candidate query grouping/order mismatch")
        cursor += count
        offsets.append(cursor)
    if cursor != len(pairs):
        raise RuntimeError("candidate query offsets do not cover every pair")

    arrays = {
        "schema": np.asarray([SCHEMA]),
        "relative_paths": np.asarray(
            [str(row["relative_path"]) for row in records]
        ),
        "raw_pids": np.asarray(
            [int(row["raw_pid"]) for row in records], dtype=np.int32
        ),
        "relabeled_pids": np.asarray(
            [int(row["train_pid"]) for row in records], dtype=np.int32
        ),
        "camids": np.asarray(
            [int(row["camera"]) for row in records], dtype=np.int16
        ),
        "image_sha256": np.asarray(image_shas, dtype="<U64"),
        "d0_global": np.asarray(descriptors, dtype=np.float32),
        "query_indices": np.asarray(
            [int(row["query_index"]) for row in pairs], dtype=np.int32
        ),
        "candidate_indices": np.asarray(
            [int(row["candidate_index"]) for row in pairs], dtype=np.int32
        ),
        "pair_is_impostor": np.asarray(
            [bool(row["impostor_positive"]) for row in pairs], dtype=np.bool_
        ),
        "d0_distance": np.asarray(
            [float(row["d0_distance"]) for row in pairs], dtype=np.float32
        ),
        "query_offsets": np.asarray(offsets, dtype=np.int64),
    }
    validate_candidate_arrays(arrays, records, bank)
    return arrays, bank


def validate_candidate_arrays(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[Mapping],
    bank: Mapping,
) -> None:
    if tuple(arrays) != NPZ_FIELDS:
        raise RuntimeError("candidate NPZ field order/schema mismatch")
    expected_dtypes = {
        "raw_pids": np.dtype(np.int32),
        "relabeled_pids": np.dtype(np.int32),
        "camids": np.dtype(np.int16),
        "d0_global": np.dtype(np.float32),
        "query_indices": np.dtype(np.int32),
        "candidate_indices": np.dtype(np.int32),
        "pair_is_impostor": np.dtype(np.bool_),
        "d0_distance": np.dtype(np.float32),
        "query_offsets": np.dtype(np.int64),
    }
    for name, dtype in expected_dtypes.items():
        if np.asarray(arrays[name]).dtype != dtype:
            raise RuntimeError("candidate dtype mismatch: " + name)
    record_count = len(records)
    pair_count = len(bank["pairs"])
    if np.asarray(arrays["schema"]).tolist() != [SCHEMA]:
        raise RuntimeError("candidate schema mismatch")
    for name in (
        "relative_paths",
        "raw_pids",
        "relabeled_pids",
        "camids",
        "image_sha256",
    ):
        if np.asarray(arrays[name]).shape != (record_count,):
            raise RuntimeError("record field shape mismatch: " + name)
    if (
        np.asarray(arrays["d0_global"]).ndim != 2
        or np.asarray(arrays["d0_global"]).shape[0] != record_count
    ):
        raise RuntimeError("D0 global matrix shape mismatch")
    for name in (
        "query_indices",
        "candidate_indices",
        "pair_is_impostor",
        "d0_distance",
    ):
        if np.asarray(arrays[name]).shape != (pair_count,):
            raise RuntimeError("pair field shape mismatch: " + name)
    offsets = np.asarray(arrays["query_offsets"])
    if (
        offsets.shape != (int(bank["eligible_query_count"]) + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != pair_count
        or np.any(np.diff(offsets) <= 0)
    ):
        raise RuntimeError("query offset contract failed")
    paths = np.asarray(arrays["relative_paths"]).tolist()
    if paths != [str(row["relative_path"]) for row in records]:
        raise RuntimeError("record path order changed")
    query = np.asarray(arrays["query_indices"]).astype(np.int64)
    candidate = np.asarray(arrays["candidate_indices"]).astype(np.int64)
    if (
        np.any(query < 0)
        or np.any(query >= record_count)
        or np.any(candidate < 0)
        or np.any(candidate >= record_count)
    ):
        raise RuntimeError("candidate indices leave the record table")
    relabeled = np.asarray(arrays["relabeled_pids"])
    cams = np.asarray(arrays["camids"])
    impostor = np.asarray(arrays["pair_is_impostor"])
    if not np.array_equal(impostor, relabeled[query] != relabeled[candidate]):
        raise RuntimeError("pair label disagrees with record PID")
    genuine = ~impostor
    if np.any(cams[query[genuine]] == cams[candidate[genuine]]):
        raise RuntimeError("same-camera genuine entered candidate bank")
    if (
        not np.isfinite(np.asarray(arrays["d0_global"])).all()
        or not np.isfinite(np.asarray(arrays["d0_distance"])).all()
        or np.any(np.asarray(arrays["d0_distance"]) < 0.0)
    ):
        raise RuntimeError("nonfinite/negative D0 quantity in candidate bank")


def _array_receipt(arrays: Mapping[str, np.ndarray]) -> dict:
    return {
        name: {
            "dtype": str(np.asarray(value).dtype),
            "shape": list(np.asarray(value).shape),
            "sha256": fuel_io.sha256_array(value),
        }
        for name, value in arrays.items()
    }


def _write_started(output_dir: Path, payload: Mapping) -> None:
    output_dir.mkdir(mode=0o700)
    fuel_io.atomic_json(output_dir / "started.json", dict(payload))
    fuel_io.readback_json(output_dir / "started.json", dict(payload))


def _seal_directory(output_dir: Path) -> None:
    for path in sorted(output_dir.iterdir()):
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    output_dir.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def _write_failure(output_dir: Path, *, stage: str, error: BaseException) -> None:
    if not output_dir.is_dir():
        return
    failure = {
        "schema": FAILURE_SCHEMA,
        "stage": str(stage),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "failed_at": utc_now(),
        "resume_allowed": False,
        "candidate_bank_complete": False,
        "pose_reads": 0,
        "openclip_reads": 0,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
    }
    path = output_dir / "failure.json"
    if not path.exists() and not path.with_name(path.name + ".tmp").exists():
        fuel_io.atomic_json(path, failure)
        fuel_io.readback_json(path, failure)


def build_and_seal(
    *,
    train_dir: Path,
    output_dir: Path,
    encode_batch: Callable[[torch.Tensor], torch.Tensor],
    started_payload: Mapping,
    provenance: Mapping,
    loader_batch: int,
    impostor_topk: int,
    seal_permissions: bool,
    post_candidate_verify: Callable[[], Mapping] | None = None,
) -> tuple[dict, dict]:
    """Execute one fresh physical D0-only bank stage."""
    stage = "fresh_namespace"
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("fresh candidate output required: " + str(output_dir))
    if not output_dir.parent.is_dir():
        raise NotADirectoryError(output_dir.parent)
    try:
        _write_started(output_dir, started_payload)
        stage = "train_only_enumeration"
        records = enumerate_train_records(train_dir)
        stage = "d0_global_extraction"
        descriptors, image_shas = extract_d0_globals(
            records,
            train_dir,
            encode_batch,
            loader_batch=int(loader_batch),
        )
        _assert_forbidden_modules_absent()
        stage = "d0_candidate_construction"
        arrays, bank = candidate_bank_arrays(
            records,
            descriptors,
            image_shas,
            impostor_topk=int(impostor_topk),
        )
        final_provenance = dict(provenance)
        if post_candidate_verify is not None:
            stage = "d0_asset_postverify"
            additions = dict(post_candidate_verify())
            overlap = set(final_provenance).intersection(additions)
            if overlap:
                raise RuntimeError(
                    "post-verify provenance overwrote fields: "
                    + ", ".join(sorted(overlap))
                )
            final_provenance.update(additions)
        stage = "candidate_npz_write"
        bank_path = output_dir / "candidate_bank.npz"
        fuel_io.atomic_npz(bank_path, arrays)
        observed = fuel_io.readback_npz(bank_path, arrays)
        validate_candidate_arrays(observed, records, bank)
        stage = "receipt_write"
        pairs = bank["pairs"]
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "created_at": utc_now(),
            "candidate_schema": SCHEMA,
            "train_only": True,
            "dataset_class_instantiations": 0,
            "query_reads": 0,
            "gallery_reads": 0,
            "pose_imports": 0,
            "pose_reads": 0,
            "openclip_imports": 0,
            "openclip_reads": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "resume_allowed": False,
            "record_count": int(len(records)),
            "raw_pid_count": int(
                len({int(row["raw_pid"]) for row in records})
            ),
            "camera_count": int(
                len({int(row["camera"]) for row in records})
            ),
            "descriptor_dimension": int(descriptors.shape[1]),
            "impostor_topk": int(impostor_topk),
            "camera_matched_impostors": bool(
                bank["camera_matched_impostors"]
            ),
            "eligible_query_count": int(bank["eligible_query_count"]),
            "excluded_query_count": int(
                len(bank["excluded_no_cross_camera_true"])
            ),
            "pair_count": int(len(pairs)),
            "genuine_pair_count": int(
                sum(not bool(row["impostor_positive"]) for row in pairs)
            ),
            "impostor_pair_count": int(
                sum(bool(row["impostor_positive"]) for row in pairs)
            ),
            "record_order_sha256": fuel_io.ordered_digest(
                row["relative_path"] for row in records
            ),
            "query_order_sha256": fuel_io.ordered_digest(
                row["query_id"] for row in bank["query_receipts"]
            ),
            "pair_order_sha256": fuel_io.ordered_digest(
                row["pair_id"] for row in pairs
            ),
            "image_asset_order_sha256": fuel_io.ordered_digest(
                "{}:{}".format(row["relative_path"], image_shas[index])
                for index, row in enumerate(records)
            ),
            "excluded_query_order_sha256": fuel_io.ordered_digest(
                bank["excluded_no_cross_camera_true"]
            ),
            "d0_distance_min": float(np.min(arrays["d0_distance"])),
            "d0_distance_max": float(np.max(arrays["d0_distance"])),
            "arrays": _array_receipt(arrays),
            "provenance": final_provenance,
        }
        receipt_path = output_dir / "receipt.json"
        fuel_io.atomic_json(receipt_path, receipt)
        fuel_io.readback_json(receipt_path, receipt)
        stage = "manifest_write"
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "sealed": True,
            "resume_allowed": False,
            "files": {
                name: {
                    "bytes": int((output_dir / name).stat().st_size),
                    "sha256": fuel_io.sha256_file(output_dir / name),
                }
                for name in (
                    "started.json",
                    "candidate_bank.npz",
                    "receipt.json",
                )
            },
        }
        manifest_path = output_dir / "manifest.json"
        fuel_io.atomic_json(manifest_path, manifest)
        fuel_io.readback_json(manifest_path, manifest)
        for name, expected in manifest["files"].items():
            path = output_dir / name
            if (
                int(path.stat().st_size) != int(expected["bytes"])
                or fuel_io.sha256_file(path) != expected["sha256"]
            ):
                raise RuntimeError("final manifest mismatch: " + name)
        if seal_permissions:
            _seal_directory(output_dir)
        return receipt, manifest
    except BaseException as error:
        try:
            _write_failure(output_dir, stage=stage, error=error)
            if seal_permissions and output_dir.is_dir():
                _seal_directory(output_dir)
        except BaseException:
            pass
        raise


def _formal_source_provenance(args) -> dict:
    expected_source = {
        "builder": args.expected_builder_sha256,
        "fuel_io": args.expected_fuel_io_sha256,
        "fuel_core": args.expected_fuel_core_sha256,
        "d0_feature_extractor": args.expected_d0_feature_extractor_sha256,
    }
    if any(not value for value in expected_source.values()):
        raise ValueError("all four frozen source SHA arguments are required")
    observed_source = {
        name: fuel_io.sha256_file(path) for name, path in SOURCE_FILES.items()
    }
    if observed_source != expected_source:
        raise RuntimeError(
            "frozen source SHA mismatch: "
            + json.dumps(
                {"expected": expected_source, "observed": observed_source},
                sort_keys=True,
            )
        )
    for path in SOURCE_FILES.values():
        if not _git_file_is_tracked(REPOSITORY_ROOT, path):
            raise RuntimeError("formal source is untracked: " + str(path))
    return observed_source


def validate_formal(args) -> dict:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("formal bank requires PYTHONDONTWRITEBYTECODE=1")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("formal bank requires PYTHONHASHSEED=0")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "formal bank requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    observed_interpreter = Path(sys.executable).resolve()
    if observed_interpreter != EXPECTED_INTERPRETER.resolve(strict=True):
        raise RuntimeError("formal interpreter mismatch")
    if REPOSITORY_ROOT.resolve(strict=True) != FIXED_REPOSITORY_ROOT:
        raise RuntimeError("formal repository path mismatch")
    fixed = {
        "data_root": FIXED_DATA_ROOT,
        "train_dir": FIXED_TRAIN_DIR,
        "d0_config": FIXED_D0_CONFIG,
        "d0_checkpoint": FIXED_D0_CHECKPOINT,
        "output_dir": FIXED_OUTPUT_DIR,
    }
    observed = {
        "data_root": Path(args.data_root).expanduser(),
        "train_dir": Path(args.data_root).expanduser()
        / "Occluded_Duke"
        / "bounding_box_train",
        "d0_config": Path(args.d0_config).expanduser(),
        "d0_checkpoint": Path(args.d0_checkpoint).expanduser(),
        "output_dir": Path(args.output_dir).expanduser(),
    }
    for name, expected in fixed.items():
        if observed[name] != expected:
            raise RuntimeError("formal fixed path mismatch: " + name)
    _canonical_existing(observed["data_root"], directory=True)
    _canonical_existing(observed["train_dir"], directory=True)
    _canonical_existing(observed["d0_config"], directory=False)
    _canonical_existing(observed["d0_checkpoint"], directory=False)
    if observed["output_dir"].exists():
        raise FileExistsError("fresh candidate output required")
    if not observed["output_dir"].parent.is_dir():
        raise NotADirectoryError(observed["output_dir"].parent)
    if (
        int(args.impostor_topk) != IMPOSTOR_TOPK
        or int(args.loader_batch) != LOADER_BATCH
        or int(args.d0_microbatch) != D0_MICROBATCH
        or str(args.device) != "cuda:0"
    ):
        raise RuntimeError("formal candidate hyperparameter mismatch")
    if (
        fuel_io.sha256_file(observed["d0_config"])
        != d0_features.SEALED_D0_CONFIG_SHA256
        or fuel_io.sha256_file(observed["d0_checkpoint"])
        != d0_features.SEALED_D0_CHECKPOINT_SHA256
    ):
        raise RuntimeError("sealed D0 asset SHA mismatch")
    head = fuel_io.git_head(REPOSITORY_ROOT)
    if not args.expected_head or str(args.expected_head) != head:
        raise RuntimeError("formal HEAD mismatch")
    if fuel_io.git_tracked_status(REPOSITORY_ROOT):
        raise RuntimeError("formal tracked worktree is dirty")
    if fuel_io.git_index_status(REPOSITORY_ROOT):
        raise RuntimeError("formal index is dirty")
    source_shas = _formal_source_provenance(args)
    _assert_forbidden_modules_absent()
    fuel_io.assert_no_cuda_compute_processes()
    records = enumerate_train_records(observed["train_dir"])
    raw_pids = {int(row["raw_pid"]) for row in records}
    cameras = {int(row["camera"]) for row in records}
    if (
        len(records) != EXPECTED_TRAIN_COUNT
        or len(raw_pids) != EXPECTED_TRAIN_PID_COUNT
        or len(cameras) != EXPECTED_CAMERA_COUNT
        or cameras != set(range(EXPECTED_CAMERA_COUNT))
    ):
        raise RuntimeError("official train cardinality mismatch")
    return {
        "paths": observed,
        "head": head,
        "source_sha256": source_shas,
        "preseal_record_order_sha256": fuel_io.ordered_digest(
            row["relative_path"] for row in records
        ),
        "interpreter": str(observed_interpreter),
    }


def run_formal(args) -> tuple[dict, dict]:
    validated = validate_formal(args)
    paths = validated["paths"]
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dataset = TrainDatasetCardinality(
        num_train_pids=EXPECTED_TRAIN_PID_COUNT,
        num_train_cams=EXPECTED_CAMERA_COUNT,
        num_train_vids=1,
    )
    extractor = None
    checkpoint_sha_before = fuel_io.sha256_file(paths["d0_checkpoint"])
    config_sha_before = fuel_io.sha256_file(paths["d0_config"])
    provenance = {
        "formal_head": validated["head"],
        "source_sha256": validated["source_sha256"],
        "interpreter": validated["interpreter"],
        "repository_root": str(REPOSITORY_ROOT),
        "official_data_root": str(paths["data_root"]),
        "official_train_dir": str(paths["train_dir"]),
        "d0_config": str(paths["d0_config"]),
        "d0_config_sha256_before": config_sha_before,
        "d0_checkpoint": str(paths["d0_checkpoint"]),
        "d0_checkpoint_sha256_before": checkpoint_sha_before,
        "preseal_record_order_sha256": validated[
            "preseal_record_order_sha256"
        ],
        "loader_batch": int(args.loader_batch),
        "d0_microbatch": int(args.d0_microbatch),
        "device": str(args.device),
        "rgb_resize": {
            "library": "PIL",
            "mode": "RGB",
            "height": IMAGE_HEIGHT,
            "width": IMAGE_WIDTH,
            "interpolation": "BILINEAR",
            "tensor_dtype": "torch.float32",
            "tensor_range": [0.0, 1.0],
        },
        "determinism": {
            "torch_seed": 1234,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
        },
    }
    started = {
        "schema": STARTED_SCHEMA,
        "started_at": utc_now(),
        "formal_head": validated["head"],
        "source_sha256": validated["source_sha256"],
        "train_only": True,
        "resume_allowed": False,
        "pose_reads": 0,
        "openclip_reads": 0,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
    }
    try:
        def encode_batch(rgb):
            nonlocal extractor
            if extractor is None:
                extractor = d0_features.SealedD0FeatureExtractor(
                    config_path=paths["d0_config"],
                    config_sha256=d0_features.SEALED_D0_CONFIG_SHA256,
                    checkpoint_path=paths["d0_checkpoint"],
                    checkpoint_sha256=d0_features.SEALED_D0_CHECKPOINT_SHA256,
                    dataset=dataset,
                    device=args.device,
                    microbatch=int(args.d0_microbatch),
                )
            result = extractor.encode(rgb)
            global_features = result["global_features"].detach().cpu()
            del result
            return global_features

        def post_candidate_verify():
            nonlocal extractor
            if extractor is None:
                raise RuntimeError("D0 extractor was never constructed")
            extractor.close()
            extractor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            checkpoint_sha_after = fuel_io.sha256_file(
                paths["d0_checkpoint"]
            )
            config_sha_after = fuel_io.sha256_file(paths["d0_config"])
            if checkpoint_sha_after != checkpoint_sha_before:
                raise RuntimeError(
                    "sealed D0 checkpoint changed during bank build"
                )
            if config_sha_after != config_sha_before:
                raise RuntimeError("sealed D0 config changed during bank build")
            return {
                "d0_config_sha256_after": config_sha_after,
                "d0_checkpoint_sha256_after": checkpoint_sha_after,
                "d0_model_closed_before_bank_write": True,
            }

        receipt, manifest = build_and_seal(
            train_dir=paths["train_dir"],
            output_dir=paths["output_dir"],
            encode_batch=encode_batch,
            started_payload=started,
            provenance=provenance,
            loader_batch=int(args.loader_batch),
            impostor_topk=int(args.impostor_topk),
            seal_permissions=True,
            post_candidate_verify=post_candidate_verify,
        )
    finally:
        if extractor is not None:
            extractor.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if fuel_io.sha256_file(paths["d0_checkpoint"]) != checkpoint_sha_before:
        raise RuntimeError("sealed D0 checkpoint changed after bank seal")
    if fuel_io.sha256_file(paths["d0_config"]) != config_sha_before:
        raise RuntimeError("sealed D0 config changed after bank seal")
    print("EXP416_CANDIDATE_BANK=SEALED")
    print("CANDIDATE_BANK_SHA256=" + manifest["files"]["candidate_bank.npz"]["sha256"])
    print("RECEIPT_SHA256=" + manifest["files"]["receipt.json"]["sha256"])
    return receipt, manifest


def _write_toy_jpeg(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.meshgrid(
        np.arange(IMAGE_HEIGHT, dtype=np.uint16),
        np.arange(IMAGE_WIDTH, dtype=np.uint16),
        indexing="ij",
    )
    array = np.stack(
        (
            (xx + int(seed) * 17) % 256,
            (yy + int(seed) * 29) % 256,
            ((xx + yy) * (int(seed) + 1)) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    noise = rng.integers(0, 4, size=array.shape, dtype=np.uint8)
    array = (array + noise).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(
        path, format="JPEG", quality=91, optimize=False
    )


def _mock_encoder(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4:
        raise ValueError("mock RGB shape mismatch")
    mean = rgb.mean(dim=(2, 3))
    variance = rgb.var(dim=(2, 3), unbiased=False)
    descriptor = torch.cat((mean, variance, mean[:, :1] + variance[:, 1:2]), dim=1)
    return torch.nn.functional.normalize(descriptor.float(), dim=1)


def run_self_test() -> None:
    _assert_forbidden_modules_absent()
    temporary_root = REPOSITORY_ROOT / "tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="exp416-bank-selftest-", dir=temporary_root
    ) as temporary:
        root = Path(temporary)
        train_dir = root / "bounding_box_train"
        train_dir.mkdir()
        # Four identities, two cameras each: every query has one genuine and
        # at least two impostors.  Filenames deliberately arrive unsorted.
        names = [
            "0004_c2_f0002.jpg",
            "0001_c1_f0001.jpg",
            "0003_c2_f0002.jpg",
            "0002_c1_f0001.jpg",
            "0001_c2_f0002.jpg",
            "0004_c1_f0001.jpg",
            "0002_c2_f0002.jpg",
            "0003_c1_f0001.jpg",
        ]
        for index, name in enumerate(names):
            _write_toy_jpeg(train_dir / name, seed=index + 1)
        started = {
            "schema": STARTED_SCHEMA,
            "started_at": "2000-01-01T00:00:00+00:00",
            "resume_allowed": False,
        }
        output = root / "bank"
        receipt, manifest = build_and_seal(
            train_dir=train_dir,
            output_dir=output,
            encode_batch=_mock_encoder,
            started_payload=started,
            provenance={"mode": "cpu-mock-self-test"},
            loader_batch=3,
            impostor_topk=2,
            seal_permissions=False,
        )
        arrays = fuel_io.load_npz_exact(
            output / "candidate_bank.npz", expected_fields=NPZ_FIELDS
        )
        records = enumerate_train_records(train_dir)
        descriptors, image_shas = extract_d0_globals(
            records, train_dir, _mock_encoder, loader_batch=3
        )
        bank = fuel_core.construct_candidate_bank(
            records, descriptors, impostor_topk=2
        )
        validate_candidate_arrays(arrays, records, bank)
        if receipt["record_count"] != 8 or receipt["eligible_query_count"] != 8:
            raise AssertionError("mock bank cardinality mismatch")
        if receipt["camera_matched_impostors"] is not True:
            raise AssertionError("mock bank camera matching is disabled")
        if receipt["impostor_pair_count"] != 16:
            raise AssertionError("mock bank impostor count mismatch")
        if manifest["files"]["candidate_bank.npz"]["sha256"] != fuel_io.sha256_file(
            output / "candidate_bank.npz"
        ):
            raise AssertionError("mock bank manifest SHA mismatch")
        if not np.array_equal(arrays["image_sha256"], image_shas):
            raise AssertionError("mock bank image SHA order mismatch")

        repeat_output = root / "bank-repeat"
        repeat_receipt, repeat_manifest = build_and_seal(
            train_dir=train_dir,
            output_dir=repeat_output,
            encode_batch=_mock_encoder,
            started_payload=started,
            provenance={"mode": "cpu-mock-self-test"},
            loader_batch=3,
            impostor_topk=2,
            seal_permissions=False,
        )
        if (
            repeat_manifest["files"]["candidate_bank.npz"]["sha256"]
            != manifest["files"]["candidate_bank.npz"]["sha256"]
            or repeat_receipt["arrays"] != receipt["arrays"]
            or repeat_receipt["record_order_sha256"]
            != receipt["record_order_sha256"]
            or repeat_receipt["pair_order_sha256"]
            != receipt["pair_order_sha256"]
        ):
            raise AssertionError("mock candidate bank is not byte deterministic")

        # Freshness is a hard once-only gate.
        try:
            build_and_seal(
                train_dir=train_dir,
                output_dir=output,
                encode_batch=_mock_encoder,
                started_payload=started,
                provenance={},
                loader_batch=3,
                impostor_topk=2,
                seal_permissions=False,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing bank namespace was reused")

        # Failure injection must leave a permanent non-resumable receipt and
        # must never emit a seemingly complete candidate NPZ.
        calls = {"count": 0}

        def exploding_encoder(rgb):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("injected D0 encoder failure")
            return _mock_encoder(rgb)

        failed = root / "failed-bank"
        try:
            build_and_seal(
                train_dir=train_dir,
                output_dir=failed,
                encode_batch=exploding_encoder,
                started_payload=started,
                provenance={},
                loader_batch=3,
                impostor_topk=2,
                seal_permissions=False,
            )
        except RuntimeError as error:
            if "injected D0 encoder failure" not in str(error):
                raise
        else:
            raise AssertionError("injected encoder failure was ignored")
        failure = fuel_io.readback_json(failed / "failure.json")
        if (
            failure["resume_allowed"]
            or failure["candidate_bank_complete"]
            or (failed / "candidate_bank.npz").exists()
        ):
            raise AssertionError("failed bank was not fail-closed")

        # A truncated batch and nonfinite descriptor are independently fatal.
        for mode in ("truncated", "nonfinite"):
            def bad_encoder(rgb, selected=mode):
                value = _mock_encoder(rgb)
                if selected == "truncated":
                    return value[:-1]
                value[0, 0] = float("nan")
                return value

            bad_output = root / ("bad-" + mode)
            try:
                build_and_seal(
                    train_dir=train_dir,
                    output_dir=bad_output,
                    encode_batch=bad_encoder,
                    started_payload=started,
                    provenance={},
                    loader_batch=3,
                    impostor_topk=2,
                    seal_permissions=False,
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError(mode + " D0 output was accepted")
            if not (bad_output / "failure.json").is_file():
                raise AssertionError(mode + " failure receipt is absent")
    _assert_forbidden_modules_absent()
    print("EXP416_CANDIDATE_BANK_SELF_TEST=PASS")


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--data-root", default=str(FIXED_DATA_ROOT))
    parser.add_argument("--d0-config", default=str(FIXED_D0_CONFIG))
    parser.add_argument("--d0-checkpoint", default=str(FIXED_D0_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(FIXED_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--loader-batch", type=int, default=LOADER_BATCH)
    parser.add_argument("--d0-microbatch", type=int, default=D0_MICROBATCH)
    parser.add_argument("--impostor-topk", type=int, default=IMPOSTOR_TOPK)
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-builder-sha256")
    parser.add_argument("--expected-fuel-io-sha256")
    parser.add_argument("--expected-fuel-core-sha256")
    parser.add_argument("--expected-d0-feature-extractor-sha256")
    args = parser.parse_args(argv)
    if args.self_test == args.formal:
        parser.error("choose exactly one of --self-test or --formal")
    return args


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    run_formal(args)


if __name__ == "__main__":
    main()
