#!/usr/bin/env python3
"""Build the once-only stage-3 feature/energy cache for exp416 PC-NEC.

The formal execution order is deliberately physical:

1. validate the already sealed D0 candidate bank and pose-only geometry;
2. load the sealed D0, re-read canonical RGB, require global descriptors to be
   bit-exact with the bank, pool ``featmaps[-1]``, and close D0;
3. only then construct OpenCLIP, re-read the same canonical RGB with exact
   tensor hashes, encode instance/canonical crops and full images, and close it;
4. compute every fixed pair/control row without changing the bank denominator;
5. atomically publish one cache and SHA-bound receipts.

The module has a CPU-only mock self-test.  It never starts a real GPU job from
the local workstation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
for import_root in (REPOSITORY_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import clip_crop_encoder
import d0_feature_extractor
import fuel_core
import fuel_io


SCHEMA = "exp416-pcnec-fuel-cache-v1"
RECEIPT_SCHEMA = "exp416-pcnec-fuel-cache-receipt-v1"
RESULT_SCHEMA = "exp416-pcnec-fuel-cache-result-v1"
MANIFEST_SCHEMA = "exp416-pcnec-fuel-cache-manifest-v1"
BANK_SCHEMA = "exp416-pcnec-candidate-bank-v1"
GEOMETRY_SCHEMA = "exp416-pcnec-geometry-v1"

EXPECTED_INTERPRETER = Path(
    "/usr/local/anaconda3/envs/mmpose-abu/bin/python"
)
FIXED_REPOSITORY_ROOT = Path(
    "/home/afr/SOLIDER-REID-exp416-pcnec-formal-v1"
)
FIXED_DATA_ROOT = Path("/mnt1/afrdata")
FIXED_IMAGE_ROOT = FIXED_DATA_ROOT / "Occluded_Duke"
FIXED_BANK = Path(
    "/home/afr/reid-clean/assets/exp416-pcnec-candidate-bank-v1/"
    "candidate_bank.npz"
)
FIXED_GEOMETRY = Path(
    "/home/afr/reid-clean/assets/exp416-pcnec-geometry-v1/geometry.npz"
)
FIXED_OUTPUT_DIR = Path(
    "/home/afr/reid-clean/assets/exp416-pcnec-fuel-v1"
)
FIXED_D0_CHECKPOINT = Path(
    "/home/afr/SOLIDER-REID-exp387-d0-0d1822a/log/occluded_duke/"
    "exp387_clean_swin_tiny_d0_s1234/transformer_120.pth"
)
FIXED_CLIP_CHECKPOINT = Path(
    "/home/afr/reid-clean/weights/"
    "exp401_clip_l14_openclip_9ce2e8a8.safetensors"
)
FIXED_D0_CONFIG_RELATIVE = Path(
    "configs/occluded_duke/swin_tiny_tapf_d0.yml"
)
EXPECTED_D0_CHECKPOINT_SHA256 = (
    "59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069"
)
EXPECTED_D0_CONFIG_SHA256 = (
    "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b"
)
EXPECTED_CLIP_SHA256 = (
    "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
)

EXPECTED_TRAIN_COUNT = 15618
SLOT_COUNT = 5
CLIP_DIMENSION = 768
D0_MICROBATCH = 8
CLIP_MICROBATCH = 8
OUTER_BATCH_SIZE = 64
FORMAL_DEVICE = "cuda:0"

BANK_FIELDS = {
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
}
GEOMETRY_FIELDS = {
    "schema",
    "relative_paths",
    "image_sha256",
    "slot_names",
    "availability",
    "slot_confidence",
    "instance_centers_xy",
    "instance_rectangles",
    "canonical_centers_xy",
    "canonical_rectangles",
    "crop_hw",
    "geometry_gate_pass",
    "query_coverage",
    "common_pair_count_by_slot",
    "query_pid_count_by_slot",
}
CACHE_FIELDS = {
    "schema",
    "bank_sha256",
    "geometry_sha256",
    "relative_paths",
    "image_sha256",
    "availability",
    "instance_rectangles",
    "canonical_rectangles",
    "d0_slot",
    "instance_clip",
    "canonical_clip",
    "global_clip",
    "instance_raw_hist",
    "canonical_raw_hist",
    "pair_row",
    "query_indices",
    "candidate_indices",
    "pair_energy",
    "common",
    "undecided",
    "wrong_donor_indices",
    "wrong_donor_invalid",
    "arm_names",
}


def _validate_relative_path(value: str) -> str:
    path = Path(str(value))
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "bounding_box_train"
        or path.suffix.lower() != ".jpg"
    ):
        raise RuntimeError("invalid official train relative path: " + str(value))
    return path.as_posix()


def _require_shape(name: str, value: np.ndarray, shape: tuple) -> None:
    if tuple(value.shape) != tuple(shape):
        raise RuntimeError(
            "{} shape mismatch: {} != {}".format(name, value.shape, shape)
        )


def _load_bank(path, expected_sha256, *, expected_count: int | None) -> dict:
    configured = Path(path).expanduser()
    if not configured.is_absolute():
        raise ValueError("candidate bank path must be absolute")
    resolved = configured.resolve(strict=True)
    if resolved != configured:
        raise RuntimeError("candidate bank path must be canonical")
    if fuel_io.sha256_file(resolved) != str(expected_sha256):
        raise RuntimeError("candidate bank SHA256 mismatch")
    arrays = fuel_io.load_npz_exact(resolved, BANK_FIELDS)
    if str(arrays["schema"].item()) != BANK_SCHEMA:
        raise RuntimeError("candidate bank schema mismatch")
    count = len(arrays["relative_paths"])
    if expected_count is not None and count != int(expected_count):
        raise RuntimeError("candidate bank train count mismatch")
    if count < 1:
        raise RuntimeError("candidate bank train table is empty")
    paths = tuple(
        _validate_relative_path(value) for value in arrays["relative_paths"]
    )
    if len(paths) != len(set(paths)):
        raise RuntimeError("candidate bank relative paths are not unique")
    if tuple(paths) != tuple(map(str, arrays["relative_paths"].tolist())):
        raise RuntimeError("candidate bank relative path normalization changed")

    for name in ("raw_pids", "relabeled_pids", "camids"):
        _require_shape(name, arrays[name], (count,))
    _require_shape(
        "candidate bank image_sha256",
        arrays["image_sha256"],
        (count,),
    )
    image_sha256 = tuple(map(str, arrays["image_sha256"].tolist()))
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in image_sha256
    ):
        raise RuntimeError("candidate bank RGB SHA256 vector is invalid")
    raw_pids = arrays["raw_pids"].astype(np.int64, copy=False)
    pids = arrays["relabeled_pids"].astype(np.int64, copy=False)
    cameras = arrays["camids"].astype(np.int64, copy=False)
    if np.any(raw_pids < 0):
        raise RuntimeError("candidate bank contains a junk raw PID")
    if sorted(set(pids.tolist())) != list(range(len(set(pids.tolist())))):
        raise RuntimeError("candidate bank relabeled PIDs are not contiguous")
    raw_to_label = {}
    label_to_raw = {}
    for raw_pid, pid in zip(raw_pids.tolist(), pids.tolist()):
        if raw_pid in raw_to_label and raw_to_label[raw_pid] != pid:
            raise RuntimeError("raw PID maps to multiple relabeled PIDs")
        if pid in label_to_raw and label_to_raw[pid] != raw_pid:
            raise RuntimeError("relabeled PID maps to multiple raw PIDs")
        raw_to_label[raw_pid] = pid
        label_to_raw[pid] = raw_pid
    expected_map = {
        raw_pid: label for label, raw_pid in enumerate(sorted(raw_to_label))
    }
    if raw_to_label != expected_map:
        raise RuntimeError("candidate bank PID relabeling is not sorted exact")
    if np.any(cameras < 0) or np.any(cameras >= 8):
        raise RuntimeError("candidate bank camera leaves [0,7]")
    for index, relative_path in enumerate(paths):
        parsed_raw, parsed_camera, _ = fuel_core.parse_train_filename(
            relative_path
        )
        if parsed_raw != int(raw_pids[index]) or parsed_camera != int(
            cameras[index]
        ):
            raise RuntimeError("candidate bank filename PID/camera mismatch")

    d0_global = arrays["d0_global"]
    if (
        d0_global.ndim != 2
        or d0_global.shape[0] != count
        or d0_global.dtype != np.float32
        or not np.isfinite(d0_global).all()
    ):
        raise RuntimeError("candidate bank D0 global feature schema mismatch")
    norms = np.linalg.norm(d0_global.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-6, rtol=2e-6):
        raise RuntimeError("candidate bank D0 globals are not normalized")

    query = arrays["query_indices"].astype(np.int64, copy=False)
    candidate = arrays["candidate_indices"].astype(np.int64, copy=False)
    impostor = arrays["pair_is_impostor"]
    distance = arrays["d0_distance"]
    if (
        query.ndim != 1
        or candidate.shape != query.shape
        or impostor.shape != query.shape
        or distance.shape != query.shape
        or impostor.dtype != np.bool_
        or len(query) == 0
    ):
        raise RuntimeError("candidate bank pair-vector schema mismatch")
    if (
        int(query.min()) < 0
        or int(candidate.min()) < 0
        or int(query.max()) >= count
        or int(candidate.max()) >= count
        or np.any(query == candidate)
    ):
        raise RuntimeError("candidate bank pair index is invalid")
    if not np.array_equal(impostor, pids[query] != pids[candidate]):
        raise RuntimeError("candidate bank pair labels disagree with PID")
    if not np.isfinite(distance).all() or np.any(distance < 0.0):
        raise RuntimeError("candidate bank D0 distance is invalid")
    offsets = arrays["query_offsets"].astype(np.int64, copy=False)
    if (
        offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != len(query)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise RuntimeError("candidate bank query offsets are invalid")
    observed_queries = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        block = query[int(start) : int(stop)]
        if not np.all(block == block[0]):
            raise RuntimeError("candidate bank query group is not contiguous")
        block_candidates = candidate[int(start) : int(stop)]
        if len(block_candidates) != len(set(block_candidates.tolist())):
            raise RuntimeError("candidate bank query has duplicate candidates")
        observed_queries.append(int(block[0]))
    if len(observed_queries) != len(set(observed_queries)):
        raise RuntimeError("candidate bank repeats a query group")

    arrays["_path_tuple"] = paths
    arrays["_resolved_path"] = np.asarray(str(resolved))
    arrays["_sha256"] = np.asarray(str(expected_sha256))
    return arrays


def _validate_rectangles(name: str, rectangles: np.ndarray, count: int) -> None:
    if rectangles.dtype.kind not in "iu":
        raise RuntimeError(name + " rectangles must be integer")
    _require_shape(name, rectangles, (count, SLOT_COUNT, 4))
    top = rectangles[:, :, 0].astype(np.int64)
    left = rectangles[:, :, 1].astype(np.int64)
    height = rectangles[:, :, 2].astype(np.int64)
    width = rectangles[:, :, 3].astype(np.int64)
    if (
        np.any(top < 0)
        or np.any(left < 0)
        or np.any(height <= 0)
        or np.any(width <= 0)
        or np.any(top + height > fuel_core.IMAGE_HEIGHT)
        or np.any(left + width > fuel_core.IMAGE_WIDTH)
    ):
        raise RuntimeError(name + " rectangle leaves canonical RGB")
    dimensions = rectangles[:, :, 2:4]
    if not np.array_equal(
        dimensions, np.broadcast_to(dimensions[:1], dimensions.shape)
    ):
        raise RuntimeError(name + " rectangle dimensions vary across rows")


def _load_geometry(
    path,
    expected_sha256,
    bank: Mapping,
    *,
    enforce_formal_gate: bool = True,
) -> dict:
    configured = Path(path).expanduser()
    if not configured.is_absolute():
        raise ValueError("geometry path must be absolute")
    resolved = configured.resolve(strict=True)
    if resolved != configured:
        raise RuntimeError("geometry path must be canonical")
    if fuel_io.sha256_file(resolved) != str(expected_sha256):
        raise RuntimeError("geometry SHA256 mismatch")
    arrays = fuel_io.load_npz_exact(resolved, GEOMETRY_FIELDS)
    if str(arrays["schema"].item()) != GEOMETRY_SCHEMA:
        raise RuntimeError("geometry schema mismatch")
    paths = tuple(map(str, arrays["relative_paths"].tolist()))
    if paths != bank["_path_tuple"]:
        raise RuntimeError("geometry and candidate bank path order differ")
    count = len(paths)
    _require_shape("geometry image_sha256", arrays["image_sha256"], (count,))
    if not np.array_equal(arrays["image_sha256"], bank["image_sha256"]):
        raise RuntimeError("geometry and candidate bank RGB SHA order differ")
    _require_shape("geometry slot_names", arrays["slot_names"], (SLOT_COUNT,))
    if tuple(map(str, arrays["slot_names"].tolist())) != (
        "head",
        "upper_torso_arms",
        "lower_torso",
        "upper_legs",
        "lower_legs_feet",
    ):
        raise RuntimeError("geometry slot ontology mismatch")
    availability = arrays["availability"]
    if availability.dtype != np.bool_:
        raise RuntimeError("geometry availability must be boolean")
    _require_shape("geometry availability", availability, (count, SLOT_COUNT))
    if not bool(availability.any(axis=0).all()):
        raise RuntimeError("one or more geometry slots are globally unavailable")
    _require_shape(
        "geometry slot_confidence",
        arrays["slot_confidence"],
        (count, SLOT_COUNT),
    )
    if not np.isfinite(arrays["slot_confidence"]).all():
        raise RuntimeError("geometry slot confidence is nonfinite")
    _require_shape(
        "geometry instance_centers",
        arrays["instance_centers_xy"],
        (count, SLOT_COUNT, 2),
    )
    _require_shape(
        "geometry canonical_centers",
        arrays["canonical_centers_xy"],
        (SLOT_COUNT, 2),
    )
    if (
        not np.isfinite(arrays["instance_centers_xy"]).all()
        or not np.isfinite(arrays["canonical_centers_xy"]).all()
    ):
        raise RuntimeError("geometry center is nonfinite")
    _require_shape("geometry crop_hw", arrays["crop_hw"], (SLOT_COUNT, 2))
    _validate_rectangles(
        "instance", arrays["instance_rectangles"], count
    )
    _validate_rectangles(
        "canonical", arrays["canonical_rectangles"], count
    )
    expected_hw = np.broadcast_to(
        arrays["crop_hw"][None],
        arrays["instance_rectangles"][:, :, 2:4].shape,
    )
    if not np.array_equal(
        arrays["instance_rectangles"][:, :, 2:4], expected_hw
    ) or not np.array_equal(
        arrays["canonical_rectangles"][:, :, 2:4], expected_hw
    ):
        raise RuntimeError("geometry correct/canonical crop H/W differ")
    canonical_rectangles = arrays["canonical_rectangles"]
    if not np.array_equal(
        canonical_rectangles,
        np.broadcast_to(
            canonical_rectangles[:1], canonical_rectangles.shape
        ),
    ):
        raise RuntimeError("canonical rectangles vary by image")
    _require_shape(
        "geometry common_pair_count_by_slot",
        arrays["common_pair_count_by_slot"],
        (SLOT_COUNT,),
    )
    _require_shape(
        "geometry query_pid_count_by_slot",
        arrays["query_pid_count_by_slot"],
        (SLOT_COUNT,),
    )
    if arrays["geometry_gate_pass"].shape != (1,):
        raise RuntimeError("geometry gate must be a scalar")
    if arrays["query_coverage"].shape != (1,):
        raise RuntimeError("geometry query coverage must be a scalar")
    query = bank["query_indices"].astype(np.int64, copy=False)
    candidate = bank["candidate_indices"].astype(np.int64, copy=False)
    common = availability[query] & availability[candidate]
    unique_queries = np.unique(query)
    covered_queries = np.unique(query[common.any(axis=1)])
    query_coverage = float(len(covered_queries) / len(unique_queries))
    pair_counts = common.sum(axis=0).astype(np.int64)
    raw_pids = bank["raw_pids"].astype(np.int64, copy=False)
    pid_counts = np.asarray(
        [
            len(np.unique(raw_pids[np.unique(query[common[:, slot]])]))
            for slot in range(SLOT_COUNT)
        ],
        dtype=np.int64,
    )
    if (
        float(arrays["query_coverage"].item()) != query_coverage
        or not np.array_equal(
            arrays["common_pair_count_by_slot"], pair_counts
        )
        or not np.array_equal(arrays["query_pid_count_by_slot"], pid_counts)
    ):
        raise RuntimeError("geometry coverage receipt differs from sealed bank")
    gate = bool(
        query_coverage >= 0.80
        and bool((pair_counts >= 100000).all())
        and bool((pid_counts >= 300).all())
    )
    if bool(arrays["geometry_gate_pass"].item()) != gate:
        raise RuntimeError("geometry gate scalar differs from recomputation")
    if enforce_formal_gate and not gate:
        raise RuntimeError(
            "PC-NEC geometry gate failed; D0/CLIP fuel cache NO-START"
        )
    arrays["_resolved_path"] = np.asarray(str(resolved))
    arrays["_sha256"] = np.asarray(str(expected_sha256))
    return arrays


def configure_formal_determinism() -> dict:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "formal stage3 requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.manual_seed(4161234)
    torch.cuda.manual_seed_all(4161234)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": 4161234,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }


def _read_canonical_rgb(image_path: Path, expected_image_sha256: str):
    from PIL import Image, UnidentifiedImageError

    image_path = Path(image_path).resolve(strict=True)
    if fuel_io.sha256_file(image_path) != str(expected_image_sha256):
        raise RuntimeError("official RGB SHA256 disagrees with geometry")
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            image = image.resize(
                (fuel_core.IMAGE_WIDTH, fuel_core.IMAGE_HEIGHT),
                resample=resampling.BILINEAR,
            )
            array = np.asarray(image, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise RuntimeError("official RGB decode failed") from error
    if array.shape != (
        fuel_core.IMAGE_HEIGHT,
        fuel_core.IMAGE_WIDTH,
        3,
    ):
        raise RuntimeError("canonical RGB shape mismatch")
    rgb = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
    if (
        tuple(rgb.shape)
        != (3, fuel_core.IMAGE_HEIGHT, fuel_core.IMAGE_WIDTH)
        or not bool(torch.isfinite(rgb).all())
    ):
        raise RuntimeError("canonical RGB tensor is invalid")
    canonical_sha256 = fuel_io.sha256_array(rgb.numpy())
    return rgb, canonical_sha256


def _load_rgb_batch(
    indices: Sequence[int],
    *,
    bank: Mapping,
    geometry: Mapping,
    image_root: Path,
    rgb_loader: Callable,
) -> tuple[torch.Tensor, list[str]]:
    rows = []
    digests = []
    image_sha = geometry["image_sha256"]
    for index in indices:
        relative_path = bank["_path_tuple"][int(index)]
        image_path = (Path(image_root) / relative_path).resolve()
        try:
            image_path.relative_to(Path(image_root).resolve())
        except ValueError as error:
            raise RuntimeError("official RGB escapes frozen image root") from error
        rgb, digest = rgb_loader(image_path, str(image_sha[int(index)]))
        if (
            not torch.is_tensor(rgb)
            or tuple(rgb.shape)
            != (3, fuel_core.IMAGE_HEIGHT, fuel_core.IMAGE_WIDTH)
            or rgb.dtype != torch.float32
            or not bool(torch.isfinite(rgb).all())
            or float(rgb.min()) < 0.0
            or float(rgb.max()) > 1.0
        ):
            raise RuntimeError("RGB loader violated canonical tensor contract")
        if str(digest) != fuel_io.sha256_array(rgb.numpy()):
            raise RuntimeError("RGB loader canonical SHA is incorrect")
        rows.append(rgb)
        digests.append(str(digest))
    return torch.stack(rows, dim=0), digests


@dataclass
class _TrainOnlyDataset:
    num_train_pids: int
    num_train_cams: int
    num_train_vids: int = 1


def _dataset_contract(bank: Mapping) -> _TrainOnlyDataset:
    pids = bank["relabeled_pids"].astype(np.int64, copy=False)
    cameras = bank["camids"].astype(np.int64, copy=False)
    return _TrainOnlyDataset(
        num_train_pids=len(set(pids.tolist())),
        num_train_cams=len(set(cameras.tolist())),
    )


def _default_d0_factory(
    *,
    args,
    bank: Mapping,
    device,
):
    return d0_feature_extractor.SealedD0FeatureExtractor(
        config_path=args.d0_config,
        config_sha256=EXPECTED_D0_CONFIG_SHA256,
        checkpoint_path=args.d0_checkpoint,
        checkpoint_sha256=EXPECTED_D0_CHECKPOINT_SHA256,
        dataset=_dataset_contract(bank),
        device=device,
        microbatch=D0_MICROBATCH,
    )


def _default_clip_factory(*, args, device):
    return clip_crop_encoder.FrozenClipCropEncoder(
        args.clip_checkpoint,
        EXPECTED_CLIP_SHA256,
        device,
        microbatch=CLIP_MICROBATCH,
    )


def _raw_histograms_for_batch(
    rgb: torch.Tensor,
    rectangles: np.ndarray,
    availability: np.ndarray,
) -> np.ndarray:
    rgb_numpy = rgb.numpy()
    output = np.zeros(
        (len(rgb_numpy), SLOT_COUNT, 512), dtype=np.float32
    )
    for row in range(len(rgb_numpy)):
        for slot in range(SLOT_COUNT):
            if not bool(availability[row, slot]):
                continue
            top, left, height, width = (
                int(value) for value in rectangles[row, slot]
            )
            crop = rgb_numpy[
                row,
                :,
                top : top + height,
                left : left + width,
            ]
            histogram = fuel_core.raw_color_histogram(crop)
            output[row, slot] = histogram.astype(np.float32)
    return output


def _build_records(bank: Mapping) -> list[dict]:
    records = []
    for index, relative_path in enumerate(bank["_path_tuple"]):
        records.append(
            {
                "record_index": int(index),
                "relative_path": str(relative_path),
                "train_pid": int(bank["relabeled_pids"][index]),
                "raw_pid": int(bank["raw_pids"][index]),
                "camera": int(bank["camids"][index]),
                "frame": int(
                    fuel_core.parse_train_filename(relative_path)[2]
                ),
            }
        )
    fuel_core.validate_train_records(records)
    return records


def _build_wrong_rgb_donor_index(
    records: Sequence[Mapping],
    availability: np.ndarray,
) -> dict:
    """Index the exact core donor pool without rescanning all rows per pair."""
    valid = np.asarray(availability, dtype=np.bool_)
    if valid.shape != (len(records), SLOT_COUNT):
        raise ValueError("wrong-RGB availability shape mismatch")
    grouped: dict[int, list[tuple[str, int, int]]] = {}
    for index, row in enumerate(records):
        if not bool(valid[index].all()):
            continue
        grouped.setdefault(int(row["camera"]), []).append(
            (
                str(row["relative_path"]),
                int(index),
                int(row["train_pid"]),
            )
        )
    output = {}
    for camera, rows in grouped.items():
        rows.sort(key=lambda value: (value[0], value[1]))
        indices = np.asarray([value[1] for value in rows], dtype=np.int64)
        pids = np.asarray([value[2] for value in rows], dtype=np.int64)
        positions_by_pid = {
            int(pid): np.flatnonzero(pids == pid).astype(np.int64)
            for pid in np.unique(pids)
        }
        output[int(camera)] = {
            "indices": indices,
            "pids": pids,
            "positions_by_pid": positions_by_pid,
        }
    return output


def _select_wrong_rgb_donor_fast(
    donor_index: Mapping,
    *,
    query_path: str,
    candidate_path: str,
    query_pid: int,
    candidate_pid: int,
    candidate_camera: int,
    slot: int,
) -> int | None:
    """Exact order-statistic form of ``fuel_core.select_wrong_rgb_donor``."""
    if not 0 <= int(slot) < SLOT_COUNT:
        raise ValueError("wrong-RGB slot is outside five-slot ontology")
    pool = donor_index.get(int(candidate_camera))
    if pool is None:
        return None
    excluded_parts = [
        pool["positions_by_pid"].get(pid, np.empty(0, dtype=np.int64))
        for pid in {int(query_pid), int(candidate_pid)}
    ]
    excluded = np.sort(np.concatenate(excluded_parts))
    pool_size = len(pool["indices"])
    eligible_size = pool_size - len(excluded)
    if eligible_size <= 0:
        return None
    key = fuel_core.stable_hash_uint64(
        fuel_core.WRONG_RGB_DONOR_SALT,
        str(query_path),
        str(candidate_path),
        int(slot),
    )
    eligible_rank = int(key % eligible_size)
    low, high = 0, pool_size - 1
    while low < high:
        middle = (low + high) // 2
        excluded_through_middle = int(
            np.searchsorted(excluded, middle, side="right")
        )
        eligible_through_middle = middle + 1 - excluded_through_middle
        if eligible_through_middle > eligible_rank:
            high = middle
        else:
            low = middle + 1
    position = low
    excluded_location = int(np.searchsorted(excluded, position))
    if (
        excluded_location < len(excluded)
        and int(excluded[excluded_location]) == position
    ):
        raise RuntimeError("wrong-RGB order statistic selected excluded PID")
    selected = int(pool["indices"][position])
    if int(pool["pids"][position]) in {
        int(query_pid),
        int(candidate_pid),
    }:
        raise RuntimeError("wrong-RGB fast selector leaked excluded PID")
    return selected


def _batched_cosine_distance(
    left: np.ndarray,
    right: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    active = np.asarray(active, dtype=np.bool_)
    if left64.shape != right64.shape or left64.shape[:-1] != active.shape:
        raise ValueError("batched cosine inputs are not aligned")
    if not np.isfinite(left64).all() or not np.isfinite(right64).all():
        raise ValueError("batched cosine input is nonfinite")
    left_norm = np.linalg.norm(left64, axis=-1)
    right_norm = np.linalg.norm(right64, axis=-1)
    if np.any(active & ((left_norm <= 0.0) | (right_norm <= 0.0))):
        raise ValueError("batched cosine active vector has zero norm")
    output = np.zeros(active.shape, dtype=np.float64)
    dot = np.sum(left64 * right64, axis=-1)
    denominator = left_norm * right_norm
    output[active] = np.clip(
        1.0 - dot[active] / denominator[active],
        0.0,
        2.0,
    )
    return output


def _batched_histogram_tv(
    left: np.ndarray,
    right: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    active = np.asarray(active, dtype=np.bool_)
    if (
        left64.shape != right64.shape
        or left64.shape[:-1] != active.shape
        or left64.shape[-1] != 512
    ):
        raise ValueError("batched histogram inputs are not aligned")
    if (
        not np.isfinite(left64).all()
        or not np.isfinite(right64).all()
        or np.any(left64 < 0.0)
        or np.any(right64 < 0.0)
    ):
        raise ValueError("batched histogram input is invalid")
    if (
        np.any(~np.isclose(left64.sum(axis=-1)[active], 1.0))
        or np.any(~np.isclose(right64.sum(axis=-1)[active], 1.0))
    ):
        raise ValueError("batched active histogram is not normalized")
    output = np.zeros(active.shape, dtype=np.float64)
    output[active] = (
        0.5 * np.abs(left64 - right64).sum(axis=-1)[active]
    )
    return output


def _batched_existential(
    distances: np.ndarray,
    common: np.ndarray,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    common = np.asarray(common, dtype=np.bool_)
    if distances.shape != common.shape or distances.ndim != 2:
        raise ValueError("batched existential inputs are not aligned")
    if np.any(common & ((distances < 0.0) | ~np.isfinite(distances))):
        raise ValueError("batched existential active distance is invalid")
    output = np.zeros(len(common), dtype=np.float64)
    covered = common.any(axis=1)
    if bool(covered.any()):
        selected = np.where(common[covered], distances[covered], -np.inf)
        output[covered] = selected.max(axis=1)
    return output


def _compute_energy_chunk(
    *,
    query: np.ndarray,
    candidate: np.ndarray,
    availability: np.ndarray,
    d0_slot: np.ndarray,
    instance_clip: np.ndarray,
    canonical_clip: np.ndarray,
    global_clip: np.ndarray,
    instance_raw_hist: np.ndarray,
    canonical_raw_hist: np.ndarray,
    d0_distance: np.ndarray,
    wrong_donor_indices: np.ndarray,
    wrong_donor_invalid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    common = availability[query] & availability[candidate]
    energies = np.empty(
        (len(query), len(fuel_core.ARM_NAMES)),
        dtype=np.float64,
    )
    column = {
        name: index for index, name in enumerate(fuel_core.ARM_NAMES)
    }

    correct_slots = _batched_cosine_distance(
        instance_clip[query],
        instance_clip[candidate],
        common,
    )
    student_slots = _batched_cosine_distance(
        d0_slot[query],
        d0_slot[candidate],
        common,
    )
    raw_slots = _batched_histogram_tv(
        instance_raw_hist[query],
        instance_raw_hist[candidate],
        common,
    )
    canonical_clip_slots = _batched_cosine_distance(
        canonical_clip[query],
        canonical_clip[candidate],
        common,
    )
    canonical_raw_slots = _batched_histogram_tv(
        canonical_raw_hist[query],
        canonical_raw_hist[candidate],
        common,
    )
    energies[:, column["correct"]] = _batched_existential(
        correct_slots, common
    )
    energies[:, column["pose_only_raw_color"]] = _batched_existential(
        raw_slots, common
    )
    energies[:, column["pose_only_student_part"]] = _batched_existential(
        student_slots, common
    )
    energies[:, column["canonical_location_clip"]] = _batched_existential(
        canonical_clip_slots, common
    )
    energies[:, column["neither"]] = _batched_existential(
        canonical_raw_slots, common
    )

    source_slot = np.broadcast_to(
        np.arange(SLOT_COUNT, dtype=np.int64),
        common.shape,
    ).copy()
    for row in range(len(common)):
        active_slots = np.flatnonzero(common[row])
        if len(active_slots):
            source_slot[row, active_slots] = np.roll(active_slots, -1)
    row_index = np.arange(len(query), dtype=np.int64)[:, None]
    shuffled_candidate = instance_clip[candidate][row_index, source_slot]
    shuffled_slots = _batched_cosine_distance(
        instance_clip[query],
        shuffled_candidate,
        common,
    )
    energies[:, column["slot_shuffle"]] = _batched_existential(
        shuffled_slots, common
    )

    donor_slot = np.arange(SLOT_COUNT, dtype=np.int64)[None, :]
    donor_features = instance_clip[wrong_donor_indices, donor_slot]
    donor_active = common & ~wrong_donor_invalid[:, None]
    wrong_slots = _batched_cosine_distance(
        instance_clip[query],
        donor_features,
        donor_active,
    )
    energies[:, column["wrong_rgb"]] = _batched_existential(
        wrong_slots, donor_active
    )
    global_active = np.ones(len(query), dtype=np.bool_)
    energies[:, column["global_clip"]] = _batched_cosine_distance(
        global_clip[query],
        global_clip[candidate],
        global_active,
    )
    energies[:, column["d0_only"]] = np.asarray(
        d0_distance, dtype=np.float64
    )
    if not np.isfinite(energies).all() or np.any(energies < 0.0):
        raise RuntimeError("batched pair energy is invalid")
    return energies, common


def _compute_pair_rows(
    *,
    bank: Mapping,
    geometry: Mapping,
    d0_slot: np.ndarray,
    instance_clip: np.ndarray,
    canonical_clip: np.ndarray,
    global_clip: np.ndarray,
    instance_raw_hist: np.ndarray,
    canonical_raw_hist: np.ndarray,
) -> dict:
    query = bank["query_indices"].astype(np.int64, copy=False)
    candidate = bank["candidate_indices"].astype(np.int64, copy=False)
    pair_count = len(query)
    availability = geometry["availability"]
    records = _build_records(bank)
    donor_index = _build_wrong_rgb_donor_index(records, availability)
    pair_energy = np.empty(
        (pair_count, len(fuel_core.ARM_NAMES)), dtype=np.float64
    )
    common = availability[query] & availability[candidate]
    undecided = ~common.any(axis=1)
    wrong_donor_indices = np.full(
        (pair_count, SLOT_COUNT), -1, dtype=np.int32
    )
    wrong_donor_invalid = np.zeros(pair_count, dtype=np.bool_)

    for pair_row, (query_index, candidate_index) in enumerate(
        zip(query.tolist(), candidate.tolist())
    ):
        selected = []
        for slot in range(SLOT_COUNT):
            donor = _select_wrong_rgb_donor_fast(
                donor_index,
                query_path=bank["_path_tuple"][query_index],
                candidate_path=bank["_path_tuple"][candidate_index],
                query_pid=int(bank["relabeled_pids"][query_index]),
                candidate_pid=int(bank["relabeled_pids"][candidate_index]),
                candidate_camera=int(bank["camids"][candidate_index]),
                slot=slot,
            )
            selected.append(donor)
        missing = [value is None for value in selected]
        if any(missing):
            if not all(missing):
                raise RuntimeError("wrong-RGB donor pool changed by slot")
            wrong_donor_invalid[pair_row] = True
        else:
            wrong_donor_indices[pair_row] = np.asarray(
                selected, dtype=np.int32
            )
        if pair_row == 0 or (pair_row + 1) % 10000 == 0:
            print(
                "wrong-RGB donor {}/{}".format(
                    pair_row + 1, pair_count
                ),
                flush=True,
            )

    energy_chunk_size = 512
    for start in range(0, pair_count, energy_chunk_size):
        stop = min(start + energy_chunk_size, pair_count)
        observed_energy, observed_common = _compute_energy_chunk(
            query=query[start:stop],
            candidate=candidate[start:stop],
            availability=availability,
            d0_slot=d0_slot,
            instance_clip=instance_clip,
            canonical_clip=canonical_clip,
            global_clip=global_clip,
            instance_raw_hist=instance_raw_hist,
            canonical_raw_hist=canonical_raw_hist,
            d0_distance=bank["d0_distance"][start:stop],
            wrong_donor_indices=wrong_donor_indices[start:stop],
            wrong_donor_invalid=wrong_donor_invalid[start:stop],
        )
        if not np.array_equal(observed_common, common[start:stop]):
            raise RuntimeError("batched energy changed common bitmap")
        pair_energy[start:stop] = observed_energy
        if start == 0 or stop == pair_count or stop % 10240 == 0:
            print(
                "pair energy {}/{}".format(stop, pair_count),
                flush=True,
            )
    if (
        not np.isfinite(pair_energy).all()
        or np.any(pair_energy < 0.0)
        or len(pair_energy) != pair_count
    ):
        raise RuntimeError("pair energy matrix is invalid")
    return {
        "pair_row": np.arange(pair_count, dtype=np.int64),
        "query_indices": query.astype(np.int32, copy=True),
        "candidate_indices": candidate.astype(np.int32, copy=True),
        "pair_energy": pair_energy,
        "common": common.astype(np.bool_, copy=True),
        "undecided": undecided.astype(np.bool_, copy=True),
        "wrong_donor_indices": wrong_donor_indices,
        "wrong_donor_invalid": wrong_donor_invalid,
    }


def build_stage3_arrays(
    *,
    bank: Mapping,
    geometry: Mapping,
    image_root: Path,
    args,
    device,
    d0_factory: Callable,
    clip_factory: Callable,
    rgb_loader: Callable,
    events: list[str] | None = None,
) -> tuple[dict, dict]:
    """Compute the stage-3 arrays; factories make the self-test CPU-only."""
    event_log = [] if events is None else events
    count = len(bank["_path_tuple"])
    availability = geometry["availability"].astype(np.bool_, copy=False)
    instance_rectangles = geometry["instance_rectangles"]
    canonical_rectangles = geometry["canonical_rectangles"]
    canonical_hashes: list[str] = [""] * count
    d0_slot_parts = []

    event_log.append("d0_construct")
    d0 = d0_factory(args=args, bank=bank, device=device)
    try:
        for start in range(0, count, OUTER_BATCH_SIZE):
            stop = min(start + OUTER_BATCH_SIZE, count)
            indices = list(range(start, stop))
            rgb, hashes = _load_rgb_batch(
                indices,
                bank=bank,
                geometry=geometry,
                image_root=image_root,
                rgb_loader=rgb_loader,
            )
            output = d0.encode(
                rgb,
                instance_rectangles[start:stop],
            )
            observed_global = (
                output["global_features"].detach().cpu().numpy()
            )
            expected_global = bank["d0_global"][start:stop]
            if (
                observed_global.dtype != expected_global.dtype
                or observed_global.shape != expected_global.shape
                or not np.array_equal(observed_global, expected_global)
            ):
                raise RuntimeError(
                    "recomputed D0 global is not bit-exact with sealed bank"
                )
            slots = output["slot_features"].detach().cpu().numpy().astype(
                np.float32, copy=True
            )
            if slots.shape[:2] != (stop - start, SLOT_COUNT):
                raise RuntimeError("D0 slot output shape mismatch")
            slots[~availability[start:stop]] = 0.0
            d0_slot_parts.append(slots)
            canonical_hashes[start:stop] = hashes
            print("D0 stage {}/{}".format(stop, count), flush=True)
    finally:
        d0.close()
        event_log.append("d0_close")
        del d0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    d0_slot = np.concatenate(d0_slot_parts, axis=0)
    if d0_slot.shape[0] != count or any(not value for value in canonical_hashes):
        raise RuntimeError("D0 stage did not cover every train row")

    event_log.append("clip_construct")
    clip = clip_factory(args=args, device=device)
    instance_clip_parts = []
    canonical_clip_parts = []
    global_clip_parts = []
    instance_hist_parts = []
    canonical_hist_parts = []
    try:
        for start in range(0, count, OUTER_BATCH_SIZE):
            stop = min(start + OUTER_BATCH_SIZE, count)
            indices = list(range(start, stop))
            rgb, hashes = _load_rgb_batch(
                indices,
                bank=bank,
                geometry=geometry,
                image_root=image_root,
                rgb_loader=rgb_loader,
            )
            if hashes != canonical_hashes[start:stop]:
                raise RuntimeError(
                    "CLIP stage canonical RGB differs from D0 stage"
                )
            active = availability[start:stop]
            instance_features = (
                clip.encode_rectangles(
                    rgb, instance_rectangles[start:stop]
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
            canonical_features = (
                clip.encode_rectangles(
                    rgb, canonical_rectangles[start:stop]
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
            whole_features = (
                clip.encode_whole_images(rgb)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
            expected_local = (stop - start, SLOT_COUNT, CLIP_DIMENSION)
            if (
                instance_features.shape != expected_local
                or canonical_features.shape != expected_local
                or whole_features.shape
                != (stop - start, CLIP_DIMENSION)
            ):
                raise RuntimeError("CLIP stage output shape mismatch")
            instance_features[~active] = 0.0
            canonical_features[~active] = 0.0
            instance_clip_parts.append(instance_features)
            canonical_clip_parts.append(canonical_features)
            global_clip_parts.append(whole_features)
            instance_hist_parts.append(
                _raw_histograms_for_batch(
                    rgb,
                    instance_rectangles[start:stop],
                    active,
                )
            )
            canonical_hist_parts.append(
                _raw_histograms_for_batch(
                    rgb,
                    canonical_rectangles[start:stop],
                    active,
                )
            )
            print("CLIP/raw stage {}/{}".format(stop, count), flush=True)
    finally:
        clip.close()
        event_log.append("clip_close")
        del clip
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if event_log.index("d0_close") > event_log.index("clip_construct"):
        raise RuntimeError("OpenCLIP was constructed before sealed D0 closed")
    instance_clip = np.concatenate(instance_clip_parts, axis=0)
    canonical_clip = np.concatenate(canonical_clip_parts, axis=0)
    global_clip = np.concatenate(global_clip_parts, axis=0)
    instance_raw_hist = np.concatenate(instance_hist_parts, axis=0)
    canonical_raw_hist = np.concatenate(canonical_hist_parts, axis=0)
    for name, value in (
        ("d0_slot", d0_slot),
        ("instance_clip", instance_clip),
        ("canonical_clip", canonical_clip),
        ("global_clip", global_clip),
        ("instance_raw_hist", instance_raw_hist),
        ("canonical_raw_hist", canonical_raw_hist),
    ):
        if value.shape[0] != count or not np.isfinite(value).all():
            raise RuntimeError(name + " does not cover every train row")

    pair_arrays = _compute_pair_rows(
        bank=bank,
        geometry=geometry,
        d0_slot=d0_slot,
        instance_clip=instance_clip,
        canonical_clip=canonical_clip,
        global_clip=global_clip,
        instance_raw_hist=instance_raw_hist,
        canonical_raw_hist=canonical_raw_hist,
    )
    arrays = {
        "schema": np.asarray(SCHEMA),
        "bank_sha256": np.asarray([str(bank["_sha256"].item())]),
        "geometry_sha256": np.asarray(
            [str(geometry["_sha256"].item())]
        ),
        "relative_paths": np.asarray(bank["_path_tuple"], dtype=np.str_),
        "image_sha256": bank["image_sha256"].copy(),
        "availability": availability.astype(np.bool_, copy=True),
        "instance_rectangles": instance_rectangles.astype(
            np.int16, copy=True
        ),
        "canonical_rectangles": canonical_rectangles.astype(
            np.int16, copy=True
        ),
        "d0_slot": d0_slot.astype(np.float32, copy=False),
        "instance_clip": instance_clip.astype(np.float32, copy=False),
        "canonical_clip": canonical_clip.astype(np.float32, copy=False),
        "global_clip": global_clip.astype(np.float32, copy=False),
        "instance_raw_hist": instance_raw_hist.astype(
            np.float32, copy=False
        ),
        "canonical_raw_hist": canonical_raw_hist.astype(
            np.float32, copy=False
        ),
        **pair_arrays,
        "arm_names": np.asarray(fuel_core.ARM_NAMES, dtype=np.str_),
    }
    if set(arrays) != CACHE_FIELDS:
        raise RuntimeError("fuel cache field set mismatch")
    pair_count = len(bank["query_indices"])
    _require_shape(
        "fuel pair_energy",
        arrays["pair_energy"],
        (pair_count, len(fuel_core.ARM_NAMES)),
    )
    _require_shape(
        "fuel common", arrays["common"], (pair_count, SLOT_COUNT)
    )
    _require_shape(
        "fuel undecided", arrays["undecided"], (pair_count,)
    )
    if not np.array_equal(
        arrays["common"],
        availability[
            bank["query_indices"].astype(np.int64)
        ]
        & availability[
            bank["candidate_indices"].astype(np.int64)
        ],
    ):
        raise RuntimeError("fuel cache common bitmap changed denominator")
    if not np.array_equal(arrays["undecided"], ~arrays["common"].any(axis=1)):
        raise RuntimeError("fuel cache UNDECIDED does not equal empty common")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "sample_count": int(count),
        "pair_count": int(pair_count),
        "arm_names": list(fuel_core.ARM_NAMES),
        "execution_order": list(event_log),
        "d0_microbatch": D0_MICROBATCH,
        "clip_microbatch": CLIP_MICROBATCH,
        "outer_batch_size": OUTER_BATCH_SIZE,
        "d0_global_bank_recompute_bit_exact": True,
        "same_canonical_rgb_sha_across_d0_clip": True,
        "availability_sha256": fuel_io.sha256_array(
            arrays["availability"]
        ),
        "pair_row_sha256": fuel_io.sha256_array(arrays["pair_row"]),
        "pair_energy_sha256": fuel_io.sha256_array(
            arrays["pair_energy"]
        ),
        "common_sha256": fuel_io.sha256_array(arrays["common"]),
        "canonical_rgb_ordered_sha256": fuel_io.ordered_digest(
            canonical_hashes
        ),
        "image_asset_ordered_sha256": fuel_io.ordered_digest(
            arrays["image_sha256"].tolist()
        ),
        "undecided_count": int(arrays["undecided"].sum()),
        "wrong_donor_invalid_count": int(
            arrays["wrong_donor_invalid"].sum()
        ),
        "common_pair_count_by_slot": arrays["common"]
        .sum(axis=0)
        .astype(np.int64)
        .tolist(),
        "arm_energy_summary": {
            name: {
                "minimum": float(arrays["pair_energy"][:, index].min()),
                "mean": float(arrays["pair_energy"][:, index].mean()),
                "maximum": float(arrays["pair_energy"][:, index].max()),
            }
            for index, name in enumerate(fuel_core.ARM_NAMES)
        },
    }
    return arrays, receipt


def _formal_path(path_value, expected: Path, name: str) -> Path:
    configured = Path(path_value).expanduser()
    if not configured.is_absolute():
        raise ValueError(name + " path must be absolute")
    resolved = configured.resolve(strict=True)
    if resolved != expected:
        raise RuntimeError(name + " path differs from frozen contract")
    if resolved != configured:
        raise RuntimeError(name + " path must be canonical")
    return resolved


def validate_formal_args(args) -> dict:
    observed_interpreter = Path(sys.executable)
    if (
        observed_interpreter != EXPECTED_INTERPRETER
        and observed_interpreter.resolve() != EXPECTED_INTERPRETER.resolve()
    ):
        raise RuntimeError("formal interpreter mismatch")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise RuntimeError("formal stage3 requires PYTHONDONTWRITEBYTECODE=1")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("formal stage3 requires PYTHONHASHSEED=0")
    if REPOSITORY_ROOT.resolve(strict=True) != FIXED_REPOSITORY_ROOT:
        raise RuntimeError("formal stage3 repository path mismatch")
    if str(args.device) != FORMAL_DEVICE:
        raise RuntimeError("formal stage3 is frozen to logical cuda:0")
    if int(args.d0_microbatch) != D0_MICROBATCH:
        raise RuntimeError("formal D0 microbatch must equal candidate builder 8")
    if int(args.clip_microbatch) != CLIP_MICROBATCH:
        raise RuntimeError("formal CLIP microbatch mismatch")
    if int(args.batch_size) != OUTER_BATCH_SIZE:
        raise RuntimeError("formal outer batch size mismatch")
    data_root = _formal_path(args.data_root, FIXED_DATA_ROOT, "data root")
    image_root = (data_root / "Occluded_Duke").resolve(strict=True)
    if image_root != FIXED_IMAGE_ROOT:
        raise RuntimeError("formal image root mismatch")
    d0_checkpoint = _formal_path(
        args.d0_checkpoint, FIXED_D0_CHECKPOINT, "D0 checkpoint"
    )
    clip_checkpoint = _formal_path(
        args.clip_checkpoint, FIXED_CLIP_CHECKPOINT, "CLIP checkpoint"
    )
    expected_config = (
        REPOSITORY_ROOT / FIXED_D0_CONFIG_RELATIVE
    ).resolve(strict=True)
    d0_config = Path(args.d0_config).expanduser().resolve(strict=True)
    if d0_config != expected_config:
        raise RuntimeError("D0 config path differs from frozen repository")
    if fuel_io.sha256_file(d0_config) != EXPECTED_D0_CONFIG_SHA256:
        raise RuntimeError("D0 config SHA256 mismatch")
    if fuel_io.sha256_file(d0_checkpoint) != EXPECTED_D0_CHECKPOINT_SHA256:
        raise RuntimeError("D0 checkpoint SHA256 mismatch")
    if fuel_io.sha256_file(clip_checkpoint) != EXPECTED_CLIP_SHA256:
        raise RuntimeError("CLIP checkpoint SHA256 mismatch")
    configured_bank = Path(args.bank).expanduser()
    configured_geometry = Path(args.geometry).expanduser()
    bank = configured_bank.resolve(strict=True)
    geometry = configured_geometry.resolve(strict=True)
    if bank != configured_bank or geometry != configured_geometry:
        raise RuntimeError("bank and geometry paths must be canonical")
    if bank != FIXED_BANK or geometry != FIXED_GEOMETRY:
        raise RuntimeError("formal bank/geometry namespace mismatch")
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute() or output_dir.exists():
        raise FileExistsError("stage3 output directory must be fresh and absolute")
    resolved_parent = output_dir.parent.resolve(strict=True)
    if resolved_parent != output_dir.parent:
        raise RuntimeError("stage3 output parent must be canonical")
    if output_dir != FIXED_OUTPUT_DIR:
        raise RuntimeError("formal stage3 output namespace mismatch")
    head = fuel_io.git_head(REPOSITORY_ROOT)
    if not args.expected_head or str(args.expected_head) != head:
        raise RuntimeError("formal stage3 HEAD mismatch")
    if fuel_io.git_tracked_status(REPOSITORY_ROOT):
        raise RuntimeError("formal stage3 tracked worktree is dirty")
    if fuel_io.git_index_status(REPOSITORY_ROOT):
        raise RuntimeError("formal stage3 index is dirty")
    expected_sources = {
        "build_fuel_cache.py": args.expected_build_fuel_cache_sha256,
        "fuel_core.py": args.expected_fuel_core_sha256,
        "fuel_io.py": args.expected_fuel_io_sha256,
        "d0_feature_extractor.py": args.expected_d0_feature_extractor_sha256,
        "clip_crop_encoder.py": args.expected_clip_crop_encoder_sha256,
        "geometry_census.py": args.expected_geometry_census_sha256,
    }
    for filename, expected in expected_sources.items():
        path = SCRIPT_DIR / filename
        if not expected or fuel_io.sha256_file(path) != str(expected):
            raise RuntimeError("formal stage3 source SHA mismatch: " + filename)
        tracked = subprocess.run(
            (
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "ls-files",
                "--error-unmatch",
                str(path.relative_to(REPOSITORY_ROOT)),
            ),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode != 0:
            raise RuntimeError("formal stage3 source is untracked: " + filename)
    bank_receipt_path = bank.parent / "receipt.json"
    bank_manifest_path = bank.parent / "manifest.json"
    bank_receipt = fuel_io.readback_json(bank_receipt_path)
    bank_manifest = fuel_io.readback_json(bank_manifest_path)
    if (
        bank_receipt.get("schema")
        != "exp416-pcnec-candidate-bank-receipt-v1"
        or bank_manifest.get("schema")
        != "exp416-pcnec-candidate-bank-manifest-v1"
        or bank_manifest.get("files", {})
        .get("candidate_bank.npz", {})
        .get("sha256")
        != str(args.bank_sha256)
        or bank_manifest.get("files", {}).get("receipt.json", {}).get("sha256")
        != fuel_io.sha256_file(bank_receipt_path)
        or bank_receipt.get("provenance", {}).get("formal_head") != head
    ):
        raise RuntimeError("candidate artifact provenance binding mismatch")
    geometry_summary_path = geometry.parent / "summary.json"
    geometry_manifest_path = geometry.parent / "manifest.json"
    geometry_summary = fuel_io.readback_json(geometry_summary_path)
    geometry_manifest = fuel_io.readback_json(geometry_manifest_path)
    if (
        geometry_summary.get("schema") != GEOMETRY_SCHEMA
        or geometry_summary.get("formal_head") != head
        or geometry_summary.get("bank_sha256") != str(args.bank_sha256)
        or geometry_summary.get("geometry_npz_sha256")
        != str(args.geometry_sha256)
        or geometry_manifest.get("formal_head") != head
        or geometry_manifest.get("bank_sha256") != str(args.bank_sha256)
        or geometry_manifest.get("geometry_npz_sha256")
        != str(args.geometry_sha256)
        or geometry_manifest.get("summary_json_sha256")
        != fuel_io.sha256_file(geometry_summary_path)
    ):
        raise RuntimeError("geometry artifact provenance binding mismatch")
    return {
        "data_root": data_root,
        "image_root": image_root,
        "d0_checkpoint": d0_checkpoint,
        "clip_checkpoint": clip_checkpoint,
        "d0_config": d0_config,
        "bank": bank,
        "geometry": geometry,
        "output_dir": output_dir,
        "head": head,
        "bank_receipt_sha256": fuel_io.sha256_file(bank_receipt_path),
        "bank_manifest_sha256": fuel_io.sha256_file(bank_manifest_path),
        "geometry_summary_sha256": fuel_io.sha256_file(
            geometry_summary_path
        ),
        "geometry_manifest_sha256": fuel_io.sha256_file(
            geometry_manifest_path
        ),
    }


def _started_payload(args, validated, determinism) -> dict:
    return {
        "schema": "exp416-pcnec-fuel-cache-started-v1",
        "status": "STARTED",
        "resume_allowed": False,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "formal_head": validated["head"],
        "interpreter": str(Path(sys.executable).resolve()),
        "device": str(args.device),
        "bank": str(validated["bank"]),
        "bank_sha256": str(args.bank_sha256),
        "geometry": str(validated["geometry"]),
        "geometry_sha256": str(args.geometry_sha256),
        "bank_receipt_sha256": validated["bank_receipt_sha256"],
        "bank_manifest_sha256": validated["bank_manifest_sha256"],
        "geometry_summary_sha256": validated["geometry_summary_sha256"],
        "geometry_manifest_sha256": validated[
            "geometry_manifest_sha256"
        ],
        "d0_config": str(validated["d0_config"]),
        "d0_config_sha256": EXPECTED_D0_CONFIG_SHA256,
        "d0_checkpoint": str(validated["d0_checkpoint"]),
        "d0_checkpoint_sha256": EXPECTED_D0_CHECKPOINT_SHA256,
        "clip_checkpoint": str(validated["clip_checkpoint"]),
        "clip_checkpoint_sha256": EXPECTED_CLIP_SHA256,
        "determinism": determinism,
        "source_files": {
            name: fuel_io.sha256_file(SCRIPT_DIR / name)
            for name in (
                "build_fuel_cache.py",
                "fuel_core.py",
                "fuel_io.py",
                "d0_feature_extractor.py",
                "clip_crop_encoder.py",
                "geometry_census.py",
            )
        },
    }


def _write_failure(output_dir, *, stage, error, started):
    failure = {
        "schema": "exp416-pcnec-fuel-cache-failure-v1",
        "status": "FAILED",
        "stage": str(stage),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "resume_allowed": False,
        "formal_head": started["formal_head"],
        "source_files": started["source_files"],
        "bank_sha256": started["bank_sha256"],
        "geometry_sha256": started["geometry_sha256"],
    }
    path = output_dir / "failure.json"
    if not path.exists() and not path.with_name(path.name + ".tmp").exists():
        fuel_io.atomic_json(path, failure)
        fuel_io.readback_json(path, failure)


def run_formal(args) -> dict:
    validated = validate_formal_args(args)
    fuel_io.assert_no_cuda_compute_processes()
    output_dir = validated["output_dir"]
    output_dir.mkdir(mode=0o755, parents=False, exist_ok=False)
    started = {
        "formal_head": validated["head"],
        "bank_sha256": str(args.bank_sha256),
        "geometry_sha256": str(args.geometry_sha256),
        "source_files": {
            name: fuel_io.sha256_file(SCRIPT_DIR / name)
            for name in (
                "build_fuel_cache.py",
                "fuel_core.py",
                "fuel_io.py",
                "d0_feature_extractor.py",
                "clip_crop_encoder.py",
                "geometry_census.py",
            )
        },
    }
    stage = "determinism_config"
    try:
        determinism = configure_formal_determinism()
        started = _started_payload(args, validated, determinism)
        stage = "started_write"
        started_path = output_dir / "started.json"
        fuel_io.atomic_json(started_path, started)
        fuel_io.readback_json(started_path, started)
        stage = "bank_geometry_gate"
        bank = _load_bank(
            validated["bank"],
            args.bank_sha256,
            expected_count=EXPECTED_TRAIN_COUNT,
        )
        geometry = _load_geometry(
            validated["geometry"],
            args.geometry_sha256,
            bank,
        )
        stage = "d0_clip_feature_cache"
        events = []
        arrays, receipt = build_stage3_arrays(
            bank=bank,
            geometry=geometry,
            image_root=validated["image_root"],
            args=args,
            device=torch.device(FORMAL_DEVICE),
            d0_factory=_default_d0_factory,
            clip_factory=_default_clip_factory,
            rgb_loader=_read_canonical_rgb,
            events=events,
        )
        receipt.update(
            {
                "formal_head": validated["head"],
                "bank_sha256": str(args.bank_sha256),
                "geometry_sha256": str(args.geometry_sha256),
                "bank_receipt_sha256": validated[
                    "bank_receipt_sha256"
                ],
                "bank_manifest_sha256": validated[
                    "bank_manifest_sha256"
                ],
                "geometry_summary_sha256": validated[
                    "geometry_summary_sha256"
                ],
                "geometry_manifest_sha256": validated[
                    "geometry_manifest_sha256"
                ],
                "determinism": determinism,
                "source_files": started["source_files"],
            }
        )
        stage = "cache_write"
        cache_path = output_dir / "fuel_cache.npz"
        fuel_io.atomic_npz(cache_path, arrays, compressed=False)
        fuel_io.readback_npz(cache_path, arrays)
        cache_sha256 = fuel_io.sha256_file(cache_path)
        receipt["fuel_cache"] = str(cache_path)
        receipt["fuel_cache_sha256"] = cache_sha256
        receipt_path = output_dir / "receipt.json"
        fuel_io.atomic_json(receipt_path, receipt)
        fuel_io.readback_json(receipt_path, receipt)
        result = {
            "schema": RESULT_SCHEMA,
            "status": "COMPLETE",
            "stage": "FUEL_CACHE_SEALED",
            "formal_head": validated["head"],
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "resume_allowed": False,
            "sample_count": int(len(arrays["relative_paths"])),
            "pair_count": int(len(arrays["pair_row"])),
            "fuel_cache_sha256": cache_sha256,
            "receipt_sha256": fuel_io.sha256_file(receipt_path),
            "d0_global_bank_recompute_bit_exact": True,
            "same_canonical_rgb_sha_across_d0_clip": True,
        }
        result_path = output_dir / "result.json"
        fuel_io.atomic_json(result_path, result)
        fuel_io.readback_json(result_path, result)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "formal_head": validated["head"],
            "source_files": started["source_files"],
            "bank_sha256": str(args.bank_sha256),
            "geometry_sha256": str(args.geometry_sha256),
            "files": {
                name: {
                    "bytes": int((output_dir / name).stat().st_size),
                    "sha256": fuel_io.sha256_file(output_dir / name),
                }
                for name in (
                    "started.json",
                    "fuel_cache.npz",
                    "receipt.json",
                    "result.json",
                )
            },
            "resume_allowed": False,
        }
        manifest_path = output_dir / "manifest.json"
        fuel_io.atomic_json(manifest_path, manifest)
        fuel_io.readback_json(manifest_path, manifest)
        fuel_io.seal_directory(output_dir)
    except BaseException as error:
        try:
            _write_failure(
                output_dir,
                stage=stage,
                error=error,
                started=started,
            )
            fuel_io.seal_directory(output_dir)
        except BaseException:
            pass
        raise
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


class _MockD0Wrapper:
    def __init__(self, events):
        self.inner = d0_feature_extractor._mock_extractor()
        self.events = events

    def encode(self, rgb, rectangles):
        return self.inner.encode(rgb, rectangles)

    def close(self):
        self.events.append("mock_d0_closed")
        self.inner.model = None


class _MockClipWrapper:
    def __init__(self, events):
        if "mock_d0_closed" not in events:
            raise RuntimeError("mock CLIP constructed before mock D0 close")
        self.inner = clip_crop_encoder._mock_encoder()
        self.events = events

    def encode_rectangles(self, rgb, rectangles):
        return self.inner.encode_rectangles(rgb, rectangles)

    def encode_whole_images(self, rgb):
        return self.inner.encode_whole_images(rgb)

    def close(self):
        self.events.append("mock_clip_closed")
        self.inner.model = None


def _mock_rgb_table(paths: Sequence[str]) -> dict[str, torch.Tensor]:
    table = {}
    for index, path in enumerate(paths):
        rgb = torch.zeros(
            3,
            fuel_core.IMAGE_HEIGHT,
            fuel_core.IMAGE_WIDTH,
            dtype=torch.float32,
        )
        rgb[0] = 0.05 + 0.05 * index
        rgb[1, : 60 + 10 * index] = 0.25 + 0.02 * index
        rgb[2, 180:] = 0.35 + 0.03 * index
        table[str(path)] = rgb.clamp(0.0, 1.0)
    return table


def _mock_bank_geometry(directory: Path):
    relative_paths = [
        "bounding_box_train/{:04d}_c{}_f{:06d}.jpg".format(
            pid, camera, pid * 10 + camera
        )
        for pid in (1, 2, 3, 4)
        for camera in (1, 2)
    ]
    records = fuel_core.build_train_records(relative_paths)
    rgb_table = _mock_rgb_table(relative_paths)
    rgb = torch.stack([rgb_table[path] for path in relative_paths])
    rectangles = np.zeros((len(records), SLOT_COUNT, 4), dtype=np.int16)
    canonical = np.zeros_like(rectangles)
    for slot in range(SLOT_COUNT):
        height = 48 + 8 * slot
        width = 24 + 4 * slot
        for row in range(len(records)):
            rectangles[row, slot] = (
                min(20 * slot + row, fuel_core.IMAGE_HEIGHT - height),
                min(6 * slot + row, fuel_core.IMAGE_WIDTH - width),
                height,
                width,
            )
            canonical[row, slot] = (
                min(30 + 55 * slot, fuel_core.IMAGE_HEIGHT - height),
                min(50, fuel_core.IMAGE_WIDTH - width),
                height,
                width,
            )
    mock_d0 = d0_feature_extractor._mock_extractor()
    d0_output = mock_d0.encode(rgb, rectangles)
    d0_global = d0_output["global_features"].numpy().astype(np.float32)
    mock_d0.model = None

    query = np.asarray((0, 0, 0, 0), dtype=np.int32)
    candidate = np.asarray((1, 2, 4, 6), dtype=np.int32)
    pids = np.asarray(
        [row["train_pid"] for row in records], dtype=np.int64
    )
    distance = np.maximum(
        2.0
        - 2.0
        * np.sum(d0_global[query] * d0_global[candidate], axis=1),
        0.0,
    ).astype(np.float32)
    bank_arrays = {
        "schema": np.asarray(BANK_SCHEMA),
        "relative_paths": np.asarray(relative_paths, dtype=np.str_),
        "raw_pids": np.asarray(
            [row["raw_pid"] for row in records], dtype=np.int64
        ),
        "relabeled_pids": pids,
        "camids": np.asarray(
            [row["camera"] for row in records], dtype=np.int16
        ),
        "image_sha256": np.asarray(
            [
                fuel_io.sha256_bytes(
                    ("mock-original-" + path).encode("utf-8")
                )
                for path in relative_paths
            ],
            dtype=np.str_,
        ),
        "d0_global": d0_global,
        "query_indices": query,
        "candidate_indices": candidate,
        "pair_is_impostor": (pids[query] != pids[candidate]),
        "d0_distance": distance,
        "query_offsets": np.asarray((0, len(query)), dtype=np.int64),
    }
    bank_path = directory / "bank.npz"
    fuel_io.atomic_npz(bank_path, bank_arrays, compressed=False)
    bank_sha = fuel_io.sha256_file(bank_path)

    availability = np.ones((len(records), SLOT_COUNT), dtype=np.bool_)
    # No camera-1 image is fully valid, making the genuine pair's matched
    # same-camera wrong-donor pool empty without dropping the pair.
    availability[1::2, 4] = False
    # One candidate has no common slot and must remain as E=0 UNDECIDED.
    availability[6] = False
    geometry_arrays = {
        "schema": np.asarray(GEOMETRY_SCHEMA),
        "relative_paths": np.asarray(relative_paths, dtype=np.str_),
        "image_sha256": bank_arrays["image_sha256"].copy(),
        "slot_names": np.asarray(
            (
                "head",
                "upper_torso_arms",
                "lower_torso",
                "upper_legs",
                "lower_legs_feet",
            ),
            dtype=np.str_,
        ),
        "availability": availability,
        "slot_confidence": availability.astype(np.float32),
        "instance_centers_xy": np.zeros(
            (len(records), SLOT_COUNT, 2), dtype=np.float32
        ),
        "instance_rectangles": rectangles,
        "canonical_centers_xy": np.zeros(
            (SLOT_COUNT, 2), dtype=np.float32
        ),
        "canonical_rectangles": canonical,
        "crop_hw": rectangles[0, :, 2:4].copy(),
        "geometry_gate_pass": np.asarray([False], dtype=np.bool_),
        "query_coverage": np.asarray([1.0], dtype=np.float64),
        "common_pair_count_by_slot": np.asarray(
            (3, 3, 3, 3, 2), dtype=np.int64
        ),
        "query_pid_count_by_slot": np.asarray(
            (1, 1, 1, 1, 1), dtype=np.int64
        ),
    }
    geometry_path = directory / "geometry.npz"
    fuel_io.atomic_npz(geometry_path, geometry_arrays, compressed=False)
    geometry_sha = fuel_io.sha256_file(geometry_path)
    return (
        bank_path,
        bank_sha,
        geometry_path,
        geometry_sha,
        rgb_table,
    )


def _run_fast_path_property_test() -> None:
    rng = np.random.default_rng(4161234)
    record_count = 64
    records = [
        {
            "relative_path": (
                "bounding_box_train/{:04d}_c{}_f{:06d}.jpg".format(
                    index % 13 + 1,
                    index % 8 + 1,
                    index,
                )
            ),
            "train_pid": int(index % 13),
            "camera": int(index % 8),
        }
        for index in range(record_count)
    ]
    donor_availability = rng.random((record_count, SLOT_COUNT)) > 0.20
    donor_availability[:24] = True
    donor_index = _build_wrong_rgb_donor_index(
        records, donor_availability
    )
    for _ in range(80):
        query_index, candidate_index = rng.integers(
            0, record_count, size=2
        ).tolist()
        for slot in range(SLOT_COUNT):
            kwargs = {
                "query_path": records[query_index]["relative_path"],
                "candidate_path": records[candidate_index]["relative_path"],
                "query_pid": records[query_index]["train_pid"],
                "candidate_pid": records[candidate_index]["train_pid"],
                "candidate_camera": records[candidate_index]["camera"],
                "slot": slot,
            }
            expected = fuel_core.select_wrong_rgb_donor(
                records,
                donor_availability,
                **kwargs,
            )
            observed = _select_wrong_rgb_donor_fast(
                donor_index,
                **kwargs,
            )
            if observed != expected:
                raise AssertionError("fast wrong-RGB donor differs from core")

    pair_count = 79
    query = rng.integers(0, record_count, size=pair_count)
    candidate = rng.integers(0, record_count, size=pair_count)
    candidate[candidate == query] = (
        candidate[candidate == query] + 1
    ) % record_count
    availability = rng.random((record_count, SLOT_COUNT)) > 0.30
    availability[query[0]] = False

    def normalized(shape):
        values = rng.normal(size=shape)
        values /= np.linalg.norm(values, axis=-1, keepdims=True)
        return values.astype(np.float32)

    instance_clip = normalized((record_count, SLOT_COUNT, 23))
    canonical_clip = normalized((record_count, SLOT_COUNT, 23))
    d0_slot = normalized((record_count, SLOT_COUNT, 17))
    global_clip = normalized((record_count, 23))
    instance_raw_hist = rng.random(
        (record_count, SLOT_COUNT, 512)
    ).astype(np.float32)
    canonical_raw_hist = rng.random(
        (record_count, SLOT_COUNT, 512)
    ).astype(np.float32)
    instance_raw_hist /= instance_raw_hist.sum(
        axis=-1, keepdims=True
    )
    canonical_raw_hist /= canonical_raw_hist.sum(
        axis=-1, keepdims=True
    )
    d0_distance = rng.random(pair_count).astype(np.float32)
    wrong_donor_indices = rng.integers(
        0,
        record_count,
        size=(pair_count, SLOT_COUNT),
        dtype=np.int32,
    )
    wrong_donor_invalid = rng.random(pair_count) < 0.20
    wrong_donor_indices[wrong_donor_invalid] = -1
    observed, observed_common = _compute_energy_chunk(
        query=query,
        candidate=candidate,
        availability=availability,
        d0_slot=d0_slot,
        instance_clip=instance_clip,
        canonical_clip=canonical_clip,
        global_clip=global_clip,
        instance_raw_hist=instance_raw_hist,
        canonical_raw_hist=canonical_raw_hist,
        d0_distance=d0_distance,
        wrong_donor_indices=wrong_donor_indices,
        wrong_donor_invalid=wrong_donor_invalid,
    )
    expected = np.empty_like(observed)
    slot_index = np.arange(SLOT_COUNT, dtype=np.int64)
    for row in range(pair_count):
        donor = None
        if not bool(wrong_donor_invalid[row]):
            donor = instance_clip[
                wrong_donor_indices[row], slot_index
            ]
        result = fuel_core.compute_pair_arm_energies(
            query_valid=availability[query[row]],
            candidate_valid=availability[candidate[row]],
            correct_clip_query=instance_clip[query[row]],
            correct_clip_candidate=instance_clip[candidate[row]],
            student_query=d0_slot[query[row]],
            student_candidate=d0_slot[candidate[row]],
            raw_hist_query=instance_raw_hist[query[row]],
            raw_hist_candidate=instance_raw_hist[candidate[row]],
            canonical_clip_query=canonical_clip[query[row]],
            canonical_clip_candidate=canonical_clip[candidate[row]],
            canonical_raw_hist_query=canonical_raw_hist[query[row]],
            canonical_raw_hist_candidate=canonical_raw_hist[candidate[row]],
            global_clip_query=global_clip[query[row]],
            global_clip_candidate=global_clip[candidate[row]],
            d0_distance=float(d0_distance[row]),
            wrong_donor_clip=donor,
        )
        if not np.array_equal(result["common"], observed_common[row]):
            raise AssertionError("batched common differs from core")
        expected[row] = [
            float(result["energies"][name])
            for name in fuel_core.ARM_NAMES
        ]
    if not np.allclose(observed, expected, rtol=2e-14, atol=2e-14):
        raise AssertionError(
            "batched pair energy differs from core: max_abs={}".format(
                float(np.max(np.abs(observed - expected)))
            )
        )


def run_self_test() -> None:
    self_test_root = REPOSITORY_ROOT / "tmp"
    self_test_root.mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="exp416-stage3-selftest-",
        dir=self_test_root,
    ) as raw:
        directory = Path(raw).resolve()
        (
            bank_path,
            bank_sha,
            geometry_path,
            geometry_sha,
            rgb_table,
        ) = _mock_bank_geometry(directory)
        bank = _load_bank(bank_path, bank_sha, expected_count=None)
        geometry = _load_geometry(
            geometry_path,
            geometry_sha,
            bank,
            enforce_formal_gate=False,
        )
        events = []

        def rgb_loader(path, expected_image_sha):
            del expected_image_sha
            relative = "/".join(Path(path).parts[-2:])
            rgb = rgb_table[relative].clone()
            return rgb, fuel_io.sha256_array(rgb.numpy())

        class Args:
            pass

        args = Args()

        def d0_factory(*, args, bank, device):
            del args, bank, device
            events.append("mock_d0_constructed")
            return _MockD0Wrapper(events)

        def clip_factory(*, args, device):
            del args, device
            events.append("mock_clip_constructed")
            return _MockClipWrapper(events)

        arrays, receipt = build_stage3_arrays(
            bank=bank,
            geometry=geometry,
            image_root=directory / "Occluded_Duke",
            args=args,
            device=torch.device("cpu"),
            d0_factory=d0_factory,
            clip_factory=clip_factory,
            rgb_loader=rgb_loader,
            events=events,
        )
        assert set(arrays) == CACHE_FIELDS
        assert str(arrays["bank_sha256"].item()) == bank_sha
        assert str(arrays["geometry_sha256"].item()) == geometry_sha
        assert arrays["pair_energy"].shape == (
            4,
            len(fuel_core.ARM_NAMES),
        )
        assert arrays["common"].shape == (4, SLOT_COUNT)
        assert arrays["undecided"].tolist() == [False, False, False, True]
        assert arrays["wrong_donor_invalid"].tolist()[0] is True
        assert np.array_equal(arrays["pair_row"], np.arange(4))
        assert np.array_equal(arrays["image_sha256"], bank["image_sha256"])
        assert receipt["d0_global_bank_recompute_bit_exact"]
        assert receipt["same_canonical_rgb_sha_across_d0_clip"]
        assert events.index("mock_d0_closed") < events.index(
            "mock_clip_constructed"
        )
        assert events.count("mock_d0_constructed") == 1
        assert events.count("mock_d0_closed") == 1
        assert events.count("mock_clip_constructed") == 1
        assert events.count("mock_clip_closed") == 1
        undecided_row = int(np.flatnonzero(arrays["undecided"])[0])
        dependent = [
            index
            for index, name in enumerate(fuel_core.ARM_NAMES)
            if name not in {"global_clip", "d0_only"}
        ]
        assert np.all(arrays["pair_energy"][undecided_row, dependent] == 0.0)

        output_dir = directory / "sealed"
        output_dir.mkdir()
        cache_path = output_dir / "fuel_cache.npz"
        fuel_io.atomic_npz(cache_path, arrays, compressed=False)
        fuel_io.readback_npz(cache_path, arrays)

        bad_geometry = dict(geometry)
        bad_geometry["relative_paths"] = geometry["relative_paths"].copy()
        bad_geometry["relative_paths"][0] = (
            "bounding_box_train/9999_c1_f000001.jpg"
        )
        try:
            if tuple(map(str, bad_geometry["relative_paths"])) != bank[
                "_path_tuple"
            ]:
                raise RuntimeError("injected path order mismatch")
        except RuntimeError:
            pass
        else:
            raise AssertionError("path-order failure injection was accepted")

        broken_bank = dict(bank)
        broken_bank["d0_global"] = bank["d0_global"].copy()
        broken_bank["d0_global"][0, 0] += np.float32(1e-3)
        try:
            build_stage3_arrays(
                bank=broken_bank,
                geometry=geometry,
                image_root=directory / "Occluded_Duke",
                args=args,
                device=torch.device("cpu"),
                d0_factory=d0_factory,
                clip_factory=clip_factory,
                rgb_loader=rgb_loader,
                events=[],
            )
        except RuntimeError as error:
            assert "bit-exact" in str(error)
        else:
            raise AssertionError("D0 global mismatch was accepted")

        try:
            _load_geometry(
                geometry_path,
                "0" * 64,
                bank,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("geometry SHA failure injection was accepted")
    _run_fast_path_property_test()
    print("EXP416_BUILD_FUEL_CACHE_SELF_TEST=PASS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--bank")
    parser.add_argument("--bank-sha256")
    parser.add_argument("--geometry")
    parser.add_argument("--geometry-sha256")
    parser.add_argument("--data-root", default=str(FIXED_DATA_ROOT))
    parser.add_argument(
        "--d0-config",
        default=str(
            (REPOSITORY_ROOT / FIXED_D0_CONFIG_RELATIVE).resolve()
        ),
    )
    parser.add_argument(
        "--d0-checkpoint", default=str(FIXED_D0_CHECKPOINT)
    )
    parser.add_argument(
        "--clip-checkpoint", default=str(FIXED_CLIP_CHECKPOINT)
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default=FORMAL_DEVICE)
    parser.add_argument("--d0-microbatch", type=int, default=D0_MICROBATCH)
    parser.add_argument(
        "--clip-microbatch", type=int, default=CLIP_MICROBATCH
    )
    parser.add_argument("--batch-size", type=int, default=OUTER_BATCH_SIZE)
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-build-fuel-cache-sha256")
    parser.add_argument("--expected-fuel-core-sha256")
    parser.add_argument("--expected-fuel-io-sha256")
    parser.add_argument("--expected-d0-feature-extractor-sha256")
    parser.add_argument("--expected-clip-crop-encoder-sha256")
    parser.add_argument("--expected-geometry-census-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    required = (
        "bank",
        "bank_sha256",
        "geometry",
        "geometry_sha256",
        "output_dir",
        "expected_head",
        "expected_build_fuel_cache_sha256",
        "expected_fuel_core_sha256",
        "expected_fuel_io_sha256",
        "expected_d0_feature_extractor_sha256",
        "expected_clip_crop_encoder_sha256",
        "expected_geometry_census_sha256",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError("missing formal arguments: " + ",".join(missing))
    run_formal(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
