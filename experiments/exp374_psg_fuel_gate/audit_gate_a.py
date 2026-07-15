#!/usr/bin/env python3
"""Audit-only runner for the frozen exp374 legacy PSG fuel gate.

The CLI is intentionally split into ``prepare``, ``run``, and ``summarize``.
No phase trains or mutates weights.  The design review authorizes writing this
file, but running any phase remains forbidden until a separate static code
review changes the exp374 test gate.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import importlib.metadata
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg as global_cfg  # noqa: E402
from datasets.occluded_duke import OccludedDukeMTMC  # noqa: E402
from datasets.pose_dataset import (  # noqa: E402
    PoseImageDataset,
    pose_val_collate_fn,
)
from model import make_model  # noqa: E402
from model.pose_backbone_model import (  # noqa: E402
    PoseBackboneModel,
    SCENE_HEATMAPS_UNSET,
)
from processor.processor import _pose_to_device  # noqa: E402
from utils.metrics import euclidean_distance, eval_func  # noqa: E402

from experiments.exp374_psg_fuel_gate.protocol import (  # noqa: E402
    ANATOMICAL_GROUPS,
    BASELINE_SEEDS,
    GateProtocolError,
    K_SEQUENCE,
    MAPPING_SEEDS,
    SCENE_METADATA_SCHEMA_V2,
    SPLIT_RELATION_SCHEMA_V2,
    SceneRecord,
    absolute_centroid_targets,
    actual_psg_input,
    aggregate_mapping_queries,
    atomic_write_bytes,
    atomic_write_json,
    canonical_scene_record_set_summary,
    canonical_json_bytes,
    core_schedule,
    create_execution_directory,
    fit_normalized_centroid_targets,
    fsync_directory,
    gate_decision,
    intervention_strength,
    per_query_metrics,
    prepare_split_mappings,
    publish_directory,
    replace_group_channels,
    require,
    sha256_bytes,
    sha256_file,
    sha256_tensor,
    simultaneous_intervals,
    summarize_scene,
    apply_scene_centroid_control,
)


DEFAULT_CHECKPOINTS = (
    {
        "seed": 1234,
        "weight": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed1234/transformer_120.pth",
        "weight_sha256": "51c37c49537119deb38bce08702fb5a3ea7fc2b4bc251f1b8f4eebd9ddf6ec69",
        "flat_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed1234/test_default.txt",
        "train_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed1234/train_log.txt",
        "expected_mAP": 58.3,
        "expected_R1": 68.1,
    },
    {
        "seed": 42,
        "weight": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed42/transformer_120.pth",
        "weight_sha256": "174e8f9316f60219cbeca292457bf976e73cc88df6fddf9d83f94a89280d2a75",
        "flat_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed42/test_default.txt",
        "train_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed42/train_log.txt",
        "expected_mAP": 57.5,
        "expected_R1": 66.7,
    },
    {
        "seed": 2024,
        "weight": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed2024/transformer_120.pth",
        "weight_sha256": "c525e9c1ba90d896b703f6eca9a117ba1a97cd08fbab02618021bf20efd09f3d",
        "flat_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed2024/test_default.txt",
        "train_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed2024/train_log.txt",
        "expected_mAP": 58.0,
        "expected_R1": 68.4,
    },
)


SCENE_METADATA_SCHEMA = SCENE_METADATA_SCHEMA_V2
RELATION_REPORT_SCHEMA = SPLIT_RELATION_SCHEMA_V2
BURNED_A02_ROOT = Path("/home/afr/exp374_gate_a_8ca57ed_a02_20260715")
OFFICIAL_REPOSITORY = "https://github.com/lightas/Occluded-DukeMTMC-Dataset"
OFFICIAL_COMMIT = "dcba185bb20cbd53d3da2c8a4bfc25aa6971ce1d"
FILENAME_REGEX = r"^(\d{4})_c([1-8])_f(\d{7})\.jpg$"
FILENAME_PATTERN = re.compile(FILENAME_REGEX)
OFFICIAL_SPLITS = {
    "train": {
        "root": "bounding_box_train",
        "list": "train.list",
        "count": 15618,
        "raw_bytes": 327978,
        "raw_sha256": "dadffee79d8601545ca2217a38406c1cb6dab39d0b4b0c6370c8486738dee059",
        "canonical_bytes": 359216,
        "canonical_sha256": "96aa7aa80a3bb09cb48e16089f04d13ef51575442ba1aab162721add27c07189",
        "pose_index_bytes": 4713227,
        "pose_index_sha256": "63dc1f5db9bab90717a484dfc5033a197ee8b95f9c94a92f2082dc18a588103b",
    },
    "query": {
        "root": "query",
        "list": "query.list",
        "count": 2210,
        "raw_bytes": 46410,
        "raw_sha256": "fb5e1b1a749a0ab8602414bc9159e7a03216c2bdc519b5a4e513e05e3f612333",
        "canonical_bytes": 50832,
        "canonical_sha256": "e7bff615f1722a10be3d108341d0e9ceb2934ebbfcfb7506957100201cdd887b",
        "pose_index_bytes": 719516,
        "pose_index_sha256": "6b60745066f9b921d347558db3ad8ee7021ad103182db8afe7fffd510bc5f7c4",
    },
    "gallery": {
        "root": "bounding_box_test",
        "list": "gallery.list",
        "count": 17661,
        "raw_bytes": 370881,
        "raw_sha256": "0393fa86344ef4c220a5589aaad409f3adda1e14e39fc8425c80e90196065fca",
        "canonical_bytes": 406205,
        "canonical_sha256": "81d57ac6b5015497133d9771b18ded6fdbb60c430341ffd589da291f3a799271",
        "pose_index_bytes": 5320783,
        "pose_index_sha256": "d5f2e14f8665ce045dfa8085dbdff031a1c9de7a7c258a594802e2a63ccefabc",
    },
}

OFFICIAL_SOURCE_PID_COUNTS = {"train": 702, "query": 519, "gallery": 1110}
OFFICIAL_QUERY_GALLERY_COUNTS = {
    "path_overlap_count": 0,
    "rgb_sha256_overlap_count": 1870,
    "pose_path_sha256_overlap_count": 0,
    "pose_content_sha256_overlap_count": 1870,
    "full_pose_person_path_overlap_count": 0,
    "full_pose_person_content_overlap_count": 3486,
    "effective_pose_person_path_overlap_count": 0,
    "effective_pose_person_content_overlap_count": 3486,
    "source_pid_overlap_count": 519,
    "rgb_content_forbidden_group_count": 0,
    "pose_content_forbidden_group_count": 0,
    "full_pose_person_content_forbidden_count": 0,
    "effective_pose_person_content_forbidden_count": 0,
    "forbidden_overlap_count": 0,
}
OFFICIAL_ALLOWED_PAIR_COUNT = 1870

RELATION_EXACT = {
    "shared_basename": (1870, 43012,
                        "e940491d5471d3b976095335d1472e734fe8e6a76c192a3e98d5d0e9dbb7567f"),
    "shared_rgb_legacy": (1870, 125291,
                          "e02e1be9b04d1428691809d81627235a3c4bb489e794d9b125c1f6c9c55b2e0c"),
    "shared_rgb": (1870, 125292,
                   "54a624b1490cecfa77677ae275229d59b68714d5216bd7ab5bf749d66b9a552d"),
    "endpoint_pairs": (1870, 21666,
                       "4135cdc4bb3cecd52dcf79423cf24d53595ce695a8b91544e2732be4bf3ebdfc"),
    "joint_metadata": (1870, 566372,
                       "e59e8e935c9aa1cb19888ad23ab4f23a052cdda0d35fbb84928ae0d1ea1c3f51"),
    "joint_pairs": (1870, 3542413,
                    "b82fd6aa1a81faf85e80b876a62bd892d259e3c7e1e9bb9d9a381641dbb3df93"),
}

OFFICIAL_QG_CACHE_FILES = {
    "query": {
        "heatmaps": (461660288,
                     "ce908ee4e57a602f03e66340ab66c16097d0ff9f26a678f5d249a4ba10f7b45f"),
        "scores": (150408,
                   "30a40c5d4c349d38b6527b8ad13b4b3f2b5e4dbfdbd0cf285d484a0be1116ce4"),
        "nuisance": (1679728,
                     "bdf7c98729b369904187d8711a39f9441013c4a0b423e22cb4e468aea6b90cfb"),
    },
    "gallery": {
        "heatmaps": (3689312384,
                     "645c352137680dcde416a33b0abe37fc32109000243a1c97b6310c9172c90d3d"),
        "scores": (1201076,
                   "8385ea376e03b460d3b5e7c3084712b2fac70b81e74b1a71b19e9d3c6096b09d"),
        "nuisance": (13422488,
                     "b3127a3bfb388a8ce5542386bde6f715be78096e9e5da44b6408054f635d2c65"),
    },
}

RELATION_REPORT_KEYS = frozenset({
    "schema", "official_source", "official_lists", "split_counts",
    "within_split", "cross_split", "relations", "pairs",
    "relation_report_sha256",
})
OFFICIAL_LIST_ROW_KEYS = frozenset({
    "rgb_root", "list", "count", "raw_bytes", "raw_sha256",
    "canonical_bytes", "canonical_sha256", "pose_index_bytes",
    "pose_index_sha256",
})
WITHIN_SPLIT_ROW_KEYS = frozenset({
    "path_duplicate_count", "rgb_sha256_duplicate_count",
    "pose_path_sha256_duplicate_count", "pose_content_sha256_duplicate_count",
    "full_pose_person_path_duplicate_count",
    "full_pose_person_content_duplicate_count",
    "effective_pose_person_path_duplicate_count",
    "effective_pose_person_content_duplicate_count", "source_pid_count",
    "target_outside_effective_count",
})
CROSS_SPLIT_ROW_KEYS = frozenset({
    "path_overlap_count", "rgb_sha256_overlap_count",
    "pose_path_sha256_overlap_count", "pose_content_sha256_overlap_count",
    "full_pose_person_path_overlap_count",
    "full_pose_person_content_overlap_count",
    "effective_pose_person_path_overlap_count",
    "effective_pose_person_content_overlap_count", "source_pid_overlap_count",
    "rgb_content_forbidden_group_count", "pose_content_forbidden_group_count",
    "full_pose_person_content_forbidden_count",
    "effective_pose_person_content_forbidden_count", "forbidden_overlap_count",
})
RELATIONS_KEYS = frozenset({
    "query_gallery_shared_basenames",
    "query_gallery_shared_rgb_sha256_legacy",
    "query_gallery_shared_rgb_sha256", "query_gallery_endpoint_pairs",
    "query_gallery_joint_metadata_pairs", "query_gallery_joint_pairs",
    "split_record_sets", "allowed_pair_count", "junk_true_count",
    "junk_false_count", "forbidden_pair_count",
})
SUMMARY_KEYS = frozenset({"count", "canonical_bytes", "sha256"})
PAIR_KEYS = frozenset({
    "basename", "camid", "effective_pose_person_sha256", "frame",
    "full_pose_person_sha256", "gallery_effective_pose_person_relpaths",
    "gallery_full_pose_person_relpaths", "gallery_index",
    "gallery_pose_path_sha256", "gallery_rgb_relpath",
    "gallery_target_person_idx", "hraw_sha256", "nuisance_sha256",
    "person_count", "pid", "pose_content_sha256",
    "query_effective_pose_person_relpaths", "query_full_pose_person_relpaths",
    "query_index", "query_pose_path_sha256", "query_rgb_relpath",
    "query_target_person_idx", "report", "rgb_sha256", "score_sha256",
    "source_camid", "source_frame_id", "source_pid", "viewid",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--config-file",
        default=str(ROOT / "configs/occluded_duke/pose_backbone_psg.yml"),
    )
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--checkpoint-manifest")
    prepare.add_argument("--resume")
    prepare.add_argument("--anchor-chunk", type=int, default=16)
    prepare.add_argument("opts", nargs=argparse.REMAINDER)

    run = subparsers.add_parser("run")
    run.add_argument("--execution-dir", required=True)
    run.add_argument("--device", default="cuda:0")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--execution-dir", required=True)
    summarize.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _is_burned_execution(path: Path) -> bool:
    resolved = str(Path(path).resolve())
    burned = str(Path(BURNED_A02_ROOT).resolve())
    return resolved == burned or resolved.startswith(burned + os.sep)


def _assert_not_burned_execution(path: Path) -> None:
    resolved = str(Path(path).resolve())
    require(not _is_burned_execution(path),
            "E_A02_BURNED_EXECUTION", resolved)


def resolved_config(config_file: Path, opts: Sequence[str]):
    local_cfg = global_cfg.clone()
    local_cfg.defrost()
    local_cfg.merge_from_file(str(config_file))
    effective_opts = list(opts)
    if effective_opts and effective_opts[0] == "--":
        effective_opts = effective_opts[1:]
    if effective_opts:
        local_cfg.merge_from_list(effective_opts)
    local_cfg.defrost()
    local_cfg.TEST.FLIP_TEST = False
    local_cfg.TEST.RE_RANKING = False
    local_cfg.TEST.NFC = False
    local_cfg.TEST.POWER_NORM = 0.0
    local_cfg.TEST.NECK_FEAT = "before"
    local_cfg.TEST.FEAT_NORM = "yes"
    local_cfg.MODEL.POSE_USE_TARGET_HEATMAP = False
    local_cfg.MODEL.POSE_PSG_STAGES = [-1]
    local_cfg.freeze()
    return local_cfg


def _torch_load_checkpoint_bytes(payload: bytes):
    buffer = io.BytesIO(payload)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=False)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")


def _normalized_checkpoint_state(raw: object) -> Dict[str, torch.Tensor]:
    if isinstance(raw, Mapping) and "state_dict" in raw and isinstance(raw["state_dict"], Mapping):
        raw = raw["state_dict"]
    require(isinstance(raw, Mapping), "E_CHECKPOINT_FORMAT", type(raw).__name__)
    state: Dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        key_text = str(key)
        normalized = key_text[len("module."):] if key_text.startswith("module.") else key_text
        require(normalized not in state, "E_CHECKPOINT_KEY_COLLISION", normalized)
        state[normalized] = value
    require(all(torch.is_tensor(value) for value in state.values()),
            "E_CHECKPOINT_FORMAT", "non-tensor state value")
    return state


def checkpoint_alias_state_audit_from_state(
    state: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    audit: Dict[str, object] = {}
    for block_index, key in enumerate(("s3_b0", "s3_b1")):
        for suffix in ("encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "encoder.2.bias"):
            canonical_key = f"psg_modules_dict.{key}.{suffix}"
            alias_key = f"psg_modules.{block_index}.{suffix}"
            require(canonical_key in state and alias_key in state,
                    "E_CHECKPOINT_ALIAS_KEY", f"{canonical_key}/{alias_key}")
            require(state[canonical_key].shape == state[alias_key].shape,
                    "E_CHECKPOINT_ALIAS_SHAPE", canonical_key)
            require(torch.equal(state[canonical_key], state[alias_key]),
                    "E_CHECKPOINT_ALIAS_VALUE", canonical_key)
            audit[canonical_key] = {
                "alias_key": alias_key,
                "shape": list(state[canonical_key].shape),
                "sha256": sha256_tensor(state[canonical_key]),
            }
    return audit


def parse_flat_log_metrics_bytes(payload: bytes, source: str) -> Dict[str, object]:
    text = payload.decode("utf-8", errors="strict")
    map_values = [float(value) for value in re.findall(
        r"\bmAP\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", text)]
    rank1_values = [float(value) for value in re.findall(
        r"\bRank-1(?![0-9])\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", text)]
    require(map_values and rank1_values, "E_FLAT_LOG_PARSE", source)
    require(len(map_values) == 1 and len(rank1_values) == 1,
            "E_FLAT_LOG_AMBIGUOUS",
            f"{source}: mAP={len(map_values)}, R1={len(rank1_values)}")
    require(all(0.0 <= value <= 100.0 for value in map_values + rank1_values),
            "E_FLAT_LOG_PARSE", "metric range")
    return {
        "mAP": map_values[-1],
        "R1": rank1_values[-1],
        "mAP_occurrences": len(map_values),
        "R1_occurrences": len(rank1_values),
    }


def checkpoint_specs(path: str | None) -> List[Dict[str, object]]:
    if path is None:
        specs = [dict(value) for value in DEFAULT_CHECKPOINTS]
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        specs = [dict(value) for value in payload]
    require(len(specs) == 3, "E_CHECKPOINT_COUNT", str(len(specs)))
    require({int(value["seed"]) for value in specs} == {42, 1234, 2024},
            "E_CHECKPOINT_SEEDS", str([value["seed"] for value in specs]))
    for spec in specs:
        expected_weight_sha = str(spec["weight_sha256"])
        flat_log_payload: bytes | None = None
        for key in ("weight", "flat_log", "train_log"):
            file_path = Path(str(spec[key])).resolve()
            require(file_path.is_file(), "E_CHECKPOINT_ASSET", str(file_path))
            spec[key] = str(file_path)
            asset_payload = file_path.read_bytes()
            actual_sha = sha256_bytes(asset_payload)
            if key == "weight":
                require(actual_sha == expected_weight_sha, "E_CHECKPOINT_SHA", str(file_path))
                spec["weight_sha256"] = actual_sha
                state = _normalized_checkpoint_state(
                    _torch_load_checkpoint_bytes(asset_payload))
                spec["psg_alias_audit"] = checkpoint_alias_state_audit_from_state(state)
            else:
                spec[f"{key}_sha256"] = actual_sha
            if key == "flat_log":
                flat_log_payload = asset_payload
        require(flat_log_payload is not None, "E_FLAT_LOG_PARSE", str(spec["seed"]))
        parsed = parse_flat_log_metrics_bytes(flat_log_payload, str(spec["flat_log"]))
        require(f"{float(spec['expected_mAP']):.1f}" == f"{parsed['mAP']:.1f}",
                "E_FLAT_LOG_MANIFEST", f"seed={spec['seed']}/mAP")
        require(f"{float(spec['expected_R1']):.1f}" == f"{parsed['R1']:.1f}",
                "E_FLAT_LOG_MANIFEST", f"seed={spec['seed']}/R1")
        spec["flat_log_metrics"] = parsed
    return sorted(specs, key=lambda value: int(value["seed"]))


def repository_manifest() -> Dict[str, object]:
    def command(*arguments: str) -> bytes:
        return subprocess.check_output(arguments, cwd=ROOT)

    commit = command("git", "rev-parse", "HEAD").decode().strip()
    status = command("git", "status", "--porcelain=v1")
    diff = command("git", "diff", "--binary", "HEAD")
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": sha256_bytes(status),
        "dirty_diff_sha256": sha256_bytes(diff),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "packages_sha256": sha256_bytes(canonical_json_bytes(packages)),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(Path(__file__).with_name("protocol.py")),
        "model_sha256": sha256_file(ROOT / "model/pose_backbone_model.py"),
    }


def runtime_environment_manifest(device: torch.device | None = None) -> Dict[str, object]:
    """Freeze every runtime property that may alter audit descriptors."""
    cuda_available = torch.cuda.is_available()
    payload: Dict[str, object] = {
        "hostname": socket.gethostname(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(getattr(
            torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if device is not None:
        require(device.type == "cuda" and cuda_available, "E_DEVICE", str(device))
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        payload["selected_device"] = {
            "index": int(index),
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "major": int(properties.major),
            "minor": int(properties.minor),
        }
    return payload


def direct_datasets(local_cfg):
    dataset = OccludedDukeMTMC(root=local_cfg.DATASETS.ROOT_DIR)
    pose_base = Path(local_cfg.MODEL.POSE_DATA_DIR) / "pose_data"
    kwargs = {
        "is_train": False,
        "img_size": tuple(local_cfg.INPUT.SIZE_TEST),
        "pixel_mean": local_cfg.INPUT.PIXEL_MEAN,
        "pixel_std": local_cfg.INPUT.PIXEL_STD,
        "heatmap_size": tuple(local_cfg.MODEL.POSE_HEATMAP_SIZE),
    }
    split_datasets = {
        "train": PoseImageDataset(dataset.train, pose_dir=str(pose_base / "train"), **kwargs),
        "query": PoseImageDataset(dataset.query, pose_dir=str(pose_base / "query"), **kwargs),
        "gallery": PoseImageDataset(dataset.gallery, pose_dir=str(pose_base / "gallery"), **kwargs),
    }
    return dataset, split_datasets


def split_loader(dataset: PoseImageDataset, local_cfg) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(local_cfg.TEST.IMS_PER_BATCH),
        shuffle=False,
        num_workers=int(local_cfg.DATALOADER.NUM_WORKERS),
        collate_fn=pose_val_collate_fn,
        pin_memory=False,
        drop_last=False,
    )


def _file_identity(value: os.stat_result) -> Tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_nlink), int(value.st_size), int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _stable_regular_file(path: Path, return_bytes: bool = False):
    """Read one non-symlink regular file through a stable descriptor."""

    path = Path(path)
    try:
        before = path.lstat()
    except OSError as error:
        raise GateProtocolError(
            "E_RELATION_FILE_IO",
            f"{path}: {error.__class__.__name__}") from error
    require(stat.S_ISREG(before.st_mode), "E_RELATION_FILE_TYPE", str(path))
    digest = hashlib.sha256()
    chunks: List[bytes] | None = [] if return_bytes else None
    try:
        handle = path.open("rb")
    except OSError as error:
        raise GateProtocolError(
            "E_RELATION_FILE_IO",
            f"{path}: {error.__class__.__name__}") from error
    try:
        with handle:
            descriptor_before = os.fstat(handle.fileno())
            require(_file_identity(before) == _file_identity(descriptor_before),
                    "E_RELATION_FILE_TOCTOU", str(path))
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            descriptor_after = os.fstat(handle.fileno())
    except GateProtocolError:
        raise
    except OSError as error:
        raise GateProtocolError(
            "E_RELATION_FILE_TOCTOU",
            f"{path}: {error.__class__.__name__}") from error
    try:
        after = path.lstat()
    except OSError as error:
        raise GateProtocolError(
            "E_RELATION_FILE_TOCTOU",
            f"{path}: {error.__class__.__name__}") from error
    identity = _file_identity(before)
    require(identity == _file_identity(descriptor_after) == _file_identity(after),
            "E_RELATION_FILE_TOCTOU", str(path))
    payload = b"".join(chunks) if chunks is not None else None
    return {
        "bytes": int(before.st_size),
        "sha256": digest.hexdigest(),
    }, identity, payload


def _register_identity(registry: MutableMapping[str, Tuple[int, ...]],
                       path: Path, identity: Tuple[int, ...]) -> None:
    key = str(Path(path))
    if key in registry:
        require(registry[key] == identity, "E_RELATION_FILE_TOCTOU", key)
    else:
        registry[key] = identity


def _recheck_identities(registry: Mapping[str, Tuple[int, ...]]) -> None:
    for value, expected in sorted(registry.items()):
        try:
            current = Path(value).lstat()
        except OSError as error:
            raise GateProtocolError(
                "E_RELATION_FILE_TOCTOU",
                f"{value}: {error.__class__.__name__}") from error
        require(stat.S_ISREG(current.st_mode), "E_RELATION_FILE_TYPE", value)
        require(_file_identity(current) == expected,
                "E_RELATION_FILE_TOCTOU", value)


def _stable_json(path: Path, registry: MutableMapping[str, Tuple[int, ...]],
                 code: str):
    report, identity, raw = _stable_regular_file(path, return_bytes=True)
    _register_identity(registry, path, identity)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateProtocolError(code, f"{path}: {error}") from error
    return payload, report


def _require_direct_child(path: Path, parent: Path, code: str) -> None:
    path = Path(path)
    parent = Path(parent)
    require(path.parent == parent and path.name not in {"", ".", ".."},
            code, str(path))


def _lexical_absolute(path: str | Path) -> Path:
    """Normalize ``.``/``..`` without resolving any symlink component."""

    return Path(os.path.abspath(os.fspath(path)))


def _exact_lexical_child(
    configured: str | Path,
    parent: str | Path,
    child_parts: Sequence[str],
    code: str,
) -> Path:
    """Bind the un-resolved configured spelling to one frozen child path."""

    raw = Path(configured)
    expected = Path(parent).joinpath(*child_parts)
    require(raw == expected and ".." not in raw.parts,
            code, f"{raw}!={expected}")
    return _lexical_absolute(raw)


def _require_real_directory(path: Path, code: str) -> None:
    """Require the final lexical path component to be a real directory."""

    path = Path(path)
    try:
        value = path.lstat()
    except OSError as error:
        raise GateProtocolError(code, f"{path}: {error}") from error
    require(stat.S_ISDIR(value.st_mode), code, str(path))


def _source_labels(basename: str) -> Tuple[int, int, int]:
    match = FILENAME_PATTERN.fullmatch(basename)
    require(match is not None, "E_RELATION_FILENAME", basename)
    return (
        int(match.group(1)),
        int(match.group(2)) - 1,
        int(match.group(3)),
    )


def _pose_asset_manifest_v2(
    dataset: PoseImageDataset,
    image_path: str,
    file_hash_cache: MutableMapping[str, str],
    identity_registry: MutableMapping[str, Tuple[int, ...]] | None = None,
) -> Dict[str, object]:
    """Freeze full and loader-effective constituent identities for one RGB."""

    basename = Path(image_path).name
    entry = dataset.index.get(basename)
    require(isinstance(entry, Mapping), "E_POSE_INDEX_MISSING", image_path)
    raw_persons = entry.get("persons")
    require(isinstance(raw_persons, list) and raw_persons,
            "E_POSE_PERSONS_EMPTY", image_path)
    require(all(isinstance(value, str) for value in raw_persons),
            "E_POSE_PERSON_ENTRY", basename)
    require(len(set(raw_persons)) == len(raw_persons),
            "E_POSE_PERSON_DUPLICATE", basename)
    pose_root = Path(dataset.pose_dir).resolve()
    full_relpaths: List[str] = []
    full_paths: List[str] = []
    full_sha256: List[str] = []
    basename_to_asset: Dict[str, Tuple[str, str, str]] = {}
    split = pose_root.name
    for value in raw_persons:
        candidate = Path(value)
        require(
            not candidate.is_absolute() and value == candidate.name
            and value not in {"", ".", ".."} and "/" not in value
            and "\\" not in value and candidate.suffix == ".npz",
            "E_POSE_PERSON_PATH", value,
        )
        path = (pose_root / value).resolve()
        require(path.parent == pose_root, "E_POSE_PERSON_ROOT", str(path))
        if str(path) not in file_hash_cache:
            report, identity, _unused = _stable_regular_file(path)
            file_hash_cache[str(path)] = str(report["sha256"])
            if identity_registry is not None:
                _register_identity(identity_registry, path, identity)
        elif identity_registry is not None:
            try:
                current = path.lstat()
            except OSError as error:
                raise GateProtocolError(
                    "E_RELATION_FILE_TOCTOU",
                    f"{path}: {error.__class__.__name__}") from error
            _register_identity(identity_registry, path, _file_identity(current))
        relpath = f"pose_data/{split}/{value}"
        digest = file_hash_cache[str(path)]
        full_relpaths.append(relpath)
        full_paths.append(str(path))
        full_sha256.append(digest)
        basename_to_asset[value] = (relpath, str(path), digest)

    raw_target = entry.get("target_person_idx", 0)
    require(type(raw_target) is int, "E_POSE_TARGET_INDEX", basename)
    target_person_idx = int(raw_target)
    require(0 <= target_person_idx < len(raw_persons),
            "E_POSE_TARGET_INDEX", basename)
    effective_names = list(raw_persons[:int(dataset.max_persons)])
    require(bool(effective_names), "E_POSE_PERSONS_EMPTY", basename)
    target_outside_effective = target_person_idx >= len(effective_names)
    if 0 < target_person_idx < len(effective_names):
        target = effective_names.pop(target_person_idx)
        effective_names.insert(0, target)
    effective_assets = [basename_to_asset[value] for value in effective_names]
    source_pid, source_camid, source_frame_id = _source_labels(basename)
    return {
        "pose_path_sha256": sha256_bytes(canonical_json_bytes(full_paths)),
        "pose_content_sha256": sha256_bytes(canonical_json_bytes(full_sha256)),
        "source_pid": source_pid,
        "source_camid": source_camid,
        "source_frame_id": source_frame_id,
        "target_person_idx": target_person_idx,
        "target_outside_effective": target_outside_effective,
        "full_pose_person_relpaths": tuple(full_relpaths),
        "full_pose_person_paths": tuple(full_paths),
        "full_pose_person_sha256": tuple(full_sha256),
        "effective_pose_person_relpaths": tuple(value[0] for value in effective_assets),
        "effective_pose_person_paths": tuple(value[1] for value in effective_assets),
        "effective_pose_person_sha256": tuple(value[2] for value in effective_assets),
    }


def _new_memmap(path: Path, shape: Tuple[int, ...], dtype: str):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


ACTIVE_PSG_BLOCK_SHAPES = {
    "s3_b0": (12, 4),
    "s3_b1": (12, 4),
}

CENTROID_RUNTIME_SECONDARY_CODES = frozenset({
    "E_CENTROID_WEAK_CHANNEL",
    "E_CENTROID_EMPTY",
    "E_CENTROID_TARGET",
    "E_CENTROID_ZERO",
    "E_CENTROID_COMMUTATION",
    "E_CENTROID_ERROR",
    "E_CENTROID_L1",
    "E_CENTROID_PEAK",
    "E_CENTROID_ENTROPY",
    "E_CENTROID_NEGATIVE_ZERO",
    "E_CENTROID_NEGATIVE_L1",
})


def _new_signed_scene_audit() -> Dict[str, object]:
    return {
        "raw_min": float("inf"),
        "negative_element_count": 0,
        "negative_sample_count": 0,
        "negative_sample_channel_count": 0,
        "negative_channel_indices_0based": set(),
        "negative_absolute_mass": 0.0,
        "sample_count": 0,
    }


def _new_actual_space_audit() -> Dict[str, object]:
    return {
        "sample_count": 0,
        "blocks": {
            key: {
                "shape": shape,
                "sraw": hashlib.sha256(),
                "spos": hashlib.sha256(),
                "delta": hashlib.sha256(),
                "delta_max_abs": 0.0,
                "delta_sum_abs": 0.0,
                "element_count": 0,
            }
            for key, shape in ACTIVE_PSG_BLOCK_SHAPES.items()
        },
    }


def _little_endian_float32_bytes(value: torch.Tensor) -> bytes:
    array = np.asarray(value.detach().cpu().numpy(), dtype=np.dtype("<f4"))
    return np.ascontiguousarray(array).tobytes(order="C")


def _update_signed_scene_audit(state: MutableMapping[str, object],
                               scene: torch.Tensor) -> None:
    raw = scene.detach().float().cpu().contiguous()
    require(raw.ndim == 4 and tuple(raw.shape[1:]) == (17, 96, 32),
            "E_SIGN_AUDIT_SHAPE", str(tuple(raw.shape)))
    require(bool(torch.isfinite(raw).all()), "E_SCENE_NONFINITE", "signed audit")
    negative = raw < 0
    state["raw_min"] = min(float(state["raw_min"]), float(raw.min().item()))
    state["negative_element_count"] = (
        int(state["negative_element_count"]) + int(negative.sum().item()))
    state["negative_sample_count"] = (
        int(state["negative_sample_count"])
        + int(negative.flatten(1).any(1).sum().item()))
    sample_channels = negative.flatten(2).any(2)
    state["negative_sample_channel_count"] = (
        int(state["negative_sample_channel_count"])
        + int(sample_channels.sum().item()))
    channel_union = negative.permute(1, 0, 2, 3).flatten(1).any(1).nonzero(
        as_tuple=False).flatten().tolist()
    state["negative_channel_indices_0based"].update(int(value) for value in channel_union)
    for sample in range(raw.shape[0]):
        state["negative_absolute_mass"] = (
            float(state["negative_absolute_mass"])
            + float((-raw[sample].clamp_max(0)).double().sum().item()))
    state["sample_count"] = int(state["sample_count"]) + int(raw.shape[0])


def _update_actual_space_audit(state: MutableMapping[str, object],
                               scene: torch.Tensor) -> None:
    raw = scene.detach().to(dtype=torch.float32).contiguous()
    require(raw.ndim == 4 and tuple(raw.shape[1:]) == (17, 96, 32),
            "E_SIGN_AUDIT_SHAPE", str(tuple(raw.shape)))
    require(bool(torch.isfinite(raw).all()), "E_SCENE_NONFINITE", "actual-space audit")
    positive = raw.clamp_min(0)
    for key, shape in ACTIVE_PSG_BLOCK_SHAPES.items():
        block = state["blocks"][key]
        sraw = actual_psg_input(raw, shape)
        spos = actual_psg_input(positive, shape)
        delta = sraw - spos
        for sample in range(raw.shape[0]):
            block["sraw"].update(_little_endian_float32_bytes(sraw[sample]))
            block["spos"].update(_little_endian_float32_bytes(spos[sample]))
            block["delta"].update(_little_endian_float32_bytes(delta[sample]))
            delta_abs = delta[sample].double().abs()
            block["delta_max_abs"] = max(
                float(block["delta_max_abs"]), float(delta_abs.max().item()))
            block["delta_sum_abs"] = (
                float(block["delta_sum_abs"]) + float(delta_abs.sum().item()))
        block["element_count"] = int(block["element_count"]) + int(delta.numel())
    state["sample_count"] = int(state["sample_count"]) + int(raw.shape[0])


def _finalize_signed_scene_audit(state: Mapping[str, object], split: str,
                                 count: int) -> Dict[str, object]:
    require(int(state["sample_count"]) == count and count > 0,
            "E_SIGN_AUDIT_COUNT", f"{split}: {state['sample_count']}!={count}")
    return {
        "transform": "positive_part_v1",
        "sample_order": "dataset_index_0_to_N_minus_1",
        "raw_shape": [count, 17, 96, 32],
        "raw_dtype": "<f4",
        "raw_element_count": count * 17 * 96 * 32,
        "raw_min": float(state["raw_min"]),
        "negative_element_count": int(state["negative_element_count"]),
        "negative_sample_count": int(state["negative_sample_count"]),
        "negative_sample_channel_count": int(state["negative_sample_channel_count"]),
        "negative_channel_indices_0based": sorted(
            int(value) for value in state["negative_channel_indices_0based"]),
        "negative_absolute_mass": float(state["negative_absolute_mass"]),
    }


def _finalize_actual_space_audit(state: Mapping[str, object], split: str,
                                 count: int, device: torch.device) -> Dict[str, object]:
    require(int(state["sample_count"]) == count and count > 0,
            "E_SIGN_AUDIT_COUNT", f"{split}: {state['sample_count']}!={count}")
    blocks: Dict[str, object] = {}
    for key, shape in ACTIVE_PSG_BLOCK_SHAPES.items():
        source = state["blocks"][key]
        element_count = int(source["element_count"])
        require(element_count == count * 17 * shape[0] * shape[1],
                "E_SIGN_AUDIT_COUNT", f"{split}/{key}: {element_count}")
        blocks[key] = {
            "shape": [count, 17, shape[0], shape[1]],
            "dtype": "<f4",
            "element_count": element_count,
            "sraw_sha256": source["sraw"].hexdigest(),
            "spos_sha256": source["spos"].hexdigest(),
            "delta_sha256": source["delta"].hexdigest(),
            "delta_max_abs": float(source["delta_max_abs"]),
            "delta_sum_abs": float(source["delta_sum_abs"]),
            "delta_mean_abs": float(source["delta_sum_abs"]) / element_count,
        }
    by_shape: Dict[Tuple[int, int], Mapping[str, object]] = {}
    for key, shape in ACTIVE_PSG_BLOCK_SHAPES.items():
        if shape in by_shape:
            require(blocks[key] == by_shape[shape], "E_SIGN_BLOCK_DRIFT", f"{split}/{key}")
        else:
            by_shape[shape] = blocks[key]
    return {
        "compute_device": str(device),
        "compute_backend": "torch_cuda" if device.type == "cuda" else "torch_cpu_test_only",
        "operator": (
            "actual_psg_input_v1=F.interpolate(mode=bilinear,align_corners=False,"
            "float32)+torch.sigmoid"
        ),
        "sample_order": "dataset_index_0_to_N_minus_1",
        "active_psg_blocks": blocks,
    }


def build_actual_space_audit(prepared: Path, split: str, count: int,
                             device: torch.device, batch_size: int) -> Dict[str, object]:
    require(device.type == "cuda" and torch.cuda.is_available(),
            "E_SIGN_AUDIT_DEVICE", str(device))
    require(batch_size > 0, "E_SIGN_AUDIT_BATCH", str(batch_size))
    source = np.load(prepared / f"{split}_scene_heatmaps.npy", mmap_mode="r")
    require(source.shape == (count, 17, 96, 32) and source.dtype == np.float32,
            "E_SIGN_AUDIT_SHAPE", f"{split}: {source.shape}/{source.dtype}")
    state = _new_actual_space_audit()
    try:
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            host = np.array(source[start:stop], dtype=np.float32, copy=True, order="C")
            raw = torch.from_numpy(host).to(device, non_blocking=False)
            _update_actual_space_audit(state, raw)
            del raw, host
        return _finalize_actual_space_audit(state, split, count, device)
    finally:
        _close_memmap(source)


def cache_split(
    split: str,
    dataset: PoseImageDataset,
    local_cfg,
    destination: Path,
) -> Dict[str, object]:
    count = len(dataset)
    heatmap_path = destination / f"{split}_scene_heatmaps.npy"
    score_path = destination / f"{split}_scene_scores.npy"
    heatmaps = _new_memmap(heatmap_path, (count, 17, 96, 32), "float32")
    scores = _new_memmap(score_path, (count, 17), "float32")
    metadata: List[Dict[str, object]] = []
    continuous = _new_memmap(destination / f"{split}_continuous.npy", (count, 95), "float64")
    file_hash_cache: Dict[str, str] = {}
    pose_identity_registry: Dict[str, Tuple[int, ...]] = {}
    cursor = 0
    expected_records = list(dataset.dataset)
    basenames = [Path(record[0]).name for record in expected_records]
    require(len(set(basenames)) == len(basenames), "E_RGB_BASENAME_COLLISION", split)
    signed_scene_audit = _new_signed_scene_audit()

    for batch in split_loader(dataset, local_cfg):
        _images, pids, camids, _camids_tensor, viewids, paths, pose_dict = batch
        scene, scene_scores, _target, _difference = PoseBackboneModel._prepare_pose(pose_dict)
        batch_size = scene.shape[0]
        require(tuple(scene.shape[1:]) == (17, 96, 32), "E_CACHE_SCENE_SHAPE", str(scene.shape))
        heatmaps[cursor:cursor + batch_size] = scene.numpy().astype(np.float32, copy=False)
        scores[cursor:cursor + batch_size] = scene_scores.numpy().astype(np.float32, copy=False)
        _update_signed_scene_audit(signed_scene_audit, scene)
        for offset in range(batch_size):
            row = cursor + offset
            expected_path, expected_pid, expected_camid, expected_viewid = expected_records[row]
            actual_path = str(Path(paths[offset]).resolve())
            require(actual_path == str(Path(expected_path).resolve()), "E_LOADER_PATH_ORDER", actual_path)
            require(int(pids[offset]) == int(expected_pid), "E_LOADER_PID_ORDER", actual_path)
            require(int(camids[offset]) == int(expected_camid), "E_LOADER_CAM_ORDER", actual_path)
            require(int(viewids[offset]) == int(expected_viewid), "E_LOADER_VIEW_ORDER", actual_path)
            rgb_sha = sha256_file(Path(actual_path))
            pose_manifest = _pose_asset_manifest_v2(
                dataset, actual_path, file_hash_cache, pose_identity_registry)
            try:
                vector, frame, report = summarize_scene(scene[offset], scene_scores[offset])
            except GateProtocolError as error:
                if split != "train" or error.code != "E_MATCH_EMPTY_SUPPORT":
                    raise
                vector = tuple([0.0] * 95)
                frame = 0
                report = {
                    "total_L1": 0.0,
                    "mean_confidence": float(scene_scores[offset].mean().item()),
                    "visible_joint_count": 0.0,
                    "scene_entropy": 0.0,
                }
            continuous[row] = np.asarray(vector, dtype=np.float64)
            person_count = int(pose_dict["num_persons"][offset])
            require(person_count == len(pose_manifest["effective_pose_person_paths"]),
                    "E_METADATA_SCHEMA_V2", f"{split}/{row}/person_count")
            metadata.append({
                "schema": SCENE_METADATA_SCHEMA,
                "index": row,
                "split": split,
                "path": actual_path,
                "rgb_sha256": rgb_sha,
                "pose_path_sha256": pose_manifest["pose_path_sha256"],
                "pose_content_sha256": pose_manifest["pose_content_sha256"],
                "pid": int(pids[offset]),
                "camid": int(camids[offset]),
                "viewid": int(viewids[offset]),
                "person_count": person_count,
                "frame": frame,
                "report": report,
                "source_pid": pose_manifest["source_pid"],
                "source_camid": pose_manifest["source_camid"],
                "source_frame_id": pose_manifest["source_frame_id"],
                "target_person_idx": pose_manifest["target_person_idx"],
                "full_pose_person_relpaths": list(
                    pose_manifest["full_pose_person_relpaths"]),
                "full_pose_person_paths": list(
                    pose_manifest["full_pose_person_paths"]),
                "full_pose_person_sha256": list(
                    pose_manifest["full_pose_person_sha256"]),
                "effective_pose_person_relpaths": list(
                    pose_manifest["effective_pose_person_relpaths"]),
                "effective_pose_person_paths": list(
                    pose_manifest["effective_pose_person_paths"]),
                "effective_pose_person_sha256": list(
                    pose_manifest["effective_pose_person_sha256"]),
            })
        cursor += batch_size
    require(cursor == count, "E_CACHE_COUNT", f"{split}: {cursor}!={count}")
    heatmaps.flush()
    scores.flush()
    continuous.flush()
    _close_memmap(heatmaps)
    _close_memmap(scores)
    _close_memmap(continuous)
    _recheck_identities(pose_identity_registry)
    atomic_write_json(destination / f"{split}_metadata.json", metadata)
    return {
        "count": count,
        "heatmaps": heatmap_path.name,
        "scores": score_path.name,
        "continuous": f"{split}_continuous.npy",
        "metadata": f"{split}_metadata.json",
        "signed_raw_audit": _finalize_signed_scene_audit(
            signed_scene_audit, split, count),
    }


def _scene_records_from_payload(metadata: object, continuous: np.ndarray,
                                split: str) -> List[SceneRecord]:
    require(isinstance(metadata, list) and metadata,
            "E_METADATA_SCHEMA_V2", f"{split}/root")
    require(isinstance(continuous, np.ndarray),
            "E_CONTINUOUS_CACHE_V2", f"{split}/root")
    require(continuous.dtype.str == "<f8" and continuous.shape == (len(metadata), 95),
            "E_CONTINUOUS_CACHE_V2", f"{split}: {continuous.dtype.str}/{continuous.shape}")
    require(bool(np.isfinite(continuous).all()), "E_CONTINUOUS_CACHE_V2", f"{split}/nonfinite")
    required = {
        "schema", "index", "split", "path", "rgb_sha256", "pose_path_sha256",
        "pose_content_sha256", "pid", "camid", "viewid", "person_count", "frame",
        "report", "source_pid", "source_camid", "source_frame_id",
        "target_person_idx", "full_pose_person_relpaths", "full_pose_person_paths",
        "full_pose_person_sha256", "effective_pose_person_relpaths",
        "effective_pose_person_paths", "effective_pose_person_sha256",
    }
    records: List[SceneRecord] = []
    for index, row in enumerate(metadata):
        require(isinstance(row, Mapping) and set(row) == required,
                "E_METADATA_SCHEMA_V2", f"{split}/{index}/keys")
        integer_fields = (
            "index", "pid", "camid", "viewid", "person_count", "frame",
            "source_pid", "source_camid", "source_frame_id", "target_person_idx",
        )
        string_fields = (
            "schema", "split", "path", "rgb_sha256",
            "pose_path_sha256", "pose_content_sha256",
        )
        tuple_fields = (
            "full_pose_person_relpaths", "full_pose_person_paths",
            "full_pose_person_sha256", "effective_pose_person_relpaths",
            "effective_pose_person_paths", "effective_pose_person_sha256",
        )
        require(
            all(type(row[field]) is int for field in integer_fields)
            and all(isinstance(row[field], str) and row[field] for field in string_fields)
            and all(isinstance(row[field], list) for field in tuple_fields)
            and isinstance(row["report"], Mapping),
            "E_METADATA_SCHEMA_V2", f"{split}/{index}/types",
        )
        require(row["schema"] == SCENE_METADATA_SCHEMA and row["split"] == split
                and row["index"] == index,
                "E_METADATA_SCHEMA_V2", f"{split}/{index}/identity")
        try:
            canonical_json_bytes(dict(row["report"]))
        except (TypeError, ValueError) as error:
            raise GateProtocolError(
                "E_METADATA_SCHEMA_V2", f"{split}/{index}/report: {error}") from error
        full_relpaths = tuple(row["full_pose_person_relpaths"])
        full_paths = tuple(row["full_pose_person_paths"])
        full_sha256 = tuple(row["full_pose_person_sha256"])
        effective_relpaths = tuple(row["effective_pose_person_relpaths"])
        effective_paths = tuple(row["effective_pose_person_paths"])
        effective_sha256 = tuple(row["effective_pose_person_sha256"])
        require(
            len(full_relpaths) == len(full_paths) == len(full_sha256) > 0
            and len(effective_relpaths) == len(effective_paths) == len(effective_sha256) > 0
            and int(row["person_count"]) == len(effective_paths)
            and 0 <= int(row["target_person_idx"]) < len(full_paths)
            and all(isinstance(value, str) and value for values in (
                full_relpaths, full_paths, full_sha256,
                effective_relpaths, effective_paths, effective_sha256,
            ) for value in values),
            "E_METADATA_SCHEMA_V2", f"{split}/{index}/constituents",
        )
        records.append(SceneRecord(
            metadata_schema=str(row["schema"]),
            index=index,
            split=split,
            path=str(row["path"]),
            rgb_sha256=str(row["rgb_sha256"]),
            pose_path_sha256=str(row["pose_path_sha256"]),
            pose_content_sha256=str(row["pose_content_sha256"]),
            pid=int(row["pid"]),
            camid=int(row["camid"]),
            viewid=int(row["viewid"]),
            person_count=int(row["person_count"]),
            continuous=tuple(float(value) for value in continuous[index]),
            frame=int(row["frame"]),
            report=dict(row["report"]),
            source_pid=int(row["source_pid"]),
            source_camid=int(row["source_camid"]),
            source_frame_id=int(row["source_frame_id"]),
            target_person_idx=int(row["target_person_idx"]),
            full_pose_person_relpaths=full_relpaths,
            full_pose_person_paths=full_paths,
            full_pose_person_sha256=full_sha256,
            effective_pose_person_relpaths=effective_relpaths,
            effective_pose_person_paths=effective_paths,
            effective_pose_person_sha256=effective_sha256,
        ))
    return records


def load_scene_records(prepared: Path, split: str) -> List[SceneRecord]:
    identity_registry: Dict[str, Tuple[int, ...]] = {}
    metadata_path = Path(prepared) / f"{split}_metadata.json"
    metadata, _metadata_report = _stable_json(
        metadata_path, identity_registry, "E_METADATA_SCHEMA_V2")
    continuous_path = Path(prepared) / f"{split}_continuous.npy"
    _continuous_report, continuous_identity, continuous_raw = (
        _stable_regular_file(continuous_path, return_bytes=True))
    _register_identity(identity_registry, continuous_path, continuous_identity)
    try:
        continuous = np.load(io.BytesIO(continuous_raw), allow_pickle=False)
    except (EOFError, OSError, ValueError) as error:
        raise GateProtocolError(
            "E_CONTINUOUS_CACHE_V2",
            f"{split}: {error.__class__.__name__}") from error
    try:
        records = _scene_records_from_payload(metadata, continuous, split)
        _recheck_identities(identity_registry)
        return records
    finally:
        if hasattr(continuous, "close"):
            continuous.close()
        else:
            _close_memmap(continuous)


def array_sha256_v1(value: np.ndarray, expected_dtype: str,
                    expected_shape: Sequence[int]) -> str:
    array = np.asarray(value)
    require(array.dtype.str == expected_dtype,
            "E_RELATION_ARRAY_DTYPE", f"{array.dtype.str}!={expected_dtype}")
    require(list(array.shape) == [int(item) for item in expected_shape],
            "E_RELATION_ARRAY_SHAPE", str(array.shape))
    require(bool(np.isfinite(array).all()), "E_RELATION_ARRAY_NONFINITE", "")
    normalized = np.ascontiguousarray(array, dtype=np.dtype(expected_dtype))
    header = canonical_json_bytes({
        "schema": "array_sha256_v1",
        "dtype": np.dtype(expected_dtype).str,
        "shape": [int(item) for item in expected_shape],
        "order": "C",
    })
    return sha256_bytes(header + normalized.tobytes(order="C"))


def _canonical_summary(payload: object) -> Dict[str, object]:
    canonical = canonical_json_bytes(payload)
    return {
        "count": len(payload),
        "canonical_bytes": len(canonical),
        "sha256": sha256_bytes(canonical),
    }


def _duplicate_count(values: Sequence[object]) -> int:
    return len(values) - len(set(values))


def _flatten_record_values(records: Sequence[SceneRecord], field: str) -> List[str]:
    output: List[str] = []
    for record in records:
        output.extend(str(value) for value in getattr(record, field))
    return output


def _close_memmap(value: np.ndarray) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _official_lists_v2(
    dataset: OccludedDukeMTMC,
    identity_registry: MutableMapping[str, Tuple[int, ...]],
) -> Tuple[Dict[str, List[str]], Dict[str, object]]:
    configured_paths = {
        "train": _exact_lexical_child(
            dataset.train_list, dataset.dataset_dir, ("train.list",),
            "E_OFFICIAL_LIST_PATH"),
        "query": _exact_lexical_child(
            dataset.query_list, dataset.dataset_dir, ("query.list",),
            "E_OFFICIAL_LIST_PATH"),
        "gallery": _exact_lexical_child(
            dataset.gallery_list, dataset.dataset_dir, ("gallery.list",),
            "E_OFFICIAL_LIST_PATH"),
    }
    lexical_root = _lexical_absolute(dataset.dataset_dir)
    root = lexical_root.resolve()
    output: Dict[str, List[str]] = {}
    report: Dict[str, object] = {}
    for split, configured_path in configured_paths.items():
        expected = OFFICIAL_SPLITS[split]
        expected_path = lexical_root / str(expected["list"])
        require(configured_path == expected_path,
                "E_OFFICIAL_LIST_PATH", f"{configured_path}!={expected_path}")
        try:
            configured_stat = configured_path.lstat()
        except OSError as error:
            raise GateProtocolError(
                "E_OFFICIAL_LIST_PATH", f"{configured_path}: {error}") from error
        require(stat.S_ISREG(configured_stat.st_mode),
                "E_OFFICIAL_LIST_PATH", str(configured_path))
        path = configured_path.resolve()
        _require_direct_child(path, root, "E_OFFICIAL_LIST_PATH")
        require(path == root / str(expected["list"]),
                "E_OFFICIAL_LIST_PATH", f"{path}!={root / str(expected['list'])}")
        raw_report, identity, raw = _stable_regular_file(path, return_bytes=True)
        _register_identity(identity_registry, path, identity)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise GateProtocolError("E_OFFICIAL_LIST_UTF8", str(path)) from error
        names = [line.strip() for line in text.splitlines() if line.strip()]
        require(len(names) == len(set(names)), "E_OFFICIAL_LIST_DUPLICATE", split)
        require(all(
            value == Path(value).name and value not in {"", ".", ".."}
            and "/" not in value and "\\" not in value
            and not Path(value).is_absolute()
            and FILENAME_PATTERN.fullmatch(value) is not None
            for value in names
        ), "E_OFFICIAL_LIST_ENTRY", split)
        ordered = sorted(names)
        canonical = canonical_json_bytes(ordered)
        output[split] = ordered
        report[split] = {
            "rgb_root": str(expected["root"]),
            "list": str(expected["list"]),
            "count": len(ordered),
            "raw_bytes": int(raw_report["bytes"]),
            "raw_sha256": str(raw_report["sha256"]),
            "canonical_bytes": len(canonical),
            "canonical_sha256": sha256_bytes(canonical),
            "pose_index_bytes": 0,
            "pose_index_sha256": "",
        }
    return output, report


def _record_metadata_projection(record: SceneRecord) -> Dict[str, object]:
    return {
        "metadata_schema": record.metadata_schema,
        "index": record.index,
        "split": record.split,
        "path": record.path,
        "rgb_sha256": record.rgb_sha256,
        "pose_path_sha256": record.pose_path_sha256,
        "pose_content_sha256": record.pose_content_sha256,
        "pid": record.pid,
        "camid": record.camid,
        "viewid": record.viewid,
        "person_count": record.person_count,
        "continuous": list(record.continuous),
        "frame": record.frame,
        "report": dict(record.report),
        "source_pid": record.source_pid,
        "source_camid": record.source_camid,
        "source_frame_id": record.source_frame_id,
        "target_person_idx": record.target_person_idx,
        "full_pose_person_relpaths": list(record.full_pose_person_relpaths),
        "full_pose_person_paths": list(record.full_pose_person_paths),
        "full_pose_person_sha256": list(record.full_pose_person_sha256),
        "effective_pose_person_relpaths": list(record.effective_pose_person_relpaths),
        "effective_pose_person_paths": list(record.effective_pose_person_paths),
        "effective_pose_person_sha256": list(record.effective_pose_person_sha256),
    }


def _record_set_summary(records: Sequence[SceneRecord]) -> Dict[str, object]:
    expected = canonical_scene_record_set_summary(records)
    local = _canonical_summary([_record_metadata_projection(record) for record in records])
    require(local == expected, "E_RELATION_RECORD_PROJECTION", str(local))
    return expected


def _joint_metadata_projection(pairs: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [{
        "basename": row["basename"], "rgb_sha256": row["rgb_sha256"],
        "pose_content_sha256": row["pose_content_sha256"],
        "query_index": row["query_index"], "gallery_index": row["gallery_index"],
        "pid": row["pid"], "camid": row["camid"], "viewid": row["viewid"],
        "person_count": row["person_count"], "frame": row["frame"],
    } for row in pairs]


def _require_exact_mapping(
    value: object,
    keys: Sequence[str],
    code: str,
    label: str,
) -> Mapping[str, object]:
    require(isinstance(value, Mapping) and set(value) == set(keys),
            code, label)
    return value


def _require_summary_schema(value: object, code: str, label: str) -> Mapping[str, object]:
    summary = _require_exact_mapping(value, SUMMARY_KEYS, code, label)
    require(type(summary["count"]) is int and int(summary["count"]) >= 0
            and type(summary["canonical_bytes"]) is int
            and int(summary["canonical_bytes"]) >= 0
            and isinstance(summary["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", str(summary["sha256"])) is not None,
            code, label)
    return summary


def _validate_relation_record_projection(record: SceneRecord) -> None:
    """Validate the full/effective pose projection used by the active loader."""

    canonical_scene_record_set_summary([record])
    full_relpaths = record.full_pose_person_relpaths
    full_paths = record.full_pose_person_paths
    full_content = record.full_pose_person_sha256
    effective_relpaths = record.effective_pose_person_relpaths
    effective_paths = record.effective_pose_person_paths
    effective_content = record.effective_pose_person_sha256
    require(len(full_relpaths) == len(full_paths) == len(full_content) > 0,
            "E_RELATION_CONSTITUENT_PROJECTION", record.path)
    require(0 <= record.target_person_idx < len(full_paths),
            "E_RELATION_EFFECTIVE_PROJECTION", record.path)
    expected_indices = list(range(min(len(full_paths), 6)))
    if 0 < record.target_person_idx < len(expected_indices):
        target = expected_indices.pop(record.target_person_idx)
        expected_indices.insert(0, target)
    require(record.person_count == len(expected_indices),
            "E_RELATION_EFFECTIVE_PROJECTION", record.path)
    require(
        effective_relpaths == tuple(full_relpaths[index] for index in expected_indices)
        and effective_paths == tuple(full_paths[index] for index in expected_indices)
        and effective_content == tuple(full_content[index] for index in expected_indices),
        "E_RELATION_EFFECTIVE_PROJECTION", record.path,
    )
    for relpath, path in zip(full_relpaths, full_paths):
        basename = Path(path).name
        require(Path(path).is_absolute() and relpath ==
                f"pose_data/{record.split}/{basename}",
                "E_RELATION_CONSTITUENT_PROJECTION", record.path)
    require(
        record.pose_path_sha256 == sha256_bytes(canonical_json_bytes(list(full_paths)))
        and record.pose_content_sha256 ==
        sha256_bytes(canonical_json_bytes(list(full_content))),
        "E_RELATION_BUNDLE_PROJECTION", record.path,
    )


def audit_split_relations_v2(
    records: Mapping[str, Sequence[SceneRecord]],
    official_lists: Mapping[str, Sequence[str]],
    official_report: Mapping[str, object],
    cache_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    """Dataset-agnostic structural relation audit with strict fail-closed aliases."""

    split_names = ("train", "query", "gallery")
    _require_exact_mapping(records, split_names, "E_RELATION_INPUT_SCHEMA", "records")
    _require_exact_mapping(
        official_lists, split_names, "E_RELATION_INPUT_SCHEMA", "official_lists")
    official_rows = _require_exact_mapping(
        official_report, split_names, "E_RELATION_INPUT_SCHEMA", "official_report")
    for split in split_names:
        require(isinstance(records[split], Sequence)
                and not isinstance(records[split], (str, bytes)),
                "E_RELATION_INPUT_SCHEMA", f"records/{split}")
        require(all(isinstance(record, SceneRecord) for record in records[split]),
                "E_RELATION_INPUT_SCHEMA", f"records/{split}/items")
        official_row = _require_exact_mapping(
            official_rows[split], OFFICIAL_LIST_ROW_KEYS,
            "E_RELATION_INPUT_SCHEMA", f"official_report/{split}")
        require(
            all(type(official_row[key]) is int and int(official_row[key]) >= 0
                for key in ("count", "raw_bytes", "canonical_bytes",
                            "pose_index_bytes"))
            and all(isinstance(official_row[key], str) and official_row[key]
                    for key in ("rgb_root", "list", "raw_sha256",
                                "canonical_sha256", "pose_index_sha256"))
            and all(re.fullmatch(r"[0-9a-f]{64}", str(official_row[key])) is not None
                    for key in ("raw_sha256", "canonical_sha256",
                                "pose_index_sha256")),
            "E_RELATION_INPUT_SCHEMA", f"official_report/{split}/types",
        )
        require(isinstance(official_lists[split], Sequence)
                and not isinstance(official_lists[split], (str, bytes)),
                "E_RELATION_INPUT_SCHEMA", f"official_lists/{split}")
        require(all(isinstance(value, str) and value
                    for value in official_lists[split]),
                "E_RELATION_INPUT_SCHEMA", f"official_lists/{split}/items")
        require(int(official_row["count"]) == len(records[split]) ==
                len(official_lists[split]),
                "E_RELATION_INPUT_SCHEMA", f"official_report/{split}/count")
    cache_root = _require_exact_mapping(
        cache_arrays, ("heatmaps", "scores", "nuisance"),
        "E_RELATION_INPUT_SCHEMA", "cache_arrays")
    expected_cache = {
        "heatmaps": ("<f4", (17, 96, 32)),
        "scores": ("<f4", (17,)),
        "nuisance": ("<f8", (95,)),
    }
    for name, (dtype, trailing_shape) in expected_cache.items():
        by_split = _require_exact_mapping(
            cache_root[name], split_names,
            "E_RELATION_INPUT_SCHEMA", f"cache_arrays/{name}")
        for split in split_names:
            try:
                value = np.asarray(by_split[split])
            except (TypeError, ValueError) as error:
                raise GateProtocolError(
                    "E_RELATION_INPUT_SCHEMA",
                    f"cache_arrays/{name}/{split}: {error}") from error
            require(value.dtype.str == dtype,
                    "E_RELATION_CACHE_DTYPE",
                    f"{name}/{split}/{value.dtype.str}!={dtype}")
            require(value.shape == (len(records[split]), *trailing_shape),
                    "E_RELATION_CACHE_SHAPE",
                    f"{name}/{split}/{value.shape}")
    within: Dict[str, object] = {}
    for split in split_names:
        values = list(records[split])
        require([record.index for record in values] == list(range(len(values))),
                "E_RELATION_RECORD_ORDER", split)
        require(all(record.split == split for record in values),
                "E_RELATION_RECORD_SPLIT", split)
        for record in values:
            _validate_relation_record_projection(record)
        require([Path(record.path).name for record in values] == list(official_lists[split]),
                "E_RELATION_OFFICIAL_ORDER", split)
        row: Dict[str, object] = {}
        for key in ("path", "rgb_sha256", "pose_path_sha256", "pose_content_sha256"):
            row[f"{key}_duplicate_count"] = _duplicate_count(
                [getattr(record, key) for record in values])
        for scope in ("full", "effective"):
            row[f"{scope}_pose_person_path_duplicate_count"] = _duplicate_count(
                _flatten_record_values(values, f"{scope}_pose_person_paths"))
            row[f"{scope}_pose_person_content_duplicate_count"] = _duplicate_count(
                _flatten_record_values(values, f"{scope}_pose_person_sha256"))
        row["source_pid_count"] = len({record.source_pid for record in values})
        row["target_outside_effective_count"] = sum(
            int(record.target_person_idx >= min(
                len(record.full_pose_person_paths), 6))
            for record in values
        )
        duplicate_codes = {
            "path_duplicate_count": "E_RELATION_WITHIN_PATH_DUPLICATE",
            "rgb_sha256_duplicate_count": "E_RELATION_WITHIN_RGB_CONTENT_DUPLICATE",
            "pose_path_sha256_duplicate_count": "E_RELATION_WITHIN_POSE_PATH_DUPLICATE",
            "pose_content_sha256_duplicate_count":
                "E_RELATION_WITHIN_POSE_CONTENT_DUPLICATE",
            "effective_pose_person_path_duplicate_count":
                "E_RELATION_WITHIN_EFFECTIVE_CONSTITUENT_PATH_DUPLICATE",
            "effective_pose_person_content_duplicate_count":
                "E_RELATION_WITHIN_EFFECTIVE_CONSTITUENT_CONTENT_DUPLICATE",
            "full_pose_person_path_duplicate_count":
                "E_RELATION_WITHIN_FULL_CONSTITUENT_PATH_DUPLICATE",
            "full_pose_person_content_duplicate_count":
                "E_RELATION_WITHIN_FULL_CONSTITUENT_CONTENT_DUPLICATE",
        }
        for key, code in duplicate_codes.items():
            require(int(row[key]) == 0, code, f"{split}/{key}")
        within[split] = row

    cross: Dict[str, object] = {}
    cross_pairs = (
        ("train", "query", "train_query"),
        ("train", "gallery", "train_gallery"),
        ("query", "gallery", "query_gallery"),
    )
    for left, right, label in cross_pairs:
        left_records = list(records[left])
        right_records = list(records[right])
        row = {}
        for key in ("path", "rgb_sha256", "pose_path_sha256", "pose_content_sha256"):
            row[f"{key}_overlap_count"] = len(
                {getattr(record, key) for record in left_records}
                & {getattr(record, key) for record in right_records})
        for scope in ("full", "effective"):
            row[f"{scope}_pose_person_path_overlap_count"] = len(
                set(_flatten_record_values(left_records, f"{scope}_pose_person_paths"))
                & set(_flatten_record_values(right_records, f"{scope}_pose_person_paths")))
            row[f"{scope}_pose_person_content_overlap_count"] = len(
                set(_flatten_record_values(left_records, f"{scope}_pose_person_sha256"))
                & set(_flatten_record_values(right_records, f"{scope}_pose_person_sha256")))
        row["source_pid_overlap_count"] = len(
            {record.source_pid for record in left_records}
            & {record.source_pid for record in right_records})
        row["rgb_content_forbidden_group_count"] = int(row["rgb_sha256_overlap_count"])
        row["pose_content_forbidden_group_count"] = int(
            row["pose_content_sha256_overlap_count"])
        row["full_pose_person_content_forbidden_count"] = int(
            row["full_pose_person_content_overlap_count"])
        row["effective_pose_person_content_forbidden_count"] = int(
            row["effective_pose_person_content_overlap_count"])
        row["forbidden_overlap_count"] = sum(int(value) for key, value in row.items()
                                             if key != "source_pid_overlap_count")
        cross[label] = row

    train_eval_alias_codes = {
        "path_overlap_count": "E_RELATION_TRAIN_EVAL_PATH_ALIAS",
        "rgb_sha256_overlap_count": "E_RELATION_TRAIN_EVAL_RGB_CONTENT_ALIAS",
        "pose_path_sha256_overlap_count": "E_RELATION_TRAIN_EVAL_POSE_PATH_ALIAS",
        "pose_content_sha256_overlap_count":
            "E_RELATION_TRAIN_EVAL_POSE_CONTENT_ALIAS",
        "effective_pose_person_path_overlap_count":
            "E_RELATION_TRAIN_EVAL_EFFECTIVE_CONSTITUENT_PATH_ALIAS",
        "effective_pose_person_content_overlap_count":
            "E_RELATION_TRAIN_EVAL_EFFECTIVE_CONSTITUENT_CONTENT_ALIAS",
        "full_pose_person_path_overlap_count":
            "E_RELATION_TRAIN_EVAL_FULL_CONSTITUENT_PATH_ALIAS",
        "full_pose_person_content_overlap_count":
            "E_RELATION_TRAIN_EVAL_FULL_CONSTITUENT_CONTENT_ALIAS",
        "source_pid_overlap_count": "E_RELATION_TRAIN_EVAL_SOURCE_PID_ALIAS",
    }
    for label in ("train_query", "train_gallery"):
        for key, code in train_eval_alias_codes.items():
            require(int(cross[label][key]) == 0, code, f"{label}/{key}")
        require(int(cross[label]["forbidden_overlap_count"]) == 0,
                "E_RELATION_TRAIN_EVAL_AGGREGATE", label)

    query = list(records["query"])
    gallery = list(records["gallery"])
    query_by_rgb = {record.rgb_sha256: record for record in query}
    gallery_by_rgb = {record.rgb_sha256: record for record in gallery}
    query_by_pose = {record.pose_content_sha256: record for record in query}
    gallery_by_pose = {record.pose_content_sha256: record for record in gallery}
    require(len(query_by_rgb) == len(query) and len(gallery_by_rgb) == len(gallery),
            "E_RELATION_RGB_GROUP", "within split")
    require(len(query_by_pose) == len(query) and len(gallery_by_pose) == len(gallery),
            "E_RELATION_POSE_GROUP", "within split")
    shared_rgb = sorted(set(query_by_rgb) & set(gallery_by_rgb))
    shared_pose = sorted(set(query_by_pose) & set(gallery_by_pose))
    rgb_endpoints = sorted((query_by_rgb[value].index, gallery_by_rgb[value].index)
                           for value in shared_rgb)
    pose_endpoints = sorted((query_by_pose[value].index, gallery_by_pose[value].index)
                            for value in shared_pose)
    endpoint_equal = rgb_endpoints == pose_endpoints

    qg = cross["query_gallery"]
    for key, code in (
        ("path_overlap_count", "E_RELATION_QUERY_GALLERY_RGB_PATH_ALIAS"),
        ("pose_path_sha256_overlap_count",
         "E_RELATION_QUERY_GALLERY_POSE_PATH_ALIAS"),
        ("effective_pose_person_path_overlap_count",
         "E_RELATION_QUERY_GALLERY_EFFECTIVE_CONSTITUENT_PATH_ALIAS"),
        ("full_pose_person_path_overlap_count",
         "E_RELATION_QUERY_GALLERY_FULL_CONSTITUENT_PATH_ALIAS"),
    ):
        require(int(qg[key]) == 0, code, key)
    require(endpoint_equal, "E_RELATION_QUERY_GALLERY_ENDPOINT_MISMATCH",
            f"rgb={rgb_endpoints}/pose={pose_endpoints}")
    qg["rgb_content_forbidden_group_count"] = 0
    qg["pose_content_forbidden_group_count"] = 0
    qg["full_pose_person_content_forbidden_count"] = 0
    qg["effective_pose_person_content_forbidden_count"] = 0
    allowed_endpoint_set = set(rgb_endpoints) if endpoint_equal else set()
    for scope in ("effective", "full"):
        left_positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        right_positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for record in query:
            for position, digest in enumerate(
                    getattr(record, f"{scope}_pose_person_sha256")):
                left_positions[digest].append((record.index, position))
        for record in gallery:
            for position, digest in enumerate(
                    getattr(record, f"{scope}_pose_person_sha256")):
                right_positions[digest].append((record.index, position))
        forbidden = 0
        for digest in set(left_positions) & set(right_positions):
            left_values = left_positions[digest]
            right_values = right_positions[digest]
            valid = (
                len(left_values) == len(right_values) == 1
                and (left_values[0][0], right_values[0][0]) in allowed_endpoint_set
                and left_values[0][1] == right_values[0][1]
            )
            require(valid, (
                "E_RELATION_QUERY_GALLERY_FULL_CONSTITUENT_CONTENT_ALIAS"
                if scope == "full" else
                "E_RELATION_QUERY_GALLERY_EFFECTIVE_CONSTITUENT_CONTENT_ALIAS"
            ), f"{scope}/{digest}")
            forbidden += int(not valid)
        qg[f"{scope}_pose_person_content_forbidden_count"] = forbidden

    pairs: List[Dict[str, object]] = []
    junk_true = 0
    junk_false = 0
    forbidden_pair = 0
    heatmaps = cache_arrays["heatmaps"]
    scores = cache_arrays["scores"]
    nuisance = cache_arrays["nuisance"]
    official_shared_basenames = sorted(
        set(official_lists["query"]) & set(official_lists["gallery"]))
    for digest in shared_rgb:
        qrecord = query_by_rgb[digest]
        grecord = gallery_by_rgb[digest]
        basename = Path(qrecord.path).name
        require(
            basename == Path(grecord.path).name
            and basename in official_shared_basenames,
            "E_RELATION_PAIR_BASENAME", basename,
        )
        is_junk = qrecord.pid == grecord.pid and qrecord.camid == grecord.camid
        require(is_junk, "E_RELATION_PAIR_NOT_JUNK", basename)
        require(
            qrecord.pid == grecord.pid
            and qrecord.camid == grecord.camid
            and qrecord.viewid == grecord.viewid
            and qrecord.person_count == grecord.person_count
            and qrecord.frame == grecord.frame
            and qrecord.source_pid == grecord.source_pid == qrecord.pid
            and qrecord.source_camid == grecord.source_camid == qrecord.camid
            and qrecord.source_frame_id == grecord.source_frame_id
            and qrecord.target_person_idx == grecord.target_person_idx
            and qrecord.pose_content_sha256 == grecord.pose_content_sha256
            and qrecord.full_pose_person_sha256 == grecord.full_pose_person_sha256
            and qrecord.effective_pose_person_sha256 ==
                grecord.effective_pose_person_sha256
            and tuple(Path(value).name for value in qrecord.full_pose_person_relpaths) ==
                tuple(Path(value).name for value in grecord.full_pose_person_relpaths)
            and tuple(Path(value).name for value in
                      qrecord.effective_pose_person_relpaths) ==
                tuple(Path(value).name for value in
                      grecord.effective_pose_person_relpaths)
            and canonical_json_bytes(dict(qrecord.report)) ==
                canonical_json_bytes(dict(grecord.report)),
            "E_RELATION_PAIR_METADATA", basename,
        )
        valid = (
            endpoint_equal
            and basename == Path(grecord.path).name
            and basename in official_shared_basenames
            and qrecord.pid == grecord.pid
            and qrecord.camid == grecord.camid
            and qrecord.viewid == grecord.viewid
            and qrecord.person_count == grecord.person_count
            and qrecord.frame == grecord.frame
            and qrecord.source_pid == grecord.source_pid == qrecord.pid
            and qrecord.source_camid == grecord.source_camid == qrecord.camid
            and qrecord.source_frame_id == grecord.source_frame_id
            and qrecord.target_person_idx == grecord.target_person_idx
            and qrecord.pose_content_sha256 == grecord.pose_content_sha256
            and qrecord.full_pose_person_sha256 == grecord.full_pose_person_sha256
            and qrecord.effective_pose_person_sha256 == grecord.effective_pose_person_sha256
            and tuple(Path(value).name for value in qrecord.full_pose_person_relpaths) ==
                tuple(Path(value).name for value in grecord.full_pose_person_relpaths)
            and tuple(Path(value).name for value in qrecord.effective_pose_person_relpaths) ==
                tuple(Path(value).name for value in grecord.effective_pose_person_relpaths)
            and canonical_json_bytes(dict(qrecord.report)) ==
                canonical_json_bytes(dict(grecord.report))
        )
        junk_true += int(is_junk)
        junk_false += int(not is_junk)
        forbidden_pair += int(not valid or not is_junk)
        if not valid or not is_junk:
            continue
        qi, gi = qrecord.index, grecord.index
        q_hraw = array_sha256_v1(heatmaps["query"][qi], "<f4", [17, 96, 32])
        g_hraw = array_sha256_v1(heatmaps["gallery"][gi], "<f4", [17, 96, 32])
        q_score = array_sha256_v1(scores["query"][qi], "<f4", [17])
        g_score = array_sha256_v1(scores["gallery"][gi], "<f4", [17])
        q_nuisance = array_sha256_v1(nuisance["query"][qi], "<f8", [95])
        g_nuisance = array_sha256_v1(nuisance["gallery"][gi], "<f8", [95])
        require(q_hraw == g_hraw,
                "E_RELATION_CACHE_HRAW_MISMATCH", basename)
        require(q_score == g_score,
                "E_RELATION_CACHE_SCORE_MISMATCH", basename)
        require(q_nuisance == g_nuisance,
                "E_RELATION_CACHE_NUISANCE_MISMATCH", basename)
        cache_equal = q_hraw == g_hraw and q_score == g_score and q_nuisance == g_nuisance
        forbidden_pair += int(not cache_equal)
        if not cache_equal:
            continue
        pairs.append({
            "basename": basename,
            "camid": qrecord.camid,
            "effective_pose_person_sha256": list(qrecord.effective_pose_person_sha256),
            "frame": qrecord.frame,
            "full_pose_person_sha256": list(qrecord.full_pose_person_sha256),
            "gallery_effective_pose_person_relpaths": list(
                grecord.effective_pose_person_relpaths),
            "gallery_full_pose_person_relpaths": list(grecord.full_pose_person_relpaths),
            "gallery_index": gi,
            "gallery_pose_path_sha256": grecord.pose_path_sha256,
            "gallery_rgb_relpath": f"bounding_box_test/{basename}",
            "gallery_target_person_idx": grecord.target_person_idx,
            "hraw_sha256": q_hraw,
            "nuisance_sha256": q_nuisance,
            "person_count": qrecord.person_count,
            "pid": qrecord.pid,
            "pose_content_sha256": qrecord.pose_content_sha256,
            "query_effective_pose_person_relpaths": list(
                qrecord.effective_pose_person_relpaths),
            "query_full_pose_person_relpaths": list(qrecord.full_pose_person_relpaths),
            "query_index": qi,
            "query_pose_path_sha256": qrecord.pose_path_sha256,
            "query_rgb_relpath": f"query/{basename}",
            "query_target_person_idx": qrecord.target_person_idx,
            "report": dict(qrecord.report),
            "rgb_sha256": digest,
            "score_sha256": q_score,
            "source_camid": qrecord.source_camid,
            "source_frame_id": qrecord.source_frame_id,
            "source_pid": qrecord.source_pid,
            "viewid": qrecord.viewid,
        })
    pairs.sort(key=lambda row: (
        row["rgb_sha256"], row["basename"], row["query_index"], row["gallery_index"]))

    qg["rgb_content_forbidden_group_count"] += abs(len(shared_rgb) - len(pairs))
    qg["pose_content_forbidden_group_count"] += (
        abs(len(shared_pose) - len(pairs)) + int(not endpoint_equal))
    require(int(qg["rgb_content_forbidden_group_count"]) == 0,
            "E_RELATION_QUERY_GALLERY_RGB_CONTENT_MULTIPLICITY", "")
    require(int(qg["pose_content_forbidden_group_count"]) == 0,
            "E_RELATION_QUERY_GALLERY_POSE_CONTENT_MULTIPLICITY", "")
    require(int(qg["full_pose_person_content_forbidden_count"]) == 0,
            "E_RELATION_QUERY_GALLERY_FULL_CONSTITUENT_CONTENT_ALIAS", "")
    require(int(qg["effective_pose_person_content_forbidden_count"]) == 0,
            "E_RELATION_QUERY_GALLERY_EFFECTIVE_CONSTITUENT_CONTENT_ALIAS", "")
    require(forbidden_pair == 0,
            "E_RELATION_QUERY_GALLERY_FORBIDDEN_PAIR", str(forbidden_pair))
    qg["forbidden_overlap_count"] = sum(int(qg[key]) for key in (
        "path_overlap_count", "pose_path_sha256_overlap_count",
        "full_pose_person_path_overlap_count", "effective_pose_person_path_overlap_count",
        "rgb_content_forbidden_group_count", "pose_content_forbidden_group_count",
        "full_pose_person_content_forbidden_count",
        "effective_pose_person_content_forbidden_count",
    )) + forbidden_pair
    require(int(qg["forbidden_overlap_count"]) == 0,
            "E_RELATION_QUERY_GALLERY_ALIAS", str(qg))

    require([row["basename"] for row in sorted(pairs, key=lambda row: row["basename"])] ==
            official_shared_basenames,
            "E_RELATION_PAIR_BASENAME_PROJECTION", "basename")
    require([row["rgb_sha256"] for row in pairs] == shared_rgb,
            "E_RELATION_PAIR_RGB_PROJECTION", "rgb")
    metadata_projection = _joint_metadata_projection(pairs)
    legacy_rgb_payload = json.dumps(
        shared_rgb, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    relations = {
        "query_gallery_shared_basenames": _canonical_summary(official_shared_basenames),
        "query_gallery_shared_rgb_sha256_legacy": {
            "count": len(shared_rgb),
            "canonical_bytes": len(legacy_rgb_payload),
            "sha256": sha256_bytes(legacy_rgb_payload),
        },
        "query_gallery_shared_rgb_sha256": _canonical_summary(shared_rgb),
        "query_gallery_endpoint_pairs": {
            "equal": endpoint_equal,
            "rgb": _canonical_summary(rgb_endpoints),
            "pose": _canonical_summary(pose_endpoints),
        },
        "query_gallery_joint_metadata_pairs": _canonical_summary(metadata_projection),
        "query_gallery_joint_pairs": _canonical_summary(pairs),
        "split_record_sets": {
            split: _record_set_summary(records[split]) for split in split_names
        },
        "allowed_pair_count": len(pairs),
        "junk_true_count": junk_true,
        "junk_false_count": junk_false,
        "forbidden_pair_count": forbidden_pair,
    }
    report = {
        "schema": RELATION_REPORT_SCHEMA,
        "official_source": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "filename_regex": FILENAME_REGEX,
        },
        "official_lists": dict(official_report),
        "split_counts": {split: len(records[split]) for split in split_names},
        "within_split": within,
        "cross_split": cross,
        "relations": relations,
        "pairs": pairs,
    }
    report["relation_report_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def _validate_official_report_schema(report: object) -> Mapping[str, object]:
    report_map = _require_exact_mapping(
        report, RELATION_REPORT_KEYS, "E_OFFICIAL_REPORT_SCHEMA", "report")
    require(report_map.get("schema") == RELATION_REPORT_SCHEMA,
            "E_OFFICIAL_REPORT_SCHEMA", str(report_map.get("schema")))
    source = _require_exact_mapping(
        report_map["official_source"], ("repository", "commit", "filename_regex"),
        "E_OFFICIAL_REPORT_SCHEMA", "official_source")
    require(all(isinstance(source[key], str) and source[key]
                for key in source), "E_OFFICIAL_REPORT_SCHEMA", "official_source")

    split_names = ("train", "query", "gallery")
    split_counts = _require_exact_mapping(
        report_map["split_counts"], split_names,
        "E_OFFICIAL_REPORT_SCHEMA", "split_counts")
    require(all(type(split_counts[split]) is int and int(split_counts[split]) >= 0
                for split in split_names),
            "E_OFFICIAL_REPORT_SCHEMA", "split_counts")

    official_lists = _require_exact_mapping(
        report_map["official_lists"], split_names,
        "E_OFFICIAL_LIST_SCHEMA", "official_lists")
    list_integer_keys = (
        "count", "raw_bytes", "canonical_bytes", "pose_index_bytes")
    list_string_keys = (
        "rgb_root", "list", "raw_sha256", "canonical_sha256",
        "pose_index_sha256")
    for split in split_names:
        row = _require_exact_mapping(
            official_lists[split], OFFICIAL_LIST_ROW_KEYS,
            "E_OFFICIAL_LIST_SCHEMA", split)
        require(all(type(row[key]) is int and int(row[key]) >= 0
                    for key in list_integer_keys)
                and all(isinstance(row[key], str) and row[key]
                        for key in list_string_keys)
                and all(re.fullmatch(r"[0-9a-f]{64}", str(row[key])) is not None
                        for key in ("raw_sha256", "canonical_sha256",
                                    "pose_index_sha256")),
                "E_OFFICIAL_LIST_SCHEMA", split)

    within = _require_exact_mapping(
        report_map["within_split"], split_names,
        "E_OFFICIAL_WITHIN_SCHEMA", "within_split")
    for split in split_names:
        row = _require_exact_mapping(
            within[split], WITHIN_SPLIT_ROW_KEYS,
            "E_OFFICIAL_WITHIN_SCHEMA", split)
        require(all(type(row[key]) is int and int(row[key]) >= 0 for key in row),
                "E_OFFICIAL_WITHIN_SCHEMA", split)

    cross_labels = ("train_query", "train_gallery", "query_gallery")
    cross = _require_exact_mapping(
        report_map["cross_split"], cross_labels,
        "E_OFFICIAL_CROSS_SCHEMA", "cross_split")
    for label in cross_labels:
        row = _require_exact_mapping(
            cross[label], CROSS_SPLIT_ROW_KEYS,
            "E_OFFICIAL_CROSS_SCHEMA", label)
        require(all(type(row[key]) is int and int(row[key]) >= 0 for key in row),
                "E_OFFICIAL_CROSS_SCHEMA", label)

    relations = _require_exact_mapping(
        report_map["relations"], RELATIONS_KEYS,
        "E_OFFICIAL_RELATIONS_SCHEMA", "relations")
    for key in (
        "query_gallery_shared_basenames",
        "query_gallery_shared_rgb_sha256_legacy",
        "query_gallery_shared_rgb_sha256",
        "query_gallery_joint_metadata_pairs",
        "query_gallery_joint_pairs",
    ):
        _require_summary_schema(
            relations[key], "E_OFFICIAL_SUMMARY_SCHEMA", key)
    endpoint = _require_exact_mapping(
        relations["query_gallery_endpoint_pairs"], ("equal", "rgb", "pose"),
        "E_OFFICIAL_RELATIONS_SCHEMA", "endpoint")
    require(type(endpoint["equal"]) is bool,
            "E_OFFICIAL_RELATIONS_SCHEMA", "endpoint/equal")
    _require_summary_schema(
        endpoint["rgb"], "E_OFFICIAL_SUMMARY_SCHEMA", "endpoint/rgb")
    _require_summary_schema(
        endpoint["pose"], "E_OFFICIAL_SUMMARY_SCHEMA", "endpoint/pose")
    record_sets = _require_exact_mapping(
        relations["split_record_sets"], split_names,
        "E_OFFICIAL_RELATIONS_SCHEMA", "split_record_sets")
    for split in split_names:
        summary = _require_summary_schema(
            record_sets[split], "E_OFFICIAL_SUMMARY_SCHEMA",
            f"split_record_sets/{split}")
        require(summary["count"] == split_counts[split],
                "E_OFFICIAL_SUMMARY_SCHEMA", f"split_record_sets/{split}/count")
    for key in (
        "allowed_pair_count", "junk_true_count", "junk_false_count",
        "forbidden_pair_count",
    ):
        require(type(relations[key]) is int and int(relations[key]) >= 0,
                "E_OFFICIAL_RELATIONS_SCHEMA", key)

    pairs = report_map["pairs"]
    require(isinstance(pairs, list), "E_OFFICIAL_PAIR_SCHEMA", "pairs")
    pair_integer_keys = (
        "camid", "frame", "gallery_index", "gallery_target_person_idx",
        "person_count", "pid", "query_index", "query_target_person_idx",
        "source_camid", "source_frame_id", "source_pid", "viewid")
    pair_string_keys = (
        "basename", "gallery_pose_path_sha256", "gallery_rgb_relpath",
        "hraw_sha256", "nuisance_sha256", "pose_content_sha256",
        "query_pose_path_sha256", "query_rgb_relpath", "rgb_sha256",
        "score_sha256")
    pair_list_keys = (
        "effective_pose_person_sha256", "full_pose_person_sha256",
        "gallery_effective_pose_person_relpaths",
        "gallery_full_pose_person_relpaths",
        "query_effective_pose_person_relpaths", "query_full_pose_person_relpaths")
    pair_sha_keys = (
        "gallery_pose_path_sha256", "hraw_sha256", "nuisance_sha256",
        "pose_content_sha256", "query_pose_path_sha256", "rgb_sha256",
        "score_sha256")
    normalized_pairs: List[Dict[str, object]] = []
    for index, row_value in enumerate(pairs):
        row = _require_exact_mapping(
            row_value, PAIR_KEYS, "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}")
        require(all(type(row[key]) is int for key in pair_integer_keys)
                and all(isinstance(row[key], str) and row[key]
                        for key in pair_string_keys)
                and all(isinstance(row[key], list) and row[key]
                        and all(isinstance(item, str) and item for item in row[key])
                        for key in pair_list_keys)
                and isinstance(row["report"], Mapping),
                "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}/types")
        require(FILENAME_PATTERN.fullmatch(str(row["basename"])) is not None
                and int(row["query_index"]) >= 0
                and int(row["gallery_index"]) >= 0,
                "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}/identity")
        require(all(re.fullmatch(r"[0-9a-f]{64}", str(row[key])) is not None
                    for key in pair_sha_keys)
                and all(re.fullmatch(r"[0-9a-f]{64}", str(item)) is not None
                        for key in ("effective_pose_person_sha256",
                                    "full_pose_person_sha256")
                        for item in row[key]),
                "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}/sha256")
        try:
            normalized_report = dict(row["report"])
            canonical_json_bytes(normalized_report)
        except (TypeError, ValueError) as error:
            raise GateProtocolError(
                "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}/report: {error}") from error

        full_sha = list(row["full_pose_person_sha256"])
        effective_sha = list(row["effective_pose_person_sha256"])
        query_full = list(row["query_full_pose_person_relpaths"])
        gallery_full = list(row["gallery_full_pose_person_relpaths"])
        query_effective = list(row["query_effective_pose_person_relpaths"])
        gallery_effective = list(row["gallery_effective_pose_person_relpaths"])
        require(
            len(full_sha) == len(query_full) == len(gallery_full) > 0
            and len(effective_sha) == len(query_effective) ==
                len(gallery_effective) > 0,
            "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}/constituent lengths",
        )
        query_target = int(row["query_target_person_idx"])
        gallery_target = int(row["gallery_target_person_idx"])
        expected_indices = list(range(min(len(full_sha), 6)))
        require(query_target == gallery_target
                and 0 <= query_target < len(expected_indices),
                "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}/target")
        if query_target > 0:
            target = expected_indices.pop(query_target)
            expected_indices.insert(0, target)
        require(
            int(row["person_count"]) == len(expected_indices)
            and effective_sha == [full_sha[position] for position in expected_indices],
            "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}/effective projection",
        )
        query_full_names = [Path(value).name for value in query_full]
        gallery_full_names = [Path(value).name for value in gallery_full]
        query_effective_names = [Path(value).name for value in query_effective]
        gallery_effective_names = [Path(value).name for value in gallery_effective]
        require(
            query_full_names == gallery_full_names
            and query_effective_names == gallery_effective_names
            and query_effective_names ==
                [query_full_names[position] for position in expected_indices]
            and all(value == f"pose_data/query/{Path(value).name}"
                    and Path(value).name not in ("", ".", "..")
                    for value in query_full + query_effective)
            and all(value == f"pose_data/gallery/{Path(value).name}"
                    and Path(value).name not in ("", ".", "..")
                    for value in gallery_full + gallery_effective),
            "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}/constituent projection",
        )
        basename = str(row["basename"])
        require(
            row["query_rgb_relpath"] == f"query/{basename}"
            and row["gallery_rgb_relpath"] == f"bounding_box_test/{basename}"
            and row["pose_content_sha256"] ==
                sha256_bytes(canonical_json_bytes(full_sha)),
            "E_OFFICIAL_PAIR_SCHEMA", f"pairs/{index}/asset projection",
        )
        normalized_row = dict(row)
        normalized_row["report"] = normalized_report
        normalized_pairs.append(normalized_row)

    expected_order = sorted(pairs, key=lambda row: (
        row["rgb_sha256"], row["basename"], row["query_index"], row["gallery_index"]))
    require(pairs == expected_order,
            "E_OFFICIAL_PAIR_SCHEMA", "pair order")
    for key in ("basename", "rgb_sha256", "query_index", "gallery_index"):
        require(len({row[key] for row in pairs}) == len(pairs),
                "E_OFFICIAL_PAIR_SCHEMA", f"pair uniqueness/{key}")

    shared_basenames = sorted(str(row["basename"]) for row in normalized_pairs)
    shared_rgb = [str(row["rgb_sha256"]) for row in normalized_pairs]
    endpoint_pairs = sorted(
        (int(row["query_index"]), int(row["gallery_index"]))
        for row in normalized_pairs)
    legacy_rgb_payload = json.dumps(
        shared_rgb, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    recomputed = {
        "query_gallery_shared_basenames": _canonical_summary(shared_basenames),
        "query_gallery_shared_rgb_sha256_legacy": {
            "count": len(shared_rgb),
            "canonical_bytes": len(legacy_rgb_payload),
            "sha256": sha256_bytes(legacy_rgb_payload),
        },
        "query_gallery_shared_rgb_sha256": _canonical_summary(shared_rgb),
        "query_gallery_joint_metadata_pairs": _canonical_summary(
            _joint_metadata_projection(normalized_pairs)),
        "query_gallery_joint_pairs": _canonical_summary(normalized_pairs),
    }
    for key, expected in recomputed.items():
        require(relations[key] == expected,
                "E_OFFICIAL_PAIR_RELATION_CLOSURE", key)
    expected_endpoint = _canonical_summary(endpoint_pairs)
    require(relations["query_gallery_endpoint_pairs"] == {
        "equal": True, "rgb": expected_endpoint, "pose": expected_endpoint,
    }, "E_OFFICIAL_PAIR_RELATION_CLOSURE", "query_gallery_endpoint_pairs")
    require(
        int(relations["allowed_pair_count"]) == len(normalized_pairs)
        and int(relations["junk_true_count"]) == len(normalized_pairs)
        and int(relations["junk_false_count"]) == 0
        and int(relations["forbidden_pair_count"]) == 0,
        "E_OFFICIAL_PAIR_RELATION_CLOSURE", "pair counts",
    )

    without_self = dict(report_map)
    frozen_sha = without_self.pop("relation_report_sha256")
    try:
        expected_self_hash = sha256_bytes(canonical_json_bytes(without_self))
    except (TypeError, ValueError) as error:
        raise GateProtocolError(
            "E_RELATION_REPORT_SELF_HASH", f"canonicalization: {error}") from error
    require(isinstance(frozen_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", frozen_sha) is not None
            and frozen_sha == expected_self_hash,
            "E_RELATION_REPORT_SELF_HASH", str(frozen_sha))
    return report_map


def assert_occluded_duke_official_v1(report: Mapping[str, object]) -> None:
    report = _validate_official_report_schema(report)
    require(report.get("official_source") == {
        "repository": OFFICIAL_REPOSITORY,
        "commit": OFFICIAL_COMMIT,
        "filename_regex": FILENAME_REGEX,
    }, "E_OFFICIAL_SOURCE", str(report.get("official_source")))
    require(report.get("split_counts") == {
        split: int(OFFICIAL_SPLITS[split]["count"])
        for split in ("train", "query", "gallery")
    }, "E_OFFICIAL_SPLIT_COUNT", str(report.get("split_counts")))
    for split, expected in OFFICIAL_SPLITS.items():
        actual = report["official_lists"][split]
        require(actual == {
            "rgb_root": expected["root"], "list": expected["list"],
            "count": expected["count"], "raw_bytes": expected["raw_bytes"],
            "raw_sha256": expected["raw_sha256"],
            "canonical_bytes": expected["canonical_bytes"],
            "canonical_sha256": expected["canonical_sha256"],
            "pose_index_bytes": expected["pose_index_bytes"],
            "pose_index_sha256": expected["pose_index_sha256"],
        }, "E_OFFICIAL_LIST_DIGEST", split)
        within = report["within_split"][split]
        require(int(within["target_outside_effective_count"]) == 0,
                "E_OFFICIAL_TARGET_OUTSIDE", split)
        require(all(int(value) == 0 for key, value in within.items()
                    if key.endswith("duplicate_count")),
                "E_OFFICIAL_WITHIN_DUPLICATE", split)
    require({split: int(report["within_split"][split]["source_pid_count"])
             for split in OFFICIAL_SOURCE_PID_COUNTS} == OFFICIAL_SOURCE_PID_COUNTS,
            "E_OFFICIAL_SOURCE_PID_COUNT", str(report["within_split"]))
    for label in ("train_query", "train_gallery"):
        require(all(int(value) == 0 for value in report["cross_split"][label].values()),
                "E_OFFICIAL_TRAIN_EVAL_ALIAS", label)
    qg = report["cross_split"]["query_gallery"]
    require(qg == OFFICIAL_QUERY_GALLERY_COUNTS,
            "E_OFFICIAL_QUERY_GALLERY_RELATION", str(qg))
    relations = report["relations"]
    for key, constant in (
        ("query_gallery_shared_basenames", RELATION_EXACT["shared_basename"]),
        ("query_gallery_shared_rgb_sha256_legacy", RELATION_EXACT["shared_rgb_legacy"]),
        ("query_gallery_shared_rgb_sha256", RELATION_EXACT["shared_rgb"]),
        ("query_gallery_joint_metadata_pairs", RELATION_EXACT["joint_metadata"]),
        ("query_gallery_joint_pairs", RELATION_EXACT["joint_pairs"]),
    ):
        count, size, digest = constant
        require(relations[key] == {
            "count": count, "canonical_bytes": size, "sha256": digest,
        }, "E_OFFICIAL_RELATION_DIGEST", key)
    count, size, digest = RELATION_EXACT["endpoint_pairs"]
    endpoint = relations["query_gallery_endpoint_pairs"]
    expected_endpoint = {"count": count, "canonical_bytes": size, "sha256": digest}
    require(endpoint == {"equal": True, "rgb": expected_endpoint, "pose": expected_endpoint},
            "E_OFFICIAL_ENDPOINT_DIGEST", str(endpoint))
    require(
        len(report["pairs"]) == OFFICIAL_ALLOWED_PAIR_COUNT
        and int(relations["allowed_pair_count"]) == OFFICIAL_ALLOWED_PAIR_COUNT
        and int(relations["junk_true_count"]) == OFFICIAL_ALLOWED_PAIR_COUNT
        and int(relations["junk_false_count"]) == 0
        and int(relations["forbidden_pair_count"]) == 0,
        "E_OFFICIAL_PAIR_COUNT", str(relations),
    )
    without_self = dict(report)
    frozen_sha = str(without_self.pop("relation_report_sha256", ""))
    require(frozen_sha == sha256_bytes(canonical_json_bytes(without_self)),
            "E_RELATION_REPORT_SELF_HASH", frozen_sha)


def build_relation_report_v2(
    dataset: OccludedDukeMTMC,
    split_datasets: Mapping[str, PoseImageDataset],
    prepared: Path,
) -> Tuple[Dict[str, object], Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]:
    """Build the full active-asset report and return full/quick identities."""

    started = time.monotonic()
    prepared = Path(prepared).resolve()
    lexical_data_root = _lexical_absolute(dataset.dataset_dir)
    data_root = lexical_data_root.resolve()
    require(data_root.is_dir(), "E_RELATION_DATA_ROOT", str(data_root))
    require(set(split_datasets) == {"train", "query", "gallery"},
            "E_RELATION_SPLITS", str(split_datasets.keys()))
    identity_registry: Dict[str, Tuple[int, ...]] = {}
    quick_paths: List[Path] = []
    official_lists, official_report = _official_lists_v2(dataset, identity_registry)
    quick_paths.extend([
        Path(dataset.train_list).resolve(),
        Path(dataset.query_list).resolve(),
        Path(dataset.gallery_list).resolve(),
    ])

    records: Dict[str, List[SceneRecord]] = {}
    cache_arrays: Dict[str, Dict[str, np.ndarray]] = {
        "heatmaps": {}, "scores": {}, "nuisance": {},
    }
    cache_reports: Dict[Path, Tuple[Dict[str, object], Tuple[int, ...]]] = {}
    file_hash_cache: Dict[str, str] = {}
    try:
        for split in ("train", "query", "gallery"):
            split_dataset = split_datasets[split]
            expected = OFFICIAL_SPLITS[split]
            require(int(split_dataset.max_persons) == 6,
                    "E_OFFICIAL_MAX_PERSONS", f"{split}/{split_dataset.max_persons}")
            configured_rgb_root = _exact_lexical_child(
                getattr(dataset, f"{split}_dir"),
                dataset.dataset_dir,
                (str(expected["root"]),),
                "E_RELATION_RGB_ROOT",
            )
            expected_rgb_root = lexical_data_root / str(expected["root"])
            require(configured_rgb_root == expected_rgb_root,
                    "E_RELATION_RGB_ROOT",
                    f"{split}/{configured_rgb_root}!={expected_rgb_root}")
            _require_real_directory(configured_rgb_root, "E_RELATION_RGB_ROOT")
            rgb_root = configured_rgb_root.resolve()
            require(rgb_root == data_root / str(expected["root"])
                    and rgb_root.parent == data_root,
                    "E_RELATION_RGB_ROOT", f"{split}/{rgb_root}")
            configured_pose_base = lexical_data_root / "pose_data"
            configured_pose_root = _exact_lexical_child(
                split_dataset.pose_dir,
                dataset.dataset_dir,
                ("pose_data", split),
                "E_RELATION_POSE_ROOT",
            )
            expected_pose_root = configured_pose_base / split
            require(configured_pose_root == expected_pose_root,
                    "E_RELATION_POSE_ROOT",
                    f"{split}/{configured_pose_root}!={expected_pose_root}")
            _require_real_directory(configured_pose_base, "E_RELATION_POSE_ROOT")
            _require_real_directory(configured_pose_root, "E_RELATION_POSE_ROOT")
            pose_root = configured_pose_root.resolve()
            require(pose_root == data_root / "pose_data" / split,
                    "E_RELATION_POSE_ROOT", f"{split}/{pose_root}")
            index_path = pose_root / "index.json"
            _require_direct_child(index_path, pose_root, "E_RELATION_POSE_INDEX_PATH")
            index, index_report = _stable_json(
                index_path, identity_registry, "E_RELATION_POSE_INDEX_JSON")
            require(isinstance(index, dict) and index == split_dataset.index,
                    "E_RELATION_ACTIVE_INDEX_DRIFT", split)
            require(sorted(index) == list(official_lists[split]),
                    "E_RELATION_POSE_INDEX_KEYS", split)
            official_report[split]["pose_index_bytes"] = int(index_report["bytes"])
            official_report[split]["pose_index_sha256"] = str(index_report["sha256"])
            quick_paths.append(index_path)

            metadata_path = prepared / f"{split}_metadata.json"
            metadata, _metadata_report = _stable_json(
                metadata_path, identity_registry, "E_METADATA_SCHEMA_V2")
            quick_paths.append(metadata_path)
            count = len(official_lists[split])
            specs = {
                "heatmaps": (prepared / f"{split}_scene_heatmaps.npy",
                             [count, 17, 96, 32], "<f4"),
                "scores": (prepared / f"{split}_scene_scores.npy", [count, 17], "<f4"),
                "nuisance": (prepared / f"{split}_continuous.npy", [count, 95], "<f8"),
            }
            for name, (path, shape, dtype) in specs.items():
                before_report, before_identity, _unused = _stable_regular_file(path)
                if split in OFFICIAL_QG_CACHE_FILES:
                    expected_bytes, expected_sha = OFFICIAL_QG_CACHE_FILES[split][name]
                    require(before_report == {
                        "bytes": expected_bytes, "sha256": expected_sha,
                    }, "E_OFFICIAL_CACHE_DIGEST", f"{split}/{name}")
                _register_identity(identity_registry, path, before_identity)
                try:
                    value = np.load(path, mmap_mode="r", allow_pickle=False)
                except OSError as error:
                    raise GateProtocolError(
                        "E_RELATION_FILE_TOCTOU",
                        f"{path}: {error.__class__.__name__}") from error
                except (ValueError, EOFError) as error:
                    raise GateProtocolError(
                        "E_RELATION_CACHE_LOAD",
                        f"{path}: {error.__class__.__name__}") from error
                require(value.dtype.str == dtype,
                        "E_RELATION_CACHE_DTYPE",
                        f"{path}/{value.dtype.str}!={dtype}")
                require(list(value.shape) == shape,
                        "E_RELATION_CACHE_SHAPE", f"{path}/{value.shape}")
                for start in range(0, count, 32):
                    stop = min(start + 32, count)
                    require(bool(np.isfinite(value[start:stop]).all()),
                            "E_RELATION_CACHE_NONFINITE", f"{path}[{start}:{stop}]")
                cache_arrays[name][split] = value
                cache_reports[path] = (before_report, before_identity)
                quick_paths.append(path)

            split_records = _scene_records_from_payload(
                metadata, cache_arrays["nuisance"][split], split)
            require(len(split_dataset.dataset) == len(split_records) == count,
                    "E_RELATION_ACTIVE_COUNT", split)
            for index_value, (active, record) in enumerate(
                    zip(split_dataset.dataset, split_records)):
                active_path, active_pid, active_camid, active_viewid = active
                resolved = Path(active_path).resolve()
                require(
                    record.index == index_value
                    and record.path == str(resolved)
                    and record.pid == int(active_pid)
                    and record.camid == int(active_camid)
                    and record.viewid == int(active_viewid),
                    "E_RELATION_ACTIVE_RECORD", f"{split}/{index_value}",
                )
                require(resolved.parent == rgb_root
                        and resolved.name == official_lists[split][index_value],
                        "E_RELATION_RGB_ROOT", str(resolved))
                rgb_report, rgb_identity, _unused = _stable_regular_file(resolved)
                _register_identity(identity_registry, resolved, rgb_identity)
                require(str(rgb_report["sha256"]) == record.rgb_sha256,
                        "E_RELATION_RGB_SHA", str(resolved))
                pose = _pose_asset_manifest_v2(
                    split_dataset, str(resolved), file_hash_cache, identity_registry)
                require(
                    record.pose_path_sha256 == pose["pose_path_sha256"]
                    and record.pose_content_sha256 == pose["pose_content_sha256"]
                    and record.source_pid == pose["source_pid"]
                    and record.source_camid == pose["source_camid"]
                    and record.source_frame_id == pose["source_frame_id"]
                    and record.target_person_idx == pose["target_person_idx"]
                    and record.full_pose_person_relpaths ==
                        pose["full_pose_person_relpaths"]
                    and record.full_pose_person_paths == pose["full_pose_person_paths"]
                    and record.full_pose_person_sha256 == pose["full_pose_person_sha256"]
                    and record.effective_pose_person_relpaths ==
                        pose["effective_pose_person_relpaths"]
                    and record.effective_pose_person_paths ==
                        pose["effective_pose_person_paths"]
                    and record.effective_pose_person_sha256 ==
                        pose["effective_pose_person_sha256"],
                    "E_RELATION_POSE_ASSET_DRIFT", f"{split}/{index_value}",
                )
                require(record.camid == record.source_camid,
                        "E_RELATION_SOURCE_CAM", f"{split}/{index_value}")
                if split != "train":
                    require(record.pid == record.source_pid,
                            "E_RELATION_SOURCE_PID", f"{split}/{index_value}")
            records[split] = split_records

        train_pid_map: Dict[int, set] = defaultdict(set)
        for record in records["train"]:
            train_pid_map[record.source_pid].add(record.pid)
        require(all(len(values) == 1 for values in train_pid_map.values())
                and len({next(iter(values)) for values in train_pid_map.values()}) ==
                len(train_pid_map),
                "E_RELATION_TRAIN_PID_BIJECTION", "")

        report = audit_split_relations_v2(
            records, official_lists, official_report, cache_arrays)
        assert_occluded_duke_official_v1(report)
    finally:
        for by_split in cache_arrays.values():
            for value in by_split.values():
                _close_memmap(value)

    for path, (before_report, before_identity) in cache_reports.items():
        after_report, after_identity, _unused = _stable_regular_file(path)
        require(after_report == before_report and after_identity == before_identity,
                "E_RELATION_CACHE_TOCTOU", str(path))
    _recheck_identities(identity_registry)
    quick_registry = {
        str(path): identity_registry[str(path)]
        for path in quick_paths
    }
    relation_path = prepared / "split_relations.json"
    if relation_path.is_file():
        _relation_report, relation_identity, _unused = _stable_regular_file(relation_path)
        _register_identity(identity_registry, relation_path, relation_identity)
        quick_registry[str(relation_path)] = relation_identity
    _recheck_identities(identity_registry)
    elapsed = time.monotonic() - started
    require(elapsed <= 90.0, "E_RELATION_AUDIT_TIMEOUT", f"{elapsed:.3f}s")
    require(len(quick_registry) in {18, 19},
            "E_RELATION_QUICK_SET", str(len(quick_registry)))
    return report, identity_registry, quick_registry


def save_mapping_payload(destination: Path, split: str, payload: Mapping[str, object]) -> Dict[str, str]:
    mappings = np.stack(payload["mappings"]).astype(np.int32)
    np.save(destination / f"{split}_mappings.npy", mappings, allow_pickle=False)
    adjacency = payload["selected_adjacency"]
    offsets = np.zeros(len(adjacency) + 1, dtype=np.int64)
    donors: List[int] = []
    for index, values in enumerate(adjacency):
        donors.extend(int(value) for value in values)
        offsets[index + 1] = len(donors)
    base_edges = payload["base_edges"]
    edge_left = np.asarray([edge[0] for edge in sorted(base_edges)], dtype=np.int32)
    edge_right = np.asarray([edge[1] for edge in sorted(base_edges)], dtype=np.int32)
    edge_cost = np.asarray([base_edges[edge] for edge in sorted(base_edges)], dtype=np.float64)
    randomized_edge_cost = np.stack(payload["randomized_edge_costs"]).astype(
        np.float64, copy=False)
    require(randomized_edge_cost.shape == (20, len(edge_cost)),
            "E_GUMBEL_COST_SHAPE", str(randomized_edge_cost.shape))
    np.savez(
        destination / f"{split}_candidate_graph.npz",
        offsets=offsets,
        donors=np.asarray(donors, dtype=np.int32),
        edge_left=edge_left,
        edge_right=edge_right,
        edge_cost=edge_cost,
        randomized_edge_cost=randomized_edge_cost,
        baseline_mean_costs=np.asarray(payload["baseline_mean_costs"], dtype=np.float64),
    )
    audit = {
        "scaler": payload["scaler"],
        "mapping_audits": payload["mapping_audits"],
        "minimum_hamming": payload["minimum_hamming"],
        "effective_unique_count": payload["effective_unique_count"],
        "eta_by_seed": payload["eta_by_seed"],
        "strata": payload["strata"],
        "mapping_seeds": list(MAPPING_SEEDS),
        "baseline_seeds": list(BASELINE_SEEDS),
        "solver": payload["solver"],
        "cost_formula_version": (
            "Cbase_v1=mean_m_min(abs(z_i-z_j),5)+0.25*camera_neq+0.25*frame_neq;"
            "u_dim=95;z=(u-median)/(1.4826*MAD);MAD_lt_1e-8=>z_dim=0;"
            "winsor=[-5,5];"
            "all_cost_arithmetic=float64"
        ),
        "candidate_k_sequence": list(K_SEQUENCE),
    }
    atomic_write_json(destination / f"{split}_mapping_audit.json", audit)
    return {
        "mappings": f"{split}_mappings.npy",
        "candidate_graph": f"{split}_candidate_graph.npz",
        "audit": f"{split}_mapping_audit.json",
    }


def build_centroid_cache(prepared: Path, counts: Mapping[str, int]) -> Dict[str, object]:
    train = np.load(prepared / "train_scene_heatmaps.npy", mmap_mode="r")
    try:
        targets = fit_normalized_centroid_targets(
            torch.from_numpy(np.asarray(train[index]))
            for index in range(int(counts["train"])))
    finally:
        _close_memmap(train)
    require(all(target is not None for target in targets), "E_CENTROID_TARGET", "missing train joint")
    atomic_write_json(prepared / "centroid_targets.json", targets)
    status: Dict[str, object] = {
        "status": "PASS",
        "geometry_transform": "positive_part_v1",
        "output_transform": "translate_signed_raw_v1",
        "negative_mass_ratio_bounds": [0.95, 1.05],
        "targets": targets,
    }
    for split in ("query", "gallery"):
        source = np.load(prepared / f"{split}_scene_heatmaps.npy", mmap_mode="r")
        output = _new_memmap(
            prepared / f"{split}_centroid_heatmaps.npy",
            tuple(source.shape),
            "float32",
        )
        try:
            for index in range(source.shape[0]):
                scene = torch.from_numpy(np.asarray(source[index]))
                absolute = absolute_centroid_targets(scene, targets)
                output[index] = apply_scene_centroid_control(scene, absolute).numpy()
            output.flush()
        finally:
            _close_memmap(output)
            _close_memmap(source)
    return status


def artifact_hashes(directory: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(value for value in directory.rglob("*") if value.is_file())
    }


def prepare_phase(args: argparse.Namespace) -> None:
    os.chdir(ROOT)
    config_file = Path(args.config_file).resolve()
    require(config_file.is_file(), "E_CONFIG_MISSING", str(config_file))
    local_cfg = resolved_config(config_file, args.opts)
    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is not None:
        _assert_not_burned_execution(resume_path)
    output_root = Path(args.output_root).resolve()
    _assert_not_burned_execution(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".exp374_prepare_", dir=str(output_root)))
    execution_dir: Path | None = None
    try:
        dataset, datasets = direct_datasets(local_cfg)
        cache_manifest = {
            split: cache_split(split, split_dataset, local_cfg, staging)
            for split, split_dataset in datasets.items()
        }
        relation_report, _relation_full_identity, _relation_quick_identity = (
            build_relation_report_v2(dataset, datasets, staging)
        )
        assert_occluded_duke_official_v1(relation_report)
        atomic_write_json(staging / "split_relations.json", relation_report)
        relation_payload = canonical_json_bytes(relation_report)
        relation_artifact = {
            "relpath": "split_relations.json",
            "bytes": len(relation_payload),
            "sha256": sha256_bytes(relation_payload),
        }
        records = {
            split: load_scene_records(staging, split)
            for split in ("train", "query", "gallery")
        }
        require(torch.cuda.is_available(), "E_MATCH_GPU_REQUIRED", "exact sparse candidate build")
        device = torch.device("cuda:0")
        for split in ("train", "query", "gallery"):
            cache_manifest[split]["signed_raw_audit"]["actual_space"] = (
                build_actual_space_audit(
                    staging,
                    split,
                    int(cache_manifest[split]["count"]),
                    device,
                    int(local_cfg.TEST.IMS_PER_BATCH),
                )
            )
        torch.cuda.empty_cache()
        _recheck_identities(_relation_full_identity)
        require(
            sha256_file(staging / "split_relations.json") == relation_artifact["sha256"],
            "E_RELATION_PREPARED_TRIPLE", "prepare pre-matching",
        )
        mapping_manifest = {}
        for split in ("query", "gallery"):
            payload = prepare_split_mappings(
                records[split], device=device, anchor_chunk=int(args.anchor_chunk),
                relation_report=relation_report, split=split)
            mapping_manifest[split] = save_mapping_payload(staging, split, payload)
            del payload
            gc.collect()
            torch.cuda.empty_cache()

        _recheck_identities(_relation_full_identity)
        require(
            sha256_file(staging / "split_relations.json") == relation_artifact["sha256"],
            "E_RELATION_PREPARED_TRIPLE", "prepare post-matching",
        )
        specs = checkpoint_specs(args.checkpoint_manifest)

        try:
            centroid_status = build_centroid_cache(
                staging,
                {split: int(cache_manifest[split]["count"]) for split in cache_manifest},
            )
        except GateProtocolError as error:
            for partial in staging.glob("*_centroid_heatmaps.npy"):
                partial.unlink()
            centroid_status = {
                "status": "INVALID_SECONDARY",
                "error_code": error.code,
                "message": str(error),
            }
            atomic_write_json(staging / "centroid_invalid.json", centroid_status)

        prepared_hashes = artifact_hashes(staging)
        premetric_manifest = {
            "schema": "exp374-gate-a-v1",
            "repository": repository_manifest(),
            "environment": runtime_environment_manifest(device),
            "config_file": str(config_file),
            "config_file_sha256": sha256_file(config_file),
            "config_opts": list(args.opts),
            "resolved_config": str(local_cfg),
            "resolved_config_sha256": sha256_bytes(str(local_cfg).encode("utf-8")),
            "checkpoints": specs,
            "dataset": {
                "name": "occluded_duke",
                "num_train_pids": int(dataset.num_train_pids),
                "num_train_cams": int(dataset.num_train_cams),
                "num_train_vids": int(dataset.num_train_vids),
                "num_query": len(datasets["query"]),
                "num_gallery": len(datasets["gallery"]),
                "cache": cache_manifest,
                "split_relations": relation_report,
                "split_relations_artifact": relation_artifact,
            },
            "matching": mapping_manifest,
            "centroid": centroid_status,
            "prepared_artifact_sha256": prepared_hashes,
            "schedule": core_schedule([int(spec["seed"]) for spec in specs]),
            "resource": {
                "passes_per_seed": 164,
                "total_passes": 492,
                "minimum_free_bytes": 80 * 1024 ** 3,
            },
        }
        require(
            prepared_hashes.get("split_relations.json") == relation_artifact["sha256"],
            "E_RELATION_PREPARED_TRIPLE", str(relation_artifact),
        )
        lock_context = (
            exclusive_execution_lock(resume_path, "prepare-resume")
            if resume_path is not None else contextlib.nullcontext()
        )
        with lock_context:
            execution_dir, execution_sha = create_execution_directory(
                output_root,
                premetric_manifest,
                resume_path,
            )
            prepared_destination = execution_dir / "prepared"
            if prepared_destination.exists():
                frozen_hashes = premetric_manifest["prepared_artifact_sha256"]
                actual_hashes = artifact_hashes(prepared_destination)
                require(actual_hashes == frozen_hashes,
                        "E_RESUME_HASH_DRIFT", "prepared artifacts")
                shutil.rmtree(staging)
            else:
                publish_directory(staging, prepared_destination)
            if resume_path is not None:
                verify_relation_runtime(
                    premetric_manifest, dataset, datasets,
                    prepared_destination, "prepare_resume")
            atomic_write_json(execution_dir / "PREPARED.json", {
                "execution_sha256": execution_sha,
                "prepared_artifact_sha256": prepared_hashes,
            })
        print(json.dumps({
            "status": "PREPARED_ONLY",
            "execution_dir": str(execution_dir),
            "execution_sha256": execution_sha,
            "tests_authorized": False,
            "formal_gate_authorized": False,
        }, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        if staging.exists():
            atomic_write_json(staging / "FAILED.json", {"status": "FAILED_PREPARE"})
        if execution_dir is not None and execution_dir.is_dir():
            atomic_write_json(execution_dir / "FAILED.json", {
                "status": "FAILED_NONREPORTABLE",
                "phase": "prepare",
                "failed_arm_published": False,
            })
        raise


def verify_prepared_artifacts(execution_dir: Path) -> Dict[str, object]:
    manifest_path = execution_dir / "premetric_manifest.json"
    require(manifest_path.is_file(), "E_RESUME_MANIFEST", str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(execution_dir.name.startswith("gate_a_"),
            "E_RESUME_EXECUTION_SHA", execution_dir.name)
    expected_sha = execution_dir.name[len("gate_a_"):]
    require(sha256_bytes(canonical_json_bytes(manifest)) == expected_sha,
            "E_RESUME_EXECUTION_SHA", str(execution_dir))
    prepared = execution_dir / "prepared"
    require(prepared.is_dir(), "E_PREPARED_MISSING", str(prepared))
    actual = artifact_hashes(prepared)
    require(actual == manifest["prepared_artifact_sha256"],
            "E_RESUME_HASH_DRIFT", "prepared artifacts")
    sha_path = execution_dir / "execution_sha256.txt"
    require(sha_path.is_file() and sha_path.read_text(encoding="utf-8") == expected_sha + "\n",
            "E_RESUME_EXECUTION_SHA", str(sha_path))
    prepared_marker = execution_dir / "PREPARED.json"
    require(prepared_marker.is_file(), "E_PREPARED_MISSING", str(prepared_marker))
    require(json.loads(prepared_marker.read_text(encoding="utf-8")) == {
        "execution_sha256": expected_sha,
        "prepared_artifact_sha256": manifest["prepared_artifact_sha256"],
    }, "E_PREPARED_MARKER", str(prepared_marker))
    require(not (execution_dir / "COMPLETE").exists(), "E_RESUME_COMPLETE", str(execution_dir))
    return manifest


def _verify_relation_artifact_triple(
    manifest: Mapping[str, object],
    prepared: Path,
) -> Dict[str, object]:
    dataset_manifest = manifest.get("dataset")
    require(isinstance(dataset_manifest, Mapping),
            "E_RELATION_PREPARED_TRIPLE", "dataset manifest")
    report = dataset_manifest.get("split_relations")
    descriptor = dataset_manifest.get("split_relations_artifact")
    require(isinstance(report, Mapping) and isinstance(descriptor, Mapping),
            "E_RELATION_PREPARED_TRIPLE", "missing relation object/artifact")
    assert_occluded_duke_official_v1(report)
    payload = canonical_json_bytes(report)
    expected = {
        "relpath": "split_relations.json",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }
    require(dict(descriptor) == expected,
            "E_RELATION_PREPARED_TRIPLE", "artifact descriptor")
    prepared_hashes = manifest.get("prepared_artifact_sha256")
    require(isinstance(prepared_hashes, Mapping)
            and prepared_hashes.get("split_relations.json") == expected["sha256"],
            "E_RELATION_PREPARED_TRIPLE", "prepared hash")
    relation_path = Path(prepared) / "split_relations.json"
    file_report, _identity, raw = _stable_regular_file(relation_path, return_bytes=True)
    require(raw == payload and file_report == {
        "bytes": expected["bytes"], "sha256": expected["sha256"],
    }, "E_RELATION_PREPARED_TRIPLE", str(relation_path))
    return dict(report)


def verify_relation_runtime(
    manifest: Mapping[str, object],
    dataset: OccludedDukeMTMC,
    split_datasets: Mapping[str, PoseImageDataset],
    prepared: Path,
    phase: str,
) -> Dict[str, object]:
    allowed_phases = {
        "prepare_resume", "run_entry", "run_tail",
        "summarize_entry", "summarize_pre_results",
    }
    require(phase in allowed_phases, "E_RELATION_RUNTIME_PHASE", phase)
    frozen = _verify_relation_artifact_triple(manifest, prepared)
    rebuilt, full_identity, quick_identity = build_relation_report_v2(
        dataset, split_datasets, prepared)
    require(canonical_json_bytes(rebuilt) == canonical_json_bytes(frozen),
            "E_RELATION_RUNTIME_DRIFT", phase)
    _verify_relation_artifact_triple(manifest, prepared)
    return {
        "phase": phase,
        "full_identity": full_identity,
        "quick_identity": quick_identity,
        "relation_report_sha256": rebuilt["relation_report_sha256"],
    }


def verify_relation_identity_snapshot(snapshot: Mapping[str, object],
                                      scope: str) -> None:
    require(scope in {"full", "quick"}, "E_RELATION_IDENTITY_SCOPE", scope)
    key = f"{scope}_identity"
    registry = snapshot.get(key)
    require(isinstance(registry, Mapping), "E_RELATION_IDENTITY_SCOPE", key)
    _recheck_identities(registry)


def assert_free_space(path: Path, minimum_bytes: int) -> None:
    free = shutil.disk_usage(path).free
    require(free >= minimum_bytes, "E_DISK_SPACE", f"{free} < {minimum_bytes}")


def validation_loader(split_datasets: Mapping[str, PoseImageDataset], local_cfg) -> DataLoader:
    combined = ConcatDataset([split_datasets["query"], split_datasets["gallery"]])
    return DataLoader(
        combined,
        batch_size=int(local_cfg.TEST.IMS_PER_BATCH),
        shuffle=False,
        num_workers=int(local_cfg.DATALOADER.NUM_WORKERS),
        collate_fn=pose_val_collate_fn,
        pin_memory=True,
        drop_last=False,
    )


@contextlib.contextmanager
def exclusive_execution_lock(execution_dir: Path, phase: str):
    lock = execution_dir / "RUN.lock"
    descriptor = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = canonical_json_bytes({
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "phase": phase,
        })
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if lock.exists():
            lock.unlink()
            fsync_parent = lock.parent
            directory_descriptor = os.open(str(fsync_parent), os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)


def verify_frozen_config_environment(
    manifest: Mapping[str, object],
    device: torch.device,
):
    config_file = Path(str(manifest["config_file"])).resolve()
    require(config_file.is_file(), "E_CONFIG_MISSING", str(config_file))
    require(sha256_file(config_file) == manifest["config_file_sha256"],
            "E_CONFIG_SHA", str(config_file))
    local_cfg = resolved_config(config_file, list(manifest["config_opts"]))
    require(str(local_cfg) == manifest["resolved_config"],
            "E_CONFIG_DRIFT", "resolved config text")
    require(sha256_bytes(str(local_cfg).encode("utf-8")) ==
            manifest["resolved_config_sha256"], "E_CONFIG_DRIFT", "resolved config SHA")
    require(repository_manifest() == manifest["repository"],
            "E_REPOSITORY_DRIFT", "repository/package/code manifest")
    require(runtime_environment_manifest(device) == manifest["environment"],
            "E_ENVIRONMENT_DRIFT", "runtime environment")
    return local_cfg


def verify_frozen_checkpoint_specs(
    manifest: Mapping[str, object],
    device: torch.device,
):
    require(device.type == "cuda" and torch.cuda.is_available(),
            "E_DEVICE", str(device))
    specs = [dict(value) for value in manifest["checkpoints"]]
    require({int(value["seed"]) for value in specs} == {42, 1234, 2024},
            "E_CHECKPOINT_SEEDS", str([value["seed"] for value in specs]))
    for spec in specs:
        for key in ("weight", "flat_log", "train_log"):
            path = Path(str(spec[key])).resolve()
            require(path.is_file(), "E_CHECKPOINT_ASSET", str(path))
            expected_key = "weight_sha256" if key == "weight" else f"{key}_sha256"
            asset_payload = path.read_bytes()
            require(sha256_bytes(asset_payload) == spec[expected_key],
                    "E_CHECKPOINT_SHA", str(path))
            if key == "flat_log":
                require(parse_flat_log_metrics_bytes(asset_payload, str(path)) ==
                        spec["flat_log_metrics"],
                        "E_FLAT_LOG_DRIFT", str(spec["seed"]))
    return sorted(specs, key=lambda value: int(value["seed"]))


def combined_metadata(prepared: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split in ("query", "gallery"):
        payload = json.loads(
            (prepared / f"{split}_metadata.json").read_text(encoding="utf-8"))
        for row in payload:
            copied = dict(row)
            copied["combined_index"] = len(rows)
            rows.append(copied)
    return rows


def runtime_metadata_from_relation_snapshot(
    prepared: Path,
    snapshot: Mapping[str, object],
) -> List[Dict[str, object]]:
    registry = snapshot.get("full_identity")
    require(isinstance(registry, Mapping), "E_RELATION_IDENTITY_SCOPE", "full")
    combined = combined_metadata(prepared)
    for row in combined:
        path = str(row["path"])
        require(path in registry, "E_RELATION_IDENTITY_SCOPE", path)
        row["runtime_rgb_identity"] = tuple(registry[path])
    return combined


def arm_artifact_hashes(directory: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(value for value in directory.rglob("*") if value.is_file())
        if path.name != "COMPLETE.json"
    }


def verify_published_arm(
    arm_dir: Path,
    expected_provenance: Mapping[str, object] | None = None,
) -> Dict[str, object] | None:
    if not arm_dir.exists():
        return None
    require(arm_dir.is_dir(), "E_ARM_PUBLISH_TYPE", str(arm_dir))
    marker = arm_dir / "COMPLETE.json"
    require(marker.is_file(), "E_ARM_INCOMPLETE", str(arm_dir))
    payload = json.loads(marker.read_text(encoding="utf-8"))
    require(payload.get("status") in {"PASS", "INVALID_SECONDARY"},
            "E_ARM_STATUS", str(payload.get("status")))
    require(arm_artifact_hashes(arm_dir) == payload["artifact_sha256"],
            "E_ARM_HASH", str(arm_dir))
    if expected_provenance is not None:
        expected = dict(expected_provenance)
        require(payload.get("provenance") == expected,
                "E_ARM_PROVENANCE", str(arm_dir))
        require(arm_dir.name == expected["arm_id"],
                "E_ARM_PROVENANCE", f"directory {arm_dir.name}")
        summary_path = arm_dir / "summary.json"
        require(summary_path.is_file(), "E_ARM_SUMMARY", str(summary_path))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summary.get("provenance") == expected,
                "E_ARM_PROVENANCE", f"summary {arm_dir}")
        require(summary.get("row") == expected["row"] and
                summary.get("arm_id") == expected["arm_id"],
                "E_ARM_PROVENANCE", f"row {arm_dir}")
        require(summary.get("mapping") == expected["mapping"],
                "E_ARM_PROVENANCE", f"mapping {arm_dir}")
        if payload["status"] == "INVALID_SECONDARY":
            require(expected["row"]["arm"] == "centroid" and
                    summary.get("status") == "INVALID_SECONDARY",
                    "E_ARM_STATUS", str(arm_dir))
            require(not (arm_dir / "per_query.npz").exists(),
                    "E_ARM_STATUS", "invalid centroid has metrics")
        else:
            require(summary.get("status") == "PASS" and
                    (arm_dir / "per_query.npz").is_file(),
                    "E_ARM_STATUS", str(arm_dir))
            correct_start = (
                expected["row"]["arm"] == "correct"
                and expected["row"].get("position") == "start"
            )
            require((arm_dir / "actual_psg_input.npy").is_file() == correct_start,
                    "E_ARM_FILES", str(arm_dir))
    return payload


def publish_arm(
    temporary: Path,
    published: Path,
    provenance: Mapping[str, object],
    *,
    quick_identity: Mapping[str, Tuple[int, ...]],
    status: str = "PASS",
) -> Dict[str, object]:
    require(status in {"PASS", "INVALID_SECONDARY"}, "E_ARM_STATUS", status)
    hashes = arm_artifact_hashes(temporary)
    marker = {
        "status": status,
        "artifact_sha256": hashes,
        "provenance": dict(provenance),
    }
    atomic_write_json(temporary / "COMPLETE.json", marker)
    require(len(quick_identity) == 19, "E_RELATION_QUICK_SET", str(len(quick_identity)))
    _recheck_identities(quick_identity)
    publish_directory(temporary, published)
    return marker


def assert_isolated_psg(local_cfg, model: PoseBackboneModel) -> None:
    require(type(model) is PoseBackboneModel, "E_MODEL_CLASS", type(model).__name__)
    cfg_required = {
        "POSE_ENABLED": True,
        "POSE_BACKBONE_PSG": True,
        "POSE_PSG_SPATIAL": False,
        "POSE_USE_TARGET_HEATMAP": False,
    }
    for name, expected in cfg_required.items():
        require(getattr(local_cfg.MODEL, name, None) == expected,
                "E_ACTIVE_CONFIG", f"{name}!={expected}")
    require(list(getattr(local_cfg.MODEL, "POSE_PSG_STAGES", [])) == [-1],
            "E_ACTIVE_CONFIG", "POSE_PSG_STAGES")
    require(int(getattr(local_cfg.MODEL, "POSE_PFM_HIDDEN", -1)) == 64,
            "E_ACTIVE_CONFIG", "POSE_PFM_HIDDEN")
    require(list(local_cfg.MODEL.POSE_HEATMAP_SIZE) == [96, 32],
            "E_ACTIVE_CONFIG", "POSE_HEATMAP_SIZE")
    require(list(local_cfg.INPUT.SIZE_TEST) == [384, 128],
            "E_ACTIVE_CONFIG", "SIZE_TEST")
    require(str(local_cfg.MODEL.TRANSFORMER_TYPE) == "swin_tiny_patch4_window7_224",
            "E_ACTIVE_CONFIG", "TRANSFORMER_TYPE")
    require(not bool(getattr(model, "reduce_feat_dim", False)), "E_ACTIVE_MODEL", "reduce_feat_dim")
    require(str(getattr(model, "neck_feat", "")) == "before", "E_ACTIVE_MODEL", "neck_feat")
    require(not model.training, "E_ACTIVE_MODEL", "model must be eval")
    require(len(model.base.stages) == 4, "E_ACTIVE_MODEL", "backbone stages")
    require(model.psg_stage_indices == {3}, "E_ACTIVE_MODEL", str(model.psg_stage_indices))
    require(set(model.psg_modules_dict.keys()) == {"s3_b0", "s3_b1"},
            "E_ACTIVE_MODEL", str(model.psg_modules_dict.keys()))
    require(len(model.psg_modules) == 2, "E_ACTIVE_MODEL", "PSG alias length")
    for block_index, key in enumerate(("s3_b0", "s3_b1")):
        gate = model.psg_modules_dict[key]
        require(model.psg_modules[block_index] is gate, "E_PSG_ALIAS", key)
        layers = list(gate.encoder.children())
        require(len(layers) == 3, "E_PSG_ENCODER", key)
        require(isinstance(layers[0], nn.Conv2d) and layers[0].kernel_size == (1, 1),
                "E_PSG_ENCODER", f"{key}/conv0")
        require(layers[0].in_channels == 17 and layers[0].out_channels == 64,
                "E_PSG_ENCODER", f"{key}/conv0 shape")
        require(isinstance(layers[1], nn.ReLU), "E_PSG_ENCODER", f"{key}/relu")
        require(isinstance(layers[2], nn.Conv2d) and layers[2].kernel_size == (1, 1),
                "E_PSG_ENCODER", f"{key}/conv1")
        require(layers[2].in_channels == 64 and layers[2].out_channels == 768,
                "E_PSG_ENCODER", f"{key}/conv1 shape")

    disabled = {
        "POSE_DUAL_STREAM": "use_dual_stream",
        "POSE_PSG_PART": "use_psg_part",
        "POSE_PFM_ENABLED": "use_pfm",
        "POSE_ADDITIVE_ADAPTER": "use_paa",
        "POSE_PATCH_EMBED": "use_pose_patch_embed",
        "POSE_PROMPT": "use_pose_prompt",
        "POSE_SHUFFLE": "use_pose_shuffle",
        "POSE_CHANNEL_SHUFFLE": "use_pose_channel_shuffle",
        "POSE_VCSR": "use_vcsr",
        "POSE_LGPA": "use_lgpa",
        "POSE_PBSR": "use_pbsr",
        "POSE_CLIP_ID_PROMPT": "use_clip_id_prompt",
        "POSE_PCMSC": "use_pcmsc",
        "POSE_PGPD": "use_pgpd",
        "POSE_PPA": "use_ppa",
        "POSE_FSDC": "use_fsdc",
        "POSE_SKELETON_GCN": "use_skeleton_gcn",
        "POSE_VCNORM": "use_vcnorm",
        "POSE_NORMALIZE": "use_pose_normalize",
        "POSE_STRUCTURAL_ROUTING": "use_structural_routing",
        "POSE_SPLADE": "use_splade",
        "POSE_LTCS": "use_ltcs",
        "POSE_LPCS": "use_lpcs",
        "POSE_BA_PKC": "ba_pkc",
        "POSE_BT_PKD": "bt_pkd",
    }
    for config_name, attribute_name in disabled.items():
        require(not bool(getattr(local_cfg.MODEL, config_name, False)),
                "E_ACTIVE_CONFIG", config_name)
        require(not bool(getattr(model, attribute_name, False)),
                "E_ACTIVE_MODEL", attribute_name)
    require(float(getattr(model, "pose_dropout_p", 0.0)) == 0.0,
            "E_ACTIVE_MODEL", "pose_dropout_p")
    require(float(getattr(model, "_part_grad_scale", 0.0)) == 0.0,
            "E_ACTIVE_MODEL", "part_grad_scale")
    require(not bool(getattr(model, "use_target_heatmap", False)),
            "E_ACTIVE_MODEL", "use_target_heatmap")
    require(not bool(getattr(local_cfg.TEST, "FLIP_TEST", False)),
            "E_ACTIVE_CONFIG", "FLIP_TEST")
    require(not bool(local_cfg.TEST.RE_RANKING), "E_ACTIVE_CONFIG", "RE_RANKING")
    require(not bool(getattr(local_cfg.TEST, "NFC", False)), "E_ACTIVE_CONFIG", "NFC")
    require(float(getattr(local_cfg.TEST, "POWER_NORM", 0.0)) == 0.0,
            "E_ACTIVE_CONFIG", "POWER_NORM")
    require(str(local_cfg.TEST.NECK_FEAT) == "before", "E_ACTIVE_CONFIG", "NECK_FEAT")
    require(str(local_cfg.TEST.FEAT_NORM).lower() == "yes", "E_ACTIVE_CONFIG", "FEAT_NORM")


def strict_load_checkpoint(
    model: PoseBackboneModel,
    path: Path,
    expected_sha256: str,
) -> Dict[str, object]:
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256,
            "E_CHECKPOINT_SHA", str(path))
    state = _normalized_checkpoint_state(_torch_load_checkpoint_bytes(payload))
    result = model.load_state_dict(state, strict=True)
    require(not result.missing_keys and not result.unexpected_keys,
            "E_CHECKPOINT_STRICT", str(result))
    return checkpoint_alias_state_audit_from_state(state)


@contextlib.contextmanager
def audit_override_context(model: PoseBackboneModel):
    require(not model._audit_scene_override_enabled, "E_OVERRIDE_CONTEXT", "already enabled")
    model._audit_scene_override_enabled = True
    try:
        yield
    finally:
        model._audit_scene_override_enabled = False


class PSGInputCapture:
    def __init__(self, model: PoseBackboneModel):
        self.values: Dict[str, List[torch.Tensor]] = {"s3_b0": [], "s3_b1": []}
        self.handles = []
        for key in self.values:
            encoder = model.psg_modules_dict[key].encoder
            self.handles.append(encoder.register_forward_pre_hook(self._hook(key)))

    def _hook(self, key: str):
        def capture(_module, inputs):
            require(len(inputs) == 1 and torch.is_tensor(inputs[0]),
                    "E_HOOK_INPUT", key)
            value = inputs[0]
            require(value.dtype == torch.float32, "E_HOOK_DTYPE", key)
            require(value.ndim == 4 and value.shape[1:] == (17, 12, 4),
                    "E_HOOK_SHAPE", f"{key}: {tuple(value.shape)}")
            self.values[key].append(value.detach().clone())
        return capture

    def reset(self) -> None:
        for values in self.values.values():
            values.clear()

    def pop(self, expected_calls: int) -> torch.Tensor | None:
        for key, values in self.values.items():
            require(len(values) == expected_calls, "E_HOOK_COUNT", f"{key}: {len(values)}")
        if expected_calls == 0:
            self.reset()
            return None
        first = self.values["s3_b0"][0]
        second = self.values["s3_b1"][0]
        require(torch.equal(first, second), "E_HOOK_BLOCK_DRIFT", "s3_b0 != s3_b1")
        self.reset()
        return first

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class PreparedSceneAccess:
    def __init__(self, prepared: Path, num_query: int):
        self.prepared = prepared
        self.num_query = num_query
        self.correct_query = np.load(prepared / "query_scene_heatmaps.npy", mmap_mode="r")
        self.correct_gallery = np.load(prepared / "gallery_scene_heatmaps.npy", mmap_mode="r")
        self.query_mappings = np.load(prepared / "query_mappings.npy", mmap_mode="r")
        self.gallery_mappings = np.load(prepared / "gallery_mappings.npy", mmap_mode="r")
        self.total = len(self.correct_query) + len(self.correct_gallery)
        require(len(self.correct_query) == num_query, "E_SCENE_QUERY_COUNT", "")
        require(self.query_mappings.shape == (20, len(self.correct_query)),
                "E_MAPPING_SHAPE", str(self.query_mappings.shape))
        require(self.gallery_mappings.shape == (20, len(self.correct_gallery)),
                "E_MAPPING_SHAPE", str(self.gallery_mappings.shape))
        centroid_query = prepared / "query_centroid_heatmaps.npy"
        centroid_gallery = prepared / "gallery_centroid_heatmaps.npy"
        self.centroid_query = np.load(centroid_query, mmap_mode="r") if centroid_query.is_file() else None
        self.centroid_gallery = np.load(centroid_gallery, mmap_mode="r") if centroid_gallery.is_file() else None

    def _correct(self, rows: np.ndarray) -> np.ndarray:
        output = np.empty((len(rows), 17, 96, 32), dtype=np.float32)
        query_mask = rows < self.num_query
        if bool(query_mask.any()):
            output[query_mask] = self.correct_query[rows[query_mask]]
        if bool((~query_mask).any()):
            output[~query_mask] = self.correct_gallery[rows[~query_mask] - self.num_query]
        return output

    def donor_indices(self, mapping_index: int) -> np.ndarray:
        require(0 <= mapping_index < 20, "E_MAPPING_INDEX", str(mapping_index))
        query = np.asarray(self.query_mappings[mapping_index], dtype=np.int64)
        gallery = np.asarray(self.gallery_mappings[mapping_index], dtype=np.int64) + self.num_query
        return np.concatenate([query, gallery])

    def scenes_for_rows(self, rows: np.ndarray, row: Mapping[str, object]) -> np.ndarray | None:
        arm = str(row["arm"])
        if arm == "bypass":
            return None
        correct = self._correct(rows)
        if arm == "correct":
            return correct
        if arm == "centroid":
            require(self.centroid_query is not None and self.centroid_gallery is not None,
                    "E_CENTROID_INVALID", "prepared centroid cache absent")
            output = np.empty_like(correct)
            query_mask = rows < self.num_query
            if bool(query_mask.any()):
                output[query_mask] = self.centroid_query[rows[query_mask]]
            if bool((~query_mask).any()):
                output[~query_mask] = self.centroid_gallery[rows[~query_mask] - self.num_query]
            return output
        mapping_index = int(row["mapping"])
        donors = self.donor_indices(mapping_index)[rows]
        donor_scene = self._correct(donors)
        if arm == "shuffle":
            return donor_scene
        if arm == "group":
            return replace_group_channels(
                torch.from_numpy(correct),
                torch.from_numpy(donor_scene),
                str(row["group"]),
            ).numpy()
        raise GateProtocolError("E_ARM_NAME", arm)


def schedule_arm_id(row: Mapping[str, object]) -> str:
    seed = int(row["seed"])
    arm = str(row["arm"])
    suffix = ""
    if arm == "correct":
        suffix = f"_{row['position']}"
    elif arm == "shuffle":
        suffix = f"_m{int(row['mapping']):02d}"
    elif arm == "group":
        suffix = f"_{row['group']}_m{int(row['mapping']):02d}"
    return f"seed_{seed}__{arm}{suffix}"


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = canonical_json_bytes({"dtype": str(value.dtype), "shape": list(value.shape)})
    return sha256_bytes(header + value.tobytes(order="C"))


def seam_preflight(
    model: PoseBackboneModel,
    capture: PSGInputCapture,
    loader: DataLoader,
    scenes: PreparedSceneAccess,
    device: torch.device,
) -> Dict[str, object]:
    batch = next(iter(loader))
    images, _pids, _camids, camids_tensor, viewids, _paths, pose_dict = batch
    images = images.to(device)
    camids_tensor = camids_tensor.to(device)
    viewids = viewids.to(device)
    pose_device = _pose_to_device(pose_dict, str(device))
    cached = torch.from_numpy(scenes._correct(np.arange(len(images), dtype=np.int64))).to(device)
    prepared, _scores, _target, _difference = model._prepare_pose(pose_device)
    require(torch.equal(cached, prepared), "E_CACHE_PREPARE_SEAM", "first batch")
    model.eval()
    capture.reset()
    with torch.no_grad():
        legacy_output = model(
            images,
            cam_label=camids_tensor,
            view_label=viewids,
            pose_dict=pose_device,
        )
    require(isinstance(legacy_output, tuple) and len(legacy_output) == 2,
            "E_MODEL_OUTPUT", type(legacy_output).__name__)
    legacy = legacy_output[0]
    require(torch.is_tensor(legacy), "E_MODEL_OUTPUT", type(legacy).__name__)
    legacy_actual = capture.pop(expected_calls=1)
    capture.reset()
    with audit_override_context(model), torch.no_grad():
        override_output = model(
            images,
            cam_label=camids_tensor,
            view_label=viewids,
            pose_dict=None,
            scene_heatmaps_override=cached.contiguous(),
        )
    require(isinstance(override_output, tuple) and len(override_output) == 2,
            "E_MODEL_OUTPUT", type(override_output).__name__)
    override = override_output[0]
    require(torch.is_tensor(override), "E_MODEL_OUTPUT", type(override).__name__)
    override_actual = capture.pop(expected_calls=1)
    require(torch.equal(legacy, override), "E_OVERRIDE_SEAM_DESCRIPTOR", "first batch")
    require(torch.equal(legacy_actual, override_actual), "E_OVERRIDE_SEAM_HOOK", "first batch")
    require(torch.equal(override_actual, actual_psg_input(cached, (12, 4))),
            "E_HOOK_OVERRIDE_DRIFT", "first batch")
    return {
        "descriptor_sha256": sha256_tensor(legacy),
        "actual_psg_input_sha256": sha256_tensor(legacy_actual),
        "batch_size": len(images),
    }


def _metric_payload(
    features: torch.Tensor,
    pids: Sequence[int],
    camids: Sequence[int],
    num_query: int,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    require(features.ndim == 2 and features.shape[1] == 768,
            "E_DESCRIPTOR_SHAPE", str(features.shape))
    require(bool(torch.isfinite(features).all()), "E_DESCRIPTOR_NONFINITE", "")
    raw_sha = sha256_tensor(features)
    normalized = F.normalize(features.float(), p=2, dim=1)
    require(bool(torch.isfinite(normalized).all()), "E_DESCRIPTOR_NONFINITE", "normalized")
    normalized_sha = sha256_tensor(normalized)
    qf, gf = normalized[:num_query], normalized[num_query:]
    distmat = euclidean_distance(qf, gf)
    distance_sha = _array_sha256(distmat)
    metrics = per_query_metrics(
        distmat,
        pids[:num_query], pids[num_query:],
        camids[:num_query], camids[num_query:],
    )
    repo_cmc, repo_map = eval_func(
        distmat,
        np.asarray(pids[:num_query]), np.asarray(pids[num_query:]),
        np.asarray(camids[:num_query]), np.asarray(camids[num_query:]),
    )
    require(abs(float(repo_map) - float(metrics["mAP"])) <= 1e-12,
            "E_METRIC_PARITY", "mAP")
    # Repository CMC explicitly casts query indicators to float32 before
    # averaging; the protocol keeps per-query indicators in float64.
    require(abs(float(repo_cmc[0]) - float(metrics["R1"])) <= 1e-7,
            "E_METRIC_PARITY", "R1")
    summary = {
        "mAP": 100.0 * float(metrics["mAP"]),
        "R1": 100.0 * float(metrics["R1"]),
        "R5": 100.0 * float(metrics["R5"]),
        "R10": 100.0 * float(metrics["R10"]),
        "raw_descriptor_sha256": raw_sha,
        "normalized_descriptor_sha256": normalized_sha,
        "distance_sha256": distance_sha,
        "descriptor_shape": list(features.shape),
        "distance_shape": list(distmat.shape),
        "distance_dtype": str(distmat.dtype),
    }
    arrays = {
        "AP": np.asarray(metrics["AP"], dtype=np.float64),
        "R1_indicator": np.asarray(metrics["R1_indicator"], dtype=np.float64),
        "margin": np.asarray(metrics["margin"], dtype=np.float64),
    }
    del normalized, qf, gf, distmat
    return summary, arrays


def _intervention_split_summary(
    relative: Sequence[np.ndarray],
    displacement: Sequence[np.ndarray],
    pair_digest: "hashlib._Hash",
) -> Dict[str, object]:
    rel = np.concatenate(relative).astype(np.float64, copy=False)
    disp = np.concatenate(displacement).astype(np.float64, copy=False)
    require(rel.shape == disp.shape and rel.ndim == 1,
            "E_STRENGTH_SHAPE", f"{rel.shape}/{disp.shape}")
    require(bool(np.isfinite(rel).all()) and bool(np.isfinite(disp).all()),
            "E_STRENGTH_NONFINITE", "")
    median_relative = float(np.median(rel))
    p10_relative = float(np.quantile(rel, 0.10, method="higher"))
    median_displacement = float(np.median(disp))
    require(median_relative >= 0.10, "E_WEAK_MEDIAN_L1", str(median_relative))
    require(p10_relative >= 0.01, "E_WEAK_P10_L1", str(p10_relative))
    require(median_displacement >= 0.03, "E_WEAK_CENTROID", str(median_displacement))
    return {
        "status": "PASS",
        "count": int(len(rel)),
        "median_relative_l1": median_relative,
        "p10_relative_l1": p10_relative,
        "median_centroid_displacement": median_displacement,
        "sample_pair_sha256": pair_digest.hexdigest(),
    }


def _mapping_row_hashes(scenes: PreparedSceneAccess, row: Mapping[str, object]) -> Dict[str, str]:
    if str(row["arm"]) not in {"shuffle", "group"}:
        return {}
    index = int(row["mapping"])
    return {
        "query_mapping_sha256": _array_sha256(
            np.asarray(scenes.query_mappings[index], dtype=np.int32)),
        "gallery_mapping_sha256": _array_sha256(
            np.asarray(scenes.gallery_mappings[index], dtype=np.int32)),
    }


def expected_arm_provenance(
    manifest: Mapping[str, object],
    spec: Mapping[str, object],
    row: Mapping[str, object],
    scenes: PreparedSceneAccess,
) -> Dict[str, object]:
    correct_sraw = {
        split: {
            key: manifest["dataset"]["cache"][split]["signed_raw_audit"]
            ["actual_space"]["active_psg_blocks"][key]["sraw_sha256"]
            for key in ACTIVE_PSG_BLOCK_SHAPES
        }
        for split in ("query", "gallery")
    }
    return {
        "execution_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "checkpoint_sha256": spec["weight_sha256"],
        "checkpoint_state_audit_sha256": sha256_bytes(canonical_json_bytes(
            spec["psg_alias_audit"])),
        "config_file_sha256": manifest["config_file_sha256"],
        "resolved_config_sha256": manifest["resolved_config_sha256"],
        "prepared_artifact_manifest_sha256": sha256_bytes(canonical_json_bytes(
            manifest["prepared_artifact_sha256"])),
        "row": dict(row),
        "arm_id": schedule_arm_id(row),
        "mapping": _mapping_row_hashes(scenes, row),
        "correct_actual_sraw_sha256": correct_sraw,
    }


def _actual_input_split_sha256(actual: torch.Tensor, num_query: int) -> Dict[str, str]:
    require(actual.ndim == 4 and actual.shape[1:] == (17, 12, 4),
            "E_HOOK_SHAPE", str(actual.shape))
    require(0 < num_query < len(actual), "E_SCENE_QUERY_COUNT", str(num_query))
    output: Dict[str, str] = {}
    for split, value in (
        ("query", actual[:num_query]),
        ("gallery", actual[num_query:]),
    ):
        digest = hashlib.sha256()
        for sample in value:
            digest.update(_little_endian_float32_bytes(sample))
        output[split] = digest.hexdigest()
    return output


def _actual_input_audit_summary(
    actual: torch.Tensor | None,
    arm: str,
    total: int,
    num_query: int,
    provenance: Mapping[str, object],
) -> Dict[str, object]:
    if actual is None:
        return {
            "actual_psg_input_sha256": None,
            "actual_psg_input_shape": None,
            "actual_psg_input_split_sha256": None,
        }
    require(actual.shape == (total, 17, 12, 4),
            "E_HOOK_SHAPE", str(actual.shape))
    summary: Dict[str, object] = {
        "actual_psg_input_sha256": sha256_tensor(actual),
        "actual_psg_input_shape": list(actual.shape),
        "actual_psg_input_split_sha256": None,
    }
    if arm == "correct":
        split_hashes = _actual_input_split_sha256(actual, num_query)
        expected_hashes = provenance["correct_actual_sraw_sha256"]
        for split in ("query", "gallery"):
            for key in ACTIVE_PSG_BLOCK_SHAPES:
                require(
                    split_hashes[split] == expected_hashes[split][key],
                    "E_HOOK_PREMETRIC_DRIFT",
                    f"{split}/{key}",
                )
        summary["actual_psg_input_split_sha256"] = split_hashes
    return summary


def _audited_metric_payload(
    features: torch.Tensor,
    pids: Sequence[int],
    camids: Sequence[int],
    num_query: int,
    arm: str,
    actual: torch.Tensor | None,
    total: int,
    provenance: Mapping[str, object],
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Complete every input audit before computing any ReID metric."""

    actual_summary = _actual_input_audit_summary(
        actual, arm, total, num_query, provenance)
    summary, arrays = _metric_payload(features, pids, camids, num_query)
    summary.update(actual_summary)
    return summary, arrays


def extract_arm(
    model: PoseBackboneModel,
    capture: PSGInputCapture,
    loader: DataLoader,
    scenes: PreparedSceneAccess,
    row: Mapping[str, object],
    device: torch.device,
    frozen_metadata: Sequence[Mapping[str, object]],
    temporary: Path,
    correct_actual_reference: np.ndarray | None,
    correct_sample_hashes: Sequence[str] | None,
    provenance: Mapping[str, object],
) -> Dict[str, object]:
    """Run one frozen intervention arm and publish no data until every audit passes."""
    arm = str(row["arm"])
    features: List[torch.Tensor] = []
    actual_inputs: List[torch.Tensor] = []
    pids: List[int] = []
    camids: List[int] = []
    intervention_values = {
        "query": {"relative": [], "displacement": [], "digest": hashlib.sha256()},
        "gallery": {"relative": [], "displacement": [], "digest": hashlib.sha256()},
    }
    cursor = 0
    model.eval()
    for batch in loader:
        images, batch_pids, batch_camids, camids_tensor, viewids, paths, _pose_dict = batch
        batch_size = len(images)
        rows = np.arange(cursor, cursor + batch_size, dtype=np.int64)
        require(cursor + batch_size <= len(frozen_metadata),
                "E_RUNTIME_BATCH_COUNT", str(cursor + batch_size))
        for offset, path in enumerate(paths):
            frozen = frozen_metadata[cursor + offset]
            resolved = str(Path(path).resolve())
            require(int(frozen["combined_index"]) == cursor + offset,
                    "E_RUNTIME_LOADER_ORDER", resolved)
            require(resolved == frozen["path"], "E_RUNTIME_LOADER_PATH", resolved)
            current_identity = _file_identity(Path(resolved).lstat())
            require(current_identity == frozen["runtime_rgb_identity"],
                    "E_RUNTIME_RGB_TOCTOU", resolved)
            require(int(batch_pids[offset]) == int(frozen["pid"]),
                    "E_RUNTIME_LOADER_PID", resolved)
            require(int(batch_camids[offset]) == int(frozen["camid"]),
                    "E_RUNTIME_LOADER_CAM", resolved)
            require(int(viewids[offset]) == int(frozen["viewid"]),
                    "E_RUNTIME_LOADER_VIEW", resolved)

        source = scenes.scenes_for_rows(rows, row)
        override = None
        if source is not None:
            require(source.dtype == np.float32 and source.shape == (batch_size, 17, 96, 32),
                    "E_ARM_SCENE_SHAPE", str(source.shape))
            override = torch.from_numpy(np.ascontiguousarray(source)).to(device)
        images = images.to(device, non_blocking=False)
        camids_tensor = camids_tensor.to(device, non_blocking=False)
        viewids = viewids.to(device, non_blocking=False)
        capture.reset()
        with audit_override_context(model), torch.no_grad():
            output = model(
                images,
                cam_label=camids_tensor,
                view_label=viewids,
                pose_dict=None,
                scene_heatmaps_override=override,
            )
        require(isinstance(output, tuple) and len(output) == 2,
                "E_MODEL_OUTPUT", type(output).__name__)
        descriptor = output[0]
        require(torch.is_tensor(descriptor), "E_MODEL_OUTPUT", type(descriptor).__name__)
        require(descriptor.shape == (batch_size, 768),
                "E_DESCRIPTOR_SHAPE", str(descriptor.shape))
        require(descriptor.dtype == torch.float32,
                "E_DESCRIPTOR_DTYPE", str(descriptor.dtype))
        features.append(descriptor.detach().float().cpu())
        actual = capture.pop(expected_calls=0 if arm == "bypass" else 1)
        if arm == "bypass":
            require(actual is None, "E_BYPASS_HOOK", "PSG encoder executed")
        else:
            require(actual is not None, "E_HOOK_COUNT", arm)
            expected = actual_psg_input(override, (12, 4))
            require(torch.equal(actual, expected), "E_HOOK_OVERRIDE_DRIFT", arm)
            actual_cpu = actual.cpu()
            actual_inputs.append(actual_cpu)
            if arm == "shuffle":
                require(correct_actual_reference is not None and correct_sample_hashes is not None,
                        "E_CORRECT_REFERENCE", "shuffle")
                correct_actual = torch.from_numpy(
                    np.asarray(correct_actual_reference[cursor:cursor + batch_size]))
                values = intervention_strength(correct_actual, actual_cpu)
                for offset in range(batch_size):
                    donor_hash = sha256_tensor(actual_cpu[offset])
                    correct_hash = str(correct_sample_hashes[cursor + offset])
                    require(correct_hash != donor_hash, "E_WEAK_IDENTICAL", str(cursor + offset))
                    split = "query" if cursor + offset < scenes.num_query else "gallery"
                    digest = intervention_values[split]["digest"]
                    digest.update(bytes.fromhex(correct_hash))
                    digest.update(bytes.fromhex(donor_hash))
                for split, selector in (
                    ("query", rows < scenes.num_query),
                    ("gallery", rows >= scenes.num_query),
                ):
                    if bool(selector.any()):
                        intervention_values[split]["relative"].append(
                            values["relative_l1"][selector])
                        intervention_values[split]["displacement"].append(
                            values["centroid_displacement"][selector])

        pids.extend(int(value) for value in batch_pids)
        camids.extend(int(value) for value in batch_camids)
        cursor += batch_size
        del images, camids_tensor, viewids, output, descriptor, override, source

    require(cursor == scenes.total == len(frozen_metadata),
            "E_RUNTIME_BATCH_COUNT", f"{cursor}/{scenes.total}/{len(frozen_metadata)}")
    intervention_audit: Dict[str, object] | None = None
    if arm == "shuffle":
        intervention_audit = {
            split: _intervention_split_summary(
                intervention_values[split]["relative"],
                intervention_values[split]["displacement"],
                intervention_values[split]["digest"],
            )
            for split in ("query", "gallery")
        }

    # Metric computation is deliberately below every intervention and actual-input gate.
    feature_tensor = torch.cat(features, dim=0)
    actual_tensor = torch.cat(actual_inputs, dim=0) if actual_inputs else None
    summary, arrays = _audited_metric_payload(
        feature_tensor,
        pids,
        camids,
        scenes.num_query,
        arm,
        actual_tensor,
        scenes.total,
        provenance,
    )
    summary.update({
        "status": "PASS",
        "row": dict(row),
        "arm_id": schedule_arm_id(row),
        "num_query": scenes.num_query,
        "num_gallery": scenes.total - scenes.num_query,
        "intervention_audit": intervention_audit,
        "mapping": _mapping_row_hashes(scenes, row),
        "provenance": dict(provenance),
    })
    np.savez(
        temporary / "per_query.npz",
        AP=arrays["AP"],
        R1_indicator=arrays["R1_indicator"],
        margin=arrays["margin"],
    )
    if arm == "correct" and row.get("position") == "start":
        require(actual_tensor is not None, "E_CORRECT_REFERENCE", "start")
        np.save(temporary / "actual_psg_input.npy", actual_tensor.numpy(), allow_pickle=False)
    atomic_write_json(temporary / "summary.json", summary)
    del feature_tensor, features, actual_inputs, actual_tensor, arrays
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def extract_and_publish_unpublished_arm(
    extract_call: Callable[[], object],
    temporary: Path,
    arm_dir: Path,
    row: Mapping[str, object],
    provenance: Mapping[str, object],
    quick_identity: Mapping[str, Tuple[int, ...]],
) -> str:
    """Publish one arm; only a centroid protocol failure is secondary-invalid."""

    try:
        extract_call()
    except GateProtocolError as error:
        if (str(row["arm"]) != "centroid"
                or error.code not in CENTROID_RUNTIME_SECONDARY_CODES):
            raise
        require(temporary.is_dir(), "E_ARM_TEMP", str(temporary))
        shutil.rmtree(temporary)
        temporary.mkdir(exist_ok=False)
        atomic_write_json(temporary / "summary.json", {
            "status": "INVALID_SECONDARY",
            "row": dict(row),
            "arm_id": schedule_arm_id(row),
            "reason": {
                "error_code": error.code,
                "message": str(error),
                "phase": "runtime_extract",
            },
            "mapping": {},
            "provenance": dict(provenance),
        })
        publish_arm(
            temporary,
            arm_dir,
            provenance,
            quick_identity=quick_identity,
            status="INVALID_SECONDARY",
        )
        return "INVALID_SECONDARY"
    publish_arm(
        temporary, arm_dir, provenance, quick_identity=quick_identity)
    return "PASS"


def read_arm_summary(
    arm_dir: Path,
    expected_provenance: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    verify_published_arm(arm_dir, expected_provenance)
    path = arm_dir / "summary.json"
    require(path.is_file(), "E_ARM_SUMMARY", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def assert_flat_log_parity(summary: Mapping[str, object], spec: Mapping[str, object]) -> None:
    parsed = spec["flat_log_metrics"]
    for metric in ("mAP", "R1"):
        actual = f"{float(summary[metric]):.1f}"
        expected = f"{float(parsed[metric]):.1f}"
        require(actual == expected, "E_FLAT_LOG_PARITY",
                f"seed={spec['seed']} {metric}: {actual}!={expected}")


def assert_correct_repeat(
    start: Mapping[str, object],
    end: Mapping[str, object],
) -> None:
    exact_keys = (
        "mAP", "R1", "R5", "R10",
        "raw_descriptor_sha256", "normalized_descriptor_sha256",
        "distance_sha256", "actual_psg_input_sha256",
        "actual_psg_input_split_sha256",
        "descriptor_shape", "distance_shape", "actual_psg_input_shape",
    )
    for key in exact_keys:
        require(start[key] == end[key], "E_CORRECT_REPEAT", key)


def _remove_stale_arm_temporaries(execution_dir: Path) -> None:
    for path in execution_dir.glob(".*.tmp-arm"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def publish_run_completion(
    execution_dir: Path,
    ordered_arm_ids: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Freeze the arm markers, then atomically publish the run marker."""
    arm_ids = [str(arm_id) for arm_id in ordered_arm_ids]
    require(len(set(arm_ids)) == len(arm_ids),
            "E_RUN_ARM_MANIFEST", "duplicate arm ids")
    arms_root = execution_dir / "arms"
    arm_manifest: Dict[str, str] = {}
    for arm_id in arm_ids:
        marker = arms_root / arm_id / "COMPLETE.json"
        require(marker.is_file(), "E_RUN_ARM_MANIFEST", arm_id)
        arm_manifest[arm_id] = sha256_file(marker)
    run_payload: Dict[str, object] = {
        "status": "PASS",
        "published_arms": len(arm_ids),
        "metrics_summarized": False,
        "arm_manifest_sha256": sha256_bytes(canonical_json_bytes(arm_manifest)),
    }
    atomic_write_json(execution_dir / "RUN_ARM_MANIFEST.json", arm_manifest)
    atomic_write_json(execution_dir / "RUN_COMPLETE.json", run_payload)
    return arm_manifest, run_payload


def verify_run_completion(
    execution_dir: Path,
    ordered_arm_ids: Sequence[str],
) -> Dict[str, str]:
    """Verify both levels of the frozen run-completion marker chain."""
    arm_ids = [str(arm_id) for arm_id in ordered_arm_ids]
    require(len(set(arm_ids)) == len(arm_ids),
            "E_RUN_ARM_MANIFEST", "duplicate arm ids")
    run_marker = execution_dir / "RUN_COMPLETE.json"
    require(run_marker.is_file(), "E_RUN_INCOMPLETE", str(run_marker))
    arm_manifest_path = execution_dir / "RUN_ARM_MANIFEST.json"
    require(arm_manifest_path.is_file(),
            "E_RUN_INCOMPLETE", str(arm_manifest_path))
    arm_manifest = json.loads(arm_manifest_path.read_text(encoding="utf-8"))
    require(isinstance(arm_manifest, dict),
            "E_RUN_ARM_MANIFEST", "manifest type")
    run_payload = json.loads(run_marker.read_text(encoding="utf-8"))
    require(run_payload == {
        "status": "PASS",
        "published_arms": len(arm_ids),
        "metrics_summarized": False,
        "arm_manifest_sha256": sha256_bytes(canonical_json_bytes(arm_manifest)),
    }, "E_RUN_INCOMPLETE", str(run_payload))
    require(set(arm_manifest) == set(arm_ids),
            "E_RUN_ARM_MANIFEST", "arm ids")
    arms_root = execution_dir / "arms"
    for arm_id, expected_sha in arm_manifest.items():
        marker = arms_root / arm_id / "COMPLETE.json"
        require(marker.is_file(), "E_RUN_ARM_MANIFEST", arm_id)
        require(sha256_file(marker) == expected_sha,
                "E_RUN_ARM_MANIFEST", arm_id)
    return {str(key): str(value) for key, value in arm_manifest.items()}


def run_phase(args: argparse.Namespace) -> None:
    os.chdir(ROOT)
    execution_dir = Path(args.execution_dir).resolve()
    _assert_not_burned_execution(execution_dir)
    require(execution_dir.is_dir(), "E_RESUME_MISSING", str(execution_dir))
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "E_DEVICE", str(device))
    with exclusive_execution_lock(execution_dir, "run"):
        manifest = verify_prepared_artifacts(execution_dir)
        local_cfg = verify_frozen_config_environment(manifest, device)
        dataset, split_datasets = direct_datasets(local_cfg)
        prepared = execution_dir / "prepared"
        relation_entry = verify_relation_runtime(
            manifest, dataset, split_datasets, prepared, "run_entry")
        specs = verify_frozen_checkpoint_specs(manifest, device)
        assert_free_space(execution_dir, int(manifest["resource"]["minimum_free_bytes"]))
        expected_schedule = core_schedule([int(spec["seed"]) for spec in specs])
        spec_by_seed = {int(spec["seed"]): spec for spec in specs}
        require(manifest["schedule"] == expected_schedule,
                "E_SCHEDULE_DRIFT", "frozen schedule")
        require(len({schedule_arm_id(row) for row in expected_schedule}) == 492,
                "E_SCHEDULE_ID", "arm ids not unique")
        _remove_stale_arm_temporaries(execution_dir)

        frozen_metadata = runtime_metadata_from_relation_snapshot(
            prepared, relation_entry)
        num_query = int(manifest["dataset"]["num_query"])
        require(num_query == len(split_datasets["query"]), "E_SCENE_QUERY_COUNT", "")
        loader = validation_loader(split_datasets, local_cfg)
        scenes = PreparedSceneAccess(prepared, num_query)
        require(scenes.total == len(frozen_metadata), "E_SCENE_TOTAL_COUNT", "")
        arms_root = execution_dir / "arms"
        arms_root.mkdir(exist_ok=True)
        expected_ids = {schedule_arm_id(row) for row in expected_schedule}
        unexpected = {path.name for path in arms_root.iterdir()} - expected_ids
        require(not unexpected, "E_ARM_UNEXPECTED", str(sorted(unexpected)))

        for spec in specs:
            seed = int(spec["seed"])
            verify_relation_identity_snapshot(relation_entry, "full")
            seed_rows = [row for row in expected_schedule if int(row["seed"]) == seed]
            require(len(seed_rows) == 164, "E_SCHEDULE_ARM_COUNT", str(seed))
            model = make_model(
                local_cfg,
                num_class=int(dataset.num_train_pids),
                camera_num=int(dataset.num_train_cams),
                view_num=int(dataset.num_train_vids),
                semantic_weight=local_cfg.MODEL.SEMANTIC_WEIGHT,
            )
            alias_audit = strict_load_checkpoint(
                model, Path(str(spec["weight"])), str(spec["weight_sha256"]))
            require(alias_audit == spec["psg_alias_audit"],
                    "E_CHECKPOINT_ALIAS_DRIFT", str(seed))
            model.to(device)
            model.eval()
            assert_isolated_psg(local_cfg, model)
            capture = PSGInputCapture(model)
            correct_actual = None
            try:
                preflight = seam_preflight(model, capture, loader, scenes, device)
                atomic_write_json(execution_dir / f"seed_{seed}_preflight.json", {
                    "status": "PASS",
                    "seed": seed,
                    "checkpoint_sha256": spec["weight_sha256"],
                    "psg_alias_audit": alias_audit,
                    "override_seam": preflight,
                })

                start_row = seed_rows[0]
                require(start_row == {"seed": seed, "arm": "correct", "position": "start"},
                        "E_SCHEDULE_ORDER", str(start_row))
                start_id = schedule_arm_id(start_row)
                start_dir = arms_root / start_id
                start_provenance = expected_arm_provenance(
                    manifest, spec, start_row, scenes)
                if verify_published_arm(start_dir, start_provenance) is None:
                    temporary = execution_dir / f".{start_id}.tmp-arm"
                    temporary.mkdir(exist_ok=False)
                    extract_arm(
                        model, capture, loader, scenes, start_row, device,
                        frozen_metadata, temporary, None, None, start_provenance,
                    )
                    publish_arm(
                        temporary, start_dir, start_provenance,
                        quick_identity=relation_entry["quick_identity"])
                start_summary = read_arm_summary(start_dir, start_provenance)
                assert_flat_log_parity(start_summary, spec)
                actual_path = start_dir / "actual_psg_input.npy"
                require(actual_path.is_file(), "E_CORRECT_REFERENCE", str(actual_path))
                correct_actual = np.load(actual_path, mmap_mode="r")
                require(correct_actual.shape == (scenes.total, 17, 12, 4) and
                        correct_actual.dtype == np.float32,
                        "E_CORRECT_REFERENCE", str(correct_actual.shape))
                require(sha256_tensor(torch.from_numpy(np.asarray(correct_actual))) ==
                        start_summary["actual_psg_input_sha256"],
                        "E_CORRECT_REFERENCE_DRIFT", str(seed))
                correct_sample_hashes = [
                    sha256_tensor(torch.from_numpy(np.asarray(correct_actual[index])))
                    for index in range(scenes.total)
                ]

                for position, row in enumerate(seed_rows[1:], start=1):
                    arm_id = schedule_arm_id(row)
                    arm_dir = arms_root / arm_id
                    provenance = expected_arm_provenance(manifest, spec, row, scenes)
                    published = verify_published_arm(arm_dir, provenance)
                    if published is None:
                        temporary = execution_dir / f".{arm_id}.tmp-arm"
                        temporary.mkdir(exist_ok=False)
                        if (str(row["arm"]) == "centroid" and
                                manifest["centroid"]["status"] != "PASS"):
                            atomic_write_json(temporary / "summary.json", {
                                "status": "INVALID_SECONDARY",
                                "row": dict(row),
                                "arm_id": arm_id,
                                "reason": manifest["centroid"],
                                "mapping": {},
                                "provenance": provenance,
                            })
                            publish_arm(
                                temporary, arm_dir, provenance,
                                quick_identity=relation_entry["quick_identity"],
                                status="INVALID_SECONDARY")
                        else:
                            extract_and_publish_unpublished_arm(
                                lambda: extract_arm(
                                    model, capture, loader, scenes, row, device,
                                    frozen_metadata, temporary, correct_actual,
                                    correct_sample_hashes, provenance,
                                ),
                                temporary,
                                arm_dir,
                                row,
                                provenance,
                                relation_entry["quick_identity"],
                            )
                    verify_published_arm(arm_dir, provenance)
                    atomic_write_json(execution_dir / "RUN_PROGRESS.json", {
                        "status": "RUNNING",
                        "seed": seed,
                        "completed_arm_index_within_seed": position,
                        "completed_arm_id": arm_id,
                    })

                end_dir = arms_root / schedule_arm_id(seed_rows[-1])
                end_provenance = expected_arm_provenance(
                    manifest, spec, seed_rows[-1], scenes)
                end_summary = read_arm_summary(end_dir, end_provenance)
                assert_correct_repeat(start_summary, end_summary)
                assert_flat_log_parity(end_summary, spec)
                verify_relation_identity_snapshot(relation_entry, "full")
            finally:
                capture.close()
                if correct_actual is not None:
                    _close_memmap(correct_actual)
                del model
                gc.collect()
                torch.cuda.empty_cache()

        relation_tail = verify_relation_runtime(
            manifest, dataset, split_datasets, prepared, "run_tail")
        require(
            relation_tail["relation_report_sha256"] ==
            relation_entry["relation_report_sha256"],
            "E_RELATION_RUNTIME_DRIFT", "run_tail/entry",
        )
        for row in expected_schedule:
            spec = spec_by_seed[int(row["seed"])]
            provenance = expected_arm_provenance(manifest, spec, row, scenes)
            require(verify_published_arm(
                arms_root / schedule_arm_id(row), provenance) is not None,
                    "E_ARM_MISSING", schedule_arm_id(row))
        verify_relation_identity_snapshot(relation_tail, "quick")
        publish_run_completion(
            execution_dir,
            [schedule_arm_id(row) for row in expected_schedule],
        )


def load_per_query(
    arm_dir: Path,
    num_query: int,
    expected_provenance: Mapping[str, object] | None = None,
) -> Dict[str, np.ndarray]:
    marker = verify_published_arm(arm_dir, expected_provenance)
    require(marker is not None and marker["status"] == "PASS",
            "E_ARM_STATUS", str(arm_dir))
    path = arm_dir / "per_query.npz"
    require(path.is_file(), "E_PER_QUERY_MISSING", str(path))
    with np.load(path, allow_pickle=False) as payload:
        require(set(payload.files) == {"AP", "R1_indicator", "margin"},
                "E_PER_QUERY_KEYS", str(payload.files))
        arrays = {
            key: np.asarray(payload[key], dtype=np.float64).copy()
            for key in payload.files
        }
    for key, value in arrays.items():
        require(value.shape == (num_query,), "E_PER_QUERY_SHAPE", f"{key}/{value.shape}")
        require(bool(np.isfinite(value).all()), "E_PER_QUERY_NONFINITE", key)
    return arrays


def _mcse_summary(values: np.ndarray) -> Dict[str, float]:
    aggregated = aggregate_mapping_queries(values)
    mcse = aggregated["mcse"]
    return {
        "median": float(np.median(mcse)),
        "p95": float(np.quantile(mcse, 0.95, method="higher")),
    }


def verify_results_directory(results_dir: Path) -> Dict[str, object]:
    require(results_dir.is_dir(), "E_RESULTS_MISSING", str(results_dir))
    marker_path = results_dir / "COMPLETE.json"
    require(marker_path.is_file(), "E_RESULTS_INCOMPLETE", str(results_dir))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    require(marker.get("status") == "PASS", "E_RESULTS_INCOMPLETE", str(marker))
    require(arm_artifact_hashes(results_dir) == marker["artifact_sha256"],
            "E_RESULTS_HASH", str(results_dir))
    return marker


def publish_or_verify_results(
    execution_dir: Path,
    results: Mapping[str, object],
    aggregate_arrays: Mapping[str, np.ndarray],
) -> Dict[str, str]:
    """Publish results once, or verify an identical crash-surviving publish."""
    results_dir = execution_dir / "results"
    temporary = execution_dir / ".results.tmp"
    if temporary.exists():
        if temporary.is_dir():
            shutil.rmtree(temporary)
        else:
            temporary.unlink()
    if results_dir.exists():
        results_marker = verify_results_directory(results_dir)
        frozen_results = json.loads(
            (results_dir / "gate_a_results.json").read_text(encoding="utf-8"))
        normalized_results = json.loads(canonical_json_bytes(results))
        require(frozen_results == normalized_results,
                "E_RESULTS_DRIFT", "gate_a_results.json")
        with np.load(
            results_dir / "primary_query_aggregates.npz",
            allow_pickle=False,
        ) as payload:
            require(set(payload.files) == set(aggregate_arrays),
                    "E_RESULTS_DRIFT", "aggregate keys")
            for key, expected in aggregate_arrays.items():
                require(np.array_equal(payload[key], expected),
                        "E_RESULTS_DRIFT", key)
        return {
            str(key): str(value)
            for key, value in results_marker["artifact_sha256"].items()
        }

    temporary.mkdir(exist_ok=False)
    atomic_write_json(temporary / "gate_a_results.json", results)
    np.savez(temporary / "primary_query_aggregates.npz", **dict(aggregate_arrays))
    result_hashes = arm_artifact_hashes(temporary)
    atomic_write_json(temporary / "COMPLETE.json", {
        "status": "PASS",
        "artifact_sha256": result_hashes,
    })
    publish_directory(temporary, results_dir)
    published_marker = verify_results_directory(results_dir)
    require(published_marker["artifact_sha256"] == result_hashes,
            "E_RESULTS_HASH", str(results_dir))
    return result_hashes


def archive_transient_state(execution_dir: Path) -> None:
    candidates = [
        path for path in (execution_dir / "FAILED.json", execution_dir / "RUN_PROGRESS.json")
        if path.exists()
    ]
    if not candidates:
        return
    history = execution_dir / "state_history"
    history.mkdir(exist_ok=True)
    for source in candidates:
        index = 1
        while True:
            destination = history / f"{source.stem}_{index:04d}{source.suffix}"
            if not destination.exists():
                break
            index += 1
        os.replace(source, destination)
    fsync_directory(history)
    fsync_directory(execution_dir)


def summarize_phase(args: argparse.Namespace) -> None:
    os.chdir(ROOT)
    execution_dir = Path(args.execution_dir).resolve()
    _assert_not_burned_execution(execution_dir)
    require(execution_dir.is_dir(), "E_RESUME_MISSING", str(execution_dir))
    with exclusive_execution_lock(execution_dir, "summarize"):
        manifest = verify_prepared_artifacts(execution_dir)
        device = torch.device(args.device)
        require(device.type == "cuda" and torch.cuda.is_available(), "E_DEVICE", str(device))
        local_cfg = verify_frozen_config_environment(manifest, device)
        dataset, split_datasets = direct_datasets(local_cfg)
        prepared = execution_dir / "prepared"
        relation_entry = verify_relation_runtime(
            manifest, dataset, split_datasets, prepared, "summarize_entry")
        verified_specs = verify_frozen_checkpoint_specs(manifest, device)
        schedule = manifest["schedule"]
        require(schedule == core_schedule([
            int(spec["seed"]) for spec in manifest["checkpoints"]
        ]), "E_SCHEDULE_DRIFT", "summarize")
        arms_root = execution_dir / "arms"
        verify_run_completion(
            execution_dir,
            [schedule_arm_id(row) for row in schedule],
        )
        num_query = int(manifest["dataset"]["num_query"])
        scenes = PreparedSceneAccess(prepared, num_query)
        query_metadata = json.loads(
            (prepared / "query_metadata.json").read_text(encoding="utf-8"))
        query_pids = np.asarray([int(row["pid"]) for row in query_metadata], dtype=np.int64)
        require(query_pids.shape == (num_query,), "E_QUERY_PID_SHAPE", str(query_pids.shape))

        correct_ap: Dict[int, np.ndarray] = {}
        correct_r1: Dict[int, np.ndarray] = {}
        control_ap: Dict[str, Dict[int, np.ndarray]] = {"shuffle": {}, "bypass": {}}
        control_r1: Dict[str, Dict[int, np.ndarray]] = {"shuffle": {}, "bypass": {}}
        shuffle_by_seed: Dict[int, Dict[str, np.ndarray]] = {}
        mcse_report: Dict[int, Dict[str, object]] = {}
        weak_audits: Dict[int, Dict[int, object]] = {}
        correct_summaries: Dict[int, Dict[str, object]] = {}
        bypass_summaries: Dict[int, Dict[str, object]] = {}
        seeds = [int(spec["seed"]) for spec in verified_specs]
        spec_by_seed = {int(spec["seed"]): spec for spec in verified_specs}

        def provenance_for(seed: int, row: Mapping[str, object]) -> Dict[str, object]:
            return expected_arm_provenance(
                manifest, spec_by_seed[seed], row, scenes)

        for spec in verified_specs:
            seed = int(spec["seed"])
            start_row = {"seed": seed, "arm": "correct", "position": "start"}
            end_row = {"seed": seed, "arm": "correct", "position": "end"}
            start_dir = arms_root / f"seed_{seed}__correct_start"
            end_dir = arms_root / f"seed_{seed}__correct_end"
            start_provenance = provenance_for(seed, start_row)
            end_provenance = provenance_for(seed, end_row)
            start_summary = read_arm_summary(start_dir, start_provenance)
            end_summary = read_arm_summary(end_dir, end_provenance)
            assert_correct_repeat(start_summary, end_summary)
            assert_flat_log_parity(start_summary, spec)
            correct_summaries[seed] = start_summary
            correct_arrays = load_per_query(start_dir, num_query, start_provenance)
            correct_ap[seed] = correct_arrays["AP"]
            correct_r1[seed] = correct_arrays["R1_indicator"]

            shuffle_arrays = {key: [] for key in ("AP", "R1_indicator", "margin")}
            weak_audits[seed] = {}
            for mapping_index in range(20):
                arm_dir = arms_root / f"seed_{seed}__shuffle_m{mapping_index:02d}"
                row = {"seed": seed, "arm": "shuffle", "mapping": mapping_index}
                provenance = provenance_for(seed, row)
                arrays = load_per_query(arm_dir, num_query, provenance)
                for key in shuffle_arrays:
                    shuffle_arrays[key].append(arrays[key])
                summary = read_arm_summary(arm_dir, provenance)
                audit = summary.get("intervention_audit")
                require(isinstance(audit, Mapping), "E_WEAK_AUDIT_MISSING", str(arm_dir))
                for split, expected_count in (
                    ("query", num_query),
                    ("gallery", int(manifest["dataset"]["num_gallery"])),
                ):
                    require(audit[split]["status"] == "PASS" and
                            int(audit[split]["count"]) == expected_count,
                            "E_WEAK_AUDIT_MISSING", f"{arm_dir}/{split}")
                weak_audits[seed][mapping_index] = audit
            stacked = {
                key: np.stack(values).astype(np.float64, copy=False)
                for key, values in shuffle_arrays.items()
            }
            shuffle_by_seed[seed] = stacked
            control_ap["shuffle"][seed] = aggregate_mapping_queries(stacked["AP"])["mean"]
            control_r1["shuffle"][seed] = aggregate_mapping_queries(
                stacked["R1_indicator"])["mean"]
            mcse_report[seed] = {
                "AP": _mcse_summary(stacked["AP"]),
                "R1_indicator": _mcse_summary(stacked["R1_indicator"]),
                "margin": _mcse_summary(stacked["margin"]),
            }

            bypass_dir = arms_root / f"seed_{seed}__bypass"
            bypass_row = {"seed": seed, "arm": "bypass"}
            bypass_provenance = provenance_for(seed, bypass_row)
            bypass_summaries[seed] = read_arm_summary(bypass_dir, bypass_provenance)
            bypass_arrays = load_per_query(bypass_dir, num_query, bypass_provenance)
            control_ap["bypass"][seed] = bypass_arrays["AP"]
            control_r1["bypass"][seed] = bypass_arrays["R1_indicator"]

        map_family = simultaneous_intervals(correct_ap, control_ap, query_pids)
        r1_family = simultaneous_intervals(correct_r1, control_r1, query_pids)
        decision = gate_decision(map_family, r1_family, audits_passed=True)

        loo_shuffle: List[float] = []
        bypass_theta = float(map_family["intervals"]["bypass"]["estimate"])
        for omitted in range(20):
            per_seed = []
            keep = [index for index in range(20) if index != omitted]
            for seed in seeds:
                reduced = shuffle_by_seed[seed]["AP"][keep].mean(axis=0)
                per_seed.append(100.0 * float(np.mean(correct_ap[seed] - reduced)))
            loo_shuffle.append(float(np.mean(per_seed)))
        leave_one_mapping_out = {
            "shuffle": {
                "min": float(min(loo_shuffle)),
                "max": float(max(loo_shuffle)),
                "by_omitted_mapping": loo_shuffle,
            },
            "bypass": {"min": bypass_theta, "max": bypass_theta},
        }

        centroid_report: Dict[int, object] = {}
        for seed in seeds:
            arm_dir = arms_root / f"seed_{seed}__centroid"
            row = {"seed": seed, "arm": "centroid"}
            provenance = provenance_for(seed, row)
            marker = verify_published_arm(arm_dir, provenance)
            require(marker is not None, "E_ARM_MISSING", str(arm_dir))
            if marker["status"] == "INVALID_SECONDARY":
                centroid_report[seed] = read_arm_summary(arm_dir, provenance)
            else:
                arrays = load_per_query(arm_dir, num_query, provenance)
                centroid_report[seed] = {
                    "status": "PASS",
                    "mAP_effect_pp": 100.0 * float(np.mean(correct_ap[seed] - arrays["AP"])),
                    "R1_effect_pp": 100.0 * float(
                        np.mean(correct_r1[seed] - arrays["R1_indicator"])),
                    "arm_summary": read_arm_summary(arm_dir, provenance),
                }

        group_report: Dict[str, object] = {}
        for group in ANATOMICAL_GROUPS:
            per_seed_report = {}
            map_effects = []
            r1_effects = []
            for seed in seeds:
                ap = []
                r1 = []
                for mapping_index in range(20):
                    arm_dir = arms_root / (
                        f"seed_{seed}__group_{group}_m{mapping_index:02d}")
                    row = {
                        "seed": seed, "arm": "group", "group": group,
                        "mapping": mapping_index,
                    }
                    arrays = load_per_query(
                        arm_dir, num_query, provenance_for(seed, row))
                    ap.append(arrays["AP"])
                    r1.append(arrays["R1_indicator"])
                ap_mean = aggregate_mapping_queries(np.stack(ap))["mean"]
                r1_mean = aggregate_mapping_queries(np.stack(r1))["mean"]
                map_effect = 100.0 * float(np.mean(correct_ap[seed] - ap_mean))
                r1_effect = 100.0 * float(np.mean(correct_r1[seed] - r1_mean))
                map_effects.append(map_effect)
                r1_effects.append(r1_effect)
                per_seed_report[seed] = {
                    "mAP_effect_pp": map_effect,
                    "R1_effect_pp": r1_effect,
                }
            group_report[group] = {
                "mean_mAP_effect_pp": float(np.mean(map_effects)),
                "mean_R1_effect_pp": float(np.mean(r1_effects)),
                "per_seed": per_seed_report,
            }

        results = {
            "schema": "exp374-gate-a-results-v1",
            "status": "COMPLETE",
            "scope": "LEGACY_FROZEN_FUEL_SCREEN",
            "decision": decision,
            "primary": {
                "mAP": map_family,
                "R1": r1_family,
                "correct_arm_summaries": correct_summaries,
                "bypass_arm_summaries": bypass_summaries,
                "shuffle_mcse": mcse_report,
                "leave_one_mapping_out": leave_one_mapping_out,
            },
            "audits": {
                "prepared_artifact_sha256": manifest["prepared_artifact_sha256"],
                "split_relations": {
                    "relation_report_sha256": manifest["dataset"]
                    ["split_relations"]["relation_report_sha256"],
                    "artifact": manifest["dataset"]["split_relations_artifact"],
                },
                "weak_intervention": weak_audits,
                "all_primary_audits_passed": True,
            },
            "secondary": {
                "centroid": centroid_report,
                "anatomical_groups": group_report,
            },
            "interpretation_limit": (
                "Gate A 仅筛查冻结历史 PSG checkpoint 对匹配图像—姿态对应的依赖；"
                "GO 只授权 Gate B 设计审查，不授权新机制训练。"
            ),
        }
        aggregate_arrays = {}
        for seed in seeds:
            aggregate_arrays[f"seed_{seed}_correct_AP"] = correct_ap[seed]
            aggregate_arrays[f"seed_{seed}_correct_R1"] = correct_r1[seed]
            aggregate_arrays[f"seed_{seed}_shuffle_AP"] = control_ap["shuffle"][seed]
            aggregate_arrays[f"seed_{seed}_shuffle_R1"] = control_r1["shuffle"][seed]
            aggregate_arrays[f"seed_{seed}_bypass_AP"] = control_ap["bypass"][seed]
            aggregate_arrays[f"seed_{seed}_bypass_R1"] = control_r1["bypass"][seed]
        relation_pre_results = verify_relation_runtime(
            manifest, dataset, split_datasets, prepared, "summarize_pre_results")
        require(
            relation_pre_results["relation_report_sha256"] ==
            relation_entry["relation_report_sha256"],
            "E_RELATION_RUNTIME_DRIFT", "summarize_pre_results/entry",
        )
        verify_run_completion(
            execution_dir, [schedule_arm_id(row) for row in schedule])
        for row in schedule:
            seed = int(row["seed"])
            provenance = provenance_for(seed, row)
            require(verify_published_arm(
                arms_root / schedule_arm_id(row), provenance) is not None,
                    "E_ARM_MISSING", schedule_arm_id(row))
        result_hashes = publish_or_verify_results(
            execution_dir, results, aggregate_arrays)
        verify_relation_identity_snapshot(relation_pre_results, "quick")
        _verify_relation_artifact_triple(manifest, prepared)
        result_marker = verify_results_directory(execution_dir / "results")
        require(result_marker["artifact_sha256"] == result_hashes,
                "E_RESULTS_HASH", "pre-COMPLETE")
        archive_transient_state(execution_dir)
        atomic_write_json(execution_dir / "COMPLETE", {
            "status": "COMPLETE",
            "decision": decision["decision"],
            "results_artifact_sha256": result_hashes,
        })
        print(json.dumps({
            "status": "COMPLETE",
            "decision": decision["decision"],
            "execution_dir": str(execution_dir),
        }, ensure_ascii=False, indent=2), flush=True)


def _write_failure_manifest(args: argparse.Namespace, error: BaseException) -> None:
    execution_value = getattr(args, "execution_dir", None)
    if execution_value is None:
        return
    execution_dir = Path(execution_value).resolve()
    if _is_burned_execution(execution_dir):
        return
    if not execution_dir.is_dir():
        return
    if (execution_dir / "COMPLETE").exists():
        return
    code = error.code if isinstance(error, GateProtocolError) else "E_INTERNAL"
    atomic_write_json(execution_dir / "FAILED.json", {
        "status": "FAILED_NONREPORTABLE",
        "phase": getattr(args, "phase", None),
        "error_code": code,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "failed_arm_published": False,
        "previously_published_arms_reusable_after_hash_verification": True,
    })


def main() -> None:
    args = parse_args()
    try:
        if args.phase == "prepare":
            prepare_phase(args)
        elif args.phase == "run":
            run_phase(args)
        elif args.phase == "summarize":
            summarize_phase(args)
        else:
            raise GateProtocolError("E_PHASE", str(args.phase))
    except BaseException as error:
        _write_failure_manifest(args, error)
        raise


if __name__ == "__main__":
    main()
