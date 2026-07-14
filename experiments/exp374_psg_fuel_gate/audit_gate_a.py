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
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

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
    SceneRecord,
    absolute_centroid_targets,
    actual_psg_input,
    aggregate_mapping_queries,
    atomic_write_bytes,
    atomic_write_json,
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
        "flat_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed1234/test_default/test_log.txt",
        "train_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed1234/train_log.txt",
        "expected_mAP": 58.3,
        "expected_R1": 68.1,
    },
    {
        "seed": 42,
        "weight": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed42/transformer_120.pth",
        "weight_sha256": "174e8f9316f60219cbeca292457bf976e73cc88df6fddf9d83f94a89280d2a75",
        "flat_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed42/test_default/test_log.txt",
        "train_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed42/train_log.txt",
        "expected_mAP": 57.5,
        "expected_R1": 66.7,
    },
    {
        "seed": 2024,
        "weight": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed2024/transformer_120.pth",
        "weight_sha256": "c525e9c1ba90d896b703f6eca9a117ba1a97cd08fbab02618021bf20efd09f3d",
        "flat_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed2024/test_default/test_log.txt",
        "train_log": "/home/afr/SOLIDER-REID/log/multiseed/exp007_psg_seed2024/train_log.txt",
        "expected_mAP": 58.0,
        "expected_R1": 68.4,
    },
)


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
        normalized = str(key).removeprefix("module.")
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


def _pose_asset_identity(dataset: PoseImageDataset, image_path: str,
                         file_hash_cache: MutableMapping[str, str]) -> Tuple[str, str]:
    entry = dataset.index.get(Path(image_path).name)
    require(entry is not None, "E_POSE_INDEX_MISSING", image_path)
    person_paths = [
        str((Path(value) if os.path.isabs(value) else Path(dataset.pose_dir) / value).resolve())
        for value in entry.get("persons", [])
    ]
    require(person_paths, "E_POSE_PERSONS_EMPTY", image_path)
    for person_path in person_paths:
        require(Path(person_path).is_file(), "E_POSE_ASSET_MISSING", person_path)
        if person_path not in file_hash_cache:
            file_hash_cache[person_path] = sha256_file(Path(person_path))
    path_sha = sha256_bytes(canonical_json_bytes(person_paths))
    content_sha = sha256_bytes(canonical_json_bytes([
        file_hash_cache[person_path] for person_path in person_paths
    ]))
    return path_sha, content_sha


def _new_memmap(path: Path, shape: Tuple[int, ...], dtype: str):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


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
    cursor = 0
    expected_records = list(dataset.dataset)
    basenames = [Path(record[0]).name for record in expected_records]
    require(len(set(basenames)) == len(basenames), "E_RGB_BASENAME_COLLISION", split)

    for batch in split_loader(dataset, local_cfg):
        _images, pids, camids, _camids_tensor, viewids, paths, pose_dict = batch
        scene, scene_scores, _target, _difference = PoseBackboneModel._prepare_pose(pose_dict)
        batch_size = scene.shape[0]
        require(tuple(scene.shape[1:]) == (17, 96, 32), "E_CACHE_SCENE_SHAPE", str(scene.shape))
        heatmaps[cursor:cursor + batch_size] = scene.numpy().astype(np.float32, copy=False)
        scores[cursor:cursor + batch_size] = scene_scores.numpy().astype(np.float32, copy=False)
        for offset in range(batch_size):
            row = cursor + offset
            expected_path, expected_pid, expected_camid, expected_viewid = expected_records[row]
            actual_path = str(Path(paths[offset]).resolve())
            require(actual_path == str(Path(expected_path).resolve()), "E_LOADER_PATH_ORDER", actual_path)
            require(int(pids[offset]) == int(expected_pid), "E_LOADER_PID_ORDER", actual_path)
            require(int(camids[offset]) == int(expected_camid), "E_LOADER_CAM_ORDER", actual_path)
            require(int(viewids[offset]) == int(expected_viewid), "E_LOADER_VIEW_ORDER", actual_path)
            rgb_sha = sha256_file(Path(actual_path))
            pose_path_sha, pose_content_sha = _pose_asset_identity(
                dataset, actual_path, file_hash_cache)
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
            metadata.append({
                "index": row,
                "split": split,
                "path": actual_path,
                "rgb_sha256": rgb_sha,
                "pose_path_sha256": pose_path_sha,
                "pose_content_sha256": pose_content_sha,
                "pid": int(pids[offset]),
                "camid": int(camids[offset]),
                "viewid": int(viewids[offset]),
                "person_count": int(pose_dict["num_persons"][offset]),
                "frame": frame,
                "report": report,
            })
        cursor += batch_size
    require(cursor == count, "E_CACHE_COUNT", f"{split}: {cursor}!={count}")
    heatmaps.flush()
    scores.flush()
    continuous.flush()
    atomic_write_json(destination / f"{split}_metadata.json", metadata)
    return {
        "count": count,
        "heatmaps": heatmap_path.name,
        "scores": score_path.name,
        "continuous": f"{split}_continuous.npy",
        "metadata": f"{split}_metadata.json",
    }


def load_scene_records(prepared: Path, split: str) -> List[SceneRecord]:
    metadata = json.loads((prepared / f"{split}_metadata.json").read_text(encoding="utf-8"))
    continuous = np.load(prepared / f"{split}_continuous.npy", mmap_mode="r")
    require(continuous.shape == (len(metadata), 95), "E_NUISANCE_MATRIX", split)
    return [SceneRecord(
        index=int(row["index"]),
        split=str(row["split"]),
        path=str(row["path"]),
        rgb_sha256=str(row["rgb_sha256"]),
        pose_path_sha256=str(row["pose_path_sha256"]),
        pose_content_sha256=str(row["pose_content_sha256"]),
        pid=int(row["pid"]),
        camid=int(row["camid"]),
        person_count=int(row["person_count"]),
        continuous=tuple(float(value) for value in continuous[index]),
        frame=int(row["frame"]),
        report=dict(row["report"]),
    ) for index, row in enumerate(metadata)]


def assert_disjoint_records(groups: Mapping[str, Sequence[SceneRecord]]) -> None:
    keys = ("path", "rgb_sha256", "pose_path_sha256", "pose_content_sha256")
    names = list(groups)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            for key in keys:
                left = {getattr(record, key) for record in groups[first]}
                right = {getattr(record, key) for record in groups[second]}
                require(not left.intersection(right), "E_SPLIT_CONTENT_OVERLAP",
                        f"{first}/{second}/{key}")


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
    targets = fit_normalized_centroid_targets(
        torch.from_numpy(np.asarray(train[index])) for index in range(int(counts["train"])))
    require(all(target is not None for target in targets), "E_CENTROID_TARGET", "missing train joint")
    atomic_write_json(prepared / "centroid_targets.json", targets)
    status: Dict[str, object] = {"status": "PASS", "targets": targets}
    for split in ("query", "gallery"):
        source = np.load(prepared / f"{split}_scene_heatmaps.npy", mmap_mode="r")
        output = _new_memmap(
            prepared / f"{split}_centroid_heatmaps.npy",
            tuple(source.shape),
            "float32",
        )
        for index in range(source.shape[0]):
            scene = torch.from_numpy(np.asarray(source[index]))
            absolute = absolute_centroid_targets(scene, targets)
            output[index] = apply_scene_centroid_control(scene, absolute).numpy()
        output.flush()
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
    specs = checkpoint_specs(args.checkpoint_manifest)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".exp374_prepare_", dir=str(output_root)))
    execution_dir: Path | None = None
    try:
        dataset, datasets = direct_datasets(local_cfg)
        cache_manifest = {
            split: cache_split(split, split_dataset, local_cfg, staging)
            for split, split_dataset in datasets.items()
        }
        records = {
            split: load_scene_records(staging, split)
            for split in ("train", "query", "gallery")
        }
        assert_disjoint_records(records)
        require(torch.cuda.is_available(), "E_MATCH_GPU_REQUIRED", "exact sparse candidate build")
        device = torch.device("cuda:0")
        mapping_manifest = {}
        for split in ("query", "gallery"):
            payload = prepare_split_mappings(
                records[split], device=device, anchor_chunk=int(args.anchor_chunk))
            mapping_manifest[split] = save_mapping_payload(staging, split, payload)
            del payload
            gc.collect()
            torch.cuda.empty_cache()

        try:
            centroid_status = build_centroid_cache(
                staging,
                {split: int(cache_manifest[split]["count"]) for split in cache_manifest},
            )
        except GateProtocolError as error:
            for partial in staging.glob("*_centroid_heatmaps.npy"):
                partial.unlink()
            centroid_status = {
                "status": "INVALID",
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
        resume_path = Path(args.resume).resolve() if args.resume else None
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
    expected_sha = execution_dir.name.removeprefix("gate_a_")
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


def verify_frozen_runtime(
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
    return local_cfg, sorted(specs, key=lambda value: int(value["seed"]))


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


def verify_runtime_datasets(
    split_datasets: Mapping[str, PoseImageDataset],
    prepared: Path,
) -> List[Dict[str, object]]:
    file_hash_cache: Dict[str, str] = {}
    rgb_stats: Dict[str, Dict[str, int]] = {}
    by_split = {
        split: json.loads(
            (prepared / f"{split}_metadata.json").read_text(encoding="utf-8"))
        for split in ("train", "query", "gallery")
    }
    for split, dataset in split_datasets.items():
        frozen = by_split[split]
        require(len(dataset.dataset) == len(frozen), "E_RUNTIME_DATASET_COUNT", split)
        for index, (record, row) in enumerate(zip(dataset.dataset, frozen)):
            path, pid, camid, viewid = record
            resolved = str(Path(path).resolve())
            require(index == int(row["index"]), "E_RUNTIME_DATASET_ORDER", split)
            require(resolved == row["path"], "E_RUNTIME_DATASET_PATH", resolved)
            require(int(pid) == int(row["pid"]), "E_RUNTIME_DATASET_PID", resolved)
            require(int(camid) == int(row["camid"]), "E_RUNTIME_DATASET_CAM", resolved)
            require(int(viewid) == int(row["viewid"]), "E_RUNTIME_DATASET_VIEW", resolved)
            stat_before = Path(resolved).stat()
            require(sha256_file(Path(resolved)) == row["rgb_sha256"],
                    "E_RUNTIME_RGB_SHA", resolved)
            stat = Path(resolved).stat()
            require((stat_before.st_dev, stat_before.st_ino, stat_before.st_size,
                     stat_before.st_mtime_ns) ==
                    (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
                    "E_RUNTIME_RGB_TOCTOU", resolved)
            rgb_stats[resolved] = {
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
            pose_path_sha, pose_content_sha = _pose_asset_identity(
                dataset, resolved, file_hash_cache)
            require(pose_path_sha == row["pose_path_sha256"],
                    "E_RUNTIME_POSE_PATH_SHA", resolved)
            require(pose_content_sha == row["pose_content_sha256"],
                    "E_RUNTIME_POSE_CONTENT_SHA", resolved)
    assert_disjoint_records({
        split: load_scene_records(prepared, split)
        for split in ("train", "query", "gallery")
    })
    combined = combined_metadata(prepared)
    for row in combined:
        row["runtime_rgb_stat"] = rgb_stats[str(row["path"])]
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
    }


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
            stat = Path(resolved).stat()
            current_stat = {
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
            require(current_stat == frozen["runtime_rgb_stat"],
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

    # Metric computation is deliberately below the complete shuffle intervention gate.
    feature_tensor = torch.cat(features, dim=0)
    summary, arrays = _metric_payload(feature_tensor, pids, camids, scenes.num_query)
    actual_tensor = torch.cat(actual_inputs, dim=0) if actual_inputs else None
    if actual_tensor is not None:
        require(actual_tensor.shape == (scenes.total, 17, 12, 4),
                "E_HOOK_SHAPE", str(actual_tensor.shape))
        summary["actual_psg_input_sha256"] = sha256_tensor(actual_tensor)
        summary["actual_psg_input_shape"] = list(actual_tensor.shape)
    else:
        summary["actual_psg_input_sha256"] = None
        summary["actual_psg_input_shape"] = None
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


def run_phase(args: argparse.Namespace) -> None:
    os.chdir(ROOT)
    execution_dir = Path(args.execution_dir).resolve()
    require(execution_dir.is_dir(), "E_RESUME_MISSING", str(execution_dir))
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "E_DEVICE", str(device))
    with exclusive_execution_lock(execution_dir, "run"):
        manifest = verify_prepared_artifacts(execution_dir)
        assert_free_space(execution_dir, int(manifest["resource"]["minimum_free_bytes"]))
        local_cfg, specs = verify_frozen_runtime(manifest, device)
        expected_schedule = core_schedule([int(spec["seed"]) for spec in specs])
        spec_by_seed = {int(spec["seed"]): spec for spec in specs}
        require(manifest["schedule"] == expected_schedule,
                "E_SCHEDULE_DRIFT", "frozen schedule")
        require(len({schedule_arm_id(row) for row in expected_schedule}) == 492,
                "E_SCHEDULE_ID", "arm ids not unique")
        _remove_stale_arm_temporaries(execution_dir)

        dataset, split_datasets = direct_datasets(local_cfg)
        prepared = execution_dir / "prepared"
        frozen_metadata = verify_runtime_datasets(split_datasets, prepared)
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
            frozen_metadata = verify_runtime_datasets(split_datasets, prepared)
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
                    publish_arm(temporary, start_dir, start_provenance)
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
                                status="INVALID_SECONDARY")
                        else:
                            extract_arm(
                                model, capture, loader, scenes, row, device,
                                frozen_metadata, temporary, correct_actual,
                                correct_sample_hashes, provenance,
                            )
                            publish_arm(temporary, arm_dir, provenance)
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
                verify_runtime_datasets(split_datasets, prepared)
            finally:
                capture.close()
                del model
                gc.collect()
                torch.cuda.empty_cache()

        for row in expected_schedule:
            spec = spec_by_seed[int(row["seed"])]
            provenance = expected_arm_provenance(manifest, spec, row, scenes)
            require(verify_published_arm(
                arms_root / schedule_arm_id(row), provenance) is not None,
                    "E_ARM_MISSING", schedule_arm_id(row))
        verify_runtime_datasets(split_datasets, prepared)
        verify_frozen_runtime(manifest, device)
        arm_manifest = {
            schedule_arm_id(row): sha256_file(
                arms_root / schedule_arm_id(row) / "COMPLETE.json")
            for row in expected_schedule
        }
        atomic_write_json(execution_dir / "RUN_ARM_MANIFEST.json", arm_manifest)
        atomic_write_json(execution_dir / "RUN_COMPLETE.json", {
            "status": "PASS",
            "published_arms": 492,
            "metrics_summarized": False,
            "arm_manifest_sha256": sha256_bytes(canonical_json_bytes(arm_manifest)),
        })


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
    require(execution_dir.is_dir(), "E_RESUME_MISSING", str(execution_dir))
    with exclusive_execution_lock(execution_dir, "summarize"):
        manifest = verify_prepared_artifacts(execution_dir)
        device = torch.device(args.device)
        require(device.type == "cuda" and torch.cuda.is_available(), "E_DEVICE", str(device))
        local_cfg, verified_specs = verify_frozen_runtime(manifest, device)
        _dataset, split_datasets = direct_datasets(local_cfg)
        verify_runtime_datasets(split_datasets, execution_dir / "prepared")
        run_marker = execution_dir / "RUN_COMPLETE.json"
        require(run_marker.is_file(), "E_RUN_INCOMPLETE", str(run_marker))
        run_payload = json.loads(run_marker.read_text(encoding="utf-8"))
        arm_manifest_path = execution_dir / "RUN_ARM_MANIFEST.json"
        require(arm_manifest_path.is_file(), "E_RUN_INCOMPLETE", str(arm_manifest_path))
        arm_manifest = json.loads(arm_manifest_path.read_text(encoding="utf-8"))
        require(run_payload == {
            "status": "PASS",
            "published_arms": 492,
            "metrics_summarized": False,
            "arm_manifest_sha256": sha256_bytes(canonical_json_bytes(arm_manifest)),
        }, "E_RUN_INCOMPLETE", str(run_payload))
        schedule = manifest["schedule"]
        require(schedule == core_schedule([
            int(spec["seed"]) for spec in manifest["checkpoints"]
        ]), "E_SCHEDULE_DRIFT", "summarize")
        arms_root = execution_dir / "arms"
        require(set(arm_manifest) == {schedule_arm_id(row) for row in schedule},
                "E_RUN_ARM_MANIFEST", "arm ids")
        for arm_id, expected_sha in arm_manifest.items():
            require(sha256_file(arms_root / arm_id / "COMPLETE.json") == expected_sha,
                    "E_RUN_ARM_MANIFEST", arm_id)
        num_query = int(manifest["dataset"]["num_query"])
        scenes = PreparedSceneAccess(execution_dir / "prepared", num_query)
        query_metadata = json.loads(
            (execution_dir / "prepared/query_metadata.json").read_text(encoding="utf-8"))
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
        results_dir = execution_dir / "results"
        temporary = execution_dir / ".results.tmp"
        aggregate_arrays = {}
        for seed in seeds:
            aggregate_arrays[f"seed_{seed}_correct_AP"] = correct_ap[seed]
            aggregate_arrays[f"seed_{seed}_correct_R1"] = correct_r1[seed]
            aggregate_arrays[f"seed_{seed}_shuffle_AP"] = control_ap["shuffle"][seed]
            aggregate_arrays[f"seed_{seed}_shuffle_R1"] = control_r1["shuffle"][seed]
            aggregate_arrays[f"seed_{seed}_bypass_AP"] = control_ap["bypass"][seed]
            aggregate_arrays[f"seed_{seed}_bypass_R1"] = control_r1["bypass"][seed]
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
            with np.load(results_dir / "primary_query_aggregates.npz", allow_pickle=False) as payload:
                require(set(payload.files) == set(aggregate_arrays),
                        "E_RESULTS_DRIFT", "aggregate keys")
                for key, expected in aggregate_arrays.items():
                    require(np.array_equal(payload[key], expected),
                            "E_RESULTS_DRIFT", key)
            result_hashes = results_marker["artifact_sha256"]
        else:
            temporary.mkdir(exist_ok=False)
            atomic_write_json(temporary / "gate_a_results.json", results)
            np.savez(temporary / "primary_query_aggregates.npz", **aggregate_arrays)
            result_hashes = arm_artifact_hashes(temporary)
            atomic_write_json(temporary / "COMPLETE.json", {
                "status": "PASS",
                "artifact_sha256": result_hashes,
            })
            publish_directory(temporary, results_dir)
            published_marker = verify_results_directory(results_dir)
            require(published_marker["artifact_sha256"] == result_hashes,
                    "E_RESULTS_HASH", str(results_dir))
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
