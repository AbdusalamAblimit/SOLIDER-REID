#!/usr/bin/env python3
"""Once-only train-split measurement of the exp405 region-isolated CLIP teacher."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


EXPECTED_SAMPLES = 15618
FORMAL_DIAGNOSTIC_SAMPLES = 2000
SAMPLES_PER_SLOT = FORMAL_DIAGNOSTIC_SAMPLES // 5
VIEW_SEED = 20260720
BOOTSTRAP_SEED = 4052026
BOOTSTRAP_REPEATS = 2000
DELETION_FRACTIONS = (0.25, 0.50, 0.75)
PREFLIGHT_SAMPLES = 512
PREFLIGHT_RECIPIENTS_PER_SLOT = 4
MAX_NO_TARGET_FRACTION = 0.01
MIN_TARGET_PID_FRACTION = 0.99
MATCH_PRIMARY_CALIPER = 8.0
MATCH_PREFERENCE_LIMIT = 64
PREFLIGHT_EXECUTION = "exp405-p0b-preflight-v1"
FORMAL_EXECUTION = "exp405-p0b-iso-teacher-v1"
REMOTE_AUDIT_ROOT = Path("/home/afr/reid-clean/audits")
PREFLIGHT_OUTPUT_ROOT = REMOTE_AUDIT_ROOT / PREFLIGHT_EXECUTION
FORMAL_OUTPUT_ROOT = REMOTE_AUDIT_ROOT / FORMAL_EXECUTION
PREFLIGHT_COMPLETE_PATH = PREFLIGHT_OUTPUT_ROOT / "complete.json"
FORMAL_MANIFEST_SCHEMA = "exp405-p0b-formal-manifest-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_module(name: str, path: Path, expected_sha256: str | None = None):
    source = path.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    if expected_sha256 is not None and source_sha != expected_sha256:
        raise RuntimeError("module SHA256 mismatch: %s" % path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def write_json_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise RuntimeError("result already exists: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError("stale temporary result: %s" % temporary)
    encoded = json.dumps(
        payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def write_cache_once(path: Path, payload: dict) -> str:
    if path.exists():
        raise RuntimeError("cache already exists: %s" % path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError("stale temporary cache: %s" % temporary)
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    loaded = torch.load(temporary, map_location="cpu", weights_only=True)
    if loaded.get("schema") != payload.get("schema"):
        raise RuntimeError("cache self-check failed")
    temporary.replace(path)
    fsync_directory(path.parent)
    return sha256_file(path)


def acquire_execution_seal(output_dir: Path, execution: str, owner) -> Path:
    seal = output_dir.parent / (execution + ".started")
    seal.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(seal), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    owner._execution_seal_acquired = True
    try:
        payload = (execution + "\n").encode("utf-8")
        written = 0
        while written < len(payload):
            amount = os.write(descriptor, payload[written:])
            if amount <= 0:
                raise RuntimeError("short write while publishing execution seal")
            written += amount
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(seal.parent)
    return seal


def runtime_fingerprint() -> dict:
    package_artifacts = {}
    module_by_distribution = {
        "torch": "torch",
        "open_clip_torch": "open_clip",
        "torchvision": "torchvision",
        "numpy": "numpy",
    }
    for distribution, module_name in module_by_distribution.items():
        try:
            metadata = importlib.metadata.distribution(distribution)
            record = metadata.read_text("RECORD")
            module_spec = importlib.util.find_spec(module_name)
            module_origin = (
                Path(module_spec.origin).resolve()
                if module_spec is not None and module_spec.origin is not None
                else None
            )
            if module_origin is None or not module_origin.is_file() or record is None:
                raise RuntimeError(
                    "runtime package lacks frozen origin/RECORD bytes: %s" % distribution
                )
            package_artifacts[distribution] = {
                "version": metadata.version,
                "module_origin": str(module_origin) if module_origin is not None else None,
                "module_origin_sha256": sha256_file(module_origin),
                "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
            }
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "required runtime package is not installed: %s" % distribution
            ) from error
    try:
        nvidia = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        nvidia = []
    return {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": sha256_file(Path(sys.executable).resolve()),
            "version": sys.version,
            "build": list(platform.python_build()),
        },
        "packages": package_artifacts,
        "torch": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "config": torch.__config__.show(),
            "cuda_toolkit": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "gpu": nvidia,
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "numerics": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def failure_receipt_path(args) -> Path | None:
    """Return this invocation's failure receipt path without mutating prior state."""
    if not getattr(args, "_execution_seal_acquired", False):
        return None
    formal = args.mode == "formal"
    output_dir = Path(args.output_dir).resolve()
    expected_output_dir = FORMAL_OUTPUT_ROOT if formal else PREFLIGHT_OUTPUT_ROOT
    if output_dir != expected_output_dir:
        return None
    if (output_dir / "complete.json").is_file():
        return None
    if output_dir.is_dir():
        return output_dir / "failure.json"
    execution = FORMAL_EXECUTION if formal else PREFLIGHT_EXECUTION
    return output_dir.parent / (execution + ".failed.json")


def stable_digest(*values: object) -> int:
    payload = "\0".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def ordered_train_manifest(records, relative_paths, pose_store) -> tuple[list[dict], str]:
    rows = []
    for index, ((_, pid, camid, trackid), relative_path) in enumerate(
        zip(records, relative_paths)
    ):
        rgb_sha, image_size, _, _ = pose_store._records[relative_path]
        rows.append({
            "index": int(index),
            "relative_path": relative_path,
            "pid": int(pid),
            "camid": int(camid),
            "trackid": int(trackid),
            "image_sha256": rgb_sha,
            "image_size": list(map(int, image_size)),
        })
    return rows, canonical_sha256(rows)


def choose_preflight_indices(records, relative_paths) -> list[int]:
    ranked = sorted(
        range(len(records)),
        key=lambda index: (
            stable_digest(
                "preflight",
                int(records[index][1]),
                int(records[index][2]),
                relative_paths[index],
            ),
            relative_paths[index],
        ),
    )
    first_by_pid_camera = {}
    for index in ranked:
        key = (int(records[index][1]), int(records[index][2]))
        first_by_pid_camera.setdefault(key, index)
    diverse = sorted(
        first_by_pid_camera.values(),
        key=lambda index: (
            stable_digest("preflight-diverse", relative_paths[index]),
            relative_paths[index],
        ),
    )
    selected = diverse[:PREFLIGHT_SAMPLES]
    if len(selected) < PREFLIGHT_SAMPLES:
        used = set(selected)
        selected.extend(index for index in ranked if index not in used)
        selected = selected[:PREFLIGHT_SAMPLES]
    if len(selected) != PREFLIGHT_SAMPLES or len(set(selected)) != PREFLIGHT_SAMPLES:
        raise RuntimeError("could not freeze the preflight sample table")
    return selected


def load_and_validate_formal_manifest(
    args,
    *,
    repo_root: Path,
    core_path: Path,
    teacher_path: Path,
    pose_artifact: Path,
    clip_checkpoint: Path,
    source_commit: str,
    ordered_manifest_sha256: str,
) -> tuple[dict, str]:
    if not args.execution_manifest or not args.execution_manifest_sha256:
        raise RuntimeError("formal mode requires a frozen execution manifest and SHA256")
    manifest_path = Path(args.execution_manifest).resolve()
    expected_path = repo_root / "experiments/exp405_clip_anatomical_view_transport/phase0_p0b_formal_manifest.json"
    if manifest_path != expected_path:
        raise RuntimeError("unexpected formal manifest path")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != args.execution_manifest_sha256:
        raise RuntimeError("formal manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FORMAL_MANIFEST_SCHEMA:
        raise RuntimeError("unsupported formal manifest schema")
    if manifest.get("execution") != FORMAL_EXECUTION:
        raise RuntimeError("formal execution id mismatch")
    expected_files = {
        "runner": sha256_file(Path(__file__).resolve()),
        "core": sha256_file(core_path),
        "teacher": sha256_file(teacher_path),
        "protocol": sha256_file(repo_root / "experiments/exp405_clip_anatomical_view_transport/protocol.md"),
    }
    expected_inputs = {
        "source_commit": source_commit,
        "pose_artifact": str(pose_artifact),
        "pose_manifest_sha256": args.pose_manifest_sha256,
        "clip_checkpoint": str(clip_checkpoint),
        "clip_checkpoint_sha256": args.clip_sha256,
        "official_train_manifest_sha256": ordered_manifest_sha256,
        "preflight_complete_sha256": sha256_file(PREFLIGHT_COMPLETE_PATH),
    }
    expected_arguments = {
        "batch_size": int(args.batch_size),
        "clip_microbatch": int(args.clip_microbatch),
        "workers": int(args.workers),
        "data_root": str(Path(args.data_root).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    expected_thresholds = {
        "max_no_target_fraction": MAX_NO_TARGET_FRACTION,
        "min_target_pid_fraction": MIN_TARGET_PID_FRACTION,
        "match_primary_caliper": MATCH_PRIMARY_CALIPER,
        "match_preference_limit": MATCH_PREFERENCE_LIMIT,
    }
    current_runtime = runtime_fingerprint()
    if manifest.get("files") != expected_files:
        raise RuntimeError("formal source file binding mismatch")
    if manifest.get("inputs") != expected_inputs:
        raise RuntimeError("formal input binding mismatch")
    if manifest.get("arguments") != expected_arguments:
        raise RuntimeError("formal argument binding mismatch")
    if manifest.get("thresholds") != expected_thresholds:
        raise RuntimeError("formal scientific threshold binding mismatch")
    if manifest.get("runtime") != current_runtime:
        raise RuntimeError("formal runtime binding mismatch")
    if manifest.get("runtime_sha256") != canonical_sha256(current_runtime):
        raise RuntimeError("formal runtime freeze SHA256 mismatch")
    return manifest, manifest_sha


def validate_preflight_receipt(path: Path) -> tuple[dict, dict]:
    if path != PREFLIGHT_COMPLETE_PATH or not path.is_file():
        raise RuntimeError("formal mode requires the fixed preflight COMPLETE receipt")
    if (
        (PREFLIGHT_OUTPUT_ROOT / "failure.json").exists()
        or (PREFLIGHT_OUTPUT_ROOT.parent / (PREFLIGHT_EXECUTION + ".failed.json")).exists()
    ):
        raise RuntimeError("failed preflight cannot authorize formal measurement")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("execution") != PREFLIGHT_EXECUTION
        or receipt.get("formal") is not False
        or receipt.get("status") != "PREFLIGHT_PASS"
        or receipt.get("decision") != "PREFLIGHT_ONLY_PASS"
        or receipt.get("formal_measurement_authorized") is not True
        or receipt.get("transport_oracle_authorized") is not False
    ):
        raise RuntimeError("preflight receipt does not authorize formal measurement")
    result_path = PREFLIGHT_OUTPUT_ROOT / "result.json"
    cache_path = PREFLIGHT_OUTPUT_ROOT / "preflight_cache.pt"
    if sha256_file(result_path) != receipt.get("result_sha256"):
        raise RuntimeError("preflight result/COMPLETE digest mismatch")
    if sha256_file(cache_path) != receipt.get("cache_sha256"):
        raise RuntimeError("preflight cache/COMPLETE digest mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PREFLIGHT_PASS"
        or result.get("decision") != "PREFLIGHT_ONLY_PASS"
        or result.get("scientific_evaluated") is not False
    ):
        raise RuntimeError("preflight result is not a mechanical-only PASS")
    return receipt, result


class CanonicalTeacherDataset(Dataset):
    def __init__(self, records, pose_store, teacher_module, *, verify_rgb: bool):
        self.records = list(records)
        self.pose_store = pose_store
        self.teacher = teacher_module
        self.verify_rgb = bool(verify_rgb)
        self.relative_paths = []
        for path, _, _, _ in self.records:
            relative = Path(path).resolve().relative_to(pose_store.dataset_root).as_posix()
            if relative not in pose_store._records:
                raise RuntimeError("record missing from frozen pose artifact")
            self.relative_paths.append(relative)
        if len(set(self.relative_paths)) != len(self.relative_paths):
            raise RuntimeError("duplicate official train path")

    def __len__(self):
        return len(self.records)

    def pose_masks(self, index: int):
        path = self.records[int(index)][0]
        pose = self.pose_store.get(path, verify_image_sha=False)
        geometry = self.teacher.deterministic_geometry(pose.relative_path, VIEW_SEED)
        points, scores, valid = self.teacher.transform_pose(
            pose.keypoints, pose.scores, pose.valid, pose.image_size, geometry
        )
        masks, confidence, geometry_valid = self.teacher.render_anatomical_regions(
            points.unsqueeze(0), scores.unsqueeze(0), valid.unsqueeze(0)
        )
        return masks[0], confidence[0], geometry_valid[0], geometry

    def __getitem__(self, index: int):
        from datasets.bases import read_image
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode

        path, pid, camid, trackid = self.records[int(index)]
        image = read_image(path)
        pose = self.pose_store.get(path, verify_image_sha=self.verify_rgb)
        if image.size != pose.image_size:
            raise RuntimeError("RGB/pose image-size mismatch")
        masks, confidence, geometry_valid, geometry = self.pose_masks(index)
        image = TF.resize(
            image,
            [self.teacher.HEIGHT, self.teacher.WIDTH],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if bool(geometry["flipped"]):
            image = TF.hflip(image)
        image = TF.pad(image, 10)
        image = TF.crop(
            image,
            int(geometry["crop_top"]),
            int(geometry["crop_left"]),
            self.teacher.HEIGHT,
            self.teacher.WIDTH,
        )
        rgb = TF.to_tensor(image)
        if rgb.shape != (3, self.teacher.HEIGHT, self.teacher.WIDTH):
            raise RuntimeError("canonical RGB shape mismatch")
        return {
            "rgb": rgb,
            "masks": masks,
            "confidence": confidence,
            "geometry_valid": geometry_valid,
            "pid": int(pid),
            "camid": int(camid),
            "trackid": int(trackid),
            "path": pose.relative_path,
            "image_sha256": pose.image_sha256,
            "index": int(index),
            "flipped": bool(geometry["flipped"]),
            "crop_top": int(geometry["crop_top"]),
            "crop_left": int(geometry["crop_left"]),
        }


def collate_rows(rows: list[dict]) -> dict:
    tensor_keys = ("rgb", "masks", "confidence", "geometry_valid")
    output = {key: torch.stack([row[key] for row in rows]) for key in tensor_keys}
    for key in (
        "pid", "camid", "trackid", "index", "flipped", "crop_top", "crop_left"
    ):
        output[key] = torch.tensor([row[key] for row in rows])
    output["path"] = tuple(row["path"] for row in rows)
    output["image_sha256"] = tuple(row["image_sha256"] for row in rows)
    return output


class DiagnosticPairDataset(Dataset):
    def __init__(self, base: CanonicalTeacherDataset, indices, wrong_mask_indices, slots):
        self.base = base
        self.indices = [int(value) for value in indices]
        self.wrong_mask_indices = [int(value) for value in wrong_mask_indices]
        self.slots = [int(value) for value in slots]
        if not (len(self.indices) == len(self.wrong_mask_indices) == len(self.slots)):
            raise ValueError("diagnostic map length mismatch")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[int(position)]
        row = self.base[index]
        donor_masks, _, donor_valid, _ = self.base.pose_masks(
            self.wrong_mask_indices[int(position)]
        )
        slot = self.slots[int(position)]
        if not bool(donor_valid[slot]):
            raise RuntimeError("wrong-mask donor slot became invalid")
        row["target_slot"] = slot
        row["wrong_mask"] = donor_masks[slot]
        row["wrong_mask_index"] = self.wrong_mask_indices[int(position)]
        return row


def collate_diagnostics(rows: list[dict]) -> dict:
    output = collate_rows(rows)
    output["target_slot"] = torch.tensor([row["target_slot"] for row in rows])
    output["wrong_mask"] = torch.stack([row["wrong_mask"] for row in rows])
    output["wrong_mask_index"] = torch.tensor(
        [row["wrong_mask_index"] for row in rows]
    )
    return output


def slot_semantics(visual: torch.Tensor, slots: torch.Tensor, teacher) -> dict[str, torch.Tensor]:
    visual = F.normalize(visual.float(), dim=-1)
    logits = teacher.logit_scale * (visual @ teacher.part_text.float().T)
    distribution = logits.softmax(-1)
    row = torch.arange(len(slots), device=slots.device)
    visible = teacher.visible_text.float().index_select(0, slots)
    occluded = teacher.occluded_text.float().index_select(0, slots)
    support = torch.sigmoid(
        teacher.logit_scale
        * ((visual * visible).sum(-1) - (visual * occluded).sum(-1))
    )
    correct = distribution[row, slots]
    wrong = distribution.masked_fill(
        F.one_hot(slots, num_classes=distribution.shape[1]).bool(), -1.0
    ).amax(-1)
    margin = correct - wrong
    return {
        "distribution": distribution,
        "support": support,
        "margin": margin,
        "correct_probability": correct,
    }


def summarize_vector(values: torch.Tensor) -> dict[str, float | int | None]:
    vector = values.detach().cpu().double().numpy()
    if len(vector) == 0:
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "count": int(len(vector)),
        "mean": float(vector.mean()),
        "p05": float(np.quantile(vector, 0.05)),
        "p50": float(np.quantile(vector, 0.50)),
        "p95": float(np.quantile(vector, 0.95)),
    }


def pid_bootstrap_interval(values: torch.Tensor, pids: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().double()
    pids = pids.detach().cpu().long()
    unique = torch.unique(pids, sorted=True)
    if len(values) == 0 or len(unique) < 2:
        raise ValueError("PID bootstrap requires at least two identities")
    pid_means = torch.stack([values[pids == pid].mean() for pid in unique])
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    draws = torch.empty(BOOTSTRAP_REPEATS, dtype=torch.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled = torch.randint(len(pid_means), (len(pid_means),), generator=generator)
        draws[repeat] = pid_means.index_select(0, sampled).mean()
    return {
        "mean": float(pid_means.mean()),
        "lower_95": float(torch.quantile(draws, 0.025)),
        "upper_95": float(torch.quantile(draws, 0.975)),
        "pids": int(len(unique)),
        "repeats": BOOTSTRAP_REPEATS,
    }


def pid_effect_summary(values: torch.Tensor, pids: torch.Tensor) -> dict:
    values = values.detach().cpu().double()
    pids = pids.detach().cpu().long()
    unique = torch.unique(pids, sorted=True)
    if len(values) == 0 or len(unique) < 2:
        raise ValueError("PID effect summary requires at least two identities")
    pid_means = torch.stack([values[pids == pid].mean() for pid in unique])
    return {
        "sample": summarize_vector(values),
        "pid_bootstrap": pid_bootstrap_interval(values, pids),
        "pid_sign": {
            "positive": int((pid_means > 0).sum()),
            "zero": int((pid_means == 0).sum()),
            "negative": int((pid_means < 0).sum()),
        },
    }


def conservative_two_cluster_summary(
    values: torch.Tensor,
    recipient_pids: torch.Tensor,
    donor_pids: torch.Tensor,
) -> dict:
    recipient = pid_effect_summary(values, recipient_pids)
    donor = pid_effect_summary(values, donor_pids)
    values = values.detach().cpu().double()
    recipient_pids = recipient_pids.detach().cpu().long()
    donor_pids = donor_pids.detach().cpu().long()
    recipient_unique, recipient_inverse = torch.unique(
        recipient_pids, sorted=True, return_inverse=True
    )
    donor_unique, donor_inverse = torch.unique(
        donor_pids, sorted=True, return_inverse=True
    )
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED + 17)
    draws = torch.empty(BOOTSTRAP_REPEATS, dtype=torch.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        for _ in range(32):
            recipient_count = torch.bincount(
                torch.randint(
                    len(recipient_unique),
                    (len(recipient_unique),),
                    generator=generator,
                ),
                minlength=len(recipient_unique),
            ).double()
            donor_count = torch.bincount(
                torch.randint(
                    len(donor_unique),
                    (len(donor_unique),),
                    generator=generator,
                ),
                minlength=len(donor_unique),
            ).double()
            weights = (
                recipient_count.index_select(0, recipient_inverse)
                * donor_count.index_select(0, donor_inverse)
            )
            if float(weights.sum()) > 0:
                draws[repeat] = (values * weights).sum() / weights.sum()
                break
        else:
            raise RuntimeError("two-way PID bootstrap produced an empty resample")
    two_way = {
        "mean": float(values.mean()),
        "lower_95": float(torch.quantile(draws, 0.025)),
        "upper_95": float(torch.quantile(draws, 0.975)),
        "recipient_pids": int(len(recipient_unique)),
        "donor_pids": int(len(donor_unique)),
        "repeats": BOOTSTRAP_REPEATS,
    }
    return {
        "recipient_pid": recipient,
        "donor_pid": donor,
        "two_way_pid_bootstrap": two_way,
        "conservative_lower_95": min(
            recipient["pid_bootstrap"]["lower_95"],
            donor["pid_bootstrap"]["lower_95"],
            two_way["lower_95"],
        ),
        "conservative_upper_95": max(
            recipient["pid_bootstrap"]["upper_95"],
            donor["pid_bootstrap"]["upper_95"],
            two_way["upper_95"],
        ),
    }


def non_torso_macro_effect(
    values: torch.Tensor,
    slots: torch.Tensor,
    pids: torch.Tensor,
) -> dict:
    values = values.detach().cpu().double()
    slots = slots.detach().cpu().long()
    pids = pids.detach().cpu().long()
    slot_pid_means = {}
    slot_pid_indices = {}
    non_torso = (slots == 0) | (slots == 3) | (slots == 4)
    global_pids = torch.unique(pids[non_torso], sorted=True)
    global_pid_index = {int(pid): index for index, pid in enumerate(global_pids)}
    for slot in (0, 3, 4):
        active = slots == slot
        unique = torch.unique(pids[active], sorted=True)
        if len(unique) < 2:
            raise ValueError("each non-torso slot requires at least two identities")
        slot_pid_means[slot] = torch.stack([
            values[active & (pids == pid)].mean() for pid in unique
        ])
        slot_pid_indices[slot] = torch.tensor([
            global_pid_index[int(pid)] for pid in unique
        ], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED + 31)
    draws = torch.empty(BOOTSTRAP_REPEATS, dtype=torch.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        for _ in range(64):
            pid_count = torch.bincount(
                torch.randint(
                    len(global_pids), (len(global_pids),), generator=generator
                ),
                minlength=len(global_pids),
            ).double()
            slot_draws = []
            for slot in (0, 3, 4):
                weights = pid_count.index_select(0, slot_pid_indices[slot])
                if float(weights.sum()) <= 0:
                    break
                slot_draws.append(
                    (slot_pid_means[slot] * weights).sum() / weights.sum()
                )
            if len(slot_draws) == 3:
                draws[repeat] = torch.stack(slot_draws).mean()
                break
        else:
            raise RuntimeError("non-torso PID bootstrap produced an empty slot")
    cells = torch.cat([slot_pid_means[slot] for slot in (0, 3, 4)])
    return {
        "slot_equal_weight": True,
        "global_pid_cluster": True,
        "slot_pid_cells": int(len(cells)),
        "slot_pid_count": {
            str(slot): int(len(slot_pid_means[slot])) for slot in (0, 3, 4)
        },
        "pid_bootstrap": {
            "mean": float(torch.stack([
                slot_pid_means[slot].mean() for slot in (0, 3, 4)
            ]).mean()),
            "lower_95": float(torch.quantile(draws, 0.025)),
            "upper_95": float(torch.quantile(draws, 0.975)),
            "repeats": BOOTSTRAP_REPEATS,
        },
        "pid_sign": {
            "positive": int((cells > 0).sum()),
            "zero": int((cells == 0).sum()),
            "negative": int((cells < 0).sum()),
        },
    }


def non_torso_two_cluster_summary(
    values: torch.Tensor,
    slots: torch.Tensor,
    recipient_pids: torch.Tensor,
    donor_pids: torch.Tensor,
) -> dict:
    values = values.detach().cpu().double()
    slots = slots.detach().cpu().long()
    recipient_pids = recipient_pids.detach().cpu().long()
    donor_pids = donor_pids.detach().cpu().long()
    non_torso = (slots == 0) | (slots == 3) | (slots == 4)
    values = values[non_torso]
    slots = slots[non_torso]
    recipient_pids = recipient_pids[non_torso]
    donor_pids = donor_pids[non_torso]
    recipient = non_torso_macro_effect(values, slots, recipient_pids)
    donor = non_torso_macro_effect(values, slots, donor_pids)
    recipient_unique, recipient_inverse = torch.unique(
        recipient_pids, sorted=True, return_inverse=True
    )
    donor_unique, donor_inverse = torch.unique(
        donor_pids, sorted=True, return_inverse=True
    )
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED + 47)
    draws = torch.empty(BOOTSTRAP_REPEATS, dtype=torch.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        for _ in range(64):
            recipient_count = torch.bincount(
                torch.randint(
                    len(recipient_unique), (len(recipient_unique),), generator=generator
                ),
                minlength=len(recipient_unique),
            ).double()
            donor_count = torch.bincount(
                torch.randint(
                    len(donor_unique), (len(donor_unique),), generator=generator
                ),
                minlength=len(donor_unique),
            ).double()
            weights = (
                recipient_count.index_select(0, recipient_inverse)
                * donor_count.index_select(0, donor_inverse)
            )
            slot_draws = []
            for slot in (0, 3, 4):
                active = slots == slot
                denominator = weights[active].sum()
                if float(denominator) <= 0:
                    break
                slot_draws.append(
                    (values[active] * weights[active]).sum() / denominator
                )
            if len(slot_draws) == 3:
                draws[repeat] = torch.stack(slot_draws).mean()
                break
        else:
            raise RuntimeError("non-torso two-way bootstrap produced an empty slot")
    point = torch.stack([
        values[slots == slot].mean() for slot in (0, 3, 4)
    ]).mean()
    two_way = {
        "mean": float(point),
        "lower_95": float(torch.quantile(draws, 0.025)),
        "upper_95": float(torch.quantile(draws, 0.975)),
        "recipient_pids": int(len(recipient_unique)),
        "donor_pids": int(len(donor_unique)),
        "repeats": BOOTSTRAP_REPEATS,
    }
    return {
        "recipient_pid": recipient,
        "donor_pid": donor,
        "two_way_pid_bootstrap": two_way,
        "conservative_lower_95": min(
            recipient["pid_bootstrap"]["lower_95"],
            donor["pid_bootstrap"]["lower_95"],
            two_way["lower_95"],
        ),
        "conservative_upper_95": max(
            recipient["pid_bootstrap"]["upper_95"],
            donor["pid_bootstrap"]["upper_95"],
            two_way["upper_95"],
        ),
    }


def full_semantic_summary(distribution, support, valid, pids) -> dict:
    per_slot = {}
    macro_margin = []
    macro_accuracy = []
    all_margin = []
    all_margin_pids = []
    all_margin_slots = []
    for slot in range(distribution.shape[1]):
        active = valid[:, slot]
        selected = distribution[active, slot]
        labels = torch.full((len(selected),), slot, dtype=torch.long)
        row = torch.arange(len(selected))
        correct = selected[row, labels]
        wrong = selected.masked_fill(
            F.one_hot(labels, num_classes=distribution.shape[-1]).bool(), -1.0
        ).amax(-1)
        margin = correct - wrong
        rank = 1 + (selected > correct.unsqueeze(1)).sum(1)
        accuracy = (selected.argmax(1) == labels).float()
        active_pids = pids[active]
        per_slot[str(slot)] = {
            "name": (
                "head",
                "upper_torso_arms",
                "lower_torso",
                "upper_legs",
                "lower_legs_feet",
            )[slot],
            "accuracy": float(accuracy.mean()),
            "mrr": float((1.0 / rank.double()).mean()),
            "margin": summarize_vector(margin),
            "margin_pid_bootstrap": pid_bootstrap_interval(margin, active_pids),
            "margin_pid_effect": pid_effect_summary(margin, active_pids),
            "support": summarize_vector(support[active, slot]),
        }
        macro_margin.append(margin.mean())
        macro_accuracy.append(accuracy.mean())
        all_margin.append(margin)
        all_margin_pids.append(active_pids)
        all_margin_slots.append(torch.full((len(margin),), slot, dtype=torch.long))
    combined_margin = torch.cat(all_margin)
    combined_pids = torch.cat(all_margin_pids)
    return {
        "macro_accuracy": float(torch.stack(macro_accuracy).mean()),
        "macro_margin": float(torch.stack(macro_margin).mean()),
        "overall_margin_pid_bootstrap": pid_bootstrap_interval(
            combined_margin, combined_pids
        ),
        "non_torso_margin": non_torso_macro_effect(
            combined_margin,
            torch.cat(all_margin_slots),
            combined_pids,
        ),
        "per_slot": per_slot,
    }


def choose_targets(valid: torch.Tensor, sample_keys: torch.Tensor) -> torch.Tensor:
    targets = torch.full((len(valid),), -1, dtype=torch.long)
    for index in range(len(valid)):
        choices = torch.nonzero(valid[index], as_tuple=False).flatten()
        if len(choices):
            targets[index] = choices[int(sample_keys[index]) % len(choices)]
    return targets


def choose_diagnostic_subset(
    targets, valid, pids, paths, *, samples_per_slot: int = SAMPLES_PER_SLOT
) -> tuple[list[int], list[int]]:
    if int(samples_per_slot) <= 0:
        raise ValueError("samples_per_slot must be positive")
    selected_indices = []
    selected_slots = []
    for slot in range(valid.shape[1]):
        candidates = [
            index for index in range(len(targets))
            if int(targets[index]) == slot and bool(valid[index, slot])
        ]
        candidates.sort(key=lambda index: stable_digest("diagnostic", slot, pids[index], paths[index]))
        first_by_pid = {}
        for index in candidates:
            first_by_pid.setdefault(int(pids[index]), index)
        diverse = sorted(
            first_by_pid.values(),
            key=lambda index: stable_digest("pid", slot, pids[index], paths[index]),
        )
        chosen = diverse[:samples_per_slot]
        if len(chosen) < samples_per_slot:
            used = set(chosen)
            chosen.extend(index for index in candidates if index not in used)
            chosen = chosen[:samples_per_slot]
        if len(chosen) != samples_per_slot:
            raise RuntimeError("insufficient diagnostic samples for slot %d" % slot)
        selected_indices.extend(chosen)
        selected_slots.extend([slot] * len(chosen))
    return selected_indices, selected_slots


def maximum_unique_assignment(preferences) -> list[int]:
    if not preferences or any(len(row) == 0 for row in preferences):
        raise ValueError("each recipient requires at least one donor preference")
    assignment = {}
    owner = {}
    for root in range(len(preferences)):
        queue = [root]
        cursor = 0
        parent_recipient = {}
        seen_recipients = {root}
        seen_donors = set()
        free_donor = None
        terminal_recipient = None
        while cursor < len(queue) and free_donor is None:
            recipient_position = queue[cursor]
            cursor += 1
            for donor in preferences[recipient_position]:
                donor = int(donor)
                if donor in seen_donors:
                    continue
                seen_donors.add(donor)
                if donor not in owner:
                    free_donor = donor
                    terminal_recipient = recipient_position
                    break
                next_recipient = owner[donor]
                if next_recipient not in seen_recipients:
                    seen_recipients.add(next_recipient)
                    parent_recipient[next_recipient] = (recipient_position, donor)
                    queue.append(next_recipient)
        if free_donor is None:
            raise RuntimeError("no one-to-one assignment exists within frozen preferences")
        current_recipient = terminal_recipient
        current_donor = free_donor
        while True:
            assignment[current_recipient] = current_donor
            owner[current_donor] = current_recipient
            if current_recipient == root:
                break
            current_recipient, current_donor = parent_recipient[current_recipient]
    result = [assignment[position] for position in range(len(preferences))]
    if len(set(result)) != len(result):
        raise RuntimeError("assignment unexpectedly reused a donor")
    return result


def maximum_unique_assignment_with_expansion(
    preferences, *, initial_limit: int = MATCH_PREFERENCE_LIMIT
) -> tuple[list[int], int]:
    if int(initial_limit) <= 0:
        raise ValueError("initial preference limit must be positive")
    maximum = max(len(row) for row in preferences)
    limits = []
    limit = int(initial_limit)
    while limit < maximum:
        limits.append(limit)
        limit *= 2
    limits.append(maximum)
    last_error = None
    for limit in limits:
        try:
            return maximum_unique_assignment([row[:limit] for row in preferences]), limit
        except RuntimeError as error:
            last_error = error
    raise RuntimeError("no full-caliper one-to-one assignment exists") from last_error


def choose_wrong_masks(
    indices,
    slots,
    valid,
    mass,
    centroid_y,
    confidence,
    support,
    global_feature,
    pids,
    camids,
    keys,
    forbidden_donor_indices,
):
    if len(indices) != len(slots) or not indices:
        raise ValueError("wrong-mask recipients/slots must be non-empty and aligned")
    descriptor_values = {
        "mass_log": mass.clamp_min(1e-12).log().double(),
        "centroid_y": centroid_y.double(),
        "confidence": confidence.double(),
        "support": support.double(),
    }
    forbidden = torch.zeros(len(valid), dtype=torch.bool)
    forbidden[torch.tensor(sorted(set(map(int, forbidden_donor_indices))), dtype=torch.long)] = True
    slot_scales = {}
    for slot in range(valid.shape[1]):
        active = valid[:, slot]
        slot_scales[slot] = {}
        for name, values in descriptor_values.items():
            selected = values[active, slot]
            median = selected.median()
            scale = (selected - median).abs().median().clamp_min(1e-6)
            slot_scales[slot][name] = float(scale)

    preferences = []
    for recipient, slot in zip(indices, slots):
        candidate = (
            valid[:, slot]
            & (pids != pids[recipient])
            & (camids == camids[recipient])
            & ~forbidden
        )
        candidate_indices = torch.nonzero(candidate, as_tuple=False).flatten()
        if not len(candidate_indices):
            raise RuntimeError("no same-camera wrong-mask donor")
        gaps = {
            name: (values[candidate_indices, slot] - values[recipient, slot]).abs()
            for name, values in descriptor_values.items()
        }
        primary_distance = sum(
            gaps[name] / slot_scales[int(slot)][name]
            for name in descriptor_values
        )
        cosine_gap = 1.0 - (
            global_feature[candidate_indices].double()
            @ global_feature[recipient].double()
        ).clamp(-1, 1)
        feasible = torch.nonzero(
            primary_distance <= MATCH_PRIMARY_CALIPER, as_tuple=False
        ).flatten()
        if not len(feasible):
            raise RuntimeError("no wrong-mask donor satisfies the frozen balance caliper")
        feasible_candidates = candidate_indices.index_select(0, feasible)
        feasible_primary = primary_distance.index_select(0, feasible)
        feasible_cosine = cosine_gap.index_select(0, feasible)
        ranked = np.lexsort(
            (
                keys.index_select(0, feasible_candidates).cpu().numpy(),
                feasible_cosine.cpu().numpy(),
                feasible_primary.cpu().numpy(),
                (feasible_primary + feasible_cosine).cpu().numpy(),
            )
        )
        preference = feasible_candidates.cpu().numpy()[ranked].astype(np.int32, copy=False)
        preferences.append(preference)

    donors, preference_limit_used = maximum_unique_assignment_with_expansion(
        preferences
    )
    records = []
    for recipient, slot, donor in zip(indices, slots, donors):
        candidate = (
            valid[:, slot]
            & (pids != pids[recipient])
            & (camids == camids[recipient])
            & ~forbidden
        )
        candidate_indices = torch.nonzero(candidate, as_tuple=False).flatten()
        selected = torch.nonzero(
            candidate_indices == int(donor), as_tuple=False
        ).flatten()
        if len(selected) != 1:
            raise RuntimeError("assigned wrong-mask donor left the frozen candidate set")
        selected_position = int(selected.item())
        gaps = {
            name: (values[candidate_indices, slot] - values[recipient, slot]).abs()
            for name, values in descriptor_values.items()
        }
        primary_distance = sum(
            gaps[name] / slot_scales[int(slot)][name]
            for name in descriptor_values
        )
        chosen_primary = primary_distance[selected_position]
        stable_tie_before = (
            (primary_distance == chosen_primary)
            & (candidate_indices < int(donor))
        ).sum()
        primary_rank = int((primary_distance < chosen_primary).sum() + stable_tie_before)
        chosen_cosine = 1.0 - (
            global_feature[int(donor)].double()
            @ global_feature[int(recipient)].double()
        ).clamp(-1, 1)
        records.append({
            "recipient": int(recipient),
            "donor": int(donor),
            "slot": int(slot),
            "recipient_pid": int(pids[recipient]),
            "donor_pid": int(pids[donor]),
            "camera": int(camids[recipient]),
            "mass_log_gap": float(gaps["mass_log"][selected_position]),
            "centroid_y_gap": float(gaps["centroid_y"][selected_position]),
            "confidence_gap": float(gaps["confidence"][selected_position]),
            "support_gap": float(gaps["support"][selected_position]),
            "global_cosine_gap": float(chosen_cosine),
            "primary_distance": float(chosen_primary),
            "primary_rank": primary_rank,
            "candidate_count": int(len(candidate_indices)),
            "caliper_candidate_count": int(
                (primary_distance <= MATCH_PRIMARY_CALIPER).sum()
            ),
            "preference_limit_used": preference_limit_used,
        })
    if len(set(donors)) != len(donors):
        raise RuntimeError("wrong-mask donor reuse is forbidden")
    return donors, records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "formal"))
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pose-artifact", required=True)
    parser.add_argument("--pose-manifest-sha256", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--clip-sha256", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--clip-microbatch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-receipt")
    parser.add_argument("--execution-manifest")
    parser.add_argument("--execution-manifest-sha256")
    return parser.parse_args()


def adjudicate_measurement(
    *, formal: bool, validity_pass: bool, scientific_pass: bool
) -> dict:
    if formal:
        return {
            "status": "VALIDITY_PASS" if validity_pass else "VALIDITY_FAIL",
            "decision": (
                "P0B_REGION_ISOLATED_TEACHER_GO"
                if validity_pass and scientific_pass
                else "P0B_REGION_ISOLATED_TEACHER_NO_GO"
            ),
            "formal_measurement_authorization_candidate": False,
            "transport_oracle_authorization_candidate": bool(
                validity_pass and scientific_pass
            ),
        }
    return {
        "status": "PREFLIGHT_PASS" if validity_pass else "PREFLIGHT_FAIL",
        "decision": "PREFLIGHT_ONLY_PASS" if validity_pass else "PREFLIGHT_ONLY_FAIL",
        "formal_measurement_authorization_candidate": bool(validity_pass),
        "transport_oracle_authorization_candidate": False,
    }


def run(args) -> dict:
    started = time.time()
    args._execution_seal_acquired = False
    formal = args.mode == "formal"
    execution = FORMAL_EXECUTION if formal else PREFLIGHT_EXECUTION
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    pose_artifact = Path(args.pose_artifact).resolve()
    clip_checkpoint = Path(args.clip_checkpoint).resolve()
    core_path = Path(args.core).resolve()
    teacher_path = Path(args.teacher).resolve()
    output_dir = Path(args.output_dir).resolve()
    expected_output_dir = FORMAL_OUTPUT_ROOT if formal else PREFLIGHT_OUTPUT_ROOT
    seal_path = expected_output_dir.parent / (execution + ".started")
    sibling_failure_path = expected_output_dir.parent / (execution + ".failed.json")
    if data_root != Path("/mnt1/afrdata"):
        raise RuntimeError("official data root must be /mnt1/afrdata")
    if Path("/mnt1/afrderived") not in pose_artifact.parents:
        raise RuntimeError("pose artifact must remain under /mnt1/afrderived")
    if output_dir != expected_output_dir:
        raise RuntimeError("execution output must use its fixed once-only root")
    if output_dir.exists() or seal_path.exists() or sibling_failure_path.exists():
        raise RuntimeError("execution id already has an immutable terminal/start state")
    if int(args.batch_size) <= 0 or int(args.clip_microbatch) <= 0 or int(args.workers) < 0:
        raise ValueError("batch sizes must be positive")
    preflight_complete = None
    preflight_result = None
    if formal:
        if not args.preflight_receipt:
            raise RuntimeError("formal mode requires the preflight PASS receipt")
        preflight_complete, preflight_result = validate_preflight_receipt(
            Path(args.preflight_receipt).resolve()
        )
    else:
        if args.preflight_receipt or args.execution_manifest or args.execution_manifest_sha256:
            raise ValueError("preflight mode forbids formal authorization inputs")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if source_commit != args.source_commit:
        raise RuntimeError("source commit mismatch")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    ).strip():
        raise RuntimeError("tracked source is dirty")
    protocol_path = (
        repo_root / "experiments/exp405_clip_anatomical_view_transport/protocol.md"
    )
    start_files = {
        "runner": sha256_file(Path(__file__).resolve()),
        "core": sha256_file(core_path),
        "teacher": sha256_file(teacher_path),
        "protocol": sha256_file(protocol_path),
    }
    start_runtime = runtime_fingerprint()
    sys.path.insert(0, str(repo_root))
    from datasets.occluded_duke import OccludedDuke
    from datasets.pose_targets import PoseTargetStore

    dataset_object = OccludedDuke(root=str(data_root), verbose=False)
    records = list(dataset_object.train)
    expected_count = EXPECTED_SAMPLES if formal else PREFLIGHT_SAMPLES
    if len(records) != EXPECTED_SAMPLES:
        raise RuntimeError("unexpected official train size")
    pose_store = PoseTargetStore(pose_artifact, args.pose_manifest_sha256)
    if len(pose_store) != EXPECTED_SAMPLES:
        raise RuntimeError("unexpected pose artifact size")
    relative_paths = [
        Path(path).resolve().relative_to(pose_store.dataset_root).as_posix()
        for path, _, _, _ in records
    ]
    if len(set(relative_paths)) != EXPECTED_SAMPLES:
        raise RuntimeError("duplicate official train path")
    if any(path not in pose_store._records for path in relative_paths):
        raise RuntimeError("official train is not fully bound to the pose artifact")
    ordered_rows, ordered_manifest_sha = ordered_train_manifest(
        records, relative_paths, pose_store
    )
    if formal:
        execution_indices = list(range(EXPECTED_SAMPLES))
    else:
        execution_indices = choose_preflight_indices(records, relative_paths)
    execution_records = [records[index] for index in execution_indices]
    execution_relative_paths = [relative_paths[index] for index in execution_indices]
    execution_ordered_rows = [ordered_rows[index] for index in execution_indices]
    execution_manifest_sha = None
    if formal:
        execution_manifest, execution_manifest_sha = load_and_validate_formal_manifest(
            args,
            repo_root=repo_root,
            core_path=core_path,
            teacher_path=teacher_path,
            pose_artifact=pose_artifact,
            clip_checkpoint=clip_checkpoint,
            source_commit=source_commit,
            ordered_manifest_sha256=ordered_manifest_sha,
        )
        expected_preflight_provenance = {
            "source_commit": source_commit,
            "runner_sha256": start_files["runner"],
            "core_sha256": start_files["core"],
            "teacher_sha256": start_files["teacher"],
            "protocol_sha256": start_files["protocol"],
            "clip_checkpoint_sha256": args.clip_sha256,
            "pose_manifest_sha256": args.pose_manifest_sha256,
            "official_train_manifest_sha256": ordered_manifest_sha,
            "runtime": start_runtime,
        }
        if any(
            preflight_result.get("provenance", {}).get(key) != value
            for key, value in expected_preflight_provenance.items()
        ):
            raise RuntimeError("preflight provenance does not match formal inputs")
        if preflight_complete.get("source_files") != start_files:
            raise RuntimeError("preflight COMPLETE source binding mismatch")
        if preflight_complete.get("runtime_sha256") != canonical_sha256(start_runtime):
            raise RuntimeError("preflight COMPLETE runtime binding mismatch")
        core = load_module(
            "exp405_phase0_core", core_path, execution_manifest["files"]["core"]
        )
        teacher_module = load_module(
            "exp405_real_teacher", teacher_path, execution_manifest["files"]["teacher"]
        )
    else:
        core = load_module("exp405_phase0_core", core_path)
        teacher_module = load_module("exp405_real_teacher", teacher_path)
    base_dataset = CanonicalTeacherDataset(
        execution_records, pose_store, teacher_module, verify_rgb=True
    )
    if base_dataset.relative_paths != execution_relative_paths:
        raise RuntimeError("official train ordering changed during dataset construction")

    acquire_execution_seal(output_dir, execution, args)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_once(output_dir / "started.json", {
        "execution": execution,
        "formal": formal,
        "source_commit": source_commit,
        "ordered_official_train_manifest_sha256": ordered_manifest_sha,
        "execution_manifest_sha256": execution_manifest_sha,
    })
    result_path = output_dir / "result.json"
    cache_path = output_dir / ("teacher_cache.pt" if formal else "preflight_cache.pt")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    frozen_teacher = teacher_module.RegionIsolatedClipTeacher(
        clip_checkpoint,
        args.clip_sha256,
        device,
        microbatch=int(args.clip_microbatch),
    )
    loader = DataLoader(
        base_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=collate_rows,
        persistent_workers=int(args.workers) > 0,
    )
    count = len(base_dataset)
    feature_dim = int(frozen_teacher.visual.output_dim)
    global_feature = torch.empty(count, feature_dim, dtype=torch.float32)
    visual = torch.empty(count, teacher_module.REGIONS, feature_dim, dtype=torch.float32)
    distribution = torch.empty(count, teacher_module.REGIONS, teacher_module.REGIONS)
    support = torch.empty(count, teacher_module.REGIONS)
    pose_geometry_valid = torch.zeros(count, teacher_module.REGIONS, dtype=torch.bool)
    readout_valid = torch.zeros(count, teacher_module.REGIONS, dtype=torch.bool)
    patch_counts = torch.zeros(count, teacher_module.REGIONS, dtype=torch.int32)
    mass = torch.zeros(count, teacher_module.REGIONS, dtype=torch.float64)
    centroid_y = torch.zeros(count, teacher_module.REGIONS, dtype=torch.float64)
    pose_confidence = torch.zeros(count, teacher_module.REGIONS, dtype=torch.float64)
    flipped = torch.zeros(count, dtype=torch.bool)
    crop_top = torch.zeros(count, dtype=torch.int16)
    crop_left = torch.zeros(count, dtype=torch.int16)
    seen = torch.zeros(count, dtype=torch.int32)
    pids = torch.tensor([int(row[1]) for row in base_dataset.records])
    camids = torch.tensor([int(row[2]) for row in base_dataset.records])
    sample_keys = core.stable_sample_keys(base_dataset.relative_paths)

    print(json.dumps({"progress": "original_start", "rows": count}), flush=True)
    for batch_number, batch in enumerate(loader):
        indices = batch["index"].long()
        expected_paths = tuple(base_dataset.relative_paths[int(index)] for index in indices)
        if tuple(batch["path"]) != expected_paths:
            raise RuntimeError("absolute index/path mismatch")
        rgb = batch["rgb"].to(device)
        masks = batch["masks"].to(device)
        encoded = frozen_teacher.encode(rgb, masks)
        batch_pose_valid = batch["geometry_valid"].to(device)
        batch_readout_valid = encoded["readout_valid"]
        valid = batch_pose_valid & batch_readout_valid
        state = core.clip_slot_state(
            encoded["visual"],
            frozen_teacher.part_text,
            frozen_teacher.visible_text,
            frozen_teacher.occluded_text,
            valid,
            logit_scale=frozen_teacher.logit_scale,
        )
        global_feature[indices] = encoded["global"].cpu()
        visual[indices] = state["visual"].cpu()
        distribution[indices] = state["distribution"].cpu()
        support[indices] = state["support"].cpu()
        pose_geometry_valid[indices] = batch_pose_valid.cpu()
        readout_valid[indices] = batch_readout_valid.cpu()
        patch_counts[indices] = encoded["patch_selected"].sum(-1).cpu().int()
        flat = batch["masks"].double().flatten(2)
        batch_mass = flat.sum(-1)
        y = torch.arange(teacher_module.MASK_HEIGHT, dtype=torch.float64).repeat_interleave(
            teacher_module.MASK_WIDTH
        )
        mass[indices] = batch_mass
        centroid_y[indices] = (flat * y).sum(-1) / batch_mass.clamp_min(1e-12)
        centroid_y[indices] /= max(teacher_module.MASK_HEIGHT - 1, 1)
        pose_confidence[indices] = batch["confidence"].double()
        flipped[indices] = batch["flipped"].bool()
        crop_top[indices] = batch["crop_top"].short()
        crop_left[indices] = batch["crop_left"].short()
        seen[indices] += 1
        if batch_number % 100 == 0:
            print(json.dumps({"progress": "original", "batch": batch_number}), flush=True)
    if not bool((seen == 1).all()):
        raise RuntimeError("original teacher coverage is not exactly once")
    if not bool(torch.isfinite(global_feature).all() and torch.isfinite(visual).all()):
        raise RuntimeError("non-finite original teacher cache")

    analysis_valid = pose_geometry_valid & readout_valid
    if formal:
        semantic_summary = full_semantic_summary(
            distribution, support, analysis_valid, pids
        )
    else:
        semantic_summary = {
            "scientific_evaluated": False,
            "analysis_valid_count_per_slot": analysis_valid.sum(0).tolist(),
        }
    targets = choose_targets(analysis_valid, sample_keys)
    if formal:
        diagnostic_indices, diagnostic_slots = choose_diagnostic_subset(
            targets, analysis_valid, pids.tolist(), base_dataset.relative_paths
        )
    else:
        diagnostic_indices, diagnostic_slots = choose_diagnostic_subset(
            targets,
            analysis_valid,
            pids.tolist(),
            base_dataset.relative_paths,
            samples_per_slot=PREFLIGHT_RECIPIENTS_PER_SLOT,
        )
    wrong_mask_indices, wrong_mask_records = choose_wrong_masks(
        diagnostic_indices,
        diagnostic_slots,
        analysis_valid,
        mass,
        centroid_y,
        pose_confidence,
        support,
        global_feature,
        pids,
        camids,
        sample_keys,
        diagnostic_indices,
    )
    diagnostic_dataset = DiagnosticPairDataset(
        base_dataset, diagnostic_indices, wrong_mask_indices, diagnostic_slots
    )
    diagnostic_loader = DataLoader(
        diagnostic_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0 if not formal else int(args.workers),
        collate_fn=collate_diagnostics,
    )
    diagnostic = {
        "original_support": [],
        "original_margin": [],
        "wrong_mask_margin": [],
        "pid": [],
        "donor_pid": [],
        "slot": [],
    }
    deletion_hashers = {
        fraction: hashlib.sha256() for fraction in DELETION_FRACTIONS
    }
    for fraction in DELETION_FRACTIONS:
        diagnostic["support_%.2f" % fraction] = []
        diagnostic["margin_%.2f" % fraction] = []
        diagnostic["support_count_%.2f" % fraction] = []
        diagnostic["erase_count_%.2f" % fraction] = []
        diagnostic["realized_%.2f" % fraction] = []

    clip_fill = torch.tensor(teacher_module.CLIP_MEAN, device=device).view(1, 3, 1, 1)
    print(json.dumps({"progress": "diagnostic_start", "rows": len(diagnostic_dataset)}), flush=True)
    for batch_number, batch in enumerate(diagnostic_loader):
        indices = batch["index"].long()
        slots_cpu = batch["target_slot"].long()
        slots = slots_cpu.to(device)
        rows = torch.arange(len(indices))
        rgb = batch["rgb"].to(device)
        masks = batch["masks"].to(device)
        selected_mask = masks[torch.arange(len(rgb), device=device), slots]
        full_mask = F.interpolate(
            selected_mask.unsqueeze(1),
            size=(teacher_module.HEIGHT, teacher_module.WIDTH),
            mode="nearest",
        )[:, 0] > 0
        original_visual = visual[indices, slots_cpu].to(device)
        original_state = slot_semantics(original_visual, slots, frozen_teacher)
        diagnostic["original_support"].append(original_state["support"].cpu())
        diagnostic["original_margin"].append(original_state["margin"].cpu())
        diagnostic["pid"].append(pids[indices])
        diagnostic["donor_pid"].append(pids[batch["wrong_mask_index"].long()])
        diagnostic["slot"].append(slots_cpu)
        sample_key = sample_keys[indices].to(device)
        support_count = full_mask.flatten(1).sum(1).long()
        if bool((support_count < 4).any()):
            raise RuntimeError("diagnostic deletion support is too small")
        for fraction in DELETION_FRACTIONS:
            _, erased, realized = core.deterministic_slot_delete(
                rgb,
                full_mask,
                sample_key,
                slots,
                fraction=fraction,
                fill=0.0,
                ordering_seed=VIEW_SEED,
            )
            erase_count = erased.flatten(1).sum(1).long()
            expected_erase = torch.floor(
                support_count.double() * float(fraction) + 0.5
            ).long()
            exact_realized = erase_count.double() / support_count.double()
            if not torch.equal(erase_count, expected_erase):
                raise RuntimeError("deletion count violates the frozen rounding rule")
            if bool((erased & ~full_mask).any()):
                raise RuntimeError("deletion escaped the frozen slot support")
            if not torch.equal(realized.cpu(), exact_realized.cpu()):
                raise RuntimeError("reported deletion fraction is not exact")
            diagnostic["support_count_%.2f" % fraction].append(support_count.cpu())
            diagnostic["erase_count_%.2f" % fraction].append(erase_count.cpu())
            diagnostic["realized_%.2f" % fraction].append(realized.cpu())
            deletion_hashers[fraction].update(indices.numpy().tobytes())
            deletion_hashers[fraction].update(slots_cpu.numpy().tobytes())
            deletion_hashers[fraction].update(
                erased.detach().cpu().to(torch.uint8).contiguous().numpy().tobytes()
            )
            deleted = torch.where(erased.unsqueeze(1), clip_fill, rgb)
            deleted_encoded = frozen_teacher.encode_selected(deleted, masks, slots)
            if not bool(deleted_encoded["readout_valid"].all()):
                raise RuntimeError("deletion invalidated region-isolated readout")
            deleted_state = slot_semantics(
                deleted_encoded["visual"], slots, frozen_teacher
            )
            diagnostic["support_%.2f" % fraction].append(
                deleted_state["support"].cpu()
            )
            diagnostic["margin_%.2f" % fraction].append(
                deleted_state["margin"].cpu()
            )
        wrong_masks = masks.clone()
        wrong_masks[torch.arange(len(rgb), device=device), slots] = batch[
            "wrong_mask"
        ].to(device)
        wrong_encoded = frozen_teacher.encode_selected(rgb, wrong_masks, slots)
        if not bool(wrong_encoded["readout_valid"].all()):
            raise RuntimeError("wrong-mask readout invalid")
        wrong_state = slot_semantics(wrong_encoded["visual"], slots, frozen_teacher)
        diagnostic["wrong_mask_margin"].append(wrong_state["margin"].cpu())
        if batch_number % 100 == 0:
            print(json.dumps({"progress": "diagnostic", "batch": batch_number}), flush=True)
    diagnostic = {
        key: torch.cat(value) if isinstance(value, list) else value
        for key, value in diagnostic.items()
    }
    q0 = diagnostic["original_support"]
    q25 = diagnostic["support_0.25"]
    q50 = diagnostic["support_0.50"]
    q75 = diagnostic["support_0.75"]
    delta_50 = q0 - q50
    delta_25_75 = q25 - q75
    mask_margin_gap = diagnostic["original_margin"] - diagnostic["wrong_mask_margin"]
    exact_deletion = {
        "%.2f" % fraction: {
            "all_counts_exact": True,
            "support_count": summarize_vector(
                diagnostic["support_count_%.2f" % fraction].double()
            ),
            "erase_count": summarize_vector(
                diagnostic["erase_count_%.2f" % fraction].double()
            ),
            "realized": summarize_vector(diagnostic["realized_%.2f" % fraction]),
            "erased_support_sha256": deletion_hashers[fraction].hexdigest(),
        }
        for fraction in DELETION_FRACTIONS
    }
    match_fields = (
        "mass_log_gap", "centroid_y_gap", "confidence_gap", "support_gap",
        "global_cosine_gap", "primary_distance", "primary_rank",
        "candidate_count", "caliper_candidate_count", "preference_limit_used",
    )
    donor_pid_values, donor_pid_counts = torch.unique(
        diagnostic["donor_pid"], sorted=True, return_counts=True
    )
    recipient_set = {row["recipient"] for row in wrong_mask_records}
    wrong_mask_match = {
        "same_camera_all": all(
            int(camids[row["recipient"]]) == int(camids[row["donor"]])
            for row in wrong_mask_records
        ),
        "different_pid_all": all(
            row["recipient_pid"] != row["donor_pid"] for row in wrong_mask_records
        ),
        "balance_caliper_all": all(
            row["primary_distance"] <= MATCH_PRIMARY_CALIPER
            for row in wrong_mask_records
        ),
        "unique_donor_count": len({row["donor"] for row in wrong_mask_records}),
        "unique_donor_pid_count": int(len(donor_pid_values)),
        "reuse_count": len(wrong_mask_records)
        - len({row["donor"] for row in wrong_mask_records}),
        "recipient_also_used_as_donor_count": len(
            recipient_set & {row["donor"] for row in wrong_mask_records}
        ),
        "donor_pid_pair_count": summarize_vector(donor_pid_counts.double()),
        "fallback_count": 0,
        "preference_limit": MATCH_PREFERENCE_LIMIT,
        "primary_caliper": MATCH_PRIMARY_CALIPER,
        "per_slot": {},
    }
    for field in match_fields:
        wrong_mask_match[field] = summarize_vector(torch.tensor([
            row[field] for row in wrong_mask_records
        ], dtype=torch.float64))
    for slot in range(teacher_module.REGIONS):
        rows_for_slot = [row for row in wrong_mask_records if row["slot"] == slot]
        slot_donor_pids = torch.tensor([
            row["donor_pid"] for row in rows_for_slot
        ], dtype=torch.long)
        _, slot_pid_counts = torch.unique(
            slot_donor_pids, sorted=True, return_counts=True
        )
        wrong_mask_match["per_slot"][str(slot)] = {
            "count": len(rows_for_slot),
            "unique_donor_pid_count": int(torch.unique(slot_donor_pids).numel()),
            "donor_pid_pair_count": summarize_vector(slot_pid_counts.double()),
            "balance_caliper_all": all(
                row["primary_distance"] <= MATCH_PRIMARY_CALIPER
                for row in rows_for_slot
            ),
            **{
                field: summarize_vector(torch.tensor([
                    row[field] for row in rows_for_slot
                ], dtype=torch.float64))
                for field in match_fields
            },
        }
    deletion_summary = {
        "scientific_evaluated": formal,
        "sample_count": int(len(q0)),
        "support": {
            "original": summarize_vector(q0),
            "deleted_25": summarize_vector(q25),
            "deleted_50": summarize_vector(q50),
            "deleted_75": summarize_vector(q75),
        },
        "exact_deletion": exact_deletion,
        "wrong_mask_match": wrong_mask_match,
        "per_slot": {},
    }
    if formal:
        deletion_summary.update({
            "original_minus_deleted_50": pid_effect_summary(
                delta_50, diagnostic["pid"]
            ),
            "deleted_25_minus_75": pid_effect_summary(
                delta_25_75, diagnostic["pid"]
            ),
            "correct_minus_wrong_mask_margin": conservative_two_cluster_summary(
                mask_margin_gap, diagnostic["pid"], diagnostic["donor_pid"]
            ),
            "non_torso_macro": {
                "original_minus_deleted_50": non_torso_macro_effect(
                    delta_50, diagnostic["slot"], diagnostic["pid"]
                ),
                "deleted_25_minus_75": non_torso_macro_effect(
                    delta_25_75, diagnostic["slot"], diagnostic["pid"]
                ),
                "correct_minus_wrong_mask_margin": non_torso_two_cluster_summary(
                    mask_margin_gap,
                    diagnostic["slot"],
                    diagnostic["pid"],
                    diagnostic["donor_pid"],
                ),
            },
        })
    for slot in range(teacher_module.REGIONS):
        active = diagnostic["slot"] == slot
        slot_summary = {
            "count": int(active.sum()),
            "strict_mean_monotonic": bool(
                q0[active].mean() > q25[active].mean()
                and q25[active].mean() > q50[active].mean()
                and q50[active].mean() > q75[active].mean()
            ),
        }
        if formal:
            slot_summary.update({
                "support_original_minus_50": pid_effect_summary(
                    delta_50[active], diagnostic["pid"][active]
                ),
                "support_25_minus_75": pid_effect_summary(
                    delta_25_75[active], diagnostic["pid"][active]
                ),
                "correct_minus_wrong_mask_margin": conservative_two_cluster_summary(
                    mask_margin_gap[active],
                    diagnostic["pid"][active],
                    diagnostic["donor_pid"][active],
                ),
            })
        deletion_summary["per_slot"][str(slot)] = slot_summary

    cache = {
        "schema": (
            "exp405-p0b-formal-cache-v1"
            if formal else "exp405-p0b-preflight-cache-v1"
        ),
        "execution": execution,
        "formal": formal,
        "ordered_sample_manifest": execution_ordered_rows,
        "ordered_sample_manifest_sha256": canonical_sha256(execution_ordered_rows),
        "relative_paths": tuple(base_dataset.relative_paths),
        "pids": pids,
        "camids": camids,
        "trackids": torch.tensor([int(row[3]) for row in base_dataset.records]),
        "sample_keys": sample_keys,
        "flipped": flipped,
        "crop_top": crop_top,
        "crop_left": crop_left,
        "global": global_feature,
        "visual": visual,
        "distribution": distribution,
        "support": support,
        "pose_geometry_valid": pose_geometry_valid,
        "readout_valid": readout_valid,
        "analysis_valid": analysis_valid,
        "patch_counts": patch_counts,
        "mass": mass,
        "centroid_y": centroid_y,
        "pose_confidence": pose_confidence,
        "part_text": frozen_teacher.part_text.cpu(),
        "visible_text": frozen_teacher.visible_text.cpu(),
        "occluded_text": frozen_teacher.occluded_text.cpu(),
        "native_logit_scale": frozen_teacher.logit_scale,
        "targets": targets,
        "diagnostic_indices": torch.tensor(diagnostic_indices),
        "diagnostic_slots": torch.tensor(diagnostic_slots),
        "wrong_mask_indices": torch.tensor(wrong_mask_indices),
        "wrong_mask_match_records": wrong_mask_records,
        "diagnostic_pid": diagnostic["pid"],
        "diagnostic_donor_pid": diagnostic["donor_pid"],
        "diagnostic_deletion_counts": {
            "%.2f" % fraction: {
                "support_count": diagnostic["support_count_%.2f" % fraction],
                "erase_count": diagnostic["erase_count_%.2f" % fraction],
                "realized": diagnostic["realized_%.2f" % fraction],
                "erased_support_sha256": deletion_hashers[fraction].hexdigest(),
            }
            for fraction in DELETION_FRACTIONS
        },
        "provenance": {
            "source_commit": source_commit,
            "runner_sha256": start_files["runner"],
            "core_sha256": start_files["core"],
            "teacher_sha256": start_files["teacher"],
            "protocol_sha256": start_files["protocol"],
            "clip_checkpoint_sha256": args.clip_sha256,
            "pose_manifest_sha256": args.pose_manifest_sha256,
            "execution_manifest_sha256": execution_manifest_sha,
            "runtime": start_runtime,
            "runtime_sha256": canonical_sha256(start_runtime),
        },
    }
    gates = {
        "official_train_exact": count == expected_count,
        "coverage_exact_once": bool((seen == 1).all()),
        "all_outputs_finite": bool(
            torch.isfinite(global_feature).all()
            and torch.isfinite(visual).all()
            and torch.isfinite(distribution).all()
            and torch.isfinite(support).all()
        ),
        "each_slot_has_geometry": bool(pose_geometry_valid.any(0).all()),
        "each_pose_valid_slot_has_patch": bool(
            (patch_counts[pose_geometry_valid] > 0).all()
            and readout_valid[pose_geometry_valid].all()
        ),
        "exact_deletion_counts": all(
            row["all_counts_exact"]
            for row in deletion_summary["exact_deletion"].values()
        ),
        "wrong_mask_matching_strict": bool(
            deletion_summary["wrong_mask_match"]["same_camera_all"]
            and deletion_summary["wrong_mask_match"]["different_pid_all"]
            and deletion_summary["wrong_mask_match"]["balance_caliper_all"]
            and deletion_summary["wrong_mask_match"]["fallback_count"] == 0
            and deletion_summary["wrong_mask_match"]["reuse_count"] == 0
            and deletion_summary["wrong_mask_match"]["recipient_also_used_as_donor_count"] == 0
        ),
    }
    if formal:
        per_slot_semantic_positive = all(
            summary["margin_pid_bootstrap"]["lower_95"] > 0
            for summary in semantic_summary["per_slot"].values()
        )
        per_slot_deletion_positive = all(
            summary["support_original_minus_50"]["pid_bootstrap"]["lower_95"] > 0
            and summary["support_25_minus_75"]["pid_bootstrap"]["lower_95"] > 0
            and summary["correct_minus_wrong_mask_margin"]["conservative_lower_95"] > 0
            and summary["strict_mean_monotonic"]
            for summary in deletion_summary["per_slot"].values()
        )
        non_torso = deletion_summary["non_torso_macro"]
        non_torso_positive = (
            non_torso["original_minus_deleted_50"]["pid_bootstrap"]["lower_95"] > 0
            and non_torso["deleted_25_minus_75"]["pid_bootstrap"]["lower_95"] > 0
            and non_torso["correct_minus_wrong_mask_margin"]["conservative_lower_95"] > 0
        )
        target_pid_fraction = float(
            torch.unique(pids[targets >= 0]).numel() / torch.unique(pids).numel()
        )
        gates.update({
            "target_image_coverage": float((targets < 0).float().mean())
            <= MAX_NO_TARGET_FRACTION,
            "target_pid_coverage": target_pid_fraction >= MIN_TARGET_PID_FRACTION,
            "part_macro_accuracy_above_chance": semantic_summary["macro_accuracy"] > 0.20,
            "part_overall_margin_ci_positive": (
                semantic_summary["overall_margin_pid_bootstrap"]["lower_95"] > 0
            ),
            "part_each_slot_margin_positive": per_slot_semantic_positive,
            "part_non_torso_margin_ci_positive": (
                semantic_summary["non_torso_margin"]["pid_bootstrap"]["lower_95"] > 0
            ),
            "deletion_original_minus_50_ci_positive": (
                deletion_summary["original_minus_deleted_50"]["pid_bootstrap"]["lower_95"] > 0
            ),
            "deletion_25_minus_75_ci_positive": (
                deletion_summary["deleted_25_minus_75"]["pid_bootstrap"]["lower_95"] > 0
            ),
            "correct_mask_beats_wrong_mask_ci": (
                deletion_summary["correct_minus_wrong_mask_margin"]["conservative_lower_95"] > 0
            ),
            "each_slot_deletion_and_mask_direction_positive": per_slot_deletion_positive,
            "non_torso_macro_directions_positive": non_torso_positive,
        })
    validity_keys = [
        "official_train_exact",
        "coverage_exact_once",
        "all_outputs_finite",
        "each_slot_has_geometry",
        "each_pose_valid_slot_has_patch",
        "exact_deletion_counts",
        "wrong_mask_matching_strict",
    ]
    if formal:
        validity_keys.extend(("target_image_coverage", "target_pid_coverage"))
    validity_pass = all(gates[key] for key in validity_keys)
    scientific_keys = tuple(key for key in gates if key not in validity_keys)
    scientific_pass = bool(
        formal and validity_pass and all(gates[key] for key in scientific_keys)
    )
    adjudication = adjudicate_measurement(
        formal=formal,
        validity_pass=validity_pass,
        scientific_pass=scientific_pass,
    )
    decision = adjudication["decision"]
    transport_authorization_candidate = adjudication[
        "transport_oracle_authorization_candidate"
    ]
    formal_measurement_authorization_candidate = adjudication[
        "formal_measurement_authorization_candidate"
    ]
    cache["decision"] = decision
    cache_sha = write_cache_once(cache_path, cache)
    result = {
        "experiment": "exp405",
        "execution": execution,
        "status": adjudication["status"],
        "decision": decision,
        "scientific_evaluated": formal,
        "transport_oracle_authorized": False,
        "transport_oracle_authorization_candidate": transport_authorization_candidate,
        "formal_measurement_authorized": False,
        "formal_measurement_authorization_candidate": formal_measurement_authorization_candidate,
        "authorization_requires_complete_receipt": True,
        "formal": formal,
        "gates": gates,
        "semantic": semantic_summary,
        "deletion": deletion_summary,
        "coverage": {
            "pose_geometry_valid_fraction_per_slot": pose_geometry_valid.float().mean(0).tolist(),
            "readout_valid_fraction_per_slot": readout_valid.float().mean(0).tolist(),
            "analysis_valid_fraction_per_slot": analysis_valid.float().mean(0).tolist(),
            "no_valid_target_count": int((targets < 0).sum()),
            "no_valid_target_fraction": float((targets < 0).float().mean()),
            "no_valid_target_pid_count": int(torch.unique(pids[targets < 0]).numel()),
            "geometry_without_readout_count": int(
                (pose_geometry_valid & ~readout_valid).sum()
            ),
            "readout_without_geometry_count": int(
                (readout_valid & ~pose_geometry_valid).sum()
            ),
            "patch_count_per_slot": {
                str(slot): summarize_vector(
                    patch_counts[pose_geometry_valid[:, slot], slot].double()
                )
                for slot in range(teacher_module.REGIONS)
            },
        },
        "provenance": {
            "source_commit": source_commit,
            "runner_sha256": start_files["runner"],
            "core_sha256": start_files["core"],
            "teacher_sha256": start_files["teacher"],
            "protocol_sha256": start_files["protocol"],
            "clip_checkpoint_sha256": args.clip_sha256,
            "pose_manifest_sha256": args.pose_manifest_sha256,
            "official_train_manifest_sha256": ordered_manifest_sha,
            "execution_manifest_sha256": execution_manifest_sha,
            "teacher_cache_sha256": cache_sha,
            "native_logit_scale": frozen_teacher.logit_scale,
            "view_seed": VIEW_SEED,
            "taxonomy": list(teacher_module.REGION_NAMES),
            "deletion_fractions": list(DELETION_FRACTIONS),
            "runtime": start_runtime,
            "runtime_sha256": canonical_sha256(start_runtime),
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "device": str(device),
            "batch_size": int(args.batch_size),
            "clip_microbatch": int(args.clip_microbatch),
            "workers": int(args.workers),
        },
    }
    postflight_files = {
        "runner": sha256_file(Path(__file__).resolve()),
        "core": sha256_file(core_path),
        "teacher": sha256_file(teacher_path),
        "protocol": sha256_file(protocol_path),
    }
    if postflight_files != start_files:
        raise RuntimeError("source changed during teacher measurement")
    if sha256_file(clip_checkpoint) != args.clip_sha256:
        raise RuntimeError("CLIP checkpoint changed during teacher measurement")
    if sha256_file(pose_artifact / "manifest.json") != args.pose_manifest_sha256:
        raise RuntimeError("pose manifest changed during teacher measurement")
    if formal and sha256_file(Path(args.execution_manifest)) != execution_manifest_sha:
        raise RuntimeError("execution manifest changed during teacher measurement")
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip() != source_commit:
        raise RuntimeError("source commit changed during teacher measurement")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    ).strip():
        raise RuntimeError("tracked source became dirty during teacher measurement")
    if runtime_fingerprint() != start_runtime:
        raise RuntimeError("runtime freeze changed during teacher measurement")
    write_json_once(result_path, result)
    result_sha = sha256_file(result_path)
    write_json_once(output_dir / "complete.json", {
        "execution": execution,
        "formal": formal,
        "status": result["status"],
        "decision": decision,
        "result_sha256": result_sha,
        "cache_sha256": cache_sha,
        "execution_manifest_sha256": execution_manifest_sha,
        "source_files": postflight_files,
        "runtime_sha256": canonical_sha256(start_runtime),
        "formal_measurement_authorized": bool(validity_pass and not formal),
        "transport_oracle_authorized": transport_authorization_candidate,
    })
    return result


def main():
    args = parse_args()
    try:
        result = run(args)
    except BaseException as error:
        try:
            failure_path = failure_receipt_path(args)
            if failure_path is not None and not failure_path.exists():
                write_json_once(failure_path, {
                    "execution": (
                        FORMAL_EXECUTION if args.mode == "formal" else PREFLIGHT_EXECUTION
                    ),
                    "formal": args.mode == "formal",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                })
        except BaseException as receipt_error:
            error.add_note(
                "failure receipt publication also failed: %s"
                % type(receipt_error).__name__
            )
        raise
    print(json.dumps({
        "status": result["status"],
        "decision": result["decision"],
        "transport_oracle_authorized": result["transport_oracle_authorized"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
