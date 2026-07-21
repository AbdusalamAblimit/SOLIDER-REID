"""Strict train-only identity proxy asset for exp410 PC2P."""

from __future__ import annotations

import hashlib
import re
import stat
from pathlib import Path

import numpy as np
import torch


SCHEMA = "exp410-pc2p-bank-v1"
DATASET = "occluded_duke"
SPLIT = "train"
NUM_SAMPLES = 15618
NUM_IDENTITIES = 702
NUM_SLOTS = 5
FEATURE_DIM = 768
SOURCE_CACHE_SHA256 = (
    "d502a0f03fe556284fd01259ed81143dcfb171855b9b2aebaa29e3b7a682fd36"
)
SOURCE_CACHE_SCHEMA = "exp409-pchm-cache-v1"
SOURCE_PREPROCESSING = "raw-rgb-pose-resize-384x128-no-augmentation"
SOURCE_CLIP_CHECKPOINT_SHA256 = (
    "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _original_pid(path):
    match = re.fullmatch(r"(-?\d+)_c\d+_f\d+\.jpg", Path(path).name)
    if match is None or int(match.group(1)) < 0:
        raise RuntimeError("invalid Occluded-Duke train filename")
    return int(match.group(1))


def dataset_contract(records, pose_store):
    """Return deterministic train-set provenance and the relabel PID map."""
    if len(records) != NUM_SAMPLES:
        raise RuntimeError("PC2P requires the official 15,618-image train split")
    paths = []
    image_sha256 = []
    pid_map = {}
    for image_path, relabel, _, _ in records:
        relative_path = (
            Path(image_path)
            .resolve()
            .relative_to(pose_store.dataset_root)
            .as_posix()
        )
        try:
            rgb_sha256 = str(pose_store._records[relative_path][0])
        except KeyError as error:
            raise RuntimeError("PC2P official path is absent from pose artifact") from error
        relabel = int(relabel)
        original = _original_pid(image_path)
        previous = pid_map.setdefault(relabel, original)
        if previous != original:
            raise RuntimeError("PC2P relabel maps to multiple original PIDs")
        paths.append(relative_path)
        image_sha256.append(rgb_sha256)
    if len(paths) != len(set(paths)):
        raise RuntimeError("PC2P official train paths are not unique")
    if set(pid_map) != set(range(NUM_IDENTITIES)):
        raise RuntimeError("PC2P relabel PID rows are not contiguous")
    relabel_to_original = tuple(pid_map[row] for row in range(NUM_IDENTITIES))
    return {
        "paths": tuple(paths),
        "image_sha256": tuple(image_sha256),
        "paths_sha256": ordered_digest(paths),
        "rgb_binding_sha256": ordered_digest(
            "{}\0{}".format(path, rgb_sha)
            for path, rgb_sha in zip(paths, image_sha256)
        ),
        "relabel_to_original_pid": relabel_to_original,
        "pid_mapping_sha256": ordered_digest(relabel_to_original),
    }


class PoseCompleteClipProxyBank:
    """Load and validate one immutable exp410 proxy bank."""

    REQUIRED_FIELDS = {
        "schema",
        "dataset",
        "split",
        "proxy",
        "slot_counts",
        "relabel_to_original_pid",
        "official_paths_sha256",
        "rgb_binding_sha256",
        "pid_mapping_sha256",
        "source_cache_sha256",
        "source_cache_schema",
        "source_preprocessing",
        "source_pose_manifest_sha256",
        "source_clip_checkpoint_sha256",
        "source_head",
        "source_builder_sha256",
        "source_teacher_source_sha256",
        "builder_sha256",
        "loader_source_sha256",
    }

    def __init__(
        self,
        path,
        expected_sha256,
        expected_pose_manifest_sha256,
        expected_num_classes,
    ):
        configured = Path(path).expanduser()
        if not configured.is_absolute():
            raise ValueError("PC2P bank path must be absolute")
        resolved = configured.resolve(strict=True)
        if resolved != configured:
            raise RuntimeError("PC2P bank must use its canonical path")
        metadata = resolved.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("PC2P bank must be a unique regular file")
        if not expected_sha256:
            raise ValueError("PC2P bank SHA256 is required")
        self.sha256 = sha256_file(resolved)
        if self.sha256 != expected_sha256:
            raise RuntimeError("PC2P bank SHA256 mismatch")

        with np.load(str(resolved), allow_pickle=False) as payload:
            if set(payload.files) != self.REQUIRED_FIELDS:
                raise RuntimeError("unexpected PC2P bank fields")
            scalars = {
                name: str(payload[name].item())
                for name in self.REQUIRED_FIELDS
                if name
                not in {"proxy", "slot_counts", "relabel_to_original_pid"}
            }
            proxy_array = payload["proxy"]
            count_array = payload["slot_counts"]
            pid_array = payload["relabel_to_original_pid"]

        expected_scalars = {
            "schema": SCHEMA,
            "dataset": DATASET,
            "split": SPLIT,
            "source_cache_sha256": SOURCE_CACHE_SHA256,
            "source_cache_schema": SOURCE_CACHE_SCHEMA,
            "source_preprocessing": SOURCE_PREPROCESSING,
            "source_pose_manifest_sha256": str(expected_pose_manifest_sha256),
            "source_clip_checkpoint_sha256": SOURCE_CLIP_CHECKPOINT_SHA256,
        }
        for name, expected in expected_scalars.items():
            if scalars[name] != expected:
                raise RuntimeError("PC2P bank provenance mismatch: " + name)
        hex64 = re.compile(r"[0-9a-f]{64}")
        hex40 = re.compile(r"[0-9a-f]{40}")
        for name in (
            "official_paths_sha256",
            "rgb_binding_sha256",
            "pid_mapping_sha256",
            "source_builder_sha256",
            "source_teacher_source_sha256",
            "builder_sha256",
            "loader_source_sha256",
        ):
            if hex64.fullmatch(scalars[name]) is None:
                raise RuntimeError("invalid PC2P provenance digest: " + name)
        if hex40.fullmatch(scalars["source_head"]) is None:
            raise RuntimeError("invalid PC2P source HEAD")
        builder_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "exp410_pose_complete_clip_proxy_classifier"
            / "build_proxy_bank.py"
        )
        if sha256_file(builder_path) != scalars["builder_sha256"]:
            raise RuntimeError("PC2P builder source SHA mismatch")
        if sha256_file(Path(__file__).resolve()) != scalars["loader_source_sha256"]:
            raise RuntimeError("PC2P loader source SHA mismatch")

        if int(expected_num_classes) != NUM_IDENTITIES:
            raise RuntimeError("PC2P model class count must be 702")
        if proxy_array.shape != (NUM_IDENTITIES, FEATURE_DIM):
            raise RuntimeError("PC2P proxy shape mismatch")
        if proxy_array.dtype != np.float32:
            raise RuntimeError("PC2P proxy must be float32")
        if count_array.shape != (NUM_IDENTITIES, NUM_SLOTS):
            raise RuntimeError("PC2P slot count shape mismatch")
        if count_array.dtype != np.int64 or bool((count_array <= 0).any()):
            raise RuntimeError("PC2P every identity-slot must have support")
        if pid_array.shape != (NUM_IDENTITIES,) or pid_array.dtype != np.int64:
            raise RuntimeError("PC2P PID mapping shape/dtype mismatch")
        if len(set(int(value) for value in pid_array.tolist())) != NUM_IDENTITIES:
            raise RuntimeError("PC2P original PID mapping is not unique")
        proxy = torch.from_numpy(proxy_array.copy())
        if not bool(torch.isfinite(proxy).all()):
            raise RuntimeError("PC2P proxy contains non-finite values")
        norms = proxy.norm(dim=-1)
        if not bool(
            torch.allclose(
                norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6
            )
        ):
            raise RuntimeError("PC2P proxy rows must be unit normalized")
        if np.unique(proxy_array.view(np.dtype((np.void, proxy_array.dtype.itemsize * FEATURE_DIM)))).size != NUM_IDENTITIES:
            raise RuntimeError("PC2P proxy contains duplicate rows")

        self.path = resolved
        self.proxy = proxy
        self.slot_counts = torch.from_numpy(count_array.copy())
        self.relabel_to_original_pid = tuple(int(value) for value in pid_array.tolist())
        self.official_paths_sha256 = scalars["official_paths_sha256"]
        self.rgb_binding_sha256 = scalars["rgb_binding_sha256"]
        self.pid_mapping_sha256 = scalars["pid_mapping_sha256"]
        self.source_head = scalars["source_head"]
        self.builder_sha256 = scalars["builder_sha256"]

    def validate_dataset(self, records, pose_store):
        contract = dataset_contract(records, pose_store)
        expected = {
            "official_paths_sha256": contract["paths_sha256"],
            "rgb_binding_sha256": contract["rgb_binding_sha256"],
            "pid_mapping_sha256": contract["pid_mapping_sha256"],
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise RuntimeError("PC2P bank/dataset binding mismatch: " + name)
        if self.relabel_to_original_pid != contract["relabel_to_original_pid"]:
            raise RuntimeError("PC2P bank PID row order mismatch")

    def to(self, device):
        proxy = self.proxy.to(device=device, non_blocking=True)
        proxy.requires_grad_(False)
        return proxy
