"""Strict loader for clean, manifest-bound pose targets."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PoseTarget:
    relative_path: str
    image_sha256: str
    image_size: tuple
    keypoints: torch.Tensor
    scores: torch.Tensor
    valid: torch.Tensor


class PoseTargetStore:
    """Load a complete pose artifact after verifying every content digest."""

    REQUIRED_ARRAYS = (
        "relative_paths",
        "image_sha256",
        "image_sizes",
        "keypoints",
        "scores",
    )

    def __init__(self, artifact_dir, expected_manifest_sha256):
        if not expected_manifest_sha256:
            raise ValueError("expected_manifest_sha256 is required")

        self.artifact_dir = Path(artifact_dir).resolve()
        if "pose_data" in self.artifact_dir.parts:
            raise RuntimeError("Legacy pose_data is forbidden")
        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest_sha = _sha256_file(manifest_path)
        if manifest_sha != expected_manifest_sha256:
            raise RuntimeError("Pose manifest SHA256 mismatch")

        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema_version") != 1:
            raise RuntimeError("Unsupported pose target schema")
        if self.manifest.get("joint_count") != 17:
            raise RuntimeError("Expected COCO-17 pose targets")

        image_root = Path(self.manifest["image_root"]).resolve()
        if "pose_data" in image_root.parts:
            raise RuntimeError("Legacy pose_data is forbidden")
        if not image_root.is_dir():
            raise NotADirectoryError(image_root)
        self.image_root = image_root
        self.dataset_root = image_root.parent
        self.manifest_sha256 = manifest_sha
        self._records = self._load_records()

    def _load_records(self):
        records = {}
        records_digest = hashlib.sha256()
        total = 0

        for shard_meta in self.manifest.get("shards", []):
            shard_path = (self.artifact_dir / shard_meta["file"]).resolve()
            try:
                shard_path.relative_to(self.artifact_dir)
            except ValueError as error:
                raise RuntimeError("Pose shard escapes artifact directory") from error
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            if _sha256_file(shard_path) != shard_meta["sha256"]:
                raise RuntimeError("Pose shard SHA256 mismatch: {}".format(shard_path))

            with np.load(str(shard_path), allow_pickle=False) as arrays:
                if tuple(arrays.files) != self.REQUIRED_ARRAYS:
                    raise RuntimeError("Unexpected pose shard schema")
                paths = arrays["relative_paths"]
                image_sha = arrays["image_sha256"]
                image_sizes = arrays["image_sizes"]
                keypoints = arrays["keypoints"]
                scores = arrays["scores"]
                count = len(paths)
                if count != shard_meta["count"]:
                    raise RuntimeError("Pose shard count mismatch")
                if image_sizes.shape != (count, 2):
                    raise RuntimeError("Unexpected pose image_sizes shape")
                if keypoints.shape != (count, 17, 2):
                    raise RuntimeError("Unexpected pose keypoints shape")
                if scores.shape != (count, 17):
                    raise RuntimeError("Unexpected pose scores shape")
                if keypoints.dtype != np.float32 or scores.dtype != np.float32:
                    raise RuntimeError("Pose tensors must be float32")
                if image_sizes.dtype != np.int32:
                    raise RuntimeError("Pose image sizes must be int32")
                if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
                    raise RuntimeError("Non-finite pose target")

                for relative_path, rgb_sha, size, joints, confidence in zip(
                    paths.tolist(),
                    image_sha.tolist(),
                    image_sizes,
                    keypoints,
                    scores,
                ):
                    if relative_path in records:
                        raise RuntimeError(
                            "Duplicate pose target: {}".format(relative_path)
                        )
                    relative = Path(relative_path)
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or "pose_data" in relative.parts
                    ):
                        raise RuntimeError("Forbidden pose target path")
                    width, height = map(int, size)
                    if width <= 0 or height <= 0:
                        raise RuntimeError("Invalid pose target image size")

                    records_digest.update(relative_path.encode("utf-8"))
                    records_digest.update(b"\0")
                    records_digest.update(rgb_sha.encode("ascii"))
                    records_digest.update(
                        np.asarray([width, height], dtype=np.int32).tobytes()
                    )
                    records_digest.update(joints.tobytes())
                    records_digest.update(confidence.tobytes())
                    records[relative_path] = (
                        rgb_sha,
                        (width, height),
                        joints.copy(),
                        confidence.copy(),
                    )
                total += count

        if total != self.manifest.get("sample_count"):
            raise RuntimeError("Pose artifact sample count mismatch")
        if records_digest.hexdigest() != self.manifest.get(
            "records_manifest_sha256"
        ):
            raise RuntimeError("Pose records manifest SHA256 mismatch")
        return records

    def __len__(self):
        return len(self._records)

    def get(self, image_path, verify_image_sha=False):
        image_path = Path(image_path).resolve()
        if "pose_data" in image_path.parts:
            raise RuntimeError("Legacy pose_data is forbidden")
        try:
            relative_path = image_path.relative_to(self.dataset_root).as_posix()
        except ValueError as error:
            raise KeyError("Image is outside pose dataset root") from error
        if relative_path not in self._records:
            raise KeyError("No pose target for {}".format(relative_path))

        rgb_sha, image_size, keypoints, scores = self._records[relative_path]
        if verify_image_sha and _sha256_file(image_path) != rgb_sha:
            raise RuntimeError("RGB SHA256 mismatch for {}".format(relative_path))

        keypoints = torch.from_numpy(keypoints.copy())
        scores = torch.from_numpy(scores.copy())
        width, height = image_size
        valid = (
            torch.isfinite(keypoints).all(dim=1)
            & torch.isfinite(scores)
            & (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] <= width - 1)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] <= height - 1)
        )
        return PoseTarget(
            relative_path=relative_path,
            image_sha256=rgb_sha,
            image_size=image_size,
            keypoints=keypoints,
            scores=scores,
            valid=valid,
        )
