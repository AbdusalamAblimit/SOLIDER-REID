#!/usr/bin/env python3
"""Eight-image runtime/I/O smoke for the frozen exp415 PACIT interfaces.

This is deliberately not a scientific oracle.  It exercises real RGB decode,
pose geometry, whole-image OpenCLIP, sealed clean-D0 inference, and atomic
cache/result readback.  It never selects a winning proposal, evaluates an
outcome, computes a success rate, or creates the formal oracle namespace.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
for import_root in (REPOSITORY_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import asset_oracle_core as core


SCHEMA_NAME = "exp415-pacit-runtime-smoke-v3"
CACHE_SCHEMA_NAME = "exp415-pacit-runtime-smoke-cache-v3"
SMOKE_COUNT = 8
EXPECTED_TRAIN_COUNT = 15618
EXPECTED_POSE_MANIFEST_SHA256 = (
    "cc09eb6b0be91d731ce0fea77b8fa9d78e5404955ec740a1fc0f1ed00e6359f8"
)
EXPECTED_CLIP_SHA256 = (
    "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
)
EXPECTED_D0_SHA256 = (
    "59017755d61370754aa2e852a487d8e242fcee8814685f77f5388ba3a430e069"
)
EXPECTED_D0_CONFIG_SHA256 = (
    "510f52604cbb455a6f61139d266705c3292e1bb431b2f603e5e750f56edd2c8b"
)
FIXED_SMOKE_OUTPUT_DIR = Path(
    "/home/afr/reid-clean/assets/exp415-pacit-smoke-v3"
)
VARIANT_ORDER = (
    ["clean"]
    + ["pose_edit_{}".format(index) for index in range(7)]
    + ["fixed_edit_{}".format(index) for index in range(7)]
    + ["roa_{}".format(index) for index in range(core.ROA_COUNT)]
)
VARIANTS_PER_ROW = len(VARIANT_ORDER)
FORBIDDEN_RESULT_KEYS = {
    "agreement",
    "argmax",
    "factorial",
    "go",
    "outcome",
    "rate",
    "raw_score",
    "selector_index",
    "success",
    "y",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor):
    value = torch.as_tensor(tensor).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def git_head(repo_root):
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def git_tracked_status(repo_root):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()


def stable_json_bytes(payload):
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _assert_no_scientific_keys(value, path="result"):
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_RESULT_KEYS:
                raise RuntimeError(
                    "runtime smoke contains forbidden scientific key: "
                    + path
                    + "."
                    + str(key)
                )
            _assert_no_scientific_keys(item, path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_scientific_keys(item, path + "[{}]".format(index))


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("fresh JSON target required: {}".format(path))
    data = stable_json_bytes(payload)
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("fresh NPZ target required: {}".format(path))
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _safe_relative(image_path, dataset_root):
    image_path = Path(image_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError as error:
        raise RuntimeError("official RGB escapes dataset root") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("invalid official relative path")
    value = relative.as_posix()
    if not value.startswith("bounding_box_train/"):
        raise RuntimeError("runtime smoke selected a non-train RGB")
    if image_path != (dataset_root / value).resolve():
        raise RuntimeError("official RGB path normalization mismatch")
    return value


def select_smoke_rows(records, dataset_root):
    """Use the frozen oracle hash ordering but return a smoke-only row index."""
    rows = []
    seen = set()
    for record_index, (image_path, pid, camera, track) in enumerate(records):
        relative_path = _safe_relative(image_path, dataset_root)
        if relative_path in seen:
            raise RuntimeError("duplicate official train relative path")
        seen.add(relative_path)
        rows.append(
            {
                "relative_path": relative_path,
                "record_index": int(record_index),
                "pid": int(pid),
                "camera": int(camera),
                "track": int(track),
                "image_path": str(Path(image_path).resolve()),
            }
        )
    if len(rows) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError("official train count mismatch")
    selected = core.select_oracle_rows(rows, count=SMOKE_COUNT)
    output = []
    for row in selected:
        copied = dict(row)
        copied["smoke_index"] = int(copied.pop("oracle_index"))
        output.append(copied)
    if [row["smoke_index"] for row in output] != list(range(SMOKE_COUNT)):
        raise RuntimeError("smoke row index mismatch")
    if len({row["relative_path"] for row in output}) != SMOKE_COUNT:
        raise RuntimeError("smoke paths are not unique")
    return output


def read_canonical_rgb(image_path):
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_size = tuple(map(int, image.size))
        image = image.resize(
            (core.IMAGE_WIDTH, core.IMAGE_HEIGHT),
            resample=Image.Resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.uint8).copy()
    if array.shape != (core.IMAGE_HEIGHT, core.IMAGE_WIDTH, 3):
        raise RuntimeError("canonical RGB decode shape mismatch")
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
    if (
        tensor.shape != (3, core.IMAGE_HEIGHT, core.IMAGE_WIDTH)
        or not bool(torch.isfinite(tensor).all())
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise RuntimeError("canonical RGB tensor contract failed")
    return tensor, original_size


def resize_pose_to_canonical(pose):
    width, height = map(int, pose.image_size)
    if width <= 0 or height <= 0:
        raise RuntimeError("pose image size must be positive")
    keypoints = pose.keypoints.clone().float()
    keypoints[:, 0] *= core.IMAGE_WIDTH / float(width)
    keypoints[:, 1] *= core.IMAGE_HEIGHT / float(height)
    valid = pose.valid.clone().bool()
    if (
        keypoints.shape != (17, 2)
        or valid.shape != (17,)
        or not bool(torch.isfinite(keypoints).all())
    ):
        raise RuntimeError("canonical pose contract failed")
    return keypoints, valid


def _edited_stack(rgb, proposals, fill):
    if len(proposals) != core.ACTIVE_PROPOSALS_PER_IMAGE:
        raise RuntimeError("active proposal count mismatch")
    edited = []
    mask_shas = []
    for aspect_index, proposal in enumerate(proposals):
        if int(proposal["aspect_index"]) != aspect_index:
            raise RuntimeError("active proposal aspect order mismatch")
        value = core.apply_candidate(rgb, proposal["mask"], fill)
        if not torch.equal(value[:, ~proposal["mask"]], rgb[:, ~proposal["mask"]]):
            raise RuntimeError("edited RGB changed outside its mask")
        edited.append(value)
        mask_shas.append(str(proposal["mask_sha256"]))
    output = torch.stack(edited, dim=0)
    if (
        output.shape
        != (
            core.ACTIVE_PROPOSALS_PER_IMAGE,
            3,
            core.IMAGE_HEIGHT,
            core.IMAGE_WIDTH,
        )
        or not bool(torch.isfinite(output).all())
    ):
        raise RuntimeError("edited RGB stack contract failed")
    return output, mask_shas


class ExplodingPose(dict):
    accesses = 0

    @classmethod
    def reset(cls):
        cls.accesses = 0

    def _fail(self, operation):
        type(self).accesses += 1
        raise RuntimeError("D0 eval accessed external pose via " + operation)

    def __getitem__(self, key):
        return self._fail("getitem")

    def get(self, key, default=None):
        del key, default
        return self._fail("get")

    def __iter__(self):
        return self._fail("iter")


def _checkpoint_state(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("unexpected D0 checkpoint payload")
    return payload


def run_d0_interface(
    *,
    config_path,
    checkpoint_path,
    checkpoint_sha256,
    dataset,
    variant_rgb,
    variant_pid,
    device,
    microbatch,
):
    from config import cfg
    from model import make_model

    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if sha256_file(config_path) != EXPECTED_D0_CONFIG_SHA256:
        raise RuntimeError("D0 config SHA256 mismatch")
    checkpoint_sha_before = sha256_file(checkpoint_path)
    if (
        str(checkpoint_sha256) != EXPECTED_D0_SHA256
        or checkpoint_sha_before != EXPECTED_D0_SHA256
    ):
        raise RuntimeError("D0 checkpoint SHA256 mismatch")

    cfg.merge_from_file(str(config_path))
    local_cfg = cfg.clone()
    local_cfg.defrost()
    # The sealed D0 state is the only model asset used by this smoke.  Keep
    # construction independent of the config's historical pretrain path.
    local_cfg.MODEL.PRETRAIN_PATH = ""
    local_cfg.freeze()
    model = make_model(
        local_cfg,
        num_class=dataset.num_train_pids,
        camera_num=dataset.num_train_cams,
        view_num=dataset.num_train_vids,
        semantic_weight=local_cfg.MODEL.SEMANTIC_WEIGHT,
    )
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    incompatible = model.load_state_dict(_checkpoint_state(payload), strict=True)
    del payload
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict D0 checkpoint load failed")
    for name, value in model.state_dict().items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise RuntimeError("non-finite D0 state tensor: " + name)

    device = torch.device(device)
    model = model.to(device).eval()
    mean = torch.tensor(
        local_cfg.INPUT.PIXEL_MEAN, dtype=torch.float32, device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        local_cfg.INPUT.PIXEL_STD, dtype=torch.float32, device=device
    ).view(1, 3, 1, 1)
    descriptors = []
    logits = []
    ExplodingPose.reset()
    sentinel = ExplodingPose()
    with torch.inference_mode():
        for start in range(0, len(variant_rgb), int(microbatch)):
            stop = min(start + int(microbatch), len(variant_rgb))
            batch = variant_rgb[start:stop].to(device)
            normalized = (batch - mean) / std
            output = model(
                normalized,
                pose_batch=sentinel,
                tapf_epoch=None,
            )
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or not torch.is_tensor(output[0])
            ):
                raise RuntimeError("unexpected D0 eval output")
            descriptor = output[0]
            classifier_input = model.dropout(model.bottleneck(descriptor))
            cls_score = model.classifier(classifier_input)
            if (
                descriptor.ndim != 2
                or cls_score.shape
                != (len(descriptor), dataset.num_train_pids)
                or not bool(torch.isfinite(descriptor).all())
                or not bool(torch.isfinite(cls_score).all())
            ):
                raise RuntimeError("D0 descriptor/logit contract failed")
            descriptors.append(descriptor.detach().cpu())
            logits.append(cls_score.detach().cpu())
    if ExplodingPose.accesses != 0:
        raise RuntimeError("D0 eval consumed external pose")

    descriptor = torch.cat(descriptors, dim=0)
    cls_score = torch.cat(logits, dim=0)
    labels = torch.as_tensor(variant_pid, dtype=torch.long)
    if (
        descriptor.shape[0] != len(variant_rgb)
        or cls_score.shape[0] != len(variant_rgb)
        or labels.shape != (len(variant_rgb),)
        or int(labels.min()) < 0
        or int(labels.max()) >= dataset.num_train_pids
    ):
        raise RuntimeError("D0 variant/PID alignment failed")
    ce = F.cross_entropy(cls_score.float(), labels, reduction="none")
    top5 = cls_score.topk(k=5, dim=1).indices.eq(labels[:, None]).any(dim=1)
    descriptor = F.normalize(descriptor.float(), dim=1)
    grouped = descriptor.reshape(
        SMOKE_COUNT, VARIANTS_PER_ROW, descriptor.shape[1]
    )
    displacement = 1.0 - (
        grouped[:, 1:] * grouped[:, :1]
    ).sum(dim=-1)
    ce_grouped = ce.reshape(SMOKE_COUNT, VARIANTS_PER_ROW)
    ce_change = ce_grouped[:, 1:] - ce_grouped[:, :1]
    top5_grouped = top5.reshape(SMOKE_COUNT, VARIANTS_PER_ROW)
    for name, value in (
        ("descriptor", descriptor),
        ("logits", cls_score),
        ("CE", ce),
        ("displacement", displacement),
        ("CE change", ce_change),
    ):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("non-finite D0 " + name)

    model = model.cpu()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if sha256_file(checkpoint_path) != checkpoint_sha_before:
        raise RuntimeError("D0 checkpoint changed during smoke")
    return {
        "descriptor_shape": list(descriptor.shape),
        "descriptor_dtype": str(descriptor.dtype),
        "descriptor_sha256": sha256_tensor(descriptor),
        "logit_shape": list(cls_score.shape),
        "logit_dtype": str(cls_score.dtype),
        "logit_sha256": sha256_tensor(cls_score),
        "ce_shape": list(ce.shape),
        "ce_sha256": sha256_tensor(ce),
        "top5_shape": list(top5_grouped.shape),
        "top5_sha256": sha256_tensor(top5_grouped),
        "displacement_shape": list(displacement.shape),
        "displacement_sha256": sha256_tensor(displacement),
        "ce_change_shape": list(ce_change.shape),
        "ce_change_sha256": sha256_tensor(ce_change),
        "external_pose_accesses": int(ExplodingPose.accesses),
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": sha256_file(checkpoint_path),
    }


def _cache_arrays(
    *,
    selected,
    image_sha,
    region_valid,
    pose_mask_sha,
    fixed_mask_sha,
    clip_pose_sha,
    clip_fixed_sha,
    d0_summary,
):
    arrays = {
        "schema": np.asarray([CACHE_SCHEMA_NAME]),
        "smoke_index": np.asarray(
            [row["smoke_index"] for row in selected], dtype=np.int32
        ),
        "relative_paths": np.asarray(
            [row["relative_path"] for row in selected]
        ),
        "pid": np.asarray([row["pid"] for row in selected], dtype=np.int64),
        "camera": np.asarray(
            [row["camera"] for row in selected], dtype=np.int64
        ),
        "image_sha256": np.asarray(image_sha),
        "region_valid": np.asarray(region_valid, dtype=np.bool_),
        "pose_mask_sha256": np.asarray(pose_mask_sha),
        "fixed_mask_sha256": np.asarray(fixed_mask_sha),
        "clip_pose_tensor_sha256": np.asarray(clip_pose_sha),
        "clip_fixed_tensor_sha256": np.asarray(clip_fixed_sha),
        "d0_descriptor_sha256": np.asarray(
            [d0_summary["descriptor_sha256"]]
        ),
        "d0_logit_sha256": np.asarray([d0_summary["logit_sha256"]]),
        "d0_ce_sha256": np.asarray([d0_summary["ce_sha256"]]),
        "d0_top5_sha256": np.asarray([d0_summary["top5_sha256"]]),
        "d0_displacement_sha256": np.asarray(
            [d0_summary["displacement_sha256"]]
        ),
        "d0_ce_change_sha256": np.asarray(
            [d0_summary["ce_change_sha256"]]
        ),
    }
    if arrays["relative_paths"].shape != (SMOKE_COUNT,):
        raise RuntimeError("smoke cache row count mismatch")
    if arrays["region_valid"].shape != (SMOKE_COUNT, core.ANCHOR_COUNT):
        raise RuntimeError("smoke cache region-valid shape mismatch")
    expected_masks = (SMOKE_COUNT, core.ACTIVE_PROPOSALS_PER_IMAGE)
    if (
        arrays["pose_mask_sha256"].shape != expected_masks
        or arrays["fixed_mask_sha256"].shape != expected_masks
    ):
        raise RuntimeError("smoke cache mask shape mismatch")
    return arrays


def _readback_npz(path, expected):
    with np.load(str(path), allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected):
            raise RuntimeError("smoke cache schema changed on readback")
        for key, value in expected.items():
            if not np.array_equal(arrays[key], value):
                raise RuntimeError("smoke cache readback mismatch: " + key)


def _prepare_fresh_namespace(output_dir, read_only_roots):
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir != FIXED_SMOKE_OUTPUT_DIR:
        raise RuntimeError(
            "runtime smoke output must equal the frozen smoke namespace"
        )
    for root in read_only_roots:
        try:
            output_dir.relative_to(Path(root).expanduser().resolve())
        except ValueError:
            continue
        raise RuntimeError("smoke namespace enters a read-only root")
    if output_dir.exists():
        raise FileExistsError("fresh smoke namespace required")
    if not output_dir.parent.is_dir():
        raise NotADirectoryError(output_dir.parent)
    output_dir.mkdir(mode=0o755)
    return output_dir


def run_smoke(args):
    stage = "argument_validation"
    source_head = None
    output_dir = None
    if not torch.cuda.is_available():
        raise RuntimeError("runtime smoke requires the real CUDA interfaces")
    if str(args.device) != "cuda:0":
        raise RuntimeError("runtime smoke is frozen to logical cuda:0")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if int(args.microbatch) <= 0:
        raise ValueError("microbatch must be positive")
    data_root = Path(args.data_root).expanduser().resolve()
    pose_artifact = Path(args.pose_artifact).expanduser().resolve()
    clip_checkpoint = Path(args.clip_checkpoint).expanduser().resolve()
    d0_config = Path(args.d0_config).expanduser().resolve()
    d0_checkpoint = Path(args.d0_checkpoint).expanduser().resolve()
    for path in (data_root, pose_artifact):
        if not path.is_dir():
            raise NotADirectoryError(path)
    for path in (clip_checkpoint, d0_config, d0_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        str(args.pose_manifest_sha256) != EXPECTED_POSE_MANIFEST_SHA256
        or str(args.clip_sha256) != EXPECTED_CLIP_SHA256
        or str(args.d0_sha256) != EXPECTED_D0_SHA256
    ):
        raise RuntimeError("runtime smoke received a non-frozen asset SHA")
    if sha256_file(clip_checkpoint) != EXPECTED_CLIP_SHA256:
        raise RuntimeError("CLIP checkpoint SHA256 mismatch")
    source_head = git_head(REPOSITORY_ROOT)
    if git_tracked_status(REPOSITORY_ROOT):
        raise RuntimeError("formal tracked source is dirty")

    output_dir = _prepare_fresh_namespace(
        args.output_dir,
        read_only_roots=(data_root, pose_artifact),
    )
    try:
        stage = "dataset_and_pose_load"
        from datasets.occluded_duke import OccludedDuke
        from datasets.pose_targets import PoseTargetStore
        from clip_color_selector import FrozenWholeImageColorSelector

        dataset = OccludedDuke(root=str(data_root), verbose=False)
        records = list(dataset.train)
        if len(records) != EXPECTED_TRAIN_COUNT:
            raise RuntimeError("official train count mismatch")
        train_pids = sorted({int(record[1]) for record in records})
        if train_pids != list(range(dataset.num_train_pids)):
            raise RuntimeError("official train PID labels are not contiguous")
        selected = select_smoke_rows(records, Path(dataset.dataset_dir))
        pose_store = PoseTargetStore(
            pose_artifact, args.pose_manifest_sha256
        )
        if len(pose_store) != EXPECTED_TRAIN_COUNT:
            raise RuntimeError("pose artifact count mismatch")

        stage = "clip_interface"
        selector = FrozenWholeImageColorSelector(
            clip_checkpoint,
            args.clip_sha256,
            device,
            microbatch=args.microbatch,
        )
        fixed_pool = core.generate_fixed_proposals()
        image_sha = []
        region_valid_rows = []
        pose_mask_sha = []
        fixed_mask_sha = []
        clip_pose_sha = []
        clip_fixed_sha = []
        variant_rgb = []
        variant_pid = []
        clip_shapes = set()
        clip_dtypes = set()
        for row in selected:
            image_path = Path(row["image_path"]).resolve()
            pose = pose_store.get(image_path, verify_image_sha=True)
            rgb, decoded_size = read_canonical_rgb(image_path)
            if decoded_size != tuple(map(int, pose.image_size)):
                raise RuntimeError("RGB/pose image size mismatch")
            keypoints, valid = resize_pose_to_canonical(pose)
            pose_pool, fields, region_valid = core.generate_pose_proposals(
                keypoints, valid
            )
            if (
                fields.shape
                != (
                    core.ANCHOR_COUNT,
                    core.IMAGE_HEIGHT,
                    core.IMAGE_WIDTH,
                )
                or region_valid.shape != (core.ANCHOR_COUNT,)
                or not bool(torch.isfinite(fields).all())
            ):
                raise RuntimeError("pose field runtime contract failed")
            pose_active = core.active_proposals(
                pose_pool, row["smoke_index"]
            )
            fixed_active = core.active_proposals(
                fixed_pool, row["smoke_index"]
            )
            fill = core.deterministic_fill(row["relative_path"])
            pose_edited, pose_shas = _edited_stack(
                rgb, pose_active, fill
            )
            fixed_edited, fixed_shas = _edited_stack(
                rgb, fixed_active, fill
            )

            with torch.inference_mode():
                pose_drop = selector(
                    rgb.to(device), pose_edited.to(device)
                ).detach().cpu()
                fixed_drop = selector(
                    rgb.to(device), fixed_edited.to(device)
                ).detach().cpu()
            expected_drop = (
                core.ACTIVE_PROPOSALS_PER_IMAGE,
                len(core.COLOR_NAMES),
            )
            if (
                pose_drop.shape != expected_drop
                or fixed_drop.shape != expected_drop
                or not bool(torch.isfinite(pose_drop).all())
                or not bool(torch.isfinite(fixed_drop).all())
            ):
                raise RuntimeError("CLIP runtime output contract failed")
            clip_shapes.add(tuple(pose_drop.shape))
            clip_shapes.add(tuple(fixed_drop.shape))
            clip_dtypes.add(str(pose_drop.dtype))
            clip_dtypes.add(str(fixed_drop.dtype))
            image_sha.append(str(pose.image_sha256))
            region_valid_rows.append(region_valid.tolist())
            pose_mask_sha.append(pose_shas)
            fixed_mask_sha.append(fixed_shas)
            clip_pose_sha.append(sha256_tensor(pose_drop))
            clip_fixed_sha.append(sha256_tensor(fixed_drop))

            roa_edited = torch.stack(
                [
                    core.apply_candidate(rgb, mask, fill)
                    for mask in core.deterministic_roa_masks(
                        row["relative_path"]
                    )
                ],
                dim=0,
            )
            variants = torch.cat(
                (
                    rgb.unsqueeze(0),
                    pose_edited,
                    fixed_edited,
                    roa_edited,
                ),
                dim=0,
            )
            if (
                variants.shape
                != (
                    VARIANTS_PER_ROW,
                    3,
                    core.IMAGE_HEIGHT,
                    core.IMAGE_WIDTH,
                )
                or not bool(torch.isfinite(variants).all())
            ):
                raise RuntimeError("D0 smoke variant stack mismatch")
            variant_rgb.append(variants)
            variant_pid.extend([int(row["pid"])] * VARIANTS_PER_ROW)

        if clip_shapes != {
            (core.ACTIVE_PROPOSALS_PER_IMAGE, len(core.COLOR_NAMES))
        }:
            raise RuntimeError("CLIP smoke shapes are inconsistent")
        if len(clip_dtypes) != 1:
            raise RuntimeError("CLIP smoke dtypes are inconsistent")
        del selector
        gc.collect()
        torch.cuda.empty_cache()
        all_variants = torch.cat(variant_rgb, dim=0)
        if all_variants.shape[0] != SMOKE_COUNT * VARIANTS_PER_ROW:
            raise RuntimeError("D0 smoke variant count mismatch")
        stage = "d0_interface"
        d0_summary = run_d0_interface(
            config_path=d0_config,
            checkpoint_path=d0_checkpoint,
            checkpoint_sha256=args.d0_sha256,
            dataset=dataset,
            variant_rgb=all_variants,
            variant_pid=variant_pid,
            device=device,
            microbatch=args.microbatch,
        )
        arrays = _cache_arrays(
            selected=selected,
            image_sha=image_sha,
            region_valid=region_valid_rows,
            pose_mask_sha=pose_mask_sha,
            fixed_mask_sha=fixed_mask_sha,
            clip_pose_sha=clip_pose_sha,
            clip_fixed_sha=clip_fixed_sha,
            d0_summary=d0_summary,
        )
        stage = "cache_write_readback"
        cache_path = output_dir / "smoke_cache.npz"
        _atomic_npz(cache_path, arrays)
        _readback_npz(cache_path, arrays)
        cache_sha = sha256_file(cache_path)
        result = {
            "schema": SCHEMA_NAME,
            "status": "PASS",
            "scope": {
                "row_count": SMOKE_COUNT,
                "variant_count_per_row": VARIANTS_PER_ROW,
                "variant_order": list(VARIANT_ORDER),
                "scientific_metrics_written": False,
                "proposal_winner_selected": False,
                "optimizer_updates": 0,
                "checkpoint_writes": 0,
            },
            "interfaces": {
                "rgb_decode_shape": [
                    3,
                    core.IMAGE_HEIGHT,
                    core.IMAGE_WIDTH,
                ],
                "pose_field_shape": [
                    core.ANCHOR_COUNT,
                    core.IMAGE_HEIGHT,
                    core.IMAGE_WIDTH,
                ],
                "active_proposals_per_pool": (
                    core.ACTIVE_PROPOSALS_PER_IMAGE
                ),
                "clip_output_shape": list(next(iter(clip_shapes))),
                "clip_output_dtype": next(iter(clip_dtypes)),
                "d0": d0_summary,
            },
            "provenance": {
                "source_head": source_head,
                "tracked_source_clean": True,
                "relative_path_order_sha256": core.ordered_digest(
                    [row["relative_path"] for row in selected]
                ),
                "pose_manifest_sha256": args.pose_manifest_sha256,
                "clip_checkpoint_sha256": sha256_file(clip_checkpoint),
                "d0_config_sha256": sha256_file(d0_config),
                "d0_checkpoint_sha256": sha256_file(d0_checkpoint),
                "runtime_smoke_source_sha256": sha256_file(Path(__file__)),
                "asset_oracle_core_source_sha256": sha256_file(
                    Path(core.__file__).resolve()
                ),
                "cache_sha256": cache_sha,
            },
            "gates": {
                "official_train_exact": True,
                "pose_lookup_rgb_sha_exact": True,
                "canonical_rgb_finite_in_range": True,
                "pose_and_fixed_active7_exact": True,
                "mask_exterior_byte_exact": True,
                "clip_shape_and_finite": True,
                "d0_descriptor_logit_ce_top5_finite": True,
                "cache_readback_exact": True,
                "formal_oracle_namespace_touched": False,
            },
        }
        _assert_no_scientific_keys(result)
        stage = "result_write_readback"
        result_path = output_dir / "result.json"
        _atomic_json(result_path, result)
        readback = json.loads(result_path.read_text(encoding="utf-8"))
        if readback != result:
            raise RuntimeError("smoke result JSON readback mismatch")
        final_cache = output_dir / "smoke_cache.npz"
        final_result = output_dir / "result.json"
        _readback_npz(final_cache, arrays)
        final_payload = json.loads(final_result.read_text(encoding="utf-8"))
        if final_payload != result:
            raise RuntimeError("final smoke result readback mismatch")
        if sha256_file(final_cache) != cache_sha:
            raise RuntimeError("smoke cache SHA changed after final readback")
        return {
            "status": "PASS",
            "output_dir": str(output_dir),
            "cache_sha256": sha256_file(final_cache),
            "result_sha256": sha256_file(final_result),
            "row_count": SMOKE_COUNT,
            "scientific_metrics_written": False,
        }
    except BaseException as error:
        if output_dir is not None and output_dir.is_dir():
            failure = output_dir / "failure.json"
            if not failure.exists() and not failure.with_name(
                failure.name + ".tmp"
            ).exists():
                _atomic_json(
                    failure,
                    {
                        "schema": "exp415-pacit-runtime-smoke-failure-v3",
                        "status": "FAIL",
                        "stage": stage,
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                        "source_head": source_head,
                        "runtime_smoke_source_sha256": sha256_file(
                            Path(__file__)
                        ),
                        "formal_oracle_namespace_touched": False,
                        "resume_allowed": False,
                    },
                )
        raise


def run_self_test():
    synthetic_records = [
        (
            str(
                REPOSITORY_ROOT
                / "bounding_box_train"
                / "{:05d}.jpg".format(index)
            ),
            index % 7,
            index % 3,
            index,
        )
        for index in range(EXPECTED_TRAIN_COUNT)
    ]
    selected = select_smoke_rows(synthetic_records, REPOSITORY_ROOT)
    reversed_selected = select_smoke_rows(
        list(reversed(synthetic_records)), REPOSITORY_ROOT
    )
    if [row["relative_path"] for row in selected] != [
        row["relative_path"] for row in reversed_selected
    ]:
        raise AssertionError("smoke hash selection depends on record order")
    fake_d0 = {
        "descriptor_sha256": "a" * 64,
        "logit_sha256": "b" * 64,
        "ce_sha256": "c" * 64,
        "top5_sha256": "d" * 64,
        "displacement_sha256": "e" * 64,
        "ce_change_sha256": "f" * 64,
    }
    arrays = _cache_arrays(
        selected=selected,
        image_sha=["1" * 64] * SMOKE_COUNT,
        region_valid=[[True] * core.ANCHOR_COUNT] * SMOKE_COUNT,
        pose_mask_sha=[
            ["2" * 64] * core.ACTIVE_PROPOSALS_PER_IMAGE
        ]
        * SMOKE_COUNT,
        fixed_mask_sha=[
            ["3" * 64] * core.ACTIVE_PROPOSALS_PER_IMAGE
        ]
        * SMOKE_COUNT,
        clip_pose_sha=["4" * 64] * SMOKE_COUNT,
        clip_fixed_sha=["5" * 64] * SMOKE_COUNT,
        d0_summary=fake_d0,
    )
    with tempfile.TemporaryDirectory(
        prefix=".exp415-smoke-selftest-", dir=str(SCRIPT_DIR)
    ) as directory:
        directory = Path(directory)
        cache = directory / "cache.npz"
        _atomic_npz(cache, arrays)
        _readback_npz(cache, arrays)
        payload = {
            "schema": SCHEMA_NAME,
            "status": "PASS",
            "scientific_metrics_written": False,
        }
        _assert_no_scientific_keys(payload)
        result = directory / "result.json"
        _atomic_json(result, payload)
        if json.loads(result.read_text(encoding="utf-8")) != payload:
            raise AssertionError("self-test JSON readback failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "self_test": SCHEMA_NAME,
                "real_data_loaded": False,
                "cuda_used": False,
            },
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the non-scientific exp415 eight-image runtime smoke."
    )
    parser.add_argument("--data-root")
    parser.add_argument("--pose-artifact")
    parser.add_argument(
        "--pose-manifest-sha256",
        default=EXPECTED_POSE_MANIFEST_SHA256,
    )
    parser.add_argument("--clip-checkpoint")
    parser.add_argument(
        "--clip-sha256", default=EXPECTED_CLIP_SHA256
    )
    parser.add_argument("--d0-config")
    parser.add_argument("--d0-checkpoint")
    parser.add_argument("--d0-sha256", default=EXPECTED_D0_SHA256)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    required = (
        "data_root",
        "pose_artifact",
        "clip_checkpoint",
        "d0_config",
        "d0_checkpoint",
        "output_dir",
    )
    missing = ["--" + name.replace("_", "-") for name in required if not getattr(args, name)]
    if missing:
        raise ValueError("missing required arguments: " + ", ".join(missing))
    print(json.dumps(run_smoke(args), sort_keys=True))


if __name__ == "__main__":
    main()
