#!/usr/bin/env python3
"""Once-only 512-row PACIT asset oracle.

The formal path reads only the frozen official RGB, pose, OpenCLIP, clean-D0,
configuration, and source assets.  It never consumes any preflight output.
Scientific rows are never dropped: an explicitly classified per-row data or
finite-output failure produces six present records with zero outcomes, while
model construction/runtime, CUDA, schema, source, or asset failures invalidate
the whole once-only run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

# The formal launcher also requires PYTHONDONTWRITEBYTECODE=1.  Set this before
# importing NumPy, Torch, or repository modules so this script cannot create a
# new pycache even when used for local self-tests.
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
for import_root in (REPOSITORY_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import asset_oracle_core as core


SCHEMA_NAME = "exp415-pacit-asset-oracle-v3"
CACHE_SCHEMA_NAME = "exp415-pacit-asset-oracle-cache-v3"
ROWS_SCHEMA_NAME = "exp415-pacit-asset-oracle-rows-v3"
SUMMARY_SCHEMA_NAME = "exp415-pacit-asset-oracle-summary-v3"
RESULT_SCHEMA_NAME = "exp415-pacit-asset-oracle-result-v3"
MANIFEST_SCHEMA_NAME = "exp415-pacit-asset-oracle-manifest-v3"
FAILURE_SCHEMA_NAME = "exp415-pacit-asset-oracle-failure-v3"

EXPECTED_INTERPRETER = Path(
    "/usr/local/anaconda3/envs/mmpose-abu/bin/python"
)
FIXED_DATA_ROOT = Path("/mnt1/afrdata")
FIXED_POSE_ARTIFACT = Path(
    "/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train"
)
FIXED_CLIP_CHECKPOINT = Path(
    "/home/afr/reid-clean/weights/"
    "exp401_clip_l14_openclip_9ce2e8a8.safetensors"
)
FIXED_D0_CHECKPOINT = Path(
    "/home/afr/SOLIDER-REID-exp387-d0-0d1822a/log/occluded_duke/"
    "exp387_clean_swin_tiny_d0_s1234/transformer_120.pth"
)
FIXED_OUTPUT_DIR = Path(
    "/home/afr/reid-clean/assets/exp415-pacit-oracle-v3"
)
FIXED_D0_CONFIG_RELATIVE = Path(
    "configs/occluded_duke/swin_tiny_tapf_d0.yml"
)

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

MICROBATCH = 4
FORMAL_SEED = 1234
VARIANT_ORDER = (
    ["clean"]
    + ["pose_edit_{}".format(index) for index in range(7)]
    + ["fixed_edit_{}".format(index) for index in range(7)]
    + ["roa_{}".format(index) for index in range(core.ROA_COUNT)]
)
VARIANTS_PER_ROW = len(VARIANT_ORDER)
EXPECTED_VARIANT_ORDER = tuple(
    ["clean"]
    + ["pose_edit_{}".format(index) for index in range(7)]
    + ["fixed_edit_{}".format(index) for index in range(7)]
    + ["roa_{}".format(index) for index in range(8)]
)
if tuple(VARIANT_ORDER) != EXPECTED_VARIANT_ORDER or VARIANTS_PER_ROW != 23:
    raise RuntimeError("frozen 23-variant order changed")
ARM_ORDER = tuple(core.FACTORIAL_ARM_NAMES)
STRONG_ORDER = ("raw_color", "d0_hard")
EDGE_ORDER = tuple(core.QUARTET_EDGE_NAMES)
SOURCE_FILES = {
    "runner": Path(__file__).resolve(),
    "core": (SCRIPT_DIR / "asset_oracle_core.py").resolve(),
    "selector": (SCRIPT_DIR / "clip_color_selector.py").resolve(),
    "prompt": (SCRIPT_DIR / "prompt_spec.py").resolve(),
    "design": (SCRIPT_DIR / "design.md").resolve(),
    "protocol": (SCRIPT_DIR / "protocol.md").resolve(),
}
RAW_PID_PATTERN = re.compile(r"^(-?\d+)_c(\d+)_f(\d+)\.jpg$")


def frozen_contract_payload():
    return {
        "oracle_count": core.ORACLE_COUNT,
        "selection_salt": core.SELECTION_SALT,
        "caliper_blind_salt": core.CALIPER_BLIND_SALT,
        "bootstrap_repetitions": 10000,
        "formal_seed": FORMAL_SEED,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "anchor_count": core.ANCHOR_COUNT,
        "fixed_anchors": [list(value) for value in core.FIXED_ANCHORS],
        "aspect_ratios": list(core.ASPECT_RATIOS),
        "area_fraction": core.AREA_FRACTION,
        "alpha": core.ALPHA,
        "roa_count": core.ROA_COUNT,
        "color_names": list(core.COLOR_NAMES),
        "color_prototype_rgb": [
            list(value) for value in core.COLOR_PROTOTYPE_RGB
        ],
        "color_lab_radius": list(core.COLOR_LAB_RADIUS),
        "blind_thresholds": {
            "anatomy_target_min": core.ANATOMY_TARGET_MIN,
            "anatomy_non_target_mean_max": (
                core.ANATOMY_NON_TARGET_MEAN_MAX
            ),
            "anatomy_non_target_single_max": (
                core.ANATOMY_NON_TARGET_SINGLE_MAX
            ),
            "color_presence_min": core.COLOR_PRESENCE_MIN,
            "color_capture_min": core.COLOR_CAPTURE_MIN,
            "color_purity_min": core.COLOR_PURITY_MIN,
            "color_absolute_drop_min": core.COLOR_ABSOLUTE_DROP_MIN,
            "color_relative_drop_min": core.COLOR_RELATIVE_DROP_MIN,
            "color_component_pixels_min": (
                core.COLOR_COMPONENT_PIXELS_MIN
            ),
            "color_component_ratio_min": (
                core.COLOR_COMPONENT_RATIO_MIN
            ),
        },
        "caliper_thresholds": {
            "area_pixel_difference_max": 1,
            "centroid_linf_max": 0.01,
            "aspect_ratio_factor_max": 1.25,
            "d0_displacement_difference_max": 0.010,
            "d0_ce_change_difference_max": 0.25,
            "clean_reference_candidate_top5_required": True,
        },
        "go_thresholds": {
            "quartet_matched_min": 461,
            "strong_pair_matched_min": 461,
            "pc_y_min": 359,
            "slot_pc_y_min": 64,
            "delta_c_given_p1_min": 0.08,
            "delta_p_given_c1_min": 0.08,
            "delta_c_given_p0_min": 0.04,
            "delta_p_given_c0_min": 0.04,
            "interaction_min": 0.04,
            "selector_agreement_min": 0.60,
            "agreement_gap_min": 0.10,
            "top5_drop_max": 0.05,
            "top5_intersection_min": 0.90,
        },
        "variant_order": list(VARIANT_ORDER),
    }


def frozen_contract_sha256():
    return hashlib.sha256(
        stable_json_bytes(frozen_contract_payload())
    ).hexdigest()


class RowDataFailure(Exception):
    """A frozen-data failure that keeps the row but does not invalidate code."""

    def __init__(self, code, message):
        self.code = str(code)
        self.message = str(message)
        super().__init__(self.code + ": " + self.message)


class ExplodingPose(dict):
    """Sentinel proving the sealed D0 never consumes external pose."""

    accesses = 0

    @classmethod
    def reset(cls):
        cls.accesses = 0

    def _fail(self, operation):
        type(self).accesses += 1
        raise RuntimeError("sealed D0 accessed external pose via " + operation)

    def __getitem__(self, key):
        del key
        return self._fail("getitem")

    def get(self, key, default=None):
        del key, default
        return self._fail("get")

    def __iter__(self):
        return self._fail("iter")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
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


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("fresh JSON target required: {}".format(path))
    with temporary.open("xb") as handle:
        handle.write(stable_json_bytes(payload))
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


def _readback_json(path, expected):
    observed = json.loads(Path(path).read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("JSON exact readback mismatch: {}".format(path))
    return observed


def _readback_npz(path, expected):
    output = {}
    with np.load(str(path), allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected):
            raise RuntimeError("NPZ schema mismatch: {}".format(path))
        for key, value in expected.items():
            observed = arrays[key]
            if observed.dtype != value.dtype or observed.shape != value.shape:
                raise RuntimeError("NPZ dtype/shape mismatch: " + key)
            if not np.array_equal(observed, value):
                raise RuntimeError("NPZ exact readback mismatch: " + key)
            if observed.dtype.kind in "fc" and not np.isfinite(observed).all():
                raise RuntimeError("NPZ nonfinite value: " + key)
            output[key] = observed.copy()
    return output


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


def git_file_is_tracked(repo_root, path):
    relative = Path(path).resolve().relative_to(Path(repo_root).resolve())
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "--error-unmatch",
            relative.as_posix(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def assert_no_cuda_compute_processes():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if output:
        raise RuntimeError("pre-seal CUDA compute process exists: " + output)


def configure_formal_determinism():
    random.seed(FORMAL_SEED)
    np.random.seed(FORMAL_SEED)
    torch.manual_seed(FORMAL_SEED)
    torch.cuda.manual_seed_all(FORMAL_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    deterministic = getattr(torch, "use_deterministic_algorithms", None)
    if deterministic is None:
        raise RuntimeError("PyTorch lacks deterministic-algorithm enforcement")
    deterministic(True)
    state = {
        "python_random_seed": FORMAL_SEED,
        "numpy_seed": FORMAL_SEED,
        "torch_seed": FORMAL_SEED,
        "cuda_seed": FORMAL_SEED,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    expected = {
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "deterministic_algorithms": True,
    }
    if any(state[name] is not value for name, value in expected.items()):
        raise RuntimeError("formal deterministic state readback mismatch")
    return state


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
        raise RuntimeError("oracle selected a non-train RGB")
    if image_path != (dataset_root / value).resolve():
        raise RuntimeError("official RGB path normalization mismatch")
    return value


def select_rows(records, dataset_root):
    rows = []
    seen = set()
    for record_index, (image_path, train_label, camera, track) in enumerate(
        records
    ):
        relative_path = _safe_relative(image_path, dataset_root)
        if relative_path in seen:
            raise RuntimeError("duplicate official train relative path")
        seen.add(relative_path)
        match = RAW_PID_PATTERN.fullmatch(Path(image_path).name)
        if match is None:
            raise RuntimeError("could not parse raw PID")
        raw_pid = int(match.group(1))
        rows.append(
            {
                "relative_path": relative_path,
                "record_index": int(record_index),
                "train_label": int(train_label),
                "raw_pid": int(raw_pid),
                "camera": int(camera),
                "track": int(track),
                "image_path": str(Path(image_path).resolve()),
            }
        )
    if len(rows) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError("official train count mismatch")
    selected = core.select_oracle_rows(rows, count=core.ORACLE_COUNT)
    if [row["oracle_index"] for row in selected] != list(
        range(core.ORACLE_COUNT)
    ):
        raise RuntimeError("oracle row index mismatch")
    if len({row["relative_path"] for row in selected}) != core.ORACLE_COUNT:
        raise RuntimeError("oracle relative paths are not unique")
    return selected


def read_canonical_rgb(image_path):
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_size = tuple(map(int, image.size))
            resampling = getattr(Image, "Resampling", Image)
            image = image.resize(
                (core.IMAGE_WIDTH, core.IMAGE_HEIGHT),
                resample=resampling.BICUBIC,
            )
            array = np.asarray(image, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise RowDataFailure("rgb_decode", str(error)) from error
    if array.shape != (core.IMAGE_HEIGHT, core.IMAGE_WIDTH, 3):
        raise RowDataFailure("rgb_shape", str(array.shape))
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
    if (
        tensor.shape != (3, core.IMAGE_HEIGHT, core.IMAGE_WIDTH)
        or not bool(torch.isfinite(tensor).all())
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise RowDataFailure("rgb_tensor", "canonical RGB contract failed")
    return tensor, original_size


def resize_pose_to_canonical(pose):
    width, height = map(int, pose.image_size)
    if width <= 0 or height <= 0:
        raise RowDataFailure("pose_size", "non-positive image size")
    keypoints = pose.keypoints.clone().float()
    keypoints[:, 0] *= core.IMAGE_WIDTH / float(width)
    keypoints[:, 1] *= core.IMAGE_HEIGHT / float(height)
    valid = pose.valid.clone().bool()
    if (
        keypoints.shape != (17, 2)
        or valid.shape != (17,)
        or not bool(torch.isfinite(keypoints).all())
    ):
        raise RowDataFailure("pose_tensor", "canonical pose contract failed")
    return keypoints, valid


def get_verified_pose(pose_store, image_path):
    # Missing lookup/file and RGB-SHA disagreement contradict the already
    # validated frozen asset, so they invalidate the oracle rather than
    # becoming a scientific zero row.
    return pose_store.get(image_path, verify_image_sha=True)


def proposal_metadata(proposal):
    return {
        key: proposal[key]
        for key in (
            "candidate_index",
            "anchor_index",
            "aspect_index",
            "aspect",
            "anchor_valid",
            "top",
            "left",
            "height",
            "width",
            "area_pixels",
            "area_fraction",
            "centroid_y",
            "centroid_x",
            "mask_sha256",
        )
    }


def proposal_digest(proposals):
    fields = []
    for proposal in proposals:
        fields.extend(
            (
                proposal["candidate_index"],
                proposal["anchor_index"],
                proposal["aspect_index"],
                proposal["area_pixels"],
                proposal["top"],
                proposal["left"],
                proposal["height"],
                proposal["width"],
                proposal["mask_sha256"],
            )
        )
    return core.ordered_digest(fields)


def edited_stack(rgb, proposals, fill):
    if len(proposals) != core.ACTIVE_PROPOSALS_PER_IMAGE:
        raise RuntimeError("active proposal count mismatch")
    values = []
    for local_index, proposal in enumerate(proposals):
        if int(proposal["aspect_index"]) != local_index:
            raise RuntimeError("active proposal order mismatch")
        value = core.apply_candidate(rgb, proposal["mask"], fill)
        if not torch.equal(value[:, ~proposal["mask"]], rgb[:, ~proposal["mask"]]):
            raise RuntimeError("mask exterior changed")
        values.append(value)
    output = torch.stack(values, dim=0)
    if output.shape != (
        core.ACTIVE_PROPOSALS_PER_IMAGE,
        3,
        core.IMAGE_HEIGHT,
        core.IMAGE_WIDTH,
    ):
        raise RuntimeError("edited stack shape mismatch")
    if not bool(torch.isfinite(output).all()):
        raise RowDataFailure("edited_nonfinite", "nonfinite edited RGB")
    return output


def prepare_row_pixels(row, pose_store, fixed_pool, *, include_roa):
    image_path = Path(row["image_path"]).resolve()
    pose = get_verified_pose(pose_store, image_path)
    rgb, decoded_size = read_canonical_rgb(image_path)
    if decoded_size != tuple(map(int, pose.image_size)):
        raise RowDataFailure("rgb_pose_size", "RGB/pose size mismatch")
    if pose.relative_path != row["relative_path"]:
        raise RuntimeError("pose/row relative-path mismatch")
    keypoints, valid = resize_pose_to_canonical(pose)
    pose_pool, fields, region_valid = core.generate_pose_proposals(
        keypoints, valid
    )
    if (
        len(pose_pool) != core.PROPOSALS_PER_POOL
        or len(fixed_pool) != core.PROPOSALS_PER_POOL
        or fields.shape
        != (
            core.ANCHOR_COUNT,
            core.IMAGE_HEIGHT,
            core.IMAGE_WIDTH,
        )
        or region_valid.shape != (core.ANCHOR_COUNT,)
        or not bool(torch.isfinite(fields).all())
    ):
        raise RuntimeError("proposal/pose-field schema failure")
    oracle_index = int(row["oracle_index"])
    pose_active = core.active_proposals(pose_pool, oracle_index)
    fixed_active = core.active_proposals(fixed_pool, oracle_index)
    fill = core.deterministic_fill(row["relative_path"])
    pose_edited = edited_stack(rgb, pose_active, fill)
    fixed_edited = edited_stack(rgb, fixed_active, fill)
    output = {
        "rgb": rgb,
        "image_sha256": str(pose.image_sha256),
        "pose_pool": pose_pool,
        "fixed_pool": fixed_pool,
        "pose_active": pose_active,
        "fixed_active": fixed_active,
        "pose_edited": pose_edited,
        "fixed_edited": fixed_edited,
        "pose_fields": fields,
        "region_valid": region_valid,
        "fill": fill,
    }
    if include_roa:
        roa_masks = core.deterministic_roa_masks(row["relative_path"])
        if len(roa_masks) != core.ROA_COUNT:
            raise RuntimeError("ROA count mismatch")
        roa_edited = torch.stack(
            [core.apply_candidate(rgb, mask, fill) for mask in roa_masks],
            dim=0,
        )
        if roa_edited.shape != (
            core.ROA_COUNT,
            3,
            core.IMAGE_HEIGHT,
            core.IMAGE_WIDTH,
        ):
            raise RuntimeError("ROA stack shape mismatch")
        output["roa_masks"] = roa_masks
        output["roa_edited"] = roa_edited
    return output


def _checkpoint_state(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("unexpected D0 checkpoint payload")
    return payload


class D0Evaluator:
    """Strict sealed-D0 evaluator with the frozen 23-variant row order."""

    def __init__(
        self,
        *,
        config_path,
        checkpoint_path,
        dataset,
        device,
        microbatch,
    ):
        from config import cfg
        from model import make_model

        if int(microbatch) != MICROBATCH:
            raise RuntimeError("formal D0 microbatch mismatch")
        config_path = Path(config_path).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()
        if sha256_file(config_path) != EXPECTED_D0_CONFIG_SHA256:
            raise RuntimeError("D0 config SHA256 mismatch")
        if sha256_file(checkpoint_path) != EXPECTED_D0_SHA256:
            raise RuntimeError("D0 checkpoint SHA256 mismatch")
        local_cfg = cfg.clone()
        local_cfg.merge_from_file(str(config_path))
        local_cfg.defrost()
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
        incompatible = model.load_state_dict(
            _checkpoint_state(payload), strict=True
        )
        del payload
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict D0 state load failed")
        for name, value in model.state_dict().items():
            if value.is_floating_point() and not bool(
                torch.isfinite(value).all()
            ):
                raise RuntimeError("nonfinite sealed D0 state: " + name)
        self.model = model.to(device).eval()
        self.dataset = dataset
        self.device = torch.device(device)
        self.microbatch = int(microbatch)
        self.mean = torch.tensor(
            local_cfg.INPUT.PIXEL_MEAN,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            local_cfg.INPUT.PIXEL_STD,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = EXPECTED_D0_SHA256

    @torch.inference_mode()
    def infer_row(self, variants, train_label):
        if variants.shape != (
            VARIANTS_PER_ROW,
            3,
            core.IMAGE_HEIGHT,
            core.IMAGE_WIDTH,
        ):
            raise RuntimeError("D0 row variant schema mismatch")
        label = int(train_label)
        if label < 0 or label >= self.dataset.num_train_pids:
            raise RuntimeError("D0 train label is outside classifier")
        descriptors = []
        logits = []
        ExplodingPose.reset()
        sentinel = ExplodingPose()
        for start in range(0, VARIANTS_PER_ROW, self.microbatch):
            stop = min(start + self.microbatch, VARIANTS_PER_ROW)
            batch = variants[start:stop].to(self.device)
            normalized = (batch - self.mean) / self.std
            output = self.model(
                normalized,
                pose_batch=sentinel,
                tapf_epoch=None,
            )
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or not torch.is_tensor(output[0])
                or output[0].ndim != 2
            ):
                raise RuntimeError("unexpected sealed D0 eval output")
            descriptor = output[0]
            classifier_input = self.model.dropout(
                self.model.bottleneck(descriptor)
            )
            cls_score = self.model.classifier(classifier_input)
            if cls_score.shape != (
                len(descriptor),
                self.dataset.num_train_pids,
            ):
                raise RuntimeError("D0 classifier schema mismatch")
            if not bool(torch.isfinite(descriptor).all()) or not bool(
                torch.isfinite(cls_score).all()
            ):
                raise RowDataFailure(
                    "d0_nonfinite", "nonfinite descriptor or classifier logit"
                )
            descriptors.append(descriptor.detach().cpu())
            logits.append(cls_score.detach().cpu())
        if ExplodingPose.accesses != 0:
            raise RuntimeError("sealed D0 consumed external pose")
        descriptor = torch.cat(descriptors, dim=0).float()
        cls_score = torch.cat(logits, dim=0).float()
        labels = torch.full((VARIANTS_PER_ROW,), label, dtype=torch.long)
        ce = F.cross_entropy(cls_score, labels, reduction="none")
        top5 = cls_score.topk(k=5, dim=1).indices.eq(
            labels[:, None]
        ).any(dim=1)
        descriptor = F.normalize(descriptor, dim=1)
        displacement = 1.0 - (
            descriptor[1:] * descriptor[:1]
        ).sum(dim=1)
        ce_change = ce[1:] - ce[:1]
        for value in (descriptor, cls_score, ce, displacement, ce_change):
            if not bool(torch.isfinite(value).all()):
                raise RowDataFailure(
                    "d0_nonfinite", "nonfinite derived D0 quantity"
                )
        if displacement.shape != (VARIANTS_PER_ROW - 1,):
            raise RuntimeError("D0 displacement schema mismatch")
        return {
            "clean_top5": bool(top5[0]),
            "pose_top5": top5[1:8].numpy().astype(np.bool_, copy=True),
            "fixed_top5": top5[8:15].numpy().astype(np.bool_, copy=True),
            "roa_top5": top5[15:23].numpy().astype(np.bool_, copy=True),
            "pose_shift": displacement[0:7].numpy().astype(
                np.float64, copy=True
            ),
            "fixed_shift": displacement[7:14].numpy().astype(
                np.float64, copy=True
            ),
            "roa_shift": displacement[14:22].numpy().astype(
                np.float64, copy=True
            ),
            "pose_ce_change": ce_change[0:7].numpy().astype(
                np.float64, copy=True
            ),
            "fixed_ce_change": ce_change[7:14].numpy().astype(
                np.float64, copy=True
            ),
            "roa_ce_change": ce_change[14:22].numpy().astype(
                np.float64, copy=True
            ),
            "descriptor_sha256": sha256_tensor(descriptor),
            "logit_sha256": sha256_tensor(cls_score),
            "ce_sha256": sha256_tensor(ce),
            "top5_sha256": sha256_tensor(top5),
            "external_pose_accesses": int(ExplodingPose.accesses),
        }

    def close(self):
        model = self.model
        self.model = None
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if sha256_file(self.checkpoint_path) != self.checkpoint_sha256:
            raise RuntimeError("D0 checkpoint changed during oracle")


def make_cache(selected):
    n = core.ORACLE_COUNT
    seven = core.ACTIVE_PROPOSALS_PER_IMAGE
    ten = len(core.COLOR_NAMES)
    pool_count = 2
    proposal_count = core.PROPOSALS_PER_POOL
    blind_count = pool_count * seven
    arrays = {
        "schema": np.asarray([CACHE_SCHEMA_NAME]),
        "oracle_index": np.arange(n, dtype=np.int32),
        "relative_paths": np.asarray(
            [row["relative_path"] for row in selected]
        ),
        "selection_hash": np.asarray(
            [
                core.sha256_text(
                    core.SELECTION_SALT + "\0" + row["relative_path"]
                )
                for row in selected
            ]
        ),
        "train_label": np.asarray(
            [row["train_label"] for row in selected], dtype=np.int64
        ),
        "raw_pid": np.asarray(
            [row["raw_pid"] for row in selected], dtype=np.int64
        ),
        "camera": np.asarray(
            [row["camera"] for row in selected], dtype=np.int16
        ),
        "active_anchor": (
            np.arange(n, dtype=np.int16) % core.ANCHOR_COUNT
        ).astype(np.int8),
        "row_record_present": np.zeros(n, dtype=np.bool_),
        "clip_valid": np.zeros(n, dtype=np.bool_),
        "d0_valid": np.zeros(n, dtype=np.bool_),
        "numeric_finite": np.ones(n, dtype=np.bool_),
        "failure_code": np.full(n, "", dtype="<U64"),
        "image_sha256": np.full(n, "", dtype="<U64"),
        "region_valid": np.zeros((n, core.ANCHOR_COUNT), dtype=np.bool_),
        "pose_anchor_valid": np.zeros(
            (n, core.ANCHOR_COUNT), dtype=np.bool_
        ),
        "proposal_candidate_index": np.full(
            (n, pool_count, proposal_count), -1, dtype=np.int16
        ),
        "proposal_anchor_index": np.full(
            (n, pool_count, proposal_count), -1, dtype=np.int8
        ),
        "proposal_aspect_index": np.full(
            (n, pool_count, proposal_count), -1, dtype=np.int8
        ),
        "proposal_area_pixels": np.zeros(
            (n, pool_count, proposal_count), dtype=np.int32
        ),
        "proposal_centroid": np.zeros(
            (n, pool_count, proposal_count, 2), dtype=np.float64
        ),
        "proposal_anchor_valid": np.zeros(
            (n, pool_count, proposal_count), dtype=np.bool_
        ),
        "proposal_mask_sha256": np.full(
            (n, pool_count, proposal_count), "", dtype="<U64"
        ),
        "pose_candidate_index": np.full((n, seven), -1, dtype=np.int16),
        "fixed_candidate_index": np.full((n, seven), -1, dtype=np.int16),
        "pose_mask_sha256": np.full((n, seven), "", dtype="<U64"),
        "fixed_mask_sha256": np.full((n, seven), "", dtype="<U64"),
        "roa_mask_sha256": np.full(
            (n, core.ROA_COUNT), "", dtype="<U64"
        ),
        "clip_pose_drop": np.zeros((n, seven, ten), dtype=np.float64),
        "clip_fixed_drop": np.zeros((n, seven, ten), dtype=np.float64),
        "clip_selected_local": np.full((n, 2), -1, dtype=np.int8),
        "clip_selected_color": np.full((n, 2), -1, dtype=np.int8),
        "clip_selected_score": np.zeros((n, 2), dtype=np.float64),
        "clean_top5": np.zeros(n, dtype=np.bool_),
        "pose_top5": np.zeros((n, seven), dtype=np.bool_),
        "fixed_top5": np.zeros((n, seven), dtype=np.bool_),
        "pose_shift": np.zeros((n, seven), dtype=np.float64),
        "fixed_shift": np.zeros((n, seven), dtype=np.float64),
        "roa_shift": np.zeros((n, core.ROA_COUNT), dtype=np.float64),
        "pose_ce_change": np.zeros((n, seven), dtype=np.float64),
        "fixed_ce_change": np.zeros((n, seven), dtype=np.float64),
        "pose_identity_safe": np.zeros((n, seven), dtype=np.bool_),
        "fixed_identity_safe": np.zeros((n, seven), dtype=np.bool_),
        "blind_color": np.full((n, blind_count), -1, dtype=np.int8),
        "blind_target_slot": np.full(
            (n, blind_count), -1, dtype=np.int8
        ),
        "blind_score": np.zeros((n, blind_count), dtype=np.float64),
        "blind_target_coverage": np.zeros(
            (n, blind_count), dtype=np.float64
        ),
        "blind_non_target_mean": np.zeros(
            (n, blind_count), dtype=np.float64
        ),
        "blind_non_target_max": np.zeros(
            (n, blind_count), dtype=np.float64
        ),
        "blind_presence": np.zeros(
            (n, blind_count), dtype=np.float64
        ),
        "blind_capture": np.zeros((n, blind_count), dtype=np.float64),
        "blind_purity": np.zeros((n, blind_count), dtype=np.float64),
        "blind_component_pixels": np.zeros(
            (n, blind_count), dtype=np.int32
        ),
        "blind_component_ratio": np.zeros(
            (n, blind_count), dtype=np.float64
        ),
        "blind_absolute_drop": np.zeros(
            (n, blind_count), dtype=np.float64
        ),
        "blind_relative_drop": np.zeros(
            (n, blind_count), dtype=np.float64
        ),
        "blind_identity_safe": np.zeros(
            (n, blind_count), dtype=np.bool_
        ),
        "blind_anatomy_valid": np.zeros(
            (n, blind_count), dtype=np.bool_
        ),
        "blind_coherent_color": np.zeros(
            (n, blind_count), dtype=np.bool_
        ),
        "blind_y": np.zeros((n, blind_count), dtype=np.int8),
        "arm_selected_local": np.full((n, 4), -1, dtype=np.int8),
        "arm_complete": np.zeros((n, 4), dtype=np.bool_),
        "arm_y": np.zeros((n, 4), dtype=np.int8),
        "quartet_edges": np.zeros((n, 4), dtype=np.bool_),
        "strong_selected_local": np.full((n, 2), -1, dtype=np.int8),
        "strong_complete": np.zeros((n, 2), dtype=np.bool_),
        "strong_y": np.zeros((n, 2), dtype=np.int8),
        "strong_pair_match": np.zeros((n, 2), dtype=np.bool_),
        "agreement_correct_base": np.zeros(n, dtype=np.bool_),
        "agreement_shuffle_base": np.zeros(n, dtype=np.bool_),
        "pc_top5": np.zeros(n, dtype=np.bool_),
    }
    return arrays


def set_row_failure(cache, row_index, failure):
    cache["failure_code"][row_index] = str(failure.code)[:64]
    if "nonfinite" in str(failure.code):
        cache["numeric_finite"][row_index] = False


def _active_metadata_into_cache(cache, row_index, payload):
    cache["image_sha256"][row_index] = payload["image_sha256"]
    cache["region_valid"][row_index] = (
        payload["region_valid"].numpy().astype(np.bool_, copy=False)
    )
    for pool_name, active_name in (
        ("pose", "pose_active"),
        ("fixed", "fixed_active"),
    ):
        pool_index = 0 if pool_name == "pose" else 1
        complete_pool = payload[pool_name + "_pool"]
        if len(complete_pool) != core.PROPOSALS_PER_POOL:
            raise RuntimeError("complete proposal pool count mismatch")
        cache["proposal_candidate_index"][row_index, pool_index] = np.asarray(
            [item["candidate_index"] for item in complete_pool],
            dtype=np.int16,
        )
        cache["proposal_anchor_index"][row_index, pool_index] = np.asarray(
            [item["anchor_index"] for item in complete_pool],
            dtype=np.int8,
        )
        cache["proposal_aspect_index"][row_index, pool_index] = np.asarray(
            [item["aspect_index"] for item in complete_pool],
            dtype=np.int8,
        )
        cache["proposal_area_pixels"][row_index, pool_index] = np.asarray(
            [item["area_pixels"] for item in complete_pool],
            dtype=np.int32,
        )
        cache["proposal_centroid"][row_index, pool_index] = np.asarray(
            [
                (item["centroid_y"], item["centroid_x"])
                for item in complete_pool
            ],
            dtype=np.float64,
        )
        cache["proposal_anchor_valid"][row_index, pool_index] = np.asarray(
            [item["anchor_valid"] for item in complete_pool],
            dtype=np.bool_,
        )
        cache["proposal_mask_sha256"][row_index, pool_index] = np.asarray(
            [item["mask_sha256"] for item in complete_pool]
        )
        active = payload[active_name]
        cache[pool_name + "_candidate_index"][row_index] = np.asarray(
            [item["candidate_index"] for item in active], dtype=np.int16
        )
        cache[pool_name + "_mask_sha256"][row_index] = np.asarray(
            [item["mask_sha256"] for item in active]
        )
    cache["pose_anchor_valid"][row_index] = np.asarray(
        [
            bool(
                next(
                    item["anchor_valid"]
                    for item in payload["pose_pool"]
                    if int(item["anchor_index"]) == anchor_index
                )
            )
            for anchor_index in range(core.ANCHOR_COUNT)
        ],
        dtype=np.bool_,
    )


def run_clip_phase(
    selected,
    pose_store,
    fixed_pool,
    selector,
    device,
    cache,
    progress,
):
    records = [None] * core.ORACLE_COUNT
    for row in selected:
        index = int(row["oracle_index"])
        progress["stage"] = "clip_phase"
        progress["oracle_index"] = index
        progress["relative_path"] = row["relative_path"]
        try:
            payload = prepare_row_pixels(
                row, pose_store, fixed_pool, include_roa=False
            )
            with torch.inference_mode():
                pose_drop = selector(
                    payload["rgb"].to(device),
                    payload["pose_edited"].to(device),
                ).detach().cpu()
                fixed_drop = selector(
                    payload["rgb"].to(device),
                    payload["fixed_edited"].to(device),
                ).detach().cpu()
            expected = (
                core.ACTIVE_PROPOSALS_PER_IMAGE,
                len(core.COLOR_NAMES),
            )
            if (
                pose_drop.shape != expected
                or fixed_drop.shape != expected
            ):
                raise RuntimeError("CLIP output shape contract failed")
            if (
                not bool(torch.isfinite(pose_drop).all())
                or not bool(torch.isfinite(fixed_drop).all())
            ):
                raise RowDataFailure(
                    "clip_nonfinite", "nonfinite CLIP selector output"
                )
            pose_selection = core.select_clip_candidate(pose_drop)
            fixed_selection = core.select_clip_candidate(fixed_drop)
            _active_metadata_into_cache(cache, index, payload)
            cache["clip_pose_drop"][index] = pose_drop.numpy()
            cache["clip_fixed_drop"][index] = fixed_drop.numpy()
            cache["clip_selected_local"][index] = (
                pose_selection["aspect_index"],
                fixed_selection["aspect_index"],
            )
            cache["clip_selected_color"][index] = (
                pose_selection["selector_color_index"],
                fixed_selection["selector_color_index"],
            )
            cache["clip_selected_score"][index] = (
                pose_selection["selector_score"],
                fixed_selection["selector_score"],
            )
            cache["clip_valid"][index] = True
            records[index] = {
                "pose_selection": pose_selection,
                "fixed_selection": fixed_selection,
                "pose_pool_digest": proposal_digest(payload["pose_pool"]),
                "fixed_pool_digest": proposal_digest(payload["fixed_pool"]),
                "pose_active_digest": proposal_digest(
                    payload["pose_active"]
                ),
                "fixed_active_digest": proposal_digest(
                    payload["fixed_active"]
                ),
                "pose_drop_sha256": sha256_tensor(pose_drop),
                "fixed_drop_sha256": sha256_tensor(fixed_drop),
            }
        except RowDataFailure as failure:
            set_row_failure(cache, index, failure)
            records[index] = {
                "failure_code": failure.code,
                "failure_message": failure.message,
            }
        progress["clip_completed_rows"] = index + 1
        if (index + 1) % 8 == 0 or index + 1 == core.ORACLE_COUNT:
            print(
                json.dumps(
                    {
                        "stage": "clip_phase",
                        "completed_rows": index + 1,
                        "total_rows": core.ORACLE_COUNT,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if any(record is None for record in records):
        raise RuntimeError("CLIP phase lost an oracle row")
    return records


def _identity_rows(d0):
    pose_identity = []
    fixed_identity = []
    for local_index in range(core.ACTIVE_PROPOSALS_PER_IMAGE):
        pose_identity.append(
            core.d0_identity_gate(
                d0["clean_top5"],
                d0["pose_top5"][local_index],
                d0["pose_shift"][local_index],
                d0["roa_shift"],
            )
        )
        fixed_identity.append(
            core.d0_identity_gate(
                d0["clean_top5"],
                d0["fixed_top5"][local_index],
                d0["fixed_shift"][local_index],
                d0["roa_shift"],
            )
        )
    return pose_identity, fixed_identity


def _blind_candidates(payload, pose_identity, fixed_identity):
    expected_anchor = int(
        payload["pose_active"][0]["anchor_index"]
    )
    pose_evaluations = []
    fixed_evaluations = []
    for local_index in range(core.ACTIVE_PROPOSALS_PER_IMAGE):
        pose_evaluations.append(
            core.blind_evaluate(
                payload["rgb"],
                payload["pose_edited"][local_index],
                payload["pose_active"][local_index]["mask"],
                payload["pose_fields"],
                payload["region_valid"],
                expected_anchor_index=expected_anchor,
                identity_safe=pose_identity[local_index]["identity_safe"],
            )
        )
        fixed_evaluations.append(
            core.blind_evaluate(
                payload["rgb"],
                payload["fixed_edited"][local_index],
                payload["fixed_active"][local_index]["mask"],
                payload["pose_fields"],
                payload["region_valid"],
                expected_anchor_index=expected_anchor,
                identity_safe=fixed_identity[local_index]["identity_safe"],
            )
        )
    return pose_evaluations, fixed_evaluations


def _arm_record(
    row_id,
    arm_name,
    selected_local,
    active,
    evaluation,
    edges,
):
    complete = selected_local is not None and evaluation is not None
    if complete:
        proposal = active[int(selected_local)]
        blind = dict(evaluation)
        outcome = int(blind["Y"])
        candidate_index = int(proposal["candidate_index"])
        mask_sha = str(proposal["mask_sha256"])
    else:
        blind = None
        outcome = 0
        candidate_index = -1
        mask_sha = None
    return {
        "row_id": str(row_id),
        "arm_name": str(arm_name),
        "arm_complete": bool(complete),
        "selected_local_index": (
            int(selected_local) if selected_local is not None else -1
        ),
        "candidate_index": candidate_index,
        "mask_sha256": mask_sha,
        "match_edges": dict(edges),
        "blind": blind,
        "Y": int(outcome),
    }


def _failed_row_record(row, failure_code, failure_message):
    row_id = row["relative_path"]
    edges = {name: False for name in EDGE_ORDER}
    arms = {
        name: {
            "row_id": row_id,
            "arm_name": name,
            "arm_complete": False,
            "selected_local_index": -1,
            "candidate_index": -1,
            "mask_sha256": None,
            "match_edges": dict(edges),
            "blind": None,
            "Y": 0,
        }
        for name in ARM_ORDER
    }
    strong = {}
    for name in STRONG_ORDER:
        strong[name] = {
            "reference": {
                "row_id": row_id,
                "arm_complete": False,
                "pair_match_ok": False,
                "Y": 0,
                "blind": None,
                "candidate": None,
            },
            "control": {
                "row_id": row_id,
                "arm_complete": False,
                "pair_match_ok": False,
                "Y": 0,
                "blind": None,
                "candidate": None,
            },
            "selected_local_index": -1,
            "equivalent_to_pc": False,
        }
    return {
        "oracle_index": int(row["oracle_index"]),
        "row_id": row_id,
        "selection_hash": core.sha256_text(
            core.SELECTION_SALT + "\0" + row_id
        ),
        "train_label": int(row["train_label"]),
        "raw_pid": int(row["raw_pid"]),
        "camera": int(row["camera"]),
        "active_anchor": int(row["oracle_index"]) % core.ANCHOR_COUNT,
        "record_complete": True,
        "data_failure": True,
        "failure_code": str(failure_code),
        "failure_message": str(failure_message),
        "arms": arms,
        "strong_controls": strong,
        "agreement_correct_base": False,
        "agreement_shuffle_base": False,
        "d0": None,
        "proposals": None,
    }


def _process_valid_row(row, clip_record, payload, d0, cache):
    index = int(row["oracle_index"])
    if (
        proposal_digest(payload["pose_pool"])
        != clip_record["pose_pool_digest"]
        or proposal_digest(payload["fixed_pool"])
        != clip_record["fixed_pool_digest"]
        or proposal_digest(payload["pose_active"])
        != clip_record["pose_active_digest"]
        or proposal_digest(payload["fixed_active"])
        != clip_record["fixed_active_digest"]
    ):
        raise RuntimeError("CLIP/D0 phase proposal digest mismatch")
    if cache["image_sha256"][index] != payload["image_sha256"]:
        raise RuntimeError("CLIP/D0 phase image SHA mismatch")
    roa_shas = [
        core.sha256_bytes(
            mask.numpy().astype(np.uint8, copy=False).tobytes()
        )
        for mask in payload["roa_masks"]
    ]
    cache["roa_mask_sha256"][index] = np.asarray(roa_shas)

    pose_identity, fixed_identity = _identity_rows(d0)
    pose_evaluations, fixed_evaluations = _blind_candidates(
        payload, pose_identity, fixed_identity
    )
    cache["clean_top5"][index] = d0["clean_top5"]
    cache["pose_top5"][index] = d0["pose_top5"]
    cache["fixed_top5"][index] = d0["fixed_top5"]
    cache["pose_shift"][index] = d0["pose_shift"]
    cache["fixed_shift"][index] = d0["fixed_shift"]
    cache["roa_shift"][index] = d0["roa_shift"]
    cache["pose_ce_change"][index] = d0["pose_ce_change"]
    cache["fixed_ce_change"][index] = d0["fixed_ce_change"]
    cache["pose_identity_safe"][index] = np.asarray(
        [item["identity_safe"] for item in pose_identity], dtype=np.bool_
    )
    cache["fixed_identity_safe"][index] = np.asarray(
        [item["identity_safe"] for item in fixed_identity], dtype=np.bool_
    )
    for local_index, evaluation in enumerate(
        pose_evaluations + fixed_evaluations
    ):
        cache["blind_color"][index, local_index] = evaluation[
            "blind_color_index"
        ]
        cache["blind_score"][index, local_index] = evaluation["blind_score"]
        cache["blind_target_slot"][index, local_index] = evaluation[
            "target_slot"
        ]
        cache["blind_target_coverage"][index, local_index] = evaluation[
            "target_coverage"
        ]
        cache["blind_non_target_mean"][index, local_index] = evaluation[
            "non_target_coverage_mean"
        ]
        cache["blind_non_target_max"][index, local_index] = evaluation[
            "non_target_coverage_max"
        ]
        cache["blind_presence"][index, local_index] = evaluation["presence"]
        cache["blind_capture"][index, local_index] = evaluation["capture"]
        cache["blind_purity"][index, local_index] = evaluation["purity"]
        cache["blind_component_pixels"][index, local_index] = evaluation[
            "component_pixels"
        ]
        cache["blind_component_ratio"][index, local_index] = evaluation[
            "component_ratio"
        ]
        cache["blind_absolute_drop"][index, local_index] = evaluation[
            "absolute_drop"
        ]
        cache["blind_relative_drop"][index, local_index] = evaluation[
            "relative_drop"
        ]
        cache["blind_identity_safe"][index, local_index] = evaluation[
            "identity_safe"
        ]
        cache["blind_anatomy_valid"][index, local_index] = evaluation[
            "anatomy_valid"
        ]
        cache["blind_coherent_color"][index, local_index] = evaluation[
            "coherent_color_removal"
        ]
        cache["blind_y"][index, local_index] = evaluation["Y"]

    pc_local = int(clip_record["pose_selection"]["aspect_index"])
    clip_only_local = int(
        clip_record["fixed_selection"]["aspect_index"]
    )
    pc_proposal = payload["pose_active"][pc_local]
    clip_only_proposal = payload["fixed_active"][clip_only_local]
    pose_eligible = core.caliper_eligible(
        pc_proposal,
        payload["pose_active"],
        d0["pose_shift"][pc_local],
        d0["pose_shift"],
        d0["pose_ce_change"][pc_local],
        d0["pose_ce_change"],
        d0["clean_top5"],
        d0["pose_top5"][pc_local],
        d0["pose_top5"],
        require_centroid=True,
        allow_reference=False,
    )
    fixed_eligible = core.caliper_eligible(
        clip_only_proposal,
        payload["fixed_active"],
        d0["fixed_shift"][clip_only_local],
        d0["fixed_shift"],
        d0["fixed_ce_change"][clip_only_local],
        d0["fixed_ce_change"],
        d0["clean_top5"],
        d0["fixed_top5"][clip_only_local],
        d0["fixed_top5"],
        require_centroid=True,
        allow_reference=False,
    )
    pose_only_local = core.select_caliper_hash_candidate(
        row["relative_path"], payload["pose_active"], pose_eligible
    )
    neither_local = core.select_caliper_hash_candidate(
        row["relative_path"], payload["fixed_active"], fixed_eligible
    )

    c_given_p1 = bool(
        pose_only_local is not None
        and core.direct_pair_caliper(
            pc_proposal,
            payload["pose_active"][pose_only_local],
            d0["pose_shift"][pc_local],
            d0["pose_shift"][pose_only_local],
            d0["pose_ce_change"][pc_local],
            d0["pose_ce_change"][pose_only_local],
            d0["clean_top5"],
            d0["pose_top5"][pc_local],
            d0["pose_top5"][pose_only_local],
            require_centroid=True,
            require_different_mask=True,
        )
    )
    c_given_p0 = bool(
        neither_local is not None
        and core.direct_pair_caliper(
            clip_only_proposal,
            payload["fixed_active"][neither_local],
            d0["fixed_shift"][clip_only_local],
            d0["fixed_shift"][neither_local],
            d0["fixed_ce_change"][clip_only_local],
            d0["fixed_ce_change"][neither_local],
            d0["clean_top5"],
            d0["fixed_top5"][clip_only_local],
            d0["fixed_top5"][neither_local],
            require_centroid=True,
            require_different_mask=True,
        )
    )
    p_given_c1 = core.direct_pair_caliper(
        pc_proposal,
        clip_only_proposal,
        d0["pose_shift"][pc_local],
        d0["fixed_shift"][clip_only_local],
        d0["pose_ce_change"][pc_local],
        d0["fixed_ce_change"][clip_only_local],
        d0["clean_top5"],
        d0["pose_top5"][pc_local],
        d0["fixed_top5"][clip_only_local],
        require_centroid=False,
        require_different_mask=False,
    )
    p_given_c0 = bool(
        pose_only_local is not None
        and neither_local is not None
        and core.direct_pair_caliper(
            payload["pose_active"][pose_only_local],
            payload["fixed_active"][neither_local],
            d0["pose_shift"][pose_only_local],
            d0["fixed_shift"][neither_local],
            d0["pose_ce_change"][pose_only_local],
            d0["fixed_ce_change"][neither_local],
            d0["clean_top5"],
            d0["pose_top5"][pose_only_local],
            d0["fixed_top5"][neither_local],
            require_centroid=False,
            require_different_mask=False,
        )
    )
    edges = {
        "c_given_p1": bool(c_given_p1),
        "c_given_p0": bool(c_given_p0),
        "p_given_c1": bool(p_given_c1),
        "p_given_c0": bool(p_given_c0),
    }
    arm_selected = {
        "pc": pc_local,
        "pose_only": pose_only_local,
        "clip_only": clip_only_local,
        "neither": neither_local,
    }
    arm_pool = {
        "pc": payload["pose_active"],
        "pose_only": payload["pose_active"],
        "clip_only": payload["fixed_active"],
        "neither": payload["fixed_active"],
    }
    arm_evaluation = {
        "pc": pose_evaluations[pc_local],
        "pose_only": (
            pose_evaluations[pose_only_local]
            if pose_only_local is not None
            else None
        ),
        "clip_only": fixed_evaluations[clip_only_local],
        "neither": (
            fixed_evaluations[neither_local]
            if neither_local is not None
            else None
        ),
    }
    arms = {
        name: _arm_record(
            row["relative_path"],
            name,
            arm_selected[name],
            arm_pool[name],
            arm_evaluation[name],
            edges,
        )
        for name in ARM_ORDER
    }

    strong_base = core.caliper_eligible(
        pc_proposal,
        payload["pose_active"],
        d0["pose_shift"][pc_local],
        d0["pose_shift"],
        d0["pose_ce_change"][pc_local],
        d0["pose_ce_change"],
        d0["clean_top5"],
        d0["pose_top5"][pc_local],
        d0["pose_top5"],
        require_centroid=True,
        allow_reference=True,
    )
    strong_eligible = core.strong_control_eligible(
        strong_base,
        pose_identity[pc_local]["identity_safe"],
        np.asarray(
            [item["identity_safe"] for item in pose_identity],
            dtype=np.bool_,
        ),
    )
    if (
        pose_identity[pc_local]["identity_safe"]
        and not bool(strong_eligible[pc_local])
    ):
        raise RuntimeError("identity-safe P+C reference left strong caliper")
    raw_local = core.select_raw_color_candidate(
        pose_evaluations, strong_eligible
    )
    hard_local = core.select_d0_hard_candidate(
        d0["pose_shift"], strong_eligible
    )
    strong = {}
    for strong_name, local_index in (
        ("raw_color", raw_local),
        ("d0_hard", hard_local),
    ):
        pair_ok = bool(
            local_index is not None
            and strong_eligible[int(local_index)]
            and core.direct_pair_caliper(
                pc_proposal,
                payload["pose_active"][int(local_index)],
                d0["pose_shift"][pc_local],
                d0["pose_shift"][int(local_index)],
                d0["pose_ce_change"][pc_local],
                d0["pose_ce_change"][int(local_index)],
                d0["clean_top5"],
                d0["pose_top5"][pc_local],
                d0["pose_top5"][int(local_index)],
                require_centroid=True,
                require_different_mask=False,
            )
        )
        control_y = (
            int(pose_evaluations[int(local_index)]["Y"])
            if local_index is not None
            else 0
        )
        strong[strong_name] = {
            "reference": {
                "row_id": row["relative_path"],
                "arm_complete": True,
                "pair_match_ok": pair_ok,
                "Y": int(pose_evaluations[pc_local]["Y"]),
                "blind": dict(pose_evaluations[pc_local]),
                "candidate": proposal_metadata(pc_proposal),
            },
            "control": {
                "row_id": row["relative_path"],
                "arm_complete": local_index is not None,
                "pair_match_ok": pair_ok,
                "Y": control_y,
                "blind": (
                    dict(pose_evaluations[int(local_index)])
                    if local_index is not None
                    else None
                ),
                "candidate": (
                    proposal_metadata(
                        payload["pose_active"][int(local_index)]
                    )
                    if local_index is not None
                    else None
                ),
            },
            "selected_local_index": (
                int(local_index) if local_index is not None else -1
            ),
            "candidate_index": (
                int(
                    payload["pose_active"][int(local_index)][
                        "candidate_index"
                    ]
                )
                if local_index is not None
                else -1
            ),
            "equivalent_to_pc": bool(
                local_index is not None and int(local_index) == pc_local
            ),
        }

    pc_blind_color = int(
        pose_evaluations[pc_local]["blind_color_index"]
    )
    selector_color = int(
        clip_record["pose_selection"]["selector_color_index"]
    )
    blind_label_valid = pc_blind_color >= 0
    agreement_correct_base = bool(
        blind_label_valid
        and pose_evaluations[pc_local]["anatomy_valid"]
        and pose_evaluations[pc_local]["coherent_color_removal"]
        and selector_color == pc_blind_color
    )
    agreement_shuffle_base = bool(
        blind_label_valid
        and pose_evaluations[pc_local]["anatomy_valid"]
        and pose_evaluations[pc_local]["coherent_color_removal"]
        and ((selector_color + 1) % len(core.COLOR_NAMES))
        == pc_blind_color
    )

    cache["d0_valid"][index] = True
    cache["pose_identity_safe"][index] = np.asarray(
        [item["identity_safe"] for item in pose_identity], dtype=np.bool_
    )
    cache["fixed_identity_safe"][index] = np.asarray(
        [item["identity_safe"] for item in fixed_identity], dtype=np.bool_
    )
    cache["arm_selected_local"][index] = np.asarray(
        [
            arm_selected[name]
            if arm_selected[name] is not None
            else -1
            for name in ARM_ORDER
        ],
        dtype=np.int8,
    )
    cache["arm_complete"][index] = np.asarray(
        [arms[name]["arm_complete"] for name in ARM_ORDER],
        dtype=np.bool_,
    )
    cache["arm_y"][index] = np.asarray(
        [arms[name]["Y"] for name in ARM_ORDER], dtype=np.int8
    )
    cache["quartet_edges"][index] = np.asarray(
        [edges[name] for name in EDGE_ORDER], dtype=np.bool_
    )
    cache["strong_selected_local"][index] = np.asarray(
        [
            strong[name]["selected_local_index"]
            for name in STRONG_ORDER
        ],
        dtype=np.int8,
    )
    cache["strong_complete"][index] = np.asarray(
        [
            strong[name]["control"]["arm_complete"]
            for name in STRONG_ORDER
        ],
        dtype=np.bool_,
    )
    cache["strong_y"][index] = np.asarray(
        [strong[name]["control"]["Y"] for name in STRONG_ORDER],
        dtype=np.int8,
    )
    cache["strong_pair_match"][index] = np.asarray(
        [
            strong[name]["control"]["pair_match_ok"]
            for name in STRONG_ORDER
        ],
        dtype=np.bool_,
    )
    cache["agreement_correct_base"][index] = agreement_correct_base
    cache["agreement_shuffle_base"][index] = agreement_shuffle_base
    cache["pc_top5"][index] = d0["pose_top5"][pc_local]

    return {
        "oracle_index": index,
        "row_id": row["relative_path"],
        "selection_hash": core.sha256_text(
            core.SELECTION_SALT + "\0" + row["relative_path"]
        ),
        "train_label": int(row["train_label"]),
        "raw_pid": int(row["raw_pid"]),
        "camera": int(row["camera"]),
        "active_anchor": index % core.ANCHOR_COUNT,
        "record_complete": True,
        "data_failure": False,
        "failure_code": "",
        "failure_message": "",
        "image_sha256": payload["image_sha256"],
        "proposals": {
            "pose_pool_digest": proposal_digest(payload["pose_pool"]),
            "fixed_pool_digest": proposal_digest(payload["fixed_pool"]),
            "pose_pool": [
                proposal_metadata(item) for item in payload["pose_pool"]
            ],
            "fixed_pool": [
                proposal_metadata(item) for item in payload["fixed_pool"]
            ],
            "pose_active": [
                proposal_metadata(item)
                for item in payload["pose_active"]
            ],
            "fixed_active": [
                proposal_metadata(item)
                for item in payload["fixed_active"]
            ],
            "roa_mask_sha256": roa_shas,
            "region_valid": payload["region_valid"].tolist(),
        },
        "clip": {
            "pose": dict(clip_record["pose_selection"]),
            "fixed": dict(clip_record["fixed_selection"]),
            "pose_drop_sha256": clip_record["pose_drop_sha256"],
            "fixed_drop_sha256": clip_record["fixed_drop_sha256"],
        },
        "d0": {
            "clean_top5": bool(d0["clean_top5"]),
            "descriptor_sha256": d0["descriptor_sha256"],
            "logit_sha256": d0["logit_sha256"],
            "ce_sha256": d0["ce_sha256"],
            "top5_sha256": d0["top5_sha256"],
            "external_pose_accesses": d0["external_pose_accesses"],
            "pose_identity": pose_identity,
            "fixed_identity": fixed_identity,
            "roa_shift": d0["roa_shift"].tolist(),
        },
        "arms": arms,
        "strong_controls": strong,
        "agreement_correct_base": agreement_correct_base,
        "agreement_shuffle_base": agreement_shuffle_base,
    }


def run_d0_and_blind_phase(
    selected,
    clip_records,
    pose_store,
    fixed_pool,
    evaluator,
    cache,
    progress,
):
    rows = [None] * core.ORACLE_COUNT
    for row in selected:
        index = int(row["oracle_index"])
        progress["stage"] = "d0_and_blind_phase"
        progress["oracle_index"] = index
        progress["relative_path"] = row["relative_path"]
        if not bool(cache["clip_valid"][index]):
            record = clip_records[index]
            rows[index] = _failed_row_record(
                row,
                record["failure_code"],
                record["failure_message"],
            )
            cache["row_record_present"][index] = True
            progress["scientific_completed_rows"] = index + 1
            if (index + 1) % 8 == 0 or index + 1 == core.ORACLE_COUNT:
                print(
                    json.dumps(
                        {
                            "stage": "d0_and_blind_phase",
                            "completed_rows": index + 1,
                            "total_rows": core.ORACLE_COUNT,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            continue
        try:
            payload = prepare_row_pixels(
                row, pose_store, fixed_pool, include_roa=True
            )
            variants = torch.cat(
                (
                    payload["rgb"].unsqueeze(0),
                    payload["pose_edited"],
                    payload["fixed_edited"],
                    payload["roa_edited"],
                ),
                dim=0,
            )
            if variants.shape != (
                VARIANTS_PER_ROW,
                3,
                core.IMAGE_HEIGHT,
                core.IMAGE_WIDTH,
            ):
                raise RuntimeError("formal 23-variant order mismatch")
            d0 = evaluator.infer_row(variants, row["train_label"])
            rows[index] = _process_valid_row(
                row, clip_records[index], payload, d0, cache
            )
        except RowDataFailure as failure:
            set_row_failure(cache, index, failure)
            rows[index] = _failed_row_record(
                row, failure.code, failure.message
            )
        cache["row_record_present"][index] = True
        progress["scientific_completed_rows"] = index + 1
        if (index + 1) % 8 == 0 or index + 1 == core.ORACLE_COUNT:
            print(
                json.dumps(
                    {
                        "stage": "d0_and_blind_phase",
                        "completed_rows": index + 1,
                        "total_rows": core.ORACLE_COUNT,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if any(record is None for record in rows):
        raise RuntimeError("D0/blind phase lost an oracle row")
    return rows


def _effect_payload(left, right, salt):
    bootstrap = core.paired_bootstrap_difference(
        left, right, repetitions=10000, salt=salt
    )
    return {
        "estimate": float(np.asarray(left).mean() - np.asarray(right).mean()),
        "bootstrap": bootstrap,
    }


def summarize_rows(rows_payload, cache):
    rows = rows_payload["rows"]
    if (
        rows_payload.get("schema") != ROWS_SCHEMA_NAME
        or len(rows) != core.ORACLE_COUNT
    ):
        raise RuntimeError("formal row JSON schema/count mismatch")
    expected_ids = cache["relative_paths"].tolist()
    if [str(row["row_id"]) for row in rows] != expected_ids:
        raise RuntimeError("row JSON/cache order mismatch")
    if [int(row["oracle_index"]) for row in rows] != list(
        range(core.ORACLE_COUNT)
    ):
        raise RuntimeError("row JSON oracle-index mismatch")

    arm_rows = {
        name: [row["arms"][name] for row in rows] for name in ARM_ORDER
    }
    factorial = core.finalize_factorial_rows(arm_rows)
    outcomes = factorial["outcomes"]
    quartet = factorial["quartet_matched"]
    raw_pair = core.finalize_paired_control_rows(
        [
            row["strong_controls"]["raw_color"]["reference"]
            for row in rows
        ],
        [
            row["strong_controls"]["raw_color"]["control"]
            for row in rows
        ],
    )
    hard_pair = core.finalize_paired_control_rows(
        [
            row["strong_controls"]["d0_hard"]["reference"]
            for row in rows
        ],
        [
            row["strong_controls"]["d0_hard"]["control"]
            for row in rows
        ],
    )

    effects = {
        "delta_c_given_p1": _effect_payload(
            outcomes["pc"], outcomes["pose_only"], "delta-c-given-p1"
        ),
        "delta_p_given_c1": _effect_payload(
            outcomes["pc"], outcomes["clip_only"], "delta-p-given-c1"
        ),
        "delta_c_given_p0": _effect_payload(
            outcomes["clip_only"],
            outcomes["neither"],
            "delta-c-given-p0",
        ),
        "delta_p_given_c0": _effect_payload(
            outcomes["pose_only"],
            outcomes["neither"],
            "delta-p-given-c0",
        ),
    }
    interaction = core.paired_bootstrap_interaction(
        outcomes["pc"],
        outcomes["pose_only"],
        outcomes["clip_only"],
        outcomes["neither"],
        repetitions=10000,
    )
    effects["interaction"] = interaction

    agreement_correct = (
        cache["agreement_correct_base"].astype(np.bool_) & quartet
    )
    agreement_shuffle = (
        cache["agreement_shuffle_base"].astype(np.bool_) & quartet
    )
    correct_rate = float(agreement_correct.mean())
    shuffle_rate = float(agreement_shuffle.mean())
    agreement_gap = correct_rate - shuffle_rate

    clean_top5 = cache["clean_top5"].astype(np.bool_)
    pc_top5 = cache["pc_top5"].astype(np.bool_)
    clean_top5_rate = float(clean_top5.mean())
    pc_top5_rate = float(pc_top5.mean())
    top5_intersection_rate = float((clean_top5 & pc_top5).mean())
    top5_drop = clean_top5_rate - pc_top5_rate

    counts = {
        name: int(outcomes[name].sum()) for name in ARM_ORDER
    }
    rates = {
        name: float(outcomes[name].mean()) for name in ARM_ORDER
    }
    slot_success = []
    active_anchor = cache["active_anchor"].astype(np.int64)
    for slot in range(core.ANCHOR_COUNT):
        mask = active_anchor == slot
        slot_success.append(int(outcomes["pc"][mask].sum()))

    pair_stats = {
        "raw_color": {
            "matched_count": int(raw_pair["pair_matched"].sum()),
            "reference_count": int(raw_pair["reference"].sum()),
            "control_count": int(raw_pair["control"].sum()),
            "reference_rate": float(raw_pair["reference"].mean()),
            "control_rate": float(raw_pair["control"].mean()),
        },
        "d0_hard": {
            "matched_count": int(hard_pair["pair_matched"].sum()),
            "reference_count": int(hard_pair["reference"].sum()),
            "control_count": int(hard_pair["control"].sum()),
            "reference_rate": float(hard_pair["reference"].mean()),
            "control_rate": float(hard_pair["control"].mean()),
        },
    }
    selection_hash_exact = all(
        str(row["selection_hash"]) == str(cache["selection_hash"][index])
        for index, row in enumerate(rows)
    )
    proposal_pools_exact = True
    active_proposals_exact = True
    roa_exact = True
    expected_complete = np.arange(
        core.PROPOSALS_PER_POOL, dtype=np.int16
    )
    for index in range(core.ORACLE_COUNT):
        for pool_index in range(2):
            proposal_pools_exact = bool(
                proposal_pools_exact
                and np.array_equal(
                    cache["proposal_candidate_index"][
                        index, pool_index
                    ],
                    expected_complete,
                )
                and np.all(
                    cache["proposal_mask_sha256"][
                        index, pool_index
                    ]
                    != ""
                )
            )
        expected_active = (
            int(cache["active_anchor"][index])
            * core.ACTIVE_PROPOSALS_PER_IMAGE
            + np.arange(
                core.ACTIVE_PROPOSALS_PER_IMAGE, dtype=np.int16
            )
        )
        active_proposals_exact = bool(
            active_proposals_exact
            and np.array_equal(
                cache["pose_candidate_index"][index],
                expected_active,
            )
            and np.array_equal(
                cache["fixed_candidate_index"][index],
                expected_active,
            )
        )
        roa_exact = bool(
            roa_exact
            and np.all(cache["roa_mask_sha256"][index] != "")
        )
    d0_external_pose_zero = all(
        row["d0"] is None
        or int(row["d0"]["external_pose_accesses"]) == 0
        for row in rows
    )

    gates = {
        "row_count_512": len(rows) == core.ORACLE_COUNT,
        "row_order_exact": [row["row_id"] for row in rows] == expected_ids,
        "row_records_present_512": int(
            cache["row_record_present"].sum()
        )
        == core.ORACLE_COUNT,
        "clip_valid_512": int(cache["clip_valid"].sum())
        == core.ORACLE_COUNT,
        "d0_valid_512": int(cache["d0_valid"].sum())
        == core.ORACLE_COUNT,
        "selection_hash_exact": selection_hash_exact,
        "recorded_complete_proposal_pools_exact": proposal_pools_exact,
        "recorded_active7_indices_exact": active_proposals_exact,
        "recorded_roa8_exact": roa_exact,
        "d0_external_pose_accesses_zero": d0_external_pose_zero,
        "quartet_matched_at_least_461": int(quartet.sum()) >= 461,
        "raw_pair_matched_at_least_461": pair_stats["raw_color"][
            "matched_count"
        ]
        >= 461,
        "d0_pair_matched_at_least_461": pair_stats["d0_hard"][
            "matched_count"
        ]
        >= 461,
        "pc_y_at_least_359": counts["pc"] >= 359,
        "five_slot_pc_each_at_least_64": all(
            value >= 64 for value in slot_success
        ),
        "delta_c_given_p1_at_least_008": effects[
            "delta_c_given_p1"
        ]["estimate"]
        >= 0.08,
        "delta_p_given_c1_at_least_008": effects[
            "delta_p_given_c1"
        ]["estimate"]
        >= 0.08,
        "delta_c_given_p0_at_least_004": effects[
            "delta_c_given_p0"
        ]["estimate"]
        >= 0.04,
        "delta_p_given_c0_at_least_004": effects[
            "delta_p_given_c0"
        ]["estimate"]
        >= 0.04,
        "interaction_at_least_004": interaction["estimate"] >= 0.04,
        "delta_c_given_p1_lower_gt_0": effects[
            "delta_c_given_p1"
        ]["bootstrap"]["one_sided_95_lower"]
        > 0.0,
        "delta_p_given_c1_lower_gt_0": effects[
            "delta_p_given_c1"
        ]["bootstrap"]["one_sided_95_lower"]
        > 0.0,
        "interaction_lower_gt_0": interaction["one_sided_95_lower"]
        > 0.0,
        "pc_strictly_above_three_factorial": all(
            counts["pc"] > counts[name]
            for name in ("pose_only", "clip_only", "neither")
        ),
        "pc_strictly_above_raw_pair_control": pair_stats["raw_color"][
            "reference_rate"
        ]
        > pair_stats["raw_color"]["control_rate"],
        "pc_strictly_above_d0_pair_control": pair_stats["d0_hard"][
            "reference_rate"
        ]
        > pair_stats["d0_hard"]["control_rate"],
        "selector_agreement_at_least_060": correct_rate >= 0.60,
        "agreement_gap_at_least_010": agreement_gap >= 0.10,
        "pc_top5_drop_at_most_005": top5_drop <= 0.05,
        "clean_pc_top5_intersection_at_least_090": (
            top5_intersection_rate >= 0.90
        ),
        "all_numeric_finite": bool(cache["numeric_finite"].all()),
        "optimizer_updates_zero": True,
        "checkpoint_writes_zero": True,
    }
    go = bool(all(gates.values()))
    failure_counts = {}
    for code in cache["failure_code"].tolist():
        if code:
            failure_counts[code] = failure_counts.get(code, 0) + 1
    return {
        "schema": SUMMARY_SCHEMA_NAME,
        "fixed_denominator": core.ORACLE_COUNT,
        "factorial": {
            "counts": counts,
            "rates": rates,
            "quartet_matched_count": int(quartet.sum()),
            "slot_pc_success_counts": slot_success,
            "effects": effects,
        },
        "strong_pairs": pair_stats,
        "agreement": {
            "correct_count": int(agreement_correct.sum()),
            "correct_rate": correct_rate,
            "text_shuffle_count": int(agreement_shuffle.sum()),
            "text_shuffle_rate": shuffle_rate,
            "correct_minus_shuffle": agreement_gap,
        },
        "top5": {
            "clean_count": int(clean_top5.sum()),
            "clean_rate": clean_top5_rate,
            "pc_edited_count": int(pc_top5.sum()),
            "pc_edited_rate": pc_top5_rate,
            "clean_pc_intersection_count": int(
                (clean_top5 & pc_top5).sum()
            ),
            "clean_pc_intersection_rate": top5_intersection_rate,
            "clean_minus_pc_edited": top5_drop,
        },
        "failure_counts": failure_counts,
        "diagnostics": {
            "clip_valid_count": int(cache["clip_valid"].sum()),
            "d0_valid_count": int(cache["d0_valid"].sum()),
            "raw_equivalent_to_pc_count": int(
                sum(
                    row["strong_controls"]["raw_color"][
                        "equivalent_to_pc"
                    ]
                    for row in rows
                )
            ),
            "d0_hard_equivalent_to_pc_count": int(
                sum(
                    row["strong_controls"]["d0_hard"][
                        "equivalent_to_pc"
                    ]
                    for row in rows
                )
            ),
        },
        "gates": gates,
        "go": go,
    }


def write_scientific_outputs(output_dir, cache, rows_payload, provenance):
    output_dir = Path(output_dir)
    cache_path = output_dir / "oracle_cache.npz"
    rows_path = output_dir / "rows.json"
    _atomic_npz(cache_path, cache)
    cache_readback = _readback_npz(cache_path, cache)
    _atomic_json(rows_path, rows_payload)
    rows_readback = _readback_json(rows_path, rows_payload)

    summary = summarize_rows(rows_readback, cache_readback)
    summary_path = output_dir / "summary.json"
    _atomic_json(summary_path, summary)
    _readback_json(summary_path, summary)
    verdict = (
        "JOINT ASSET ORACLE GO"
        if summary["go"]
        else "EXP415 PACIT ASSET NO-GO / FORMAL E120 NO-START"
    )
    result = {
        "schema": RESULT_SCHEMA_NAME,
        "status": "COMPLETE",
        "verdict": verdict,
        "go": bool(summary["go"]),
        "fixed_denominator": core.ORACLE_COUNT,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "resume_allowed": False,
        "external_gate_inputs_read": False,
        "provenance": dict(provenance),
        "summary_sha256": sha256_file(summary_path),
        "cache_sha256": sha256_file(cache_path),
        "rows_sha256": sha256_file(rows_path),
    }
    result_path = output_dir / "result.json"
    _atomic_json(result_path, result)
    _readback_json(result_path, result)
    manifest = {
        "schema": MANIFEST_SCHEMA_NAME,
        "files": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "bytes": int((output_dir / name).stat().st_size),
            }
            for name in (
                "started.json",
                "oracle_cache.npz",
                "rows.json",
                "summary.json",
                "result.json",
            )
        },
        "resume_allowed": False,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    _readback_json(manifest_path, manifest)
    for name, metadata in manifest["files"].items():
        if sha256_file(output_dir / name) != metadata["sha256"]:
            raise RuntimeError("manifest file SHA mismatch: " + name)
    return result, summary, manifest


def _validate_source_hashes(args):
    expected = {
        "runner": args.expected_runner_sha256,
        "core": args.expected_core_sha256,
        "selector": args.expected_selector_sha256,
        "prompt": args.expected_prompt_sha256,
    }
    missing = [name for name, value in expected.items() if not value]
    if missing:
        raise ValueError(
            "missing expected frozen source SHA: " + ", ".join(missing)
        )
    observed = {name: sha256_file(path) for name, path in SOURCE_FILES.items()}
    for name, expected_sha in expected.items():
        if observed[name] != str(expected_sha):
            raise RuntimeError("frozen source SHA mismatch: " + name)
    return observed


def validate_formal_args(args):
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError(
            "formal oracle requires PYTHONDONTWRITEBYTECODE=1"
        )
    if not sys.dont_write_bytecode:
        raise RuntimeError("formal oracle bytecode writes are not disabled")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("formal oracle requires PYTHONHASHSEED=0")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "formal oracle requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    observed_interpreter = Path(sys.executable)
    if (
        observed_interpreter != EXPECTED_INTERPRETER
        and observed_interpreter.resolve() != EXPECTED_INTERPRETER.resolve()
    ):
        raise RuntimeError("formal interpreter mismatch")
    if str(args.device) != "cuda:0":
        raise RuntimeError("formal oracle is frozen to logical cuda:0")
    if int(args.microbatch) != MICROBATCH:
        raise RuntimeError("formal microbatch mismatch")
    assert_no_cuda_compute_processes()
    fixed_paths = {
        "data_root": FIXED_DATA_ROOT,
        "pose_artifact": FIXED_POSE_ARTIFACT,
        "clip_checkpoint": FIXED_CLIP_CHECKPOINT,
        "d0_checkpoint": FIXED_D0_CHECKPOINT,
        "output_dir": FIXED_OUTPUT_DIR,
    }
    observed_paths = {
        name: Path(getattr(args, name)).expanduser().resolve()
        for name in fixed_paths
    }
    for name, expected in fixed_paths.items():
        if observed_paths[name] != expected:
            raise RuntimeError("formal frozen path mismatch: " + name)
    expected_config = (REPOSITORY_ROOT / FIXED_D0_CONFIG_RELATIVE).resolve()
    d0_config = Path(args.d0_config).expanduser().resolve()
    if d0_config != expected_config:
        raise RuntimeError("formal D0 config path mismatch")
    for name in ("data_root", "pose_artifact"):
        if not observed_paths[name].is_dir():
            raise NotADirectoryError(observed_paths[name])
    for name in ("clip_checkpoint", "d0_checkpoint"):
        if not observed_paths[name].is_file():
            raise FileNotFoundError(observed_paths[name])
    if not d0_config.is_file():
        raise FileNotFoundError(d0_config)
    if observed_paths["output_dir"].exists():
        raise FileExistsError("once-only oracle namespace already exists")
    if not observed_paths["output_dir"].parent.is_dir():
        raise NotADirectoryError(observed_paths["output_dir"].parent)
    if str(args.pose_manifest_sha256) != EXPECTED_POSE_MANIFEST_SHA256:
        raise RuntimeError("pose manifest argument mismatch")
    if sha256_file(observed_paths["clip_checkpoint"]) != EXPECTED_CLIP_SHA256:
        raise RuntimeError("CLIP checkpoint SHA mismatch")
    if sha256_file(observed_paths["d0_checkpoint"]) != EXPECTED_D0_SHA256:
        raise RuntimeError("D0 checkpoint SHA mismatch")
    if sha256_file(d0_config) != EXPECTED_D0_CONFIG_SHA256:
        raise RuntimeError("D0 config SHA mismatch")
    head = git_head(REPOSITORY_ROOT)
    if not args.expected_head or head != str(args.expected_head):
        raise RuntimeError("formal expected HEAD mismatch")
    if git_tracked_status(REPOSITORY_ROOT):
        raise RuntimeError("formal tracked source is dirty")
    for name in ("runner", "core", "selector", "prompt"):
        if not git_file_is_tracked(REPOSITORY_ROOT, SOURCE_FILES[name]):
            raise RuntimeError("formal source file is untracked: " + name)
    source_shas = _validate_source_hashes(args)
    return {
        "paths": observed_paths,
        "d0_config": d0_config,
        "head": head,
        "source_shas": source_shas,
        "interpreter_invoked": str(observed_interpreter),
        "interpreter_resolved": str(observed_interpreter.resolve()),
    }


def _started_payload(validated, selected):
    return {
        "schema": SCHEMA_NAME + "-started",
        "started_at": utc_now(),
        "source_head": validated["head"],
        "source_sha256": validated["source_shas"],
        "pose_manifest_sha256": EXPECTED_POSE_MANIFEST_SHA256,
        "clip_checkpoint_sha256": EXPECTED_CLIP_SHA256,
        "d0_checkpoint_sha256": EXPECTED_D0_SHA256,
        "d0_config_sha256": EXPECTED_D0_CONFIG_SHA256,
        "frozen_contract_sha256": frozen_contract_sha256(),
        "row_count": core.ORACLE_COUNT,
        "row_order_sha256": core.ordered_digest(
            [row["relative_path"] for row in selected]
        ),
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "resume_allowed": False,
        "external_gate_inputs_read": False,
    }


def run_formal(args):
    stage = "preseal_validation"
    progress = {
        "stage": stage,
        "oracle_index": None,
        "relative_path": None,
        "clip_completed_rows": 0,
        "scientific_completed_rows": 0,
    }
    output_dir = None
    namespace_owned = False
    validated = validate_formal_args(args)
    from datasets.occluded_duke import OccludedDuke
    from datasets.pose_targets import PoseTargetStore
    from clip_color_selector import FrozenWholeImageColorSelector

    dataset = OccludedDuke(
        root=str(validated["paths"]["data_root"]), verbose=False
    )
    records = list(dataset.train)
    labels = sorted({int(record[1]) for record in records})
    if labels != list(range(dataset.num_train_pids)):
        raise RuntimeError("official relabeled train PID contract failed")
    selected = select_rows(records, Path(dataset.dataset_dir))
    pose_store = PoseTargetStore(
        validated["paths"]["pose_artifact"],
        args.pose_manifest_sha256,
    )
    if len(pose_store) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError("pose artifact count mismatch")
    fixed_pool = core.generate_fixed_proposals()
    if len(fixed_pool) != core.PROPOSALS_PER_POOL:
        raise RuntimeError("fixed proposal pool mismatch")

    output_dir = validated["paths"]["output_dir"]
    try:
        stage = "once_only_started_seal"
        progress["stage"] = stage
        assert_no_cuda_compute_processes()
        output_dir.mkdir(mode=0o755, exist_ok=False)
        namespace_owned = True
        started = _started_payload(validated, selected)
        _atomic_json(output_dir / "started.json", started)
        _readback_json(output_dir / "started.json", started)
        stage = "cuda_initialization"
        progress["stage"] = stage
        if not torch.cuda.is_available():
            raise RuntimeError("formal oracle requires CUDA")
        torch.cuda.set_device(torch.device("cuda:0"))
        deterministic_state = configure_formal_determinism()
        stage = "clip_phase"
        progress["stage"] = stage
        device = torch.device("cuda:0")
        selector = FrozenWholeImageColorSelector(
            validated["paths"]["clip_checkpoint"],
            EXPECTED_CLIP_SHA256,
            device,
            microbatch=MICROBATCH,
        )
        cache = make_cache(selected)
        clip_records = run_clip_phase(
            selected,
            pose_store,
            fixed_pool,
            selector,
            device,
            cache,
            progress,
        )
        del selector
        gc.collect()
        torch.cuda.empty_cache()
        if (
            sha256_file(validated["paths"]["clip_checkpoint"])
            != EXPECTED_CLIP_SHA256
        ):
            raise RuntimeError("CLIP checkpoint changed during oracle")

        stage = "d0_and_blind_phase"
        progress["stage"] = stage
        evaluator = D0Evaluator(
            config_path=validated["d0_config"],
            checkpoint_path=validated["paths"]["d0_checkpoint"],
            dataset=dataset,
            device=device,
            microbatch=MICROBATCH,
        )
        rows = run_d0_and_blind_phase(
            selected,
            clip_records,
            pose_store,
            fixed_pool,
            evaluator,
            cache,
            progress,
        )
        evaluator.close()
        del evaluator

        stage = "postcompute_integrity"
        progress["stage"] = stage
        if git_head(REPOSITORY_ROOT) != validated["head"]:
            raise RuntimeError("formal HEAD changed during oracle")
        if git_tracked_status(REPOSITORY_ROOT):
            raise RuntimeError("formal tracked source changed during oracle")
        for name, expected_sha in validated["source_shas"].items():
            if sha256_file(SOURCE_FILES[name]) != expected_sha:
                raise RuntimeError("formal source changed during oracle: " + name)
        if (
            sha256_file(validated["paths"]["d0_checkpoint"])
            != EXPECTED_D0_SHA256
            or sha256_file(validated["paths"]["clip_checkpoint"])
            != EXPECTED_CLIP_SHA256
            or sha256_file(validated["d0_config"])
            != EXPECTED_D0_CONFIG_SHA256
        ):
            raise RuntimeError("frozen asset changed during oracle")
        rows_payload = {
            "schema": ROWS_SCHEMA_NAME,
            "row_count": core.ORACLE_COUNT,
            "row_order_sha256": core.ordered_digest(
                [row["row_id"] for row in rows]
            ),
            "rows": rows,
        }
        provenance = {
            "source_head": validated["head"],
            "source_sha256": validated["source_shas"],
            "pose_manifest_sha256": EXPECTED_POSE_MANIFEST_SHA256,
            "clip_checkpoint_sha256": EXPECTED_CLIP_SHA256,
            "d0_checkpoint_sha256": EXPECTED_D0_SHA256,
            "d0_config_sha256": EXPECTED_D0_CONFIG_SHA256,
            "row_order_sha256": rows_payload["row_order_sha256"],
            "variant_order": list(VARIANT_ORDER),
            "frozen_contract": frozen_contract_payload(),
            "frozen_contract_sha256": frozen_contract_sha256(),
            "interpreter": str(Path(sys.executable).resolve()),
            "interpreter_invoked": validated["interpreter_invoked"],
            "interpreter_resolved": validated["interpreter_resolved"],
            "device": "cuda:0",
            "microbatch": MICROBATCH,
            "formal_seed": FORMAL_SEED,
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
            "cublas_workspace_config": os.environ[
                "CUBLAS_WORKSPACE_CONFIG"
            ],
            "deterministic_algorithms": True,
            "deterministic_state_readback": deterministic_state,
            "external_gate_inputs_read": False,
        }
        stage = "atomic_outputs"
        progress["stage"] = stage
        result, summary, manifest = write_scientific_outputs(
            output_dir, cache, rows_payload, provenance
        )
        return {
            "status": "COMPLETE",
            "verdict": result["verdict"],
            "go": result["go"],
            "output_dir": str(output_dir),
            "result_sha256": sha256_file(output_dir / "result.json"),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "manifest_sha256": sha256_file(output_dir / "manifest.json"),
            "row_count": core.ORACLE_COUNT,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
        }
    except BaseException as error:
        if namespace_owned and output_dir is not None and output_dir.is_dir():
            failure_path = output_dir / "failure.json"
            failure_tmp = failure_path.with_name(failure_path.name + ".tmp")
            if not failure_path.exists() and not failure_tmp.exists():
                _atomic_json(
                    failure_path,
                    {
                        "schema": FAILURE_SCHEMA_NAME,
                        "status": "ORACLE INVALID",
                        "stage": progress["stage"],
                        "oracle_index": progress["oracle_index"],
                        "relative_path": progress["relative_path"],
                        "clip_completed_rows": int(
                            progress["clip_completed_rows"]
                        ),
                        "completed_rows": int(
                            progress["scientific_completed_rows"]
                        ),
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                        "source_head": validated["head"],
                        "source_sha256": validated["source_shas"],
                        "started_sha256": (
                            sha256_file(output_dir / "started.json")
                            if (output_dir / "started.json").is_file()
                            else None
                        ),
                        "optimizer_updates": 0,
                        "checkpoint_writes": 0,
                        "resume_allowed": False,
                        "external_gate_inputs_read": False,
                    },
                )
        raise


def _synthetic_rows_and_cache():
    selected = [
        {
            "oracle_index": index,
            "relative_path": "bounding_box_train/{:05d}.jpg".format(index),
            "train_label": index % 32,
            "raw_pid": index % 32,
            "camera": index % 8,
        }
        for index in range(core.ORACLE_COUNT)
    ]
    cache = make_cache(selected)
    cache["row_record_present"][:] = True
    cache["clip_valid"][:] = True
    cache["d0_valid"][:] = True
    cache["numeric_finite"][:] = True
    cache["clean_top5"][:] = True
    cache["pc_top5"][:500] = True
    cache["agreement_correct_base"][:400] = True
    cache["agreement_shuffle_base"][:150] = True
    complete_indices = np.arange(
        core.PROPOSALS_PER_POOL, dtype=np.int16
    )
    for index in range(core.ORACLE_COUNT):
        for pool_index in range(2):
            cache["proposal_candidate_index"][
                index, pool_index
            ] = complete_indices
            cache["proposal_mask_sha256"][
                index, pool_index
            ] = np.asarray(
                [
                    "{:064x}".format(
                        index * 1000
                        + pool_index * core.PROPOSALS_PER_POOL
                        + candidate
                        + 1
                    )
                    for candidate in range(core.PROPOSALS_PER_POOL)
                ]
            )
        active = (
            index % core.ANCHOR_COUNT
        ) * core.ACTIVE_PROPOSALS_PER_IMAGE + np.arange(
            core.ACTIVE_PROPOSALS_PER_IMAGE, dtype=np.int16
        )
        cache["pose_candidate_index"][index] = active
        cache["fixed_candidate_index"][index] = active
        cache["roa_mask_sha256"][index] = np.asarray(
            [
                "{:064x}".format(index * 100 + roa_index + 1)
                for roa_index in range(core.ROA_COUNT)
            ]
        )
    rows = []
    all_edges = {name: True for name in EDGE_ORDER}
    for index, selected_row in enumerate(selected):
        neither_y = int(index < 200)
        pose_y = int(index < 230)
        clip_y = int(index < 200 or 230 <= index < 260)
        pc_y = int(index < 370)
        values = {
            "pc": pc_y,
            "pose_only": pose_y,
            "clip_only": clip_y,
            "neither": neither_y,
        }
        arms = {}
        for arm_index, name in enumerate(ARM_ORDER):
            arms[name] = {
                "row_id": selected_row["relative_path"],
                "arm_name": name,
                "arm_complete": True,
                "selected_local_index": arm_index,
                "candidate_index": arm_index,
                "mask_sha256": "{:064x}".format(index * 8 + arm_index),
                "match_edges": dict(all_edges),
                "blind": {"Y": values[name], "blind_color_index": 0},
                "Y": values[name],
            }
        strong = {}
        for strong_index, name in enumerate(STRONG_ORDER):
            control_y = int(index < (240 - 10 * strong_index))
            strong[name] = {
                "reference": {
                    "row_id": selected_row["relative_path"],
                    "arm_complete": True,
                    "pair_match_ok": True,
                    "Y": pc_y,
                },
                "control": {
                    "row_id": selected_row["relative_path"],
                    "arm_complete": True,
                    "pair_match_ok": True,
                    "Y": control_y,
                },
                "selected_local_index": strong_index,
                "candidate_index": strong_index,
                "equivalent_to_pc": False,
            }
        rows.append(
            {
                "oracle_index": index,
                "row_id": selected_row["relative_path"],
                "selection_hash": cache["selection_hash"][index].item(),
                "train_label": selected_row["train_label"],
                "raw_pid": selected_row["raw_pid"],
                "camera": selected_row["camera"],
                "active_anchor": index % core.ANCHOR_COUNT,
                "record_complete": True,
                "data_failure": False,
                "failure_code": "",
                "failure_message": "",
                "arms": arms,
                "strong_controls": strong,
                "agreement_correct_base": bool(index < 400),
                "agreement_shuffle_base": bool(index < 150),
                "d0": {"external_pose_accesses": 0},
                "proposals": {},
            }
        )
    rows_payload = {
        "schema": ROWS_SCHEMA_NAME,
        "row_count": core.ORACLE_COUNT,
        "row_order_sha256": core.ordered_digest(
            [row["row_id"] for row in rows]
        ),
        "rows": rows,
    }
    return cache, rows_payload


def run_self_test():
    cache, rows_payload = _synthetic_rows_and_cache()
    summary = summarize_rows(rows_payload, cache)
    if not summary["go"]:
        failed = [name for name, value in summary["gates"].items() if not value]
        raise AssertionError("synthetic GO failed: " + ", ".join(failed))
    with tempfile.TemporaryDirectory(
        prefix=".exp415-oracle-selftest-", dir=str(SCRIPT_DIR)
    ) as directory:
        output_dir = Path(directory)
        started = {
            "schema": SCHEMA_NAME + "-started",
            "self_test": True,
            "resume_allowed": False,
        }
        _atomic_json(output_dir / "started.json", started)
        provenance = {
            "self_test": True,
            "external_gate_inputs_read": False,
        }
        result, observed_summary, manifest = write_scientific_outputs(
            output_dir, cache, rows_payload, provenance
        )
        if not result["go"] or observed_summary != summary:
            raise AssertionError("self-test output summary mismatch")
        if set(manifest["files"]) != {
            "started.json",
            "oracle_cache.npz",
            "rows.json",
            "summary.json",
            "result.json",
        }:
            raise AssertionError("self-test manifest mismatch")
    print("EXP415_ASSET_ORACLE_SELF_TEST=PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the once-only exp415 512-row PACIT asset oracle."
    )
    parser.add_argument("--data-root", default=str(FIXED_DATA_ROOT))
    parser.add_argument(
        "--pose-artifact", default=str(FIXED_POSE_ARTIFACT)
    )
    parser.add_argument(
        "--pose-manifest-sha256",
        default=EXPECTED_POSE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--clip-checkpoint", default=str(FIXED_CLIP_CHECKPOINT)
    )
    parser.add_argument(
        "--d0-checkpoint", default=str(FIXED_D0_CHECKPOINT)
    )
    parser.add_argument(
        "--d0-config",
        default=str((REPOSITORY_ROOT / FIXED_D0_CONFIG_RELATIVE).resolve()),
    )
    parser.add_argument("--output-dir", default=str(FIXED_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", type=int, default=MICROBATCH)
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--expected-core-sha256")
    parser.add_argument("--expected-selector-sha256")
    parser.add_argument("--expected-prompt-sha256")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    result = run_formal(args)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
