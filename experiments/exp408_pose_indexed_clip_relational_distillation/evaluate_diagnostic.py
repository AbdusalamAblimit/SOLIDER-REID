#!/usr/bin/env python3
"""Evaluate the frozen exp408 64-image PICRD relation diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from datasets.bases import read_image
from datasets.occluded_duke import OccludedDuke
from datasets.paired_pose_transform import PairedPoseTransform
from datasets.pose_targets import PoseTargetStore
from model import make_model
from model.pose_clip_relation import PoseClipRelationCache, sha256_file


SOURCE_HEAD = "86496f0062d7553062567e7d2bbcb371a24ef500"
CONFIG_PATH = REPO_ROOT / "configs/occluded_duke/swin_tiny_tapf_picrd_exp408.yml"
CONFIG_SHA256 = "798b806f04f63627ee2e5db4f67ec6c56a91d8447ee8c41f521307c634846433"
CHECKPOINT_PATH = Path(
    "/home/afr/reid-clean/logs/exp408-picrd-s1234-v1/transformer_120.pth"
)
CHECKPOINT_SHA256 = "6e6f9f4cdc64b54d9cbf8c2d6013f8303ae6b84c9b4a0d79ab4d1106d8f6d321"
RUNNER_LOG_PATH = Path(
    "/home/afr/reid-clean/train-logs/exp408-picrd-s1234-v1.runner.log"
)
CACHE_PATH = Path(
    "/home/afr/reid-clean/assets/exp408-picrd-cache-v2/picrd_cache.npz"
)
CACHE_SHA256 = "80db6448a38745a7846bbb1ffb63d868b4efcda8851bc069cd8166dc311cebee"
DIAGNOSTIC_PATH = Path(
    "/home/afr/reid-clean/assets/exp408-picrd-cache-v2/diagnostic_manifest.json"
)
DIAGNOSTIC_SHA256 = "8ef842f98a1172d7c8c197828cb3d4fda2006ced52062c9608569da5be62cff8"
SOURCE_SHA256 = {
    "model/pose_clip_relation.py": "fbd3e137a729f44d3179864f9978bd8846b22e8627a3c311747b0a2541092864",
    "model/tapf.py": "79b12d764c2c72be76ae3a2a3f19b2168f07ad15ac470b517415bba7dd0dea37",
    "model/make_model.py": "b68d5a0e1d85edf4411cb69c86646d40d11223b506eaccd8a42ad57335950487",
    "model/backbones/swin_transformer.py": "45e020d20e42db3695a27b123ec9ad76c7c6d4498255c340537a75d6c3665036",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_frozen_source():
    actual_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_head != SOURCE_HEAD:
        raise RuntimeError("diagnostic repository HEAD differs from training source")
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise RuntimeError("frozen training config SHA256 mismatch")
    for relative, expected in SOURCE_SHA256.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError("frozen source SHA256 mismatch: " + relative)
    if CHECKPOINT_PATH.resolve() != CHECKPOINT_PATH or not CHECKPOINT_PATH.is_file():
        raise RuntimeError("frozen e120 checkpoint path is missing or non-canonical")
    if sha256_file(CHECKPOINT_PATH) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen e120 checkpoint SHA256 mismatch")
    if RUNNER_LOG_PATH.resolve() != RUNNER_LOG_PATH or not RUNNER_LOG_PATH.is_file():
        raise RuntimeError("frozen runner log is missing or non-canonical")
    runner_text = RUNNER_LOG_PATH.read_text(encoding="utf-8")
    required_receipts = (
        "Epoch 120 done.",
        "Validation Results - Epoch: 120",
        "mAP: 57.1%",
        "CMC curve, Rank-1  :67.7%",
    )
    if any(receipt not in runner_text for receipt in required_receipts):
        raise RuntimeError("frozen runner log lacks the e120 terminal receipt")
    return actual_head, sha256_file(RUNNER_LOG_PATH)


def load_manifest():
    path = DIAGNOSTIC_PATH.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path != DIAGNOSTIC_PATH or sha256_file(path) != DIAGNOSTIC_SHA256:
        raise RuntimeError("diagnostic manifest SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "exp408-picrd-diagnostic-v1":
        raise RuntimeError("unexpected diagnostic schema")
    if payload.get("preprocessing") != "raw-rgb-pose-resize-384x128-no-augmentation":
        raise RuntimeError("unexpected diagnostic preprocessing")
    if payload.get("cache_sha256") != CACHE_SHA256:
        raise RuntimeError("diagnostic manifest is not bound to the frozen cache")
    if payload.get("wrong_rgb_cyclic_offset") != 4:
        raise RuntimeError("unexpected diagnostic wrong-RGB offset")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 64:
        raise RuntimeError("diagnostic manifest must contain 64 rows")
    paths = [row.get("relative_path") for row in rows]
    identities = [row.get("pid") for row in rows]
    if any(not isinstance(value, str) for value in paths):
        raise RuntimeError("diagnostic relative paths are invalid")
    if len(set(paths)) != len(paths):
        raise RuntimeError("diagnostic relative paths are not unique")
    if any(not isinstance(value, int) for value in identities):
        raise RuntimeError("diagnostic identities are invalid")
    if any(len(set(identities[start : start + 4])) != 1 for start in range(0, 64, 4)):
        raise RuntimeError("diagnostic identities are not grouped four-at-a-time")
    if len(set(identities)) != 16:
        raise RuntimeError("diagnostic manifest must contain 16 identities")
    shifted = identities[4:] + identities[:4]
    if any(left == right for left, right in zip(identities, shifted)):
        raise RuntimeError("diagnostic offset four is not different-PID")
    return path, rows, paths, identities


def deterministic_batch(rows, dataset, pose_store, transform):
    official = {}
    for image_path, pid, _camid, _trackid in dataset.train:
        relative = (
            Path(image_path)
            .resolve()
            .relative_to(pose_store.dataset_root)
            .as_posix()
        )
        official[relative] = (Path(image_path).resolve(), int(pid))
    images = []
    keypoints = []
    scores = []
    valid = []
    for row in rows:
        relative = row["relative_path"]
        if relative not in official:
            raise RuntimeError("diagnostic path is absent from official train")
        image_path, official_pid = official[relative]
        if official_pid != row["pid"]:
            raise RuntimeError("diagnostic PID disagrees with official train")
        image = read_image(str(image_path))
        pose = pose_store.get(image_path, verify_image_sha=False)
        image, augmented = transform(image, pose)
        if augmented.relative_path != relative:
            raise RuntimeError("diagnostic pose path changed")
        if augmented.flipped or augmented.crop_offset != (0, 0):
            raise RuntimeError("diagnostic transform is not deterministic resize")
        images.append(image)
        keypoints.append(augmented.keypoints)
        scores.append(augmented.scores)
        valid.append(augmented.valid)
    return (
        torch.stack(images),
        torch.stack(keypoints),
        torch.stack(scores),
        torch.stack(valid),
    )


def diagnostic_once(model, images, pose_batch):
    model.eval()
    if model.base.training:
        raise RuntimeError("model base must begin in eval mode")
    if any(module.training for module in model.base.children()):
        raise RuntimeError("all base children must remain in eval mode")
    model.base.training = True
    try:
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            _descriptor, _maps, state = model.base(
                images, pose_batch=pose_batch, tapf_epoch=120
            )
    finally:
        model.base.training = False
    keys = (
        "picrd_loss",
        "picrd_correct",
        "picrd_wrong_rgb",
        "picrd_generic",
        "picrd_zero",
        "picrd_ranking",
        "picrd_common_valid_fraction",
    )
    values = {key: state[key].detach().float().cpu() for key in keys}
    values["picrd_wrong_shift"] = int(state["picrd_wrong_shift"])
    stacked = torch.stack([values[key] for key in keys])
    if not bool(torch.isfinite(stacked).all()):
        raise RuntimeError("diagnostic values are non-finite")
    return values


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.parent.exists():
        raise FileExistsError("diagnostic output must use a fresh directory")
    actual_head, runner_log_sha256 = validate_frozen_source()
    manifest_path, rows, paths, identities = load_manifest()

    cfg.merge_from_file(str(CONFIG_PATH))
    cfg.freeze()
    if not (cfg.MODEL.TAPF.ENABLED and cfg.MODEL.TAPF.PICRD_ENABLED):
        raise RuntimeError("PICRD config is not enabled")
    if cfg.SOLVER.IMS_PER_BATCH != 64 or cfg.SOLVER.SEED != 1234:
        raise RuntimeError("diagnostic requires frozen batch64/seed1234 config")
    if Path(cfg.MODEL.TAPF.PICRD_CACHE).resolve() != CACHE_PATH:
        raise RuntimeError("config does not select the frozen PICRD cache")
    if cfg.MODEL.TAPF.PICRD_CACHE_SHA256 != CACHE_SHA256:
        raise RuntimeError("config PICRD cache SHA256 changed")

    dataset = OccludedDuke(root=cfg.DATASETS.ROOT_DIR, verbose=False)
    pose_store = PoseTargetStore(
        cfg.MODEL.TAPF.ARTIFACT_DIR,
        cfg.MODEL.TAPF.MANIFEST_SHA256,
    )
    transform = PairedPoseTransform(
        size_train=cfg.INPUT.SIZE_TRAIN,
        flip_probability=0.0,
        padding=0,
        pixel_mean=cfg.INPUT.PIXEL_MEAN,
        pixel_std=cfg.INPUT.PIXEL_STD,
        erasing_probability=0.0,
    )
    images, keypoints, scores, geometry_valid = deterministic_batch(
        rows, dataset, pose_store, transform
    )
    cache = PoseClipRelationCache(
        cfg.MODEL.TAPF.PICRD_CACHE,
        cfg.MODEL.TAPF.PICRD_CACHE_SHA256,
    )
    clip_features, clip_valid = cache.lookup(paths)

    torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    model = make_model(
        cfg,
        num_class=dataset.num_train_pids,
        camera_num=dataset.num_train_cams,
        view_num=dataset.num_train_vids,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    device = torch.device("cuda", 0)
    model.to(device)
    pose_batch = {
        "keypoints": keypoints.to(device),
        "scores": scores.to(device),
        "valid": geometry_valid.to(device),
        "clip_slot_features": clip_features.to(device),
        "clip_slot_valid": clip_valid.to(device),
        "identity": torch.as_tensor(identities, device=device),
    }
    images = images.to(device)
    first = diagnostic_once(model, images, pose_batch)
    second = diagnostic_once(model, images, pose_batch)
    scalar_keys = tuple(key for key in first if key != "picrd_wrong_shift")
    if first["picrd_wrong_shift"] != 4 or second["picrd_wrong_shift"] != 4:
        raise RuntimeError("diagnostic wrong-RGB shift changed")
    if any(not torch.equal(first[key], second[key]) for key in scalar_keys):
        raise RuntimeError("diagnostic is not repeat-exact")

    correct = float(first["picrd_correct"])
    controls = {
        "wrong_rgb": float(first["picrd_wrong_rgb"]),
        "generic": float(first["picrd_generic"]),
        "zero": float(first["picrd_zero"]),
    }
    order_pass = all(correct < value for value in controls.values())
    result = {
        "schema": "exp408-picrd-diagnostic-result-v1",
        "experiment": "exp408_pose_indexed_clip_relational_distillation",
        "source_head": actual_head,
        "source_sha256": SOURCE_SHA256,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "config": str(CONFIG_PATH),
        "config_sha256": CONFIG_SHA256,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runner_log": str(RUNNER_LOG_PATH),
        "runner_log_sha256": runner_log_sha256,
        "cache_sha256": cache.sha256,
        "diagnostic_manifest": str(manifest_path),
        "diagnostic_manifest_sha256": DIAGNOSTIC_SHA256,
        "samples": 64,
        "identities": 16,
        "wrong_rgb_shift": 4,
        "repeat_exact": True,
        "common_valid_fraction": float(first["picrd_common_valid_fraction"]),
        "loss": float(first["picrd_loss"]),
        "ranking": float(first["picrd_ranking"]),
        "correct": correct,
        "controls": controls,
        "correct_is_strict_minimum": order_pass,
        "mechanism_order_pass": order_pass,
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
