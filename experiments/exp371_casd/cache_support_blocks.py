#!/usr/bin/env python3
"""Extract target-only LGPA support blocks and pose-response metadata.

This script only creates frozen caches for exp371 Gate C.  It does not build
same-ID support, evaluate an oracle, fit a projection, or train a student.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = "exp371_target_support_cache_v1"
POSE_SOURCE = "target_person_index_0; POSE_USE_TARGET_HEATMAP=True"
# Kept import-light for unit tests.  main() asserts this exact value against
# model.modules.clip_part_head.PART_KPS before real extraction.
PART_KPS = (
    (0, 1, 2, 3, 4),
    (5, 6, 11, 12),
    (5, 6, 7, 8, 9, 10),
    (11, 12, 13, 14),
    (15, 16),
)
FLIP_PAIRS = (
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
)
PART_COUNT = len(PART_KPS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val"), default=("train", "val")
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--expected-block-dim", type=int, default=768)
    parser.add_argument("--consistency-atol", type=float, default=2e-5)
    parser.add_argument("--flip-raw-atol", type=float, default=2e-6)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def pose_to_device(pose_dict: Mapping[str, object], device: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pose_dict.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, Mapping):
            result[key] = pose_to_device(value, device)
        else:
            result[key] = value
    return result


def flip_support_batch(
    imgs: torch.Tensor, pose_dict: Mapping[str, object]
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Mirror images and pose tensors using the repository COCO convention."""
    flipped: Dict[str, object] = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in pose_dict.items()
    }
    if "heatmaps" in flipped:
        heatmaps = flipped["heatmaps"].flip(-1)
        for left, right in FLIP_PAIRS:
            heatmaps[:, :, [left, right]] = heatmaps[:, :, [right, left]]
        flipped["heatmaps"] = heatmaps
    if "keypoints" in flipped:
        keypoints = flipped["keypoints"].clone()
        keypoints[..., 0] = imgs.shape[-1] - 1 - keypoints[..., 0]
        for left, right in FLIP_PAIRS:
            keypoints[:, :, [left, right]] = keypoints[:, :, [right, left]]
        flipped["keypoints"] = keypoints
    if "scores" in flipped:
        scores = flipped["scores"].clone()
        for left, right in FLIP_PAIRS:
            scores[:, :, [left, right]] = scores[:, :, [right, left]]
        flipped["scores"] = scores
    return imgs.flip(-1), flipped


def _target_heatmaps(pose_dict: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("heatmaps", "person_mask")
    missing = [key for key in required if key not in pose_dict]
    if missing:
        raise ValueError("pose_dict missing keys: %s" % missing)
    heatmaps = pose_dict["heatmaps"]
    person_mask = pose_dict["person_mask"]
    if heatmaps.ndim != 5 or heatmaps.shape[2] != 17:
        raise ValueError("heatmaps must have shape [B,P,17,H,W]")
    if person_mask.ndim != 2 or person_mask.shape[:2] != heatmaps.shape[:2]:
        raise ValueError("person_mask must have shape [B,P]")
    valid = person_mask[:, 0].bool()
    count = person_mask.bool().sum(dim=1).to(torch.int64)
    target = heatmaps[:, 0].float() * valid[:, None, None, None].float()
    return target, valid, count


def raw_part_response(
    pose_dict: Mapping[str, torch.Tensor], feature_hw: Tuple[int, int]
) -> Dict[str, torch.Tensor]:
    """Compute target-person raw and relative five-slot pose responses."""
    target, valid, count = _target_heatmaps(pose_dict)
    resized = F.interpolate(
        target, size=feature_hw, mode="bilinear", align_corners=False
    )
    raw = torch.stack(
        [
            resized[:, indices].max(dim=1).values.mean(dim=(1, 2))
            for indices in PART_KPS
        ],
        dim=1,
    )
    raw = raw * valid[:, None].float()
    relative = raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-6)
    if not torch.isfinite(raw).all() or not torch.isfinite(relative).all():
        raise ValueError("raw pose response contains NaN/Inf")
    if (raw[~valid] != 0).any():
        raise AssertionError("invalid target person must have zero raw response")
    return {
        "raw_pose_response": raw,
        "raw_response_relative_allocation": relative,
        "target_person_valid": valid,
        "person_count": count,
    }


def _feature_hw(extra: object) -> Tuple[int, int]:
    if not isinstance(extra, (list, tuple)) or not extra:
        raise TypeError("model second output must be a non-empty feature-map sequence")
    last = extra[-1]
    if not isinstance(last, torch.Tensor) or last.ndim != 4:
        raise TypeError("last feature map must have shape [B,C,H,W]")
    return int(last.shape[-2]), int(last.shape[-1])


def _forward(model, img, pose_dict, camids, target_view):
    return model(
        img,
        cam_label=camids,
        view_label=target_view,
        pose_dict=pose_dict,
    )


def _set_test_mode(model, mode: str) -> str:
    target = unwrap_model(model)
    previous = str(getattr(target, "pose_test_feat", "global"))
    target.pose_test_feat = mode
    return previous


def _restore_test_mode(model, previous: str) -> None:
    unwrap_model(model).pose_test_feat = previous


def _equal_concat_pair(
    feat: torch.Tensor, feat_flip: torch.Tensor, block_dim: int
) -> torch.Tensor:
    if not isinstance(feat, torch.Tensor) or not isinstance(feat_flip, torch.Tensor):
        raise TypeError("equal_concat forward must return tensors")
    expected = 7 * block_dim
    if feat.ndim != 2 or feat.shape != feat_flip.shape or feat.shape[1] != expected:
        raise ValueError("equal_concat must have shape [B,7x%d]" % block_dim)
    merged = (feat.float() + feat_flip.float()) / 2.0
    merged = merged.view(merged.shape[0], 7, block_dim)
    merged = F.normalize(merged, p=2, dim=2)
    return merged.contiguous()


def _maxsim_pair(
    feat: Mapping[str, torch.Tensor],
    feat_flip: Mapping[str, torch.Tensor],
    block_dim: int,
) -> Dict[str, torch.Tensor]:
    required = ("global_feat", "kp_feats", "kp_weights")
    if not isinstance(feat, Mapping) or not isinstance(feat_flip, Mapping):
        raise TypeError("maxsim_hybrid forward must return mappings")
    missing = [key for key in required if key not in feat or key not in feat_flip]
    if missing:
        raise ValueError("maxsim output missing keys: %s" % missing)
    # ``equal_concat`` normalizes every block inside each forward before the
    # original/flip arithmetic mean is formed.  Mirror that exact order here;
    # averaging raw MaxSim outputs first is mathematically close but not the
    # same when the two views have slightly different norms.
    global_feat = F.normalize(
        (
            F.normalize(feat["global_feat"].float(), p=2, dim=1)
            + F.normalize(feat_flip["global_feat"].float(), p=2, dim=1)
        ) / 2.0,
        p=2,
        dim=1,
    )
    kp_feats = F.normalize(
        (
            F.normalize(feat["kp_feats"].float(), p=2, dim=2)
            + F.normalize(feat_flip["kp_feats"].float(), p=2, dim=2)
        ) / 2.0,
        p=2,
        dim=2,
    )
    relative = (
        feat["kp_weights"].float() + feat_flip["kp_weights"].float()
    ) / 2.0
    if global_feat.ndim != 2 or global_feat.shape[1] != block_dim:
        raise ValueError("maxsim global feature has wrong shape")
    if kp_feats.ndim != 3 or kp_feats.shape[1:] != (PART_COUNT, block_dim):
        raise ValueError("maxsim kp_feats must have shape [B,5,%d]" % block_dim)
    if relative.shape != kp_feats.shape[:2]:
        raise ValueError("kp_weights must have shape [B,5]")
    if not all(torch.isfinite(value).all() for value in (global_feat, kp_feats, relative)):
        raise ValueError("maxsim metadata contains NaN/Inf")
    if (relative < 0).any():
        raise ValueError("relative allocation contains negative values")
    return {
        "global_feat": global_feat,
        "kp_feats": kp_feats,
        "relative_allocation": relative,
    }


@torch.no_grad()
def extract_support_batch(
    model,
    img: torch.Tensor,
    pose_dict: Mapping[str, torch.Tensor],
    camids: torch.Tensor,
    target_view: torch.Tensor,
    *,
    flip_test: bool,
    block_dim: int,
    consistency_atol: float = 2e-5,
    flip_raw_atol: float = 2e-6,
) -> Dict[str, torch.Tensor]:
    """Extract equal/maxsim views and target-only pose metadata for one batch."""
    model.eval()
    img_flip, pose_flip = flip_support_batch(img, pose_dict)

    previous = _set_test_mode(model, "equal_concat")
    try:
        equal, extra = _forward(model, img, pose_dict, camids, target_view)
        if flip_test:
            equal_flip, extra_flip = _forward(
                model, img_flip, pose_flip, camids, target_view
            )
        else:
            equal_flip, extra_flip = equal, extra
        equal_blocks = _equal_concat_pair(equal, equal_flip, block_dim)

        _set_test_mode(model, "maxsim_hybrid")
        maxsim, maxsim_extra = _forward(model, img, pose_dict, camids, target_view)
        if flip_test:
            maxsim_flip, maxsim_extra_flip = _forward(
                model, img_flip, pose_flip, camids, target_view
            )
        else:
            maxsim_flip, maxsim_extra_flip = maxsim, maxsim_extra
        maxsim_merged = _maxsim_pair(maxsim, maxsim_flip, block_dim)
    finally:
        _restore_test_mode(model, previous)

    feature_hw = _feature_hw(maxsim_extra)
    if _feature_hw(extra) != feature_hw or _feature_hw(maxsim_extra_flip) != feature_hw:
        raise AssertionError("equal/maxsim/flip feature-map resolutions differ")
    if flip_test and _feature_hw(extra_flip) != feature_hw:
        raise AssertionError("equal flip feature-map resolution differs")

    global_error = float(
        (equal_blocks[:, 0] - maxsim_merged["global_feat"]).abs().max().item()
    )
    part_error = float(
        (equal_blocks[:, 2:7] - maxsim_merged["kp_feats"]).abs().max().item()
    )
    if global_error > consistency_atol or part_error > consistency_atol:
        raise AssertionError(
            "equal/maxsim mismatch: global=%.9g parts=%.9g atol=%.9g"
            % (global_error, part_error, consistency_atol)
        )

    pose_orig = raw_part_response(pose_dict, feature_hw)
    pose_flipped = raw_part_response(pose_flip, feature_hw) if flip_test else pose_orig
    if not torch.equal(
        pose_orig["target_person_valid"], pose_flipped["target_person_valid"]
    ) or not torch.equal(pose_orig["person_count"], pose_flipped["person_count"]):
        raise AssertionError("flip changed target validity or person count")
    raw_flip_error = float(
        (
            pose_orig["raw_pose_response"]
            - pose_flipped["raw_pose_response"]
        ).abs().max().item()
    )
    if raw_flip_error > flip_raw_atol:
        raise AssertionError(
            "raw pose response changed under flip: %.9g > %.9g"
            % (raw_flip_error, flip_raw_atol)
        )
    raw = (
        pose_orig["raw_pose_response"]
        + pose_flipped["raw_pose_response"]
    ) / 2.0
    raw_relative = raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-6)
    allocation_error = float(
        (raw_relative - maxsim_merged["relative_allocation"]).abs().max().item()
    )
    if allocation_error > consistency_atol:
        raise AssertionError(
            "head/raw relative allocation mismatch: %.9g > %.9g"
            % (allocation_error, consistency_atol)
        )

    return {
        "features": equal_blocks.reshape(equal_blocks.shape[0], -1).contiguous(),
        "kp_feats": maxsim_merged["kp_feats"].contiguous(),
        "relative_allocation": maxsim_merged["relative_allocation"].contiguous(),
        "raw_pose_response": raw.contiguous(),
        "raw_response_relative_allocation": raw_relative.contiguous(),
        "target_person_valid": pose_orig["target_person_valid"].contiguous(),
        "person_count": pose_orig["person_count"].contiguous(),
        "global_consistency_max_abs": torch.tensor(global_error),
        "part_consistency_max_abs": torch.tensor(part_error),
        "allocation_consistency_max_abs": torch.tensor(allocation_error),
        "raw_flip_max_abs_diff": torch.tensor(raw_flip_error),
    }


def cache_payload(
    tensors: Mapping[str, torch.Tensor],
    *,
    pids: Sequence[int],
    camids: Sequence[int],
    paths: Sequence[str],
    split: str,
    num_query: int,
    block_dim: int,
    weight_sha256: str,
    script_sha256: str,
    flip_test: bool,
) -> Dict[str, object]:
    count = int(tensors["features"].shape[0])
    if not (len(pids) == len(camids) == len(paths) == count):
        raise ValueError("PID/CAM/path lengths must match tensor count")
    tensor_keys = (
        "features", "kp_feats", "relative_allocation", "raw_pose_response",
        "raw_response_relative_allocation", "target_person_valid", "person_count",
    )
    hashes = {key: tensor_sha256(tensors[key]) for key in tensor_keys}
    audit_keys = (
        "global_consistency_max_abs", "part_consistency_max_abs",
        "allocation_consistency_max_abs", "raw_flip_max_abs_diff",
    )
    audit = {
        key: float(tensors[key].item())
        for key in audit_keys
    }
    payload: Dict[str, object] = {
        **{key: tensors[key].cpu() for key in tensor_keys},
        "pids": [int(value) for value in pids],
        "camids": [int(value) for value in camids],
        "paths": [os.path.normpath(str(value)) for value in paths],
        "split": str(split),
        "mode": "target_only_correct",
        "num_query": int(num_query),
        "block_dim": int(block_dim),
        "schema_version": SCHEMA_VERSION,
        "pose_source": POSE_SOURCE,
        "relative_allocation_semantics": (
            "within-image five-slot relative pose-response allocation; "
            "not absolute visibility"
        ),
        "raw_response_definition": (
            "resize target-person heatmap to final feature map; per PART_KPS "
            "spatial mean of channelwise max; orig/flip arithmetic mean"
        ),
        "part_kps": [list(group) for group in PART_KPS],
        "part_kps_sha256": json_sha256([list(group) for group in PART_KPS]),
        "weight_sha256": str(weight_sha256),
        "script_sha256": str(script_sha256),
        "flip_test": bool(flip_test),
        "tensor_sha256": hashes,
        "audit": {
            **audit,
            "sample_count": count,
            "target_person_valid_count": int(
                tensors["target_person_valid"].sum().item()
            ),
            "multi_person_count": int((tensors["person_count"] > 1).sum().item()),
        },
    }
    payload["metadata_sha256"] = json_sha256({
        "pids": payload["pids"],
        "camids": payload["camids"],
        "paths": payload["paths"],
        "split": payload["split"],
        "num_query": payload["num_query"],
        "schema_version": payload["schema_version"],
        "pose_source": payload["pose_source"],
    })
    return payload


def assert_protocol(config, model, expected_block_dim: int) -> None:
    target = unwrap_model(model)
    checks = {
        "POSE_LGPA": bool(getattr(config.MODEL, "POSE_LGPA", False)),
        "POSE_LGPA_DETACH": bool(
            getattr(config.MODEL, "POSE_LGPA_DETACH", False)
        ),
        "POSE_USE_TARGET_HEATMAP": bool(
            getattr(config.MODEL, "POSE_USE_TARGET_HEATMAP", False)
        ),
        "POSE_BACKBONE_PSG": bool(
            getattr(config.MODEL, "POSE_BACKBONE_PSG", False)
        ),
        "POSE_PSG_STAGES_EMPTY": list(
            getattr(config.MODEL, "POSE_PSG_STAGES", [])
        ) == [],
        "POSE_SKELETON_GCN_OFF": not bool(
            getattr(config.MODEL, "POSE_SKELETON_GCN", False)
        ),
        "POSE_PPA_OFF": not bool(getattr(config.MODEL, "POSE_PPA", False)),
        "POSE_VCSR_OFF": not bool(getattr(config.MODEL, "POSE_VCSR", False)),
        "POSE_STRUCTURAL_ROUTING_OFF": not bool(
            getattr(config.MODEL, "POSE_STRUCTURAL_ROUTING", False)
        ),
        "POSE_OA_SD_OFF": not bool(getattr(config.MODEL, "POSE_OA_SD", False)),
        "POSE_PARALLEL_AUG_OFF": not bool(
            getattr(config.MODEL, "POSE_PARALLEL_AUG", False)
        ),
        "LGPA_NO_POSE_OFF": not bool(getattr(target, "_lgpa_no_pose", False)),
        "LGPA_FIXED_BANDS_OFF": not bool(
            getattr(target, "_lgpa_fixed_bands", False)
        ),
        "BLOCK_DIM": int(getattr(target, "in_planes", 0)) == expected_block_dim,
        "MODEL_TARGET_HEATMAP": bool(getattr(target, "use_target_heatmap", False)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("support-cache protocol failed: " + ", ".join(failed))


@torch.no_grad()
def extract_loader(
    model,
    loader,
    *,
    device: str,
    flip_test: bool,
    block_dim: int,
    consistency_atol: float,
    flip_raw_atol: float,
) -> Tuple[Dict[str, torch.Tensor], List[int], List[int], List[str]]:
    collected: Dict[str, List[torch.Tensor]] = {}
    pids: List[int] = []
    camids_all: List[int] = []
    paths: List[str] = []
    audit_max = {
        "global_consistency_max_abs": 0.0,
        "part_consistency_max_abs": 0.0,
        "allocation_consistency_max_abs": 0.0,
        "raw_flip_max_abs_diff": 0.0,
    }
    for batch in loader:
        img, pid, camid, camids, target_view, imgpath, pose_dict = batch
        pose_dict = pose_to_device(pose_dict, device)
        batch_out = extract_support_batch(
            model,
            img.to(device),
            pose_dict,
            camids.to(device),
            target_view.to(device),
            flip_test=flip_test,
            block_dim=block_dim,
            consistency_atol=consistency_atol,
            flip_raw_atol=flip_raw_atol,
        )
        for key in (
            "features", "kp_feats", "relative_allocation", "raw_pose_response",
            "raw_response_relative_allocation", "target_person_valid", "person_count",
        ):
            collected.setdefault(key, []).append(batch_out[key].cpu())
        for key in audit_max:
            audit_max[key] = max(audit_max[key], float(batch_out[key].item()))
        pids.extend(int(value) for value in pid)
        camids_all.extend(int(value) for value in camid)
        paths.extend(str(value) for value in imgpath)

    if not collected:
        raise ValueError("loader produced no batches")
    tensors = {key: torch.cat(values, dim=0) for key, values in collected.items()}
    tensors.update({key: torch.tensor(value) for key, value in audit_max.items()})
    return tensors, pids, camids_all, paths


def main() -> None:
    # Keep training-stack imports out of the pure-function/unit-test path.
    from config import cfg
    from datasets import make_dataloader
    from model import make_model
    from model.modules.clip_part_head import PART_KPS as MODEL_PART_KPS
    from datasets.pose_dataset import FLIP_PAIRS as DATASET_FLIP_PAIRS

    if tuple(tuple(group) for group in MODEL_PART_KPS) != PART_KPS:
        raise AssertionError("cache PART_KPS drifted from CLIPPartHead.PART_KPS")
    if tuple(tuple(pair) for pair in DATASET_FLIP_PAIRS) != FLIP_PAIRS:
        raise AssertionError("cache FLIP_PAIRS drifted from pose dataset")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weight = Path(args.weight).resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)

    cfg.merge_from_file(args.config_file)
    opts = list(args.opts)
    if opts and opts[0] == "--":
        opts = opts[1:]
    if opts:
        cfg.merge_from_list(opts)
    cfg.defrost()
    cfg.MODEL.POSE_USE_TARGET_HEATMAP = True
    cfg.MODEL.POSE_TEST_FEAT = "equal_concat"
    cfg.TEST.WEIGHT = str(weight)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    (
        _train_loader,
        train_loader_normal,
        val_loader,
        num_query,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    model.load_param(str(weight))
    model.to(args.device)
    assert_protocol(cfg, model, args.expected_block_dim)

    weight_sha = file_sha256(weight)
    script_sha = file_sha256(Path(__file__).resolve())
    flip_test = bool(getattr(cfg.TEST, "FLIP_TEST", True))
    split_specs = {
        "train": (train_loader_normal, 0, "train_normal"),
        "val": (val_loader, int(num_query), "val"),
    }
    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "config_file": str(Path(args.config_file).resolve()),
        "weight": str(weight),
        "weight_sha256": weight_sha,
        "script_sha256": script_sha,
        "pose_source": POSE_SOURCE,
        "splits": list(args.splits),
        "flip_test": flip_test,
        "outputs": {},
    }
    for requested in args.splits:
        loader, split_num_query, split_name = split_specs[requested]
        tensors, pids, camids, paths = extract_loader(
            model,
            loader,
            device=args.device,
            flip_test=flip_test,
            block_dim=args.expected_block_dim,
            consistency_atol=args.consistency_atol,
            flip_raw_atol=args.flip_raw_atol,
        )
        payload = cache_payload(
            tensors,
            pids=pids,
            camids=camids,
            paths=paths,
            split=split_name,
            num_query=split_num_query,
            block_dim=args.expected_block_dim,
            weight_sha256=weight_sha,
            script_sha256=script_sha,
            flip_test=flip_test,
        )
        output_path = output_dir / (requested + "_target_support.pt")
        atomic_torch_save(output_path, payload)
        manifest["outputs"][requested] = {
            "path": str(output_path),
            "file_sha256": file_sha256(output_path),
            "sample_count": payload["audit"]["sample_count"],
            "tensor_sha256": payload["tensor_sha256"],
            "metadata_sha256": payload["metadata_sha256"],
        }
        atomic_json(output_dir / "manifest.json", manifest)
        print(json.dumps({requested: manifest["outputs"][requested]}, indent=2))
    print("COMPLETE: %s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
