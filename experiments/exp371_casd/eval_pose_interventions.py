#!/usr/bin/env python3
"""Evaluate exp336 under controlled test-time pose interventions.

The script never changes model weights.  It reports correct/canonical/
wrong-PID-shuffled/uniform/no-pose descriptors from the same checkpoint and
optionally caches the correct-pose train/validation descriptors for Gate D.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg  # noqa: E402
from datasets import make_dataloader  # noqa: E402
from model import make_model  # noqa: E402
from processor.processor import _extract_feat_flip, _pose_to_device  # noqa: E402
from utils.metrics import euclidean_distance, eval_func  # noqa: E402

from experiments.exp371_casd.intervention_utils import (  # noqa: E402
    PoseDonorDataset,
    build_wrong_pid_donors,
    tensor_sha256,
    uniformize_pose_dict,
    validate_equal_concat,
)


VALID_MODES = ("correct", "canonical", "shuffled", "uniform", "no_pose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--modes", nargs="+", default=list(VALID_MODES), choices=VALID_MODES
    )
    parser.add_argument(
        "--cache-modes", nargs="*", default=["correct"], choices=VALID_MODES
    )
    parser.add_argument(
        "--cache-train-correct", action="store_true",
        help="also cache correct-pose descriptors from train_loader_normal",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def dataset_pids(dataset) -> List[int]:
    records = getattr(dataset, "dataset", None)
    if records is None:
        raise ValueError("expected PoseImageDataset.dataset records")
    return [int(record[1]) for record in records]


def shuffled_loader(base_loader: DataLoader, num_query: int) -> Tuple[DataLoader, Dict[str, Dict[str, object]]]:
    pids = dataset_pids(base_loader.dataset)
    query_donors, query_stats = build_wrong_pid_donors(pids[:num_query])
    gallery_local, gallery_stats = build_wrong_pid_donors(pids[num_query:])
    donors = query_donors + [num_query + index for index in gallery_local]
    wrapped = PoseDonorDataset(base_loader.dataset, donors)
    loader = DataLoader(
        wrapped,
        batch_size=base_loader.batch_size,
        shuffle=False,
        num_workers=base_loader.num_workers,
        collate_fn=base_loader.collate_fn,
        pin_memory=bool(getattr(base_loader, "pin_memory", False)),
        drop_last=False,
    )
    query_stats["donor_map_sha256"] = tensor_sha256(
        torch.tensor(query_donors, dtype=torch.int64))
    gallery_stats["donor_map_sha256"] = tensor_sha256(
        torch.tensor(gallery_local, dtype=torch.int64))
    return loader, {"query": query_stats, "gallery": gallery_stats}


def assert_exp336_protocol(cfg, model) -> None:
    target = model.module if hasattr(model, "module") else model
    failures = []
    checks = {
        "POSE_LGPA": bool(getattr(cfg.MODEL, "POSE_LGPA", False)),
        "POSE_LGPA_DETACH": bool(getattr(cfg.MODEL, "POSE_LGPA_DETACH", False)),
        "POSE_BACKBONE_PSG": bool(getattr(cfg.MODEL, "POSE_BACKBONE_PSG", False)),
        "POSE_PSG_STAGES_EMPTY": list(getattr(cfg.MODEL, "POSE_PSG_STAGES", [])) == [],
        "POSE_OA_SD_OFF": not bool(getattr(cfg.MODEL, "POSE_OA_SD", False)),
        "POSE_PARALLEL_AUG_OFF": not bool(getattr(cfg.MODEL, "POSE_PARALLEL_AUG", False)),
        "RE_RANKING_OFF": not bool(getattr(cfg.TEST, "RE_RANKING", False)),
        "NFC_OFF": not bool(getattr(cfg.TEST, "NFC", False)),
        "POWER_NORM_OFF": float(getattr(cfg.TEST, "POWER_NORM", 0.0)) == 0.0,
        "EMPTY_PSG_MODULES": len(getattr(target, "psg_modules_dict", {})) == 0,
        "BLOCK_DIM_768": int(getattr(target, "in_planes", 0)) == 768,
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    if failures:
        raise AssertionError(
            "Gate B is valid only for the isolated exp336 protocol; failed: "
            + ", ".join(failures)
        )


def set_model_intervention(model, mode: str) -> None:
    target = model.module if hasattr(model, "module") else model
    target._lgpa_no_pose = mode == "no_pose"
    target._lgpa_fixed_bands = mode == "canonical"
    target.pose_test_feat = "equal_concat"


def unpack_batch(batch_data, device: str, mode: str):
    img, pid, camid, camids, target_view, imgpath, pose_dict = batch_data
    pose_dict = _pose_to_device(pose_dict, device)
    if mode == "uniform":
        pose_dict = uniformize_pose_dict(pose_dict)
    return (
        img.to(device), pid, camid, camids.to(device), target_view.to(device),
        list(imgpath), pose_dict,
    )


def extract_features(
    model,
    loader: DataLoader,
    mode: str,
    device: str,
    flip_test: bool,
) -> Dict:
    set_model_intervention(model, mode)
    model.eval()
    features: List[torch.Tensor] = []
    pids: List[int] = []
    camids_all: List[int] = []
    paths: List[str] = []

    with torch.no_grad():
        for batch_data in loader:
            img, pid, camid, camids, target_view, imgpath, pose_dict = unpack_batch(
                batch_data, device, mode
            )
            feat = _extract_feat_flip(
                model, img, pose_dict, camids, target_view, True, flip_test
            )
            if not isinstance(feat, torch.Tensor):
                raise TypeError(f"expected tensor descriptor, got {type(feat)!r}")
            features.append(feat.detach().float().cpu())
            pids.extend(int(x) for x in pid)
            camids_all.extend(int(x) for x in camid)
            paths.extend(str(x) for x in imgpath)

    return {
        "features": torch.cat(features, dim=0),
        "pids": pids,
        "camids": camids_all,
        "paths": paths,
    }


def evaluate(features: torch.Tensor, pids: Sequence[int], camids: Sequence[int], num_query: int) -> Dict[str, float]:
    normed = F.normalize(features.float(), p=2, dim=1)
    qf, gf = normed[:num_query], normed[num_query:]
    pids_np = np.asarray(pids)
    camids_np = np.asarray(camids)
    distmat = euclidean_distance(qf, gf)
    cmc, mean_ap = eval_func(
        distmat,
        pids_np[:num_query], pids_np[num_query:],
        camids_np[:num_query], camids_np[num_query:],
    )
    return {
        "mAP": float(mean_ap),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
    }


def cache_payload(
    path: Path,
    extracted: Dict,
    split: str,
    mode: str,
    num_query: int,
    block_dim: int,
    weight_sha256: str,
) -> None:
    torch.save(
        {
            **extracted,
            "split": split,
            "mode": mode,
            "num_query": int(num_query),
            "block_dim": int(block_dim),
            "weight_sha256": weight_sha256,
        },
        path,
    )


def main() -> None:
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
    cfg.TEST.WEIGHT = str(weight)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.MODEL.POSE_TEST_FEAT = "equal_concat"
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exp371 feature extraction")

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
    model.to(device)
    assert_exp336_protocol(cfg, model)

    target = model.module if hasattr(model, "module") else model
    block_dim = int(getattr(target, "in_planes", 0) or getattr(target, "num_features", 0))
    if block_dim <= 0:
        raise ValueError("could not infer model block dimension")

    weight_sha = file_sha256(weight)
    script_sha = file_sha256(Path(__file__).resolve())
    manifest = {
        "config_file": str(Path(args.config_file).resolve()),
        "weight": str(weight),
        "weight_sha256": weight_sha,
        "script_sha256": script_sha,
        "modes": list(args.modes),
        "cache_modes": list(args.cache_modes),
        "cache_train_correct": bool(args.cache_train_correct),
        "num_query": int(num_query),
        "num_classes": int(num_classes),
        "block_dim": block_dim,
        "flip_test": bool(getattr(cfg.TEST, "FLIP_TEST", True)),
        "config": str(cfg),
    }
    atomic_json(output_dir / "manifest.json", manifest)

    flip_test = bool(getattr(cfg.TEST, "FLIP_TEST", True))
    results: Dict[str, Dict] = {}
    global_sha_reference: Optional[str] = None
    for mode in args.modes:
        loader = val_loader
        donor_stats = None
        if mode == "shuffled":
            loader, donor_stats = shuffled_loader(val_loader, num_query)
        extracted = extract_features(model, loader, mode, device, flip_test)
        features = extracted["features"]
        layout = validate_equal_concat(features, block_dim)
        global_features = features[:, :block_dim]
        global_sha = tensor_sha256(global_features)
        if global_sha_reference is None:
            global_sha_reference = global_sha
        if global_sha != global_sha_reference:
            raise AssertionError(
                f"global descriptor changed under {mode}: {global_sha} != {global_sha_reference}"
            )

        result = {
            "full": evaluate(features, extracted["pids"], extracted["camids"], num_query),
            "global": evaluate(global_features, extracted["pids"], extracted["camids"], num_query),
            "layout": layout,
            "global_sha256": global_sha,
            "donor_stats": donor_stats,
        }
        results[mode] = result
        atomic_json(output_dir / "results.json", results)
        print(json.dumps({mode: result}, ensure_ascii=False, indent=2), flush=True)

        if mode in args.cache_modes:
            cache_payload(
                output_dir / f"val_{mode}.pt",
                extracted, "val", mode, num_query, block_dim, weight_sha,
            )

    if args.cache_train_correct:
        extracted = extract_features(
            model, train_loader_normal, "correct", device, flip_test
        )
        validate_equal_concat(extracted["features"], block_dim)
        cache_payload(
            output_dir / "train_correct.pt",
            extracted, "train_normal", "correct", 0, block_dim, weight_sha,
        )
        manifest["train_cache_num_samples"] = len(extracted["pids"])
        manifest["train_cache_sha256"] = tensor_sha256(extracted["features"])
        atomic_json(output_dir / "manifest.json", manifest)

    print(f"COMPLETE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
