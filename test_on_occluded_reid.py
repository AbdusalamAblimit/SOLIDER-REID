#!/usr/bin/env python
"""Evaluate trained models on Occluded-ReID.

Supports all model families used so far in this repo:
  - baseline / PSG-only global models
  - pose part-pooling models
  - PSG + part-pooling models
  - PDS / PDS+StopGrad / PDS+GCN models
  - PSG + GCN models

For pose-enabled models, pose data must exist under:
  data/occluded_reid/pose_data/query/
  data/occluded_reid/pose_data/gallery/
"""

import argparse
import glob
import logging
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn
from datasets.occluded_reid import OccludedREID
from datasets.pose_dataset import PoseImageDataset, pose_val_collate_fn
from model import make_model
from processor.processor import _pose_to_device
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on Occluded-ReID")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str)
    parser.add_argument(
        "--dataset-root", default="data/occluded_reid",
        help="Occluded-ReID root directory")
    parser.add_argument(
        "--modes", default="auto",
        help="Comma-separated eval modes. Default: auto")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only validate config, dataset, checkpoint, and modes")
    parser.add_argument(
        "opts", help="Modify config options using the command-line",
        default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def _has_override(opts, key):
    return any(opt == key for opt in opts)


def _resolve_checkpoint(output_dir):
    ckpts = glob.glob(os.path.join(output_dir, "transformer_*.pth"))
    if not ckpts:
        return ""
    return max(ckpts, key=os.path.getmtime)


def _resolve_output_dir(train_output_dir):
    base = train_output_dir or "./log"
    return os.path.join(base, "occluded_reid_eval")


def _build_val_loader(cfg, dataset_root):
    dataset = OccludedREID(dataset_dir=dataset_root)
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    if cfg.MODEL.POSE_ENABLED:
        pose_root = os.path.join(dataset_root, "pose_data")
        query_pose_dir = os.path.join(pose_root, "query")
        gallery_pose_dir = os.path.join(pose_root, "gallery")
        required = [
            os.path.join(query_pose_dir, "index.json"),
            os.path.join(gallery_pose_dir, "index.json"),
        ]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            missing_str = "\n".join(missing)
            raise RuntimeError(
                "Pose-enabled model requires Occluded-ReID pose data.\n"
                f"Missing:\n{missing_str}\n"
                "Generate it with:\n"
                "python scripts/extract_pose.py "
                "--data-root data/occluded_reid "
                "--output-dir data/occluded_reid/pose_data "
                "--splits query gallery")

        hm_size = None
        if hasattr(cfg.MODEL, 'POSE_HEATMAP_SIZE'):
            hm_size = tuple(cfg.MODEL.POSE_HEATMAP_SIZE)

        common_kwargs = dict(
            img_size=tuple(cfg.INPUT.SIZE_TEST),
            is_train=False,
            pixel_mean=cfg.INPUT.PIXEL_MEAN,
            pixel_std=cfg.INPUT.PIXEL_STD,
            heatmap_size=hm_size,
        )
        query_set = PoseImageDataset(
            dataset.query, pose_dir=query_pose_dir, **common_kwargs)
        gallery_set = PoseImageDataset(
            dataset.gallery, pose_dir=gallery_pose_dir, **common_kwargs)
        val_set = ConcatDataset([query_set, gallery_set])
        collate_fn = pose_val_collate_fn
    else:
        query_set = ImageDataset(dataset.query, val_transforms)
        gallery_set = ImageDataset(dataset.gallery, val_transforms)
        val_set = ConcatDataset([query_set, gallery_set])
        collate_fn = val_collate_fn

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_fn)
    return dataset, val_loader


def _plain_pose_part_model(cfg):
    return (cfg.MODEL.POSE_ENABLED and
            not getattr(cfg.MODEL, 'POSE_DUAL_STREAM', False) and
            not getattr(cfg.MODEL, 'POSE_BACKBONE_PSG', False) and
            not getattr(cfg.MODEL, 'POSE_PSG_PART', False))


def _single_branch_global_model(cfg):
    if not cfg.MODEL.POSE_ENABLED:
        return True
    if getattr(cfg.MODEL, 'POSE_DUAL_STREAM', False):
        return False
    if getattr(cfg.MODEL, 'POSE_PSG_PART', False):
        return False
    if _plain_pose_part_model(cfg):
        return False
    if getattr(cfg.MODEL, 'POSE_SKELETON_GCN', False):
        return False
    return True


def _supported_modes(cfg):
    if _single_branch_global_model(cfg):
        return ['global']
    return ['global', 'part_only', 'equal_concat', 'concat_scaled']


def _normalize_modes(mode_arg, cfg):
    aliases = {
        'default': 'global',
        'part': 'part_only',
        'gcn_only': 'part_only',
        'concat': 'concat_scaled',
    }
    supported = _supported_modes(cfg)
    if mode_arg == 'auto':
        return supported

    modes = []
    for raw in mode_arg.split(','):
        mode = aliases.get(raw.strip(), raw.strip())
        if mode not in supported:
            raise ValueError(
                f"Unsupported mode '{raw}'. Supported: {supported}")
        if mode not in modes:
            modes.append(mode)
    return modes


def _set_model_mode(model, cfg, mode):
    if not hasattr(model, 'pose_test_feat'):
        return

    if getattr(cfg.MODEL, 'POSE_BACKBONE_PSG', False) and \
            getattr(cfg.MODEL, 'POSE_SKELETON_GCN', False) and \
            not getattr(cfg.MODEL, 'POSE_DUAL_STREAM', False):
        model.pose_test_feat = 'gcn_only' if mode == 'part_only' else mode
        return

    model.pose_test_feat = mode


def _extract_plain_pose_part_features(model, img, pose_dict, mode):
    global_feat, featmaps = model.base(img)
    part_featmap = featmaps[model.pose_part_stage]

    scene_heatmaps, scene_scores = model._prepare_pose(pose_dict)
    if getattr(model, 'pfm_enabled', False):
        modulated = model.pfm(featmaps[-1], scene_heatmaps)
        global_feat = modulated.mean(dim=(2, 3))

    if model.reduce_feat_dim:
        global_feat = model.fcneck(global_feat)
    feat = model.bottleneck(global_feat)
    test_feat_global = feat if model.neck_feat == 'after' else global_feat

    _, part_feats, _ = model.pose_part(
        part_featmap, scene_heatmaps, scene_scores)

    if mode == 'global':
        return test_feat_global
    if mode == 'part_only':
        return torch.cat(part_feats, dim=1)
    if mode == 'equal_concat':
        return torch.cat([test_feat_global] + part_feats, dim=1)

    scale = 1.0 / len(part_feats)
    return torch.cat([test_feat_global] + [f * scale for f in part_feats], dim=1)


def _extract_features(model, cfg, img, camids, target_view, pose_dict, mode):
    if not cfg.MODEL.POSE_ENABLED:
        feat, _ = model(img, cam_label=camids, view_label=target_view)
        return feat

    if _plain_pose_part_model(cfg):
        return _extract_plain_pose_part_features(model, img, pose_dict, mode)

    _set_model_mode(model, cfg, mode)
    feat, _ = model(img, cam_label=camids, view_label=target_view,
                    pose_dict=pose_dict)
    return feat


def _evaluate_mode(cfg, model, val_loader, num_query, mode, device):
    logger = logging.getLogger("transreid.test")
    evaluator = R1_mAP_eval(
        num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    model.eval()
    for batch_data in val_loader:
        with torch.no_grad():
            if cfg.MODEL.POSE_ENABLED:
                img, pid, camid, camids, target_view, _, pose_dict = batch_data
                pose_dict = _pose_to_device(pose_dict, device)
            else:
                img, pid, camid, camids, target_view, _ = batch_data
                pose_dict = None

            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            feat = _extract_features(
                model, cfg, img, camids, target_view, pose_dict, mode)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("[%s] Validation Results", mode)
    logger.info("[%s] mAP: %.1f%%", mode, mAP * 100.0)
    for r in [1, 5, 10]:
        logger.info("[%s] CMC curve, Rank-%-3d:%.1f%%", mode, r, cmc[r - 1] * 100.0)
    return {
        'mode': mode,
        'mAP': mAP,
        'R1': cmc[0],
        'R5': cmc[4],
        'R10': cmc[9],
    }


def main():
    args = parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    train_output_dir = cfg.OUTPUT_DIR
    if not cfg.TEST.WEIGHT:
        ckpt = _resolve_checkpoint(train_output_dir)
        if not ckpt:
            raise RuntimeError(
                "TEST.WEIGHT is empty and no checkpoint was found under "
                f"'{train_output_dir}'.")
        cfg.TEST.WEIGHT = ckpt

    if not _has_override(args.opts, 'OUTPUT_DIR'):
        cfg.OUTPUT_DIR = _resolve_output_dir(train_output_dir)

    modes = _normalize_modes(args.modes, cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)
    if args.config_file:
        logger.info("Loaded configuration file %s", args.config_file)
    logger.info("Using checkpoint: %s", cfg.TEST.WEIGHT)
    logger.info("Dataset root: %s", args.dataset_root)
    logger.info("Eval modes: %s", modes)

    dataset, val_loader = _build_val_loader(cfg, args.dataset_root)
    num_eval_pids = len({pid for _, pid, _, _ in dataset.query + dataset.gallery})
    logger.info("Occluded-ReID stats: query=%d gallery=%d ids=%d",
                len(dataset.query), len(dataset.gallery), num_eval_pids)

    if args.dry_run:
        logger.info("Dry run complete.")
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = make_model(
        cfg,
        num_class=max(num_eval_pids, 1),
        camera_num=2,
        view_num=1,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device)

    results = []
    for mode in modes:
        results.append(_evaluate_mode(
            cfg, model, val_loader, len(dataset.query), mode, device))

    logger.info("=== Summary ===")
    for result in results:
        logger.info(
            "[%s] mAP=%.1f%% R1=%.1f%% R5=%.1f%% R10=%.1f%%",
            result['mode'],
            result['mAP'] * 100.0,
            result['R1'] * 100.0,
            result['R5'] * 100.0,
            result['R10'] * 100.0,
        )


if __name__ == "__main__":
    main()
