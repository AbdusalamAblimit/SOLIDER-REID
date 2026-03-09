"""Test exp001 model with different feature modes: global, part, concat.

Usage:
    python scripts/test_feature_modes.py \
        --config_file configs/occluded_duke/pose_part_pooling.yml \
        TEST.WEIGHT log/occluded_duke/exp001_pose_part_pooling/transformer_120.pth
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.metrics import R1_mAP_eval
from processor.processor import _pose_to_device
from model.modules.pose_utils import heatmaps_to_parts, NUM_PARTS
from model.pose_model import PoseReIDModel
from utils.logger import setup_logger


def extract_features(model, img, camids, target_view, pose_dict, mode='concat'):
    """Extract features in different modes without modifying model code.

    mode='global': only global feature (after BN)
    mode='part':   only concatenated part features
    mode='concat': global + scaled part features (original behavior)
    """
    model.eval()
    with torch.no_grad():
        # Run backbone
        global_feat, featmaps = model.base(img)
        last_featmap = featmaps[-1]

        # Determine which feature map to use for part pooling
        pose_part_stage = getattr(model, 'pose_part_stage', -1)
        part_featmap = featmaps[pose_part_stage]

        # Apply PFM if model has it
        pfm_enabled = getattr(model, 'pfm_enabled', False)
        if pfm_enabled and pose_dict is not None:
            scene_heatmaps, scene_scores = PoseReIDModel._prepare_pose(pose_dict)
            last_featmap = model.pfm(last_featmap, scene_heatmaps)
            global_feat = last_featmap.mean(dim=(2, 3))  # Re-GAP from modulated
        else:
            scene_heatmaps, scene_scores = None, None

        if model.reduce_feat_dim:
            global_feat = model.fcneck(global_feat)
        feat = model.bottleneck(global_feat)

        if model.neck_feat == 'after':
            test_feat_global = feat
        else:
            test_feat_global = global_feat

        if mode == 'global':
            return test_feat_global

        # Part features — prepare pose if not already done
        if scene_heatmaps is None:
            scene_heatmaps, scene_scores = PoseReIDModel._prepare_pose(pose_dict)
        _, part_feats, part_valid = model.pose_part(
            part_featmap, scene_heatmaps, scene_scores)

        if mode == 'part':
            # Concatenate all part features (no global)
            scale = 1.0 / len(part_feats)
            return torch.cat([f * scale for f in part_feats], dim=1)

        if mode == 'part_noscale':
            # Part features without 1/N scaling
            return torch.cat(part_feats, dim=1)

        if mode == 'norm_concat':
            # L2-normalize each sub-feature before concat
            g_norm = F.normalize(test_feat_global, p=2, dim=1)
            p_norms = [F.normalize(f, p=2, dim=1) for f in part_feats]
            return torch.cat([g_norm] + p_norms, dim=1)

        if mode == 'norm_part_only':
            # L2-normalize each part feature before concat (no global)
            p_norms = [F.normalize(f, p=2, dim=1) for f in part_feats]
            return torch.cat(p_norms, dim=1)

        if mode == 'equal_concat':
            # Global + parts without scaling
            return torch.cat(
                [test_feat_global] + part_feats, dim=1)

        # concat mode (original, with 1/N scaling on parts)
        scale = 1.0 / len(part_feats)
        return torch.cat(
            [test_feat_global] + [f * scale for f in part_feats], dim=1)


def do_inference_mode(cfg, model, val_loader, num_query, mode='concat'):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info(f"Inferencing with mode: {mode}")

    evaluator = R1_mAP_eval(num_query, max_rank=50,
                            feat_norm=cfg.TEST.FEAT_NORM,
                            reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()
    model.to(device)
    model.eval()

    for n_iter, batch_data in enumerate(val_loader):
        with torch.no_grad():
            img, pid, camid, camids, target_view, imgpath, pose_dict = batch_data
            pose_dict = _pose_to_device(pose_dict, device)
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            feat = extract_features(model, img, camids, target_view,
                                    pose_dict, mode=mode)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info(f"[{mode}] mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"[{mode}] Rank-{r}: {cmc[r-1]:.1%}")
    return mAP, cmc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    logger = setup_logger("transreid", None, if_train=False)

    train_loader, train_loader_normal, val_loader, num_query, \
        num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num,
                       semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    if cfg.TEST.WEIGHT:
        model.load_param(cfg.TEST.WEIGHT)

    print("=" * 60)
    for mode in ['global', 'part', 'part_noscale', 'norm_part_only',
                 'norm_concat', 'equal_concat', 'concat']:
        mAP, cmc = do_inference_mode(cfg, model, val_loader, num_query,
                                     mode=mode)
        print(f"[{mode:>14s}]  mAP={mAP:.1%}  R1={cmc[0]:.1%}  "
              f"R5={cmc[4]:.1%}  R10={cmc[9]:.1%}")
    print("=" * 60)
