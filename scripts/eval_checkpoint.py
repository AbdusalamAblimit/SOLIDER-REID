#!/usr/bin/env python
"""Evaluate a saved checkpoint with a specific config.

Usage:
    python scripts/eval_checkpoint.py \
        --config_file configs/occluded_duke/pose_psg_gilt.yml \
        --weight log/occluded_duke/exp008_psg_part/transformer_120.pth
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor.processor import do_inference
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--weight', required=True)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=False)
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Weight: {args.weight}")

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)

    state_dict = torch.load(args.weight, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded checkpoint from {args.weight}")

    model.cuda()
    model.eval()

    do_inference(cfg, model, val_loader, num_query)


if __name__ == '__main__':
    main()
