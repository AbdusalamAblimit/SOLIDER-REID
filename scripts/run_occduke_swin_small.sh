#!/bin/bash
# Run Swin-Small baseline on Occluded-Duke
# Prerequisites:
#   1. Occluded-Duke dataset in data/occluded_duke/
#   2. Pretrained weights: pretrained/swin_small.pth
#
# No pose data needed - this is a pure baseline run.

mkdir -p log/occluded_duke/swin_small_baseline

PYTHONUNBUFFERED=1 python train.py \
  --config_file configs/occluded_duke/swin_small.yml \
  2>&1 | tee log/occluded_duke/swin_small_baseline/train_log.txt
