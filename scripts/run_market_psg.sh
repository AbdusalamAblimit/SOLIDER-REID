#!/bin/bash
# Run exp007 PSG on Market-1501
# Prerequisites:
#   1. Market-1501 dataset in data/market1501/
#   2. Pretrained weights: pretrained/swin_tiny.pth
#   3. Pretrained detection model: pretrained/rtmdet_s_8xb32-300e_coco*.pth + .py
#   4. Pretrained pose model: pretrained/td-hm_ViTPose-huge*.pth + .py
#
# Step 1: Extract pose data for Market-1501 (MUST do this first!)
#   python scripts/extract_pose.py \
#     --data-root data/market1501 \
#     --output-dir data/market1501/pose_data \
#     --det-config pretrained/rtmdet_s_8xb32-300e_coco.py \
#     --det-checkpoint pretrained/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth \
#     --pose-config pretrained/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py \
#     --pose-checkpoint pretrained/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth
#
# Step 2: Train
mkdir -p log/market1501/exp007_backbone_psg

PYTHONUNBUFFERED=1 python train.py \
  --config_file configs/market/pose_backbone_psg.yml \
  2>&1 | tee log/market1501/exp007_backbone_psg/train_log.txt
