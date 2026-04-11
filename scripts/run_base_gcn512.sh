#!/bin/bash
# 4090: Base GCN512 + 2-stage PSG — 目标 75% mAP
# 这是当前最有可能突破 75% 的实验
set -e
PYTHON="python"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

CONFIG="configs/occluded_duke/pose_psg_lgpa_gcn_base.yml"

echo "=============================================="
echo "Base GCN512 + 2-stage PSG (target: 75% mAP)"
echo "Start: $(date)"
echo "=============================================="

OUT="./log/occluded_duke/base_gcn512_2stage_lr4"
$PYTHON train.py --config_file $CONFIG \
    SOLVER.BASE_LR 0.0004 \
    MODEL.POSE_GCN_HIDDEN 512 \
    MODEL.POSE_PSG_STAGES '[-2,-1]' \
    OUTPUT_DIR $OUT

echo "Training done. Running evals..."

# equal_concat eval (already done during training)
echo "=== MaxSim eval ==="
$PYTHON test.py --config_file $CONFIG \
    MODEL.POSE_GCN_HIDDEN 512 \
    MODEL.POSE_PSG_STAGES '[-2,-1]' \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT}/transformer_120.pth \
    OUTPUT_DIR ${OUT}_maxsim

echo "=== MaxSim on ep100 ==="
$PYTHON test.py --config_file $CONFIG \
    MODEL.POSE_GCN_HIDDEN 512 \
    MODEL.POSE_PSG_STAGES '[-2,-1]' \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT}/transformer_100.pth \
    OUTPUT_DIR ${OUT}_maxsim_ep100

echo "=============================================="
echo "DONE: $(date)"
echo "Results in: $OUT"
echo "=============================================="
