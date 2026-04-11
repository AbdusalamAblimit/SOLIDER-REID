#!/bin/bash
# 4090: Base GCN512 + 2-stage PSG 全套 (OccDuke + Market)
# LR=4e-4, 120ep, 每个自动跑 MaxSim
set -e
PYTHON="python"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

CONFIG_OCC="configs/occluded_duke/pose_psg_lgpa_gcn_base.yml"
CONFIG_MKT="configs/market/pose_psg_lgpa_gcn_base.yml"

# 公共参数
GCN_ARGS="MODEL.POSE_GCN_HIDDEN 512 MODEL.POSE_PSG_STAGES [-2,-1]"

echo "=============================================="
echo "Base GCN512 + 2-stage PSG (target: 75% mAP)"
echo "=============================================="

# 1. Occluded-Duke
echo "[1/4] OccDuke train"
echo "Start: $(date)"
OUT1="./log/occluded_duke/base_gcn512_2stage_lr4"
$PYTHON train.py --config_file $CONFIG_OCC \
    SOLVER.BASE_LR 0.0004 $GCN_ARGS \
    OUTPUT_DIR $OUT1

echo "[2/4] OccDuke MaxSim"
$PYTHON test.py --config_file $CONFIG_OCC \
    $GCN_ARGS MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT1}/transformer_120.pth \
    OUTPUT_DIR ${OUT1}_maxsim
echo "OccDuke done: $(date)"

# 2. Market (无 PLBOA, 用 ROA)
echo "[3/4] Market train (ROA, no PLBOA)"
echo "Start: $(date)"
OUT2="./log/market1501/base_gcn512_2stage_roa_lr4"
$PYTHON train.py --config_file $CONFIG_MKT \
    SOLVER.BASE_LR 0.0004 $GCN_ARGS \
    MODEL.POSE_LOWER_BODY_OCC False \
    MODEL.POSE_ROA True MODEL.POSE_ROA_PROB 0.7 \
    OUTPUT_DIR $OUT2

echo "[4/4] Market MaxSim"
$PYTHON test.py --config_file $CONFIG_MKT \
    $GCN_ARGS \
    MODEL.POSE_LOWER_BODY_OCC False \
    MODEL.POSE_ROA True \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT2}/transformer_120.pth \
    OUTPUT_DIR ${OUT2}_maxsim
echo "Market done: $(date)"

echo "=============================================="
echo "ALL DONE: $(date)"
echo "OccDuke: $OUT1"
echo "Market:  $OUT2"
echo "=============================================="
