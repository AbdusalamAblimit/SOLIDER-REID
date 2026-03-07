#!/bin/bash
# Occluded-Duke Swin-Small 训练脚本: Baseline vs PCFC+GiLt
# 用法: bash scripts/run_occluded_duke_small.sh [baseline|gilt|both]

set -e

MODE=${1:-both}
CONDA_ENV="solider-reid"

echo "============================================"
echo "Occluded-Duke Swin-Small Training Script"
echo "Mode: $MODE"
echo "============================================"

# 检查 pose 数据
if [ ! -f "data/occluded_duke/pose_train.npz" ]; then
    echo "ERROR: pose data not found at data/occluded_duke/pose_train.npz"
    echo "Please run pose extraction first (see scripts/extract_pose_generic.py)"
    exit 1
fi

if [ "$MODE" = "baseline" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo ">>> [1/2] Training Baseline (Swin-Small, SW=0.2)"
    echo "    Config: configs/occluded_duke/baseline_small.yml"
    echo "    Output: ./log/occluded_duke/baseline_small"
    conda run -n $CONDA_ENV python train.py \
        --config_file configs/occluded_duke/baseline_small.yml
    echo ">>> Baseline training complete."
fi

if [ "$MODE" = "gilt" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo ">>> [2/2] Training PCFC+GiLt (Swin-Small, SW=0.2)"
    echo "    Config: configs/occluded_duke/gilt_pcfc_small.yml"
    echo "    Output: ./log/occluded_duke/gilt_pcfc_small"
    conda run -n $CONDA_ENV python train.py \
        --config_file configs/occluded_duke/gilt_pcfc_small.yml
    echo ">>> PCFC+GiLt training complete."
fi

echo ""
echo "============================================"
echo "All training complete!"
echo "Check results in ./log/occluded_duke/"
echo "============================================"
