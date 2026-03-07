#!/bin/bash
# Market-1501 训练脚本: Baseline vs PCFC+GiLt
# 用法: bash scripts/run_market.sh [baseline|gilt|both]

set -e

MODE=${1:-both}
CONDA_ENV="solider-reid"

echo "============================================"
echo "Market-1501 Training Script"
echo "Mode: $MODE"
echo "============================================"

# 先检查 pose 数据是否存在
if [ ! -f "data/market1501/pose_train.npz" ]; then
    echo "ERROR: pose data not found at data/market1501/pose_train.npz"
    echo "Please run pose extraction first (see data/extract_pose.py)"
    exit 1
fi

if [ "$MODE" = "baseline" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo ">>> [1/2] Training Baseline (Swin-Tiny, SW=0.2)"
    echo "    Config: configs/market/baseline_tiny.yml"
    echo "    Output: ./log/market1501/exp001_baseline"
    conda run -n $CONDA_ENV python train.py \
        --config_file configs/market/baseline_tiny.yml
    echo ">>> Baseline training complete."
fi

if [ "$MODE" = "gilt" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo ">>> [2/2] Training PCFC+GiLt (Swin-Tiny, SW=0.2)"
    echo "    Config: configs/market/gilt_pcfc_tiny.yml"
    echo "    Output: ./log/market1501/gilt_pcfc"
    conda run -n $CONDA_ENV python train.py \
        --config_file configs/market/gilt_pcfc_tiny.yml
    echo ">>> PCFC+GiLt training complete."
fi

echo ""
echo "============================================"
echo "All training complete!"
echo "Check results in ./log/market1501/"
echo "============================================"
