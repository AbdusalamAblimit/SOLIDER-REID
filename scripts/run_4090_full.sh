#!/bin/bash
# =================================================================
# 4090 全方法矩阵实验脚本
# 方法: PSG + GCN + PAA + ROA (当前最强配置)
#
# 实验矩阵:
#   Occluded-Duke: Swin-Small, Swin-Base  (Tiny 已有)
#   Market-1501:   Swin-Tiny, Swin-Small, Swin-Base
#
# 用法:
#   bash scripts/run_4090_full.sh
#
# 前置条件:
#   1. conda env: solider-reid
#   2. 数据: data/occluded_duke, data/market
#   3. VOC: data/VOCdevkit/VOC2012 (用 scripts/download_voc.sh 下载)
#   4. 预训练: pretrained/swin_tiny.pth, swin_small.pth, swin_base.pth
#   5. Pose data: data/occluded_duke/pose_data, data/market/pose_data
# =================================================================

set -e

# 检查 VOC
if [ ! -d "data/VOCdevkit/VOC2012/JPEGImages" ]; then
    echo "VOC2012 not found. Downloading..."
    bash scripts/download_voc.sh data
fi

# 检查 pose data
for ds in occluded_duke market; do
    if [ ! -d "data/$ds/pose_data" ]; then
        echo "WARNING: Pose data not found at data/$ds/pose_data"
        echo "Run scripts/extract_pose.py first for $ds dataset"
    fi
done

PYTHON=${PYTHON:-python}
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "4090 Full Experiment Matrix"
echo "Method: PSG + GCN + PAA + ROA"
echo "=========================================="

# ----- Helper function -----
run_exp() {
    local CONFIG=$1
    local OUTPUT=$2
    local EXTRA_ARGS="${@:3}"

    if [ -f "$OUTPUT/transformer_120.pth" ]; then
        echo "SKIP: $OUTPUT (checkpoint exists)"
        return
    fi

    echo ""
    echo ">>> Starting: $OUTPUT"
    echo "    Config: $CONFIG"
    echo "    Extra: $EXTRA_ARGS"

    $PYTHON train.py \
        --config_file "$CONFIG" \
        OUTPUT_DIR "$OUTPUT" \
        $EXTRA_ARGS \
        2>&1 | tee "$OUTPUT/train_log.txt"

    echo "<<< Finished: $OUTPUT"
}

# ===========================================================
# 1. Occluded-Duke: Swin-Small (lr=4e-4, best from 4090 log)
# ===========================================================
echo ""
echo "===== Occluded-Duke / Swin-Small ====="
run_exp configs/occluded_duke/pose_psg_gcn_paa_roa.yml \
    ./log/occluded_duke/4090_full_occ_small \
    MODEL.PRETRAIN_PATH pretrained/swin_small.pth \
    MODEL.TRANSFORMER_TYPE swin_small_patch4_window7_224 \
    SOLVER.BASE_LR 0.0004

# ===========================================================
# 2. Occluded-Duke: Swin-Base (lr=4e-4)
# ===========================================================
echo ""
echo "===== Occluded-Duke / Swin-Base ====="
run_exp configs/occluded_duke/pose_psg_gcn_paa_roa.yml \
    ./log/occluded_duke/4090_full_occ_base \
    MODEL.PRETRAIN_PATH pretrained/swin_base.pth \
    MODEL.TRANSFORMER_TYPE swin_base_patch4_window7_224 \
    SOLVER.BASE_LR 0.0004

# ===========================================================
# 3. Market-1501: Swin-Tiny (lr=8e-4, default)
# ===========================================================
echo ""
echo "===== Market-1501 / Swin-Tiny ====="
run_exp configs/occluded_duke/pose_psg_gcn_paa_roa.yml \
    ./log/market1501/4090_full_market_tiny \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR data \
    MODEL.POSE_DATA_DIR data/market \
    SOLVER.BASE_LR 0.0008

# ===========================================================
# 4. Market-1501: Swin-Small (lr=4e-4)
# ===========================================================
echo ""
echo "===== Market-1501 / Swin-Small ====="
run_exp configs/occluded_duke/pose_psg_gcn_paa_roa.yml \
    ./log/market1501/4090_full_market_small \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR data \
    MODEL.PRETRAIN_PATH pretrained/swin_small.pth \
    MODEL.TRANSFORMER_TYPE swin_small_patch4_window7_224 \
    MODEL.POSE_DATA_DIR data/market \
    SOLVER.BASE_LR 0.0004

# ===========================================================
# 5. Market-1501: Swin-Base (lr=4e-4)
# ===========================================================
echo ""
echo "===== Market-1501 / Swin-Base ====="
run_exp configs/occluded_duke/pose_psg_gcn_paa_roa.yml \
    ./log/market1501/4090_full_market_base \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR data \
    MODEL.PRETRAIN_PATH pretrained/swin_base.pth \
    MODEL.TRANSFORMER_TYPE swin_base_patch4_window7_224 \
    MODEL.POSE_DATA_DIR data/market \
    SOLVER.BASE_LR 0.0004

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results summary:"
for d in ./log/occluded_duke/4090_full_* ./log/market1501/4090_full_*; do
    if [ -f "$d/train_log.txt" ]; then
        echo "  $(basename $d):"
        grep "mAP:\|Rank-1" "$d/train_log.txt" | tail -2 | sed 's/^/    /'
    fi
done
