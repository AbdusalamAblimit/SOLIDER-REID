#!/bin/bash
# Market-1501 Base LGPA-D+GCN: ROA instead of PLBOA
# ROA = Random Object Augmentation (paste VOC objects, prob=0.7)
# 对比 PLBOA 版本 (93.8/96.8) 看 ROA 是否在无遮挡数据集上更优
#
# 用法: cd /path/to/SOLIDER-REID && bash scripts/run_base_market_roa.sh

set -e
PYTHON="python"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

CONFIG="configs/market/pose_psg_lgpa_gcn_base.yml"

echo "=============================================="
echo "Market Base LGPA-D+GCN with ROA (no PLBOA)"
echo "=============================================="

# ROA replaces PLBOA: disable PLBOA, enable ROA
echo "[1/2] Market Base ROA prob=0.7 LR=4e-4"
echo "Start: $(date)"
OUT="./log/market1501/base_lgpa_gcn_roa_lr4"
$PYTHON train.py --config_file $CONFIG \
    SOLVER.BASE_LR 0.0004 \
    MODEL.POSE_LOWER_BODY_OCC False \
    MODEL.POSE_ROA True \
    MODEL.POSE_ROA_PROB 0.7 \
    OUTPUT_DIR $OUT

echo "Training done. Running MaxSim test..."
$PYTHON test.py --config_file $CONFIG \
    MODEL.POSE_LOWER_BODY_OCC False \
    MODEL.POSE_ROA True \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT}/transformer_120.pth \
    OUTPUT_DIR ${OUT}_maxsim

echo "[1/2] Done: $(date)"

# Also run a no-augmentation version for clean comparison
echo ""
echo "[2/2] Market Base NO occlusion aug LR=4e-4"
echo "Start: $(date)"
OUT2="./log/market1501/base_lgpa_gcn_noaug_lr4"
$PYTHON train.py --config_file $CONFIG \
    SOLVER.BASE_LR 0.0004 \
    MODEL.POSE_LOWER_BODY_OCC False \
    MODEL.POSE_ROA False \
    OUTPUT_DIR $OUT2

echo "Training done. Running MaxSim test..."
$PYTHON test.py --config_file $CONFIG \
    MODEL.POSE_LOWER_BODY_OCC False \
    MODEL.POSE_ROA False \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT2}/transformer_120.pth \
    OUTPUT_DIR ${OUT2}_maxsim

echo "[2/2] Done: $(date)"

echo ""
echo "=============================================="
echo "DONE! Compare:"
echo "  PLBOA:  4090_log/market1501/base_lgpa_gcn_lr4  (93.8/96.8)"
echo "  ROA:    $OUT"
echo "  No aug: $OUT2"
echo "=============================================="
