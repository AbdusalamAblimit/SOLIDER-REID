#!/bin/bash
# Swin-Base LGPA-D+GCN+OA-SD 全套实验 (4090)
# 顺序跑: OccDuke lr4 → OccDuke lr2 → Market lr4 → Market lr2
# 每个实验结束后自动跑 MaxSim test
#
# 使用方法:
#   cd /root/work/SOLIDER-REID  (或你的项目目录)
#   bash scripts/run_base_4090.sh
#
# 预计时间: 每个 ~12-15h (Base+WITH_CP 较慢), 总计 ~50-60h
# 如果 4090 比 3090 快, 可能 ~8-10h/experiment

set -e

PYTHON="python"  # 4090 上的 python 路径，按需修改
# 如果用 conda env:
# PYTHON="/path/to/conda/envs/solider-reid-pt2/bin/python"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

CONFIG_OCC="configs/occluded_duke/pose_psg_lgpa_gcn_base.yml"
CONFIG_MKT="configs/market/pose_psg_lgpa_gcn_base.yml"

echo "=============================================="
echo "Swin-Base LGPA-D+GCN+OA-SD 4-experiment suite"
echo "=============================================="

# ============================================
# 1. Occluded-Duke LR=4e-4
# ============================================
echo ""
echo "[1/4] Occluded-Duke Base LR=4e-4"
echo "Start: $(date)"
OUT_OCC_LR4="./log/occluded_duke/base_lgpa_gcn_lr4"
$PYTHON train.py --config_file $CONFIG_OCC \
    SOLVER.BASE_LR 0.0004 \
    OUTPUT_DIR $OUT_OCC_LR4

echo "[1/4] Training done. Running MaxSim test..."
$PYTHON test.py --config_file $CONFIG_OCC \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT_OCC_LR4}/transformer_120.pth \
    OUTPUT_DIR ${OUT_OCC_LR4}_maxsim
echo "[1/4] Done: $(date)"

# ============================================
# 2. Occluded-Duke LR=2e-4
# ============================================
echo ""
echo "[2/4] Occluded-Duke Base LR=2e-4"
echo "Start: $(date)"
OUT_OCC_LR2="./log/occluded_duke/base_lgpa_gcn_lr2"
$PYTHON train.py --config_file $CONFIG_OCC \
    SOLVER.BASE_LR 0.0002 \
    OUTPUT_DIR $OUT_OCC_LR2

echo "[2/4] Training done. Running MaxSim test..."
$PYTHON test.py --config_file $CONFIG_OCC \
    SOLVER.BASE_LR 0.0002 \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT_OCC_LR2}/transformer_120.pth \
    OUTPUT_DIR ${OUT_OCC_LR2}_maxsim
echo "[2/4] Done: $(date)"

# ============================================
# 3. Market-1501 LR=4e-4
# ============================================
echo ""
echo "[3/4] Market-1501 Base LR=4e-4"
echo "Start: $(date)"
OUT_MKT_LR4="./log/market1501/base_lgpa_gcn_lr4"
$PYTHON train.py --config_file $CONFIG_MKT \
    SOLVER.BASE_LR 0.0004 \
    OUTPUT_DIR $OUT_MKT_LR4

echo "[3/4] Training done. Running MaxSim test..."
$PYTHON test.py --config_file $CONFIG_MKT \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT_MKT_LR4}/transformer_120.pth \
    OUTPUT_DIR ${OUT_MKT_LR4}_maxsim
echo "[3/4] Done: $(date)"

# ============================================
# 4. Market-1501 LR=2e-4
# ============================================
echo ""
echo "[4/4] Market-1501 Base LR=2e-4"
echo "Start: $(date)"
OUT_MKT_LR2="./log/market1501/base_lgpa_gcn_lr2"
$PYTHON train.py --config_file $CONFIG_MKT \
    SOLVER.BASE_LR 0.0002 \
    OUTPUT_DIR $OUT_MKT_LR2

echo "[4/4] Training done. Running MaxSim test..."
$PYTHON test.py --config_file $CONFIG_MKT \
    SOLVER.BASE_LR 0.0002 \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT_MKT_LR2}/transformer_120.pth \
    OUTPUT_DIR ${OUT_MKT_LR2}_maxsim
echo "[4/4] Done: $(date)"

# ============================================
# 5. Occluded-Duke Multi-Stage PSG + PAA, LR=4e-4
# ============================================
echo ""
echo "[5/6] Occluded-Duke Base Multi-Stage PSG + PAA LR=4e-4"
echo "Start: $(date)"
OUT_OCC_MSPSG="./log/occluded_duke/base_lgpa_gcn_mspsg_paa_lr4"
$PYTHON train.py --config_file $CONFIG_OCC \
    SOLVER.BASE_LR 0.0004 \
    MODEL.POSE_PSG_STAGES '[-2,-1]' \
    MODEL.POSE_ADDITIVE_ADAPTER True \
    MODEL.POSE_PAA_BOTTLENECK 32 \
    OUTPUT_DIR $OUT_OCC_MSPSG

echo "[5/6] Training done. Running MaxSim test..."
$PYTHON test.py --config_file $CONFIG_OCC \
    MODEL.POSE_PSG_STAGES '[-2,-1]' \
    MODEL.POSE_ADDITIVE_ADAPTER True \
    MODEL.POSE_PAA_BOTTLENECK 32 \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT_OCC_MSPSG}/transformer_120.pth \
    OUTPUT_DIR ${OUT_OCC_MSPSG}_maxsim
echo "[5/6] Done: $(date)"

# ============================================
# 6. Market Multi-Stage PSG + PAA, LR=4e-4
# ============================================
echo ""
echo "[6/6] Market-1501 Base Multi-Stage PSG + PAA LR=4e-4"
echo "Start: $(date)"
OUT_MKT_MSPSG="./log/market1501/base_lgpa_gcn_mspsg_paa_lr4"
$PYTHON train.py --config_file $CONFIG_MKT \
    SOLVER.BASE_LR 0.0004 \
    MODEL.POSE_PSG_STAGES '[-2,-1]' \
    MODEL.POSE_ADDITIVE_ADAPTER True \
    MODEL.POSE_PAA_BOTTLENECK 32 \
    OUTPUT_DIR $OUT_MKT_MSPSG

echo "[6/6] Training done. Running MaxSim test..."
$PYTHON test.py --config_file $CONFIG_MKT \
    MODEL.POSE_PSG_STAGES '[-2,-1]' \
    MODEL.POSE_ADDITIVE_ADAPTER True \
    MODEL.POSE_PAA_BOTTLENECK 32 \
    MODEL.POSE_TEST_FEAT maxsim_hybrid \
    TEST.WEIGHT ${OUT_MKT_MSPSG}/transformer_120.pth \
    OUTPUT_DIR ${OUT_MKT_MSPSG}_maxsim
echo "[6/6] Done: $(date)"

echo ""
echo "=============================================="
echo "ALL 6 EXPERIMENTS COMPLETE!"
echo "=============================================="
echo ""
echo "Results summary — check train_log.txt in each directory:"
echo "  $OUT_OCC_LR4/train_log.txt"
echo "  $OUT_OCC_LR2/train_log.txt"
echo "  $OUT_MKT_LR4/train_log.txt"
echo "  $OUT_MKT_LR2/train_log.txt"
echo "  $OUT_OCC_MSPSG/train_log.txt"
echo "  $OUT_MKT_MSPSG/train_log.txt"
echo ""
echo "MaxSim results in *_maxsim/ directories"
