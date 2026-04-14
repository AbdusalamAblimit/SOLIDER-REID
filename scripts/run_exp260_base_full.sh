#!/bin/bash
# exp260: Base GCN512 + 2-stage PSG 全套
# Step 1: OccDuke 训练 + MaxSim eval
# Step 2: Market 训练
# Step 3: Market 权重 → Occluded-ReID 跨数据集测试
set -e
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
PY="/root/miniconda3/envs/solider-reid-pt2/bin/python"

CONFIG_OCC="configs/occluded_duke/pose_psg_lgpa_gcn_base.yml"
CONFIG_MKT="configs/market/pose_psg_lgpa_gcn_base.yml"

echo "=============================================="
echo "exp260: Base GCN512 + 2-stage PSG"
echo "=============================================="

# ====== Step 1: Occluded-Duke ======
echo "[1/5] OccDuke train — $(date)"
OUT_OCC="./log/occluded_duke/exp260_base_gcn512_2stage"
$PY train.py --config_file $CONFIG_OCC \
    OUTPUT_DIR $OUT_OCC

echo "[2/5] OccDuke MaxSim+FlipTest eval — $(date)"
$PY scripts/eval_fliptest_maxsim.py \
    --config_file $CONFIG_OCC \
    --weight ${OUT_OCC}/transformer_120.pth \
    DATALOADER.NUM_WORKERS 4 \
    2>&1 | tee ${OUT_OCC}/fliptest_maxsim.log

echo "OccDuke done: $(date)"
echo ""

# ====== Step 2: Market-1501 ======
echo "[3/5] Market train (no PLBOA) — $(date)"
OUT_MKT="./log/market1501/exp260_base_gcn512_2stage"
$PY train.py --config_file $CONFIG_MKT \
    OUTPUT_DIR $OUT_MKT

echo "[4/5] Market MaxSim+FlipTest eval — $(date)"
$PY scripts/eval_fliptest_maxsim.py \
    --config_file $CONFIG_MKT \
    --weight ${OUT_MKT}/transformer_120.pth \
    DATALOADER.NUM_WORKERS 4 \
    2>&1 | tee ${OUT_MKT}/fliptest_maxsim.log

echo "Market done: $(date)"
echo ""

# ====== Step 3: Cross-dataset (Market → Occluded-ReID) ======
echo "[5/5] Cross-dataset: Market weights → Occluded-ReID — $(date)"
$PY test.py --config_file $CONFIG_MKT \
    DATASETS.NAMES occluded_reid \
    DATASETS.ROOT_DIR data \
    MODEL.POSE_DATA_DIR data/occluded_reid \
    TEST.WEIGHT ${OUT_MKT}/transformer_120.pth \
    OUTPUT_DIR ${OUT_MKT}_occluded_reid \
    2>&1 | tee ${OUT_MKT}/cross_dataset_occ_reid.log

echo "=============================================="
echo "ALL DONE: $(date)"
echo "OccDuke: $OUT_OCC"
echo "Market:  $OUT_MKT"
echo "=============================================="
