#!/bin/bash
# PAMS experiments on Market-1501
# Usage: bash scripts/run_pams_market.sh [GPU_ID]

GPU=${1:-0}
DATA_DIR="/root/work/SOLIDER-REID/data"
PRETRAIN="pretrained/swin_tiny.pth"
POSE_CFG="pose/config_vispredict.py"
POSE_CKPT="pretrained/best_coco_AP_epoch_210.pth"

echo "========================================"
echo "PAMS Experiments on Market-1501"
echo "GPU: ${GPU}"
echo "========================================"

# --- PAMS_full on Market ---
echo ""
echo ">>> PAMS_full on Market-1501"
python train.py --config_file configs/market/pams_tiny.yml \
    MODEL.DEVICE_ID "'${GPU}'" \
    DATASETS.ROOT_DIR "'${DATA_DIR}'" \
    MODEL.PRETRAIN_PATH "'${PRETRAIN}'" \
    MODEL.POSE.CFG "'${POSE_CFG}'" \
    MODEL.POSE.CKPT "'${POSE_CKPT}'" \
    OUTPUT_DIR "'./log/market/pams_full'"

echo ""
echo "========================================"
echo "Market-1501 experiment done!"
echo "========================================"
