#!/bin/bash
# PAMS experiments on Occluded-Duke
# Usage: bash scripts/run_pams_occ_duke.sh [GPU_ID]

GPU=${1:-0}
DATA_DIR="/root/work/SOLIDER-REID/data"
PRETRAIN="pretrained/swin_tiny.pth"
POSE_CFG="pose/config_vispredict.py"
POSE_CKPT="pretrained/best_coco_AP_epoch_210.pth"

echo "========================================"
echo "PAMS Experiments on Occluded-Duke"
echo "GPU: ${GPU}"
echo "========================================"

# --- PAMS_full: Full PAMS (MSF + BPA + GiLt + Push + PartDist) ---
echo ""
echo ">>> [1/4] PAMS_full: Full PAMS"
python train.py --config_file configs/occluded_duke/pams_tiny.yml \
    MODEL.DEVICE_ID "'${GPU}'" \
    DATASETS.ROOT_DIR "'${DATA_DIR}'" \
    MODEL.PRETRAIN_PATH "'${PRETRAIN}'" \
    MODEL.POSE.CFG "'${POSE_CFG}'" \
    MODEL.POSE.CKPT "'${POSE_CKPT}'" \
    OUTPUT_DIR "'./log/occluded_duke/pams_full'"

# --- PAMS_no_push: Without Push Loss ---
echo ""
echo ">>> [2/4] PAMS_no_push: Without Push Loss"
python train.py --config_file configs/occluded_duke/pams_tiny.yml \
    MODEL.DEVICE_ID "'${GPU}'" \
    DATASETS.ROOT_DIR "'${DATA_DIR}'" \
    MODEL.PRETRAIN_PATH "'${PRETRAIN}'" \
    MODEL.POSE.CFG "'${POSE_CFG}'" \
    MODEL.POSE.CKPT "'${POSE_CKPT}'" \
    MODEL.PAMS.PUSH_WEIGHT 0.0 \
    OUTPUT_DIR "'./log/occluded_duke/pams_no_push'"

# --- PAMS_no_bpa: Without BPA supervision ---
echo ""
echo ">>> [3/4] PAMS_no_bpa: Without BPA supervision"
python train.py --config_file configs/occluded_duke/pams_tiny.yml \
    MODEL.DEVICE_ID "'${GPU}'" \
    DATASETS.ROOT_DIR "'${DATA_DIR}'" \
    MODEL.PRETRAIN_PATH "'${PRETRAIN}'" \
    MODEL.POSE.CFG "'${POSE_CFG}'" \
    MODEL.POSE.CKPT "'${POSE_CKPT}'" \
    MODEL.PAMS.BPA_WEIGHT 0.0 \
    OUTPUT_DIR "'./log/occluded_duke/pams_no_bpa'"

# --- PAMS_no_pose: Without pose (no BPA, part classifier learns from scratch) ---
echo ""
echo ">>> [4/4] PAMS_no_pose: Without pose predictor"
python train.py --config_file configs/occluded_duke/pams_tiny.yml \
    MODEL.DEVICE_ID "'${GPU}'" \
    DATASETS.ROOT_DIR "'${DATA_DIR}'" \
    MODEL.PRETRAIN_PATH "'${PRETRAIN}'" \
    MODEL.POSE.ENABLE False \
    MODEL.PAMS.BPA_WEIGHT 0.0 \
    OUTPUT_DIR "'./log/occluded_duke/pams_no_pose'"

echo ""
echo "========================================"
echo "All Occluded-Duke experiments done!"
echo "========================================"
