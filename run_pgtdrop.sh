#!/bin/bash
set -e

BASE_DUKE="configs/occluded_duke/pgtdrop_tiny.yml"
BASE_MARKET="configs/market/pgtdrop_tiny.yml"
LOG_FILE="pgtdrop_experiments.log"

run() {
    NAME=$1; CFG=$2; shift 2
    echo "===== Running $NAME =====" | tee -a "$LOG_FILE"
    python train.py --config_file "$CFG" OUTPUT_DIR "./log/pgtdrop/$NAME" "$@" 2>&1 | tee -a "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

# Phase 1: Occ-Duke experiments

# PGT_base: Single Swin baseline (no masking, no vis-pool)
run "PGT_base_duke" "$BASE_DUKE" \
    MODEL.PGTDROP.KEEP_RATIO 1.0 \
    MODEL.PGTDROP.VIS_POOL False \
    MODEL.POSE.ENABLE False

# PGT_s1_k70: Main experiment — drop 30% after stage 1
run "PGT_s1_k70_duke" "$BASE_DUKE" \
    MODEL.PGTDROP.DROP_STAGE 1 \
    MODEL.PGTDROP.KEEP_RATIO 0.7

# PGT_s1_k50: Aggressive — drop 50%
run "PGT_s1_k50_duke" "$BASE_DUKE" \
    MODEL.PGTDROP.DROP_STAGE 1 \
    MODEL.PGTDROP.KEEP_RATIO 0.5

# PGT_s1_k70_rd15: With random augmentation
run "PGT_s1_k70_rd15_duke" "$BASE_DUKE" \
    MODEL.PGTDROP.DROP_STAGE 1 \
    MODEL.PGTDROP.KEEP_RATIO 0.7 \
    MODEL.PGTDROP.RANDOM_DROP 0.15

# PGT_s0_k70: Early drop — after stage 0
run "PGT_s0_k70_duke" "$BASE_DUKE" \
    MODEL.PGTDROP.DROP_STAGE 0 \
    MODEL.PGTDROP.KEEP_RATIO 0.7

echo "All PGTDrop Occ-Duke experiments completed!" | tee -a "$LOG_FILE"

# Uncomment below for Phase 2: Market (if Occ-Duke shows improvement)
# run "PGT_base_market" "$BASE_MARKET" \
#     MODEL.PGTDROP.KEEP_RATIO 1.0 \
#     MODEL.PGTDROP.VIS_POOL False \
#     MODEL.POSE.ENABLE False
#
# run "PGT_s1_k70_market" "$BASE_MARKET" \
#     MODEL.PGTDROP.DROP_STAGE 1 \
#     MODEL.PGTDROP.KEEP_RATIO 0.7
