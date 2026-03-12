#!/bin/bash
# Multi-seed experiments — variance analysis
# Purpose: Determine whether PSG+GCN improvements are real or training variance.
#
# Configs (3):
#   1. exp007: PSG only (1.0x loss)     — configs/occluded_duke/pose_backbone_psg.yml
#   2. exp007a: PSG + 0.5x loss scale   — configs/occluded_duke/pose_psg_half_loss.yml
#   3. exp030a: PSG + GCN (equal_concat) — configs/occluded_duke/pose_psg_gcn.yml
#
# Seeds: 1234 (original), 42, 2024
# Expected time: ~4.5h on 4090, ~18h on 3090
#
# Usage: nohup bash scripts/run_multiseed_3090.sh > log/multiseed/run.log 2>&1 &
# Prerequisites: conda activate solider-reid, cd to repo root

set -e

# Use conda env python — adjust path if different on your machine
PYTHON="${PYTHON:-python}"
SEEDS=(1234 42 2024)
LOGDIR="./log/multiseed"

mkdir -p "$LOGDIR"

run_experiment() {
    local CONFIG=$1
    local OUTPUT_DIR=$2
    local EXTRA_OPTS=$3
    local HAS_GCN=$4  # "yes" or "no"

    echo "========================================"
    echo "Training: $OUTPUT_DIR"
    echo "Config:   $CONFIG"
    echo "Extra:    $EXTRA_OPTS"
    echo "Has GCN:  $HAS_GCN"
    echo "Start:    $(date)"
    echo "========================================"

    mkdir -p "$OUTPUT_DIR"

    # Train
    PYTHONUNBUFFERED=1 $PYTHON train.py \
        --config_file "$CONFIG" \
        OUTPUT_DIR "$OUTPUT_DIR" \
        $EXTRA_OPTS \
        2>&1 | tee "$OUTPUT_DIR/train_log.txt"

    # Find checkpoint
    local CKPT=$(ls -t "$OUTPUT_DIR"/transformer_*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found in $OUTPUT_DIR"
        return 1
    fi
    echo "Checkpoint: $CKPT"

    # Extract seed from EXTRA_OPTS for test (remove POSE_TEST_FEAT if present)
    local SEED_OPTS=$(echo "$EXTRA_OPTS" | sed 's/MODEL.POSE_TEST_FEAT [a-z_]*//g')

    if [ "$HAS_GCN" = "yes" ]; then
        # GCN models: test global + equal_concat
        # POSE_TEST_FEAT must come LAST to override any value in EXTRA_OPTS
        for MODE in global equal_concat; do
            echo "Testing: $MODE"
            PYTHONUNBUFFERED=1 $PYTHON test.py \
                --config_file "$CONFIG" \
                TEST.WEIGHT "$CKPT" \
                $SEED_OPTS \
                MODEL.POSE_TEST_FEAT "$MODE" \
                2>&1 | tee "$OUTPUT_DIR/test_${MODE}.txt"
        done
    else
        # Single-stream: default test
        echo "Testing: default"
        PYTHONUNBUFFERED=1 $PYTHON test.py \
            --config_file "$CONFIG" \
            TEST.WEIGHT "$CKPT" \
            $SEED_OPTS \
            2>&1 | tee "$OUTPUT_DIR/test_default.txt"
    fi

    echo "Done: $OUTPUT_DIR ($(date))"
    echo ""
}

# ============================================================
# Run all experiments
# ============================================================

for SEED in "${SEEDS[@]}"; do
    echo "############################################"
    echo "# SEED: $SEED ($(date))"
    echo "############################################"

    # --- exp007: PSG only (1.0x loss) ---
    run_experiment \
        "configs/occluded_duke/pose_backbone_psg.yml" \
        "${LOGDIR}/exp007_psg_seed${SEED}" \
        "SOLVER.SEED $SEED" \
        "no"

    # --- exp007a: PSG + 0.5x loss ---
    run_experiment \
        "configs/occluded_duke/pose_psg_half_loss.yml" \
        "${LOGDIR}/exp007a_psg_half_seed${SEED}" \
        "SOLVER.SEED $SEED" \
        "no"

    # --- exp030a: PSG + GCN (equal_concat) ---
    run_experiment \
        "configs/occluded_duke/pose_psg_gcn.yml" \
        "${LOGDIR}/exp030a_psg_gcn_seed${SEED}" \
        "SOLVER.SEED $SEED MODEL.POSE_TEST_FEAT equal_concat" \
        "yes"
done

echo ""
echo "============================================"
echo "All multi-seed experiments complete! ($(date))"
echo "Results in: ${LOGDIR}/"
echo "============================================"

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
printf "%-45s %8s %8s\n" "Experiment" "mAP" "R1"
echo "-------------------------------------------------------------"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- SEED $SEED ---"

    # exp007 PSG
    local_log="${LOGDIR}/exp007_psg_seed${SEED}/test_default.txt"
    printf "  %-43s " "exp007 PSG (1.0x)"
    if [ -f "$local_log" ]; then
        grep -oP "mAP: \K[0-9.]+%" "$local_log" | tr -d '\n'
        echo -n "  "
        grep -oP "Rank-1\s*:\K[0-9.]+%" "$local_log"
    else
        echo "N/A"
    fi

    # exp007a PSG + 0.5x
    local_log="${LOGDIR}/exp007a_psg_half_seed${SEED}/test_default.txt"
    printf "  %-43s " "exp007a PSG (0.5x)"
    if [ -f "$local_log" ]; then
        grep -oP "mAP: \K[0-9.]+%" "$local_log" | tr -d '\n'
        echo -n "  "
        grep -oP "Rank-1\s*:\K[0-9.]+%" "$local_log"
    else
        echo "N/A"
    fi

    # exp030a global
    local_log="${LOGDIR}/exp030a_psg_gcn_seed${SEED}/test_global.txt"
    printf "  %-43s " "exp030a PSG+GCN (global)"
    if [ -f "$local_log" ]; then
        grep -oP "mAP: \K[0-9.]+%" "$local_log" | tr -d '\n'
        echo -n "  "
        grep -oP "Rank-1\s*:\K[0-9.]+%" "$local_log"
    else
        echo "N/A"
    fi

    # exp030a equal_concat
    local_log="${LOGDIR}/exp030a_psg_gcn_seed${SEED}/test_equal_concat.txt"
    printf "  %-43s " "exp030a PSG+GCN (equal_concat)"
    if [ -f "$local_log" ]; then
        grep -oP "mAP: \K[0-9.]+%" "$local_log" | tr -d '\n'
        echo -n "  "
        grep -oP "Rank-1\s*:\K[0-9.]+%" "$local_log"
    else
        echo "N/A"
    fi
done
