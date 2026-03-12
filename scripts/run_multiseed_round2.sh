#!/bin/bash
# Multi-seed Round 2 — confirm loss scaling and GCN contributions
#
# Round 1 已完成: baseline × 3, PSG × 3, PDS+StopGrad × 3
# Round 2 目标:
#   1. exp007a: PSG + 0.5x loss — 确认 loss scaling 是否为真实 sweet spot
#   2. exp030a: PSG + GCN        — 确认 GCN 互补特征贡献
#
# Seeds: 1234, 42, 2024
# Expected time: ~3h on 4090 (2 configs × 3 seeds × ~30min)
#
# Usage: nohup bash scripts/run_multiseed_round2.sh > log/multiseed/round2.log 2>&1 &

set -e

PYTHON="${PYTHON:-python}"
SEEDS=(1234 42 2024)
LOGDIR="./log/multiseed"

mkdir -p "$LOGDIR"

run_experiment() {
    local CONFIG=$1
    local OUTPUT_DIR=$2
    local EXTRA_OPTS=$3
    local TEST_MODES=$4  # comma-separated: "default" or "global,equal_concat,concat_scaled,gcn_only"

    echo "========================================"
    echo "Training: $OUTPUT_DIR"
    echo "Config:   $CONFIG"
    echo "Extra:    $EXTRA_OPTS"
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

    # Clean EXTRA_OPTS: remove POSE_TEST_FEAT if present
    local CLEAN_OPTS=$(echo "$EXTRA_OPTS" | sed 's/MODEL.POSE_TEST_FEAT [a-z_]*//g')

    # Test each mode
    IFS=',' read -ra MODES <<< "$TEST_MODES"
    for MODE in "${MODES[@]}"; do
        echo "Testing: $MODE"
        mkdir -p "$OUTPUT_DIR/test_${MODE}"
        if [ "$MODE" = "default" ]; then
            PYTHONUNBUFFERED=1 $PYTHON test.py \
                --config_file "$CONFIG" \
                TEST.WEIGHT "$CKPT" \
                $CLEAN_OPTS \
                2>&1 | tee "$OUTPUT_DIR/test_${MODE}/test_log.txt"
        else
            PYTHONUNBUFFERED=1 $PYTHON test.py \
                --config_file "$CONFIG" \
                TEST.WEIGHT "$CKPT" \
                $CLEAN_OPTS \
                MODEL.POSE_TEST_FEAT "$MODE" \
                2>&1 | tee "$OUTPUT_DIR/test_${MODE}/test_log.txt"
        fi
    done

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

    # --- exp007a: PSG + 0.5x loss scale ---
    # KEY: this confirms whether 0.5x loss scaling is a real sweet spot
    run_experiment \
        "configs/occluded_duke/pose_psg_half_loss.yml" \
        "${LOGDIR}/exp007a_psg_half_seed${SEED}" \
        "SOLVER.SEED $SEED" \
        "default"

    # --- exp030a: PSG + GCN ---
    # Tests global (isolate PSG effect) and equal_concat (PSG + GCN combined)
    run_experiment \
        "configs/occluded_duke/pose_psg_gcn.yml" \
        "${LOGDIR}/exp030a_psg_gcn_seed${SEED}" \
        "SOLVER.SEED $SEED" \
        "global,equal_concat,concat_scaled,gcn_only"
done

echo ""
echo "============================================"
echo "Round 2 multi-seed experiments complete! ($(date))"
echo "============================================"

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
printf "%-50s %8s %8s\n" "Experiment" "mAP" "R1"
echo "-------------------------------------------------------------------"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- SEED $SEED ---"

    # exp007a
    local_log="${LOGDIR}/exp007a_psg_half_seed${SEED}/train_log.txt"
    printf "  %-48s " "exp007a PSG+0.5x"
    if [ -f "$local_log" ]; then
        mAP=$(grep "mAP:" "$local_log" | tail -1 | grep -oP "[0-9.]+%")
        R1=$(grep "Rank-1" "$local_log" | tail -1 | grep -oP "[0-9.]+%")
        echo "$mAP  $R1"
    else
        echo "N/A"
    fi

    # exp030a global
    local_log="${LOGDIR}/exp030a_psg_gcn_seed${SEED}/test_global/test_log.txt"
    printf "  %-48s " "exp030a PSG+GCN (global)"
    if [ -f "$local_log" ]; then
        mAP=$(grep "mAP:" "$local_log" | tail -1 | grep -oP "[0-9.]+%")
        R1=$(grep "Rank-1" "$local_log" | tail -1 | grep -oP "[0-9.]+%")
        echo "$mAP  $R1"
    else
        echo "N/A"
    fi

    # exp030a equal_concat
    local_log="${LOGDIR}/exp030a_psg_gcn_seed${SEED}/test_equal_concat/test_log.txt"
    printf "  %-48s " "exp030a PSG+GCN (equal_concat)"
    if [ -f "$local_log" ]; then
        mAP=$(grep "mAP:" "$local_log" | tail -1 | grep -oP "[0-9.]+%")
        R1=$(grep "Rank-1" "$local_log" | tail -1 | grep -oP "[0-9.]+%")
        echo "$mAP  $R1"
    else
        echo "N/A"
    fi
done

echo ""
echo "=== ROUND 1 REFERENCE (已完成) ==="
echo "Baseline: 56.50% mean (56.7/55.9/56.9)"
echo "PSG:      57.83% mean (58.3/57.9/57.3)"
echo "PDS+SG:   59.20% mean (59.7/59.2/58.7) [global-only]"
