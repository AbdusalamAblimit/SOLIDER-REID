#!/bin/bash
# Multi-seed experiments for 4090
# Purpose: Verify whether PDS+StopGrad truly improves over PSG-only,
#          or if the +1.2% gap is just training variance.
#
# Experiments: exp000 (baseline), exp007 (PSG), exp023 (PDS+StopGrad)
# Seeds: 1234 (original), 42, 2024
#
# Usage: bash scripts/run_multiseed_4090.sh
# Expected time: ~18 hours (9 runs × ~2h each)

set -e

PYTHON="python"  # adjust if needed, e.g. /path/to/python
SEEDS=(1234 42 2024)

# ============================================================
# Helper: run training + 4 test modes
# ============================================================
run_experiment() {
    local CONFIG=$1
    local OUTPUT_DIR=$2
    local EXTRA_OPTS=$3

    echo "========================================"
    echo "Training: $OUTPUT_DIR"
    echo "Config:   $CONFIG"
    echo "Extra:    $EXTRA_OPTS"
    echo "========================================"

    mkdir -p "$OUTPUT_DIR"

    # Train
    PYTHONUNBUFFERED=1 $PYTHON train.py \
        --config_file "$CONFIG" \
        OUTPUT_DIR "$OUTPUT_DIR" \
        $EXTRA_OPTS \
        2>&1 | tee "$OUTPUT_DIR/train_log.txt"

    # Find the checkpoint
    local CKPT=$(ls -t "$OUTPUT_DIR"/transformer_*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found in $OUTPUT_DIR"
        return 1
    fi
    echo "Checkpoint: $CKPT"

    # Test: dual-stream models get 4 modes, others get single test
    if echo "$CONFIG" | grep -q "pose_pds"; then
        # Dual-stream (PDS) models: 4 test modes
        for MODE in global part_only equal_concat concat_scaled; do
            local TEST_DIR="$OUTPUT_DIR/test_${MODE}"
            echo "Testing: $MODE -> $TEST_DIR"
            mkdir -p "$TEST_DIR"
            PYTHONUNBUFFERED=1 $PYTHON test.py \
                --config_file "$CONFIG" \
                TEST.WEIGHT "$CKPT" \
                MODEL.POSE_TEST_FEAT "$MODE" \
                OUTPUT_DIR "$TEST_DIR" \
                $EXTRA_OPTS \
                2>&1 | tee "$TEST_DIR/test_log.txt"
        done
    else
        # Single-stream models (baseline, PSG): single test
        local TEST_DIR="$OUTPUT_DIR/test_default"
        echo "Testing: default -> $TEST_DIR"
        mkdir -p "$TEST_DIR"
        PYTHONUNBUFFERED=1 $PYTHON test.py \
            --config_file "$CONFIG" \
            TEST.WEIGHT "$CKPT" \
            OUTPUT_DIR "$TEST_DIR" \
            $EXTRA_OPTS \
            2>&1 | tee "$TEST_DIR/test_log.txt"
    fi

    echo "Done: $OUTPUT_DIR"
    echo ""
}

# ============================================================
# Run all experiments
# ============================================================

for SEED in "${SEEDS[@]}"; do
    echo "############################################"
    echo "# SEED: $SEED"
    echo "############################################"

    # --- exp000: Baseline ---
    run_experiment \
        "configs/occluded_duke/swin_tiny.yml" \
        "./log/multiseed/exp000_baseline_seed${SEED}" \
        "SOLVER.SEED $SEED"

    # --- exp007: PSG (Pose Spatial Gate) ---
    run_experiment \
        "configs/occluded_duke/pose_backbone_psg.yml" \
        "./log/multiseed/exp007_psg_seed${SEED}" \
        "SOLVER.SEED $SEED"

    # --- exp023: PDS + StopGrad ---
    run_experiment \
        "configs/occluded_duke/pose_pds_stopgrad.yml" \
        "./log/multiseed/exp023_pds_stopgrad_seed${SEED}" \
        "SOLVER.SEED $SEED"
done

echo ""
echo "============================================"
echo "All multi-seed experiments complete!"
echo "Results in: ./log/multiseed/"
echo "============================================"

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- SEED $SEED ---"

    # Baseline
    echo -n "  exp000 baseline:       "
    grep "mAP:" "./log/multiseed/exp000_baseline_seed${SEED}/test_default/test_log.txt" 2>/dev/null || echo "N/A"

    # PSG (single-stream, only global)
    echo -n "  exp007 PSG:            "
    grep "mAP:" "./log/multiseed/exp007_psg_seed${SEED}/test_default/test_log.txt" 2>/dev/null || echo "N/A"

    # PDS+StopGrad (4 modes)
    for MODE in global part_only equal_concat concat_scaled; do
        printf "  exp023 PDS+SG (%-15s): " "$MODE"
        grep "mAP:" "./log/multiseed/exp023_pds_stopgrad_seed${SEED}/test_${MODE}/test_log.txt" 2>/dev/null || echo "N/A"
    done
done
