#!/bin/bash
# Overnight experiment pipeline - runs sequentially, server never idles
# Started: $(date)
# Each experiment ~1.5-2.5h depending on single/dual branch
set -e

MARKET="configs/market/sptrans_tiny.yml"
DUKE="configs/occluded_duke/sptrans_tiny.yml"
LOG="overnight_results.log"

run() {
    local NAME=$1 CFG=$2; shift 2
    local DIR="./log/sptrans_v2/$NAME"
    mkdir -p "$DIR"

    # Skip if already completed (has final epoch eval in log)
    if [ -f "$DIR/train_log.txt" ] && grep -q "Epoch: 120" "$DIR/train_log.txt" 2>/dev/null; then
        echo "[SKIP] $NAME already completed" | tee -a "$LOG"
        return
    fi

    echo "" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"
    echo "[START] $NAME  $(date)" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"
    python train.py --config_file "$CFG" OUTPUT_DIR "$DIR" "$@" 2>&1 | tee -a "$DIR/run.log"

    # Extract final results
    echo "[DONE] $NAME  $(date)" | tee -a "$LOG"
    grep -E "(Validation Results|mAP:|Rank-1)" "$DIR/train_log.txt" | tail -12 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

echo "========== Overnight Pipeline Started: $(date) ==========" | tee -a "$LOG"

# --- Currently running: single_branch_duke (will be skipped if done) ---

# 1. Single-branch + vis-pool on Occ-Duke (may already be running/done)
run "single_branch_duke" "$DUKE" \
    MODEL.SPTRANS.SINGLE_BRANCH True \
    MODEL.SPTRANS.LOSS_STRATEGY unified

# 2. Single-branch + vis-pool on Market (comparison)
run "single_branch_market" "$MARKET" \
    MODEL.SPTRANS.SINGLE_BRANCH True \
    MODEL.SPTRANS.LOSS_STRATEGY unified

# 3. Baseline dual-branch on Occ-Duke (no innovations, just dual Swin)
run "SP2_E0_baseline_duke" "$DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT False \
    MODEL.SPTRANS.MID_ROUTING False MODEL.SPTRANS.SINGLE_BRANCH False \
    MODEL.SPTRANS.LOSS_STRATEGY unified

# 4. Full v2 (MoE+Expert+AdaptSem) on Occ-Duke
run "SP2_E6_full_part_expert_duke" "$DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM True MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.SINGLE_BRANCH False \
    MODEL.SPTRANS.LOSS_STRATEGY part_expert

# 5. Expert+Routing only (no AdaptSem) on Occ-Duke
run "SP2_E4_expert_routing_duke" "$DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.SINGLE_BRANCH False \
    MODEL.SPTRANS.LOSS_STRATEGY part_expert

# 6. Baseline dual-branch on Market (clean reference)
run "SP2_E0_baseline_market" "$MARKET" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT False \
    MODEL.SPTRANS.MID_ROUTING False MODEL.SPTRANS.SINGLE_BRANCH False \
    MODEL.SPTRANS.LOSS_STRATEGY unified

# --- Summary ---
echo "" | tee -a "$LOG"
echo "========== Overnight Pipeline Finished: $(date) ==========" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "===== RESULTS SUMMARY =====" | tee -a "$LOG"
for d in ./log/sptrans_v2/*/; do
    name=$(basename "$d")
    if [ -f "$d/train_log.txt" ]; then
        echo "--- $name ---" | tee -a "$LOG"
        grep -E "(Validation Results|mAP:|Rank-1)" "$d/train_log.txt" | tail -12 | tee -a "$LOG"
        echo "" | tee -a "$LOG"
    fi
done
