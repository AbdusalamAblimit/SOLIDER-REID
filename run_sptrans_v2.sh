#!/bin/bash
set -e

BASE_MARKET="configs/market/sptrans_tiny.yml"
BASE_DUKE="configs/occluded_duke/sptrans_tiny.yml"
LOG_FILE="sptrans_v2_experiments.log"

run() {
    NAME=$1; CFG=$2; shift 2
    echo "===== Running $NAME =====" | tee -a "$LOG_FILE"
    python train.py --config_file "$CFG" OUTPUT_DIR "./log/sptrans_v2/$NAME" "$@" 2>&1 | tee -a "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

# SP2_E0: baseline-dual (no innovations)
run "SP2_E0_baseline_market" "$BASE_MARKET" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT False \
    MODEL.SPTRANS.MID_ROUTING False MODEL.SPTRANS.LOSS_STRATEGY unified

run "SP2_E0_baseline_duke" "$BASE_DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT False \
    MODEL.SPTRANS.MID_ROUTING False MODEL.SPTRANS.LOSS_STRATEGY unified

# SP2_E1: adaptive-sem-only
run "SP2_E1_adaptive_sem_market" "$BASE_MARKET" \
    MODEL.SPTRANS.ADAPTIVE_SEM True MODEL.SPTRANS.PART_EXPERT False \
    MODEL.SPTRANS.MID_ROUTING False MODEL.SPTRANS.LOSS_STRATEGY unified

run "SP2_E1_adaptive_sem_duke" "$BASE_DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM True MODEL.SPTRANS.PART_EXPERT False \
    MODEL.SPTRANS.MID_ROUTING False MODEL.SPTRANS.LOSS_STRATEGY unified

# SP2_E2: part-expert-only (PartExpertHead, no MidRouting)
run "SP2_E2_part_expert_market" "$BASE_MARKET" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING False MODEL.SPTRANS.LOSS_STRATEGY part_expert

run "SP2_E2_part_expert_duke" "$BASE_DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING False MODEL.SPTRANS.LOSS_STRATEGY part_expert

# SP2_E3: mid-routing-only (MoE in stages, no PartExpertHead at output)
run "SP2_E3_mid_routing_market" "$BASE_MARKET" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT False \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY unified

run "SP2_E3_mid_routing_duke" "$BASE_DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT False \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY unified

# SP2_E4: expert+routing (Direction 3 complete, no AdaptiveSem)
run "SP2_E4_expert_routing_market" "$BASE_MARKET" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY part_expert

run "SP2_E4_expert_routing_duke" "$BASE_DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM False MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY part_expert

# SP2_E5: full-unified (all features ON, unified loss)
run "SP2_E5_full_unified_market" "$BASE_MARKET" \
    MODEL.SPTRANS.ADAPTIVE_SEM True MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY unified

run "SP2_E5_full_unified_duke" "$BASE_DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM True MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY unified

# SP2_E6: full-part-expert (all features ON, part_expert loss)
run "SP2_E6_full_part_expert_market" "$BASE_MARKET" \
    MODEL.SPTRANS.ADAPTIVE_SEM True MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY part_expert

run "SP2_E6_full_part_expert_duke" "$BASE_DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM True MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY part_expert

# SP2_E7: full-split (all features ON, split loss for comparison)
run "SP2_E7_full_split_market" "$BASE_MARKET" \
    MODEL.SPTRANS.ADAPTIVE_SEM True MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY split

run "SP2_E7_full_split_duke" "$BASE_DUKE" \
    MODEL.SPTRANS.ADAPTIVE_SEM True MODEL.SPTRANS.PART_EXPERT True \
    MODEL.SPTRANS.MID_ROUTING True MODEL.SPTRANS.LOSS_STRATEGY split

echo "All SPTrans v2 experiments completed!" | tee -a "$LOG_FILE"
