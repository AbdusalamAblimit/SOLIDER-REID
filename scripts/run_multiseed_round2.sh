#!/bin/bash
# Summary-only helper for the old "round 2" experiment set.
#
# NOTE:
#   The actual round-2 experiments (exp007a + exp030a) are already fully
#   covered by scripts/run_multiseed_3090.sh.
#   To avoid duplicate training, this script no longer launches any jobs.
#   It only summarizes existing logs.
#
# Usage:
#   bash scripts/run_multiseed_round2.sh
#   LOGDIR=./4090_log/multiseed bash scripts/run_multiseed_round2.sh

set -euo pipefail

LOGDIR="${LOGDIR:-./log/multiseed}"
SEEDS=(1234 42 2024)

resolve_log() {
    local base_dir=$1
    local flat_path=$2
    local nested_path=$3

    if [ -f "${base_dir}/${flat_path}" ]; then
        echo "${base_dir}/${flat_path}"
    elif [ -f "${base_dir}/${nested_path}" ]; then
        echo "${base_dir}/${nested_path}"
    fi
}

print_metric_line() {
    local label=$1
    local log_file=$2

    printf "  %-48s " "$label"
    if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
        local map_value
        local rank1_value
        map_value=$(grep "mAP:" "${log_file}" | tail -1 | grep -oP "[0-9.]+%" || true)
        rank1_value=$(grep "Rank-1" "${log_file}" | tail -1 | grep -oP "[0-9.]+%" || true)
        if [ -n "${map_value}" ] && [ -n "${rank1_value}" ]; then
            echo "${map_value}  ${rank1_value}"
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

echo "scripts/run_multiseed_round2.sh no longer runs training."
echo "exp007a and exp030a are already included in scripts/run_multiseed_3090.sh."
echo "Summarizing existing logs from: ${LOGDIR}"
echo ""
echo "=== ROUND 2 SUMMARY ==="
printf "%-50s %8s %8s\n" "Experiment" "mAP" "R1"
echo "-------------------------------------------------------------------"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- SEED ${SEED} ---"

    exp007a_dir="${LOGDIR}/exp007a_psg_half_seed${SEED}"
    exp030a_dir="${LOGDIR}/exp030a_psg_gcn_seed${SEED}"

    exp007a_log=$(resolve_log "${exp007a_dir}" "test_default.txt" "test_default/test_log.txt")
    exp030a_global_log=$(resolve_log "${exp030a_dir}" "test_global.txt" "test_global/test_log.txt")
    exp030a_eq_log=$(resolve_log "${exp030a_dir}" "test_equal_concat.txt" "test_equal_concat/test_log.txt")
    exp030a_cs_log=$(resolve_log "${exp030a_dir}" "test_concat_scaled.txt" "test_concat_scaled/test_log.txt")
    exp030a_gcn_log=$(resolve_log "${exp030a_dir}" "test_gcn_only.txt" "test_gcn_only/test_log.txt")

    print_metric_line "exp007a PSG+0.5x" "${exp007a_log}"
    print_metric_line "exp030a PSG+GCN (global)" "${exp030a_global_log}"
    print_metric_line "exp030a PSG+GCN (equal_concat)" "${exp030a_eq_log}"
    print_metric_line "exp030a PSG+GCN (concat_scaled)" "${exp030a_cs_log}"
    print_metric_line "exp030a PSG+GCN (gcn_only)" "${exp030a_gcn_log}"
done

echo ""
echo "If you need to run the experiments, use scripts/run_multiseed_3090.sh."
