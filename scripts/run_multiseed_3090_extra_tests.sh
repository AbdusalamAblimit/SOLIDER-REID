#!/bin/bash
# Supplement missing exp030a test modes after scripts/run_multiseed_3090.sh.
#
# scripts/run_multiseed_3090.sh trains:
#   - exp007
#   - exp007a
#   - exp030a
# but for exp030a it only tests:
#   - global
#   - equal_concat
#
# This helper adds any missing exp030a test modes from:
#   - global
#   - equal_concat
#   - concat_scaled
#   - gcn_only
#
# Existing result files with a valid "mAP:" line are skipped.
#
# Usage:
#   bash scripts/run_multiseed_3090_extra_tests.sh
#   LOGDIR=./log/multiseed bash scripts/run_multiseed_3090_extra_tests.sh
#   PYTHON=/path/to/python bash scripts/run_multiseed_3090_extra_tests.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
LOGDIR="${LOGDIR:-./log/multiseed}"
SEEDS=(1234 42 2024)
TEST_MODES=(global equal_concat concat_scaled gcn_only)
CONFIG="configs/occluded_duke/pose_psg_gcn.yml"

has_valid_result() {
    local result_file=$1
    [ -f "${result_file}" ] && grep -q "mAP:" "${result_file}"
}

run_test_mode() {
    local output_dir=$1
    local ckpt=$2
    local seed=$3
    local mode=$4
    local result_file="${output_dir}/test_${mode}.txt"

    if has_valid_result "${result_file}"; then
        echo "Skip ${output_dir##*/} ${mode}: existing result found"
        return 0
    fi

    echo "Testing ${output_dir##*/} mode=${mode}"
    PYTHONUNBUFFERED=1 "${PYTHON}" test.py \
        --config_file "${CONFIG}" \
        TEST.WEIGHT "${ckpt}" \
        SOLVER.SEED "${seed}" \
        MODEL.POSE_TEST_FEAT "${mode}" \
        2>&1 | tee "${result_file}"
}

echo "Supplementing missing exp030a test modes in: ${LOGDIR}"
echo ""

for seed in "${SEEDS[@]}"; do
    output_dir="${LOGDIR}/exp030a_psg_gcn_seed${seed}"

    if [ ! -d "${output_dir}" ]; then
        echo "Skip seed ${seed}: missing directory ${output_dir}"
        continue
    fi

    ckpt=$(ls -t "${output_dir}"/transformer_*.pth 2>/dev/null | head -1 || true)
    if [ -z "${ckpt}" ]; then
        echo "Skip seed ${seed}: no checkpoint found in ${output_dir}"
        continue
    fi

    echo "========================================"
    echo "Seed:       ${seed}"
    echo "Output dir: ${output_dir}"
    echo "Checkpoint: ${ckpt}"
    echo "========================================"

    for mode in "${TEST_MODES[@]}"; do
        run_test_mode "${output_dir}" "${ckpt}" "${seed}" "${mode}"
    done

    echo ""
done

echo "Done."
