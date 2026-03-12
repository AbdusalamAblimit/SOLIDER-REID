#!/bin/bash
# Run exp030 variants on a 4090 box.
#
# Default runs:
#   1. market_tiny
#   2. market_small
#   3. market_base
#   4. occluded_small
#   5. occluded_base
#
# Usage:
#   bash scripts/run_exp030_4090.sh
#   bash scripts/run_exp030_4090.sh market_tiny occluded_base
#   GPU_ID=0 PYTHON=/root/miniconda3/envs/solider-reid/bin/python bash scripts/run_exp030_4090.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1234}"
TEST_MODES=(global part_only equal_concat concat_scaled)

EXPERIMENTS=(
  "market_tiny|configs/market/pose_pds_sg_gcn.yml|log/market1501/exp030_pds_sg_gcn|pretrained/swin_tiny.pth|data/market1501/pose_data"
  "market_small|configs/market/pose_pds_sg_gcn_small.yml|log/market1501/exp030_pds_sg_gcn_small|pretrained/swin_small.pth|data/market1501/pose_data"
  "market_base|configs/market/pose_pds_sg_gcn_base.yml|log/market1501/exp030_pds_sg_gcn_base|pretrained/swin_base.pth|data/market1501/pose_data"
  "occluded_small|configs/occluded_duke/pose_pds_sg_gcn_small.yml|log/occluded_duke/exp030_pds_sg_gcn_small|pretrained/swin_small.pth|data/occluded_duke/pose_data"
  "occluded_base|configs/occluded_duke/pose_pds_sg_gcn_base.yml|log/occluded_duke/exp030_pds_sg_gcn_base|pretrained/swin_base.pth|data/occluded_duke/pose_data"
)

require_file() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
}

require_pose_data() {
  local pose_dir="$1"
  require_file "$pose_dir/train/index.json"
  require_file "$pose_dir/query/index.json"
  require_file "$pose_dir/gallery/index.json"
}

should_run() {
  local label="$1"
  if [ "$#" -eq 1 ] && [ "$SELECT_ALL" = "1" ]; then
    return 0
  fi

  shift
  local selected
  for selected in "$@"; do
    if [ "$selected" = "$label" ]; then
      return 0
    fi
  done
  return 1
}

run_experiment() {
  local label="$1"
  local config="$2"
  local output_dir="$3"
  local pretrain="$4"
  local pose_dir="$5"
  local final_ckpt="$output_dir/transformer_120.pth"
  local ckpt=""
  local device_opts=(MODEL.DEVICE_ID "('$GPU_ID')" SOLVER.SEED "$SEED" OUTPUT_DIR "$output_dir")

  echo "=================================================="
  echo "Experiment: $label"
  echo "Config:     $config"
  echo "Output:     $output_dir"
  echo "GPU_ID:     $GPU_ID"
  echo "SEED:       $SEED"
  echo "=================================================="

  require_file "$config"
  require_file "$pretrain"
  require_pose_data "$pose_dir"
  mkdir -p "$output_dir"

  if [ -f "$final_ckpt" ]; then
    echo "Found final checkpoint, skipping training: $final_ckpt"
    ckpt="$final_ckpt"
  else
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" train.py \
      --config_file "$config" \
      "${device_opts[@]}" \
      2>&1 | tee "$output_dir/train_log.txt"

    ckpt="$(ls -t "$output_dir"/transformer_*.pth 2>/dev/null | head -1 || true)"
    if [ -z "$ckpt" ]; then
      echo "ERROR: no checkpoint found in $output_dir" >&2
      exit 1
    fi
  fi

  echo "Using checkpoint: $ckpt"

  local mode
  for mode in "${TEST_MODES[@]}"; do
    local test_dir="$output_dir/test_${mode}"
    mkdir -p "$test_dir"
    echo "Testing mode: $mode"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" test.py \
      --config_file "$config" \
      TEST.WEIGHT "$ckpt" \
      MODEL.DEVICE_ID "('$GPU_ID')" \
      MODEL.POSE_TEST_FEAT "$mode" \
      OUTPUT_DIR "$test_dir" \
      2>&1 | tee "$test_dir/test_log.txt"
  done
}

print_summary() {
  echo
  echo "==================== Summary ===================="
  local spec
  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r label _ output_dir _ _ <<< "$spec"
    if ! should_run "$label" "${REQUESTED[@]}"; then
      continue
    fi

    echo
    echo "[$label]"
    local mode
    for mode in "${TEST_MODES[@]}"; do
      local test_log="$output_dir/test_${mode}/test_log.txt"
      if [ -f "$test_log" ]; then
        local map_line
        local rank1_line
        map_line="$(grep 'mAP:' "$test_log" | tail -1 | sed 's/^.*INFO: //' || true)"
        rank1_line="$(grep 'CMC curve, Rank-1' "$test_log" | tail -1 | sed 's/^.*INFO: //' || true)"
        printf '  %-13s %s | %s\n' "$mode" "${map_line:-mAP: N/A}" "${rank1_line:-CMC curve, Rank-1 : N/A}"
      else
        printf '  %-13s %s\n' "$mode" "missing test log"
      fi
    done
  done
  echo "================================================="
}

if [ "$#" -eq 0 ]; then
  SELECT_ALL=1
  REQUESTED=()
else
  SELECT_ALL=0
  REQUESTED=("$@")
fi

for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r label config output_dir pretrain pose_dir <<< "$spec"
  if ! should_run "$label" "${REQUESTED[@]}"; then
    continue
  fi
  run_experiment "$label" "$config" "$output_dir" "$pretrain" "$pose_dir"
done

print_summary
