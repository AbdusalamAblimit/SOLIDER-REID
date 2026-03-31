#!/bin/bash
# 4090 上运行最有希望的实验
# 用法: bash scripts/run_4090_best.sh [1|2|3]
# 1 = GCN+PAA+CE+OA-SD on Small (目标: 超过 70.8)
# 2 = STD-PR+SupCon+3-view on Small (目标: 复现/超过 69.3，4090 不需 CP 可能更快更好)
# 3 = GCN+PAA+CE+OA-SD+3-view on Small (终极 CE 路线，目标: 73+)

set -e
cd "$(dirname "$0")/.."

EXP=${1:-1}

case $EXP in
  1)
    echo "=== Experiment 1: GCN+PAA+CE+OA-SD on Swin-Small ==="
    echo "Target: 70.8 (4090 PAA baseline) + OA-SD boost → 73+"
    PYTHONUNBUFFERED=1 python train.py \
      --config_file configs/occluded_duke/pose_psg_gcn_paa_roa.yml \
      MODEL.TRANSFORMER_TYPE swin_small_patch4_window7_224 \
      MODEL.PRETRAIN_PATH pretrained/swin_small.pth \
      MODEL.POSE_LOWER_BODY_OCC True \
      MODEL.POSE_LOWER_BODY_OCC_PROB 0.7 \
      MODEL.POSE_OA_SD True \
      SOLVER.BASE_LR 0.0004 \
      SOLVER.CHECKPOINT_PERIOD 20 \
      OUTPUT_DIR ./log/occluded_duke/4090_gcn_paa_oa_sd_small
    ;;
  2)
    echo "=== Experiment 2: STD-PR+SupCon+3-view on Swin-Small (no CP needed on 4090) ==="
    echo "Target: 69.3 (exp202b with CP) → 70+ without CP overhead"
    PYTHONUNBUFFERED=1 python train.py \
      --config_file configs/occluded_duke/pose_psg_stdpr_pertoken_plboa_pape_ms_supcon_small.yml \
      MODEL.POSE_PARALLEL_AUG True \
      SOLVER.CHECKPOINT_PERIOD 20 \
      OUTPUT_DIR ./log/occluded_duke/4090_stdpr_supcon_3view_small
    ;;
  3)
    echo "=== Experiment 3: GCN+PAA+CE+OA-SD+3-view on Swin-Small (Ultimate CE) ==="
    echo "Target: GCN arch (70.8 ceiling) + OA-SD (+2.9) + 3-view (+1.1) → 73-74+"
    PYTHONUNBUFFERED=1 python train.py \
      --config_file configs/occluded_duke/pose_psg_gcn_paa_roa.yml \
      MODEL.TRANSFORMER_TYPE swin_small_patch4_window7_224 \
      MODEL.PRETRAIN_PATH pretrained/swin_small.pth \
      MODEL.POSE_LOWER_BODY_OCC True \
      MODEL.POSE_LOWER_BODY_OCC_PROB 0.7 \
      MODEL.POSE_OA_SD True \
      MODEL.POSE_PARALLEL_AUG True \
      SOLVER.BASE_LR 0.0004 \
      SOLVER.CHECKPOINT_PERIOD 20 \
      OUTPUT_DIR ./log/occluded_duke/4090_gcn_paa_oa_sd_3view_small
    ;;
  *)
    echo "Usage: bash scripts/run_4090_best.sh [1|2|3]"
    echo "  1 = GCN+PAA+CE+OA-SD (expected: 73+)"
    echo "  2 = STD-PR+SupCon+3-view (expected: 70+)"
    echo "  3 = GCN+PAA+CE+OA-SD+3-view (expected: 73-74+)"
    exit 1
    ;;
esac
