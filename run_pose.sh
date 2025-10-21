#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
BASE_CFG="${2:-configs/msmt17/swin_tiny_pose.yml}"
TAG="${3:-msmt_pose_sweep}"

EPOCHS="${EPOCHS:-120}"
EVAL_PERIOD="${EVAL_PERIOD:-10}"
CKPT_PERIOD="${CKPT_PERIOD:-60}"

DATA_ROOT="${DATA_ROOT:-data}"
BATCH="${BATCH:-64}"
BASE_LR="${BASE_LR:-0.0008}"

# 注意：布尔值用 True/False（大写）
FUSION_MODES=("mul" "add" "gate")
FUSE_STAGES=(1 2)
SCALES=(0.1 0.3 1.0)
HM_NORMS=("sigmoid" "softmax")
USE_VIS_OPTS=("True" "False")
DETACH="True"

RUN_BASELINES="${RUN_BASELINES:-1}"

PY=python
ENTRY=train.py

ROOT_OUT="./log/msmt17/${TAG}"
mkdir -p "${ROOT_OUT}"

run_job() {
  local mode="$1"
  local stage="$2"
  local scale="$3"
  local hmn="$4"
  local usevis="$5"
  local detach="$6"

  local name="fm-${mode}_fs-${stage}_sc-${scale}_hn-${hmn}_uv-${usevis}_dt-${detach}"
  local out="${ROOT_OUT}/${name}"
  mkdir -p "${out}"

  if [[ -f "${out}/last_checkpoint" ]]; then
    echo "[SKIP] ${name} (found last_checkpoint)"
    return 0
  fi

  echo "==> RUN ${name}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  ${PY} -u ${ENTRY} \
    --config_file "${BASE_CFG}" \
    DATASETS.ROOT_DIR "${DATA_ROOT}" \
    SOLVER.IMS_PER_BATCH ${BATCH} \
    SOLVER.BASE_LR ${BASE_LR} \
    SOLVER.MAX_EPOCHS ${EPOCHS} \
    SOLVER.EVAL_PERIOD ${EVAL_PERIOD} \
    SOLVER.CHECKPOINT_PERIOD ${CKPT_PERIOD} \
    MODEL.TRANSFORMER_TYPE pose_swin_tiny_patch4_window7_224 \
    MODEL.POSE.ENABLE True \
    MODEL.POSE.SAVE_VIS False \
    MODEL.POSE.FUSION_MODE "${mode}" \
    MODEL.POSE.FUSE_STAGE ${stage} \
    MODEL.POSE.SCALE ${scale} \
    MODEL.POSE.HM_NORM "${hmn}" \
    MODEL.POSE.USE_VIS ${usevis} \
    MODEL.POSE.DETACH ${detach} \
    OUTPUT_DIR "${out}" \
    | tee "${out}/train.log"
}

# baseline: 关闭 pose
if [[ "${RUN_BASELINES}" == "1" ]]; then
  name_base="baseline_no_pose"
  out_base="${ROOT_OUT}/${name_base}"
  mkdir -p "${out_base}"
  if [[ ! -f "${out_base}/last_checkpoint" ]]; then
    echo "==> RUN ${name_base}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    ${PY} -u ${ENTRY} \
      --config_file "${BASE_CFG}" \
      DATASETS.ROOT_DIR "${DATA_ROOT}" \
      SOLVER.IMS_PER_BATCH ${BATCH} \
      SOLVER.BASE_LR ${BASE_LR} \
      SOLVER.MAX_EPOCHS ${EPOCHS} \
      SOLVER.EVAL_PERIOD ${EVAL_PERIOD} \
      SOLVER.CHECKPOINT_PERIOD ${CKPT_PERIOD} \
      MODEL.TRANSFORMER_TYPE pose_swin_tiny_patch4_window7_224 \
      MODEL.POSE.ENABLE False \
      OUTPUT_DIR "${out_base}" \
      | tee "${out_base}/train.log"
  else
    echo "[SKIP] ${name_base}"
  fi

  # baseline2: 开 pose 但 SCALE=0
  name_zero="pose_scale0"
  out_zero="${ROOT_OUT}/${name_zero}"
  mkdir -p "${out_zero}"
  if [[ ! -f "${out_zero}/last_checkpoint" ]]; then
    echo "==> RUN ${name_zero}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    ${PY} -u ${ENTRY} \
      --config_file "${BASE_CFG}" \
      DATASETS.ROOT_DIR "${DATA_ROOT}" \
      SOLVER.IMS_PER_BATCH ${BATCH} \
      SOLVER.BASE_LR ${BASE_LR} \
      SOLVER.MAX_EPOCHS ${EPOCHS} \
      SOLVER.EVAL_PERIOD ${EVAL_PERIOD} \
      SOLVER.CHECKPOINT_PERIOD ${CKPT_PERIOD} \
      MODEL.TRANSFORMER_TYPE pose_swin_tiny_patch4_window7_224 \
      MODEL.POSE.ENABLE True \
      MODEL.POSE.SAVE_VIS False \
      MODEL.POSE.FUSION_MODE mul \
      MODEL.POSE.FUSE_STAGE 2 \
      MODEL.POSE.SCALE 0.0 \
      MODEL.POSE.HM_NORM sigmoid \
      MODEL.POSE.USE_VIS True \
      MODEL.POSE.DETACH True \
      OUTPUT_DIR "${out_zero}" \
      | tee "${out_zero}/train.log"
  else
    echo "[SKIP] ${name_zero}"
  fi
fi

# 主扫
for mode in "${FUSION_MODES[@]}"; do
  for stage in "${FUSE_STAGES[@]}"; do
    for scale in "${SCALES[@]}"; do
      for hmn in "${HM_NORMS[@]}"; do
        for usevis in "${USE_VIS_OPTS[@]}"; do
          run_job "${mode}" "${stage}" "${scale}" "${hmn}" "${usevis}" "${DETACH}"
        done
      done
    done
  done
done

# 汇总
python -u tools/summarize_pose_logs.py "${ROOT_OUT}" > "${ROOT_OUT}/sweep_results.csv"
echo "==> DONE. Summary saved at ${ROOT_OUT}/sweep_results.csv"
