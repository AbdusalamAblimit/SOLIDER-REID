#!/usr/bin/env bash
set -uo pipefail
set -C

ROOT=/home/afr/reid-clean/audits/exp403_elo_cur_final_v1
REPO=/home/afr/SOLIDER-REID-exp403-elo-cur-fe854ea
CONFIG=/home/afr/reid-clean/formal/exp403_elo_cur/swin_tiny_elo_cur_formal.yml
OUTPUT=${REPO}/log/occluded_duke/exp403_elo_cur_s1234
CHECKPOINT=${OUTPUT}/transformer_120.pth
PYTHON=/home/afr/reid-clean/runtimes/exp394-openclip-reid-py310/bin/python
CONTRACT=${ROOT}/checkpoint_contract.json
AUDIT=${ROOT}/final_counterfactual_audit.py
BASE_AUDIT=${ROOT}/base_actual_counterfactual_audit.py
CORE=${ROOT}/final_counterfactual_core.py
BASE_CORE=${ROOT}/base_counterfactual_core.py
POSTFLIGHT=${ROOT}/final_postflight.py
RESULT=${ROOT}/formal_result_once.json
RUNNER=${ROOT}/formal_runner_once.log
MANIFEST=${ROOT}/formal_manifest_once.json

export EXP403_CONTRACT=${CONTRACT}
export EXP403_BASE_AUDIT=${BASE_AUDIT}
export EXP403_BASE_CORE=${BASE_CORE}
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${AUDIT}" \
  --repo-root "${REPO}" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT}" \
  --core "${CORE}" \
  --result "${RESULT}" \
  > "${RUNNER}" 2>&1 &
audit_pid=$!
wait "${audit_pid}"
audit_exit=$?

"${PYTHON}" "${POSTFLIGHT}" \
  --result "${RESULT}" \
  --runner "${RUNNER}" \
  --manifest "${MANIFEST}" \
  --contract "${CONTRACT}" \
  --audit-wrapper "${AUDIT}" \
  --base-audit "${BASE_AUDIT}" \
  --core "${CORE}" \
  --base-core "${BASE_CORE}" \
  --execution-wrapper "$0" \
  --audit-pid "${audit_pid}" \
  --exit-code "${audit_exit}"
postflight_exit=$?

if [[ "${audit_exit}" -ne 0 ]]; then
  exit "${audit_exit}"
fi
exit "${postflight_exit}"
