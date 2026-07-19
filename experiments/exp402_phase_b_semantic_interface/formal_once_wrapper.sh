#!/usr/bin/env bash
set -uo pipefail
set -C

AUDIT_ROOT=/home/afr/reid-clean/audits/exp402_phase_b_semantic_interface
REPO=/home/afr/SOLIDER-REID-exp401-rich-budget-c0-formal-11d7a35
CONFIG=/home/afr/reid-clean/formal/exp401_rich_budget_c0/swin_tiny_tapf_rich_budget_c0_formal.yml
OUTPUT=/home/afr/SOLIDER-REID-exp401-rich-budget-c0-formal-11d7a35/log/occluded_duke/exp401_clean_swin_tiny_rich_budget_c0_s1234
CHECKPOINT=${OUTPUT}/transformer_120.pth
PYTHON=/home/afr/reid-clean/runtimes/exp394-openclip-reid-py310/bin/python
AUDIT=${AUDIT_ROOT}/actual_counterfactual_audit_run2.py
CORE=${AUDIT_ROOT}/counterfactual_core_run2.py
POSTFLIGHT=${AUDIT_ROOT}/postflight_manifest.py
RESULT=${AUDIT_ROOT}/formal_result_once.json
RUNNER=${AUDIT_ROOT}/formal_runner_once.log
MANIFEST=${AUDIT_ROOT}/formal_manifest_once.json

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
  --audit-script "${AUDIT}" \
  --core "${CORE}" \
  --audit-pid "${audit_pid}" \
  --exit-code "${audit_exit}"
postflight_exit=$?

if [[ "${audit_exit}" -ne 0 ]]; then
  exit "${audit_exit}"
fi
exit "${postflight_exit}"
