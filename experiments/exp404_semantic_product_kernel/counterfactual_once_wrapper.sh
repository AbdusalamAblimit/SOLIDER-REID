#!/usr/bin/env bash
set -uo pipefail
set -C

REPO=/home/afr/SOLIDER-REID-exp404-spk-formal-v1
ASSET_DIR=${REPO}/experiments/exp404_semantic_product_kernel
AUDIT_ROOT=/home/afr/reid-clean/audits/exp404-spk-counterfactual-v1
CONFIG=${ASSET_DIR}/swin_tiny_spk_formal.yml
OUTPUT=${REPO}/log/occluded_duke/exp404_spk_s1234
CHECKPOINT=${OUTPUT}/transformer_120.pth
PYTHON=/home/afr/reid-clean/runtimes/exp404-spk-py310/bin/python
RUNTIME_FREEZE=/home/afr/reid-clean/formal/exp404_spk/runtime_freeze.txt
AUDIT=${ASSET_DIR}/actual_counterfactual_audit.py
CORE=${ASSET_DIR}/counterfactual_core.py
POSTFLIGHT=${ASSET_DIR}/counterfactual_postflight.py
WRAPPER=${ASSET_DIR}/counterfactual_once_wrapper.sh
RESULT=${AUDIT_ROOT}/formal_result_once.json
RUNNER=${AUDIT_ROOT}/formal_runner_once.log
MANIFEST=${AUDIT_ROOT}/formal_manifest_once.json
LOCK=${AUDIT_ROOT}/formal_once.lock
PREFLIGHT=${AUDIT_ROOT}/preflight_result_v2.json

mkdir -p "${AUDIT_ROOT}"
for path in "${RESULT}" "${RUNNER}" "${MANIFEST}" "${LOCK}" "${RESULT}.tmp" "${MANIFEST}.tmp"; do
  if [[ -e "${path}" ]]; then
    echo "freshness gate failed"
    exit 20
  fi
done
if [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]; then
  echo "remote tracked repo is not clean"
  exit 21
fi
if [[ "$(git -C "${REPO}" status --porcelain)" != "?? log/" ]]; then
  echo "remote untracked set is not the sealed formal output"
  exit 25
fi
if [[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" != "03dbebb341e9d085e3d697505b8793cca217fca4a3b8f2a1f28fc512336e7d23" ]]; then
  echo "checkpoint SHA gate failed"
  exit 22
fi
if [[ "$(sha256sum "${CONFIG}" | awk '{print $1}')" != "2bd191ef96da0158a57f917831ea70627f1fef163397219ce1168e3e30bb297d" ]]; then
  echo "config SHA gate failed"
  exit 23
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "fresh runtime executable gate failed"
  exit 26
fi
if [[ "$(sha256sum "${RUNTIME_FREEZE}" | awk '{print $1}')" != "3d38c99c7f06502d8b40467d2674c966723e5c913d2edf962c5a7088ec60cddb" ]]; then
  echo "fresh runtime freeze gate failed"
  exit 27
fi
if [[ "$(sha256sum "${PREFLIGHT}" | awk '{print $1}')" != "cf7cfc5afbf1a865a95f60dd785964ae9288ad9965ad6e3bc9cdb424e8057f8c" ]]; then
  echo "counterfactual preflight SHA gate failed"
  exit 28
fi
preflight_gate=$("${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p.get("status"),p.get("decision"),p.get("formal_full_authorized"))' "${PREFLIGHT}")
if [[ "${preflight_gate}" != "PASS EXP404_COUNTERFACTUAL_PREFLIGHT_PASS True" ]]; then
  echo "counterfactual preflight authorization gate failed"
  exit 29
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU exclusivity gate failed"
  exit 24
fi
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LOCK}"

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
  --wrapper "${WRAPPER}" \
  --audit-pid "${audit_pid}" \
  --exit-code "${audit_exit}"
postflight_exit=$?

if [[ "${audit_exit}" -ne 0 ]]; then
  exit "${audit_exit}"
fi
exit "${postflight_exit}"
