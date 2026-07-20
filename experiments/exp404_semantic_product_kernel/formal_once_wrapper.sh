#!/usr/bin/env bash
set -euo pipefail

REPO=/home/afr/SOLIDER-REID-exp404-spk-formal-v1
ASSET=/home/afr/reid-clean/formal/exp404_spk
CONFIG=$ASSET/swin_tiny_spk_formal.yml
PREFLIGHT=$ASSET/cuda_amp_preflight_v3_result.json
PYTHON=/home/afr/reid-clean/runtimes/exp404-spk-py310/bin/python
OUTPUT=$REPO/log/occluded_duke/exp404_spk_s1234
RUNNER=$ASSET/formal_train_v1.runner.log
LAUNCH=$ASSET/formal_train_v1.launch.json
LOCK=$ASSET/formal_train_v1.launch.lock

test ! -e "$OUTPUT"
test ! -e "$RUNNER"
test ! -e "$LAUNCH"
test ! -e "$LOCK"
test -x "$PYTHON"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "2bd191ef96da0158a57f917831ea70627f1fef163397219ce1168e3e30bb297d"
test "$(sha256sum "$PREFLIGHT" | awk '{print $1}')" = "70566973f0387d0b335040ff20fe2c1f091563cc18f4a65370b25aac303d58bf"
test "$(sha256sum "$ASSET/runtime_freeze.txt" | awk '{print $1}')" = "3d38c99c7f06502d8b40467d2674c966723e5c913d2edf962c5a7088ec60cddb"

cd "$REPO"
test -z "$(git status --short)"
test "$(sha256sum model/tapf.py | awk '{print $1}')" = "72ff5a609c7a080d848e96a2c12239795388441cc13b85519ef2cbf42f04bf2a"
test "$(sha256sum model/make_model.py | awk '{print $1}')" = "44de28f34b675366606e4ae4734567f50c6ede755fd85280073c514543d61f76"
test "$(sha256sum processor/processor.py | awk '{print $1}')" = "bc98121ab179e44f091ef6e7cabf9f75b6e2cfa3390ccba930d1324553a4beb1"

GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')
test -z "$GPU_PIDS"
mkdir "$LOCK"

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 nohup "$PYTHON" train.py \
  --config_file "$CONFIG" > "$RUNNER" 2>&1 &
PID=$!
sleep 2
kill -0 "$PID"
HEAD=$(git rev-parse HEAD)
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
umask 022
printf '{\n  "config_sha256": "%s",\n  "execution": "exp404_formal_train_v1",\n  "head": "%s",\n  "main_pid": %s,\n  "preflight_sha256": "%s",\n  "resume": false,\n  "started_utc": "%s"\n}\n' \
  "2bd191ef96da0158a57f917831ea70627f1fef163397219ce1168e3e30bb297d" \
  "$HEAD" "$PID" \
  "70566973f0387d0b335040ff20fe2c1f091563cc18f4a65370b25aac303d58bf" \
  "$STARTED" > "$LAUNCH"
chmod 0444 "$LAUNCH"
printf '%s\n' "$PID"
