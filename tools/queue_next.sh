#!/bin/bash
# queue_next.sh — wait for current training's main PID to exit, then auto-launch next
#   if current finished OK (final checkpoint exists + no crash markers).
# Usage: queue_next.sh <current_pid> <current_log> <current_output_dir> <max_epochs> \
#                     <next_config> <next_output_dir> <next_log> <tag>
set -u
CURRENT_PID=$1
CURRENT_LOG=$2
CURRENT_OUTPUT_DIR=$3
MAX_EPOCHS=$4
NEXT_CONFIG=$5
NEXT_OUTPUT_DIR=$6
NEXT_LOG=$7
TAG=$8

QLOG=/tmp/queue_${TAG}.log
REPO=/root/work/SOLIDER-REID

echo "[$(date)] queue_next started tag=$TAG wait_pid=$CURRENT_PID" >> "$QLOG"

# Wait for main pid to exit
while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep 60
done
sleep 10  # let log flush

echo "[$(date)] pid $CURRENT_PID exited" >> "$QLOG"

# Success check: final checkpoint exists AND no crash markers in tail
FINAL_CKPT="${CURRENT_OUTPUT_DIR}/transformer_${MAX_EPOCHS}.pth"

if [ ! -f "$FINAL_CKPT" ]; then
    echo "[$(date)] ABORT: final ckpt $FINAL_CKPT not found. Current did not reach epoch $MAX_EPOCHS." >> "$QLOG"
    ls -la "$CURRENT_OUTPUT_DIR" >> "$QLOG" 2>&1
    tail -30 "$CURRENT_LOG" >> "$QLOG"
    exit 1
fi

if tail -200 "$CURRENT_LOG" | grep -qE "Traceback|OOM|Killed|CUDA error|RuntimeError.*assert|NaN|Inf"; then
    echo "[$(date)] ABORT: crash markers found in tail of $CURRENT_LOG" >> "$QLOG"
    tail -30 "$CURRENT_LOG" >> "$QLOG"
    exit 1
fi

echo "[$(date)] OK: final ckpt present, no crash markers. Launching next." >> "$QLOG"

cd "$REPO"
export PYTHONUNBUFFERED=1
echo "[$(date)] launching: python3 train.py --config_file $NEXT_CONFIG SOLVER.SEED 42 OUTPUT_DIR $NEXT_OUTPUT_DIR" >> "$QLOG"

nohup python3 train.py --config_file "$NEXT_CONFIG" SOLVER.SEED 42 OUTPUT_DIR "$NEXT_OUTPUT_DIR" \
    > "$NEXT_LOG" 2>&1 &
NEW_PID=$!
echo "[$(date)] next launched PID=$NEW_PID log=$NEXT_LOG" >> "$QLOG"
