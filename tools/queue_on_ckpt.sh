#!/bin/bash
# queue_on_ckpt.sh — wait for a specific final checkpoint to appear, then launch next.
# Used to chain a 2nd-level run behind a 1st-level queue_next.sh launch: level 1
# already launches the intermediate exp via nohup, and this daemon simply watches
# the intermediate exp's final ckpt file.
#
# Usage: queue_on_ckpt.sh <wait_ckpt> <next_config> <next_output_dir> <next_log> <tag>
set -u
WAIT_CKPT=$1
NEXT_CONFIG=$2
NEXT_OUTPUT_DIR=$3
NEXT_LOG=$4
TAG=$5

QLOG=/tmp/queue_${TAG}.log
REPO=/root/work/SOLIDER-REID

echo "[$(date)] queue_on_ckpt started tag=$TAG waiting for $WAIT_CKPT" >> "$QLOG"

# Poll for the checkpoint (polls every 2 min; may wait many hours)
while [ ! -f "$WAIT_CKPT" ]; do
    sleep 120
done

echo "[$(date)] detected $WAIT_CKPT" >> "$QLOG"

# Wait for the training process to fully exit and release GPU
# Detect by looking for any python train.py process with OUTPUT_DIR matching
# the previous run's dir. When none is left, it's safe to start next.
PREV_DIR=$(dirname "$WAIT_CKPT")
while ps -eo cmd | grep -E "python3?.*train.py" | grep -v grep | grep -qF "OUTPUT_DIR $PREV_DIR"; do
    echo "[$(date)] previous run still alive, sleeping" >> "$QLOG"
    sleep 60
done

sleep 20  # extra safety

# Sanity check: no crash markers in previous log
PREV_LOG="$PREV_DIR/train_log.txt"
if [ -f "$PREV_LOG" ] && tail -200 "$PREV_LOG" | grep -qE "Traceback|OOM|Killed|CUDA error|RuntimeError.*assert|NaN|Inf"; then
    echo "[$(date)] ABORT: crash markers found in $PREV_LOG" >> "$QLOG"
    tail -30 "$PREV_LOG" >> "$QLOG"
    exit 1
fi

cd "$REPO"
export PYTHONUNBUFFERED=1
echo "[$(date)] launching: python3 train.py --config_file $NEXT_CONFIG SOLVER.SEED 42 OUTPUT_DIR $NEXT_OUTPUT_DIR" >> "$QLOG"
nohup python3 train.py --config_file "$NEXT_CONFIG" SOLVER.SEED 42 OUTPUT_DIR "$NEXT_OUTPUT_DIR" > "$NEXT_LOG" 2>&1 &
NEW_PID=$!
echo "[$(date)] next launched PID=$NEW_PID log=$NEXT_LOG" >> "$QLOG"
