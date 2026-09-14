#!/usr/bin/env bash
# Serialise GPU jobs from several agents on the single A10G.
# Usage:  bash scripts/gpu_queue.sh <job-name> -- <command ...>
# Waits (up to GPU_QUEUE_WAIT_S, default 4 h) for an exclusive lock, runs the command with the project venv and
# HF_HOME set, logs start/end/rc to logs/gpu_queue.log. Exit code is the command's.
set -uo pipefail
cd "$(dirname "$0")/.."
name="$1"; shift; [ "${1:-}" = "--" ] && shift
LOCK=logs/gpu.lock; LOG=logs/gpu_queue.log; mkdir -p logs
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/hf}
. .venv/bin/activate
exec 9>"$LOCK"
t0=$(date +%s)
if ! flock -x -w "${GPU_QUEUE_WAIT_S:-14400}" 9; then
  echo "$(date -u +%FT%TZ) TIMEOUT waiting for GPU lock: $name" | tee -a "$LOG" >&2; exit 75
fi
echo "$(date -u +%FT%TZ) START $name (waited $(( $(date +%s) - t0 ))s): $*" | tee -a "$LOG"
t1=$(date +%s)
"$@"; rc=$?
echo "$(date -u +%FT%TZ) END   $name rc=$rc ($(( $(date +%s) - t1 ))s)" | tee -a "$LOG"
exit $rc
