#!/usr/bin/env bash
# Serialise GPU jobs from several agents on the single A10G.
# Usage:  bash scripts/gpu_queue.sh <job-name> -- <command ...>
# Waits (up to GPU_QUEUE_WAIT_S, default 4 h) for an exclusive lock, runs the command with the project venv and
# HF_HOME set, logs start/end/rc to logs/gpu_queue.log. Exit code is the command's.
# Priority (Phase 6, 2026-10-17): jobs whose name matches the regex in logs/gpu_queue_lowprio.txt (one ERE per file,
# default none) are LOW priority: they only take the lock while no high-priority job is waiting (waiting jobs leave a
# marker logs/gpu_wait/<pid>.hi). Low-priority jobs never time out while yielding.
set -uo pipefail
cd "$(dirname "$0")/.."
name="$1"; shift; [ "${1:-}" = "--" ] && shift
LOCK=logs/gpu.lock; LOG=logs/gpu_queue.log; WAITD=logs/gpu_wait; mkdir -p logs "$WAITD"
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/hf}
. .venv/bin/activate
exec 9>"$LOCK"
t0=$(date +%s)
low=0
if [ -s logs/gpu_queue_lowprio.txt ] && echo "$name" | grep -Eq -f logs/gpu_queue_lowprio.txt; then low=1; fi
hi_waiting() {
  local f p n=0
  for f in "$WAITD"/*.hi; do
    [ -e "$f" ] || continue
    p=$(basename "$f" .hi)
    if kill -0 "$p" 2>/dev/null; then n=$((n+1)); else rm -f "$f"; fi
  done
  [ "$n" -gt 0 ]
}
if [ "$low" = 1 ]; then
  while true; do
    if hi_waiting; then sleep 20; continue; fi
    if flock -x -w 20 9; then
      if hi_waiting; then flock -u 9; sleep 5; continue; fi
      break
    fi
  done
else
  touch "$WAITD/$$.hi"; trap 'rm -f "$WAITD/$$.hi"' EXIT
  if ! flock -x -w "${GPU_QUEUE_WAIT_S:-14400}" 9; then
    echo "$(date -u +%FT%TZ) TIMEOUT waiting for GPU lock: $name" | tee -a "$LOG" >&2; exit 75
  fi
  rm -f "$WAITD/$$.hi"
fi
echo "$(date -u +%FT%TZ) START $name (waited $(( $(date +%s) - t0 ))s$([ "$low" = 1 ] && echo ', low priority')): $*" | tee -a "$LOG"
t1=$(date +%s)
"$@"; rc=$?
echo "$(date -u +%FT%TZ) END   $name rc=$rc ($(( $(date +%s) - t1 ))s)" | tee -a "$LOG"
exit $rc
