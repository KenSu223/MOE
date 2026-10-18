#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): tier 1 of one or more cells of ONE model inside a single GPU-queue job (throughput fix, coordinator
# 2026-10-17): for each cell the pending stages (scripts/ext14_jobs.py --stages: sweep -> joint (-> joint_shared) -> dla -> direct
# (-> heads)) run back-to-back; every stage writes its .done_<stage> marker, so a failure resumes at the failed stage. A per-cell lock
# (results/p6_<cell>_<proto>/.lock_t1 with the PID) keeps two jobs from running the same cell. The base analysis stays outside (CPU,
# in the calling chain). Exit status = that of the FIRST cell (the calling chain retries it with 3/4 budgets); later cells of the bundle
# are picked up again by their own chain if they fail.
# Usage (inside gpu_queue.sh): bash scripts/ext14_chain_base_cell.sh <scale> <proto> <cell> [<cell> ...]
cd "$(dirname "$0")/.."
export HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
scale=$1; proto=$2; shift 2
first_rc=0; i=0
for cell in "$@"; do
  d="results/p6_${cell}_${proto}"; mkdir -p "$d"
  lock="$d/.lock_t1"
  if [ -f "$lock" ] && kill -0 "$(cat $lock 2>/dev/null)" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) $cell/$proto locked by $(cat $lock); skipped"; [ $i -eq 0 ] && first_rc=3; i=$((i + 1)); continue
  fi
  echo $$ > "$lock"
  rc=0
  while IFS=$'\t' read -r name st cmd; do
    [ -z "$name" ] && continue
    case "$name" in SKIP|WAIT) break;; esac
    t0=$(date +%s)
    $cmd > "logs/$name.log" 2>&1 < /dev/null; rc=$?
    echo "$(date -u +%FT%TZ) $name rc=$rc ($(( $(date +%s) - t0 ))s, in bundle job)"
    if [ $rc -eq 0 ]; then touch "$d/.done_$st"; rm -f "$d/.done_analysis"; else break; fi
  done <<< "$(python scripts/ext14_jobs.py "$cell" "$proto" --scale "$scale" --stages)"
  rm -f "$lock"
  [ $i -eq 0 ] && first_rc=$rc
  i=$((i + 1))
done
exit $first_rc
