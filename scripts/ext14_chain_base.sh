#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): tier-1 base runners per cell and proto (sweep, joint, dla, direct; heads only where needed; scope cut 2026-10-17)
# through the GPU queue, then the per-cell base analysis (CPU), which writes "[p6-data] BASE READY <cell>/<proto>".
# Cells are processed in the given order, both protos per cell (interleaved), skipping finished stages (.done_<stage> markers).
# A cell without a registry entry yet, or under REVIEW, is deferred until "[coordinator] REVIEW OK <cell>/<proto>" (or skipped on VOID); the chain loops over deferred
# cells (the first runnable cell of the list always goes next; nothing runnable -> sleep 5 min). A failed job is retried once with 3/4 of the row budgets (OOM), then the cell is left.
# Usage: nohup setsid bash scripts/ext14_chain_base.sh "<cell ...>" "<proto ...>" > logs/ext14_base_chain_<tag>.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
CELLS="$1"
PROTOS="${2:-qwen3 mixtral_bos}"
echo "$(date -u +%FT%TZ) chain_base start cells=[$CELLS] protos=[$PROTOS] pid=$$" >> $LOG

run_cell() {  # returns 0 done / skipped, 2 deferred (review), 1 failed
  local cell=$1 proto=$2 scale=1.0 tries=0
  # not registered yet (case sets pending): defer, the main loop comes back
  python -c "import json,sys; r=json.load(open('data/dial/p6_conditions.json'))['conditions']; sys.exit(0 if '$proto' in r.get('$cell', {}) else 1)" 2>/dev/null || return 2
  local d="results/p6_${cell}_${proto}"
  while :; do
    jobs=$(python scripts/ext14_jobs.py "$cell" "$proto" --scale $scale)
    first=$(echo "$jobs" | head -1 | cut -f1)
    if [ "$first" = "SKIP" ]; then echo "$(date -u +%FT%TZ) $cell/$proto skipped: $(echo "$jobs" | head -1 | cut -f2)" >> $LOG; return 0; fi
    if [ "$first" = "WAIT" ]; then return 2; fi
    local failed=0
    while IFS=$'\t' read -r name st cmd; do
      [ -z "$name" ] && continue
      bash scripts/gpu_queue.sh "$name" -- $cmd > "logs/$name.log" 2>&1 < /dev/null; rc=$?
      echo "$(date -u +%FT%TZ) $name rc=$rc scale=$scale" >> $LOG
      if [ $rc -eq 0 ]; then mkdir -p "$d"; touch "$d/.done_$st"; rm -f "$d/.done_analysis"; else failed=1; break; fi
    done <<< "$jobs"
    if [ $failed -eq 0 ]; then break; fi
    tries=$((tries + 1))
    if [ $tries -ge 2 ]; then echo "$(date -u +%FT%TZ) $cell/$proto FAILED after $tries attempts" >> $LOG; return 1; fi
    scale=0.75
  done
  # every required stage done (scope cut: sweep / joint / dla / direct, + heads where needed) -> base analysis
  if [ -f "$d/.done_sweep" ] && [ -f "$d/.done_joint" ] && [ -f "$d/.done_dla" ] && [ -f "$d/.done_direct" ] && [ ! -f "$d/.done_analysis" ]; then
    python scripts/ext14_base_analyze.py "$cell" "$proto" >> "logs/ext14_base_analyze_${cell}_${proto}.log" 2>&1 < /dev/null; rc=$?
    echo "$(date -u +%FT%TZ) analyze $cell $proto rc=$rc" >> $LOG
    [ $rc -eq 0 ] && touch "$d/.done_analysis"
  fi
  return 0
}

# priority loop: always run the FIRST runnable cell of the list (registered, no pending review), so that cells registered later
# but listed earlier (e.g. Dial E after its extension scans) overtake later-listed cells
pending=""
for cell in $CELLS; do for proto in $PROTOS; do pending="$pending $cell:$proto"; done; done
while [ -n "$(echo $pending)" ]; do
  ran=0; next=""
  for cp in $pending; do
    if [ $ran -eq 1 ]; then next="$next $cp"; continue; fi
    cell=${cp%%:*}; proto=${cp##*:}
    run_cell "$cell" "$proto"; rc=$?
    if [ $rc -eq 2 ]; then next="$next $cp"; else ran=1; fi
  done
  pending="$next"
  if [ $ran -eq 0 ] && [ -n "$(echo $pending)" ]; then echo "$(date -u +%FT%TZ) deferred (case sets or review pending):$pending" >> $LOG; sleep 300; fi
done
echo "$(date -u +%FT%TZ) chain_base done cells=[$CELLS] protos=[$PROTOS]" >> $LOG
