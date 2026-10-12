#!/usr/bin/env bash
# ext13ko / ext13wr detached chain for one protocol (Phase 5, agent p5-expert).
#   nohup bash scripts/ext13ko_chain.sh qwen3|mixtral_bos > logs/ext13ko_chain_<proto>.log 2>&1 &
# Waits for the exact PROGRESS lines "[p5-data] BASE READY <cond>/<proto>", then queues knockout + writer jobs
# (scripts/ext13ko_run.py --with-wr, <= ~22 min per queue job, resumable) for every ready, unfinished condition; after each
# job runs the CPU analyses and appends a PROGRESS line per finished condition. When every condition is finished (or
# logs/ext13ko_<proto>.nomore exists and nothing is pending) it runs the cross phase (other conditions' selected experts).
# Stop: touch logs/ext13ko_<proto>.stop
cd "$(dirname "$0")/.."
proto=$1
CONDS=${CONDS:-"f1 dir_f1 dir_opt cp_nat cp_vt_orig cp_vt_neu vt_e2a_orig vt_e2a_neu vt_a2e_orig vt_a2e_neu f2e"}
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
log() { echo "$(date -u +%FT%TZ) $*"; }
deadline=$(( $(date +%s) + ${CHAIN_HOURS:-60} * 3600 ))
declare -A announced
analyse() {
  flock -x logs/ext13ko_analyze.lock bash -c "python scripts/ext13ko_analyze.py > logs/ext13ko_analyze.log 2>&1; python scripts/ext13wr_analyze.py > logs/ext13wr_analyze.log 2>&1"
}
log "chain start $proto: $CONDS"
while [ "$(date +%s)" -lt "$deadline" ]; do
  [ -f "logs/ext13ko_${proto}.stop" ] && { log "stop file"; exit 0; }
  todo=(); pending=0; ndone=0
  for c in $CONDS; do
    if python scripts/ext13ko_status.py "$proto" "$c" 2>/dev/null; then
      ndone=$((ndone + 1))
      if [ -z "${announced[$c]}" ]; then
        announced[$c]=1
        if ! grep -qF "[p5-expert] KO+WR DONE $c/$proto" logs/PROGRESS.md; then
          analyse
          line=$(python scripts/ext13ko_progress_line.py "$proto" "$c" 2>&1 | tail -1)
          echo "- $(date -u +%FT%TZ) [p5-expert] KO+WR DONE $c/$proto: $line" >> logs/PROGRESS.md
          log "announced $c"
        fi
      fi
      continue
    fi
    if grep -qF "[p5-data] BASE READY $c/$proto" logs/PROGRESS.md; then todo+=("$c"); else pending=$((pending + 1)); fi
  done
  if [ ${#todo[@]} -gt 0 ]; then
    list=$(IFS=,; echo "${todo[*]}")
    if ! python scripts/ext13ko_run.py "$proto" --conds "$list" --plan-only >> "logs/ext13ko_$proto.log" 2>&1; then
      log "plan-only (CPU) failed for $list; retry in 10 min"; sleep 600; continue
    fi
    log "queue job for $list"
    bash scripts/gpu_queue.sh "ext13ko-$proto" -- python scripts/ext13ko_run.py "$proto" --conds "$list" --with-wr --time-budget 1320 >> "logs/ext13ko_$proto.log" 2>&1
    rc=$?
    log "job rc=$rc"
    if [ $rc -ne 0 ]; then
      fails=$((fails + 1)); [ $fails -ge 3 ] && { log "3 failed jobs, exit"; exit 1; }
      sleep 60
    fi
    continue
  fi
  if [ $pending -eq 0 ] || { [ -f "logs/ext13ko_${proto}.nomore" ] && [ $ndone -gt 0 ]; }; then
    # cross phase over the finished conditions
    fin=(); for c in $CONDS; do python scripts/ext13ko_status.py "$proto" "$c" 2>/dev/null && fin+=("$c"); done
    left=(); for c in "${fin[@]}"; do python scripts/ext13ko_status.py "$proto" "$c" --phase cross 2>/dev/null || left+=("$c"); done
    if [ ${#left[@]} -gt 0 ] && [ -z "$NO_CROSS" ]; then
      list=$(IFS=,; echo "${left[*]}")
      log "queue cross job for $list"
      bash scripts/gpu_queue.sh "ext13ko-cross-$proto" -- python scripts/ext13ko_run.py "$proto" --conds "$list" --phase cross --time-budget 1320 >> "logs/ext13ko_$proto.log" 2>&1
      log "cross job rc=$?"
      continue
    fi
    analyse
    echo "- $(date -u +%FT%TZ) [p5-expert] chain $proto finished (conditions: ${fin[*]}; cross phase done)" >> logs/PROGRESS.md
    log "chain done"
    exit 0
  fi
  sleep 120
done
log "deadline"
