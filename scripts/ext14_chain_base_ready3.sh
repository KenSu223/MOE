#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): start the Stage-2 pipeline of the remaining models (v3: the engine lines must START the PROGRESS line; quoted mentions in other lines do not count):
#   qwen2_57b waits for "[p6-engine] PROTOCOL qwen2_57b" (bf16-fragile routers; coordinator 2026-10-17 02:37Z), the others for
#   "[p6-engine] MODEL READY <proto>"; runners read precision flags from moetrace/models.py at run time.
# Scans: scripts/ext14_chain_base_stage2b.sh; orchestrator scripts/ext14_chain_base_s2.sh (not for qwen3_1p7b: competence only).
# Usage: nohup setsid bash scripts/ext14_chain_base_ready2.sh qwen2_7b qwen2_57b qwen3_14b qwen3_1p7b > logs/ext14_base_ready.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
LOG=logs/ext14_base_chain.log
CELLS_ALL=e1_a2e,e1_e2a,e3p3_a2e,e3_a2e,e3_e2a,e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,sc_agree_a2e,sc_conflict_a2e,l_ant,l_prop,l_bind,l_ind,l_ioi
ready() {
  if [ "$1" = "qwen2_57b" ]; then grep -Eq "^[0-9T:Z.-]+ \[p6-engine\] PROTOCOL qwen2_57b" logs/PROGRESS.md
  else grep -Eq "^[0-9T:Z.-]+ \[p6-engine\] MODEL READY $1( |$)" logs/PROGRESS.md; fi
}
pending="$*"
while [ -n "$(echo $pending)" ]; do
  next=""
  for p in $pending; do
    if ready $p; then
      CELLS=$CELLS_ALL nohup setsid bash scripts/ext14_chain_base_stage2b.sh $p > logs/ext14_base_stage2_$p.out 2>&1 < /dev/null &
      [ "$p" != "qwen3_1p7b" ] && nohup setsid bash scripts/ext14_chain_base_s2.sh $p > logs/ext14_base_s2_$p.out 2>&1 < /dev/null &
      echo "$(date -u +%FT%TZ) ready-watcher3: started Stage 2 for $p" >> $LOG
    else
      next="$next $p"
    fi
  done
  pending="$next"
  [ -n "$(echo $pending)" ] && sleep 120
done
echo "$(date -u +%FT%TZ) ready-watcher3 done" >> $LOG
