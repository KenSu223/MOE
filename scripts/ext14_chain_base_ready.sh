#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): start the Stage-2 pipeline of every remaining model when p6-engine writes "[p6-engine] MODEL READY <proto>":
# scans (scripts/ext14_chain_base_stage2b.sh) and, except for qwen3_1p7b (competence scan only), the orchestrator
# scripts/ext14_chain_base_s2.sh (extension scans, case sets, smoke, tier-1 chain). All children nohup setsid.
# Usage: nohup setsid bash scripts/ext14_chain_base_ready.sh qwen2_7b qwen2_57b qwen3_14b qwen3_1p7b > logs/ext14_base_ready.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
LOG=logs/ext14_base_chain.log
CELLS_ALL=e1_a2e,e1_e2a,e3p3_a2e,e3_a2e,e3_e2a,e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,sc_agree_a2e,sc_conflict_a2e,l_ant,l_prop,l_bind,l_ind,l_ioi
pending="$*"
while [ -n "$(echo $pending)" ]; do
  next=""
  for p in $pending; do
    if grep -q "\[p6-engine\] MODEL READY $p\b" logs/PROGRESS.md; then
      CELLS=$CELLS_ALL nohup setsid bash scripts/ext14_chain_base_stage2b.sh $p > logs/ext14_base_stage2_$p.out 2>&1 < /dev/null &
      if [ "$p" != "qwen3_1p7b" ]; then
        nohup setsid bash scripts/ext14_chain_base_s2.sh $p > logs/ext14_base_s2_$p.out 2>&1 < /dev/null &
      fi
      echo "$(date -u +%FT%TZ) ready-watcher: started Stage 2 for $p" >> $LOG
    else
      next="$next $p"
    fi
  done
  pending="$next"
  [ -n "$(echo $pending)" ] && sleep 120
done
echo "$(date -u +%FT%TZ) ready-watcher done" >> $LOG
