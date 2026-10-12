#!/usr/bin/env bash
# ext13 Phase 5 (p5-data): base runners per condition and model (sweep, joint, dla, direct, expert, heads, grid; Phase-3
# runners) through the GPU queue, then the base analysis (CPU) which writes "[p5-data] BASE READY <cond>/<proto>".
# Jobs come from scripts/ext13_p5_jobs.py (skips stages with a .done_<stage> marker, so the chain can be restarted).
# Usage: nohup bash scripts/ext13_p5_chain_base.sh [cond ...] > logs/ext13_p5_chain_base.out 2>&1 &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400
export HF_HOME=/opt/dlami/nvme/hf
. .venv/bin/activate
LOG=logs/ext13_p5_chain.log
CONDS="${*:-f1 dir_f1 dir_opt cp_nat cp_vt_orig cp_vt_neu vt_e2a_orig vt_e2a_neu vt_a2e_orig vt_a2e_neu f2e}"
for cond in $CONDS; do
  for proto in qwen3 mixtral_bos; do
    ok=1
    # wait until the case set of this condition / model is registered (p5-data casesets step)
    until python -c "import json,sys; r=json.load(open('data/wino_ref/p5_conditions.json'))['conditions']; sys.exit(0 if '$proto' in r.get('$cond', {}) else 1)" 2>/dev/null; do sleep 60; done
    jobs=$(python scripts/ext13_p5_jobs.py "$cond" "$proto")
    while IFS=$'\t' read -r name st cmd; do
      [ -z "$name" ] && continue
      bash scripts/gpu_queue.sh "$name" -- $cmd > "logs/$name.log" 2>&1 < /dev/null; rc=$?
      echo "$(date -u +%FT%TZ) $name rc=$rc" >> $LOG
      if [ $rc -eq 0 ]; then mkdir -p "results/p5_${cond}_${proto}"; touch "results/p5_${cond}_${proto}/.done_$st"; else ok=0; break; fi
    done <<< "$jobs"
    if [ $ok -eq 1 ] && [ -f "results/p5_${cond}_${proto}/.done_grid" ]; then
      python scripts/ext13_p5_analyze.py "$cond" "$proto" >> "logs/ext13_p5_analyze_${cond}_${proto}.log" 2>&1 < /dev/null
      echo "$(date -u +%FT%TZ) analyze $cond $proto rc=$?" >> $LOG
    fi
  done
done
echo "$(date -u +%FT%TZ) chain_base done ($CONDS)" >> $LOG
