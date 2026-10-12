#!/usr/bin/env bash
# Coordinator fix (2026-10-11 16:10Z): cp_nat / mixtral_bos base runners OOMed in the sweep (60k rows, longer prompts with the
# options prefix, the largest pool). Same jobs as scripts/ext13_p5_jobs.py with smaller row budgets and expandable segments;
# .done_<stage> markers as in scripts/ext13_p5_chain_base.sh, then the base analysis (writes BASE READY cp_nat/mixtral_bos).
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext13_p5_chain.log; cond=cp_nat; proto=mixtral_bos; ok=1
jobs=$(python scripts/ext13_p5_jobs.py $cond $proto | sed -e 's/--max-rows 60000/--max-rows 25000/g' -e 's/--max-rows 50000/--max-rows 25000/g' -e 's/--max-rows 40000/--max-rows 25000/g' -e 's/--token-budget 200000/--token-budget 120000/g')
while IFS=$'\t' read -r name st cmd; do
  [ -z "$name" ] && continue
  bash scripts/gpu_queue.sh "$name" -- $cmd > "logs/$name.log" 2>&1 < /dev/null; rc=$?
  echo "$(date -u +%FT%TZ) $name rc=$rc (fix chain)" >> $LOG
  if [ $rc -eq 0 ]; then mkdir -p "results/p5_${cond}_${proto}"; touch "results/p5_${cond}_${proto}/.done_$st"; else ok=0; break; fi
done <<< "$jobs"
if [ $ok -eq 1 ] && [ -f "results/p5_${cond}_${proto}/.done_grid" ]; then
  python scripts/ext13_p5_analyze.py $cond $proto >> "logs/ext13_p5_analyze_${cond}_${proto}.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) analyze $cond $proto rc=$? (fix chain)" >> $LOG
fi
