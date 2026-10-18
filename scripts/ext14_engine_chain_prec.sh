#!/usr/bin/env bash
# ext14 (p6-engine): precision effect on one Phase-6 cell (e1_a2e, Qwen pair file, 32 pairs = 64 directed units) for the
# Qwen2 twins: engine bf16 vs engine fp32 (= transformers fp32) on a T2-like plan -> attention share, rescue correlation,
# pass time and peak memory (two spawn counts with the 6-unit run of chain4 give the per-row memory for fp32 row budgets).
set -u
MAIN=/home/ubuntu/MOE
cd $MAIN
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
for m in qwen2_7b qwen2_57b; do
  bash scripts/gpu_queue.sh p6eng-prec-$m -- python scripts/ext14_engine_verify.py --model $m --units 32 --vs-fp32 \
    --pairs data/dial/pairs_e1_a2e_qwen3.parquet --tag _e1a2e > logs/ext14_engine_prec_$m.log 2>&1
  echo "$(ts) $m precision run rc=$?"
done
echo "$(ts) chain done"
