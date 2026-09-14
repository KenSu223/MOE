#!/usr/bin/env bash
# ext2-model-zoo: attention/MoE/block sweep for each zoo model under its intended protocol, once that run's expert pass exists.
set -uo pipefail
cd /home/ubuntu/MOE; . .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
CH=logs/ext2_zoo_chain.log
clog() { echo "$(date -u +%FT%TZ) attn-chain: $*" >> "$CH"; }
for spec in olmoe:default olmoe_instruct:chat qwen3_instruct:chat qwen3_coder:chat mixtral_instruct:chat; do
  key=${spec%%:*}; p=${spec##*:}; run="${key}_${p}"
  [ -f "results/${run}_attnsweep/sweep_summary.json" ] && continue
  clog "waiting for results/$run/expert_rows.parquet"
  while [ ! -f "results/$run/expert_rows.parquet" ] || ! grep -q expert_done_utc "results/$run/run_meta.json" 2>/dev/null; do sleep 60; done
  clog "launching $run attn sweep"
  bash scripts/gpu_queue.sh "ext2zoo-$run-attn" -- python scripts/ext2_zoo_attn.py "$key" --protocol "$p" > "logs/ext2_zoo_${run}_attn.log" 2>&1
  clog "$run attn rc=$?"
done
clog "attn DONE"
