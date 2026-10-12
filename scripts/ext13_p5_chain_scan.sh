#!/usr/bin/env bash
# ext13 Phase 5 (p5-data): probe v2 + condition scans (prefill only), both models, through the GPU queue.
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400
q() { name=$1; shift; bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1; echo "$(date -u +%FT%TZ) $name rc=$?" >> logs/ext13_p5_chain.log; }
for p in qwen3 mixtral_bos; do
  q ext13-p5-probes-$p python scripts/ext13_p5_scan.py $p --part probes
  q ext13-p5-conds-$p  python scripts/ext13_p5_scan.py $p --part conds
done
