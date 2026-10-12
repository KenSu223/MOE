#!/usr/bin/env bash
# ext13 Phase 5 step 2: competence scan + association probe, Qwen3 then Mixtral BOS (GPU, via gpu_queue.sh).
cd "$(dirname "$0")/.."
for p in qwen3 mixtral_bos; do
  bash scripts/gpu_queue.sh ext13-scan-$p -- python scripts/ext13_scan.py $p > logs/ext13_scan_$p.log 2>&1
  echo "$(date -u +%FT%TZ) ext13-scan-$p rc=$?" >> logs/ext13_chain.log
done
