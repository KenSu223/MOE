#!/usr/bin/env bash
# ext13 Phase 5 step 1a: full F1 pool scan (competence + association probe), Qwen3 then Mixtral BOS (GPU).
cd "$(dirname "$0")/.."
for p in qwen3 mixtral_bos; do
  bash scripts/gpu_queue.sh ext13-f1full-$p -- python scripts/ext13_scan.py $p --formats F1 --f1-sample 0 > logs/ext13_f1full_$p.log 2>&1
  echo "$(date -u +%FT%TZ) ext13-f1full-$p rc=$?" >> logs/ext13_chain.log
done
