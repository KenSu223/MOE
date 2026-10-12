#!/usr/bin/env bash
# ext13 P5-3: competence scan of the verb-swap pairs (orig / neutral / unrelated), Qwen3 then Mixtral BOS (GPU).
cd "$(dirname "$0")/.."
for p in qwen3 mixtral_bos; do
  bash scripts/gpu_queue.sh ext13-verbs-$p -- python scripts/ext13_scan.py $p --verbs > logs/ext13_verbs_$p.log 2>&1
  echo "$(date -u +%FT%TZ) ext13-verbs-$p rc=$?" >> logs/ext13_chain.log
done
