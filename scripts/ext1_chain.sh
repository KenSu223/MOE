#!/usr/bin/env bash
# Direction 1 (ext1-joint-search): expert pass at EVERY layer for the three base runs, serialised through the GPU queue.
# Stops at the first failing job. Logs: logs/ext1_expert_<run>.log, queue log: logs/gpu_queue.log.
set -uo pipefail
cd /home/ubuntu/MOE
run() {  # run <queue-name> <log> <args...>
  local name=$1 logf=$2; shift 2
  echo "$(date -u +%FT%TZ) ext1_chain: launching $name" >> logs/ext1_chain.log
  bash scripts/gpu_queue.sh "$name" -- python scripts/run_expert.py "$@" > "$logf" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) ext1_chain: $name rc=$rc" >> logs/ext1_chain.log
  [ $rc -eq 0 ] || { echo "$(date -u +%FT%TZ) ext1_chain: ABORT after $name" >> logs/ext1_chain.log; exit $rc; }
}
run ext1-qwen3-alllayers        logs/ext1_expert_qwen3_bos_alllayers.log    qwen3   --layers "$(seq -s, 0 47)" --no-pairs --layer-chunks 4 --out qwen3_bos_alllayers
run ext1-mixtral-bos-alllayers  logs/ext1_expert_mixtral_bos_alllayers.log  mixtral --layers "$(seq -s, 0 31)" --no-pairs --layer-chunks 2 --out mixtral_bos_alllayers
run ext1-mixtral-nobos-alllayers logs/ext1_expert_mixtral_nobos_alllayers.log mixtral --layers "$(seq -s, 0 31)" --no-pairs --layer-chunks 2 --no-special-tokens --out mixtral_nobos_alllayers
echo "$(date -u +%FT%TZ) ext1_chain: DONE" >> logs/ext1_chain.log
