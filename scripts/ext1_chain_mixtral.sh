#!/usr/bin/env bash
# Direction 1 (ext1-joint-search): all-layer expert pass for the two Mixtral runs (4 layer chunks each to bound GPU memory).
set -uo pipefail
cd /home/ubuntu/MOE
run() {  # run <queue-name> <log> <args...>
  local name=$1 logf=$2; shift 2
  echo "$(date -u +%FT%TZ) ext1_chain_mixtral: launching $name" >> logs/ext1_chain.log
  bash scripts/gpu_queue.sh "$name" -- python scripts/run_expert.py "$@" > "$logf" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) ext1_chain_mixtral: $name rc=$rc" >> logs/ext1_chain.log
  [ $rc -eq 0 ] || { echo "$(date -u +%FT%TZ) ext1_chain_mixtral: ABORT after $name" >> logs/ext1_chain.log; exit $rc; }
}
run ext1-mixtral-bos-alllayers   logs/ext1_expert_mixtral_bos_alllayers.log   mixtral --layers "$(seq -s, 0 31)" --no-pairs --layer-chunks 4 --out mixtral_bos_alllayers
run ext1-mixtral-nobos-alllayers logs/ext1_expert_mixtral_nobos_alllayers.log mixtral --layers "$(seq -s, 0 31)" --no-pairs --layer-chunks 4 --no-special-tokens --out mixtral_nobos_alllayers
echo "$(date -u +%FT%TZ) ext1_chain_mixtral: DONE" >> logs/ext1_chain.log
