#!/usr/bin/env bash
# Direction 4 GPU chain for one model/protocol: scan (filter + layer sweep, queued) -> select (CPU) -> expert pass (queued).
# Usage: bash scripts/ext4_chain.sh <model_key> <protocol raw|nobos|chat> <run_name> <scan_chunk> <layer_chunks> [per_category_max]
# Every GPU command goes through scripts/gpu_queue.sh (shared lock). Logs: logs/ext4_<run>_{scan,expert}.log, logs/ext4_chain.log
set -uo pipefail
cd "$(dirname "$0")/.."
model="$1"; proto="$2"; run="$3"; chunk="${4:-1024}"; lc="${5:-12}"; pcm="${6:-}"
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/hf}
. .venv/bin/activate
log() { echo "$(date -u +%FT%TZ) [ext4-chain $run] $*" | tee -a logs/ext4_chain.log; }
pcm_arg=""; [ -n "$pcm" ] && pcm_arg="--per-category-max $pcm"
if [ ! -f "results/$run/scan_meta.json" ] || ! python -c "import json,sys; m=json.load(open('results/$run/scan_meta.json')); sys.exit(0 if 'total_gpu_s' in m else 1)"; then
  log "scan start (chunk $chunk $pcm_arg)"
  bash scripts/gpu_queue.sh "ext4-scan-$run" -- python scripts/ext4_scan.py "$model" --out "$run" --protocol "$proto" --chunk "$chunk" $pcm_arg > "logs/ext4_${run}_scan.log" 2>&1
  rc=$?; log "scan rc=$rc"; [ $rc -ne 0 ] && exit $rc
else
  log "scan already complete, skipping"
fi
log "select"
python scripts/ext4_select.py "$run" --figure > "logs/ext4_${run}_select.log" 2>&1 || { log "select failed"; exit 1; }
tail -3 "logs/ext4_${run}_select.log" | tee -a logs/ext4_chain.log
python scripts/ext4_run_expert.py "$model" --out "$run" --protocol "$proto" --no-pairs --layer-chunks "$lc" --dry-run 2>&1 | tail -3 | tee -a logs/ext4_chain.log
log "expert pass start (layer chunks $lc)"
bash scripts/gpu_queue.sh "ext4-expert-$run" -- python scripts/ext4_run_expert.py "$model" --out "$run" --protocol "$proto" --no-pairs --layer-chunks "$lc" > "logs/ext4_${run}_expert.log" 2>&1
rc=$?; log "expert rc=$rc"; [ $rc -ne 0 ] && exit $rc
log "chain done"
