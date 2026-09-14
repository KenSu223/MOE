#!/usr/bin/env bash
# Direction 4, remaining GPU sequence (resumed 10:10 UTC): memory-bounded expert passes for the two finished scans, then the
# Qwen3-Coder runs (raw and chat protocols: scan -> select -> expert). Every GPU command goes through scripts/gpu_queue.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
log() { echo "$(date -u +%FT%TZ) [ext4-chain2] $*" | tee -a logs/ext4_chain.log; }
expert() {  # model proto run prefill_tokens wf_rows
  log "expert $3 start (prefill-tokens $4, wf-rows $5)"
  bash scripts/gpu_queue.sh "ext4-expert-$3" -- python scripts/ext4_run_expert.py "$1" --out "$3" --protocol "$2" --no-pairs --prefill-tokens "$4" --wf-rows "$5" > "logs/ext4_$3_expert.log" 2>&1
  rc=$?; log "expert $3 rc=$rc"; return $rc
}
scan_select() {  # model proto run
  if [ ! -f "results/$3/scan_meta.json" ] || ! python -c "import json,sys; sys.exit(0 if 'total_gpu_s' in json.load(open('results/$3/scan_meta.json')) else 1)"; then
    log "scan $3 start"
    bash scripts/gpu_queue.sh "ext4-scan-$3" -- python scripts/ext4_scan.py "$1" --out "$3" --protocol "$2" --chunk 1024 > "logs/ext4_$3_scan.log" 2>&1
    rc=$?; log "scan $3 rc=$rc"; [ $rc -ne 0 ] && return $rc
  else
    log "scan $3 already complete"
  fi
  python scripts/ext4_select.py "$3" --figure > "logs/ext4_$3_select.log" 2>&1 || { log "select $3 failed"; return 1; }
  tail -2 "logs/ext4_$3_select.log" | tee -a logs/ext4_chain.log
}
expert qwen3 raw codefact_qwen3_raw 160000 45000 || log "qwen3 expert failed, continuing"
expert mixtral nobos codefact_mixtral_nobos 110000 24000 || log "mixtral expert failed, continuing"
scan_select qwen3_coder raw codefact_qwen3_coder_raw && expert qwen3_coder raw codefact_qwen3_coder_raw 160000 45000 || log "coder raw failed"
scan_select qwen3_coder chat codefact_qwen3_coder_chat && expert qwen3_coder chat codefact_qwen3_coder_chat 160000 45000 || log "coder chat failed"
log "chain2 done"
