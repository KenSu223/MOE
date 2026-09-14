#!/usr/bin/env bash
# ext2-model-zoo: download the four additional checkpoints sequentially to the NVMe HF cache, checking free space
# before each one (never let free space drop below 15 GB). Log: logs/ext2_zoo_download.log
set -uo pipefail
export HF_HOME=/opt/dlami/nvme/hf
. /opt/dlami/nvme/hfenv/bin/activate
LOG=/home/ubuntu/MOE/logs/ext2_zoo_download.log
MARGIN_GB=15
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
free_gb() { df -BG --output=avail /opt/dlami/nvme | tail -1 | tr -dc '0-9'; }
dl() {  # dl <repo> <size_gb>
  local repo="$1" need="$2"
  local have; have=$(free_gb)
  if [ "$have" -lt $(( need + MARGIN_GB )) ]; then
    log "STOP: $repo needs ${need} GB + ${MARGIN_GB} GB margin but only ${have} GB free"; return 75
  fi
  log "START $repo (need ${need} GB, free ${have} GB)"
  local t0=$(date +%s) rc=1
  for hp in 1 0 0; do
    HF_XET_HIGH_PERFORMANCE=$hp hf download "$repo" --include "*.safetensors" --include "*.json" --include "*.txt" \
      --include "*.model" --include "tokenizer*" --include "merges*" --include "vocab*" --include "*.jinja" >> "$LOG.$(echo "$repo" | tr '/' '_')" 2>&1
    rc=$?; [ $rc -eq 0 ] && break
    log "$repo attempt (hp=$hp) failed rc=$rc, retrying in 20s"; sleep 20
  done
  log "DONE $repo rc=$rc elapsed=$(( $(date +%s) - t0 ))s free_after=$(free_gb) GB"
  return $rc
}
while pgrep -x hf >/dev/null; do sleep 15; done
dl allenai/OLMoE-1B-7B-0125-Instruct 14 && touch "$LOG.olmoe_instruct.done"
dl Qwen/Qwen3-30B-A3B-Instruct-2507 62 && touch "$LOG.qwen3_instruct.done"
dl Qwen/Qwen3-Coder-30B-A3B-Instruct 62 && touch "$LOG.qwen3_coder.done"
dl mistralai/Mixtral-8x7B-Instruct-v0.1 94 && touch "$LOG.mixtral_instruct.done"
log "ALL_DOWNLOADS_DONE"; du -sh "$HF_HOME"/hub/models--* | tee -a "$LOG"
