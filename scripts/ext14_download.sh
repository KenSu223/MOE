#!/usr/bin/env bash
# Phase 6 Stage 0: download the dense / twin checkpoints to the NVMe (same dl() as download_models.sh).
set -uo pipefail
export HF_HOME=/opt/dlami/nvme/hf
. /opt/dlami/nvme/hfenv/bin/activate
export HF_XET_HIGH_PERFORMANCE=1
dl() {
  local repo="$1"; local log="/home/ubuntu/MOE/logs/dl_$(echo "$repo" | tr '/' '_').log"
  local t0=$(date +%s); local rc=1
  for attempt in 1 2 3 4 5 6 7 8; do
    hf download "$repo" --include "*.safetensors" --include "*.json" --include "*.txt" --include "*.model" --include "tokenizer*" --include "merges*" --include "vocab*" >> "$log" 2>&1
    rc=$?; [ $rc -eq 0 ] && break
    echo "$(date +%T) $repo attempt $attempt failed rc=$rc, retrying in 20s" >> "$log"; sleep 20
  done
  echo "$(date -u +%FT%TZ) DONE $repo rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
}
# wave 1: small dense models (needed first for the engine verification and the upcycling check)
dl mistralai/Mistral-7B-v0.1 & dl Qwen/Qwen2-7B & dl Qwen/Qwen3-4B-Base &
wait
# wave 2
dl Qwen/Qwen2-57B-A14B & dl Qwen/Qwen3-8B-Base & dl Qwen/Qwen3-14B-Base & dl Qwen/Qwen3-1.7B-Base &
wait
echo "$(date -u +%FT%TZ) EXT14_DOWNLOADS_DONE"
du -sh "$HF_HOME"/hub/models--* 2>/dev/null
df -h /opt/dlami/nvme | tail -1
