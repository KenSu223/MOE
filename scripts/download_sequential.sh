#!/usr/bin/env bash
# Phase 0b (fixed): one download at a time. Three concurrent Xet downloads got OOM-killed.
set -uo pipefail
export HF_HOME=/opt/dlami/nvme/hf
. /opt/dlami/nvme/hfenv/bin/activate
# wait for any running download to finish first
while pgrep -f "hf download" >/dev/null; do sleep 15; done
dl() {
  local repo="$1" hp="$2"; local log="/home/ubuntu/MOE/logs/dl_$(echo "$repo" | tr '/' '_').log"
  local t0=$(date +%s)
  HF_XET_HIGH_PERFORMANCE=$hp hf download "$repo" --include "*.safetensors" --include "*.json" --include "*.txt" --include "*.model" --include "tokenizer*" --include "merges*" --include "vocab*" >> "$log" 2>&1
  local rc=$?; echo "DONE $repo hp=$hp rc=$rc elapsed=$(( $(date +%s) - t0 ))s"; return $rc
}
for repo in allenai/OLMoE-1B-7B-0125 Qwen/Qwen3-30B-A3B-Base mistralai/Mixtral-8x7B-v0.1; do
  dl "$repo" 1 || dl "$repo" 0 || dl "$repo" 0   # retry without high-performance mode if killed
done
echo ALL_DOWNLOADS_DONE; du -sh "$HF_HOME"/hub/models--*
