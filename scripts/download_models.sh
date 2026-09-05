#!/usr/bin/env bash
# Phase 0b: download checkpoints to the NVMe instance store (ephemeral, 521 GB free, 1.6 GB/s).
set -uo pipefail
export HF_HOME=/opt/dlami/nvme/hf
mkdir -p "$HF_HOME" /home/ubuntu/MOE/logs
[ -d /opt/dlami/nvme/hfenv ] || python3.14 -m venv /opt/dlami/nvme/hfenv
. /opt/dlami/nvme/hfenv/bin/activate
pip install -q -U pip "huggingface_hub[hf_xet]"
export HF_XET_HIGH_PERFORMANCE=1
# reuse an existing HF token if the user has one (faster, higher rate limits); never printed
[ -f "$HOME/.cache/huggingface/token" ] && [ ! -f "$HF_HOME/token" ] && cp "$HOME/.cache/huggingface/token" "$HF_HOME/token" || true
dl() {
  local repo="$1"; local log="/home/ubuntu/MOE/logs/dl_$(echo "$repo" | tr '/' '_').log"
  local t0=$(date +%s); local rc=1
  for attempt in 1 2 3 4 5 6 7 8; do
    hf download "$repo" --include "*.safetensors" --include "*.json" --include "*.txt" --include "*.model" --include "tokenizer*" --include "merges*" --include "vocab*" >> "$log" 2>&1
    rc=$?; [ $rc -eq 0 ] && break
    echo "$(date +%T) $repo attempt $attempt failed rc=$rc, retrying in 20s" >> "$log"; sleep 20
  done
  echo "$(date +%T) DONE $repo rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
}
dl allenai/OLMoE-1B-7B-0125 &
dl Qwen/Qwen3-30B-A3B-Base &
dl mistralai/Mixtral-8x7B-v0.1 &
wait
echo ALL_DOWNLOADS_DONE
du -sh "$HF_HOME"/hub/models--* 2>/dev/null
