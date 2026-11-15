#!/usr/bin/env bash
# RelP-MoE SVA (2026-11-07): download the three checkpoints at PINNED revisions to the NVMe (ephemeral).
# OLMoE = the revision pinned in data/relp_sva/sources.lock.json (small-model reference for the streamed implementation);
# Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 = their main revisions on 2026-11-07 (recorded in the RelP-MoE manifests).
# Same retry pattern as scripts/download_models.sh. Launch detached:  setsid nohup bash scripts/relp_moe_download.sh &
set -uo pipefail
export HF_HOME=/opt/dlami/nvme/hf
mkdir -p "$HF_HOME" /home/ubuntu/MOE/logs
[ -d /opt/dlami/nvme/hfenv ] || python3.14 -m venv /opt/dlami/nvme/hfenv
. /opt/dlami/nvme/hfenv/bin/activate
pip install -q -U pip "huggingface_hub[hf_xet]"
export HF_XET_HIGH_PERFORMANCE=1
dl() {
  local repo="$1" rev="$2"; local log="/home/ubuntu/MOE/logs/dl_$(echo "$repo" | tr '/' '_').log"
  local t0=$(date +%s); local rc=1
  for attempt in 1 2 3 4 5 6 7 8; do
    hf download "$repo" --revision "$rev" --include "*.safetensors" --include "*.json" --include "*.txt" \
      --include "*.model" --include "tokenizer*" --include "merges*" --include "vocab*" >> "$log" 2>&1
    rc=$?; [ $rc -eq 0 ] && break
    echo "$(date +%T) $repo attempt $attempt failed rc=$rc, retrying in 20s" >> "$log"; sleep 20
  done
  echo "$(date -u +%FT%TZ) DONE $repo@$rev rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
}
dl allenai/OLMoE-1B-7B-0125 9b0c1aa87e34a20052389dce1f0cf01da783f654 &
dl Qwen/Qwen3-30B-A3B-Base 1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 &
dl mistralai/Mixtral-8x7B-v0.1 fc7ac94680e38d7348cfa806e51218e6273104b0 &
wait
echo "$(date -u +%FT%TZ) ALL_DOWNLOADS_DONE"
du -sh "$HF_HOME"/hub/models--* 2>/dev/null
