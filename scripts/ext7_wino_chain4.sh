#!/usr/bin/env bash
# ext7-wino chain 4: W4 direct-path split with ext7-controls' shared implementation (moetrace/ext7_controls.py direct_split_pairs,
# read-only use, = ext8 direct_split), main + replication families, Qwen3 and Mixtral BOS -> results/wino_<m>_str/direct_split.parquet
set -uo pipefail
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date -u +%FT%TZ; }
step() { local name=$1; shift; echo "$(ts) [chain4] START $name"; $Q "$name" -- "$@"; local rc=$?; echo "$(ts) [chain4] END $name rc=$rc"; return $rc; }
step ext7-wino-direct-qwen3 python -m moetrace.ext7_controls direct --model qwen3 --pairs data/wino_str/pairs_train_xl_qwen3.parquet \
  --case-sets data/wino_str/case_sets.json --out wino_qwen3_str --splits discovery,validation,rep_discovery,rep_validation || exit 1
step ext7-wino-direct-mixtral_bos python -m moetrace.ext7_controls direct --model mixtral --pairs data/wino_str/pairs_train_xl_mixtral_bos.parquet \
  --case-sets data/wino_str/case_sets.json --out wino_mixtral_bos_str --splits discovery,validation,rep_discovery,rep_validation || exit 1
echo "$(ts) [chain4] DONE"
