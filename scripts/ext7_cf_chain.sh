#!/usr/bin/env bash
# ext7-controls task 1: CounterFact STR attention / MoE / block sweeps (Qwen3, Mixtral BOS). Each job via gpu_queue.sh.
cd /home/ubuntu/MOE
set -u
bash scripts/gpu_queue.sh ext7c-cf-attnsweep-qwen3 -- python scripts/ext7_cf_attnsweep.py qwen3 --str-run qwen3_str --out qwen3_str_attnsweep --layer-chunks 4 > logs/ext7c_cf_attnsweep_qwen3.log 2>&1
echo "$(date -u +%FT%TZ) qwen3 rc=$?"
bash scripts/gpu_queue.sh ext7c-cf-attnsweep-mixtral_bos -- python scripts/ext7_cf_attnsweep.py mixtral --str-run mixtral_bos_str --out mixtral_bos_str_attnsweep --layer-chunks 3 > logs/ext7c_cf_attnsweep_mixtral_bos.log 2>&1
echo "$(date -u +%FT%TZ) mixtral_bos rc=$?"
