#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/MOE; . .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
echo "$(date -u +%T) chain1 start"
python scripts/funnel_hypotheses2.py qwen3 390 > logs/funnel2_qwen3.log 2>&1; echo "$(date -u +%T) funnel2 rc=$?"
python scripts/run_noise.py qwen3 --layer 44 --expert 69 --set paper > logs/noise_qwen3.log 2>&1; echo "$(date -u +%T) noise rc=$?"
python scripts/run_filter.py mixtral > logs/filter_mixtral.log 2>&1; echo "$(date -u +%T) mixtral filter rc=$?"
python scripts/run_sweep.py mixtral > logs/sweep_mixtral.log 2>&1; echo "$(date -u +%T) mixtral sweep rc=$?"
bash scripts/auto_expert.sh mixtral > logs/expert_mixtral.log 2>&1; echo "$(date -u +%T) mixtral expert rc=$?"
echo "$(date -u +%T) chain1 done"
