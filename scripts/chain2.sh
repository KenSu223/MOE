#!/usr/bin/env bash
# Waits for chain1, then: paper_like token-rule runs (paper case set) for both models, then HF reference checks.
set -uo pipefail
cd /home/ubuntu/MOE; . .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
while ! grep -q "chain1 done" logs/chain1.log; do sleep 20; done
echo "$(date -u +%T) chain2 start"
for m in qwen3 mixtral; do
  mkdir -p results/${m}_alt; cp results/$m/case_sets.json results/${m}_alt/case_sets.json
  python scripts/run_sweep.py $m --sets paper --out ${m}_alt --token-rule paper_like > logs/sweep_${m}_alt.log 2>&1; echo "$(date -u +%T) $m alt sweep rc=$?"
  bash scripts/auto_expert.sh $m ${m}_alt paper_like > logs/expert_${m}_alt.log 2>&1; echo "$(date -u +%T) $m alt expert rc=$?"
done
for m in qwen3 mixtral; do
  timeout 3600 python scripts/hf_reference_check.py $m 5 > logs/hf_ref_$m.log 2>&1; echo "$(date -u +%T) hf ref $m rc=$?"
done
echo "$(date -u +%T) chain2 done"
