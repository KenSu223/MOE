#!/usr/bin/env bash
# Phase 8 smoke run (not a result): every stage-2 command on 8 rows of one paper task, into results/relp_moe_dev/p8_smoke/.
# Usage: bash scripts/relp_moe_p8_smoke.sh <qwen3|mixtral> <task>
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
model=$1; task=$2
D=$(.venv-relp/bin/python -c "import sys; sys.path.insert(0,'scripts'); import relp_moe_p8_data as D; print(D.MODELS['$model'])")
b=results/relp_sva_moe/$model/paper/$task
o=results/relp_moe_dev/p8_smoke/${model}_$task
c=(--model-dir "$D" --manifest-dir "data/relp_sva/${model}_paper_v1" --task "$task" --verification "$b/verify/verification.json" --phase7-dir "$b" --max-rows 8)
for cmd in probe rules path freeze neuron addback; do
  extra=(); [ "$cmd" = addback ] && extra=(--rw-variant relp_rw)
  [ "$cmd" = path ] && extra=(--chunk 4)
  t=$(date +%s)
  bash scripts/gpu_queue.sh "relp-dev-p8smoke-$model-$cmd" -- "$PY" scripts/relp_moe_p8_run.py "$cmd" "${c[@]}" "${extra[@]}" --output-dir "$o/p8_$cmd" \
    >> "logs/relp_moe_p8_smoke.log" 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) smoke $model $task $cmd rc=$rc ($(( $(date +%s) - t ))s)" | tee -a logs/relp_moe_p8_smoke.log
  [ $rc -ne 0 ] && exit $rc
done
