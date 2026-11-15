#!/usr/bin/env bash
# Phase 8 stage 3 smoke run (not a result): every run3 command on a few rows, into results/relp_moe_dev/p8_smoke3/.
# Usage: bash scripts/relp_moe_p8_smoke3.sh <qwen3|mixtral> <task> [rows]
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
model=$1; task=$2; n=${3:-6}
D=$(.venv-relp/bin/python -c "import sys; sys.path.insert(0,'scripts'); import relp_moe_p8_data as D; print(D.MODELS['$model'])")
o=results/relp_moe_dev/p8_smoke3/${model}_$task
c=(--model-dir "$D" --manifest-dir "data/relp_p8/${model}_${task}_v1" --task "$task" --base-dir "$o" --max-rows "$n")
for cmd in verify probe rules path patch noise freeze addback; do
  t=$(date +%s)
  bash scripts/gpu_queue.sh "relp-dev-p8smoke3-$model-$cmd" -- "$PY" scripts/relp_moe_p8_run3.py "$cmd" "${c[@]}" >> logs/relp_moe_p8_smoke.log 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) smoke3 $model $task $cmd rc=$rc ($(( $(date +%s) - t ))s)" | tee -a logs/relp_moe_p8_smoke.log
  [ $rc -ne 0 ] && exit $rc
done
