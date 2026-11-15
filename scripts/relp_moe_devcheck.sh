#!/usr/bin/env bash
# Development check of the streamed RelP engine on a real checkpoint (NOT a formal result): the full runner pipeline on
# the small smoke nounpp manifest (verify on 4 calibration pairs -> score 20 discovery pairs with the full budgets ->
# mean-score reuse -> one evaluate-multi pass of 8 validation rows x 2 baselines x 3 units), into results/relp_moe_dev/.
# Purpose: native bit-exactness of the real checkpoint and per-pass timings for the run estimate.
# Usage: bash scripts/relp_moe_devcheck.sh <qwen3|mixtral>
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
model=$1
declare -A DIRS=(
  [qwen3]=/opt/dlami/nvme/hf/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  [mixtral]=/opt/dlami/nvme/hf/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0)
out=results/relp_moe_dev/$model/smoke_nounpp
c=(--model-dir "${DIRS[$model]}" --manifest-dir "data/relp_sva/${model}_smoke_v1" --task nounpp)
step() { local name=$1; shift; bash scripts/gpu_queue.sh "relp-dev-$model-$name" -- "$PY" scripts/relp_sva_moe_run.py "$@" \
  >> "logs/relp_dev_${model}.log" 2>&1; local rc=$?; echo "$(date -u +%FT%TZ) $model $name rc=$rc" | tee -a "logs/relp_dev_${model}.log"; return $rc; }
step verify verify "${c[@]}" --output-dir "$out/verify" || exit 1
V=$out/verify/verification.json
step score score "${c[@]}" --verification "$V" --output-dir "$out/score_zero" || exit 1
step score_mean score "${c[@]}" --verification "$V" --baseline mean_q_routed --reuse-scores "$out/score_zero" \
  --output-dir "$out/score_mean" || exit 1
step validation evaluate-multi "${c[@]}" --verification "$V" --split validation --score-dirs "$out/score_zero" \
  "$out/score_mean" --units neuron expert_position expert_all_positions --output-dir "$out/validation" || exit 1
echo "$(date -u +%FT%TZ) $model DEVCHECK DONE" | tee -a "logs/relp_dev_${model}.log"
