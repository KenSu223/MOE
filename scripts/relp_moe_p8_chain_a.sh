#!/usr/bin/env bash
# Phase 8 chain A (docs/relp_moe_phase8_plan.md section 7): stage 2 on the SVA paper profile, both models.
# Per model and task: probe -> rules -> path (simple, nounpp) -> freeze -> neuron; then the router-aware variant is selected
# on discovery (scripts/relp_moe_p8_select.py, CPU) and addback runs for every task. GPU jobs through gpu_queue.sh;
# complete steps are skipped, incomplete ones resume (part files) or restart.
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
PA=/home/ubuntu/MOE/.venv-analysis/bin/python
LOG=logs/relp_moe_p8_chain.log
declare -A DIRS=(
  [qwen3]=/opt/dlami/nvme/hf/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  [mixtral]=/opt/dlami/nvme/hf/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0)
declare -A SUB=([qwen3]=3000 [mixtral]=1500)
log() { echo "$(date -u +%FT%TZ) [relp-p8/A] $*" | tee -a "$LOG"; }
progress() { echo "$(date -u +%FT%TZ) [relp-p8] $*" >> logs/PROGRESS.md; }
complete() { [ -f "$1/run_meta.json" ] && grep -q '"status": "complete"' "$1/run_meta.json"; }
fail=0
step() {   # step <model> <task> <command> [extra args...]
  local model=$1 task=$2 cmd=$3; shift 3
  local b=results/relp_sva_moe/$model/paper/$task
  complete "$b/p8_$cmd" && { log "skip $model/$task/$cmd (complete)"; return 0; }
  log "start $model-$task-$cmd"
  bash scripts/gpu_queue.sh "relp-p8-$model-$task-$cmd" -- "$PY" scripts/relp_moe_p8_run.py "$cmd" \
    --model-dir "${DIRS[$model]}" --manifest-dir "data/relp_sva/${model}_paper_v1" --task "$task" \
    --verification "$b/verify/verification.json" --phase7-dir "$b" --output-dir "$b/p8_$cmd" --sub-rows "${SUB[$model]}" "$@" \
    >> "logs/relp_moe_p8_jobs_${model}.log" 2>&1
  local rc=$?
  log "end $model-$task-$cmd rc=$rc"
  if [ $rc -ne 0 ]; then fail=1; progress "FAIL $model/$task/$cmd rc=$rc (logs/relp_moe_p8_jobs_${model}.log)"; fi
  return $rc
}
log "chain A start"
for model in qwen3 mixtral; do for task in nounpp simple rc within_rc; do
  step $model $task probe || continue
  step $model $task rules
  case $task in simple|nounpp) step $model $task path ;; esac
  step $model $task freeze
  step $model $task neuron
  progress "stage 2 GPU (probe/rules/path/freeze/neuron) done for $model/$task"
done; done
"$PA" scripts/relp_moe_p8_select.py >> logs/relp_moe_p8_analyze.log 2>&1 || { log "selection failed rc=$?"; progress "FAIL chain A: router-aware selection (logs/relp_moe_p8_analyze.log)"; exit 1; }
for model in qwen3 mixtral; do
  rw=$("$PA" -c "import json; print(json.load(open('results/relp_sva_moe/p8/selected_rw.json'))['$model']['variant'])")
  for task in nounpp simple rc within_rc; do step $model $task addback --rw-variant "$rw"; done
done
[ $fail -eq 0 ] && progress "chain A DONE: stage 2 complete for 2 models x 4 tasks" || progress "chain A ended with failures (see $LOG)"
log "chain A end (fail=$fail)"
