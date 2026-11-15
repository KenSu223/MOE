#!/usr/bin/env bash
# Phase 8 chain B (docs/relp_moe_phase8_plan.md section 8d): stage 3, CounterFact then two-hop capitals, Qwen3 then Mixtral.
# Per task and model: verify -> probe -> rules -> path -> patch -> noise -> freeze -> addback. Two-hop first runs the competence
# scan and builds its manifest (go / no-go rule of the plan). GPU jobs through gpu_queue.sh; complete steps are skipped.
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
LOG=logs/relp_moe_p8_chain.log
declare -A DIRS=(
  [qwen3]=/opt/dlami/nvme/hf/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  [mixtral]=/opt/dlami/nvme/hf/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0)
declare -A SUB=([qwen3]=3000 [mixtral]=1500)
log() { echo "$(date -u +%FT%TZ) [relp-p8/B] $*" | tee -a "$LOG"; }
progress() { echo "$(date -u +%FT%TZ) [relp-p8] $*" >> logs/PROGRESS.md; }
complete() { [ -f "$1/run_meta.json" ] && grep -q '"status": "complete"' "$1/run_meta.json"; }
fail=0
step() {   # step <model> <task> <command> [extra args...]
  local model=$1 task=$2 cmd=$3; shift 3
  local b=results/relp_p8/$model/$task
  complete "$b/$cmd" && { log "skip $model/$task/$cmd (complete)"; return 0; }
  log "start $model-$task-$cmd"
  bash scripts/gpu_queue.sh "relp-p8-$model-$task-$cmd" -- "$PY" scripts/relp_moe_p8_run3.py "$cmd" \
    --model-dir "${DIRS[$model]}" --manifest-dir "data/relp_p8/${model}_${task}_v1" --task "$task" --base-dir "$b" \
    --sub-rows "${SUB[$model]}" "$@" >> "logs/relp_moe_p8_jobs_${model}.log" 2>&1
  local rc=$?
  log "end $model-$task-$cmd rc=$rc"
  if [ $rc -ne 0 ]; then fail=1; progress "FAIL stage 3 $model/$task/$cmd rc=$rc (logs/relp_moe_p8_jobs_${model}.log)"; fi
  return $rc
}
run_task() {   # run_task <model> <task> <patch directions...>
  local model=$1 task=$2; shift 2
  step $model $task verify || return 1
  step $model $task probe
  step $model $task rules || return 1
  step $model $task path
  step $model $task patch --directions "$@" || return 1
  step $model $task noise
  step $model $task freeze
  step $model $task addback
  progress "stage 3 GPU done for $model/$task"
}
log "chain B start"
for model in qwen3 mixtral; do run_task $model counterfact denoise noise; done
for model in qwen3 mixtral; do
  b=results/relp_p8/$model/twohop
  if ! complete "$b/scan"; then
    log "start $model-twohop-scan"
    bash scripts/gpu_queue.sh "relp-p8-$model-twohop-scan" -- "$PY" scripts/relp_moe_p8_run3.py scan --model-dir "${DIRS[$model]}" \
      --task twohop --base-dir "$b" --candidates "data/relp_p8/twohop_candidates_${model}.json" >> "logs/relp_moe_p8_jobs_${model}.log" 2>&1
    log "end $model-twohop-scan rc=$?"
  fi
  if [ ! -f "data/relp_p8/${model}_twohop_v1/manifest.json" ]; then
    "$PY" scripts/relp_moe_p8_prepare.py twohop --model $model --scan "$b/scan/scan.parquet" > "logs/relp_moe_p8_twohop_${model}.log" 2>&1 \
      || { log "twohop manifest failed for $model (see logs/relp_moe_p8_twohop_${model}.log)"; progress "STOP two-hop $model: manifest not built (logs/relp_moe_p8_twohop_${model}.log)"; fail=1; continue; }
  fi
  run_task $model twohop denoise
done
[ $fail -eq 0 ] && progress "chain B DONE: stage 3 complete (counterfact, twohop; 2 models)" || progress "chain B ended with failures (see $LOG)"
log "chain B end (fail=$fail)"
