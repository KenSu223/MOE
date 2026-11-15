#!/usr/bin/env bash
# RelP -> MoE on SVA (docs/relp_sva_experiment_protocol.md), Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1, layer-streamed engine
# (scripts/relp_sva_moe_run.py). Order: smoke (both models) -> paper (both) -> grouped (both); per task:
#   verify (native acceptance gate) -> score_zero (paired RelP, clean endpoint A, pool non_special, zero baseline)
#   -> score_mean (mean_q_routed; reuses the baseline-independent discovery scores, adds the calibration mean)
#   -> validation (neuron / expert_position / expert_all_positions x zero / mean, one streamed pass)
# and after all tasks of a profile: test (same, held-out test split). Smoke = protocol section 7 (small budgets, no test).
# Every GPU job goes through gpu_queue.sh. Complete steps (run_meta status complete) are skipped, incomplete ones moved
# aside, so the chain can be re-launched after any interruption. A failed verify blocks that task (logged, no workaround).
# Launch detached:  setsid nohup bash scripts/relp_moe_chain.sh > logs/relp_moe_chain.out 2>&1 < /dev/null &
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
MODELS=${RELP_MODELS:-"qwen3 mixtral"}
PROFILES=${RELP_PROFILES:-"smoke paper grouped"}
LOG=logs/relp_moe_chain.log
declare -A DIRS=(
  [qwen3]=/opt/dlami/nvme/hf/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  [mixtral]=/opt/dlami/nvme/hf/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0)

log() { echo "$(date -u +%FT%TZ) [relp-moe] $*" | tee -a "$LOG"; }
progress() { echo "$(date -u +%FT%TZ) [relp-moe] $*" >> logs/PROGRESS.md; }
complete() { [ -f "$1/run_meta.json" ] && grep -q '"status": "complete"' "$1/run_meta.json"; }

run() {  # run <job-name> <output-dir> <runner args...>
  local name=$1 dir=$2; shift 2
  if complete "$dir"; then log "skip $name (complete)"; return 0; fi
  if [ -e "$dir" ]; then mv "$dir" "${dir}_incomplete_$(date +%s)"; log "moved incomplete $dir aside"; fi
  mkdir -p "$(dirname "$dir")"
  log "start $name"
  bash scripts/gpu_queue.sh "relp-$name" -- "$PY" scripts/relp_sva_moe_run.py "$@" --output-dir "$dir" \
    >> "logs/relp_moe_jobs_${name%%-*}.log" 2>&1
  local rc=$?
  log "end $name rc=$rc"
  return $rc
}

tasks_of() { [ "$1" = smoke ] && echo "simple nounpp" || echo "simple rc within_rc nounpp"; }

task_flow() {  # task_flow <model> <profile> <task>
  local model=$1 profile=$2 task=$3
  local base=results/relp_sva_moe/$model/$profile/$task
  local c=(--model-dir "${DIRS[$model]}" --manifest-dir "data/relp_sva/${model}_${profile}_v1" --task "$task")
  local tag="$model-$profile-$task"
  if ! run "$tag-verify" "$base/verify" verify "${c[@]}"; then
    log "VERIFY FAILED $tag: downstream steps of this task skipped"
    progress "VERIFY FAILED $model/$profile/$task (see $base/verify*/verification.json); task blocked"
    return 1
  fi
  local V=$base/verify/verification.json
  if [ "$profile" = smoke ]; then
    run "$tag-score" "$base/score_zero" score "${c[@]}" --verification "$V" --mode paired --pool non_special \
      --baseline zero --neuron-budgets 1 5 --expert-budgets 1 2 --random-seeds 101 || return 1
    run "$tag-validation" "$base/validation" evaluate-multi "${c[@]}" --verification "$V" --split validation \
      --score-dirs "$base/score_zero" --units neuron expert_position || return 1
    return 0
  fi
  run "$tag-score_zero" "$base/score_zero" score "${c[@]}" --verification "$V" --mode paired --pool non_special \
    --baseline zero || return 1
  run "$tag-score_mean" "$base/score_mean" score "${c[@]}" --verification "$V" --mode paired --pool non_special \
    --baseline mean_q_routed --reuse-scores "$base/score_zero" || return 1
  run "$tag-validation" "$base/validation" evaluate-multi "${c[@]}" --verification "$V" --split validation \
    --score-dirs "$base/score_zero" "$base/score_mean" --units neuron expert_position expert_all_positions || return 1
}

test_flow() {  # test_flow <model> <profile> <task>  (plan locked at score time; validation complete)
  local model=$1 profile=$2 task=$3
  local base=results/relp_sva_moe/$model/$profile/$task
  complete "$base/validation" || { log "skip test $model/$profile/$task (validation not complete)"; return 1; }
  local c=(--model-dir "${DIRS[$model]}" --manifest-dir "data/relp_sva/${model}_${profile}_v1" --task "$task")
  run "$model-$profile-$task-test" "$base/test" evaluate-multi "${c[@]}" --verification "$base/verify/verification.json" \
    --split test --score-dirs "$base/score_zero" "$base/score_mean" --units neuron expert_position expert_all_positions
}

log "chain start: models=[$MODELS] profiles=[$PROFILES]"
progress "chain start (scripts/relp_moe_chain.sh): models=[$MODELS] profiles=[$PROFILES]"
for profile in $PROFILES; do
  for model in $MODELS; do
    for task in $(tasks_of "$profile"); do task_flow "$model" "$profile" "$task"; done
    if [ "$profile" != smoke ]; then
      for task in $(tasks_of "$profile"); do test_flow "$model" "$profile" "$task"; done
    fi
    progress "PROFILE DONE $model/$profile"
  done
done
log "chain end"
progress "chain end"
