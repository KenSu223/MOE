#!/usr/bin/env bash
# Phase 8 follow-up: SVA add-back curves under five patch semantics (scripts/relp_moe_p8_mix.py), both models, 4 tasks.
# GPU jobs through gpu_queue.sh; complete steps are skipped.
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
PA=/home/ubuntu/MOE/.venv-analysis/bin/python
LOG=logs/relp_moe_p8_mix.log
declare -A DIRS=(
  [qwen3]=/opt/dlami/nvme/hf/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  [mixtral]=/opt/dlami/nvme/hf/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0)
log() { echo "$(date -u +%FT%TZ) [relp-p8/mix] $*" | tee -a "$LOG"; }
progress() { echo "$(date -u +%FT%TZ) [relp-p8] $*" >> logs/PROGRESS.md; }
complete() { [ -f "$1/run_meta.json" ] && grep -q '"status": "complete"' "$1/run_meta.json"; }
fail=0
log "chain mix start"
for model in qwen3 mixtral; do for task in nounpp simple rc within_rc; do
  b=results/relp_sva_moe/$model/paper/$task
  complete "$b/p8_mix" && { log "skip $model/$task (complete)"; continue; }
  log "start $model-$task-mix"
  bash scripts/gpu_queue.sh "relp-p8-$model-$task-mix" -- "$PY" scripts/relp_moe_p8_mix.py \
    --model-dir "${DIRS[$model]}" --manifest-dir "data/relp_sva/${model}_paper_v1" --task "$task" \
    --verification "$b/verify/verification.json" --phase7-dir "$b" --output-dir "$b/p8_mix" \
    >> "logs/relp_moe_p8_mix_jobs.log" 2>&1
  rc=$?
  log "end $model-$task-mix rc=$rc"
  if [ $rc -ne 0 ]; then fail=1; progress "FAIL $model/$task/mix rc=$rc (logs/relp_moe_p8_mix_jobs.log)"; fi
done; done
[ $fail -eq 0 ] && progress "chain mix DONE: SVA add-back under five patch semantics, 2 models x 4 tasks" || progress "chain mix ended with failures (see $LOG)"
log "chain mix end (fail=$fail)"
