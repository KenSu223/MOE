#!/usr/bin/env bash
# Phase 8, S2.5 (docs/relp_moe_phase8_plan.md): repeat patches for the noise floor of exact patching. The Phase-7 patch
# command on the first 40 discovery pairs of each paper task (every position from the changed word, every layer, every
# routed expert, both directions): the same (pair, unit) as in patch_full, in another batch composition.
# GPU jobs through gpu_queue.sh; complete steps skipped.
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
LOG=logs/relp_moe_p8_chain.log
declare -A DIRS=(
  [qwen3]=/opt/dlami/nvme/hf/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  [mixtral]=/opt/dlami/nvme/hf/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0)
log() { echo "$(date -u +%FT%TZ) [relp-p8/noise] $*" | tee -a "$LOG"; }
progress() { echo "$(date -u +%FT%TZ) [relp-p8] $*" >> logs/PROGRESS.md; }
complete() { [ -f "$1/run_meta.json" ] && grep -q '"status": "complete"' "$1/run_meta.json"; }
log "chain start"
fail=0
for model in qwen3 mixtral; do for task in nounpp simple rc within_rc; do
  b=results/relp_sva_moe/$model/paper/$task
  complete "$b/p8_noise" && { log "skip $model/$task (complete)"; continue; }
  [ -d "$b/p8_noise" ] && mv "$b/p8_noise" "$b/p8_noise.incomplete.$(date -u +%s)"
  log "start $model-$task-noise"
  bash scripts/gpu_queue.sh "relp-p8-$model-$task-noise" -- "$PY" scripts/relp_moe_compare_run.py patch \
    --model-dir "${DIRS[$model]}" --manifest-dir "data/relp_sva/${model}_paper_v1" --task "$task" \
    --verification "$b/verify/verification.json" --splits discovery --max-rows 40 --output-dir "$b/p8_noise" \
    >> "logs/relp_moe_p8_jobs_${model}.log" 2>&1
  rc=$?
  log "end $model-$task-noise rc=$rc"
  [ $rc -ne 0 ] && { fail=1; progress "FAIL S2.5 noise $model/$task rc=$rc (logs/relp_moe_p8_jobs_${model}.log)"; }
done; done
[ $fail -eq 0 ] && progress "S2.5 noise repeats DONE (8/8 paper tasks, 40 discovery pairs each)"
log "chain end (fail=$fail)"
