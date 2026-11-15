#!/usr/bin/env bash
# Phase 7 chain 3 (docs/relp_moe_plan.md S4f), after chain 2 (scripts/relp_moe_patch_chain.sh) has ended:
# analysis (writes diagnose_units.json per task: largest RelP-vs-exact gaps, both methods' selected experts) -> diagnose
# (natural / frozen downstream routing / frozen attention / both + direct effect, discovery rows, both directions) per
# task -> analysis + figures again. GPU jobs through gpu_queue.sh; complete steps skipped.
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
PA=/home/ubuntu/MOE/.venv-analysis/bin/python
LOG=logs/relp_moe_chain.log
declare -A DIRS=(
  [qwen3]=/opt/dlami/nvme/hf/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  [mixtral]=/opt/dlami/nvme/hf/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0)
log() { echo "$(date -u +%FT%TZ) [relp-moe/chain3] $*" | tee -a "$LOG"; }
progress() { echo "$(date -u +%FT%TZ) [relp-moe] $*" >> logs/PROGRESS.md; }
complete() { [ -f "$1/run_meta.json" ] && grep -q '"status": "complete"' "$1/run_meta.json"; }

while pgrep -f "scripts/relp_moe_patch_chain.sh" > /dev/null; do sleep 60; done
log "chain3 start (chain2 no longer running)"
"$PA" scripts/relp_moe_analyze.py >> logs/relp_moe_analyze.log 2>&1 || { log "analysis failed rc=$?"; progress "FAIL chain3: analysis before diagnostics failed (logs/relp_moe_analyze.log)"; exit 1; }
for profile in paper grouped; do for model in qwen3 mixtral; do for task in simple rc within_rc nounpp; do
  b=results/relp_sva_moe/$model/$profile/$task
  [ -f "$b/diagnose_units.json" ] || { log "skip diagnose $model/$profile/$task (no unit list)"; continue; }
  complete "$b/diagnose" && { log "skip diagnose $model/$profile/$task (complete)"; continue; }
  log "start $model-$profile-$task-diagnose"
  bash scripts/gpu_queue.sh "relp-$model-$profile-$task-diagnose" -- "$PY" scripts/relp_moe_compare_run.py diagnose \
    --model-dir "${DIRS[$model]}" --manifest-dir "data/relp_sva/${model}_${profile}_v1" --task "$task" \
    --verification "$b/verify/verification.json" --units-json "$b/diagnose_units.json" --output-dir "$b/diagnose" \
    >> "logs/relp_moe_jobs_${model}.log" 2>&1
  log "end $model-$profile-$task-diagnose rc=$?"
done; done; done
progress "S4f diagnostics DONE"
"$PA" scripts/relp_moe_analyze.py >> logs/relp_moe_analyze.log 2>&1 && "$PA" scripts/relp_moe_figures.py >> logs/relp_moe_analyze.log 2>&1
log "chain3 end (analysis rc=$?)"
progress "chain3 end: diagnostics, analysis and figures refreshed; ready for the final report"
