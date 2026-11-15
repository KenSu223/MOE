#!/usr/bin/env bash
# Phase 7 chain 2 (docs/relp_moe_plan.md S2-S4), started after the RelP chain (scripts/relp_moe_chain.sh) has ended.
#   S3 endpoint B (paper + grouped) -> S2 stability re-score (paper, --score-chunk 100) -> S4c/e grouped: layer x position
#   grid (all positions from the changed word) + Lu original scope (final position, experts of every layer) ->
#   S4d paper: full grid (layer x position and every expert routed at A or B, both directions; discovery+validation+test).
# Budget check: after the first S4d task, the S4d total is projected from its measured time; if it exceeds 6 h (the
# estimate given to the user), the chain stops with a "[relp-moe] REVIEW" PROGRESS line instead of overrunning.
# Every GPU job goes through gpu_queue.sh; complete steps are skipped (re-launch resumes; patch runs resume per pass).
set -uo pipefail
cd /home/ubuntu/MOE
export HF_HOME=/opt/dlami/nvme/hf
PY=/home/ubuntu/MOE/.venv-relp/bin/python
LOG=logs/relp_moe_chain.log
declare -A DIRS=(
  [qwen3]=/opt/dlami/nvme/hf/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  [mixtral]=/opt/dlami/nvme/hf/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0)
TASKS="simple rc within_rc nounpp"
log() { echo "$(date -u +%FT%TZ) [relp-moe/chain2] $*" | tee -a "$LOG"; }
progress() { echo "$(date -u +%FT%TZ) [relp-moe] $*" >> logs/PROGRESS.md; }
complete() { [ -f "$1/run_meta.json" ] && grep -q '"status": "complete"' "$1/run_meta.json"; }

while pgrep -f "scripts/relp_moe_chain.sh" > /dev/null; do sleep 60; done
log "chain2 start (RelP chain no longer running)"
progress "chain2 start: S3 endpoint B -> S2 stability -> S4 patching grids (scripts/relp_moe_patch_chain.sh)"

job() {  # job <name> <output-dir> <script> <args...>
  local name=$1 dir=$2 script=$3; shift 3
  if complete "$dir"; then log "skip $name (complete)"; return 0; fi
  log "start $name"
  local t0=$(date +%s)
  bash scripts/gpu_queue.sh "relp-$name" -- "$PY" "$script" "$@" >> "logs/relp_moe_jobs_${name%%-*}.log" 2>&1
  local rc=$?
  log "end $name rc=$rc ($(( $(date +%s) - t0 ))s)"
  return $rc
}

base_of() { echo "results/relp_sva_moe/$1/$2/$3"; }
cargs() { echo "--model-dir ${DIRS[$1]} --manifest-dir data/relp_sva/$1_$2_v1 --task $3"; }
verified() { complete "$(base_of "$1" "$2" "$3")/verify"; }

# S3 endpoint B
for profile in paper grouped; do for model in qwen3 mixtral; do for task in $TASKS; do
  b=$(base_of $model $profile $task); verified $model $profile $task || { log "skip S3 $model/$profile/$task (no passing verify)"; continue; }
  job "$model-$profile-$task-endpointB" "$b/score_endpointB" scripts/relp_moe_compare_run.py endpointB $(cargs $model $profile $task) \
    --verification "$b/verify/verification.json" --output-dir "$b/score_endpointB"
done; done; done
progress "S3 endpoint B DONE (paper + grouped)"
# S2 stability: same score, another batch composition (3 chunks of 100 pairs)
for model in qwen3 mixtral; do for task in $TASKS; do
  b=$(base_of $model paper $task); verified $model paper $task || continue
  [ -e "$b/score_zero_chunk100" ] && ! complete "$b/score_zero_chunk100" && mv "$b/score_zero_chunk100" "$b/score_zero_chunk100_incomplete_$(date +%s)"
  job "$model-paper-$task-stability" "$b/score_zero_chunk100" scripts/relp_sva_moe_run.py score $(cargs $model paper $task) \
    --verification "$b/verify/verification.json" --mode paired --pool non_special --baseline zero --score-chunk 100 \
    --output-dir "$b/score_zero_chunk100"
done; done
progress "S2 stability re-score DONE (paper)"
# S4c/e grouped: layer x position grid; Lu original scope (final position, all layers' experts)
for model in qwen3 mixtral; do for task in $TASKS; do
  b=$(base_of $model grouped $task); verified $model grouped $task || continue
  job "$model-grouped-$task-patch_layerpos" "$b/patch_layerpos" scripts/relp_moe_compare_run.py patch $(cargs $model grouped $task) \
    --verification "$b/verify/verification.json" --no-experts --output-dir "$b/patch_layerpos"
  job "$model-grouped-$task-patch_final" "$b/patch_final" scripts/relp_moe_compare_run.py patch $(cargs $model grouped $task) \
    --verification "$b/verify/verification.json" --positions final --output-dir "$b/patch_final"
done; done
progress "S4c/S4e grouped patching DONE"
# S4d paper full grid (contains the paper layer x position grid), budget check after the first task
first=1; t_first=0
for model in qwen3 mixtral; do for task in $TASKS; do
  b=$(base_of $model paper $task); verified $model paper $task || continue
  t0=$(date +%s)
  job "$model-paper-$task-patch_full" "$b/patch_full" scripts/relp_moe_compare_run.py patch $(cargs $model paper $task) \
    --verification "$b/verify/verification.json" --output-dir "$b/patch_full"
  if [ $first = 1 ]; then
    first=0; t_first=$(( $(date +%s) - t0 ))
    # projection: Qwen3 tasks ~ the first; Mixtral ~ 1.6x (measured stream / compute ratios)
    proj=$(python3 -c "print(round($t_first*(4+4*1.6)/3600,2))")
    log "S4d budget check: first task ${t_first}s -> projected S4d total ${proj} h"
    if python3 -c "import sys; sys.exit(0 if $t_first*(4+4*1.6)/3600 > 6 else 1)"; then
      progress "REVIEW S4d budget: projected ${proj} h > 6 h estimate; chain2 paused before the remaining S4d tasks"
      log "chain2 paused (budget)"; exit 0
    fi
  fi
done; done
progress "S4d paper full patching grids DONE"
log "chain2 end"
progress "chain2 end"
