#!/usr/bin/env bash
# ext14 (p6-engine) model verification chain, batched (fewer GPU-lock acquisitions than ext14_engine_chain_models.sh):
#   job A: qwen2_7b, qwen3_4b, qwen3_8b, qwen3_1p7b (full mode on the GPU + per-layer mode), ~10 min
#   job B: qwen3_14b (full mode with transformers offloaded to CPU RAM, light plan + per-layer mode)
#   job C: qwen2_57b (per-layer mode: engine identities, sanity prompts, transformers decoder layers), after the second
#          engine merge (logs/.ext14_engine_merged2)
# Each model is gated by scripts/ext14_engine_gate.py; on PASS the exact line '[p6-engine] MODEL READY <proto>' is appended
# to logs/PROGRESS.md. Models with logs/.ext14_engine_ready_<key> are skipped (re-runnable).
#   nohup setsid bash scripts/ext14_engine_chain_models2.sh > logs/ext14_engine_chain_models2.log 2>&1 < /dev/null &
set -u
MAIN=/home/ubuntu/MOE
cd $MAIN
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q="bash $MAIN/scripts/gpu_queue.sh"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

cmd_of() {  # key "full args" "per-layer layers" -> shell command (full args empty = per-layer only)
  local key=$1 fargs=$2 pl=$3 c=""
  if [ -n "$fargs" ]; then
    c="python scripts/ext14_engine_verify.py --model $key $fargs > logs/ext14_engine_verify_$key.log 2>&1"
  else
    c=": > logs/ext14_engine_verify_$key.log"
  fi
  echo "$c; python scripts/ext14_engine_verify.py --model $key --units 3 --per-layer $pl >> logs/ext14_engine_verify_$key.log 2>&1"
}

gate() {  # key proto has_full
  local key=$1 proto=$2 full=$3
  local jsons="results/verify_ext14_${key}_perlayer.json"
  [ "$full" = 1 ] && jsons="results/verify_ext14_$key.json $jsons"
  # only JSONs written by this chain's job count (guards against stale files)
  for j in $jsons; do [ -f $j ] && [ $j -nt logs/.ext14_engine_job_$key ] || { echo "$(ts) $key: $j missing or stale"; \
    echo "$(ts) [p6-engine] $key verification FAILED (no fresh $j): see logs/ext14_engine_verify_$key.log" >> logs/PROGRESS.md; return 1; }; done
  python scripts/ext14_engine_gate.py $jsons > logs/ext14_engine_gate_$key.log 2>&1; local g=$?
  echo "$(ts) $key gate=$g"; cat logs/ext14_engine_gate_$key.log
  if [ $g = 0 ]; then
    local det=$(grep -m1 "rescue corr\|update relerr" logs/ext14_engine_gate_$key.log | sed 's/^ *//')
    echo "$(ts) [p6-engine] MODEL READY $proto (gate PASS: $jsons; $det)" >> logs/PROGRESS.md
    touch logs/.ext14_engine_ready_$key
    return 0
  fi
  echo "$(ts) [p6-engine] $key verification FAILED (gate): see logs/ext14_engine_gate_$key.log" >> logs/PROGRESS.md
  return 1
}

batch() {  # job-name "key:proto:fargs:layers" ...
  local name=$1; shift
  local todo=() cmd="cd $MAIN"
  for spec in "$@"; do
    IFS=: read -r key proto fargs pl <<< "$spec"
    [ -f logs/.ext14_engine_ready_$key ] && { echo "$(ts) $key already ready"; continue; }
    todo+=("$spec")
    touch logs/.ext14_engine_job_$key
    cmd="$cmd; $(cmd_of $key "$fargs" $pl)"
  done
  [ ${#todo[@]} = 0 ] && return 0
  sleep 1
  $Q $name -- bash -c "$cmd"; echo "$(ts) job $name rc=$?"
  for spec in "${todo[@]}"; do
    IFS=: read -r key proto fargs pl <<< "$spec"
    gate $key $proto $([ -n "$fargs" ] && echo 1 || echo 0)
  done
}

batch p6eng-verify-dense "qwen2_7b:qwen2_7b:--units 6:1,14,27" "qwen3_4b:qwen3_4b:--units 6:1,18,35" \
      "qwen3_8b:qwen3_8b:--units 6:1,18,35" "qwen3_1p7b:qwen3_1p7b:--units 6:1,14,27"
batch p6eng-verify-qwen3_14b "qwen3_14b:qwen3_14b:--units 3 --offload --light:1,20,39"
until [ -f logs/.ext14_engine_merged2 ]; do sleep 60; done
batch p6eng-verify-qwen2_57b "qwen2_57b:qwen2_57b::1,13,27"
echo "$(ts) chain done"
