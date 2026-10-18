#!/usr/bin/env bash
# ext14 (p6-engine) model verification chain (main tree, after the engine merge). One GPU-queue job per model:
# full mode (transformers hooks; Qwen3-14B with CPU offload and the light plan) + per-layer mode (engine identities,
# sanity prompts, transformers decoder layers on the engine's residual stream). The gate (scripts/ext14_engine_gate.py)
# decides; on PASS the exact handshake line '[p6-engine] MODEL READY <proto>' is appended to logs/PROGRESS.md.
# After Mistral-7B passes (with the tiny fp32 and OLMoE checks of the merge job), '[p6-engine] NEURONS READY' is written.
# Re-runnable: models with logs/.ext14_engine_ready_<key> are skipped.
#   nohup setsid bash scripts/ext14_engine_chain_models.sh > logs/ext14_engine_chain_models.log 2>&1 < /dev/null &
set -u
MAIN=/home/ubuntu/MOE
cd $MAIN
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q="bash $MAIN/scripts/gpu_queue.sh"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

verify() {  # key proto "full args" "per-layer layers" (full args empty = per-layer only)
  local key=$1 proto=$2 fargs=$3 pl=$4
  [ -f logs/.ext14_engine_ready_$key ] && { echo "$(ts) $key already ready"; return 0; }
  local cmd="cd $MAIN"
  local jsons="results/verify_ext14_${key}_perlayer.json"
  if [ -n "$fargs" ]; then
    cmd="$cmd && python scripts/ext14_engine_verify.py --model $key $fargs > logs/ext14_engine_verify_$key.log 2>&1"
    jsons="results/verify_ext14_$key.json $jsons"
  else
    cmd="$cmd && : > logs/ext14_engine_verify_$key.log"
  fi
  cmd="$cmd && python scripts/ext14_engine_verify.py --model $key --units 3 --per-layer $pl >> logs/ext14_engine_verify_$key.log 2>&1"
  $Q p6eng-verify-$key -- bash -c "$cmd"; local rc=$?
  python scripts/ext14_engine_gate.py $jsons > logs/ext14_engine_gate_$key.log 2>&1; local g=$?
  echo "$(ts) $key verify rc=$rc gate=$g"; cat logs/ext14_engine_gate_$key.log
  if [ $rc = 0 ] && [ $g = 0 ]; then
    local det=$(grep -m1 "rescue corr\|update relerr" logs/ext14_engine_gate_$key.log | sed 's/^ *//')
    echo "$(ts) [p6-engine] MODEL READY $proto (gate PASS: $jsons; $det)" >> logs/PROGRESS.md
    touch logs/.ext14_engine_ready_$key
    return 0
  fi
  echo "$(ts) [p6-engine] $key verification FAILED (rc $rc, gate $g): see logs/ext14_engine_gate_$key.log" >> logs/PROGRESS.md
  return 1
}

verify mistral7b mistral7b_bos "--units 6" "1,16,31"
if [ -f logs/.ext14_engine_ready_mistral7b ] && [ ! -f logs/.ext14_engine_neurons_ready ]; then
  ok=1
  for f in mistral qwen2 qwen3 qwen2_moe; do
    python scripts/ext14_engine_gate.py results/verify_ext14_tiny_$f.json results/verify_ext14_tiny_${f}_perlayer.json --fp32 >> logs/ext14_engine_gate_neurons.log 2>&1 || ok=0
  done
  python scripts/ext14_engine_gate.py results/verify_ext14_olmoe.json >> logs/ext14_engine_gate_neurons.log 2>&1 || ok=0
  if [ $ok = 1 ]; then
    echo "$(ts) [p6-engine] NEURONS READY (DiagSpec.expert_neurons + kind expert_neurons; = transformers to fp32 precision on tiny mistral / qwen2 / qwen3 / qwen2_moe, bf16 on OLMoE and Mistral-7B; API logs/ext14_engine_api.md)" >> logs/PROGRESS.md
    touch logs/.ext14_engine_neurons_ready
  else
    echo "$(ts) [p6-engine] NEURONS check FAILED: logs/ext14_engine_gate_neurons.log" >> logs/PROGRESS.md
  fi
fi
verify qwen2_7b qwen2_7b "--units 6" "1,14,27"
verify qwen3_4b qwen3_4b "--units 6" "1,18,35"
verify qwen3_8b qwen3_8b "--units 6" "1,18,35"
verify qwen3_1p7b qwen3_1p7b "--units 6" "1,14,27"
verify qwen3_14b qwen3_14b "--units 3 --offload --light" "1,20,39"
# Qwen2-57B after the second engine merge (shared-expert route mask, coordinator request 2026-10-16 20:47Z)
until [ -f logs/.ext14_engine_merged2 ]; do sleep 60; done
verify qwen2_57b qwen2_57b "" "1,13,27"
echo "$(ts) chain done"
