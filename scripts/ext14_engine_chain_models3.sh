#!/usr/bin/env bash
# ext14 (p6-engine) third model-verification chain (coordinator request after the failed gates of 2026-10-16 22:20-22:41Z).
# Adds transformers' own bf16 noise floor: chain mode (transformers decoder layers chained one at a time in bf16 and fp32,
# engine bf16 and fp32, on the units' clean prompts + sanity prompts) and an fp32 reference layer in the per-layer mode;
# full-mode runs in fp32 (engine fp32 vs transformers fp32) prove implementation exactness on the real weights.
#   job 1 qwen2_57b : chain + per-layer (fp32 floor)
#   job 2 qwen2_7b  : chain + per-layer (fp32 floor) + full fp32 (transformers offloaded to CPU RAM, light plan)
#   job 3 qwen3_14b : full bf16 (offloaded, light plan; multi-step layer bug of the light plan fixed) + chain + per-layer
#   job 4 qwen3_1p7b: full bf16 (competent-unit statistic) + full fp32 + chain + per-layer
# Gates: scripts/ext14_engine_gate.py (bf16 full-mode thresholds scaled by the chain's noise floor via --noise).
#   nohup setsid bash scripts/ext14_engine_chain_models3.sh > logs/ext14_engine_chain_models3.log 2>&1 < /dev/null &
set -u
MAIN=/home/ubuntu/MOE
cd $MAIN
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q="bash $MAIN/scripts/gpu_queue.sh"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
V="python scripts/ext14_engine_verify.py"

run_gate() {  # key proto json...
  local key=$1 proto=$2; shift 2
  for j in "$@"; do
    [ "$j" = "--noise" ] && continue
    case "$j" in results/verify_ext14_${key}*) [ -f $j ] && [ $j -nt logs/.ext14_engine_job3_$key ] || { echo "$(ts) $key: $j missing or stale";
      echo "$(ts) [p6-engine] $key verification FAILED (no fresh $j): see logs/ext14_engine_verify3_$key.log" >> logs/PROGRESS.md; return 1; };; esac
  done
  python scripts/ext14_engine_gate.py "$@" > logs/ext14_engine_gate3_$key.log 2>&1; local g=$?
  echo "$(ts) $key gate=$g"; cat logs/ext14_engine_gate3_$key.log
  if [ $g = 0 ]; then
    local det=$(grep -m1 "chain eng_bf16__hf_fp32\|rescue corr" logs/ext14_engine_gate3_$key.log | sed 's/^ *//' | cut -c1-200)
    echo "$(ts) [p6-engine] MODEL READY $proto (gate PASS: $*; $det)" >> logs/PROGRESS.md
    touch logs/.ext14_engine_ready_$key
    return 0
  fi
  echo "$(ts) [p6-engine] $key verification FAILED (gate): see logs/ext14_engine_gate3_$key.log" >> logs/PROGRESS.md
  return 1
}

job() {  # key proto job-name "commands"
  local key=$1 proto=$2 name=$3 cmd=$4
  [ -f logs/.ext14_engine_ready_$key ] && { echo "$(ts) $key already ready"; return 1; }
  touch logs/.ext14_engine_job3_$key; sleep 1
  $Q $name -- bash -c "cd $MAIN; $cmd" > /dev/null; echo "$(ts) job $name rc=$?"
  return 0
}

R=results/verify_ext14
if job qwen2_57b qwen2_57b p6eng-v3-qwen2_57b "$V --model qwen2_57b --units 3 --chain > logs/ext14_engine_verify3_qwen2_57b.log 2>&1; $V --model qwen2_57b --units 3 --per-layer 1,13,27 >> logs/ext14_engine_verify3_qwen2_57b.log 2>&1"; then
  run_gate qwen2_57b qwen2_57b ${R}_qwen2_57b_chain.json ${R}_qwen2_57b_perlayer.json ${R}_tiny_qwen2_moe.json ${R}_tiny_qwen2_moe_perlayer.json
fi
if job qwen2_7b qwen2_7b p6eng-v3-qwen2_7b "$V --model qwen2_7b --units 3 --chain > logs/ext14_engine_verify3_qwen2_7b.log 2>&1; $V --model qwen2_7b --units 6 >> logs/ext14_engine_verify3_qwen2_7b.log 2>&1; $V --model qwen2_7b --units 3 --per-layer 1,14,27 >> logs/ext14_engine_verify3_qwen2_7b.log 2>&1; $V --model qwen2_7b --units 2 --dtype fp32 --offload --light >> logs/ext14_engine_verify3_qwen2_7b.log 2>&1"; then
  run_gate qwen2_7b qwen2_7b ${R}_qwen2_7b_fp32.json ${R}_qwen2_7b_chain.json ${R}_qwen2_7b_perlayer.json ${R}_qwen2_7b.json --noise ${R}_qwen2_7b_chain.json
fi
if job qwen3_14b qwen3_14b p6eng-v3-qwen3_14b "$V --model qwen3_14b --units 3 --offload --light > logs/ext14_engine_verify3_qwen3_14b.log 2>&1; $V --model qwen3_14b --units 3 --chain >> logs/ext14_engine_verify3_qwen3_14b.log 2>&1; $V --model qwen3_14b --units 3 --per-layer 1,20,39 >> logs/ext14_engine_verify3_qwen3_14b.log 2>&1"; then
  run_gate qwen3_14b qwen3_14b ${R}_qwen3_14b.json ${R}_qwen3_14b_chain.json ${R}_qwen3_14b_perlayer.json --noise ${R}_qwen3_14b_chain.json
fi
if job qwen3_1p7b qwen3_1p7b p6eng-v3-qwen3_1p7b "$V --model qwen3_1p7b --units 6 > logs/ext14_engine_verify3_qwen3_1p7b.log 2>&1; $V --model qwen3_1p7b --units 3 --dtype fp32 >> logs/ext14_engine_verify3_qwen3_1p7b.log 2>&1; $V --model qwen3_1p7b --units 3 --chain >> logs/ext14_engine_verify3_qwen3_1p7b.log 2>&1; $V --model qwen3_1p7b --units 3 --per-layer 1,14,27 >> logs/ext14_engine_verify3_qwen3_1p7b.log 2>&1"; then
  run_gate qwen3_1p7b qwen3_1p7b ${R}_qwen3_1p7b.json ${R}_qwen3_1p7b_fp32.json ${R}_qwen3_1p7b_chain.json ${R}_qwen3_1p7b_perlayer.json --noise ${R}_qwen3_1p7b_chain.json
fi
echo "$(ts) chain done"
