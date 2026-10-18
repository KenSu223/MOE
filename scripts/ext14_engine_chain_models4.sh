#!/usr/bin/env bash
# ext14 (p6-engine) fourth verification chain (after merge3: model numerics protocols in the engine).
#   job 1 qwen2_57b : protocol check (chain + engine-vs-fp32 with fp32 router logits); the PROTOCOL decision and line are
#                     written by hand from these JSONs (coordinator rule 2026-10-17).
#   job 2 qwen2_7b  : chain, full bf16 (light, 2 pairs), full fp32 (transformers offloaded to CPU RAM, same plan), engine bf16
#                     vs fp32, per-layer -> noise compare (is the bf16 deviation transformers' own bf16 noise?) + gate
#   job 3 qwen3_14b : full bf16 (offloaded, light; plan bug fixed), chain, per-layer -> gate with the chain noise floor
#   job 4 qwen3_1p7b: full bf16 (competent-unit statistic), full fp32, chain, engine bf16 vs fp32, per-layer -> gate
#   nohup setsid bash scripts/ext14_engine_chain_models4.sh > logs/ext14_engine_chain_models4.log 2>&1 < /dev/null &
set -u
MAIN=/home/ubuntu/MOE
cd $MAIN
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q="bash $MAIN/scripts/gpu_queue.sh"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
V="python scripts/ext14_engine_verify.py"
R=results/verify_ext14
until [ -f logs/.ext14_engine_merged3 ]; do sleep 60; done

fresh() {  # key files... : all exist and are newer than the job marker
  local key=$1; shift
  for j in "$@"; do [ -f $j ] && [ $j -nt logs/.ext14_engine_job4_$key ] || { echo "$(ts) $key: $j missing or stale"; return 1; }; done
}
gate() {  # key proto gate-args...
  local key=$1 proto=$2; shift 2
  python scripts/ext14_engine_gate.py "$@" > logs/ext14_engine_gate4_$key.log 2>&1; local g=$?
  echo "$(ts) $key gate=$g"; cat logs/ext14_engine_gate4_$key.log
  if [ $g = 0 ]; then
    if [ ! -f logs/.ext14_engine_ready_$key ]; then
      echo "$(ts) [p6-engine] MODEL READY $proto (gate PASS: $(echo "$@" | tr ' ' '\n' | grep results | tr '\n' ' '); log logs/ext14_engine_gate4_$key.log)" >> logs/PROGRESS.md
      touch logs/.ext14_engine_ready_$key
    fi
    return 0
  fi
  echo "$(ts) [p6-engine] $key verification FAILED (gate): see logs/ext14_engine_gate4_$key.log" >> logs/PROGRESS.md
  return 1
}
job() {  # key name cmd
  touch logs/.ext14_engine_job4_$1; sleep 1
  $Q $2 -- bash -c "cd $MAIN; $3" > /dev/null; echo "$(ts) job $2 rc=$?"
}

# fp32 router logits alone did not fix the bf16 fragility (transformers chain: routing agreement 0.19, sanity Thursday rank
# 124639; results/verify_ext14_qwen2_57b_router.json) -> measure bf16 and bf16 + fp32-router engines against the fp32 engine
# on a T2-like plan (rescue correlation, attention share, pass time, peak memory of the fp32 engine)
job qwen2_57b p6eng-v4-qwen2_57b "$V --model qwen2_57b --units 3 --vs-fp32 > logs/ext14_engine_verify4_qwen2_57b.log 2>&1; $V --model qwen2_57b --units 3 --vs-fp32 --router-fp32 >> logs/ext14_engine_verify4_qwen2_57b.log 2>&1"
fresh qwen2_57b ${R}_qwen2_57b_vsfp32.json ${R}_qwen2_57b_r32_vsfp32.json && \
  python scripts/ext14_engine_gate.py ${R}_qwen2_57b_vsfp32.json ${R}_qwen2_57b_r32_vsfp32.json > logs/ext14_engine_gate4_qwen2_57b.log 2>&1
cat logs/ext14_engine_gate4_qwen2_57b.log; echo "$(ts) [p6-engine] qwen2_57b fp32-router protocol check finished: logs/ext14_engine_gate4_qwen2_57b.log (decision follows)" >> logs/PROGRESS.md

if [ ! -f logs/.ext14_engine_ready_qwen2_7b ]; then
  job qwen2_7b p6eng-v4-qwen2_7b "$V --model qwen2_7b --units 3 --chain > logs/ext14_engine_verify4_qwen2_7b.log 2>&1; $V --model qwen2_7b --units 2 --light >> logs/ext14_engine_verify4_qwen2_7b.log 2>&1; $V --model qwen2_7b --units 2 --light --dtype fp32 --offload >> logs/ext14_engine_verify4_qwen2_7b.log 2>&1; $V --model qwen2_7b --units 3 --vs-fp32 >> logs/ext14_engine_verify4_qwen2_7b.log 2>&1; $V --model qwen2_7b --units 3 --per-layer 1,14,27 >> logs/ext14_engine_verify4_qwen2_7b.log 2>&1"
  if fresh qwen2_7b ${R}_qwen2_7b.json ${R}_qwen2_7b_fp32.json ${R}_qwen2_7b_chain.json ${R}_qwen2_7b_vsfp32.json ${R}_qwen2_7b_perlayer.json; then
    python scripts/ext14_engine_noise_compare.py ${R}_qwen2_7b.json ${R}_qwen2_7b_fp32.json --out ${R}_qwen2_7b_noise.json > /dev/null 2>&1
    gate qwen2_7b qwen2_7b ${R}_qwen2_7b_fp32.json ${R}_qwen2_7b_noise.json ${R}_qwen2_7b_chain.json ${R}_qwen2_7b_perlayer.json ${R}_qwen2_7b_vsfp32.json
  fi
fi
if [ ! -f logs/.ext14_engine_ready_qwen3_14b ]; then
  job qwen3_14b p6eng-v4-qwen3_14b "$V --model qwen3_14b --units 3 --offload --light > logs/ext14_engine_verify4_qwen3_14b.log 2>&1; $V --model qwen3_14b --units 3 --chain >> logs/ext14_engine_verify4_qwen3_14b.log 2>&1; $V --model qwen3_14b --units 3 --per-layer 1,20,39 >> logs/ext14_engine_verify4_qwen3_14b.log 2>&1"
  fresh qwen3_14b ${R}_qwen3_14b.json ${R}_qwen3_14b_chain.json ${R}_qwen3_14b_perlayer.json && \
    gate qwen3_14b qwen3_14b ${R}_qwen3_14b.json ${R}_qwen3_14b_chain.json ${R}_qwen3_14b_perlayer.json --noise ${R}_qwen3_14b_chain.json
fi
if [ ! -f logs/.ext14_engine_ready_qwen3_1p7b ]; then
  job qwen3_1p7b p6eng-v4-qwen3_1p7b "$V --model qwen3_1p7b --units 6 > logs/ext14_engine_verify4_qwen3_1p7b.log 2>&1; $V --model qwen3_1p7b --units 3 --dtype fp32 >> logs/ext14_engine_verify4_qwen3_1p7b.log 2>&1; $V --model qwen3_1p7b --units 3 --chain >> logs/ext14_engine_verify4_qwen3_1p7b.log 2>&1; $V --model qwen3_1p7b --units 3 --vs-fp32 >> logs/ext14_engine_verify4_qwen3_1p7b.log 2>&1; $V --model qwen3_1p7b --units 3 --per-layer 1,14,27 >> logs/ext14_engine_verify4_qwen3_1p7b.log 2>&1"
  if fresh qwen3_1p7b ${R}_qwen3_1p7b.json ${R}_qwen3_1p7b_fp32.json ${R}_qwen3_1p7b_chain.json ${R}_qwen3_1p7b_vsfp32.json ${R}_qwen3_1p7b_perlayer.json; then
    python scripts/ext14_engine_noise_compare.py ${R}_qwen3_1p7b.json ${R}_qwen3_1p7b_fp32.json --out ${R}_qwen3_1p7b_noise.json > /dev/null 2>&1
    gate qwen3_1p7b qwen3_1p7b ${R}_qwen3_1p7b.json ${R}_qwen3_1p7b_fp32.json ${R}_qwen3_1p7b_noise.json ${R}_qwen3_1p7b_chain.json ${R}_qwen3_1p7b_vsfp32.json ${R}_qwen3_1p7b_perlayer.json --noise ${R}_qwen3_1p7b_chain.json
  fi
fi
echo "$(ts) chain done"
