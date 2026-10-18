#!/usr/bin/env bash
# ext14 (p6-engine): merge job (engine files -> main tree, regression, official tiny + OLMoE verification) through the GPU
# queue; on success writes '[p6-engine] HEADMASK READY' and continues with the model verification chain.
set -u
MAIN=/home/ubuntu/MOE
cd $MAIN
. .venv/bin/activate
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
if [ ! -f logs/.ext14_engine_merged ]; then
  bash scripts/gpu_queue.sh p6eng-merge -- bash scripts/ext14_engine_merge.sh; rc=$?
  echo "$(ts) merge job rc=$rc"
  [ $rc = 0 ] || { echo "$(ts) [p6-engine] engine merge FAILED (rc $rc): logs/ext14_engine_chain_merge.log" >> logs/PROGRESS.md; exit 1; }
  touch logs/.ext14_engine_merged
fi
ok=1
python scripts/ext14_engine_gate.py results/verify_ext14_olmoe.json > logs/ext14_engine_gate_olmoe.log 2>&1 || ok=0
for f in mistral qwen2 qwen3 qwen2_moe; do
  python scripts/ext14_engine_gate.py results/verify_ext14_tiny_$f.json results/verify_ext14_tiny_${f}_perlayer.json --fp32 >> logs/ext14_engine_gate_olmoe.log 2>&1 || ok=0
done
cat logs/ext14_engine_gate_olmoe.log
if [ $ok = 1 ] && [ ! -f logs/.ext14_engine_headmask_ready ]; then
  det=$(grep -m1 "rescue corr" logs/ext14_engine_gate_olmoe.log | sed 's/^ *//')
  echo "$(ts) [p6-engine] HEADMASK READY (engine merged into moetrace/; PrefillSpec.head_mask zero|mean and PrefillSpec.route_mask_inherit=True, wavefront rows inherit both masks; regression identical (verify_olmoe / ext5 / ext8 / ext9 engine checks, Phase-5 sweep passes p5_f2e_qwen3 / p5_f2e_mixtral_bos bit for bit); vs transformers hooks: tiny fp32 exact, OLMoE bf16 $det (results/verify_ext14_olmoe.json); API logs/ext14_engine_api.md)" >> logs/PROGRESS.md
  touch logs/.ext14_engine_headmask_ready
fi
[ $ok = 1 ] || { echo "$(ts) [p6-engine] official verification gate FAILED: logs/ext14_engine_gate_olmoe.log" >> logs/PROGRESS.md; exit 1; }
exec bash scripts/ext14_engine_chain_models.sh
