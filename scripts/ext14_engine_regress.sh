#!/usr/bin/env bash
# ext14 (p6-engine) engine regression in a given tree (the dev worktree, or the main tree after the merge).
# Re-runs the official OLMoE verifications (verify_olmoe, ext5/ext8/ext9 engine checks, with the arguments of their
# official runs) and one stored Phase-5 sweep pass, and compares them with the reference copies
# /home/ubuntu/MOE/results/verify_ext14_before_*.json (non-timing fields identical) / the stored rows (bit for bit).
#   bash scripts/gpu_queue.sh p6eng-regress -- bash <tree>/scripts/ext14_engine_regress.sh <tree> <tag>
set -u
TREE=${1:?tree}; TAG=${2:?tag}
MAIN=/home/ubuntu/MOE
cd "$TREE"
. $MAIN/.venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
RW="python $TREE/scripts/ext14_engine_runwt.py $TREE"
L=$MAIN/logs/ext14_engine_regress_$TAG
mkdir -p results
$RW $MAIN/scripts/verify_olmoe.py > ${L}_verify_olmoe.log 2>&1; r1=$?
$RW $MAIN/scripts/ext5_engine_verify.py > ${L}_ext5.log 2>&1; r2=$?
$RW $MAIN/scripts/ext8_engine_verify.py --engine main --n-cf 6 --n-wino-pairs 3 --stress 20000 --out results/verify_ext8_engine_olmoe.json > ${L}_ext8.log 2>&1; r3=$?
$RW $MAIN/scripts/ext9_engine_verify.py --engine main --stress 10000 --out results/verify_ext9_engine_olmoe.json > ${L}_ext9.log 2>&1; r4=$?
ok=1
[ $r1 = 0 ] && [ $r2 = 0 ] && [ $r3 = 0 ] && [ $r4 = 0 ] || ok=0
for f in verify_olmoe verify_ext5_engine_olmoe verify_ext8_engine_olmoe verify_ext9_engine_olmoe; do
  python $MAIN/scripts/ext9_regress_compare.py results/$f.json $MAIN/results/verify_ext14_before_$f.json >> ${L}_compare.log 2>&1 || ok=0
done
$RW $TREE/scripts/ext14_engine_regress_sweep.py --run p5_f2e_qwen3 --out results/verify_ext14_regress_sweep_p5_f2e_qwen3_$TAG.json > ${L}_sweep_qwen3.log 2>&1 || ok=0
$RW $TREE/scripts/ext14_engine_regress_sweep.py --run p5_f2e_mixtral_bos --out results/verify_ext14_regress_sweep_p5_f2e_mixtral_bos_$TAG.json > ${L}_sweep_mixtral.log 2>&1 || ok=0
echo "$(ts) regression $TAG in $TREE: rc $r1 $r2 $r3 $r4 ok=$ok" | tee -a ${L}_compare.log
[ $ok = 1 ]
