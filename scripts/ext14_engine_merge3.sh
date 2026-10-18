#!/usr/bin/env bash
# ext14 (p6-engine) third merge: model numerics protocols (ArchSpec.router_fp32, Engine reads models.py flags
# router_fp32 / act_dtype; default behaviour unchanged) -> main tree, inside one GPU-queue job: md5 check against the second
# merge, backup, atomic copy (arch.py before engine.py), regression in the main tree (revert + restore on failure), official
# tiny verification. On success touches logs/.ext14_engine_merged3.
#   bash scripts/gpu_queue.sh p6eng-merge3 -- bash /home/ubuntu/MOE/scripts/ext14_engine_merge3.sh
set -u
MAIN=/home/ubuntu/MOE; WT=/home/ubuntu/MOE_wt_p6engine
cd $MAIN
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
declare -A EXPECT=([engine]=264d6465aa9038c97024f410dd85ca98 [arch]=d929a3bb2389bde5f76afe17645c5a3b)
BK=$MAIN/logs/ext14_engine_before_merge3
mkdir -p $BK
for f in arch engine; do
  cur=$(md5sum moetrace/$f.py | cut -d' ' -f1)
  [ "$cur" = "${EXPECT[$f]}" ] || { echo "$(ts) $f.py changed since the second merge ($cur); not merging"; exit 2; }
done
cmp -s moetrace/weights.py $WT/moetrace/weights.py || { echo "$(ts) weights.py differs from the worktree; not merging"; exit 2; }
for f in arch engine; do cp moetrace/$f.py $BK/$f.py.bak; done
for f in arch engine; do cp $WT/moetrace/$f.py moetrace/$f.py.ext14tmp && mv moetrace/$f.py.ext14tmp moetrace/$f.py; done
echo "$(ts) merged arch.py ($(md5sum moetrace/arch.py | cut -c1-8)) engine.py ($(md5sum moetrace/engine.py | cut -c1-8))"
bash $MAIN/scripts/ext14_engine_regress.sh $MAIN main3; rr=$?
for f in verify_olmoe verify_ext5_engine_olmoe verify_ext8_engine_olmoe verify_ext9_engine_olmoe; do
  cp results/$f.json results/verify_ext14_regress_main3_$f.json 2>/dev/null
  cp results/verify_ext14_before_$f.json results/$f.json   # keep the committed official JSONs (identical results)
done
if [ $rr != 0 ]; then
  for f in engine arch; do cp $BK/$f.py.bak moetrace/$f.py.ext14tmp && mv moetrace/$f.py.ext14tmp moetrace/$f.py; done
  echo "$(ts) REGRESSION FAILED: arch.py / engine.py reverted (failed runs: results/verify_ext14_regress_main3_*.json)"
  exit 1
fi
echo "$(ts) regression identical in the main tree"
ok=1
for f in qwen2_moe mistral qwen2 qwen3; do
  python scripts/ext14_engine_verify.py --model /opt/dlami/nvme/p6_tiny/$f --dtype fp32 --units 4 > logs/ext14_engine_verify_tiny_$f.log 2>&1 || ok=0
  python scripts/ext14_engine_gate.py results/verify_ext14_tiny_$f.json results/verify_ext14_tiny_${f}_perlayer.json --fp32 >> logs/ext14_engine_verify_tiny_$f.log 2>&1 || ok=0
done
echo "$(ts) official tiny verification ok=$ok"
[ $ok = 1 ] && touch logs/.ext14_engine_merged3
[ $ok = 1 ]
