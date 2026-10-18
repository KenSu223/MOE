#!/usr/bin/env bash
# ext14 (p6-engine) second merge: engine.py with the Qwen2-MoE shared-expert route mask (coordinator request 2026-10-16
# 20:47Z) -> main tree, inside one GPU-queue job: md5 check against the first merge, backup, atomic copy, regression in the
# main tree (revert + restore on failure), official tiny verification (the shared-expert knockout is checked on the tiny
# Qwen2-MoE). On success touches logs/.ext14_engine_merged2 (the model chain then runs the Qwen2-57B verification).
#   bash scripts/gpu_queue.sh p6eng-merge2 -- bash /home/ubuntu/MOE/scripts/ext14_engine_merge2.sh
set -u
MAIN=/home/ubuntu/MOE; WT=/home/ubuntu/MOE_wt_p6engine
cd $MAIN
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
EXPECT=572a24bb62b0f6d7e149f654909ff753   # md5 of moetrace/engine.py after the first merge (2026-10-16 19:54Z)
BK=$MAIN/logs/ext14_engine_before_merge2
mkdir -p $BK
cur=$(md5sum moetrace/engine.py | cut -d' ' -f1)
[ "$cur" = "$EXPECT" ] || { echo "$(ts) engine.py changed since the first merge ($cur); not merging"; exit 2; }
for f in arch weights models; do cmp -s moetrace/$f.py $WT/moetrace/$f.py || { echo "$(ts) $f.py differs from the worktree; not merging"; exit 2; }; done
cp moetrace/engine.py $BK/engine.py.bak
cp $WT/moetrace/engine.py moetrace/engine.py.ext14tmp && mv moetrace/engine.py.ext14tmp moetrace/engine.py
echo "$(ts) merged engine.py ($(md5sum moetrace/engine.py | cut -c1-8))"
bash $MAIN/scripts/ext14_engine_regress.sh $MAIN main2; rr=$?
for f in verify_olmoe verify_ext5_engine_olmoe verify_ext8_engine_olmoe verify_ext9_engine_olmoe; do
  cp results/$f.json results/verify_ext14_regress_main2_$f.json 2>/dev/null
  cp results/verify_ext14_before_$f.json results/$f.json   # keep the committed official JSONs (identical results)
done
if [ $rr != 0 ]; then
  cp $BK/engine.py.bak moetrace/engine.py.ext14tmp && mv moetrace/engine.py.ext14tmp moetrace/engine.py
  echo "$(ts) REGRESSION FAILED: engine.py reverted (failed runs: results/verify_ext14_regress_main2_*.json)"
  exit 1
fi
echo "$(ts) regression identical in the main tree"
ok=1
for f in qwen2_moe mistral qwen2 qwen3; do
  python scripts/ext14_engine_verify.py --model /opt/dlami/nvme/p6_tiny/$f --dtype fp32 --units 4 > logs/ext14_engine_verify_tiny_$f.log 2>&1 || ok=0
  python scripts/ext14_engine_verify.py --model /opt/dlami/nvme/p6_tiny/$f --dtype fp32 --units 2 --per-layer 1,2,3 >> logs/ext14_engine_verify_tiny_$f.log 2>&1 || ok=0
  python scripts/ext14_engine_gate.py results/verify_ext14_tiny_$f.json results/verify_ext14_tiny_${f}_perlayer.json --fp32 >> logs/ext14_engine_verify_tiny_$f.log 2>&1 || ok=0
done
echo "$(ts) official tiny verification ok=$ok"
if [ $ok = 1 ]; then
  touch logs/.ext14_engine_merged2
  echo "$(ts) [p6-engine] engine update merged: PrefillSpec.route_mask may knock out the Qwen2-57B shared expert (expert 64, route_mask_mode zero; reroute raises ValueError; inherited by spawns with route_mask_inherit); = transformers (hook zeroing mlp.shared_expert) to fp32 precision on the tiny Qwen2-MoE; regression identical. MODEL READY qwen2_57b follows its verification." >> logs/PROGRESS.md
fi
[ $ok = 1 ]
