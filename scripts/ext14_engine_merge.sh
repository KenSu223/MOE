#!/usr/bin/env bash
# ext14 (p6-engine) merge of the dev worktree's engine files into the main tree + regression + official verification,
# run INSIDE one GPU-queue job (no other GPU job runs between the merge and the regression; on a regression failure the
# previous files and the official verification JSONs are restored before the lock is released).
#   bash scripts/gpu_queue.sh p6eng-merge -- bash /home/ubuntu/MOE/scripts/ext14_engine_merge.sh
set -u
MAIN=/home/ubuntu/MOE; WT=/home/ubuntu/MOE_wt_p6engine
cd $MAIN
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
FILES="moetrace/arch.py moetrace/weights.py moetrace/models.py moetrace/engine.py"
BK=$MAIN/logs/ext14_engine_before_merge
mkdir -p $BK
# 1. the main tree's files must still be the branch-point versions (nobody else edited them)
for f in $FILES; do
  if ! git -C $MAIN diff --quiet 4143e09 -- $f; then echo "$(ts) $f changed in the main tree since 4143e09; not merging"; exit 2; fi
done
for f in verify_olmoe verify_ext5_engine_olmoe verify_ext8_engine_olmoe verify_ext9_engine_olmoe; do
  [ -f results/verify_ext14_before_$f.json ] || { echo "missing results/verify_ext14_before_$f.json"; exit 2; }
done
# 2. backup + atomic replacement (arch / weights / models first: the old engine runs with the new arch / weights)
for f in $FILES; do cp $f $BK/$(basename $f).bak; done
for f in $FILES; do cp $WT/$f $f.ext14tmp && mv $f.ext14tmp $f; done
echo "$(ts) merged: $(for f in $FILES; do echo -n "$(basename $f)=$(md5sum $f | cut -c1-8) "; done)"
# 3. regression in the main tree
bash $MAIN/scripts/ext14_engine_regress.sh $MAIN main; rr=$?
if [ $rr != 0 ]; then
  for f in $FILES; do cp $BK/$(basename $f).bak $f.ext14tmp && mv $f.ext14tmp $f; done
  for f in verify_olmoe verify_ext5_engine_olmoe verify_ext8_engine_olmoe verify_ext9_engine_olmoe; do
    cp results/$f.json logs/ext14_failed_$f.json 2>/dev/null; cp results/verify_ext14_before_$f.json results/$f.json
  done
  echo "$(ts) REGRESSION FAILED: files reverted, official JSONs restored (failed runs in logs/ext14_failed_*.json)"
  exit 1
fi
echo "$(ts) regression identical in the main tree"
# 4. official ext14 verifications on the merged engine: tiny fp32 models (4 families, full + per-layer) and OLMoE (bf16)
ok=1
for f in mistral qwen2 qwen3 qwen2_moe; do
  python scripts/ext14_engine_verify.py --model /opt/dlami/nvme/p6_tiny/$f --dtype fp32 --units 4 > logs/ext14_engine_verify_tiny_$f.log 2>&1 || ok=0
  python scripts/ext14_engine_verify.py --model /opt/dlami/nvme/p6_tiny/$f --dtype fp32 --units 2 --per-layer 1,2,3 >> logs/ext14_engine_verify_tiny_$f.log 2>&1 || ok=0
done
python scripts/ext14_engine_verify.py --model olmoe --units 6 > logs/ext14_engine_verify_olmoe.log 2>&1 || ok=0
echo "$(ts) official tiny + OLMoE verification ok=$ok"
[ $ok = 1 ]
