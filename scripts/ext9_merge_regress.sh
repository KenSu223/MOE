#!/usr/bin/env bash
# ext9 engine merge + regression, run INSIDE one GPU-queue job (no other agent's GPU job runs between the merge and the
# regression; on a regression failure the previous engine and the official verification JSONs are restored before the lock
# is released).
#   bash scripts/gpu_queue.sh ext9-merge-regress -- bash scripts/ext9_merge_regress.sh
set -u
cd /home/ubuntu/MOE
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
EXPECT=1603112687e707a0d68caf93b6736f1c   # md5 of moetrace/engine.py when engine_ext9_dev.py was copied from it
BK=logs/ext9_engine_before_merge.py.bak
cur=$(md5sum moetrace/engine.py | cut -d' ' -f1)
if [ "$cur" != "$EXPECT" ]; then echo "$(ts) engine.py changed since the dev copy ($cur); not merging"; exit 2; fi
for f in verify_olmoe verify_ext5_engine_olmoe verify_ext8_engine_olmoe; do
  [ -f results/${f}_before_ext9.json ] || { echo "missing results/${f}_before_ext9.json"; exit 2; }
done
cp moetrace/engine.py "$BK"
cp moetrace/engine_ext9_dev.py moetrace/.engine_ext9_tmp.py && mv moetrace/.engine_ext9_tmp.py moetrace/engine.py
echo "$(ts) merged engine_ext9_dev.py -> engine.py ($(md5sum moetrace/engine.py | cut -d' ' -f1))"
python scripts/verify_olmoe.py > logs/ext9_regress_verify_olmoe.log 2>&1; r1=$?
python scripts/ext5_engine_verify.py > logs/ext9_regress_ext5_engine.log 2>&1; r2=$?
python scripts/ext8_engine_verify.py --engine main --n-cf 6 --n-wino-pairs 3 --stress 20000 > logs/ext9_regress_ext8_engine.log 2>&1; r3=$?
ok=1
[ $r1 = 0 ] && [ $r2 = 0 ] && [ $r3 = 0 ] || ok=0
for f in verify_olmoe verify_ext5_engine_olmoe verify_ext8_engine_olmoe; do
  python scripts/ext8_regress_compare.py results/$f.json results/${f}_before_ext9.json || ok=0
done
if [ $ok = 0 ]; then
  cp "$BK" moetrace/.engine_revert_tmp.py && mv moetrace/.engine_revert_tmp.py moetrace/engine.py
  for f in verify_olmoe verify_ext5_engine_olmoe verify_ext8_engine_olmoe; do
    cp results/$f.json logs/ext9_failed_$f.json 2>/dev/null; cp results/${f}_before_ext9.json results/$f.json
  done
  echo "$(ts) REGRESSION FAILED (rc $r1 $r2 $r3): engine.py reverted, official JSONs restored, failed runs in logs/ext9_failed_*.json"
  exit 1
fi
echo "$(ts) regression identical; running the ext9 verification on the merged engine"
python scripts/ext9_engine_verify.py --engine main --stress 10000 > logs/ext9_engine_verify.log 2>&1; r4=$?
echo "$(ts) ext9_engine_verify rc=$r4"
if [ $r4 = 0 ]; then
  cat >> logs/PROGRESS.md <<EOF
- $(ts) [ext9-knockout] ENGINE READY (ext9): moetrace/engine.py now has E4a PrefillSpec.route_mask=((layer, expert), ...) + route_mask_pos all|final + route_mask_mode reroute|zero (expert knockout on prefill rows; spawns on masked parents raise), E4b multi steps (layer, "attn_head", heads) and (layer, "heads_experts", (heads, experts)) as first and later steps in both directions, E4c DiagSpec.contrib_final_vectors=(layers) -> extra["diag"]["contrib_final_vectors"][l] fp32 [B, k, H] + ["moe_out_final_fp32"][l]; regression verify_olmoe / verify_ext5_engine / verify_ext8_engine identical to *_before_ext9.json; OLMoE vs HF: results/verify_ext9_engine_olmoe.json; API: logs/ext9_engine_api.md
EOF
fi
exit $r4
