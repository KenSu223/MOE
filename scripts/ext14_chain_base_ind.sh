#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): apply the IND PILOT decision. Waits for "[coordinator] IND <frame>" in logs/PROGRESS.md or 30 min after the
# "[p6-data] IND PILOT" line, then rebuilds l_ind in that frame (the coordinator's frame if given, else the pre-set rule's from
# data/dial/qa/ind_pilot.json), rescans it in both models and registers its case sets (the tier-1 chain then picks it up).
# Usage: nohup setsid bash scripts/ext14_chain_base_ind.sh > logs/ext14_base_ind.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
until grep -q "\[p6-data\] IND PILOT" logs/PROGRESS.md; do sleep 30; done
t0=$(date +%s)
until grep -Eq "\[coordinator\] IND I[0-4]" logs/PROGRESS.md || [ $(( $(date +%s) - t0 )) -ge 1800 ]; do sleep 30; done
F=$(grep -Eo "\[coordinator\] IND I[0-4]" logs/PROGRESS.md | tail -1 | grep -Eo "I[0-4]")
[ -z "$F" ] && F=$(python -c "import json; print(json.load(open('data/dial/qa/ind_pilot.json'))['decision']['frame'])")
echo "$(date -u +%FT%TZ) l_ind frame $F (coordinator line: $(grep -Ec '\[coordinator\] IND I' logs/PROGRESS.md))" >> $LOG
python -I scripts/ext14_build_ladder.py --cells l_ind --ind-frame $F >> logs/ext14_base_ind_build.log 2>&1 || { echo "$(date -u +%FT%TZ) l_ind rebuild FAILED" >> $LOG; exit 1; }
for p in qwen3 mixtral_bos; do
  name=ext14-scan-$p-l_ind-$F
  bash scripts/gpu_queue.sh "$name" -- python scripts/ext14_scan.py $p --part cells --subset all --cells l_ind > "logs/$name.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) $name rc=$?" >> $LOG
  python scripts/ext14_casesets.py $p --cells l_ind >> logs/ext14_base_ind_build.log 2>&1
  echo "$(date -u +%FT%TZ) casesets l_ind $p rc=$?" >> $LOG
done
echo "$(date -u +%FT%TZ) l_ind ($F) registered" >> $LOG
