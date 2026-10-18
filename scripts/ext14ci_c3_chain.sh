#!/usr/bin/env bash
# ext14 C3 chain (p6-circuit), one proto: CounterFact relation pair sets -> task subsets -> per-task sweep (all layers, MoE + attention)
# -> common head / expert layers -> per-task expert singles + head singles -> C3 analysis -> validation runs (knockout matrix,
# add-back, edges) -> analysis. Every GPU job goes through scripts/gpu_queue.sh, one at a time; stages are skipped by
# .done_<stage> markers in results/p6ci_c3_<task>_<proto>/ (and results/p6ci_c3_data/.done_cf_<proto>).
# Tasks whose source pair file is missing (Phase-6 cells not built yet) are skipped; re-run the chain later to add them.
# Usage: nohup setsid bash scripts/ext14ci_c3_chain.sh <qwen3|mixtral_bos> > logs/ext14ci_c3_chain_<proto>.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
PROTO=$1
LOG=logs/ext14ci_c3_chain_$PROTO.log
case $PROTO in qwen3) MODEL=qwen3; ST="" ;; mixtral_bos) MODEL=mixtral; ST="" ;; esac
echo "$(date -u +%FT%TZ) c3 chain start proto=$PROTO pid=$$" >> $LOG
D=results/p6ci_c3_data
mkdir -p $D
if [ ! -f $D/.done_cf_$PROTO ]; then
  bash scripts/gpu_queue.sh p6ci-c3-cfbuild-$PROTO -- python scripts/ext14ci_c3_cf_build.py $MODEL --proto $PROTO > logs/ext14ci_c3_cfbuild_$PROTO.log 2>&1 < /dev/null
  rc=$?; echo "$(date -u +%FT%TZ) cf build rc=$rc" >> $LOG
  [ $rc -eq 0 ] && touch $D/.done_cf_$PROTO
fi
TASKS=$(python -c "
from moetrace import ext14_circuit_c3 as C3
out = []
for t in C3.TASK_ORDER:
    r = C3.build_subset(t, '$PROTO')
    if r and r['validation'] >= C3.MIN_VAL:
        out.append(t)
print(' '.join(out))")
echo "$(date -u +%FT%TZ) tasks: $TASKS" >> $LOG
budget() {  # rows per pass scaled by the task's longest prompt
  python -c "
import pandas as pd, json
p = pd.read_parquet('$D/pairs_$1_$PROTO.parquet'); T = max(len(json.loads(x)) for x in p.ids_a)
base = {'qwen3': $2, 'mixtral_bos': int($2 * 2 / 3)}['$PROTO']
print(int(base * min(1.0, 30.0 / T)))"
}
for t in $TASKS; do
  run=p6ci_c3_${t}_$PROTO; r=results/$run; mkdir -p $r
  if [ ! -f $r/.done_sweep ]; then
    b=$(budget $t 60000)
    bash scripts/gpu_queue.sh p6ci-c3-$t-$PROTO-sweep -- python scripts/ext7_wino_sweep.py --model $MODEL --pairs $D/pairs_${t}_$PROTO.parquet \
      --case-sets $D/case_sets_${t}_$PROTO.json --out $run --agent p6-circuit --kinds layer,attn_layer --max-rows $b > logs/ext14ci_c3_${t}_${PROTO}_sweep.log 2>&1 < /dev/null
    rc=$?; echo "$(date -u +%FT%TZ) $t sweep rc=$rc" >> $LOG
    [ $rc -eq 0 ] && touch $r/.done_sweep
  fi
done
python scripts/ext14ci_c3_analyze.py --proto $PROTO --layers-only >> logs/ext14ci_c3_analyze.log 2>&1
LE=$(python -c "import json; print(','.join(map(str, json.load(open('$D/layers_$PROTO.json'))['experts'])))")
LH=$(python -c "import json; print(','.join(map(str, json.load(open('$D/layers_$PROTO.json'))['heads'])))")
echo "$(date -u +%FT%TZ) layers experts=[$LE] heads=[$LH]" >> $LOG
for t in $TASKS; do
  run=p6ci_c3_${t}_$PROTO; r=results/$run
  [ -f $r/.done_sweep ] || continue
  if [ ! -f $r/.done_expert ]; then
    b=$(budget $t 40000)
    bash scripts/gpu_queue.sh p6ci-c3-$t-$PROTO-expert -- python scripts/ext7_wino_expert.py --model $MODEL --pairs $D/pairs_${t}_$PROTO.parquet \
      --case-sets $D/case_sets_${t}_$PROTO.json --out $run --agent p6-circuit --layers $LE --families main --max-spawn $b > logs/ext14ci_c3_${t}_${PROTO}_expert.log 2>&1 < /dev/null
    rc=$?; echo "$(date -u +%FT%TZ) $t expert rc=$rc" >> $LOG
    [ $rc -eq 0 ] && touch $r/.done_expert
  fi
  if [ ! -f $r/.done_heads ]; then
    b=$(budget $t 40000)
    bash scripts/gpu_queue.sh p6ci-c3-$t-$PROTO-heads -- python scripts/ext7_wino_heads.py --model $MODEL --pairs $D/pairs_${t}_$PROTO.parquet \
      --case-sets $D/case_sets_${t}_$PROTO.json --out $run --agent p6-circuit --layers $LH --families main --max-rows $b > logs/ext14ci_c3_${t}_${PROTO}_heads.log 2>&1 < /dev/null
    rc=$?; echo "$(date -u +%FT%TZ) $t heads rc=$rc" >> $LOG
    [ $rc -eq 0 ] && touch $r/.done_heads
  fi
done
python scripts/ext14ci_c3_analyze.py --proto $PROTO >> logs/ext14ci_c3_analyze.log 2>&1
echo "$(date -u +%FT%TZ) analyze (circuits) rc=$?" >> $LOG
if [ -f $D/circuits_$PROTO.json ] && [ ! -f $D/.done_validate_$PROTO ]; then
  bash scripts/gpu_queue.sh p6ci-c3-validate-$PROTO -- python scripts/ext14ci_c3_validate.py --proto $PROTO > logs/ext14ci_c3_validate_$PROTO.log 2>&1 < /dev/null
  rc=$?; echo "$(date -u +%FT%TZ) validate rc=$rc" >> $LOG
  [ $rc -eq 0 ] && touch $D/.done_validate_$PROTO
fi
python scripts/ext14ci_c3_analyze.py --proto $PROTO >> logs/ext14ci_c3_analyze.log 2>&1
echo "$(date -u +%FT%TZ) c3 chain done rc=$?" >> $LOG
