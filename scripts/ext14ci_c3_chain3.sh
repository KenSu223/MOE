#!/usr/bin/env bash
# ext14 C3 chain v3 (p6-circuit), same stages as ext14ci_c3_chain2.sh but with fewer, larger queued jobs (gpu_queue priorities
# 2026-10-17: p6ci-* jobs run when no tier-1 / scan / engine job waits):
#   job 1  CounterFact relation pair build + merge + head singles (selection half) + expert DLA        (one queued job)
#   wait   Mixtral: e1_e2a head singles (NEED HEADS e1_e2a/mixtral_bos), at most 12 h
#   CPU    concept circuits from existing rows
#   job 2/3 validation (knockout matrix, add-back, edges), tasks split into two halves
#   CPU    report
# Usage: nohup setsid bash scripts/ext14ci_c3_chain3.sh <qwen3|mixtral_bos> > logs/ext14ci_c3_chain3_<proto>.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=172800 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
PROTO=$1
LOG=logs/ext14ci_c3_chain3_$PROTO.log
case $PROTO in qwen3) MODEL=qwen3; HL=28,38,39,40,42,43 ;; mixtral_bos) MODEL=mixtral; HL=13,15,16,18,19,24 ;; esac
D=results/p6ci_c3_data; R=results/p6ci_c3_cfc_$PROTO; mkdir -p $D $R
echo "$(date -u +%FT%TZ) c3 chain3 start proto=$PROTO pid=$$" >> $LOG
if [ ! -f $R/.done_dla ]; then
  bash scripts/gpu_queue.sh p6ci-c3-country-$PROTO -- bash -c "
    set -e
    [ -f $D/.done_country_$PROTO ] || { python scripts/ext14ci_c3_cf_build.py $MODEL --proto $PROTO --rels P17,P27,P495,P103 --max-per-rel 400 --cap 48; touch $D/.done_country_$PROTO; }
    python scripts/ext14ci_c3_country.py merge --proto $PROTO
    [ -f $R/.done_heads ] || { python scripts/ext7_wino_heads.py --model $MODEL --pairs $D/pairs_cfc_$PROTO.parquet --case-sets $D/case_sets_cfc_$PROTO.json \
        --out p6ci_c3_cfc_$PROTO --splits discovery --families main --layers $HL --max-rows 40000 --agent p6-circuit; touch $R/.done_heads; }
    python scripts/ext14ci_c3_country.py dla --proto $PROTO" > logs/ext14ci_c3_country_$PROTO.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) country job rc=$?" >> $LOG
fi
if [ "$PROTO" = mixtral_bos ]; then
  n=0
  while [ ! -f results/p6_e1_e2a_mixtral_bos_t2/head_rows.parquet ] && [ ! -f results/p6_e1_e2a_mixtral_bos/head_rows.parquet ] && [ $n -lt 144 ]; do
    sleep 300; n=$((n + 1))
  done
  echo "$(date -u +%FT%TZ) e1_e2a/mixtral_bos heads present: $(ls results/p6_e1_e2a_mixtral_bos*/head_rows.parquet 2>/dev/null | tr '\n' ' ')" >> $LOG
fi
python scripts/ext14ci_c3_existing.py --proto $PROTO >> logs/ext14ci_c3_existing_$PROTO.log 2>&1
echo "$(date -u +%FT%TZ) circuits rc=$?" >> $LOG
TODO=$(python -c "
import json, os
t = json.load(open('$D/circuits_$PROTO.json'))['tasks']
print(' '.join(x for x in t if not os.path.exists('$D/validate_${PROTO}__' + x + '.parquet')))")
set -- $TODO
n=$#; h=$(( (n + 1) / 2 ))
A=$(echo $TODO | tr ' ' '\n' | head -n $h | tr '\n' ' ')
B=$(echo $TODO | tr ' ' '\n' | tail -n +$((h + 1)) | tr '\n' ' ')
for grp in "$A" "$B"; do
  [ -z "$(echo $grp)" ] && continue
  name=$(echo $grp | tr ' ' '+')
  bash scripts/gpu_queue.sh p6ci-c3-val-$PROTO-$name -- bash -c "for t in $grp; do python scripts/ext14ci_c3_validate.py --proto $PROTO --tasks \$t --max-rows 2048 --max-spawn 16000 || exit 1; done" \
    > logs/ext14ci_c3_val_${PROTO}_$name.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) validate [$grp] rc=$?" >> $LOG
  python scripts/ext14ci_c3_existing.py --proto $PROTO --report >> logs/ext14ci_c3_existing_$PROTO.log 2>&1
done
python scripts/ext14ci_c3_existing.py --proto $PROTO --report >> logs/ext14ci_c3_existing_$PROTO.log 2>&1
echo "$(date -u +%FT%TZ) c3 chain3 done rc=$?" >> $LOG
