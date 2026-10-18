#!/usr/bin/env bash
# ext14 C3 chain v2 (p6-circuit; scope cut 2026-10-17: existing rows + the COUNTRY CounterFact sets of NEED HEADS cf_country), one proto:
#   1 CounterFact STR pair sets P17 / P27 / P495 / P103 (prefill margin pass)      [GPU]
#   2 merge -> pairs_cfc_<proto> (selection / evaluation halves)                 [CPU]
#   3 head singles at fixed CounterFact attention layers, selection half only     [GPU]
#   4 per-expert DLA, selection half (one prefill pass)                          [GPU]
#   5 Mixtral only: wait for the e1_e2a head singles (NEED HEADS e1_e2a/mixtral_bos; at most 12 h, then go on without)
#   6 concept circuits from existing rows (scripts/ext14ci_c3_existing.py)      [CPU]
#   7 validation per task: knockout matrix, add-back, edges                      [GPU, one job per task]
#   8 report (tables, summary)                                                   [CPU]
# Markers in results/p6ci_c3_data/ (.done_country_<proto>) and results/p6ci_c3_cfc_<proto>/ (.done_heads, .done_dla).
# Usage: nohup setsid bash scripts/ext14ci_c3_chain2.sh <qwen3|mixtral_bos> > logs/ext14ci_c3_chain2_<proto>.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
PROTO=$1
LOG=logs/ext14ci_c3_chain2_$PROTO.log
case $PROTO in qwen3) MODEL=qwen3; HL=28,38,39,40,42,43 ;; mixtral_bos) MODEL=mixtral; HL=13,15,16,18,19,24 ;; esac
D=results/p6ci_c3_data; R=results/p6ci_c3_cfc_$PROTO; mkdir -p $D $R
echo "$(date -u +%FT%TZ) c3 chain2 start proto=$PROTO pid=$$" >> $LOG
if [ ! -f $D/.done_country_$PROTO ]; then
  bash scripts/gpu_queue.sh p6ci-c3-cfbuild-$PROTO -- python scripts/ext14ci_c3_cf_build.py $MODEL --proto $PROTO --rels P17,P27,P495,P103 \
    --max-per-rel 400 --cap 48 > logs/ext14ci_c3_cfbuild_$PROTO.log 2>&1 < /dev/null
  rc=$?; echo "$(date -u +%FT%TZ) cf build rc=$rc" >> $LOG
  [ $rc -eq 0 ] && touch $D/.done_country_$PROTO
fi
if [ -f $D/.done_country_$PROTO ]; then
  python scripts/ext14ci_c3_country.py merge --proto $PROTO >> $LOG 2>&1
  if [ ! -f $R/.done_heads ]; then
    bash scripts/gpu_queue.sh p6ci-c3-cfc-heads-$PROTO -- python scripts/ext7_wino_heads.py --model $MODEL --pairs $D/pairs_cfc_$PROTO.parquet \
      --case-sets $D/case_sets_cfc_$PROTO.json --out p6ci_c3_cfc_$PROTO --splits discovery --families main --layers $HL --max-rows 40000 \
      --agent p6-circuit > logs/ext14ci_c3_cfc_heads_$PROTO.log 2>&1 < /dev/null
    rc=$?; echo "$(date -u +%FT%TZ) cfc heads rc=$rc" >> $LOG
    [ $rc -eq 0 ] && touch $R/.done_heads
  fi
  if [ ! -f $R/.done_dla ]; then
    bash scripts/gpu_queue.sh p6ci-c3-cfc-dla-$PROTO -- python scripts/ext14ci_c3_country.py dla --proto $PROTO > logs/ext14ci_c3_cfc_dla_$PROTO.log 2>&1 < /dev/null
    echo "$(date -u +%FT%TZ) cfc dla rc=$?" >> $LOG
  fi
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
TASKS=$(python -c "import json; print(' '.join(json.load(open('$D/circuits_$PROTO.json'))['tasks']))")
for t in $TASKS; do
  f=$D/validate_${PROTO}__$t.parquet
  [ -f $f ] && continue
  bash scripts/gpu_queue.sh p6ci-c3-val-$PROTO-$t -- python scripts/ext14ci_c3_validate.py --proto $PROTO --tasks $t --max-rows 2048 --max-spawn 16000 \
    > logs/ext14ci_c3_val_${PROTO}_$t.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) validate $t rc=$?" >> $LOG
  python scripts/ext14ci_c3_existing.py --proto $PROTO --report >> logs/ext14ci_c3_existing_$PROTO.log 2>&1
done
python scripts/ext14ci_c3_existing.py --proto $PROTO --report >> logs/ext14ci_c3_existing_$PROTO.log 2>&1
echo "$(date -u +%FT%TZ) c3 chain2 done rc=$?" >> $LOG
