#!/usr/bin/env bash
# ext14 C2 RelP pilot chain (p6-circuit): OLMoE CounterFact pair build, RelP runs (OLMoE wino / ioi / cf, Mistral-7B wino / cf after the
# C1 Mistral score stages), verification of the functional forward on Mistral, analysis after every run. One queued GPU job at a time.
# Markers: results/p6ci_relp_<model>_<task>/.done_relp, results/p6ci_c3_data/.done_cf_olmoe_mix.
# Usage: nohup setsid bash scripts/ext14ci_c2_chain.sh > logs/ext14ci_c2_chain.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext14ci_c2_chain.log
N=${N_CASES:-48}
echo "$(date -u +%FT%TZ) c2 chain start pid=$$" >> $LOG
D=results/p6ci_c3_data; mkdir -p $D
if [ ! -f $D/.done_cf_olmoe_mix ]; then
  bash scripts/gpu_queue.sh p6ci-c2-cfbuild-olmoe -- python scripts/ext14ci_c3_cf_build.py olmoe --proto olmoe \
    --rels P17,P27,P495,P103,P178,P1412,P364,P136,P30,P101 --max-per-rel 150 --merge-as mix > logs/ext14ci_c2_cfbuild_olmoe.log 2>&1 < /dev/null
  rc=$?; echo "$(date -u +%FT%TZ) olmoe cf build rc=$rc" >> $LOG
  [ $rc -eq 0 ] && touch $D/.done_cf_olmoe_mix
fi
run() {  # model task
  local d=results/p6ci_relp_$1_$2
  [ -f $d/.done_relp ] && return 0
  bash scripts/gpu_queue.sh p6ci-c2-relp-$1-$2 -- python scripts/ext14ci_relp_run.py $1 $2 --n-cases $N > logs/ext14ci_relp_$1_$2.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) relp $1 $2 rc=$?" >> $LOG
  python scripts/ext14ci_relp_analyze.py >> logs/ext14ci_relp_analyze.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) analyze rc=$?" >> $LOG
}
run olmoe wino
[ -f $D/.done_cf_olmoe_mix ] && run olmoe cf
# C1 exact single-neuron oracle for the Qwen3 experts (768 neurons): the indirect L37E032 (F1) and the direct L44E069 / L41E117
for te in qwen3_f1:L37E032; do  # scope cut 2026-10-17: only the indirect expert (does a better-than-DLA ranking exist?)
  t=${te%%:*}; e=${te##*:}; d=results/p6ci_c1_$t
  while [ ! -f $d/.done_patch_$e ]; do sleep 300; done
  [ -f $d/.done_oracle_$e ] && continue
  bash scripts/gpu_queue.sh p6ci-c1-$t-$e-oracle -- python scripts/ext14ci_c1_run.py $t oracle --expert $e --n-rows 96 --max-spawn 30000 > logs/ext14ci_c1_${t}_${e}_oracle.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) oracle $t $e rc=$?" >> $LOG
  python scripts/ext14ci_c1_analyze.py >> logs/ext14ci_c1_analyze.log 2>&1 < /dev/null
done
# Mistral needs the C1 score stage (margin-filtered Mixtral items)
for t in wino cf; do
  while [ ! -f results/p6ci_c1_mistral7b_$t/.done_score ]; do sleep 300; done
  if [ ! -f results/p6ci_relp_verify_mistral7b.json ]; then
    bash scripts/gpu_queue.sh p6ci-c2-verify-mistral7b -- python scripts/ext14ci_relp_verify.py mistral7b $t > logs/ext14ci_relp_verify_mistral7b.log 2>&1 < /dev/null
    echo "$(date -u +%FT%TZ) verify mistral7b rc=$?" >> $LOG
  fi
  run mistral7b $t
done
echo "$(date -u +%FT%TZ) c2 chain done" >> $LOG
