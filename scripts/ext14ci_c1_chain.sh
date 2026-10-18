#!/usr/bin/env bash
# ext14 C1 chain (p6-circuit): score + per-expert neuron patch stages for every C1 task through the GPU queue (one small job at
# a time, Stage-1/2 jobs interleave), skipping finished stages (.done_score / .done_patch_<expert> in results/p6ci_c1_<task>/),
# then the CPU analysis after every stage. A failed patch job is retried once with 3/4 of the row budget.
# Usage: nohup setsid bash scripts/ext14ci_c1_chain.sh "<task ...>" > logs/ext14ci_c1_chain.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext14ci_c1_chain.log
TASKS="${1:-qwen3_cf qwen3_wino qwen3_f1 mixtral_wino mixtral_cf mistral7b_wino mistral7b_cf}"
echo "$(date -u +%FT%TZ) c1 chain start tasks=[$TASKS] pid=$$" >> $LOG

analyze() {
  python scripts/ext14ci_c1_analyze.py >> logs/ext14ci_c1_analyze.log 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) analyze rc=$?" >> $LOG
}

for task in $TASKS; do
  d=results/p6ci_c1_$task
  if [ ! -f $d/.done_score ]; then
    bash scripts/gpu_queue.sh p6ci-c1-$task-score -- python scripts/ext14ci_c1_run.py $task score > logs/ext14ci_c1_${task}_score.log 2>&1 < /dev/null
    echo "$(date -u +%FT%TZ) $task score rc=$?" >> $LOG
    [ -f $d/.done_score ] || continue
  fi
  case $task in
    qwen3_*) budget=30000 ;;
    mixtral_*) budget=24000 ;;
    *) budget=30000 ;;
  esac
  for e in $(python -c "from moetrace import ext14_circuit as C; print(' '.join(C.ename(l, e) for l, e in C.C1_TASKS['$task']['experts']))"); do
    [ -f $d/.done_patch_$e ] && continue
    for b in $budget $((budget * 3 / 4)); do
      bash scripts/gpu_queue.sh p6ci-c1-$task-$e -- python scripts/ext14ci_c1_run.py $task patch --expert $e --max-spawn $b > logs/ext14ci_c1_${task}_${e}.log 2>&1 < /dev/null
      rc=$?
      echo "$(date -u +%FT%TZ) $task $e patch rc=$rc budget=$b" >> $LOG
      [ $rc -eq 0 ] && break
    done
    analyze
  done
done
analyze
echo "$(date -u +%FT%TZ) c1 chain done" >> $LOG
