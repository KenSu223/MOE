#!/usr/bin/env bash
# p5-addback detached chain (2026-10-11): waits for "[p5-data] BASE READY <cond>/<proto>" lines (scripts/ext13ab_ready.py)
# and runs the ready add-back tasks of ONE model in shared passes, one queued job of <= ~25 min at a time
# (--max-passes), resumable; after every job the CPU analysis + figures are refreshed (flock logs/ext13ab_analyze.lock).
# A failed job (e.g. OOM) is retried with a smaller budget. Stops when ext13ab_ready.py prints DONE or after 40 h.
# Usage: setsid nohup bash scripts/ext13ab_chain.sh <qwen3|mixtral> > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
m=$1
export HF_HOME=/opt/dlami/nvme/hf
. .venv/bin/activate
LOG=logs/ext13ab_chain.log
if [ "$m" = qwen3 ]; then budget=45000; passes=25; mt=3; else budget=50000; passes=13; mt=2; fi
deadline=$(( $(date +%s) + 40*3600 ))
n=0
echo "$(date -u +%FT%TZ) [$m] chain start (budget $budget, max-passes $passes, max-tasks $mt)" >> $LOG
while [ "$(date +%s)" -lt "$deadline" ]; do
  st=$(python scripts/ext13ab_ready.py $m 2>>$LOG)
  case "$st" in
    RUN*)
      keys=${st#RUN }
      n=$((n+1))
      job=ext13ab-$m-$n
      echo "$(date -u +%FT%TZ) [$m] job $job tasks $keys (budget $budget)" >> $LOG
      bash scripts/gpu_queue.sh $job -- python scripts/ext13ab_run.py $m --tasks "$keys" --max-passes $passes --budget $budget --max-tasks $mt > logs/$job.log 2>&1
      rc=$?
      echo "$(date -u +%FT%TZ) [$m] $job rc=$rc" >> $LOG
      if [ $rc -ne 0 ]; then
        budget=$(( budget * 3 / 4 )); [ $budget -lt 15000 ] && budget=15000
        echo "$(date -u +%FT%TZ) [$m] job failed; budget -> $budget" >> $LOG
        sleep 60
      fi
      ( flock -x 8; python -W ignore scripts/ext13ab_analyze.py > logs/ext13ab_analyze.log 2>&1 && python -W ignore scripts/ext13ab_plot.py >> logs/ext13ab_analyze.log 2>&1; echo "$(date -u +%FT%TZ) [$m] analysis refreshed rc=$?" >> $LOG ) 8>logs/ext13ab_analyze.lock
      ;;
    DONE*)
      echo "$(date -u +%FT%TZ) [$m] all conditions complete" >> $LOG
      touch logs/ext13ab_chain_$m.done
      exit 0;;
    *)
      sleep 180;;
  esac
done
echo "$(date -u +%FT%TZ) [$m] deadline reached" >> $LOG
