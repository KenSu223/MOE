#!/usr/bin/env bash
# ext2-model-zoo: filter -> sweep -> all-layer expert pass for one model under each of its protocols, serialised through
# the GPU queue. Waits for the model's download marker (logs/ext2_zoo_download.log.<key>.done) unless the checkpoint is
# already present. Stops at the first failing job. Logs: logs/ext2_zoo_<run>_<stage>.log; chain log: logs/ext2_zoo_chain.log
# Usage: bash scripts/ext2_zoo_chain.sh <model_key> [protocol ...]   (default: all protocols to run for the model)
set -uo pipefail
cd /home/ubuntu/MOE; . .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
key=$1; shift
CH=logs/ext2_zoo_chain.log
clog() { echo "$(date -u +%FT%TZ) chain[$key]: $*" >> "$CH"; }
marker="logs/ext2_zoo_download.log.$key.done"
repo=$(python -c "from moetrace.models import MODELS; print(MODELS['$key']['repo'])")
if [ ! -d "$HF_HOME/hub/models--${repo//\//--}/snapshots" ] || [ -n "$(ls "$HF_HOME/hub/models--${repo//\//--}/blobs" 2>/dev/null | grep incomplete)" ]; then
  clog "waiting for download marker $marker"
  while [ ! -f "$marker" ]; do sleep 30; done
fi
[ -f "data/model_usage/$key.json" ] || python scripts/ext2_zoo_usage.py "$key" >> "$CH" 2>&1
protos=${*:-$(python -c "from moetrace.ext2_zoo import protocols_to_run; print(' '.join(protocols_to_run('$key')))")}
clog "protocols: $protos"
job() {  # job <run> <stage> <cmd...>
  local run=$1 stage=$2; shift 2
  local logf="logs/ext2_zoo_${run}_${stage}.log"
  clog "launching $run $stage"
  bash scripts/gpu_queue.sh "ext2zoo-$run-$stage" -- "$@" > "$logf" 2>&1
  local rc=$?
  clog "$run $stage rc=$rc"
  [ $rc -eq 0 ] || { clog "ABORT after $run $stage (see $logf)"; exit $rc; }
}
for p in $protos; do
  run="${key}_${p}"
  [ -f "results/$run/case_sets.json" ] || job "$run" filter python scripts/ext2_zoo_filter.py "$key" --protocol "$p"
  [ -f "results/$run/sweep_summary.json" ] || job "$run" sweep python scripts/ext2_zoo_sweep.py "$key" --protocol "$p"
  job "$run" expert python scripts/ext2_zoo_expert.py "$key" --protocol "$p" --resume
done
clog "DONE"
