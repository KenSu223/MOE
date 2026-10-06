#!/usr/bin/env bash
# ext12 4a / 4c add-back chain: multi-task driver scripts/ext12_addback_run.py, 5 passes per GPU-queue job (other agents'
# jobs interleave), resumable from each task's addback_state.pkl. Qwen3 group (WinoGrande replication split + CounterFact
# fold swap) first; the Mixtral group (WinoGrande replication BOS + CounterFact fold swap BOS + WinoGrande no BOS main)
# waits until the no-BOS all-layer expert rows exist (scripts/ext12_chain_inputs.sh). A failed job is retried once with a
# smaller budget (memory). Usage: nohup setsid bash scripts/ext12_chain_addback.sh [qwen3|mixtral|all] > logs/ext12_chain_addback.log 2>&1 &
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
WHAT=${1:-all}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
group_complete() {
  /home/ubuntu/MOE/.venv/bin/python - "$1" <<'PY'
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace import ext12_complete as C
ok = True
for k in C.GROUPS[sys.argv[1]]:
    p = f"/home/ubuntu/MOE/results/{C.TASKS[k]['out']}/run_meta.json"
    ok &= os.path.exists(p) and bool(json.load(open(p)).get("complete"))
sys.exit(0 if ok else 1)
PY
}
drive() {  # $1 group, $2 budget, $3 fallback budget
  local g=$1 b=$2 fb=$3
  for i in $(seq 1 40); do
    group_complete "$g" && { echo "$(date -u +%FT%TZ) group $g complete"; return 0; }
    if ! $Q "ext12-addback-$g-$i" -- python scripts/ext12_addback_run.py "$g" --budget "$b" --max-passes 5; then
      echo "$(date -u +%FT%TZ) job failed, retrying with budget $fb"
      $Q "ext12-addback-$g-$i-retry" -- python scripts/ext12_addback_run.py "$g" --budget "$fb" --max-passes 5 || return 1
      b=$fb
    fi
  done
  group_complete "$g"
}
if [ "$WHAT" = qwen3 ] || [ "$WHAT" = all ]; then
  drive qwen3 45000 30000 || { echo "QWEN3 ADDBACK FAILED"; exit 1; }
fi
if [ "$WHAT" = mixtral ] || [ "$WHAT" = all ]; then
  until python3 -c "import json,sys; m=json.load(open('results/wino_mixtral_nobos_str/run_meta.json')); sys.exit(0 if m.get('expert',{}).get('complete') else 1)" 2>/dev/null; do
    sleep 60
  done
  drive mixtral 60000 44000 || { echo "MIXTRAL ADDBACK FAILED"; exit 1; }
fi
echo "$(date -u +%FT%TZ) EXT12 ADDBACK CHAIN DONE"
