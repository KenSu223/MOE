#!/usr/bin/env bash
# ext10 step 2 (needs the ext9 engine: multi steps attn_head / heads_experts): joint add-back over heads + experts.
# 1. step 1 for IOI (i) S2 -> IO on Qwen3 (attention-pole reference; single heads + DLA), so its rows can share the Qwen3
#    step-2 passes; 2. step 2 Qwen3 (CounterFact + WinoGrande + IOI rows in every pass), 3. step 2 Mixtral (CounterFact +
#    WinoGrande); 10 passes per GPU-queue job so that other agents' jobs can interleave (resumable state in
#    results/<model>_circuit/); then the analysis and the section.
# Usage: bash scripts/ext10_chain2.sh
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"

s1_complete() {  # $1 = key
  python3 - "$1" <<'PY'
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.ext10_circuit import out_dir
p = os.path.join(out_dir(sys.argv[1]), "heads_state.json")
st = json.load(open(p)) if os.path.exists(p) else None
sys.exit(0 if st and len(st["done"]) == len(st["chunks"]) else 1)
PY
}
complete() {  # $1 = model
  python3 -c "import json,sys; m=json.load(open('results/$1_circuit/run_meta.json')); sys.exit(0 if m.get('complete') else 1)" 2>/dev/null
}
drive() {  # $1 = model, rest = extra args
  local m=$1; shift
  for i in $(seq 1 8); do
    complete "$m" && break
    $Q "ext10-circuit-$m-$i" -- python scripts/ext10_circuit_run.py "$m" --max-passes 10 "$@" || echo "JOB FAILED: $m (job $i)"
  done
  complete "$m" || { echo "STEP2 $m INCOMPLETE"; return 1; }
  echo "STEP2 $m DONE"
}

for i in 1 2; do
  s1_complete ioi_qwen3 && break
  $Q "ext10-heads-ioi_qwen3" -- python scripts/ext10_heads_run.py ioi_qwen3 || echo "JOB FAILED: ioi_qwen3 step 1"
done
if s1_complete ioi_qwen3; then
  echo "STEP1 ioi_qwen3 DONE"
  drive qwen3 --tasks cf,wino,ioi --budget 100000 || exit 1
else
  echo "STEP1 ioi_qwen3 INCOMPLETE: Qwen3 step 2 without IOI"
  drive qwen3 --tasks cf,wino || exit 1
fi
drive mixtral --tasks cf,wino || exit 1
. .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
python scripts/ext10_heads_analyze.py || echo "STEP1 ANALYSIS FAILED"
python scripts/ext10_circuit_analyze.py || echo "ANALYSIS FAILED"
python scripts/ext10_circuit_text.py || echo "TEXT FAILED"
echo "EXT10 CHAIN2 DONE"
