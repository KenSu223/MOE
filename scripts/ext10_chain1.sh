#!/usr/bin/env bash
# ext10 step 1: single-head patches at every (layer, head) + head / expert DLA on the ext8 validation rows, then the
# analysis. One GPU-queue job per run (resumable: finished chunks are skipped). Usage: bash scripts/ext10_chain1.sh [keys]
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
KEYS=${1:-cf_qwen3,wino_qwen3,cf_mixtral,wino_mixtral}

complete() {  # $1 = key
  python3 - "$1" <<'PY'
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.ext10_circuit import out_dir
p = os.path.join(out_dir(sys.argv[1]), "heads_state.json")
st = json.load(open(p)) if os.path.exists(p) else None
sys.exit(0 if st and len(st["done"]) == len(st["chunks"]) else 1)
PY
}

for k in ${KEYS//,/ }; do
  extra=""
  [ "$k" = wino_mixtral ] && extra="--rows-per-pass 128"
  for i in 1 2 3; do
    complete "$k" && break
    $Q "ext10-heads-$k" -- python scripts/ext10_heads_run.py "$k" $extra || echo "JOB FAILED: $k (attempt $i)"
  done
  complete "$k" || { echo "STEP1 $k INCOMPLETE"; exit 1; }
  echo "STEP1 $k DONE"
done
. .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
python scripts/ext10_heads_analyze.py || echo "ANALYSIS FAILED"
echo "EXT10 CHAIN1 DONE"
