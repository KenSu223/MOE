#!/usr/bin/env bash
# Run the expert pass at the union of per-set discovery-selected layers and the paper's layer.
# Usage: auto_expert.sh <model> [out_name] [token_rule]
set -euo pipefail
cd /home/ubuntu/MOE; . .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
model=$1; out=${2:-$1}; rule=${3:-space}
layers=$(python - "$model" "$out" <<'PY'
import json, sys
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.models import MODELS
m, out = sys.argv[1], sys.argv[2]; s = json.load(open(f"results/{out}/sweep_summary.json"))
ls = set()
for k in ("paper", "strict", "relaxed"):
    if k in s: ls.add(s[k]["L_star_discovery"])
if MODELS[m]["paper_layer"] is not None: ls.add(MODELS[m]["paper_layer"])
print(",".join(str(x) for x in sorted(ls)))
PY
)
echo "expert pass layers for $model ($out, rule=$rule): $layers"
python scripts/run_expert.py "$model" --layers "$layers" --out "$out" --token-rule "$rule"
