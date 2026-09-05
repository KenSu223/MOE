#!/usr/bin/env bash
# Remaining work after the overnight agent stopped: Mixtral no-BOS run, Mixtral paper_like token-rule run, HF reference checks.
set -uo pipefail
cd /home/ubuntu/MOE; . .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
log(){ echo "$(date -u +%T) $*"; }
layers_for(){ python - "$1" <<'PY'
import json, sys
s = json.load(open(f"results/{sys.argv[1]}/sweep_summary.json"))
ls = {19}
if "paper" in s: ls.add(s["paper"]["L_star_discovery"])
print(",".join(str(x) for x in sorted(ls)))
PY
}
log "chain3 start"
# 1) Mixtral without BOS (special tokens off), paper case set
mkdir -p results/mixtral_nobos; cp results/mixtral/case_sets.json results/mixtral_nobos/
python scripts/run_sweep.py mixtral --sets paper --out mixtral_nobos --no-special-tokens > logs/sweep_mixtral_nobos.log 2>&1; log "mixtral nobos sweep rc=$?"
L=$(layers_for mixtral_nobos); log "mixtral nobos expert layers: $L"
python scripts/run_expert.py mixtral --layers "$L" --out mixtral_nobos --no-special-tokens > logs/expert_mixtral_nobos.log 2>&1; log "mixtral nobos expert rc=$?"
# 2) Mixtral with the paper_like object-token rule (BOS on), paper case set
mkdir -p results/mixtral_alt; cp results/mixtral/case_sets.json results/mixtral_alt/
python scripts/run_sweep.py mixtral --sets paper --out mixtral_alt --token-rule paper_like > logs/sweep_mixtral_alt.log 2>&1; log "mixtral alt sweep rc=$?"
L=$(layers_for mixtral_alt); log "mixtral alt expert layers: $L"
python scripts/run_expert.py mixtral --layers "$L" --out mixtral_alt --token-rule paper_like > logs/expert_mixtral_alt.log 2>&1; log "mixtral alt expert rc=$?"
# 3) HF reference checks (slow; offload)
for m in qwen3 mixtral; do
  timeout 5400 python scripts/hf_reference_check.py $m 5 > logs/hf_ref_$m.log 2>&1; log "hf ref $m rc=$?"
done
log "chain3 done"
