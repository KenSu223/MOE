#!/usr/bin/env bash
# ext7-wino chain 2 (resume of chain 1 after the Mixtral sweep OOM; Phase 3 W2, W6, W3, W5 on WinoGrande STR pairs; Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS).
# Every GPU step goes through scripts/gpu_queue.sh (shared A10G). Gate: results/verify_ext7_wino_olmoe.json must pass.
# Usage: nohup bash scripts/ext7_wino_chain2.sh > logs/ext7_wino_chain2.log 2>&1 &
set -uo pipefail
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
CS=data/wino_str/case_sets.json
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date -u +%FT%TZ; }
step() { local name=$1; shift; echo "$(ts) [chain2] START $name"; $Q "$name" -- "$@"; local rc=$?; echo "$(ts) [chain2] END $name rc=$rc"; return $rc; }

step ext7-wino-verify4 python scripts/ext7_wino_verify.py || exit 1
# gate 1 (final-position kinds layer / attn_layer / block / resid / attn_head vs transformers hooks on OLMoE)
python -c "
import json,sys; v=json.load(open('results/verify_ext7_wino_olmoe.json'))
ok = all(v[f'final_{k}']['meanabsdiff'] <= 0.1 and v[f'final_{k}']['frac_within_0.25'] >= 0.95 and v[f'final_{k}']['rescue_corr_vs_hf'] >= 0.9
         for k in ('layer','attn_layer','block','resid','attn_head'))
sys.exit(0 if ok else 1)" || { echo "$(ts) [chain2] verification gate 1 FAILED"; exit 1; }

args_of() {  # model key -> common args
  case $1 in
    qwen3) echo "--model qwen3 --pairs data/wino_str/pairs_train_xl_qwen3.parquet --case-sets $CS --out wino_qwen3_str" ;;
    mixtral_bos) echo "--model mixtral --pairs data/wino_str/pairs_train_xl_mixtral_bos.parquet --case-sets $CS --out wino_mixtral_bos_str" ;;
  esac
}

# W2 final-position sweep (families main, rep, own pool)
for m in mixtral_bos; do
  mr=40000
  step ext7-wino-sweep-$m python scripts/ext7_wino_sweep.py $(args_of $m) --own-pool $m --max-rows $mr || exit 1
done
# W6 all-layer expert pass (families main, rep) with equal-norm pairs at the discovery MoE argmax
for m in qwen3 mixtral_bos; do
  ms=60000; [ $m = mixtral_bos ] && ms=40000
  step ext7-wino-expert-$m python scripts/ext7_wino_expert.py $(args_of $m) --families main,rep --layers all --pairs-layers auto --max-spawn $ms || exit 1
done
python - <<'EOF' && echo "- $(ts) [ext7-wino] EXPERT ROWS READY (ext7-wino): results/wino_qwen3_str/str_expert_rows.parquet, results/wino_mixtral_bos_str/str_expert_rows.parquet (schema of results/qwen3_str/str_expert_rows.parquet; case_id = directed id 2*pair_idx+d, slot 0; families in <run>/case_sets.json: main = 128/128 pairs, rep = 128/128 pairs; sweep_cases.parquet has delta_clean/delta_corrupt/drop; routing in str_sweep_routing.parquet)" >> logs/PROGRESS.md
import pandas as pd, json
for run in ("wino_qwen3_str", "wino_mixtral_bos_str"):
    e = pd.read_parquet(f"results/{run}/str_expert_rows.parquet")
    fam = json.load(open(f"results/{run}/case_sets.json"))
    ids = set(fam["main"]["discovery"] + fam["main"]["validation"] + fam["rep"]["discovery"] + fam["rep"]["validation"])
    assert set(e.case_id) >= ids and e.rescue.notna().all(), run
    print(run, len(e), e.layer.nunique(), e.kind.value_counts().to_dict())
EOF
# gate 2 (suffix executor at the STR positions; re-run of the verification with window-5 rows in a separate pass)
for i in $(seq 1 360); do
  python -c "import json,sys; v=json.load(open('results/verify_ext7_wino_olmoe.json')); sys.exit(0 if 'max_meanabsdiff' in v else 1)" && break
  sleep 10
done
python -c "import json,sys; v=json.load(open('results/verify_ext7_wino_olmoe.json')); sys.exit(0 if v.get('pass') else 1)" \
  || { echo "$(ts) [chain2] verification gate 2 FAILED"; exit 1; }
# W3 position x layer grid, window 1 (kinds layer, attn_layer, resid), family main
for m in qwen3 mixtral_bos; do
  step ext7-wino-grid-w1-$m python scripts/ext7_wino_grid.py $(args_of $m) --families main --kinds layer,attn_layer,resid --window 1 --max-rows 60000 --token-budget 200000 || exit 1
done
# W5 heads at the top-4 attention layers of W2 plus one null layer
for m in qwen3 mixtral_bos; do
  step ext7-wino-heads-$m python scripts/ext7_wino_heads.py $(args_of $m) --families main --layers auto --max-rows 50000 || exit 1
done
# W3 window 5 (MoE output only)
for m in qwen3 mixtral_bos; do
  step ext7-wino-grid-w5-$m python scripts/ext7_wino_grid.py $(args_of $m) --families main --kinds layer --window 5 --max-rows 60000 --token-budget 200000 || exit 1
done
echo "$(ts) [chain2] DONE"
