#!/usr/bin/env bash
# ext8 A0-A3 on CounterFact STR (Qwen3, Mixtral BOS): resumable add-back driver, 5 passes per GPU-queue job so that
# other agents' jobs can interleave; then the analysis. Usage: bash scripts/ext8_addback_chain.sh [cf|wino|all]
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
WHAT=${1:-cf}

complete() {  # $1 = out run
  python -c "import json,sys; m=json.load(open('results/$1/run_meta.json')); sys.exit(0 if m.get('complete') else 1)" 2>/dev/null
}

drive() {  # $1 model, $2 task, $3 src run, $4 out run, rest = extra args
  local model=$1 task=$2 src=$3 out=$4; shift 4
  for i in $(seq 1 40); do
    complete "$out" && return 0
    $Q "ext8-addback-$out-$i" -- python scripts/ext8_addback_run.py "$model" --task "$task" --src-run "$src" --out "$out" --max-passes 5 "$@" || return 1
  done
  complete "$out"
}

if [ "$WHAT" = cf ] || [ "$WHAT" = all ]; then
  drive qwen3 cf qwen3_str qwen3_str_addback --pool 32 --shap-cases 54 --budget 45000 || { echo "CF QWEN3 FAILED"; exit 1; }
  drive mixtral cf mixtral_bos_str mixtral_bos_str_addback --pool 64 --beam-cases 64 --budget 50000 || { echo "CF MIXTRAL FAILED"; exit 1; }
  . .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
  python scripts/ext8_addback_analyze.py --runs cf_qwen3,cf_mixtral || echo "ANALYSIS FAILED"
  echo "CF ADDBACK CHAIN DONE"
fi
if [ "$WHAT" = wino ] || [ "$WHAT" = all ]; then
  drive qwen3 wino wino_qwen3_str wino_qwen3_str_addback --pairs data/wino_str/pairs_train_xl_qwen3.parquet --case-sets data/wino_str/case_sets.json \
    --pool 32 --shap-cases 64 --exact-cases 128 --noise-cases 128 --beam-cases 128 --budget 45000 || { echo "WINO QWEN3 FAILED"; exit 1; }
  drive mixtral wino wino_mixtral_bos_str wino_mixtral_bos_str_addback --pairs data/wino_str/pairs_train_xl_mixtral_bos.parquet \
    --case-sets data/wino_str/case_sets.json --pool 64 --shap-cases 128 --exact-cases 128 --noise-cases 128 --beam-cases 64 --budget 50000 || { echo "WINO MIXTRAL FAILED"; exit 1; }
  . .venv/bin/activate; export HF_HOME=/opt/dlami/nvme/hf
  python scripts/ext8_addback_analyze.py || echo "ANALYSIS FAILED"
  echo "WINO ADDBACK CHAIN DONE"
fi
