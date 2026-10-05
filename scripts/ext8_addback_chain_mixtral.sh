#!/usr/bin/env bash
# ext8: extra driver chain for the two Mixtral BOS add-back runs (CounterFact, WinoGrande), started while the cf / wino chains
# (scripts/ext8_addback_chain.sh) are still on Qwen3. Jobs of all chains go through the same GPU lock and each job loads the
# run's state from disk, so two chains driving the same run interleave safely (each job continues where the last one stopped).
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
complete() { python -c "import json,sys; m=json.load(open('results/$1/run_meta.json')); sys.exit(0 if m.get('complete') else 1)" 2>/dev/null; }
drive() {
  local model=$1 task=$2 src=$3 out=$4; shift 4
  for i in $(seq 1 40); do
    complete "$out" && return 0
    $Q "ext8-addback-$out-m$i" -- python scripts/ext8_addback_run.py "$model" --task "$task" --src-run "$src" --out "$out" --max-passes 5 "$@" || return 1
  done
  complete "$out"
}
drive mixtral cf mixtral_bos_str mixtral_bos_str_addback --pool 64 --beam-cases 64 --budget 50000 || { echo "CF MIXTRAL FAILED"; exit 1; }
drive mixtral wino wino_mixtral_bos_str wino_mixtral_bos_str_addback --pairs data/wino_str/pairs_train_xl_mixtral_bos.parquet \
  --case-sets data/wino_str/case_sets.json --pool 64 --shap-cases 128 --exact-cases 128 --noise-cases 128 --beam-cases 64 --budget 50000 || { echo "WINO MIXTRAL FAILED"; exit 1; }
echo "MIXTRAL ADDBACK CHAIN DONE"
