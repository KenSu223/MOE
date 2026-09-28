#!/usr/bin/env bash
# ext6 STR chain (2026-09-28). Phase A, per protocol: donor filter -> layer sweep -> expert pass at the key layers
# (paper layer, named loci, STR discovery L* and top-3 discovery layers; Mixtral with equal-norm pairs).
# Phase B: expert pass at every remaining layer (joint layer x expert search). Then the CPU analysis.
# Every GPU job goes through scripts/gpu_queue.sh. Log: logs/ext6_str_chain.log
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
Q="bash scripts/gpu_queue.sh"
MS=25000

keylayers() {  # run, fixed layers
  python - "$1" "$2" <<'EOF'
import json, sys
s = json.load(open(f"results/{sys.argv[1]}/str_sweep_summary.json"))
ls = {int(x) for x in sys.argv[2].split(",")} | {s["L_star_discovery"]} | {int(l) for l, _ in s["disc_top5"][:3]}
print(",".join(map(str, sorted(ls))))
EOF
}
rest() {  # n_layers, key layers
  python -c "import sys; k={int(x) for x in sys.argv[2].split(',')}; print(','.join(str(l) for l in range(int(sys.argv[1])) if l not in k))" "$1" "$2"
}

phase_a() {  # model run fixed pairs [flags...]
  local model=$1 run=$2 fixed=$3 pairs=$4; shift 4
  $Q ext6-str-filter-$run -- python scripts/ext6_str_filter.py $model --out $run "$@" || return 1
  $Q ext6-str-sweep-$run -- python scripts/ext6_str_sweep.py $model --out $run --layer-chunks 2 "$@" || return 1
  local kl; kl=$(keylayers $run $fixed)
  echo "$run key layers: $kl"
  echo "$kl" > results/$run/str_key_layers.txt
  $Q ext6-str-expert-key-$run -- python scripts/ext6_str_expert.py $model --out $run --layers $kl $pairs --max-spawn $MS "$@" || return 1
}
phase_b() {  # model run n_layers [flags...]
  local model=$1 run=$2 L=$3; shift 3
  local rl; rl=$(rest $L "$(cat results/$run/str_key_layers.txt)")
  $Q ext6-str-expert-rest-$run -- python scripts/ext6_str_expert.py $model --out $run --layers $rl --max-spawn $MS "$@" || return 1
}

echo "$(date -u +%FT%TZ) chain start"
phase_a qwen3 qwen3_str 42,44 "" || echo "FAILED phase A qwen3_str"
phase_a mixtral mixtral_nobos_str 18,19 --pairs --no-special-tokens || echo "FAILED phase A mixtral_nobos_str"
phase_a mixtral mixtral_bos_str 18,19 --pairs || echo "FAILED phase A mixtral_bos_str"
python scripts/ext6_str_analyze.py > logs/ext6_str_analyze_A.log 2>&1 || echo "analysis A failed"
echo "$(date -u +%FT%TZ) phase A done"
phase_b qwen3 qwen3_str 48 || echo "FAILED phase B qwen3_str"
phase_b mixtral mixtral_nobos_str 32 --no-special-tokens || echo "FAILED phase B mixtral_nobos_str"
phase_b mixtral mixtral_bos_str 32 || echo "FAILED phase B mixtral_bos_str"
python scripts/ext6_str_analyze.py > logs/ext6_str_analyze.log 2>&1 || echo "analysis failed"
echo "$(date -u +%FT%TZ) chain done"
