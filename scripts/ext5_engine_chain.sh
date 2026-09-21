#!/usr/bin/env bash
# ext5-engine GPU chain (Phase 2 wave 1: F5 metrics runs, F2 head sweeps, F1.3 exhaustive subsets).
# Every step goes through scripts/gpu_queue.sh (shared lock, venv, HF_HOME). Launch detached:
#   setsid nohup bash scripts/ext5_engine_chain.sh > logs/ext5_engine_chain.log 2>&1 &
# Steps whose outputs exist are skipped (idempotent re-launch). Verification is NOT part of this chain (run first).
set -uo pipefail
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
R=results

step() {  # step <name> <done-file> <command...>
  local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date -u +%FT%TZ) skip $name ($done exists)"; return 0; fi
  echo "$(date -u +%FT%TZ) >>> $name"
  $Q "ext5eng-$name" -- "$@"; local rc=$?
  echo "$(date -u +%FT%TZ) <<< $name rc=$rc"
  return $rc
}

# ---- F5: base layer sweeps with metrics (paper set) + expert passes at the two candidate layers, no pairs
for spec in "qwen3|qwen3_metrics|qwen3||42,44" "mixtral|mixtral_nobos_metrics|mixtral_nobos|--no-special-tokens|18,19" "mixtral|mixtral_bos_metrics|mixtral||18,19"; do
  IFS='|' read -r model run base flag layers <<< "$spec"
  mkdir -p $R/$run
  [ -e $R/$run/case_sets.json ] || cp $R/$base/case_sets.json $R/$run/case_sets.json
  step "sweep-$run" $R/$run/sweep_summary.json python scripts/run_sweep.py $model --out $run --sets paper --metrics --agent ext5-engine $flag
  step "expert-$run" $R/$run/expert_rows.parquet python scripts/run_expert.py $model --out $run --layers $layers --no-pairs --metrics --agent ext5-engine $flag
done

# ---- F2: head sweeps at the attention peaks (+ MoE peak as a null)
step "heads-qwen3" $R/qwen3_heads/head_rows.parquet python scripts/ext5_heads_sweep.py qwen3 --out qwen3_heads --layers 40,43,44 --base-run qwen3
step "heads-mixtral_nobos" $R/mixtral_nobos_heads/head_rows.parquet python scripts/ext5_heads_sweep.py mixtral --out mixtral_nobos_heads --layers 15,18,19,24 --base-run mixtral_nobos --no-special-tokens
step "heads-mixtral_bos" $R/mixtral_bos_heads/head_rows.parquet python scripts/ext5_heads_sweep.py mixtral --out mixtral_bos_heads --layers 15,18,19,24 --base-run mixtral

# ---- F1.3: exhaustive clean-active subsets (one pass per layer; routing from the metrics runs, same 256-case batch)
step "subsets-qwen3-L44" $R/qwen3_subsets/subset_prefill_L44.parquet python scripts/ext5_subsets_run.py qwen3 --out qwen3_subsets --layers 44 --base-run qwen3_metrics
step "subsets-qwen3-L42" $R/qwen3_subsets/subset_prefill_L42.parquet python scripts/ext5_subsets_run.py qwen3 --out qwen3_subsets --layers 42 --base-run qwen3_metrics
step "subsets-mixtral_nobos" $R/mixtral_nobos_subsets/subset_prefill_L19.parquet python scripts/ext5_subsets_run.py mixtral --out mixtral_nobos_subsets --layers 18,19 --base-run mixtral_nobos_metrics --no-special-tokens
echo "$(date -u +%FT%TZ) chain finished"
