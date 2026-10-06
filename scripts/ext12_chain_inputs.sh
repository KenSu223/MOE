#!/usr/bin/env bash
# ext12 (Phase 4, Direction 12): GPU jobs that need no new driver, all through scripts/gpu_queue.sh, each skipped when its
# stage is already complete in the run's run_meta.json (resume from disk without redoing GPU work).
#   4c  Mixtral WITHOUT BOS on the 776-pair WinoGrande case set (main + rep): W2 sweep, W6 all-layer expert pass (+ equal-norm
#       rows at the auto layer), W4 joint (all-MoE / all-attention / all-block, both directions), direct-path split, sink flags
#   4b  option swap on the role-swap items (Qwen3, Mixtral BOS): W2 sweep, W4 joint, direct-path split
# Usage: nohup setsid bash scripts/ext12_chain_inputs.sh > logs/ext12_chain_inputs.log 2>&1 &
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
done_stage() {  # $1 run, $2 run_meta key; success if complete / completed_utc
  python3 - "$1" "$2" <<'PY'
import json, sys, os
p = f"/home/ubuntu/MOE/results/{sys.argv[1]}/run_meta.json"
m = json.load(open(p)) if os.path.exists(p) else {}
s = m.get(sys.argv[2], {})
sys.exit(0 if (s.get("complete") or s.get("completed_utc")) else 1)
PY
}
step() {  # $1 run, $2 stage key, $3 job name, rest = command
  local run=$1 key=$2 job=$3; shift 3
  if done_stage "$run" "$key"; then echo "$(date -u +%FT%TZ) skip $job (complete)"; return 0; fi
  echo "$(date -u +%FT%TZ) run $job"
  $Q "$job" -- "$@" || { echo "$(date -u +%FT%TZ) FAILED $job"; return 1; }
}
NB=data/wino_str/pairs_train_xl_mixtral_nobos.parquet
CS=data/wino_str/case_sets.json
S4=discovery,validation,rep_discovery,rep_validation
R=wino_mixtral_nobos_str

# ---- 4c inputs first (the Mixtral add-back of ext12_addback_run.py needs the expert rows)
step $R sweep ext12-4c-sweep python scripts/ext7_wino_sweep.py --model mixtral --pairs $NB --case-sets $CS --out $R --splits $S4 \
  --kinds layer,attn_layer,block --max-rows 40000 --agent ext12-complete || exit 1
step $R expert ext12-4c-expert python scripts/ext7_wino_expert.py --model mixtral --pairs $NB --case-sets $CS --out $R --families main,rep \
  --layers all --pairs-layers auto --max-spawn 40000 --agent ext12-complete || exit 1

# ---- 4b option swap on the role items
for proto in qwen3 mixtral_bos; do
  model=$proto; [ $proto = mixtral_bos ] && model=mixtral
  mr=60000; [ $model = mixtral ] && mr=40000
  RR=wino_roleitems_${proto}_str
  PP=data/wino_str/pairs_train_xl_${proto}.parquet
  CC=results/wino_roleitems_inputs/case_sets_${proto}.json
  step $RR sweep ext12-4b-sweep-$proto python scripts/ext7_wino_sweep.py --model $model --pairs $PP --case-sets $CC --out $RR \
    --kinds layer,attn_layer,block --max-rows $mr --agent ext12-complete || exit 1
  step $RR joint ext12-4b-joint-$proto python scripts/ext7_wino_joint.py --model $model --pairs $PP --case-sets $CC --out $RR \
    --families main,rep --max-rows 30000 --agent ext12-complete || exit 1
  step $RR direct_split ext12-4b-direct-$proto python -m moetrace.ext7_controls direct --model $model --pairs $PP --case-sets $CC --out $RR || exit 1
done

# ---- 4c W4 + sink flags
step $R joint ext12-4c-joint python scripts/ext7_wino_joint.py --model mixtral --pairs $NB --case-sets $CS --out $R --families main,rep \
  --max-rows 30000 --agent ext12-complete || exit 1
step $R direct_split ext12-4c-direct python -m moetrace.ext7_controls direct --model mixtral --pairs $NB --case-sets $CS --out $R --splits $S4 || exit 1
step $R sink_flags ext12-4c-sinkflags python scripts/ext12_sink_flags.py --out $R || exit 1
echo "$(date -u +%FT%TZ) EXT12 INPUT CHAIN DONE"
