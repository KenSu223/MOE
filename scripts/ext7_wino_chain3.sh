#!/usr/bin/env bash
# ext7-wino chain 3: W4 joint sublayer decomposition (ext8 multi steps, after ENGINE READY (ext8)) and the direct-path / DLA
# split of the final residual, Qwen3 and Mixtral BOS, families main + rep. OLMoE smoke test first.
# Usage: nohup bash scripts/ext7_wino_chain3.sh > logs/ext7_wino_chain3.log 2>&1 &
set -uo pipefail
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
CS=data/wino_str/case_sets.json
. .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date -u +%FT%TZ; }
step() { local name=$1; shift; echo "$(ts) [chain3] START $name"; $Q "$name" -- "$@"; local rc=$?; echo "$(ts) [chain3] END $name rc=$rc"; return $rc; }
SM="--model olmoe --pairs data/wino_str/pairs_train_xl_olmoe.parquet --case-sets /tmp/claude-1000/-home-ubuntu-MOE/488f7156-7059-40cb-9629-447467831e6b/scratchpad/ext7w/olmoe_smoke_cs.json --out /tmp/claude-1000/-home-ubuntu-MOE/488f7156-7059-40cb-9629-447467831e6b/scratchpad/ext7w/wino_olmoe_smoke"
step ext7-wino-smoke-joint python scripts/ext7_wino_joint.py $SM --families main,rep || exit 1
python - <<'PY' || { echo "$(ts) [chain3] smoke sanity FAILED"; exit 1; }
import pandas as pd
d = pd.read_parquet("/tmp/claude-1000/-home-ubuntu-MOE/488f7156-7059-40cb-9629-447467831e6b/scratchpad/ext7w/wino_olmoe_smoke/joint_rows.parquet")
b = d[d.kind == "all_block"]
a = d[d.kind == "all_attn"]
print("all_block |effect/drop - 1| max", float((b.effect / b["drop"] - 1).abs().max()), "all_attn", float((a.effect / a["drop"] - 1).abs().max()))
assert (b.effect / b["drop"] - 1).abs().max() < 1e-6
PY
args_of() {
  case $1 in
    qwen3) echo "--model qwen3 --pairs data/wino_str/pairs_train_xl_qwen3.parquet --case-sets $CS --out wino_qwen3_str" ;;
    mixtral_bos) echo "--model mixtral --pairs data/wino_str/pairs_train_xl_mixtral_bos.parquet --case-sets $CS --out wino_mixtral_bos_str" ;;
  esac
}
for m in qwen3 mixtral_bos; do
  step ext7-wino-joint-$m python scripts/ext7_wino_joint.py $(args_of $m) --families main,rep --max-rows 30000 || exit 1
  step ext7-wino-dla-$m python scripts/ext7_wino_dla.py $(args_of $m) --families main,rep --chunk 512 || exit 1
done
echo "$(ts) [chain3] DONE"
