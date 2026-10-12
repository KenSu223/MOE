#!/usr/bin/env bash
# ext13 Phase 5 GPU chain 1 (2026-10-11): F2R + verb attr2ent (F2R) scans, then the P5-1 F1 analyses for Qwen3
# (runners of Phase 3, generic STR pair files). Mixtral F1 analyses follow in chain 2 once its case sets exist.
cd "$(dirname "$0")/.."
q() { name=$1; shift; bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1; echo "$(date -u +%FT%TZ) $name rc=$?" >> logs/ext13_chain.log; }
for p in qwen3 mixtral_bos; do
  q ext13-scan-f2r-$p python scripts/ext13_scan.py $p --formats F2R
  q ext13-verbs-a2e-$p python scripts/ext13_scan.py $p --verbs --verb-dirs attr2ent
done
P="--model qwen3 --pairs data/wino_ref/pairs_F1_train_xl_qwen3.parquet --case-sets data/wino_ref/case_sets_F1_qwen3.json --out wino_ref_F1_qwen3"
q ext13-f1-sweep-qwen3  python scripts/ext7_wino_sweep.py $P --max-rows 60000
q ext13-f1-joint-qwen3  python scripts/ext7_wino_joint.py $P --families main,rep --max-rows 30000
q ext13-f1-dla-qwen3    python scripts/ext7_wino_dla.py $P --families main,rep --chunk 512
q ext13-f1-direct-qwen3 python -m moetrace.ext7_controls direct $P --splits discovery,validation,rep_discovery,rep_validation
q ext13-f1-expert-qwen3 python scripts/ext7_wino_expert.py $P --families main,rep --layers all --pairs-layers auto --max-spawn 60000
q ext13-f1-heads-qwen3  python scripts/ext7_wino_heads.py $P --families main --layers auto --max-rows 50000
q ext13-f1-grid-qwen3   python scripts/ext7_wino_grid.py $P --families main --kinds layer,attn_layer,resid --window 1 --max-rows 60000 --token-budget 200000
