#!/usr/bin/env bash
# ext13 Phase 5 GPU chain 2 (2026-10-11): P5-1 F1 analyses for Mixtral BOS, after its full F1 scan (case sets built here).
cd "$(dirname "$0")/.."
until grep -q "ext13-f1full-mixtral_bos rc=" logs/ext13_chain.log; do sleep 30; done
grep -q "ext13-f1full-mixtral_bos rc=0" logs/ext13_chain.log || { echo "$(date -u +%FT%TZ) chain2 abort: Mixtral F1 scan failed" >> logs/ext13_chain.log; exit 1; }
. .venv/bin/activate
python scripts/ext13_casesets.py mixtral_bos --fmt F1 >> logs/ext13_casesets.log 2>&1 || exit 1
q() { name=$1; shift; bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1; echo "$(date -u +%FT%TZ) $name rc=$?" >> logs/ext13_chain.log; }
P="--model mixtral --pairs data/wino_ref/pairs_F1_train_xl_mixtral_bos.parquet --case-sets data/wino_ref/case_sets_F1_mixtral_bos.json --out wino_ref_F1_mixtral_bos"
q ext13-f1-sweep-mixtral  python scripts/ext7_wino_sweep.py $P --max-rows 40000
q ext13-f1-joint-mixtral  python scripts/ext7_wino_joint.py $P --families main,rep --max-rows 30000
q ext13-f1-dla-mixtral    python scripts/ext7_wino_dla.py $P --families main,rep --chunk 512
q ext13-f1-direct-mixtral python -m moetrace.ext7_controls direct $P --splits discovery,validation,rep_discovery,rep_validation
q ext13-f1-expert-mixtral python scripts/ext7_wino_expert.py $P --families main,rep --layers all --pairs-layers auto --max-spawn 40000
q ext13-f1-heads-mixtral  python scripts/ext7_wino_heads.py $P --families main --layers auto --max-rows 50000
q ext13-f1-grid-mixtral   python scripts/ext7_wino_grid.py $P --families main --kinds layer,attn_layer,resid --window 1 --max-rows 60000 --token-budget 200000
