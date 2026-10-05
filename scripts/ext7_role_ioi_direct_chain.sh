#!/usr/bin/env bash
# ext7-controls W4 complement: direct-path split at the final position (prefill only, DiagSpec attn_out_final +
# resid_final; python -m moetrace.ext7_controls direct) for the role swap and IOI runs. Every job via gpu_queue.sh.
cd /home/ubuntu/MOE
set -u
q() { local name=$1; shift; bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1; echo "$(date -u +%FT%TZ) $name rc=$?"; }
MAINREP=discovery,validation,rep_discovery,rep_validation
q ext7c-direct-role-qwen3 python -m moetrace.ext7_controls direct --model qwen3 --pairs data/wino_role/pairs_qwen3.parquet --case-sets data/wino_role/case_sets_qwen3.json --out wino_role_qwen3_str --splits $MAINREP
q ext7c-direct-role-mixtral_bos python -m moetrace.ext7_controls direct --model mixtral --pairs data/wino_role/pairs_mixtral_bos.parquet --case-sets data/wino_role/case_sets_mixtral_bos.json --out wino_role_mixtral_bos_str --splits discovery,validation
for proto in qwen3 mixtral_bos; do
  model=qwen3; [ $proto = mixtral_bos ] && model=mixtral
  for corr in s2io s1io; do
    q ext7c-direct-ioi-$proto-$corr python -m moetrace.ext7_controls direct --model $model --pairs data/ioi/pairs_${corr}_$proto.parquet --case-sets data/ioi/case_sets_$corr.json --out ioi_${proto}_$corr --splits $MAINREP
  done
done
echo "$(date -u +%FT%TZ) direct chain done"
