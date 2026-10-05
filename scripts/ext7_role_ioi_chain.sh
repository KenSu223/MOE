#!/usr/bin/env bash
# ext7-controls: W2 (final-position sweep), W5 (IOI heads), W3 (grids, first 64 validation pairs, window 1) for the
# WinoGrande role swap (W7) and IOI (W8) on Qwen3 and Mixtral (BOS), with agent ext7-wino's generic pair runner.
# Every job via scripts/gpu_queue.sh. Log: logs/ext7c_role_ioi_chain.log
cd /home/ubuntu/MOE
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
q() { local name=$1; shift; bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1; echo "$(date -u +%FT%TZ) $name rc=$?"; }
MAINREP=discovery,validation,rep_discovery,rep_validation
# ---- W2 final-position sweeps (layer, attn_layer, block)
q ext7c-w2-role-qwen3 python scripts/ext7_wino_sweep.py --model qwen3 --pairs data/wino_role/pairs_qwen3.parquet --case-sets data/wino_role/case_sets_qwen3.json --out wino_role_qwen3_str --splits $MAINREP --kinds layer,attn_layer,block --agent ext7-controls
q ext7c-w2-ioi-qwen3-s2io python scripts/ext7_wino_sweep.py --model qwen3 --pairs data/ioi/pairs_s2io_qwen3.parquet --case-sets data/ioi/case_sets_s2io.json --out ioi_qwen3_s2io --splits $MAINREP --kinds layer,attn_layer,block --agent ext7-controls
q ext7c-w2-ioi-qwen3-s1io python scripts/ext7_wino_sweep.py --model qwen3 --pairs data/ioi/pairs_s1io_qwen3.parquet --case-sets data/ioi/case_sets_s1io.json --out ioi_qwen3_s1io --splits $MAINREP --kinds layer,attn_layer,block --agent ext7-controls
q ext7c-w2-role-mixtral_bos python scripts/ext7_wino_sweep.py --model mixtral --pairs data/wino_role/pairs_mixtral_bos.parquet --case-sets data/wino_role/case_sets_mixtral_bos.json --out wino_role_mixtral_bos_str --splits discovery,validation --kinds layer,attn_layer,block --max-rows 50000 --agent ext7-controls
q ext7c-w2-ioi-mixtral_bos-s2io python scripts/ext7_wino_sweep.py --model mixtral --pairs data/ioi/pairs_s2io_mixtral_bos.parquet --case-sets data/ioi/case_sets_s2io.json --out ioi_mixtral_bos_s2io --splits $MAINREP --kinds layer,attn_layer,block --max-rows 50000 --agent ext7-controls
q ext7c-w2-ioi-mixtral_bos-s1io python scripts/ext7_wino_sweep.py --model mixtral --pairs data/ioi/pairs_s1io_mixtral_bos.parquet --case-sets data/ioi/case_sets_s1io.json --out ioi_mixtral_bos_s1io --splits $MAINREP --kinds layer,attn_layer,block --max-rows 50000 --agent ext7-controls
# ---- W5 heads on IOI (auto = top-4 discovery attention layers + 1 null layer), family main
for proto in qwen3 mixtral_bos; do
  model=qwen3; [ $proto = mixtral_bos ] && model=mixtral
  for corr in s2io s1io; do
    q ext7c-w5-ioi-$proto-$corr python scripts/ext7_wino_heads.py --model $model --pairs data/ioi/pairs_${corr}_$proto.parquet --case-sets data/ioi/case_sets_$corr.json --out ioi_${proto}_$corr --splits $MAINREP --layers auto --families main --agent ext7-controls
  done
done
# ---- W3 grids (window 1, kinds layer and attn_layer; first 64 validation pairs; named positions only; separate run dirs)
for proto in qwen3 mixtral_bos; do
  model=qwen3; [ $proto = mixtral_bos ] && model=mixtral
  for corr in s2io s1io; do
    q ext7c-w3-ioi-$proto-$corr python scripts/ext7_ioi_grid.py --model $model --pairs data/ioi/pairs_${corr}_$proto.parquet --case-sets data/ioi/case_sets_${corr}_grid.json --out ioi_${proto}_${corr}_grid --kinds layer,attn_layer
  done
  q ext7c-w3-role-$proto python scripts/ext7_role_grid.py --model $model --pairs data/wino_role/pairs_$proto.parquet --case-sets data/wino_role/case_sets_${proto}_grid.json --out wino_role_${proto}_grid --kinds layer,attn_layer
done
echo "$(date -u +%FT%TZ) chain done"
