#!/usr/bin/env bash
# ext7-controls W3: all six restricted grids in ONE gpu_queue job (submitted as
#   bash scripts/gpu_queue.sh ext7c-w3-all -- bash scripts/ext7_role_ioi_grids_all.sh
# so that the GPU lock is taken once). The grid scripts skip runs whose grid_w1 is already complete, so the identical
# steps of scripts/ext7_role_ioi_chain.sh become no-ops afterwards.
cd /home/ubuntu/MOE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for proto in qwen3 mixtral_bos; do
  model=qwen3; [ $proto = mixtral_bos ] && model=mixtral
  for corr in s2io s1io; do
    python scripts/ext7_ioi_grid.py --model $model --pairs data/ioi/pairs_${corr}_$proto.parquet --case-sets data/ioi/case_sets_${corr}_grid.json --out ioi_${proto}_${corr}_grid --kinds layer,attn_layer > logs/ext7c-w3all-ioi-$proto-$corr.log 2>&1
    echo "$(date -u +%FT%TZ) ioi $proto $corr rc=$?"
  done
  python scripts/ext7_role_grid.py --model $model --pairs data/wino_role/pairs_$proto.parquet --case-sets data/wino_role/case_sets_${proto}_grid.json --out wino_role_${proto}_grid --kinds layer,attn_layer > logs/ext7c-w3all-role-$proto.log 2>&1
  echo "$(date -u +%FT%TZ) role $proto rc=$?"
done
