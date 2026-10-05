#!/usr/bin/env bash
# ext7-controls W4: joint sublayer decomposition at the final position (all attention / all MoE / both, denoise and noise;
# ext8 multi step kinds) for the WinoGrande role swap and IOI on Qwen3 and Mixtral (BOS), with agent ext7-wino's
# scripts/ext7_wino_joint.py. Same --out / --splits as the W2 sweeps. Every job via scripts/gpu_queue.sh.
cd /home/ubuntu/MOE
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
q() { local name=$1; shift; bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1; echo "$(date -u +%FT%TZ) $name rc=$?"; }
MAINREP=discovery,validation,rep_discovery,rep_validation
q ext7c-w4-role-qwen3 python scripts/ext7_wino_joint.py --model qwen3 --pairs data/wino_role/pairs_qwen3.parquet --case-sets data/wino_role/case_sets_qwen3.json --out wino_role_qwen3_str --splits $MAINREP --agent ext7-controls
q ext7c-w4-ioi-qwen3-s2io python scripts/ext7_wino_joint.py --model qwen3 --pairs data/ioi/pairs_s2io_qwen3.parquet --case-sets data/ioi/case_sets_s2io.json --out ioi_qwen3_s2io --splits $MAINREP --agent ext7-controls
q ext7c-w4-ioi-qwen3-s1io python scripts/ext7_wino_joint.py --model qwen3 --pairs data/ioi/pairs_s1io_qwen3.parquet --case-sets data/ioi/case_sets_s1io.json --out ioi_qwen3_s1io --splits $MAINREP --agent ext7-controls
q ext7c-w4-role-mixtral_bos python scripts/ext7_wino_joint.py --model mixtral --pairs data/wino_role/pairs_mixtral_bos.parquet --case-sets data/wino_role/case_sets_mixtral_bos.json --out wino_role_mixtral_bos_str --splits discovery,validation --agent ext7-controls
q ext7c-w4-ioi-mixtral_bos-s2io python scripts/ext7_wino_joint.py --model mixtral --pairs data/ioi/pairs_s2io_mixtral_bos.parquet --case-sets data/ioi/case_sets_s2io.json --out ioi_mixtral_bos_s2io --splits $MAINREP --agent ext7-controls
q ext7c-w4-ioi-mixtral_bos-s1io python scripts/ext7_wino_joint.py --model mixtral --pairs data/ioi/pairs_s1io_mixtral_bos.parquet --case-sets data/ioi/case_sets_s1io.json --out ioi_mixtral_bos_s1io --splits $MAINREP --agent ext7-controls
echo "$(date -u +%FT%TZ) w4 chain done"
