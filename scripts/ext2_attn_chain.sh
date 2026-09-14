#!/usr/bin/env bash
# ext2-attn-patch GPU chain: verification gate, then the three attention-vs-MoE sweeps (paper case sets), every
# GPU step through scripts/gpu_queue.sh. Run dirs: results/{qwen3_bos,mixtral_bos,mixtral_nobos}_attnsweep.
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
log() { echo "$(date -u +%FT%TZ) $*"; }
log "chain start"
. .venv/bin/activate; python scripts/ext2_attn_gate.py || { log "GATE FAILED: sweeps not started"; exit 1; }
log "gate passed"
$Q ext2attn-sweep-qwen3 -- python scripts/ext2_attn_sweep.py qwen3 --out qwen3_bos_attnsweep --sets paper \
  > logs/ext2_sweep_qwen3_bos.log 2>&1 || log "qwen3 sweep failed rc=$?"
$Q ext2attn-sweep-mixtral-bos -- python scripts/ext2_attn_sweep.py mixtral --out mixtral_bos_attnsweep --sets paper \
  > logs/ext2_sweep_mixtral_bos.log 2>&1 || log "mixtral bos sweep failed rc=$?"
$Q ext2attn-sweep-mixtral-nobos -- python scripts/ext2_attn_sweep.py mixtral --out mixtral_nobos_attnsweep --sets paper \
  --base-run mixtral_nobos --no-special-tokens > logs/ext2_sweep_mixtral_nobos.log 2>&1 || log "mixtral nobos sweep failed rc=$?"
log "chain done"
