#!/usr/bin/env bash
# ext7-controls: IOI final-position attention on IO / S1 / S2 for every head (prefill diag pass), both models.
cd /home/ubuntu/MOE
for proto in qwen3 mixtral_bos; do
  bash scripts/gpu_queue.sh ext7c-ioi-attn-$proto -- python scripts/ext7_ioi_attn.py $proto > logs/ext7c_ioi_attn_$proto.log 2>&1
  echo "$(date -u +%FT%TZ) ioi-attn $proto rc=$?"
done
