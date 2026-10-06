#!/usr/bin/env bash
# ext11 Part C: routing diagnostics (prefill only), Qwen3 then Mixtral BOS; every job through the GPU lock.
cd /home/ubuntu/MOE
bash scripts/gpu_queue.sh ext11-routing-qwen3 -- python scripts/ext11_writer_routing.py qwen3 --out qwen3_writer_routing
bash scripts/gpu_queue.sh ext11-routing-mixtral -- python scripts/ext11_writer_routing.py mixtral --out mixtral_bos_writer_routing
echo "$(date -u +%FT%TZ) ext11 routing chain finished"
