#!/usr/bin/env bash
# ext11 Part B: vocabulary projections (one prefill pass per model with DiagSpec.contrib_final_vectors); GPU lock per job.
cd /home/ubuntu/MOE
bash scripts/gpu_queue.sh ext11-vocab-qwen3 -- python scripts/ext11_writer_vocab.py qwen3 --out qwen3_writer_vocab
bash scripts/gpu_queue.sh ext11-vocab-mixtral -- python scripts/ext11_writer_vocab.py mixtral --out mixtral_bos_writer_vocab
echo "$(date -u +%FT%TZ) ext11 vocab chain finished"
