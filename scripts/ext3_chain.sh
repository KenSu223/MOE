#!/usr/bin/env bash
# ext3 GPU chain (agent ext3-bos-mechanism): verification gate, then the experiment passes; every step via gpu_queue.sh.
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
log() { echo "$(date -u +%FT%TZ) $*"; }
log "chain start"
$Q ext3-verify-diag2 -- python scripts/ext3_verify_diag.py 6 || { log "verify-diag failed"; exit 1; }
$Q ext3-verify-olmoe -- python scripts/verify_olmoe.py || { log "verify_olmoe failed"; exit 1; }
. .venv/bin/activate; python scripts/ext3_gate.py || { log "GATE FAILED: experiments not started"; exit 1; }
log "gate passed"
$Q ext3-mixtral-passA -- python scripts/ext3_run_variants.py mixtral --variants bos,nobos,shift1,eos,bosbos
$Q ext3-mixtral-passB -- python scripts/ext3_run_variants.py mixtral --variants nl,dot,comma,the,rare
$Q ext3-mixtral-passC -- python scripts/ext3_run_variants.py mixtral --variants bos,sinkfull,sinkkey --skip-outputs bos
$Q ext3-qwen3-pass -- python scripts/ext3_run_variants.py qwen3 --variants default,eot
$Q ext3-corpus-wiki -- python scripts/ext3_corpus_routing.py mixtral --corpus wiki
$Q ext3-corpus-code -- python scripts/ext3_corpus_routing.py mixtral --corpus code
log "chain done"
