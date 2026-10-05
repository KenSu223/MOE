#!/usr/bin/env bash
# ext8 step E after the merge into engine.py: regression (verify_olmoe, ext5 engine verification) + ext8 verification.
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
$Q ext8-regress-verify-olmoe -- python scripts/verify_olmoe.py || exit 1
$Q ext8-regress-ext5-engine -- python scripts/ext5_engine_verify.py || exit 1
$Q ext8-engine-verify -- python scripts/ext8_engine_verify.py --engine main --n-cf 6 --n-wino-pairs 3 --stress 20000 || exit 1
echo "EXT8 ENGINE CHAIN DONE"
