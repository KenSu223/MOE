#!/usr/bin/env bash
# ext8 smoke test on OLMoE: STR source run (ext6 scripts) + add-back driver with the dev engine.
cd /home/ubuntu/MOE
Q="bash scripts/gpu_queue.sh"
$Q ext8-smoke-filter -- python scripts/ext8_smoke_src.py || exit 1
$Q ext8-smoke-sweep -- python scripts/ext6_str_sweep.py olmoe --out olmoe_addback_src --layer-chunks 1 || exit 1
$Q ext8-smoke-expert -- python scripts/ext6_str_expert.py olmoe --out olmoe_addback_src --layers $(seq -s, 0 15) --max-spawn 60000 || exit 1
$Q ext8-smoke-addback -- python scripts/ext8_addback_run.py olmoe --task cf --src-run olmoe_addback_src --out olmoe_addback_smoke --pool 32 --budget 30000 --engine ${ENGINE:-dev} || exit 1
echo "SMOKE CHAIN DONE"
