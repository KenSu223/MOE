#!/usr/bin/env bash
# ext7-controls: competence scans (prefill only) for W7 role-swap pairs and W8 IOI pairs. Each job via gpu_queue.sh.
cd /home/ubuntu/MOE
set -u
for proto in qwen3 mixtral_bos; do
  bash scripts/gpu_queue.sh ext7c-role-scan-$proto -- python scripts/ext7_role_scan.py $proto > logs/ext7c_role_scan_$proto.log 2>&1
  echo "$(date -u +%FT%TZ) role $proto rc=$?"
  bash scripts/gpu_queue.sh ext7c-ioi-scan-$proto -- python scripts/ext7_ioi_scan.py $proto > logs/ext7c_ioi_scan_$proto.log 2>&1
  echo "$(date -u +%FT%TZ) ioi $proto rc=$?"
done
