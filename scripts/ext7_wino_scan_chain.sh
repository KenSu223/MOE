#!/bin/bash
# ext7: WinoGrande STR competence scans (train_xl pairs from scripts/ext7_wino_build.py), one GPU job per protocol.
cd /home/ubuntu/MOE
. .venv/bin/activate
for p in olmoe qwen3 mixtral_nobos mixtral_bos; do
  bash scripts/gpu_queue.sh ext7-wino-scan-$p -- python scripts/ext7_wino_scan.py $p || echo "scan $p failed"
done
python scripts/ext7_wino_build.py --split dev --protos qwen3,mixtral_nobos >/dev/null 2>&1
for p in qwen3 mixtral_nobos; do
  bash scripts/gpu_queue.sh ext7-wino-scan-dev-$p -- python scripts/ext7_wino_scan.py $p --split dev || echo "dev scan $p failed"
done
echo "chain done"
