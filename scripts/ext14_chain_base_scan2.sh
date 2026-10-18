#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): generic competence-scan chain (prefill only) through the GPU queue (second version; the first chain
# script is not edited while instances of it run).
# Usage: nohup setsid bash scripts/ext14_chain_base_scan2.sh <tag> <subset core|ext|all> <cells comma list> [protos] [--wait-line REGEX --wait-max-s N]
#   --wait-line: wait until logs/PROGRESS.md matches REGEX (or N seconds passed) before queueing anything
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
tag=$1; subset=$2; cells=$3; shift 3
PROTOS="qwen3 mixtral_bos"; WAIT=""; WMAX=0
while [ $# -gt 0 ]; do
  case $1 in
    --wait-line) WAIT=$2; shift 2;;
    --wait-max-s) WMAX=$2; shift 2;;
    *) PROTOS="$1"; shift;;
  esac
done
if [ -n "$WAIT" ]; then
  t0=$(date +%s)
  until grep -Eq "$WAIT" logs/PROGRESS.md || { [ $WMAX -gt 0 ] && [ $(( $(date +%s) - t0 )) -ge $WMAX ]; }; do sleep 30; done
  echo "$(date -u +%FT%TZ) scan2 $tag: wait condition met ($(grep -Ec "$WAIT" logs/PROGRESS.md) matching lines)" >> $LOG
fi
for p in $PROTOS; do
  name=ext14-scan-$p-$tag
  bash scripts/gpu_queue.sh "$name" -- python scripts/ext14_scan.py $p --part cells --subset $subset --cells "$cells" > "logs/$name.log" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) $name rc=$?" >> $LOG
done
echo "$(date -u +%FT%TZ) scan2 $tag done ($PROTOS)" >> $LOG
