#!/usr/bin/env bash
# ext14 Phase 6 (p6-data) Stage-2 scans for one model: association probes (full specs of the family), then the competence scans of the
# Set-A cells + Set C (core-776 items for the WinoGrande cells; all items for the ladder), through the GPU queue.
# Usage: nohup setsid bash scripts/ext14_chain_base_stage2.sh <proto> > logs/ext14_base_stage2_<proto>.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
p=$1
fam=$(python -c "import sys; sys.path.insert(0,'.'); from moetrace.ext14_data import PROTOS; print(PROTOS['$p'][2])")
CELLS=${CELLS:-e1_a2e,e1_e2a,e3p3_a2e,e3_a2e,e3_e2a,e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,sc_agree_a2e,sc_conflict_a2e,l_ant,l_prop,l_bind,l_ioi}
q() { name=$1; shift; t0=$(date +%s); bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1 < /dev/null; rc=$?; echo "$(date -u +%FT%TZ) $name rc=$rc" >> $LOG; return $rc; }
q ext14-scan-$p-probes python scripts/ext14_scan.py $p --part probes --specs data/dial/build/probe_specs_full_$fam.parquet
q ext14-scan-$p-cells python scripts/ext14_scan.py $p --part cells --subset core --cells $CELLS
echo "$(date -u +%FT%TZ) stage2 scans done ($p)" >> $LOG
