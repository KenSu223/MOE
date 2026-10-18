#!/usr/bin/env bash
# ext14 Phase 6 (p6-data) Stage-2 scans for one model (v2 of scripts/ext14_chain_base_stage2.sh, which is not edited while it runs):
# association probes (full specs of the family) -> competence scans of the Set-A cells + Set C (core-776 items for the WinoGrande
# cells, all items for the ladder) -> CounterFact STR filter (scripts/ext14_scan_cf.py, paper case set of the family) -> l_cf
# conversion (scripts/ext14_build_stage2.py) and its scan. Skips steps whose outputs exist.
# Usage: nohup setsid bash scripts/ext14_chain_base_stage2b.sh <proto> > logs/ext14_base_stage2_<proto>.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
p=$1
read key fam <<< $(python -c "import sys; sys.path.insert(0,'.'); from moetrace.ext14_data import PROTOS; k,s,f=PROTOS['$p']; print(k, f)")
pk=$([ "$fam" = "qwen" ] && echo Qwen3 || echo Mixtral)
CELLS=${CELLS:-e1_a2e,e1_e2a,e3p3_a2e,e3_a2e,e3_e2a,e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,sc_agree_a2e,sc_conflict_a2e,l_ant,l_prop,l_bind,l_ioi}
q() { name=$1; shift; bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1 < /dev/null; rc=$?; echo "$(date -u +%FT%TZ) $name rc=$rc" >> $LOG; return $rc; }
D=results/dial_scan_$p
[ -f $D/probe_values.parquet ] || q ext14-scan-$p-probes python scripts/ext14_scan.py $p --part probes --specs data/dial/build/probe_specs_full_$fam.parquet
[ -f $D/scan_e1_a2e_core.parquet ] || q ext14-scan-$p-cells python scripts/ext14_scan.py $p --part cells --subset core --cells $CELLS
[ -f results/${key}_str/str_candidates.parquet ] || q ext14-cf-filter-$p python scripts/ext14_scan_cf.py $key --out ${key}_str --paper-key $pk
python scripts/ext14_build_stage2.py --what l_cf --proto $p >> logs/ext14_base_stage2_build.log 2>&1 && \
  q ext14-scan-$p-l_cf python scripts/ext14_scan.py $p --part cells --subset all --cells l_cf
echo "$(date -u +%FT%TZ) stage2 scans done ($p)" >> $LOG
