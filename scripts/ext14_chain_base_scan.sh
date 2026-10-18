#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): competence scans (prefill only) through the GPU queue.
# Usage: nohup setsid bash scripts/ext14_chain_base_scan.sh <stage> [protos] > logs/ext14_base_scan_<stage>.out 2>&1 < /dev/null &
#   stage dialE_core : Dial E + Set C + R-site on the core-776 items (two jobs per model)
#   stage dialE_ext  : extension items for the cells listed in data/dial/build/need_ext_<proto>.txt (written by ext14_casesets.py)
#   stage probes     : association probes (data/dial/build/probe_specs_<fam>.parquet)
#   stage <name>     : any other cell list in data/dial/build/scan_cells_<name>.txt (one cell per line), subset all
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
stage=$1; shift
PROTOS="${*:-qwen3 mixtral_bos}"
q() { name=$1; shift; bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1 < /dev/null; rc=$?; echo "$(date -u +%FT%TZ) $name rc=$rc" >> $LOG; return $rc; }
for p in $PROTOS; do
  case $stage in
    dialE_core)
      q ext14-scan-$p-dialE-core1 python scripts/ext14_scan.py $p --part cells --subset core --cells e1_a2e,e1_e2a,e3_a2e,e3_e2a,rsite_e1_a2e
      q ext14-scan-$p-dialE-core2 python scripts/ext14_scan.py $p --part cells --subset core --cells e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,sc_agree_a2e,sc_conflict_a2e
      ;;
    dialE_ext)
      f=data/dial/build/need_ext_$p.txt
      [ -s "$f" ] && q ext14-scan-$p-dialE-ext python scripts/ext14_scan.py $p --part cells --subset ext --cells "$(paste -sd, $f)"
      ;;
    probes)
      q ext14-scan-$p-probes python scripts/ext14_scan.py $p --part probes
      ;;
    *)
      f=data/dial/build/scan_cells_$stage.txt
      q ext14-scan-$p-$stage python scripts/ext14_scan.py $p --part cells --subset all --cells "$(paste -sd, $f)"
      ;;
  esac
done
echo "$(date -u +%FT%TZ) scan chain $stage done ($PROTOS)" >> $LOG
