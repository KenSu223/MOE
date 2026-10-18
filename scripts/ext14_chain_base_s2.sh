#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): Stage-2 orchestration for one model after its competence scans (scripts/ext14_chain_base_stage2*.sh):
#   1. extension scans for the WinoGrande cells with < 256 core neutral margin pairs (skipped if an extension scan of the model ran);
#   2. case sets of Set A (14 cells + e3p3_a2e; tier 1) and Set C (behaviour only by design);
#   3. smoke: tier 1 of the first registered cell alone, base analysis must write BASE READY (dense models: T1-T5 only), minutes per
#      stage posted in PROGRESS ("[p6-data] SMOKE <proto> ...");
#   4. the full tier-1 chain of the model (scripts/ext14_chain_base.sh).
# Usage: nohup setsid bash scripts/ext14_chain_base_s2.sh <proto> > logs/ext14_base_s2_<proto>.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400 HF_HOME=/opt/dlami/nvme/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
p=$1
WINO=e1_a2e,e1_e2a,e3p3_a2e,e3_a2e,e3_e2a,e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,sc_agree_a2e,sc_conflict_a2e
SETA="e1_a2e e1_e2a e3p3_a2e e3_a2e e3_e2a e1p_a2e e1p_e2a e2_a2e e2_e2a l_ind l_bind l_ant l_prop l_cf l_ioi"
D=results/dial_scan_$p
echo "$(date -u +%FT%TZ) s2 $p start (pid $$)" >> $LOG
until grep -q "stage2 scans done ($p)" $LOG && [ -f $D/scan_l_ind_all.parquet -o -f $D/scan_l_ind_core.parquet ]; do sleep 60; done
# 1. extension scans
if ! ls $D/scan_e1_a2e_ext*.parquet > /dev/null 2>&1; then
  until ! pgrep -f "ext14_chain_base_scan2.sh dialE-ext ext .* $p" > /dev/null; do sleep 60; done
  if ! ls $D/scan_e1_a2e_ext*.parquet > /dev/null 2>&1; then
    python scripts/ext14_casesets.py $p --plan --cells $WINO >> logs/ext14_base_cases.log 2>&1
    need=$(paste -sd, data/dial/build/need_ext_$p.txt)
    if [ -n "$need" ]; then
      bash scripts/gpu_queue.sh ext14-scan-$p-dialE-ext -- python scripts/ext14_scan.py $p --part cells --subset ext --cells $need > logs/ext14-scan-$p-dialE-ext.log 2>&1 < /dev/null
      echo "$(date -u +%FT%TZ) ext14-scan-$p-dialE-ext rc=$?" >> $LOG
    fi
  fi
fi
# 2. case sets
python scripts/ext14_casesets.py $p --cells e1_a2e,e1_e2a,e3p3_a2e,e3_a2e,e3_e2a,e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,l_ind,l_bind,l_ant,l_prop,l_cf,l_ioi >> logs/ext14_base_cases.log 2>&1
echo "$(date -u +%FT%TZ) casesets $p Set A rc=$?" >> $LOG
python scripts/ext14_casesets.py $p --cells sc_agree_a2e,sc_conflict_a2e --behaviour-only >> logs/ext14_base_cases.log 2>&1
echo "$(date -u +%FT%TZ) casesets $p Set C (behaviour only) rc=$?" >> $LOG
# 3. smoke on the first runnable cell
first=$(python - <<EOF
import json
r = json.load(open("data/dial/p6_conditions.json"))["conditions"]
for c in "$SETA".split():
    e = r.get(c, {}).get("$p")
    if e and not e.get("behaviour_only") and not e.get("insufficient") and not e.get("anchor"):
        print(c); break
EOF
)
if [ -z "$first" ]; then echo "$(date -u +%FT%TZ) s2 $p: no runnable cell" >> $LOG; exit 0; fi
bash scripts/ext14_chain_base.sh "$first" "$p"
if [ ! -f results/p6_${first}_${p}/.done_analysis ]; then
  echo "$(date -u +%FT%TZ) [p6-data] SMOKE $p FAILED on $first (no BASE READY; see logs/ext14_base_analyze_${first}_${p}.log); full chain NOT armed" >> logs/PROGRESS.md
  exit 1
fi
mins=$(python - <<EOF
import re
d = {}
for l in open("logs/gpu_queue.log"):
    m = re.search(r"END   ext14-$first-$p-(\w+) rc=0 \((\d+)s\)", l)
    if m:
        d[m.group(1)] = round(int(m.group(2)) / 60, 1)
print(", ".join(f"{k} {v}" for k, v in d.items()), "| total", round(sum(d.values()), 1))
EOF
)
echo "$(date -u +%FT%TZ) [p6-data] SMOKE $p tier 1 on $first OK (BASE READY written): minutes per stage $mins" >> logs/PROGRESS.md
# 4. full chain
bash scripts/ext14_chain_base.sh "$SETA" "$p"
echo "$(date -u +%FT%TZ) s2 $p done" >> $LOG
