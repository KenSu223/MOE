#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): register case sets automatically when their scans finish (CPU; survives the agent).
# Each step waits for a line in logs/ext14_base_chain.log (the scan chains log "<job> rc=0"), then runs scripts/ext14_casesets.py.
# Usage: nohup setsid bash scripts/ext14_chain_base_cases.sh > logs/ext14_base_cases.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
step() {  # step <regex in chain log> <proto> <cells>
  until grep -Eq "$1" $LOG; do sleep 30; done
  python scripts/ext14_casesets.py $2 --cells $3 >> logs/ext14_base_cases.log 2>&1
  echo "$(date -u +%FT%TZ) casesets $2 [$3] rc=$?" >> $LOG
}
step "ext14-scan-mixtral_bos-dialA rc=0" mixtral_bos e0_a2e,e0_e2a,aagr_a2e,aagr_e2a,acon_a2e,acon_e2a &
step "ext14-scan-mixtral_bos-dialE-ext3 rc=0" mixtral_bos e1_e2a,e3_a2e,e1p_a2e,e1p_e2a,rsite_e1_a2e &
( until grep -q "ext14-scan-qwen3-dialE-ext1 rc=0" $LOG && grep -q "ext14-scan-qwen3-dialE-ext2 rc=0" $LOG; do sleep 30; done
  python scripts/ext14_casesets.py qwen3 --cells e1_e2a,e3p3_a2e,e3_a2e,e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,sc_agree_a2e,sc_conflict_a2e >> logs/ext14_base_cases.log 2>&1
  echo "$(date -u +%FT%TZ) casesets qwen3 [Dial E after extension] rc=$?" >> $LOG ) &
wait
echo "$(date -u +%FT%TZ) chain_base_cases done" >> $LOG
