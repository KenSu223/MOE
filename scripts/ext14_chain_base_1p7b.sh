#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): Qwen3-1.7B is competence-scan only (plan): after its Stage-2 scans, register every cell as behaviour only.
cd "$(dirname "$0")/.."
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
until grep -q "stage2 scans done (qwen3_1p7b)" $LOG; do sleep 120; done
python scripts/ext14_casesets.py qwen3_1p7b --cells e1_a2e,e1_e2a,e3p3_a2e,e3_a2e,e3_e2a,e1p_a2e,e1p_e2a,e2_a2e,e2_e2a,sc_agree_a2e,sc_conflict_a2e,l_ind,l_bind,l_ant,l_prop,l_cf,l_ioi --behaviour-only >> logs/ext14_base_cases.log 2>&1
echo "$(date -u +%FT%TZ) casesets qwen3_1p7b (behaviour only) rc=$?" >> $LOG
