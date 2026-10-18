#!/usr/bin/env bash
# ext14 Phase 6 (p6-data): all-association-strata E1-family cells of a Stage-2 model (coordinator decision 2026-10-17 03:15Z): after the
# model's Set-A case sets, every base cell of e1_a2e / e1_e2a / e1p_a2e / e1p_e2a whose neutral set is insufficient gets '<cell>_all'
# (twin / Stage-1 counterpart's selected items preferred), then a tier-1 chain for those cells. For the dense vs MoE / capacity
# comparisons only.
# Usage: nohup setsid bash scripts/ext14_chain_base_all.sh <proto> > logs/ext14_base_all_<proto>.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
. .venv/bin/activate
LOG=logs/ext14_base_chain.log
p=$1
until grep -q "casesets $p Set A rc=" $LOG; do sleep 60; done
cells=$(python -c "
import json
r = json.load(open('data/dial/p6_conditions.json'))['conditions']
out = []
for c in ('e1_a2e', 'e1_e2a', 'e1p_a2e', 'e1p_e2a'):
    e = r.get(c, {}).get('$p')
    if e and (e.get('behaviour_only') or e.get('insufficient')):
        out.append(c + '_all')
print(' '.join(out))
")
echo "$(date -u +%FT%TZ) all-strata cells for $p: [$cells]" >> $LOG
[ -z "$cells" ] && exit 0
python scripts/ext14_casesets.py $p --cells $(echo $cells | tr ' ' ',') >> logs/ext14_base_cases.log 2>&1
echo "$(date -u +%FT%TZ) casesets $p all-strata rc=$?" >> $LOG
bash scripts/ext14_chain_base.sh "$cells" "$p"
echo "$(date -u +%FT%TZ) all-strata chain done ($p)" >> $LOG
