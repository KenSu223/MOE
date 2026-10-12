#!/usr/bin/env bash
# Touches logs/ext13ko_<proto>.nomore once every non-behaviour-only condition of the registry is finished (knockout + writer),
# so that the running scripts/ext13ko_chain.sh <proto> starts the cross phase and its final analysis (behaviour-only
# conditions never get a BASE READY line, so the chain's own "nothing pending" test never fires).
#   nohup bash scripts/ext13ko_nomore_watch.sh > logs/ext13ko_nomore_watch.log 2>&1 &
cd "$(dirname "$0")/.."
. .venv/bin/activate
CONDS="f1 dir_f1 dir_opt cp_nat cp_vt_orig cp_vt_neu vt_e2a_orig vt_e2a_neu vt_a2e_orig vt_a2e_neu f2e"
deadline=$(( $(date +%s) + 60 * 3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  left=0
  for proto in qwen3 mixtral_bos; do
    [ -f "logs/ext13ko_${proto}.nomore" ] && continue
    need=$(python -c "
import sys; sys.path.insert(0, '.')
from moetrace import ext13ko as K
reg = K._registry(); out = []
for c in '$CONDS'.split():
    if c in reg and '$proto' in reg[c] and not K.entry(c, '$proto')['behaviour_only']:
        out.append(c)
print(' '.join(out))" 2>/dev/null)
    ok=1
    for c in $need; do python scripts/ext13ko_status.py "$proto" "$c" 2>/dev/null || { ok=0; break; }; done
    if [ $ok -eq 1 ] && [ -n "$need" ]; then
      touch "logs/ext13ko_${proto}.nomore"; echo "$(date -u +%FT%TZ) nomore $proto (finished: $need)"
    else
      left=1
    fi
  done
  [ $left -eq 0 ] && { echo "$(date -u +%FT%TZ) done"; exit 0; }
  sleep 300
done
