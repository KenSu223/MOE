#!/usr/bin/env bash
# ext13 p5-data: options-prefix copy conditions (coordinator decision 2026-10-11): rescan cp_vt / cp_nat (both models), then
# rebuild the cp_* case sets (registers them, clears rebuild_pending so the base chain proceeds) and check the pair contract.
cd "$(dirname "$0")/.."
export GPU_QUEUE_WAIT_S=86400
export HF_HOME=/opt/dlami/nvme/hf
. .venv/bin/activate
q() { name=$1; shift; bash scripts/gpu_queue.sh "$name" -- "$@" > "logs/$name.log" 2>&1 < /dev/null; rc=$?; echo "$(date -u +%FT%TZ) $name rc=$rc" >> logs/ext13_p5_chain.log; return $rc; }
ok=1
for p in qwen3 mixtral_bos; do
  q ext13-p5-cpopt-scan-$p python scripts/ext13_p5_scan.py $p --part conds --conds cp_vt,cp_nat --meta-key p5_scan_conds_cp_options || ok=0
done
if [ $ok -eq 1 ]; then
  python -W ignore scripts/ext13_p5_casesets.py --conds cp_nat,cp_vt_orig,cp_vt_neu --progress-suffix " (options prefix)" >> logs/ext13_p5_casesets_cpopt.log 2>&1
  echo "$(date -u +%FT%TZ) casesets cp options rc=$?" >> logs/ext13_p5_chain.log
  python -W ignore -c "
import json
from moetrace import ext7_pairs as P
for p in ('qwen3','mixtral_bos'):
    for c in ('cp_nat','cp_vt_orig','cp_vt_neu'):
        cs=json.load(open(f'data/wino_ref/case_sets_p5_{c}_{p}.json'))
        pairs=P.load_pairs(f'data/wino_ref/pairs_p5_{c}_{p}.parquet', cs)
        fam=P.resolve_sets(cs); ids=P.all_ids(fam); dc=P.directed_cases(pairs, ids)
        print(p, c, {k:{s:len(v) for s,v in f.items()} for k,f in fam.items()}, 'contract ok')
" >> logs/ext13_p5_casesets_cpopt.log 2>&1
  echo "$(date -u +%FT%TZ) contract check cp options rc=$?" >> logs/ext13_p5_chain.log
  python -W ignore scripts/ext13_p5_behaviour.py >> logs/ext13_p5_casesets_cpopt.log 2>&1
  python -W ignore scripts/ext13_p5_text.py >> logs/ext13_p5_casesets_cpopt.log 2>&1
fi
