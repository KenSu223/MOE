# p5-addback status (agent p5-addback) — 2026-10-11T13:57Z

## done
- Code: moetrace/ext13ab.py (registry from data/wino_ref/p5_conditions.json, loaders, queues), scripts/ext13ab_run.py
  (multi-task ext8 driver, ext8 file format, resumable), ext13ab_analyze.py (ext8 analyse + ext12 bootstrap, deletion
  bootstrap, k80-of-own-max, seed CIs, strata, paired contrasts), ext13ab_plot.py (per-condition add-back / deletion / split,
  overview, split_all; ALSO rewrites the section via ext13ab_text.py), ext13ab_text.py (templated section + readings),
  ext13ab_ready.py (BASE READY detection, DONE logic), chain scripts/ext13ab_chain.sh.
- Smoke: results/p5_F1pre_*_ab_smoke. Complete runs: p5_{f1,dir_f1,dir_opt}_{qwen3,mixtral_bos}_ab, p5_f1_qwen3_rep_ab.
- GPU so far 112 min (10 jobs incl. smoke).

## running (nothing needs this agent alive)
- Chains scripts/ext13ab_chain.sh qwen3 / mixtral (started 06:14Z, 40 h deadline): wait for "[p5-data] BASE READY <cond>/<proto>",
  run jobs ext13ab-<model>-<n> (logs/ext13ab-<model>-<n>.log), then analysis + figures + section refresh under
  flock logs/ext13ab_analyze.lock (logs/ext13ab_analyze.log). Exit with logs/ext13ab_chain_<model>.done when every
  non-behaviour-only condition of the protocol is complete.
- Remaining: cp_nat (+rep), cp_vt_orig, cp_vt_neu, vt_e2a_orig, vt_e2a_neu, vt_a2e_orig (Qwen3 only), f2e.

## resume
- Read this file, tail logs/ext13ab_chain.log; never redo GPU work (addback_state.pkl resumes, run_meta complete=true).
- If a chain died: `setsid nohup bash scripts/ext13ab_chain.sh <qwen3|mixtral> > /dev/null 2>&1 &`.
- Manual refresh: `(flock -x 8; python -W ignore scripts/ext13ab_analyze.py && python -W ignore scripts/ext13ab_plot.py) 8>logs/ext13ab_analyze.lock`.
