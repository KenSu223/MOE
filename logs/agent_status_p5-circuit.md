# p5-circuit status (Phase 5: Direction-10 heads + experts add-back for every Phase-5 condition)

Updated: 2026-10-11T13:49Z

## Plan
- Code: moetrace/ext13ci.py (condition registry from data/wino_ref/p5_conditions.json, tolerant task loader),
  scripts/ext13ci_heads_run.py (step 1: single heads every (layer, head) + attn_layer refs + head/expert DLA + final-position
  attention mass on position classes), scripts/ext13ci_circuit_run.py (step 2: mixed greedy, head greedy, static orderings,
  ceilings, sanity; several conditions of one model per pass), analysis / figures / text scripts.
- Outputs: results/p5_<cond>_<proto>_ci/ (step 1), results/p5_<proto>_ci/<group>/ (step 2).
- Smoke on results/wino_ref_F1_qwen3 (_smoke suffix), then detached chains waiting for "[p5-data] BASE READY <cond>/<proto>".

## Status
- [x] code: moetrace/ext13ci.py, scripts/ext13ci_{heads_run,circuit_run,analyze,chain,text}.py; smoke OK 06:14Z
- [x] registry format checked (data/wino_ref/p5_conditions.json, conditions/<cond>/<proto>)
- [x] step 1 done: f1, dir_f1, dir_opt (both models); step 2 g1 done (Qwen3 12:18Z, Mixtral 13:45Z)
- [x] analysis (incl. paired contrasts, overview panel), figures, section writer (DEVIATIONS + data-driven copy-head block) finished 13:55Z
- [ ] remaining (automatic, chains alive, pids 1326568 / 1326569): cp_nat, cp_vt_orig, cp_vt_neu, vt_e2a_orig, vt_e2a_neu,
      vt_a2e_orig (Qwen3 only), f2e -> step 1 each, step 2 groups g2 (cp_*), g3 (vt_*), g4 (f2e). After every step-1
      condition the chain runs scripts/ext13ci_analyze.py --step 1; after every step-2 group --step 2 + scripts/ext13ci_text.py
      (results/sections/ext13_circuit.md); at the end the text again. Nothing needs the agent alive.
- [ ] optional hand-written READINGS in scripts/ext13ci_text.py once all groups are done

## Resume
Read this file, logs/PROGRESS.md ([p5-circuit] lines), logs/ext13ci_chain_<proto>.log. If a chain process is gone, relaunch
`setsid nohup .venv/bin/python scripts/ext13ci_chain.py <proto> >> logs/ext13ci_chain_<proto>.log 2>&1 &`; it re-derives
everything from disk (heads_state.json, circuit_state.pkl, run_meta complete flags, results/p5_<proto>_ci/chain_state.json)
and never redoes finished passes. Skip a condition: add its name to results/p5_<proto>_ci/skip.txt.
- 2026-10-12T16:30Z [coordinator] Phase 5 COMPLETE: all chains finished 05:40Z, results committed and pushed (bed8247). Nothing left to run for this agent.
