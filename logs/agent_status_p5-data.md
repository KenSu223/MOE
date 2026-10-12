# p5-data status (agent p5-data, Phase 5 data + base analyses)
Updated: 2026-10-11T13:58Z
## Done
- probe v2 + pre-registered rule (Qwen3 R={v1,v2}, Mixtral R={v2}); all builds, scans, case sets, registry data/wino_ref/p5_conditions.json
- copy prefix switched to "Options: t1, t2.\n" (coordinator decision); void "Of X and Y" files in data/wino_ref/void_cp_respectively/,
  scans results/wino_ref_<proto>/scan_pairs_p5_cp_*_respectively.parquet (behaviour table ext13base_cp_respectively)
- CASES READY (options prefix) cp_* 09:43Z; BASE READY f1, dir_f1, dir_opt (both models)
- section results/sections/ext13_base.md is generated (scripts/ext13_p5_text.py): per-condition paragraphs, headline list and contrasts
  are numbers-driven from results/ext13base_summary.json; static text in scripts/ext13_p5_notes.json
## Running (detached, no agent needed)
- scripts/ext13_p5_chain_base.sh (PID 2527986; log logs/ext13_p5_chain.log): cp_nat -> cp_vt_orig -> cp_vt_neu -> vt_e2a_orig -> vt_e2a_neu
  -> vt_a2e_orig (Qwen3; others behaviour-only, skipped) -> f2e, both models. Per cond/proto: 7-8 queued jobs, then
  scripts/ext13_p5_analyze.py (base_summary, ext13base_* tables/figures, registry selected_expert / pop_top10 / status.base,
  PROGRESS "[p5-data] BASE READY <cond>/<proto>", refreshes behaviour tables + the section). Ends with "chain_base done" in the log.
## How to resume / check
- grep "rc=[1-9]" logs/ext13_p5_chain.log -> a failed stage stops that cond/proto only; fix, then
  `nohup bash scripts/ext13_p5_chain_base.sh <cond> > logs/ext13_p5_chain_base.out 2>&1 &` (skips stages with .done_<stage> markers).
- never redo GPU work: markers results/p5_<cond>_<proto>/.done_<stage>, run_meta.json "complete".
