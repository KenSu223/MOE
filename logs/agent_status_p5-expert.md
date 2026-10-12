# p5-expert status (Phase 5: knockout + writer per condition)

- 2026-10-11T06:05Z started. Read CLAUDE.md, RESEARCH_PLAN Phase 5 (review, P5-8, task list, execution plan), ext9/ext11 machinery.
- Plan: moetrace/ext13ko.py + scripts/ext13ko_run.py (knockout, results/p5_<proto>_ko/<cond>/), moetrace/ext13wr.py +
  scripts/ext13wr_* (writer pass with contrib_dla + contrib_final_vectors; Part A/B; results/p5_<proto>_wr/<cond>/),
  smoke on results/wino_ref_F1_qwen3 (_smoke), then detached chains per model waiting for "[p5-data] BASE READY <cond>/<proto>".
- 06:20Z smoke OK (results/p5_qwen3_ko_smoke/f1pre, results/p5_qwen3_wr_smoke/f1pre; tables/figures *_smoke). Chains launched:
  logs/ext13ko_chain_{qwen3,mixtral_bos}.log (wait for BASE READY lines; stop with touch logs/ext13ko_<proto>.stop;
  logs/ext13ko_<proto>.nomore = no more conditions coming -> cross phase). Next: write text generators (sections).
- 06:30Z section writers ready (scripts/ext13ko_text.py, scripts/ext13wr_text.py; tested on smoke -> results/sections/*_smoke.md).
  DLA check: per-layer sum of contrib_dla / final_rms vs ext7_wino_dla moe_c / moe_k on the smoke prompts r 0.992 / 0.994
  (signs confirmed: corrupted twin's own-direction DLA = minus the clean direction). Waiting for p5-data BASE READY lines.
- 07:58Z f1/qwen3 BASE READY -> planned (66 interventions, 19,968 rows, 5 passes; selected L37E032 = p5-data's, pop top-10 from the registry), job queued. Fixed entry() for the registry's dict-valued selected_expert. vt_a2e_neu Qwen3 and vt_a2e Mixtral are behaviour-only per p5-data: touch logs/ext13ko_<proto>.nomore after p5-data's last BASE READY to trigger the cross phase.
- 2026-10-11T13:51:35Z resumed after usage limit (no GPU redone). Chains alive (f1, dir_f1, dir_opt done for both models). Sections now regenerated automatically: scripts/ext13wr_analyze.py (called by the chains after every finished condition) runs ext13ko_text.py + ext13wr_text.py -> results/sections/ext13_knockout.md, ext13_writer.md; P5-8 readings (cp_* vs vt_e2a_*/dir_opt, paired on shared source items) are generated once cp_* exist. New detached watcher scripts/ext13ko_nomore_watch.sh (logs/ext13ko_nomore_watch.log) touches logs/ext13ko_<proto>.nomore when every non-behaviour-only condition is finished -> chains run the cross phase and the final analysis, then exit. Nothing needs the agent alive.
