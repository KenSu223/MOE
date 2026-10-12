# agent p5-atp status (attribution-patching pilot on OLMoE, P5-5)

Owned: moetrace/ext13atp*.py, scripts/ext13atp_*, results/p5_atp_*, results/tables|figures/ext13atp_*, results/sections/ext13_atp.md

## Steps
- [x] 1 code: moetrace/ext13atp.py (tasks, HF autograd estimates, patched runs), scripts (build, engine, grad, fp32, analyze)
- [x] 2 GPU J1 engine: IOI OLMoE scan + case sets (wino, ioi) + activation-patching truth
- [x] 3 GPU J2 HF: gradients / IG / patched runs (routing) per task
- [x] 4 CPU fp32 check
- [x] 5 GPU J3 engine add-back curves
- [x] 6 analysis, tables, figures, section

## Log
- 2026-10-11T14:26Z chain launched detached: scripts/ext13atp_chain.sh (log logs/ext13atp_chain.log, status logs/ext13atp_chain.status). Relaunch is safe (skips finished outputs).
- IOI OLMoE scan: 1571/1600 margin pairs; case sets results/p5_atp_data/case_sets_{wino,ioi}.json (64/64 pairs each, Phase-3 splits).
- fp32 CPU checks running detached (logs/ext13atp-fp32-{wino,ioi}.log), 16 validation cases per task, ~9 s/case.
- 14:36Z J1 done (engine truth wino+ioi, 18 s / task; smoke HF run 58 cases). 14:58Z J2 HF wino COMPLETE (583 s, 2.5-3 s/case). fp32 CPU checks (16 cases grads, 8 cases patched) done for wino.
- Analysis script scripts/ext13atp_analyze.py runs (wino only so far): tables results/tables/ext13atp_*, figures results/figures/ext13atp_*, results/ext13atp_summary.json.
- Remaining: J2 ioi (queued), J3 add-back (chain), analysis both tasks, section results/sections/ext13_atp.md, PROGRESS line, final report.
- 15:40Z ALL DONE: J3 add-back done (15:15Z), analysis + section results/sections/ext13_atp.md written (scripts/ext13atp_text.py). Nothing left to run. GPU 28 min.
- 2026-10-12T16:30Z [coordinator] Phase 5 COMPLETE: all chains finished 05:40Z, results committed and pushed (bed8247). Nothing left to run for this agent.
