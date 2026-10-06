# ext9-knockout status (resume from disk; do not redo GPU work)
Updated 2026-10-05 ~07:00Z.
## Part 1 engine E4: DONE
- moetrace/engine.py merged (md5 0f236b890a3f7fe70bba037208e72988 = moetrace/engine_ext9_dev.py). Backup of the previous engine: logs/ext9_engine_before_merge.py.bak.
- Regression after merge (logs/ext9_regress_post_merge_*.json, copied to results/): identical in every result field (scripts/ext9_regress_compare.py); only verify_ext8 peak-memory fields differ (<1 MB; nondeterministic, see that script's docstring). The first merge attempt (scripts/ext9_merge_regress.sh) auto-reverted because ext8_regress_compare counts memory fields; re-merged by hand 06:58Z.
- PROGRESS "ENGINE READY (ext9)" appended 06:58Z; API logs/ext9_engine_api.md.
- Full verification queued: job ext9-engine-verify -> results/verify_ext9_engine_olmoe.json (log logs/ext9_engine_verify.log). Dev smoke: scratchpad verify_ext9_smoke.json.
## Part 2 knockout: RUNNING
- OLMoE driver smoke queued (job ext9-ko-smoke-olmoe, results/olmoe_knockout_smoke; log in scratchpad ext9_smoke_olmoe.log).
- Chain scripts/ext9_chain.sh running (logs/ext9_chain.log): waits for the smoke, then jobs ext9-ko-<model>-<phase> for phases full, random, samelayer, final; then analysis + text. Resumable: rerun `nohup setsid bash scripts/ext9_chain.sh > logs/ext9_chain.log 2>&1 &` (EXT9_SKIP_SMOKE_WAIT=1 to skip the smoke gate); finished passes are skipped (plan frozen in results/<run>/plan_<phase>.json).
- NEXT: when done, inspect results/ext9_knockout_summary.json, write the reading in scripts/ext9_knockout_text.py, final report.
- 07:55Z: full verification done (results/verify_ext9_engine_olmoe.json). OLMoE smoke OK (analysis fixed). Qwen3 full phase DONE (17 min GPU; 240k-token passes OOM'd sometimes -> auto-split; budgets lowered for later plans: Qwen3 200k/12k rows, Mixtral 140k). Mixtral full phase running. Final phase moved to the sub scope (targets final-only + zero, pop-set final-only). Qwen3 controls.json built.
- 08:45Z: Mixtral full DONE (20 min), Qwen3 random DONE (8 min). Remaining: Mixtral random, Qwen3/Mixtral samelayer, Qwen3/Mixtral final (chain running). Analysis code tested on partial data (scratchpad ext9_qm_summary.json); reading in scripts/ext9_knockout_text.py generated from the summary.
- Noise: rows are not bit-reproducible across pass composition; null condition (sub scope) gives F ~ +0.008 WG / -0.003 CF (Qwen3); per-item cross-pass sd 0.42 (WG) / 0.19 (CF) logits, same order within pass (routing flips), so it is intrinsic bf16 noise.
- 10:25Z: samelayer DONE for both models (Qwen3 9.4 min, Mixtral 11.6 min). Remaining: final phase (Qwen3, Mixtral); then the chain runs scripts/ext9_knockout_analyze.py + scripts/ext9_knockout_text.py into results/. After that: review section, PROGRESS DONE line, final report.
- 10:40Z: ALL DONE. Section results/sections/ext9_knockout.md, summary results/ext9_knockout_summary.json. Nothing running. Final report sent to the coordinator.
