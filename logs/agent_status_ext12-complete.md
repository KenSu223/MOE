# agent_status ext12-complete (Phase 4, Direction 12: completeness checks 4a / 4b / 4c)

Started 2026-10-05T06:05Z. Resume from disk: read this file, logs/ext12_*.log, results/*_rep*/run_meta.json,
results/wino_roleitems_*/run_meta.json, results/wino_mixtral_nobos_str*/run_meta.json. Do not redo GPU work (all runners resume or skip completed stages).

## Plan
- 4b inputs: scripts/ext12_roleitems_casesets.py -> results/wino_roleitems_inputs/case_sets_<proto>.json + link_<proto>.parquet (CPU)
- chain scripts/ext12_chain_inputs.sh (GPU, via gpu_queue): 4c sweep + expert (Mixtral no BOS, main+rep), 4b option-swap W2/W4/direct on role items (Qwen3, Mixtral BOS), 4c joint + direct, 4c sink flags
- 4a + 4c add-back: scripts/ext12_addback_run.py (multi-task driver over ext8 machinery; one pass serves several tasks of one model)
  Qwen3: wino rep (rep_validation curves, ranking from rep_discovery) + CF fold swap; Mixtral: wino rep BOS + CF fold swap BOS + wino no-BOS main
- analysis scripts/ext12_analyze.py -> results/ext12_complete_summary.json, tables/figures ext12_*; text scripts/ext12_complete_text.py -> results/sections/ext12_complete.md

## Status
- 06:05Z reading done; writing 4b case sets + input chain
- 06:08Z launched scripts/ext12_chain_inputs.sh (logs/ext12_chain_inputs.log); 4c sweep done 06:13 (no BOS: MoE L20, attn L13, block L19)
- 06:14Z OLMoE smoke of scripts/ext12_addback_run.py (two copies of an OLMoE CF fold-swap task, scratchpad) part a OK (2 passes); part b (resume to the end) queued
- written: moetrace/ext12_complete.py (task registry/loaders), scripts/ext12_addback_run.py, scripts/ext12_chain_addback.sh, scripts/ext12_analyze.py (4a/4b/4c), scripts/ext12_sink_flags.py, scripts/ext12_roleitems_casesets.py
- NEXT: after smoke b passes -> nohup setsid bash scripts/ext12_chain_addback.sh all > logs/ext12_chain_addback.log 2>&1 &
- 06:27Z smoke b OK (resume to completion; the two identical OLMoE tasks agree to bf16 noise: ceiling 0.2799 vs 0.2794, oracle AUC 0.364 vs 0.365); analysis smoke OK (ext8 analyse point == bootstrap point)
- 06:22Z 4c expert pass done; 4c W6 (CPU): no BOS selects L20E000 again (Spec +0.95 [0.84, 1.06], equal-norm +0.154), replicated on rep; W2 share 0.349 vs BOS 0.344
- 06:30Z launched scripts/ext12_chain_addback.sh all (logs/ext12_chain_addback.log); Qwen3 group running (pass ~56 s)
- 06:37Z 4b Qwen3 option sweep on role items done: share role 0.304 vs option 0.306 on 282 identical twins (diff -0.002 [-0.015, +0.009]) -> Phase-3 role/option difference was an item (names) confound
- 07:08Z 4a Qwen3 group COMPLETE (results/wino_qwen3_str_addback_rep, results/qwen3_str_addback_fold_rep); analysed (results/ext12_complete_summary.json, tables ext12_4a_*); replicates
- 07:09Z 4b Qwen3 complete (sweep/joint/direct in results/wino_roleitems_qwen3_str); renaming identity found (role-swapped prompt == option-swapped prompt with names exchanged, 512/512)
- text generator scripts/ext12_complete_text.py drafted (reading() data-driven; re-check generic claims at the end)
- WAITING: input chain (4b Mixtral sweep/joint/direct, 4c joint/direct/sink) and add-back chain (Mixtral group: wino rep BOS + CF fold BOS + wino no BOS)
- FINAL STEPS: python scripts/ext12_analyze.py ; python scripts/ext12_complete_text.py ; check claims ; PROGRESS line ; SubagentHandback report
- 08:52Z Mixtral add-back group COMPLETE (15 passes); 4a fully analysed: all replicate (tables ext12_4a_*)
- 09:05Z 4b COMPLETE both models (tables ext12_4b_*); 09:54Z 4c joint + direct done (W4 identical to BOS: A_dir 0.288 vs 0.288, M 0.795 vs 0.788)
- REMAINING: 4c sink-flags job (queued), then: python scripts/ext12_analyze.py ; python scripts/ext12_complete_text.py ; PROGRESS DONE line ; final report
- 10:37Z ALL DONE: section results/sections/ext12_complete.md, summary results/ext12_complete_summary.json, PROGRESS DONE line written. Nothing left to run.
