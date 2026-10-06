# ext10-circuit status (Phase 4, Direction 10: head + expert add-back to full repair)

Updated: 2026-10-05T10:25Z

## Plan
- Step 1 (existing engine kinds): single-head attn_head patches at every (layer, head) on the ext8 validation rows
  (cf_qwen3 432 rows, cf_mixtral 436, wino_qwen3 256, wino_mixtral 256) + attn_layer reference rows per layer;
  patch-free head DLA from DiagSpec.attn_heads_final (all layers) + expert DLA (contrib_dla) of the same pass.
  Driver scripts/ext10_heads_run.py (resumable: row chunks = passes, state in results/<run>/heads_state.json),
  chain scripts/ext10_chain1.sh (gpu_queue jobs ext10-heads-<run>), analysis scripts/ext10_heads_analyze.py.
- Step 2 (after the ext9 engine, E4b attn_head steps in multi): greedy heads+experts, head-only greedy, static mixed orderings.

## Status
- [x] Step 1 code (moetrace/ext10_circuit.py, scripts/ext10_heads_run.py, scripts/ext10_heads_analyze.py)
- [x] Step 1 GPU runs (chain1 06:20-07:19Z; cf_qwen3 3 passes, wino_qwen3 2, cf_mixtral 4, wino_mixtral 3); IOI Qwen3 step 1 done 07:50Z (2 passes)
- [x] Step 1 analysis (scripts/ext10_heads_analyze.py -> summary key step1, tables ext10_heads_*, figures ext10_heads_*)
- [x] Step 2: smoke OK 07:15Z; scripts/ext10_chain2.sh relaunched 07:42Z (10 passes per job); Qwen3 (cf, wino, ioi) COMPLETE 09:04Z (19 passes, results/qwen3_circuit), analysis done; Mixtral (cf, wino) COMPLETE 10:10Z (19 passes, results/mixtral_circuit); analysis + section done (logs/ext10_chain2.log): IOI Qwen3 step 1 -> step 2 qwen3 (cf,wino,ioi) -> step 2 mixtral (cf,wino) -> analysis -> text

## Resume
"resume from disk, do not redo GPU work": rerun the chain; finished passes are skipped (heads_state.json).

- Section draft: scripts/ext10_circuit_text.py -> results/sections/ext10_circuit.md (step 1 parts filled; step-2 parts appear when the summary has key step2).

## DONE 2026-10-05T10:25Z
All work complete: section results/sections/ext10_circuit.md, summary results/ext10_circuit_summary.json. GPU ~79 min. Not done: Mixtral IOI (budget). Nothing committed.
