# ext8-addback status (agent ext8-addback) — 2026-10-04T11:15Z

## done
- Step E: engine extension merged into moetrace/engine.py (ENGINE READY (ext8) in PROGRESS.md 08:00Z); regression identical;
  results/verify_ext8_engine_olmoe.json.
- A0-A3 on CounterFact STR and WinoGrande STR, Qwen3 + Mixtral BOS: runs results/{qwen3_str,mixtral_bos_str,wino_qwen3_str,
  wino_mixtral_bos_str}_addback all complete (run_meta complete=true).
- Analysis: python scripts/ext8_addback_analyze.py (writes results/tables/ext8_*, results/figures/ext8_*,
  results/ext8_addback_summary.json); section: python scripts/ext8_addback_text.py -> results/sections/ext8_addback.md (fully
  templated from the summary JSON; re-run both after any change).

## running
- scripts/ext8_addback_chain_mixtral.sh has one queued no-op job (state complete); nothing else.

## next
- Nothing; awaiting coordinator review. Not committed (coordinator commits).
