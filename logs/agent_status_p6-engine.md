# p6-engine status

Updated: 2026-10-17T06:50Z — DONE (final report sent). No chain of mine is running.

## Handshakes (logs/PROGRESS.md)
- HEADMASK READY 2026-10-16 20:04:50Z; NEURONS READY 20:31:01Z.
- MODEL READY mistral7b_bos 20:31Z, qwen3_4b / qwen3_8b 22:20Z, qwen2_57b 2026-10-17 02:36Z, qwen2_7b 04:03Z, qwen3_14b 04:50Z,
  qwen3_1p7b 05:24Z.
- PROTOCOL qwen2_57b fp32 and PROTOCOL qwen2_7b fp32 06:46Z (models.py act_dtype='fp32', applied by Engine(repo)).

## Engine (main tree; three merges, each with identical regression)
- merge 1 (2026-10-16 19:54Z): dense families, qwen2_moe + shared expert (index 64), head masks, route_mask_inherit, neurons,
  diagnostics, fp32 engine option, models.py entries.
- merge 2 (21:35Z): shared-expert knockout (route_mask entry (l, 64), zero mode).
- merge 3 (2026-10-17 02:59Z): model numerics protocols (models.py router_fp32 / act_dtype, ArchSpec.router_fp32).
- Backups: logs/ext14_engine_before_merge{,2,3}/. Worktree /home/ubuntu/MOE_wt_p6engine (branch p6-engine) = main copies.

## Docs
- logs/ext14_engine_api.md (API, Qwen2 protocol, verification table, runner compatibility).

## Open items (not engine work)
- ext5_subject (T4 grid) and scripts/ext7_wino_dla.py:89 hard-code bf16 (grid would fail with an fp32 engine; out of scope now).
- Qwen2-57B fp32 row budgets are estimates (<= 12k spawn rows, <= 256 prefill rows at T <= 32); calibrate on the first cell.
