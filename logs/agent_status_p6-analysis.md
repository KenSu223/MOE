# p6-analysis status

Updated: 2026-10-17T16:03Z

## Done
- `moetrace/ext14_analysis.py`: CellData (pair runs + CounterFact donor-level anchor), shares under LD / normalised LD / dp,
  A_direct, M_direct, all-MoE M, DLA, clean margin, hand-off from the grid (bootstrap CI of the layer), head singles,
  paired / joint cluster bootstrap on common items (item = src_pair_id across levels, pair id across models), ladder,
  Dial E (+ E13 H-test), Dial A (+ 5-bin dose-response), Set C, R-metric, R-site, twins, series, predictions evaluation.
- `scripts/ext14_all.py` (summary JSON, tables, figures, sections via `scripts/ext14_analyze_sections.py`): smoke-run on
  Phase-5 stand-ins (shares reproduce Phase 5 exactly: dir_f1 0.71/0.52, dir_opt 0.19/0.35, IOI 0.92/0.80, CF 0.54/0.58).
- Tier 2: `scripts/ext14_t2_common.py` (mirror dirs + in-process patches of ext13ab / ext13ci / ext13ko / ext13wr),
  `ext14_t2_select.py`, `ext14_t2_run.py`, `ext14_t2_analyze.py`, `ext14_t2_chain.py`. GPU smoke on Phase-5 runs
  (results/p6_*_t2_smoke): ab, ci1, ci2, ko+wr on Qwen3, ab + ko on Mixtral all rc 0; selection reproduces Phase 5.
- R-causal: `scripts/ext14_causal_run.py` (masks injected into the unchanged ext7 runners via a PrefillSpec subclass),
  `scripts/ext14_causal_analyze.py`. REQUEST to p6-engine posted (spawns on masked prefill rows).

## Running chains
- `scripts/ext14_t2_chain.py --no-t2` (v4, pid 2526571, log logs/ext14_t2_chain.log): R-causal only (its job
  p6an-causal-e3_e2a-qwen3 waits at low queue priority since 09:33Z; then e3_e2a/mixtral_bos), then ext14_causal_analyze.py.
- `scripts/ext14_analyze_chain.py` (pid 2351975, log logs/ext14_analyze_chain.log): CPU refresher, re-runs scripts/ext14_all.py on
  every BASE READY / registry / causal / tier-2 change (the R-causal chain only refreshes between its GPU jobs). Stop:
  touch logs/ext14_analyze_chain.stop.

## Done since the restart (2026-10-17)
- ext14_all crash fixed (None-safe formatting, guarded table blocks / sections, CF anchor via anchor_runs.sweep_attention,
  cache version); pooled / behaviour-only rules; P3 frame (e3p3_a2e) in Dial E; E2 behavioural level (behaviour_rates);
  Qwen2-57B shared expert as a component (engine-result patch, task augmentation, shared singles stage); writer vocab + answer
  assert; depth-normalised interior share; pooled-split tier 2.
- Tier 2 done: e1_a2e/qwen3 (L37E032, Spec +0.21; mixed r(20) 0.93; KO F +0.06), e1_a2e/mixtral_bos (L17E007; r(20) 1.17; KO F -0.01).

- Scope cut (coordinator 02:07Z): tier 2 kept only for e1_a2e qwen3 / mixtral_bos (finished); e1_e2a/qwen3 expert singles kept
  (no selection); R-causal minimal (e1_a2e, e3_e2a; zero-mode KO of top-3 heads / experts + 3 matched random draws; sweep + joint only;
  e3_e2a experts from p5_dir_opt_*); D6 from p6-data's joint_shared_rows (moetrace.ext14_analysis.shared_routed, table ext14_qwen2_shared).
- R-causal smoke (Phase-5 stand-in, results/p6_causal_e3_e2a_qwen3_smoke): sweep under head masks OK; joint needed --splits
  validation (fixed); rerun queued (scratchpad causal_smoke2*.sh).

- (06:45Z resume) R-causal e1_a2e done for both models (R3: Qwen3 failed — attention AUC+ −0.23 vs random but share only
  −0.02; Mixtral held −0.31 / −0.07; R4 held in both: MoE AUC+ −0.21 / −0.27 vs random). e3_e2a waits for BASE READY + heads.
- Twin / series comparisons prefer the all-strata <cell>_all variant (coordinator 03:19Z); Stage-2 cells flow in automatically.

## Next
1. R-causal e3_e2a (both models) when the queue reaches it; R-site when rsite_e1_a2e is BASE READY (both automatic).
2. Sections ext14_{ladder,dial,dense,scale}.md regenerate automatically (data-driven readings); final report sent 2026-10-17.

## Open problems
- R-causal needs masked wavefront rows — requested, approved by the coordinator; p6-engine adds head_mask (inherited) and
  route_mask_inherit (part of HEADMASK READY).
- Smoke outputs removed (results/p6_*_t2_smoke*, /opt/dlami/nvme/moe_p6wr); writer raw path now carries the smoke suffix.
