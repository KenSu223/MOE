# Brief: p6-analysis (tier 2, robustness / causal checks, cross-cell analysis, figures, sections)

Read logs/agent_brief_p6_common.md first. Its rules apply.

## Goal

Turn the tier-1 base runs of p6-data into the Phase-6 answers:
- the ladder v2 test;
- the Dial E / Dial A / Set C results;
- dense vs MoE and capacity (Stage 2).

Also run tier 2 on the selected cells and the robustness / causal checks. Write the figures, tables and sections, all re-runnable from disk.

## Files you own

- `scripts/ext14_t2_*.py` and `.sh` (tier-2 runners; copy and adapt `scripts/ext13ab_*`, `ext13ci_*`, `ext13ko_*`, `ext13wr_*`, `ext7_wino_expert.py` invocations; do not edit the ext13 / ext7 originals)
- `scripts/ext14_causal_*.py`, `scripts/ext14_analyze_*.py`, `scripts/ext14_all.py`
- `moetrace/ext14_analysis.py`
- `results/p6_<cell>_<proto>_t2/`, `results/p6_causal_*/`
- `results/figures/ext14_*`, `results/tables/ext14_*`, `results/ext14_summary.json`
- `results/sections/ext14_{ladder,dial,dense,scale}.md`
- `logs/ext14_t2_*.log`, `logs/ext14_analysis*.log`, `logs/agent_status_p6-analysis.md`

Not yours:
- `moetrace/ext14_plotutil.py` — the coordinator writes it with `random_equivalent(...)` for the "random k that matches greedy's first k" marks. Import it; every add-back figure must carry these marks.
- `results/tables/ext14_prereg_check.md` — the coordinator owns its definition. You produce its numbers in `results/ext14_summary.json` under the prediction IDs of `data/dial/predictions.json`.

## Inputs

- The registry `data/dial/p6_conditions.json` and the PROGRESS handshakes `[p6-data] BASE READY <cell>/<proto>`.
- Phase-3/4/5 anchors (CounterFact, IOI, dir_f1 / dir_opt), registered by p6-data.

Measures exactly as in Phase 5:
- attention share = AUC+(attention) / (AUC+ attention + AUC+ MoE / MLP) of the single-layer curves;
- A_direct and the direct split;
- all-MoE M;
- hand-off layers from the grid;
- head singles.

Read `scripts/ext13_p5_analyze.py`, `moetrace/ext13syn.py`, `scripts/ext13_posthoc.py` for the existing definitions and reuse them (import, do not fork logic where possible). Statistics: pair bootstrap (template-clustered where templates exist), validation split, paired intersections across levels / twins.

## Tasks

1. **Tier 2** (Stage 1, `qwen3` and `mixtral_bos`, cells `e1_a2e` and `e1_e2a`; the E3 tier-2 counterparts are the Phase-5 `p5_dir_f1_*` / `p5_dir_opt_*` runs):
   - expert singles and selection;
   - add-back: adaptive greedy / per-case oracle / random only, as in Phase 5, with the random-equivalent marks;
   - mixed head + expert greedy;
   - knockout with matched controls;
   - writer DLA.
   - Stage 2: the same on `qwen2_57b` for `e1_a2e` and `e3_a2e`, including the shared-vs-routed split.
   - Start each cell's chain on its BASE READY line.
2. **R-metric** (CPU): the attention share of every cell under LD, normalised LD and Δp (the sweeps record F5 metrics). Spearman of the cell rankings per model.
3. **R-site:** compare `rsite_e1_a2e` with `e1_a2e` (shares and grids).
4. **R-causal** (needs `[p6-engine] HEADMASK READY`), on `e1_a2e` and `e3_e2a` for both models:
   - knock out the top-3 mover heads (by the cell's single-head rescue) in both prefill rows, zero mode;
   - separately, the top-3 experts (`route_mask`, reroute);
   - each with 3 matched random draws (same layers for heads; same layers and activity for experts);
   - then rerun T2 (sweep) and T3 (joint / direct) under the knockout.
   - Report the change of attention AUC+, MoE AUC+, share and task margin vs random.
5. **Cross-cell analyses:**
   - ladder: predicted grade vs observed share per model, Spearman, cue distance as covariate;
   - Dial E: share per level and direction with the predicted bands from `data/dial/predictions.json` overlaid; the direction gap per level (paired items) — the H-inference vs H-type decision;
   - Dial A: share vs prior class and the 5-bin dose–response; MoE direct sign in conflict;
   - Set C: behaviour and shares per subset;
   - position grids: hand-off layer per level.
   - Stage 2:
     - MoE vs dense parallel per-layer curves (depth-normalised l / L) for both twins;
     - share-shift arrows per cell;
     - Qwen3 4B → 8B → 14B → 30B-A3B and Mistral-7B → Mixtral series;
     - Qwen2-57B shared vs routed;
     - the hypotheses D1–D6 with their operational criteria.
6. **Outputs:**
   - `results/ext14_summary.json`: every number keyed by cell / proto, plus a `predictions` block keyed by prediction ID with the observed value and CI;
   - tables `results/tables/ext14_*.md`, figures `results/figures/ext14_*.png` (and .pdf);
   - sections `ext14_ladder.md`, `ext14_dial.md` (with robustness and causal subsections), `ext14_dense.md`, `ext14_scale.md`, in the style of the Phase-5 sections (summary first, numbers with CIs, caveats, files).
   - `scripts/ext14_all.py` rebuilds all of them from disk. Hook it into your chains after every stage, and smoke-run it on the first BASE READY cell before arming the chains.

## Done when

The Stage-1 and Stage-2 analyses are complete for every cell that reached BASE READY, tier 2 and the robustness / causal checks are done, and the sections are written. Stage 3 (Mixtral-8x22B) is not part of this brief. Then send your final report.
