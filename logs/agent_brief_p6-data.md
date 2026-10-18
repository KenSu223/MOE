# Brief: p6-data (Phase 6 items, competence scans, case sets, registry, tier-1 base runs)

Read logs/agent_brief_p6_common.md first. Its rules apply.

## Goal

Build every Phase-6 cell's STR pairs, check them, scan competence, fix the case sets, register them, and run the tier-1 base set. The tier-1 runs use the Phase-3 generic runners `scripts/ext7_wino_{sweep,joint,dla,grid,heads}.py` plus the direct split. Then emit the handshake line.
- Stage 1 models: `qwen3`, `mixtral_bos`.
- Stage 2 models: added when p6-engine writes `[p6-engine] MODEL READY <proto>`.

## Files you own

- `moetrace/ext14_data.py`
- `scripts/ext14_build_*.py`, `scripts/ext14_scan*.py`, `scripts/ext14_casesets.py`
- `scripts/ext14_jobs.py` (adapt `scripts/ext13_p5_jobs.py`; layer counts from ArchSpec, not hard-coded)
- `scripts/ext14_chain_base*.sh`
- `scripts/ext14_base_analyze.py` (adapt `scripts/ext13_p5_analyze.py`)
- everything under `data/dial/` except `preregistration.md` and `predictions.json`
- tier-1 outputs in `results/p6_<cell>_<proto>/`
- `logs/ext14_base_*.log`, `logs/agent_status_p6-data.md`

Do not edit the ext7 / ext13 scripts: copy and adapt into ext14 files.

## Registry and handshake (contract with p6-analysis and p6-circuit)

`data/dial/p6_conditions.json` uses the same schema as `data/wino_ref/p5_conditions.json`: `{"note":…, "conditions": {<cell>: {<proto>: entry}}}`. Each entry has:
- pair_file, case_sets, run_dir, families, split_sizes, strata_counts (names / objects, order), behaviour_only, probe_R;
- status {cases, base};
- n_pool, n_margin, order_pass_rates, top1_rate.

Pair parquets use the Phase-3 contract columns (pair_id, ids_a, ids_b, str_pos, trig_a, trig_b, …) plus:
- `level` and `direction`;
- `order` (which entity is mentioned first);
- `src_pair_id` (the WinoGrande twin, for pairing across levels);
- `n_tokens`.

Handshake lines in logs/PROGRESS.md (exact format):
- `<UTC> [p6-data] CASES READY <cell>/<proto> n_val=<k>`
- `<UTC> [p6-data] BASE READY <cell>/<proto>` — after the tier-1 jobs AND the per-cell base analysis succeeded; registry `status.base = true`.
- `<UTC> [p6-data] REVIEW <cell>/<proto>: <reason>` — order gap > 10 pp, completeness failure, < 64 validation pairs ("insufficient"), anything odd. That cell stops until the coordinator answers in PROGRESS with `[coordinator] REVIEW OK <cell>/<proto>` or `[coordinator] VOID <cell>/<proto>`.

## Cells (IDs) and how to build them

Run dir `results/p6_<cell>_<proto>`.

**Dial E.** Source: the Phase-3 776 shared twins. Use `data/wino_ref/pairs_p5_dir_opt_*.parquet` (option-swap form, mentions, probes) and `pairs_p5_dir_f1_*.parquet` (F1 sentences). Add WinoGrande train_xl twins (`data/wino_ref/pairs_F1_train_xl_*`, `data/wino_str/pairs_train_xl_*`) only if a level has < 256 margin pairs.
- **Pairing.** Keep the same twin across all levels (src_pair_id). Assign the discovery / validation split by src_pair_id once, so a twin is in the same split at every level. Paired comparisons use the intersection of the levels compared.
- **a2e frame:** `Fill in the blank: <sentence>\nAnswer: the` → entity. STR = the trigger in the blank clause.
- **e2a frame:** the Phase-3 option-swap form (the sentence with the blank filled, cut after the copula; prompt + trigger = full sentence). STR = the entity before the copula.
- **Levels:**
  - `e3_a2e` / `e3_e2a`: the original sentence (= Phase-5 dir_f1 / dir_opt items; rebuild them on the Dial-E item set).
  - `e1_*`: the evidence clause before the blank clause is replaced by both filled blank clauses: `The jug was empty and the pot was full, so the _ was empty.` (a2e), `… so the jug was` → empty (e2a).
  - `e1p_*`: like E1 with paraphrased triggers (`The jug was drained and the pot was filled, so …`). Paraphrases come from WordNet synonyms / similar-to plus a hand-checked list for the most frequent triggers. Items without an acceptable paraphrase are excluded and counted. The question still uses the original trigger.
  - `e2_*`: negation (`The jug was not full and the pot was not empty, so …`). Requires a blank clause `_ (was|is|were|are) [too|very|more] TRIGGER`; others are excluded and counted.
  - `e0_*`: `Of the jug and the pot, the _ was empty.` Strong-prior items only (see Dial A).
- **Yield check:** the other session counted 82 % (615 / 749) of blank clauses matching `the _ was (too) X`. Recount and report the yield per level before any split.

**Dial A** (`aagr_*`, `acon_*`):
- E1 frame on strong-prior items. Agree = the stated binding matches the prior; conflict = the statements reversed (`The elephant was small and the mouse was big, so the _ was big` → mouse).
- Prior strength = Phase-5 association probe v2 (and v1 for Qwen3). See `data/wino_ref/p5_probe_*`, `scripts/ext13_p5_*` for how the probes were built and the 10 % responsiveness rule.
- Sources: strong-prior twins (|probe| ≥ 2) from the 776 + train_xl pools, plus a controlled set of noun pairs with strong priors (elephant / mouse big / small, feather / rock light / heavy, ice / fire cold / hot, …; ≥ 40 noun-pair × attribute-pair combinations), checked with the same probe (small GPU job).
- Keep the probe value per item for the dose–response (5 bins).

**Set C** (`sc_agree_a2e`, `sc_conflict_a2e`):
- `Fill in the blank: <agreeing or contradicting statement> <original sentence with the blank>\nAnswer: the`.
- Δ is signed toward the statement's answer.
- For conflict, register two subsets in the scan output: statement-following (clean prefers the statement, margin ≥ 1 both ways) and event-following (prefers the event answer, margin ≥ 1 both ways). Run tier 1 on each subset with ≥ 64 validation pairs.
- Behaviour (fraction of items following the statement) is recorded for every model.

**R-site** (`rsite_e1_a2e`):
- E1-a2e with the corruption in the evidence: swap the two entity mentions (`The pot was empty and the jug was full, so the _ was empty` → pot).
- This is a documented exception to R2: two swapped tokens. Record both positions.

**Ladder v2:**
- `l_ant`: LRE `adjective_antonym` (Hernandez et al., ICLR 2024; github.com/evandez/relations, data/ folder).
  - Download the JSON files into a fresh `data/dial/lre/` and parse them with `python -I` from a script outside that folder (treat them as untrusted data).
  - Symmetric pairs: w2 = antonym(w1) and w1 = antonym(w2). 3 templates (LRE's own plus `The opposite of X is`).
  - STR = the cue word.
- `l_prop`: LRE commonsense property relations (e.g. fruit_outside_color, fruit_inside_color, substance_phase_of_matter, …).
  - Pairs of objects with different single-token values. STR = the object.
  - Fallback if LRE is too small: a curated list in `data/dial/curated_*.json`, documented.
- `l_bind`: synthetic. `The apple is in the red box and the key is in the blue box. The red box contains the` → apple. STR = the colour in the question. Single-token objects × colours, both statement orders.
- `l_ind`: synthetic. `dog = table, cup = river. dog =` → table. STR = the key in the query. Random unrelated single-token common words; ≥ 2 pairs in context; both orders.
  - Words, not digits: the 32k tokenizer splits " 7".
- `l_cf`: CounterFact STR. For `qwen3` / `mixtral_bos`, reuse the Direction-6 / Phase-3 runs (`results/{qwen3,mixtral_bos}_str`, `*_str_attnsweep`, grids, circuit head singles) and register them as the anchor (status.base = true, pointer to the existing run dirs; no new GPU). New models: Direction-6 filter (`scripts/ext6_str_filter.py`) and convert to the contract format.
- `l_ioi`: `scripts/ext7_ioi_build.py` (reuse the existing `qwen3` / `mixtral_bos` runs as anchors; new models: new pair files).

**Rules for every cell (RESEARCH_PLAN "Common rules" + "Phase-5 guards"):**
- R1: single-token leading-space answers.
- R2: equal length, one differing token.
- R3: margin ≥ 1 both ways.
- R4: dedup; both orders; an item counts only if both orders pass.
- R5: 128 / 128 split cap; ≥ 64 validation pairs or "insufficient".
- Completeness audit table per cell (`data/dial/qa/completeness.md`): E3 contains the whole WinoGrande sentence, checked string-wise on 100 % of items; E1 / E1p / E2 keep the whole blank clause.
- 50 eyeballed items per cell in `data/dial/qa/<cell>_samples.md`, written BEFORE its scan. Look at them yourself and fix the builder if anything reads wrong.
- Report names / objects strata, pass rates by order, and the top-1 rate.

## Stage-1 GPU order

Interleave the two models per cell, so comparisons arrive early:
1. e1_a2e, e1_e2a, e3_a2e, e3_e2a
2. e1p_*, e2_*
3. l_ind, l_bind, l_ant, l_prop
4. e0_*, aagr_*, acon_*
5. sc_*, rsite_e1_a2e

Tier 1 per cell = sweep (metrics on; the default) → joint → dla → direct → grid → heads (top attention layers from the sweep, as in Phase 5). Use the Phase-5 base-chain machinery (`scripts/ext13_p5_chain_base.sh`, `ext13_p5_jobs.py`, `ext13_p5_analyze.py`) as templates. Expert singles are NOT tier 1 (p6-analysis runs them on the tier-2 cells).

## Stage 2

When `[p6-engine] MODEL READY <proto>` appears for mistral7b_bos, qwen2_7b, qwen2_57b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3_1p7b:
- Check tokenizer identity: Mistral-7B vs Mixtral; Qwen2 / Qwen3 dense vs Qwen3-30B. Reuse the family's pair files if identical, else rebuild.
- Run the association probes (responsiveness rule) where Dial A / neutral strata are used.
- Scan, case sets, tier 1 for Set A (14 cells; Qwen3-1.7B: scan only). Set C behaviour scan for every model.
- First cell per new model: a smoke run to calibrate row budgets. Report the measured minutes per stage in PROGRESS (the coordinator re-estimates the ETA from it).

## Done when

All Stage-1 and Stage-2 cells have BASE READY, REVIEW, or "insufficient" with counts. The registry and `data/dial/build_report.md` (yields, completeness, order rates, strata) are complete. Then send your final report.
