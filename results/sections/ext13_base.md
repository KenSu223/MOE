**Summary.** Phase 5's data and base analyses (agent p5-data). Every list below is generated from the per-run summaries and refreshes after each condition, so pending conditions appear when their base analysis finishes.

Headlines per condition (validation pairs; generated from `results/ext13base_summary.json`):

- `f1` (attribute → entity, fill-in-the-blank (task 1)): Qwen3 share 0.76, A_direct 1.20, expert L37E032 (A); Mixtral BOS share 0.56, A_direct 1.01, expert L20E001 (B).
- `dir_f1` (direction on identical items: fill-in-the-blank (task 2)): Qwen3 share 0.71, A_direct 1.08, expert L44E116 (A); Mixtral BOS share 0.52, A_direct 0.86, expert L20E001 (A).
- `dir_opt` (direction on identical items: Phase-3 option swap (task 2)): Qwen3 share 0.19, A_direct 0.05, expert L41E117 (A); Mixtral BOS share 0.35, A_direct 0.31, expert L20E000 (A).
- `cp_nat` (copy prefix on the natural Phase-3 twins (task 6)): Qwen3 share 0.21, A_direct -0.01, expert L41E117 (A); Mixtral BOS share 0.37, A_direct 0.38, expert L21E006 (B).
- `cp_vt_orig` (copy prefix on the verb templates, original verb (task 6)): Qwen3 share 0.31, A_direct 0.11, expert L39E001 (A); Mixtral BOS share 0.39, A_direct 0.47, expert L19E006 (A).
- `cp_vt_neu` (copy prefix on the verb templates, neutral verb (task 6)): Qwen3 share 0.26, A_direct 0.09, expert L42E123 (B); Mixtral BOS share 0.38, A_direct 0.43, expert L19E006 (B).
- `vt_e2a_orig` (verb templates, entity → attribute, original verb (task 4)): Qwen3 share 0.28, A_direct 0.13, expert L39E121 (A); Mixtral BOS share 0.41, A_direct 0.38, expert L19E006 (A).
- `vt_e2a_neu` (verb templates, entity → attribute, neutral verb (task 4)): Qwen3 share 0.22, A_direct 0.13, expert L42E123 (A); Mixtral BOS share 0.42, A_direct 0.42, expert L19E006 (A).
- `vt_a2e_orig` (verb templates, attribute → entity fill-in-the-blank, original verb (task 5)): Qwen3 share 0.79, A_direct 1.47, expert L37E032 (A).
- `f2e` (attribute → entity, natural sentence cut at a final blank (task 3)): Qwen3 share 0.69, A_direct 0.99, expert L43E036 (A); Mixtral BOS share 0.65, A_direct 1.35, expert L19E004 (A).

Contrasts (validation means; the CIs are in the summary table; only `dir_f1` / `dir_opt` share their pairs):

- Direction on identical pairs, attribute → entity vs entity → attribute (`dir_f1` vs `dir_opt`): Qwen3 share 0.71 vs 0.19, A_direct 1.08 vs 0.05, all-MoE M 0.11 vs 0.81; Mixtral BOS share 0.52 vs 0.35, A_direct 0.86 vs 0.31, all-MoE M 0.41 vs 0.75.
- Copy vs generate on natural twins (different item filters) (`cp_nat` vs `dir_opt`): Qwen3 share 0.21 vs 0.19, A_direct -0.01 vs 0.05, all-MoE M 0.79 vs 0.81; Mixtral BOS share 0.37 vs 0.35, A_direct 0.38 vs 0.31, all-MoE M 0.73 vs 0.75.
- Copy vs generate on templates, original verb (`cp_vt_orig` vs `vt_e2a_orig`): Qwen3 share 0.31 vs 0.28, A_direct 0.11 vs 0.13, all-MoE M 0.69 vs 0.82; Mixtral BOS share 0.39 vs 0.41, A_direct 0.47 vs 0.38, all-MoE M 0.54 vs 0.68.
- Copy vs generate on templates, neutral verb (`cp_vt_neu` vs `vt_e2a_neu`): Qwen3 share 0.26 vs 0.22, A_direct 0.09 vs 0.13, all-MoE M 0.74 vs 0.82; Mixtral BOS share 0.38 vs 0.42, A_direct 0.43 vs 0.42, all-MoE M 0.46 vs 0.62.
- Original vs neutral verb, entity → attribute (`vt_e2a_orig` vs `vt_e2a_neu`): Qwen3 share 0.28 vs 0.22, A_direct 0.13 vs 0.13, all-MoE M 0.82 vs 0.82; Mixtral BOS share 0.41 vs 0.42, A_direct 0.38 vs 0.42, all-MoE M 0.68 vs 0.62.
- Direction on the templates (Qwen3 only) (`vt_a2e_orig` vs `vt_e2a_orig`): Qwen3 share 0.79 vs 0.28, A_direct 1.47 vs 0.13, all-MoE M 0.21 vs 0.82.
- Natural sentence vs fill-in-the-blank, attribute → entity (`f2e` vs `f1`): Qwen3 share 0.69 vs 0.76, A_direct 0.99 vs 1.20, all-MoE M 0.21 vs 0.24; Mixtral BOS share 0.65 vs 0.56, A_direct 1.35 vs 1.01, all-MoE M -0.06 vs 0.23.

Data findings (behaviour section):

1. **Probe.** The pre-registered association rule uses both probes in Qwen3 but only v2 in Mixtral, whose v1 does not respond. Half of the margin pairs are neutral.
2. **Copy prefix.** The first copy prefix, `Of X and Y, one … t1 and the other … t2`, is answered by position ('respectively', X ↔ t1). Mixtral solves 43 % of the unrelated-verb template items in the congruent order and 0 % in the other. That prefix is void for tracing; the copy conditions use `Options: t1, t2.`, whose orders pass about equally (natural twins 0.73 / 0.74 Qwen3, 0.90 / 0.89 Mixtral).
3. **Task 5.** Fill-in-the-blank on the verb templates fails the margin (Qwen3 3.7 % / 1.9 %, Mixtral 0.1 % / 0.2 % vs 39 % / 33 % for the same items entity → attribute), so task 5 is traced for Qwen3 orig only (5 validation seeds).
4. **Memorisation.** Original WinoGrande names give no advantage over new names.
5. **Restated format.** It passes the margin on 9 % of the pairs.

Status: base analysis complete for 19 condition × model runs (f1/qwen3, f1/mixtral_bos, dir_f1/qwen3, dir_f1/mixtral_bos, dir_opt/qwen3, dir_opt/mixtral_bos, cp_nat/qwen3, cp_nat/mixtral_bos, cp_vt_orig/qwen3, cp_vt_orig/mixtral_bos, cp_vt_neu/qwen3, cp_vt_neu/mixtral_bos, vt_e2a_orig/qwen3, vt_e2a_orig/mixtral_bos, vt_e2a_neu/qwen3, vt_e2a_neu/mixtral_bos, vt_a2e_orig/qwen3, f2e/qwen3, f2e/mixtral_bos); pending: none; behaviour only: vt_a2e_orig/mixtral_bos, vt_a2e_neu/qwen3, vt_a2e_neu/mixtral_bos.

### What was run

Phase 5 (RESEARCH_PLAN.md "Phase 5 execution plan") asks whether the final-token attention share measured on WinoGrande in Phase 3 indexes the reasoning direction (entity → attribute vs attribute → entity, i.e. reference resolution), the availability of the answer in the context (copy vs generate), or the semi-synthetic verb templates' content. This section holds the data build and the Phase-3 base analyses of every condition; add-back, head + expert circuits, knockout and writers are in the sections of p5-addback, p5-circuit and p5-expert. Conditions (STR = the twin's own difference, both directions of every pair, Δ = logit(r) − logit(r′) at the final token):

- `f1` (task 1) — attribute → entity in the fill-in-the-blank format on all WinoGrande train_xl twins: `Fill in the blank: <sentence with _>\nAnswer: the` → " mirror"; STR = the trigger; pool = margin (Δ_A ≥ 1, Δ_B ≤ −1) & content swap & probe-neutral; Qwen3 128 / 128 + replication 128 / 128; Mixtral's neutral pool (203 pairs) is split half / half, 101 / 101, without replication.
- `dir_f1` / `dir_opt` (task 2) — direction on identical items: the 776 Phase-3 twins whose fill-in-the-blank pair passes the margin in the model, in the fill-in-the-blank format (attribute → entity) and in the Phase-3 option-swap format (fill the blank, predict the trigger: entity → attribute); same pairs, same seed-0 half / half split; no association filter (strata only).
- `f2e` (task 3) — attribute → entity in the natural sentence: twins that differ before the blank and have no word after it, cut at the blank; margin & content & neutral; half / half.
- `vt_e2a_{orig,neu}` (task 4) — the 55 curated entity → attribute verb templates × 48 IOI name pairs (Phase-3 format), original / neutral verb; `vt_a2e_{orig,neu}` (task 5) — the same items (identical name pairs, rng `ext13-verbs-<seed>`) in the fill-in-the-blank format (blank = entity, STR = trigger); seed split S = the 55 seeds (sorted ids) shuffled with random.Random(0), 27 discovery / 28 validation seeds, ≤ 128 margin pairs per split drawn round-robin over the split's seeds (within a seed by name-pair index, then attribute order).
- `cp_vt_{orig,neu}`, `cp_nat` (task 6, P5-8 copy vs generate) — the Phase-3 entity → attribute prompt preceded by `Options: {t1}, {t2}.` and a newline, which names both candidate attributes and pairs nothing with the names (coordinator decision 2026-10-11). Both attribute orders are separate pairs; an order preference is identical in both twins and cancels in the STR drop. Built with the Phase-3 builder on the prefixed twin. Checks: each prefix attribute is ONE token with the continuation's id (leading-space form after `:` / `,`), and token alignment holds. All 55 template seeds (orig / neutral / unrelated × 48 name pairs × 2 orders) and all 776 natural twins pass under both tokenizers. An item enters a case set only if BOTH orders pass the margin. Templates use the seed split S; cp_nat uses the twin's Phase-3 split (discovery / validation / replication). The first prefix, `Of X and Y, one {frame} t1 and the other {frame} t2.`, is void for tracing and kept as a behaviour result (`data/wino_ref/void_cp_respectively/`, table `ext13base_cp_respectively`).
- `mem` (task 7, behaviour only) — the 55 seeds with their ORIGINAL WinoGrande names × orig / neutral / unrelated verb (Phase-3 builder).
- Restated format (F2R): behaviour only.

pair_idx: natural twins = index of the HIT id in the sorted train_xl twins (as Phase 3); cp_nat = 2 · Phase-3 pair_idx + order − 1; templates = 1000 · seed rank + 2 · name-pair index + order − 1 (the same template item has the same pair_idx in vt_e2a, vt_a2e and cp order 1). Files: `data/wino_ref/pairs_p5_<cond>_<proto>.parquet` (pair contract of `moetrace/ext7_pairs.py` + strata `names`, `stratum_assoc`, `stratum_neutral`, `stratum_local_assoc`, `seed`, `variant`, `order`; token positions `mention_a_pos` / `mention_b_pos` (+ `blank_pos`; copy prompts `attr_a_pos` / `attr_b_pos` (prefix attributes), `prefix_len_tokens`)), `data/wino_ref/case_sets_p5_<cond>_<proto>.json`, registry `data/wino_ref/p5_conditions.json`. Base runners (the Phase-3 generic STR-pair runners, unchanged, through `scripts/gpu_queue.sh`; `scripts/ext13_p5_chain_base.sh`, job list `scripts/ext13_p5_jobs.py`): W2 sweep (MoE / attention / block at every layer), W4 joint (all-MoE, denoise and noise), DLA and the direct-path split, W6 experts (all layers, equal-norm pairs at L*), W5 heads (top-4 attention layers + a null layer), W3 grid (MoE / attention / residual, window 1, main family). Analysis `scripts/ext13_p5_analyze.py` (the functions of `scripts/ext7_wino_analyze.py` per run, as ext12 part 4c) plus a cluster bootstrap of the headline numbers with the condition's independent unit (seed for templates, Phase-3 twin for cp_nat, pair otherwise; 5,000 resamples, seed 0), the grid by site, and the copy-prefix attention of the top heads.

**Phase-5 conditions: split sizes and strata (data/wino_ref/case_sets_p5_<cond>_<proto>.json)**

| Condition | Model | Split | Pairs | Names | Objects | Associative | One-sided | Reverse | Neutral | No probe | Local assoc | Seeds | Flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1 | Qwen3-30B-A3B-Base | discovery | 128 | 62 | 66 | 0 | 0 | 0 | 128 | 0 |  |  |  |
| f1 | Qwen3-30B-A3B-Base | validation | 128 | 70 | 58 | 0 | 0 | 0 | 128 | 0 |  |  |  |
| f1 | Qwen3-30B-A3B-Base | rep_discovery | 128 | 67 | 61 | 0 | 0 | 0 | 128 | 0 |  |  |  |
| f1 | Qwen3-30B-A3B-Base | rep_validation | 128 | 60 | 68 | 0 | 0 | 0 | 128 | 0 |  |  |  |
| f1 | Mixtral-8x7B, BOS | discovery | 101 | 11 | 90 | 0 | 0 | 0 | 101 | 0 |  |  |  |
| f1 | Mixtral-8x7B, BOS | validation | 101 | 12 | 89 | 0 | 0 | 0 | 101 | 0 |  |  |  |
| dir_f1 | Qwen3-30B-A3B-Base | discovery | 90 | 8 | 82 | 6 | 26 | 19 | 39 | 0 | 21 |  |  |
| dir_f1 | Qwen3-30B-A3B-Base | validation | 90 | 6 | 84 | 4 | 16 | 20 | 50 | 0 | 7 |  |  |
| dir_f1 | Mixtral-8x7B, BOS | discovery | 94 | 0 | 94 | 5 | 21 | 27 | 41 | 0 | 19 |  |  |
| dir_f1 | Mixtral-8x7B, BOS | validation | 94 | 1 | 93 | 2 | 20 | 27 | 45 | 0 | 20 |  |  |
| dir_opt | Qwen3-30B-A3B-Base | discovery | 90 | 8 | 82 | 6 | 26 | 19 | 39 | 0 | 21 |  |  |
| dir_opt | Qwen3-30B-A3B-Base | validation | 90 | 6 | 84 | 4 | 16 | 20 | 50 | 0 | 7 |  |  |
| dir_opt | Mixtral-8x7B, BOS | discovery | 94 | 0 | 94 | 5 | 21 | 27 | 41 | 0 | 19 |  |  |
| dir_opt | Mixtral-8x7B, BOS | validation | 94 | 1 | 93 | 2 | 20 | 27 | 45 | 0 | 20 |  |  |
| cp_nat | Qwen3-30B-A3B-Base | discovery | 146 | 12 | 134 | 14 | 26 | 34 | 64 | 8 | 22 |  |  |
| cp_nat | Qwen3-30B-A3B-Base | validation | 140 | 10 | 130 | 8 | 14 | 56 | 60 | 2 | 12 |  |  |
| cp_nat | Qwen3-30B-A3B-Base | rep_discovery | 162 | 2 | 160 | 2 | 32 | 48 | 70 | 10 | 20 |  |  |
| cp_nat | Qwen3-30B-A3B-Base | rep_validation | 158 | 14 | 144 | 4 | 26 | 60 | 64 | 4 | 24 |  |  |
| cp_nat | Mixtral-8x7B, BOS | discovery | 216 | 20 | 196 | 0 | 48 | 54 | 94 | 20 | 22 |  |  |
| cp_nat | Mixtral-8x7B, BOS | validation | 204 | 16 | 188 | 6 | 36 | 70 | 78 | 14 | 22 |  |  |
| cp_nat | Mixtral-8x7B, BOS | rep_discovery | 214 | 10 | 204 | 0 | 32 | 50 | 112 | 20 | 30 |  |  |
| cp_nat | Mixtral-8x7B, BOS | rep_validation | 222 | 20 | 202 | 4 | 34 | 68 | 106 | 10 | 38 |  |  |
| cp_vt_orig | Qwen3-30B-A3B-Base | discovery | 128 | 128 | 0 | 0 | 14 | 24 | 90 | 0 |  | 14 |  |
| cp_vt_orig | Qwen3-30B-A3B-Base | validation | 128 | 128 | 0 | 0 | 8 | 19 | 101 | 0 |  | 13 |  |
| cp_vt_orig | Mixtral-8x7B, BOS | discovery | 128 | 128 | 0 | 0 | 16 | 10 | 102 | 0 |  | 9 |  |
| cp_vt_orig | Mixtral-8x7B, BOS | validation | 128 | 128 | 0 | 0 | 8 | 4 | 116 | 0 |  | 10 |  |
| cp_vt_neu | Qwen3-30B-A3B-Base | discovery | 128 | 128 | 0 | 0 | 14 | 20 | 94 | 0 |  | 12 |  |
| cp_vt_neu | Qwen3-30B-A3B-Base | validation | 128 | 128 | 0 | 0 | 8 | 28 | 92 | 0 |  | 8 |  |
| cp_vt_neu | Mixtral-8x7B, BOS | discovery | 128 | 128 | 0 | 0 | 2 | 6 | 120 | 0 |  | 8 |  |
| cp_vt_neu | Mixtral-8x7B, BOS | validation | 128 | 128 | 0 | 0 | 8 | 6 | 114 | 0 |  | 8 |  |
| vt_e2a_orig | Qwen3-30B-A3B-Base | discovery | 128 | 128 | 0 | 0 | 7 | 19 | 102 | 0 |  | 16 |  |
| vt_e2a_orig | Qwen3-30B-A3B-Base | validation | 128 | 128 | 0 | 0 | 7 | 12 | 109 | 0 |  | 17 |  |
| vt_e2a_orig | Mixtral-8x7B, BOS | discovery | 128 | 128 | 0 | 0 | 8 | 6 | 114 | 0 |  | 14 |  |
| vt_e2a_orig | Mixtral-8x7B, BOS | validation | 128 | 128 | 0 | 0 | 4 | 7 | 117 | 0 |  | 18 |  |
| vt_e2a_neu | Qwen3-30B-A3B-Base | discovery | 128 | 128 | 0 | 0 | 9 | 22 | 97 | 0 |  | 16 |  |
| vt_e2a_neu | Qwen3-30B-A3B-Base | validation | 128 | 128 | 0 | 0 | 10 | 18 | 100 | 0 |  | 12 |  |
| vt_e2a_neu | Mixtral-8x7B, BOS | discovery | 128 | 128 | 0 | 0 | 7 | 4 | 117 | 0 |  | 11 |  |
| vt_e2a_neu | Mixtral-8x7B, BOS | validation | 128 | 128 | 0 | 0 | 9 | 8 | 111 | 0 |  | 12 |  |
| vt_a2e_orig | Qwen3-30B-A3B-Base | discovery | 60 | 60 | 0 | 0 | 4 | 5 | 51 | 0 |  | 8 |  |
| vt_a2e_orig | Qwen3-30B-A3B-Base | validation | 37 | 37 | 0 | 0 | 4 | 7 | 26 | 0 |  | 5 |  |
| vt_a2e_orig | Mixtral-8x7B, BOS | discovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | 0 | behaviour only |
| vt_a2e_orig | Mixtral-8x7B, BOS | validation | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |  | 1 | behaviour only |
| vt_a2e_neu | Qwen3-30B-A3B-Base | discovery | 45 | 45 | 0 | 0 | 0 | 7 | 38 | 0 |  | 5 | behaviour only |
| vt_a2e_neu | Qwen3-30B-A3B-Base | validation | 5 | 5 | 0 | 0 | 1 | 2 | 2 | 0 |  | 3 | behaviour only |
| vt_a2e_neu | Mixtral-8x7B, BOS | discovery | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |  | 1 | behaviour only |
| vt_a2e_neu | Mixtral-8x7B, BOS | validation | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |  | 2 | behaviour only |
| f2e | Qwen3-30B-A3B-Base | discovery | 32 | 3 | 29 | 0 | 0 | 0 | 32 | 0 |  |  |  |
| f2e | Qwen3-30B-A3B-Base | validation | 32 | 6 | 26 | 0 | 0 | 0 | 32 | 0 |  |  |  |
| f2e | Mixtral-8x7B, BOS | discovery | 18 | 3 | 15 | 0 | 0 | 0 | 18 | 0 |  |  |  |
| f2e | Mixtral-8x7B, BOS | validation | 18 | 2 | 16 | 0 | 0 | 0 | 18 | 0 |  |  |  |


### Association probe (pre-registered rule)

Both probes respond in Qwen3: 23.0 % (v1) and 22.7 % (v2) of the trigger probes of its 1,569 margin pairs reach |probe| ≥ 1, against 7–9 % on the random-name templates, where nothing but name-specific bias can drive the probe. R = {v1, v2}, so a Qwen3 pair is neutral only if both triggers stay below 1 under both probes. In Mixtral v1 does not respond (2.0 %, SD 0.41: the flat v1 behaviour noted in the review is confirmed) while v2 does (37.3 %, SD 1.22; templates 5.6 %), so R = {v2}; neither model needed the 'none responds' fallback. The probes agree only moderately (r = 0.51 Qwen3, 0.28 Mixtral) and objects respond far more than names (Qwen3 29 % vs 13 %, Mixtral v2 40 % vs 11 %). Strata of the margin pairs (associative / one-sided / reverse / neutral): Qwen3 86 / 299 / 394 / 790, Mixtral 12 / 111 / 135 / 235. Neutral is the largest stratum in both models (50 % / 48 %) and covers most name pairs (Qwen3 407 of 597, Mixtral 39 of 46). 'Reverse' is the second largest: a trigger alone points to the other candidate by ≥ 1 under some responding probe. It overlaps with own-answer pointing under the other probe in 245 (Qwen3) and 98 (Mixtral) pairs, so reverse / one-sided / associative are noisy partial-input labels and serve as descriptive strata only. The neutral rule shapes the main sets of `f1` and `f2e`; for the other conditions the strata are recorded per pair (`stratum_assoc`) and reported, not filtered.

**Association probes v1 / v2 on the margin pairs of the valid attribute → entity formats (trigger probes of both twins; pre-registered rule: a probe responds if ≥ 10 % reach |probe| ≥ 1)**

| Model | Probe | Margin pairs (unique twins) | Share |probe| ≥ 1 | Names | Objects | F1 / F2E / F2R | SD | Responds (≥ 0.10) | Templates (random names): share |probe| ≥ 1 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | v1 | 1569 | 0.230 | 0.131 | 0.291 | 0.235 / 0.218 / 0.174 | 0.98 | yes | 0.072 |
| Qwen3-30B-A3B-Base | v2 | 1569 | 0.227 | 0.136 | 0.283 | 0.232 / 0.248 / 0.140 | 0.94 | yes | 0.093 |
| Mixtral-8x7B, BOS | v1 | 493 | 0.020 | 0.011 | 0.021 | 0.019 / 0.034 / 0.000 | 0.41 | no | 0.008 |
| Mixtral-8x7B, BOS | v2 | 493 | 0.373 | 0.109 | 0.400 | 0.386 / 0.349 / 0.075 | 1.22 | yes | 0.056 |


**Probe outcome and association strata of the margin pairs (F1 / F2E / F2R pooled, unique twins)**

| Model | r(v1, v2) | Responding | R (used) | Margin pairs: assoc. / one-sided / reverse / neutral / no probe | Names | Objects | Conflicts (reverse & own) |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 0.51 | v1, v2 | v1, v2 | 86 / 299 / 394 / 790 / 0 | 0 / 57 / 133 / 407 | 86 / 242 / 261 / 383 | 245 |
| Mixtral-8x7B, BOS | 0.28 | v2 | v2 | 12 / 111 / 135 / 235 / 0 | 1 / 3 / 3 / 39 | 11 / 108 / 132 / 196 | 98 |


### Summary across conditions and models

The table lists the validation pairs of every condition and model, with cluster-bootstrap CIs (unit = seed for templates, Phase-3 twin for cp_nat, pair otherwise). Attention share = AUC+(attention) / (AUC+(attention) + AUC+(MoE)) over the single-layer final-position curves (Direction-2b definition). A_direct / M_direct = direct path of all final-position attention / MoE outputs, with the exact final norm. All-MoE M = all final-position MoE outputs patched (denoising). Peaks are normalised (rescue / mean drop) at the validation argmax. The selected expert is the paper's two-stage, recurrence-first selection at the discovery MoE argmax: pattern A = one positive specific expert, B = layer effect without a specific expert. Stratified rows follow the table.

| Condition | Direction / format | Model | Val. pairs (units) | Attention share (AUC+) | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (val) | Attention peak (val) | Selected expert: Spec [95% CI], pattern | Top heads (val rescue) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f1 | attr2ent, fill-in-the-blank (entity readout) | Qwen3 | 128 (128 pairs) | 0.76 [0.73, 0.77] | 1.20 [1.14, 1.26] | -0.20 [-0.26, -0.14] | 0.24 [0.18, 0.30] | L37 0.137 | L39 0.292 | **L37E032** +0.47 [+0.39, +0.56], A; rep L37E032 | L42H10 +0.97, L39H29 +0.55, L43H24 +0.51 |
| f1 | attr2ent, fill-in-the-blank (entity readout) | Mixtral BOS | 101 (101 pairs) | 0.56 [0.53, 0.59] | 1.01 [0.91, 1.12] | -0.01 [-0.12, 0.09] | 0.23 [0.13, 0.32] | L20 0.128 | L16 0.287 | **L20E001** -0.01 [-0.08, +0.06], B | L21H11 +0.31, L19H12 +0.28, L16H1 +0.27 |
| dir_f1 | attr2ent, fill-in-the-blank (entity readout) | Qwen3 | 90 (90 pairs) | 0.71 [0.68, 0.73] | 1.08 [1.00, 1.17] | -0.08 [-0.17, 0.00] | 0.11 [0.05, 0.17] | L43 0.086 | L43 0.213 | **L44E116** +0.17 [+0.11, +0.24], A | L42H10 +0.53, L42H31 +0.48, L39H29 +0.45 |
| dir_f1 | attr2ent, fill-in-the-blank (entity readout) | Mixtral BOS | 94 (94 pairs) | 0.52 [0.50, 0.54] | 0.86 [0.77, 0.95] | 0.14 [0.05, 0.23] | 0.41 [0.35, 0.46] | L20 0.151 | L16 0.278 | **L20E001** +0.11 [+0.02, +0.20], A | L21H11 +0.49, L19H12 +0.42, L15H8 +0.37 |
| dir_opt | ent2attr, Phase-3 option swap (fill the blank, predict the trigger) | Qwen3 | 90 (90 pairs) | 0.19 [0.16, 0.24] | 0.05 [0.02, 0.08] | 0.95 [0.92, 0.98] | 0.81 [0.79, 0.83] | L41 0.185 | L42 0.038 | **L41E117** +0.46 [+0.29, +0.66], A | L38H18 +0.46, L38H21 +0.31, L42H30 +0.21 |
| dir_opt | ent2attr, Phase-3 option swap (fill the blank, predict the trigger) | Mixtral BOS | 94 (94 pairs) | 0.35 [0.33, 0.36] | 0.31 [0.29, 0.33] | 0.69 [0.67, 0.71] | 0.75 [0.73, 0.78] | L20 0.146 | L13 0.124 | **L20E000** +0.90 [+0.79, +1.02], A | L25H9 +0.63, L19H13 +0.42, L13H4 +0.21 |
| cp_nat | ent2attr, Phase-3 + copy prefix on the 776 Phase-3 twins | Qwen3 | 140 (70 Phase-3 twins) | 0.21 [0.20, 0.24] | -0.01 [-0.07, 0.04] | 1.01 [0.96, 1.07] | 0.79 [0.76, 0.81] | L43 0.163 | L41 0.059 | **L41E117** +0.42 [+0.35, +0.49], A; rep L43E081 | L41H27 +0.58, L42H30 +0.14, L41H0 +0.13 |
| cp_nat | ent2attr, Phase-3 + copy prefix on the 776 Phase-3 twins | Mixtral BOS | 204 (102 Phase-3 twins) | 0.37 [0.36, 0.39] | 0.38 [0.34, 0.41] | 0.62 [0.59, 0.66] | 0.73 [0.70, 0.76] | L20 0.167 | L13 0.153 | **L21E006** +0.07 [-0.02, +0.15], B; rep L20E000 | L25H9 +0.53, L19H13 +0.36, L13H18 +0.24 |
| cp_vt_orig | ent2attr, Phase-3 + copy prefix on verb templates | Qwen3 | 128 (13 seeds) | 0.31 [0.26, 0.38] | 0.11 [-0.01, 0.28] | 0.89 [0.72, 1.01] | 0.69 [0.57, 0.79] | L42 0.130 | L41 0.081 | **L39E001** +0.12 [+0.07, +0.17], A | L41H0 +0.43, L42H30 +0.41, L41H5 +0.27 |
| cp_vt_orig | ent2attr, Phase-3 + copy prefix on verb templates | Mixtral BOS | 128 (10 seeds) | 0.39 [0.33, 0.47] | 0.47 [0.39, 0.60] | 0.53 [0.40, 0.61] | 0.54 [0.38, 0.68] | L19 0.194 | L19 0.146 | **L19E006** +0.51 [+0.39, +0.62], A | L13H18 +0.41, L19H13 +0.23, L19H1 +0.19 |
| cp_vt_neu | ent2attr, Phase-3 + copy prefix on verb templates (neutral verb) | Qwen3 | 128 (8 seeds) | 0.26 [0.23, 0.38] | 0.09 [-0.10, 0.27] | 0.91 [0.73, 1.10] | 0.74 [0.62, 0.83] | L43 0.180 | L42 0.098 | **L42E123** -0.00 [-0.04, +0.03], B | L41H0 +0.57, L42H30 +0.35, L42H27 +0.23 |
| cp_vt_neu | ent2attr, Phase-3 + copy prefix on verb templates (neutral verb) | Mixtral BOS | 128 (8 seeds) | 0.38 [0.33, 0.42] | 0.43 [0.37, 0.52] | 0.57 [0.48, 0.63] | 0.46 [0.37, 0.63] | L21 0.171 | L13 0.132 | **L19E006** +0.05 [-0.03, +0.13], B | L13H18 +0.37, L19H13 +0.35, L19H12 +0.15 |
| vt_e2a_orig | ent2attr, Phase-3 option swap on verb templates | Qwen3 | 128 (17 seeds) | 0.28 [0.20, 0.36] | 0.13 [0.08, 0.17] | 0.87 [0.83, 0.92] | 0.82 [0.79, 0.86] | L43 0.168 | L42 0.049 | **L39E121** +0.24 [+0.14, +0.36], A | L42H30 +0.32, L38H3 +0.30, L41H27 +0.29 |
| vt_e2a_orig | ent2attr, Phase-3 option swap on verb templates | Mixtral BOS | 128 (18 seeds) | 0.41 [0.36, 0.46] | 0.38 [0.31, 0.47] | 0.62 [0.53, 0.69] | 0.68 [0.58, 0.76] | L19 0.184 | L19 0.152 | **L19E006** +0.62 [+0.50, +0.74], A | L13H18 +0.44, L19H13 +0.37, L19H12 +0.30 |
| vt_e2a_neu | ent2attr, Phase-3 option swap on verb templates (neutral verb) | Qwen3 | 128 (12 seeds) | 0.22 [0.17, 0.30] | 0.13 [0.05, 0.19] | 0.87 [0.81, 0.95] | 0.82 [0.77, 0.87] | L43 0.170 | L42 0.061 | **L42E123** +0.23 [+0.16, +0.30], A | L38H3 +0.35, L42H30 +0.29, L41H0 +0.22 |
| vt_e2a_neu | ent2attr, Phase-3 option swap on verb templates (neutral verb) | Mixtral BOS | 128 (12 seeds) | 0.42 [0.37, 0.48] | 0.42 [0.32, 0.55] | 0.58 [0.45, 0.68] | 0.62 [0.48, 0.75] | L19 0.174 | L19 0.144 | **L19E006** +0.29 [+0.19, +0.39], A | L13H18 +0.40, L19H13 +0.35, L19H12 +0.21 |
| vt_a2e_orig | attr2ent, fill-in-the-blank on verb templates | Qwen3 | 37 (5 seeds) | 0.79 [0.64, 0.80] | 1.47 [1.34, 1.55] | -0.47 [-0.55, -0.34] | 0.21 [0.19, 0.53] | L37 0.080 | L39 0.374 | **L37E032** +0.17 [+0.13, +0.21], A | L42H10 +1.13, L43H24 +0.54, L39H27 +0.51 |
| vt_a2e_orig | attr2ent, fill-in-the-blank on verb templates | Mixtral BOS | behaviour only |  |  |  |  |  |  |  |  |
| vt_a2e_neu | attr2ent, fill-in-the-blank on verb templates (neutral verb) | Qwen3 | behaviour only |  |  |  |  |  |  |  |  |
| vt_a2e_neu | attr2ent, fill-in-the-blank on verb templates (neutral verb) | Mixtral BOS | behaviour only |  |  |  |  |  |  |  |  |
| f2e | attr2ent, natural truncation, blank at the sentence end | Qwen3 | 32 (32 pairs) | 0.69 [0.60, 0.75] | 0.99 [0.87, 1.12] | 0.01 [-0.12, 0.13] | 0.21 [0.05, 0.35] | L37 0.072 | L42 0.209 | **L43E036** +0.15 [+0.07, +0.25], A | L42H10 +0.98, L43H28 +0.58, L42H12 +0.48 |
| f2e | attr2ent, natural truncation, blank at the sentence end | Mixtral BOS | 18 (18 pairs) | 0.65 [0.59, 0.70] | 1.35 [1.16, 1.55] | -0.35 [-0.55, -0.16] | -0.06 [-0.34, 0.19] | L19 0.084 | L19 0.247 | **L19E004** +0.35 [+0.16, +0.55], A | L20H21 +0.80, L18H10 +0.70, L16H0 +0.68 |


| Condition | Model | Stratum | Pairs (main, all) | Attention share | A_direct / drop | M_direct / drop | All-MoE M / drop |
|---|---|---|---|---|---|---|---|
| f1 | Qwen3 | names | 132 | 0.79 [0.77, 0.81] | 1.32 [1.28, 1.36] | -0.32 [-0.36, -0.28] | 0.34 [0.29, 0.40] |
| f1 | Qwen3 | objects | 124 | 0.73 [0.71, 0.75] | 1.11 [1.05, 1.17] | -0.11 [-0.17, -0.05] | 0.11 [0.06, 0.16] |
| f1 | Qwen3 | probe-neutral | 256 | 0.76 [0.75, 0.78] | 1.20 [1.16, 1.24] | -0.20 [-0.24, -0.16] | 0.21 [0.17, 0.26] |
| f1 | Mixtral BOS | names | 23 | 0.67 [0.63, 0.72] | 1.44 [1.26, 1.62] | -0.44 [-0.62, -0.26] | -0.11 [-0.40, 0.19] |
| f1 | Mixtral BOS | objects | 179 | 0.54 [0.52, 0.55] | 0.94 [0.87, 1.01] | 0.06 [-0.01, 0.13] | 0.29 [0.23, 0.35] |
| f1 | Mixtral BOS | probe-neutral | 202 | 0.55 [0.53, 0.57] | 0.99 [0.92, 1.06] | 0.01 [-0.06, 0.08] | 0.26 [0.19, 0.32] |
| dir_f1 | Qwen3 | objects | 166 | 0.68 [0.66, 0.70] | 1.03 [0.96, 1.09] | -0.03 [-0.09, 0.04] | 0.12 [0.08, 0.16] |
| dir_f1 | Qwen3 | probe-neutral | 89 | 0.74 [0.71, 0.75] | 1.20 [1.14, 1.27] | -0.20 [-0.27, -0.14] | 0.06 [0.00, 0.11] |
| dir_f1 | Qwen3 | not neutral | 91 | 0.65 [0.62, 0.68] | 0.91 [0.83, 0.99] | 0.09 [0.01, 0.17] | 0.20 [0.14, 0.26] |
| dir_f1 | Mixtral BOS | objects | 187 | 0.51 [0.49, 0.52] | 0.83 [0.77, 0.90] | 0.17 [0.10, 0.23] | 0.41 [0.37, 0.44] |
| dir_f1 | Mixtral BOS | probe-neutral | 86 | 0.51 [0.49, 0.53] | 0.92 [0.83, 1.01] | 0.08 [-0.01, 0.17] | 0.39 [0.33, 0.44] |
| dir_f1 | Mixtral BOS | not neutral | 102 | 0.51 [0.49, 0.53] | 0.77 [0.69, 0.85] | 0.23 [0.15, 0.31] | 0.41 [0.36, 0.46] |
| dir_opt | Qwen3 | objects | 166 | 0.18 [0.16, 0.21] | 0.04 [0.02, 0.07] | 0.96 [0.93, 0.98] | 0.81 [0.80, 0.83] |
| dir_opt | Qwen3 | probe-neutral | 89 | 0.14 [0.12, 0.18] | 0.00 [-0.04, 0.04] | 1.00 [0.96, 1.04] | 0.80 [0.78, 0.82] |
| dir_opt | Qwen3 | not neutral | 91 | 0.23 [0.20, 0.27] | 0.11 [0.08, 0.14] | 0.89 [0.86, 0.92] | 0.82 [0.80, 0.83] |
| dir_opt | Mixtral BOS | objects | 187 | 0.34 [0.33, 0.35] | 0.31 [0.29, 0.32] | 0.69 [0.68, 0.71] | 0.76 [0.74, 0.77] |
| dir_opt | Mixtral BOS | probe-neutral | 86 | 0.33 [0.32, 0.34] | 0.28 [0.26, 0.30] | 0.72 [0.70, 0.74] | 0.78 [0.76, 0.80] |
| dir_opt | Mixtral BOS | not neutral | 102 | 0.35 [0.34, 0.37] | 0.33 [0.31, 0.35] | 0.67 [0.65, 0.69] | 0.74 [0.71, 0.76] |
| cp_nat | Qwen3 | names | 22 | 0.35 [0.30, 0.41] | 0.31 [0.19, 0.43] | 0.69 [0.57, 0.81] | 0.63 [0.53, 0.72] |
| cp_nat | Qwen3 | objects | 264 | 0.19 [0.18, 0.22] | -0.03 [-0.07, 0.01] | 1.03 [0.99, 1.07] | 0.79 [0.77, 0.80] |
| cp_nat | Qwen3 | probe-neutral | 124 | 0.19 [0.17, 0.24] | -0.06 [-0.14, 0.01] | 1.06 [0.99, 1.14] | 0.76 [0.72, 0.78] |
| cp_nat | Qwen3 | not neutral | 162 | 0.22 [0.21, 0.25] | 0.04 [-0.01, 0.08] | 0.96 [0.92, 1.01] | 0.78 [0.76, 0.81] |
| cp_nat | Mixtral BOS | names | 36 | 0.51 [0.45, 0.57] | 0.70 [0.58, 0.82] | 0.30 [0.18, 0.42] | 0.45 [0.30, 0.59] |
| cp_nat | Mixtral BOS | objects | 384 | 0.37 [0.36, 0.38] | 0.37 [0.34, 0.39] | 0.63 [0.61, 0.66] | 0.74 [0.72, 0.75] |
| cp_nat | Mixtral BOS | probe-neutral | 172 | 0.38 [0.36, 0.40] | 0.40 [0.36, 0.46] | 0.60 [0.54, 0.64] | 0.72 [0.67, 0.76] |
| cp_nat | Mixtral BOS | not neutral | 248 | 0.39 [0.37, 0.40] | 0.39 [0.36, 0.42] | 0.61 [0.58, 0.64] | 0.71 [0.68, 0.73] |
| cp_vt_orig | Qwen3 | names | 256 | 0.29 [0.25, 0.35] | 0.10 [-0.00, 0.23] | 0.90 [0.77, 1.00] | 0.69 [0.59, 0.77] |
| cp_vt_orig | Qwen3 | probe-neutral | 191 | 0.29 [0.24, 0.35] | 0.09 [-0.01, 0.21] | 0.91 [0.79, 1.01] | 0.71 [0.62, 0.78] |
| cp_vt_orig | Qwen3 | not neutral | 65 | 0.31 [0.24, 0.37] | 0.13 [-0.02, 0.32] | 0.87 [0.68, 1.02] | 0.65 [0.46, 0.78] |
| cp_vt_orig | Mixtral BOS | names | 256 | 0.42 [0.38, 0.47] | 0.53 [0.43, 0.65] | 0.47 [0.35, 0.57] | 0.52 [0.41, 0.62] |
| cp_vt_orig | Mixtral BOS | probe-neutral | 218 | 0.42 [0.37, 0.47] | 0.52 [0.43, 0.63] | 0.48 [0.37, 0.57] | 0.53 [0.42, 0.63] |
| cp_vt_orig | Mixtral BOS | not neutral | 38 | 0.45 [0.40, 0.51] | 0.63 [0.44, 0.79] | 0.37 [0.21, 0.56] | 0.46 [0.32, 0.60] |
| cp_vt_neu | Qwen3 | names | 256 | 0.28 [0.24, 0.36] | 0.11 [-0.03, 0.28] | 0.89 [0.72, 1.03] | 0.68 [0.56, 0.77] |
| cp_vt_neu | Qwen3 | probe-neutral | 186 | 0.27 [0.23, 0.35] | 0.10 [-0.04, 0.25] | 0.90 [0.75, 1.04] | 0.68 [0.59, 0.76] |
| cp_vt_neu | Qwen3 | not neutral | 70 | 0.32 [0.24, 0.40] | 0.15 [-0.03, 0.39] | 0.85 [0.61, 1.03] | 0.66 [0.46, 0.80] |
| cp_vt_neu | Mixtral BOS | names | 256 | 0.41 [0.37, 0.44] | 0.46 [0.38, 0.57] | 0.54 [0.43, 0.62] | 0.52 [0.39, 0.63] |
| cp_vt_neu | Mixtral BOS | probe-neutral | 234 | 0.41 [0.37, 0.44] | 0.46 [0.38, 0.56] | 0.54 [0.44, 0.62] | 0.52 [0.40, 0.64] |
| cp_vt_neu | Mixtral BOS | not neutral | 22 | 0.42 [0.37, 0.49] | 0.52 [0.38, 0.70] | 0.48 [0.30, 0.62] | 0.43 [0.28, 0.59] |
| vt_e2a_orig | Qwen3 | names | 256 | 0.27 [0.22, 0.33] | 0.11 [0.08, 0.14] | 0.89 [0.86, 0.92] | 0.82 [0.78, 0.85] |
| vt_e2a_orig | Qwen3 | probe-neutral | 211 | 0.27 [0.22, 0.33] | 0.11 [0.08, 0.14] | 0.89 [0.86, 0.92] | 0.82 [0.79, 0.85] |
| vt_e2a_orig | Qwen3 | not neutral | 45 | 0.30 [0.22, 0.38] | 0.10 [0.05, 0.14] | 0.90 [0.86, 0.95] | 0.81 [0.75, 0.87] |
| vt_e2a_orig | Mixtral BOS | names | 256 | 0.39 [0.36, 0.43] | 0.35 [0.30, 0.41] | 0.65 [0.59, 0.70] | 0.71 [0.65, 0.77] |
| vt_e2a_orig | Mixtral BOS | probe-neutral | 231 | 0.39 [0.36, 0.42] | 0.35 [0.31, 0.40] | 0.65 [0.60, 0.69] | 0.72 [0.66, 0.77] |
| vt_e2a_orig | Mixtral BOS | not neutral | 25 | 0.38 [0.31, 0.46] | 0.39 [0.27, 0.54] | 0.61 [0.46, 0.73] | 0.66 [0.48, 0.80] |
| vt_e2a_neu | Qwen3 | names | 256 | 0.25 [0.20, 0.31] | 0.12 [0.07, 0.16] | 0.88 [0.84, 0.93] | 0.81 [0.77, 0.85] |
| vt_e2a_neu | Qwen3 | probe-neutral | 197 | 0.24 [0.20, 0.30] | 0.11 [0.07, 0.15] | 0.89 [0.85, 0.93] | 0.82 [0.78, 0.85] |
| vt_e2a_neu | Qwen3 | not neutral | 59 | 0.27 [0.19, 0.36] | 0.13 [0.06, 0.18] | 0.87 [0.82, 0.94] | 0.80 [0.75, 0.86] |
| vt_e2a_neu | Mixtral BOS | names | 256 | 0.40 [0.36, 0.45] | 0.39 [0.33, 0.47] | 0.61 [0.53, 0.67] | 0.67 [0.57, 0.75] |
| vt_e2a_neu | Mixtral BOS | probe-neutral | 228 | 0.40 [0.36, 0.45] | 0.38 [0.32, 0.44] | 0.62 [0.56, 0.68] | 0.69 [0.61, 0.76] |
| vt_e2a_neu | Mixtral BOS | not neutral | 28 | 0.41 [0.36, 0.47] | 0.48 [0.34, 0.64] | 0.52 [0.36, 0.66] | 0.55 [0.37, 0.73] |
| vt_a2e_orig | Qwen3 | names | 97 | 0.76 [0.73, 0.80] | 1.51 [1.42, 1.59] | -0.51 [-0.59, -0.42] | 0.20 [0.16, 0.27] |
| vt_a2e_orig | Qwen3 | probe-neutral | 77 | 0.76 [0.72, 0.80] | 1.52 [1.45, 1.59] | -0.52 [-0.59, -0.45] | 0.19 [0.16, 0.26] |
| vt_a2e_orig | Qwen3 | not neutral | 20 | 0.78 [0.75, 0.80] | 1.45 [1.32, 1.57] | -0.45 [-0.57, -0.32] | 0.21 [0.14, 0.34] |
| f2e | Qwen3 | objects | 55 | 0.67 [0.60, 0.72] | 0.93 [0.85, 1.02] | 0.07 [-0.02, 0.15] | 0.08 [-0.01, 0.17] |
| f2e | Qwen3 | probe-neutral | 64 | 0.68 [0.62, 0.72] | 0.99 [0.91, 1.08] | 0.01 [-0.08, 0.09] | 0.15 [0.05, 0.24] |
| f2e | Mixtral BOS | objects | 31 | 0.67 [0.63, 0.70] | 1.30 [1.19, 1.43] | -0.30 [-0.43, -0.19] | -0.17 [-0.30, -0.07] |
| f2e | Mixtral BOS | probe-neutral | 36 | 0.67 [0.64, 0.70] | 1.32 [1.22, 1.43] | -0.32 [-0.43, -0.22] | -0.07 [-0.22, 0.07] |


#### f1: attribute → entity, fill-in-the-blank (task 1)

- Qwen3 (128 validation pairs, 128 pairs; drop 4.67): attention share 0.76 [0.73, 0.77], A_direct 1.20 [1.14, 1.26], M_direct -0.20 [-0.26, -0.14], all-MoE M 0.24 [0.18, 0.30]; peaks MoE L37 0.137, attention L39 0.292 of the drop; expert L37E032 (Spec +0.47 [+0.39, +0.56], pattern A; replication re-selects L37E032); Phase-3 WinoGrande expert L41E117: rescue +0.00, Spec -0.02 (discovery-active 1); top heads L42H10 +0.97, L39H29 +0.55, L43H24 +0.51; share names 0.79 (132 pairs) vs objects 0.73 (124); grid: MoE at the STR site peaks L0 0.15, attention at the final position L39 0.30.
- Mixtral BOS (101 validation pairs, 101 pairs; drop 4.32): attention share 0.56 [0.53, 0.59], A_direct 1.01 [0.91, 1.12], M_direct -0.01 [-0.12, 0.09], all-MoE M 0.23 [0.13, 0.32]; peaks MoE L20 0.128, attention L16 0.287 of the drop; expert L20E001 (Spec -0.01 [-0.08, +0.06], pattern B); Phase-3 WinoGrande expert L20E000: rescue +0.03, Spec -0.23 (discovery-active 48); top heads L21H11 +0.31, L19H12 +0.28, L16H1 +0.27; share names 0.67 (23 pairs) vs objects 0.54 (179); grid: MoE at the STR site peaks L0 0.48, attention at the final position L16 0.28.

Qwen3 and Mixtral differ in pool size and composition. Qwen3 has 1,451 margin & content pairs, 50 % neutral (names 70 / 128 validation pairs); Mixtral has 443, of which 203 are neutral (names 12 / 101). The replication family exists for Qwen3 only, and it reproduces every headline: share 0.78, A_direct 1.20, L37E032 re-selected with Spec +0.45. The grid by site (`ext13base_f1_w3_sites`) puts the decision read-out at the final position. Single attention layers there carry up to 0.29 (Qwen3 L39) and 0.29 (Mixtral L16) of the drop; the MoE output at the trigger site peaks at layer 0 (Qwen3 0.15, Mixtral 0.49 of the drop), like the early subject-site effect of Phase 2 / Direction 6b.

Tables: `results/tables/ext13base_f1_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`; figures `results/figures/ext13base_f1_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**f1: headline numbers with cluster-bootstrap 95% CIs (unit = pair; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 128 | 128 | 0.76 [0.73, 0.77] | 1.782 [1.699, 1.902] | 0.576 [0.523, 0.670] | 1.197 [1.137, 1.260] | -0.197 [-0.260, -0.137] | 0.243 [0.184, 0.301] | L37 0.137 | L39 0.292 |
| Qwen3-30B-A3B-Base | main, all | 256 | 256 | 0.76 [0.75, 0.78] | 1.742 [1.677, 1.831] | 0.541 [0.501, 0.603] | 1.202 [1.163, 1.244] | -0.202 [-0.244, -0.163] | 0.215 [0.174, 0.256] | L37 0.119 | L39 0.298 |
| Qwen3-30B-A3B-Base | replication, validation | 128 | 128 | 0.78 [0.75, 0.80] | 1.753 [1.652, 1.878] | 0.502 [0.446, 0.586] | 1.201 [1.139, 1.265] | -0.201 [-0.265, -0.139] | 0.171 [0.108, 0.231] | L37 0.128 | L39 0.280 |
| Qwen3-30B-A3B-Base | names (main, all) | 132 | 132 | 0.79 [0.77, 0.81] | 2.161 [2.082, 2.289] | 0.568 [0.507, 0.684] | 1.317 [1.281, 1.357] | -0.317 [-0.357, -0.281] | 0.345 [0.294, 0.396] | L37 0.181 | L39 0.400 |
| Qwen3-30B-A3B-Base | objects (main, all) | 124 | 124 | 0.73 [0.71, 0.75] | 1.428 [1.362, 1.515] | 0.532 [0.484, 0.605] | 1.108 [1.050, 1.172] | -0.108 [-0.172, -0.050] | 0.109 [0.056, 0.161] | L44 0.091 | L43 0.231 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 256 | 256 | 0.76 [0.75, 0.78] | 1.742 [1.677, 1.831] | 0.541 [0.501, 0.603] | 1.202 [1.163, 1.244] | -0.202 [-0.244, -0.163] | 0.215 [0.174, 0.256] | L37 0.119 | L39 0.298 |
| Mixtral-8x7B, BOS | main, validation | 101 | 101 | 0.56 [0.53, 0.59] | 1.346 [1.259, 1.447] | 1.056 [0.956, 1.171] | 1.010 [0.906, 1.115] | -0.010 [-0.115, 0.094] | 0.227 [0.134, 0.318] | L20 0.128 | L16 0.287 |
| Mixtral-8x7B, BOS | main, all | 202 | 202 | 0.55 [0.53, 0.57] | 1.320 [1.260, 1.388] | 1.087 [1.014, 1.171] | 0.988 [0.917, 1.057] | 0.012 [-0.057, 0.083] | 0.255 [0.190, 0.319] | L20 0.133 | L16 0.278 |
| Mixtral-8x7B, BOS | names (main, all) | 23 | 23 | 0.67 [0.63, 0.72] | 1.627 [1.438, 1.842] | 0.789 [0.580, 1.047] | 1.442 [1.258, 1.618] | -0.442 [-0.618, -0.258] | -0.108 [-0.396, 0.190] | L17 0.129 | L19 0.350 |
| Mixtral-8x7B, BOS | objects (main, all) | 179 | 179 | 0.54 [0.52, 0.55] | 1.289 [1.227, 1.358] | 1.120 [1.045, 1.208] | 0.941 [0.871, 1.013] | 0.059 [-0.013, 0.129] | 0.293 [0.233, 0.351] | L20 0.141 | L16 0.274 |
| Mixtral-8x7B, BOS | probe-neutral (main, all) | 202 | 202 | 0.55 [0.53, 0.57] | 1.320 [1.260, 1.388] | 1.087 [1.014, 1.171] | 0.988 [0.917, 1.057] | 0.012 [-0.057, 0.083] | 0.255 [0.190, 0.319] | L20 0.133 | L16 0.278 |


**f1 W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L37 | +0.633 [+0.542, +0.731] | 0.137 [0.117, 0.159] | L37E032 | 248 (≥ 128) | +0.495 [+0.415, +0.582] | +0.470 [+0.389, +0.556] | +0.646 [+0.554, +0.746] | A | +0.172 [+0.141, +0.205] |
| Qwen3-30B-A3B-Base | main | interior | L37 | +0.633 [+0.542, +0.731] | 0.137 [0.117, 0.159] | L37E032 | 248 (≥ 128) | +0.495 [+0.415, +0.582] | +0.470 [+0.389, +0.556] | +0.646 [+0.554, +0.746] | A | +0.172 [+0.141, +0.205] |
| Qwen3-30B-A3B-Base | rep | two_stage | L37 | +0.602 [+0.511, +0.700] | 0.128 [0.110, 0.148] | L37E032 | 248 (≥ 128) | +0.468 [+0.387, +0.555] | +0.452 [+0.373, +0.537] | +0.606 [+0.514, +0.703] | A |  |
| Qwen3-30B-A3B-Base | rep | interior | L37 | +0.602 [+0.511, +0.700] | 0.128 [0.110, 0.148] | L37E032 | 248 (≥ 128) | +0.468 [+0.387, +0.555] | +0.452 [+0.373, +0.537] | +0.606 [+0.514, +0.703] | A |  |
| Mixtral-8x7B, BOS | main | two_stage | L20 | +0.554 [+0.473, +0.638] | 0.128 [0.111, 0.144] | L20E001 | 140 (≥ 101) | +0.227 [+0.177, +0.278] | -0.011 [-0.078, +0.058] | +0.534 [+0.457, +0.616] | B | +0.003 [-0.027, +0.034] |
| Mixtral-8x7B, BOS | main | interior | L20 | +0.554 [+0.473, +0.638] | 0.128 [0.111, 0.144] | L20E001 | 140 (≥ 101) | +0.227 [+0.177, +0.278] | -0.011 [-0.078, +0.058] | +0.534 [+0.457, +0.616] | B | +0.003 [-0.027, +0.034] |


#### dir_f1: direction on identical items: fill-in-the-blank (task 2)

- Qwen3 (90 validation pairs, 90 pairs; drop 5.20): attention share 0.71 [0.68, 0.73], A_direct 1.08 [1.00, 1.17], M_direct -0.08 [-0.17, 0.00], all-MoE M 0.11 [0.05, 0.17]; peaks MoE L43 0.086, attention L43 0.213 of the drop; expert L44E116 (Spec +0.17 [+0.11, +0.24], pattern A); Phase-3 WinoGrande expert L41E117: rescue +0.00, Spec -0.03 (discovery-active 1); top heads L42H10 +0.53, L42H31 +0.48, L39H29 +0.45; grid: MoE at the STR site peaks L2 0.21, attention at the final position L43 0.21.
- Mixtral BOS (94 validation pairs, 94 pairs; drop 4.62): attention share 0.52 [0.50, 0.54], A_direct 0.86 [0.77, 0.95], M_direct 0.14 [0.05, 0.23], all-MoE M 0.41 [0.35, 0.46]; peaks MoE L20 0.151, attention L16 0.278 of the drop; expert L20E001 (Spec +0.11 [+0.02, +0.20], pattern A); Phase-3 WinoGrande expert L20E000: rescue +0.01, Spec -0.36 (discovery-active 21); top heads L21H11 +0.49, L19H12 +0.42, L15H8 +0.37; grid: MoE at the STR site peaks L0 0.53, attention at the final position L16 0.26.

Tables: `results/tables/ext13base_dir_f1_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`; figures `results/figures/ext13base_dir_f1_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**dir_f1: headline numbers with cluster-bootstrap 95% CIs (unit = pair; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 90 | 90 | 0.71 [0.68, 0.73] | 1.418 [1.307, 1.561] | 0.578 [0.520, 0.698] | 1.082 [0.998, 1.166] | -0.082 [-0.166, 0.002] | 0.110 [0.046, 0.173] | L43 0.086 | L43 0.213 |
| Qwen3-30B-A3B-Base | main, all | 180 | 180 | 0.70 [0.67, 0.71] | 1.440 [1.358, 1.537] | 0.627 [0.570, 0.721] | 1.051 [0.992, 1.107] | -0.051 [-0.107, 0.008] | 0.128 [0.086, 0.170] | L44 0.089 | L43 0.213 |
| Qwen3-30B-A3B-Base | objects (main, all) | 166 | 166 | 0.68 [0.66, 0.70] | 1.401 [1.318, 1.500] | 0.653 [0.591, 0.753] | 1.026 [0.965, 1.086] | -0.026 [-0.086, 0.035] | 0.120 [0.077, 0.162] | L43 0.094 | L43 0.206 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 89 | 89 | 0.74 [0.71, 0.75] | 1.561 [1.469, 1.678] | 0.557 [0.507, 0.667] | 1.204 [1.136, 1.270] | -0.204 [-0.270, -0.136] | 0.055 [0.002, 0.106] | L44 0.088 | L43 0.240 |
| Qwen3-30B-A3B-Base | not neutral (main, all) | 91 | 91 | 0.65 [0.62, 0.68] | 1.338 [1.219, 1.485] | 0.712 [0.612, 0.852] | 0.906 [0.827, 0.989] | 0.094 [0.011, 0.173] | 0.198 [0.138, 0.256] | L44 0.090 | L43 0.188 |
| Mixtral-8x7B, BOS | main, validation | 94 | 94 | 0.52 [0.50, 0.54] | 1.281 [1.198, 1.378] | 1.165 [1.091, 1.249] | 0.857 [0.775, 0.945] | 0.143 [0.055, 0.225] | 0.409 [0.355, 0.458] | L20 0.151 | L16 0.278 |
| Mixtral-8x7B, BOS | main, all | 188 | 188 | 0.51 [0.49, 0.53] | 1.221 [1.165, 1.287] | 1.174 [1.120, 1.237] | 0.834 [0.772, 0.895] | 0.166 [0.105, 0.228] | 0.402 [0.363, 0.438] | L20 0.156 | L16 0.268 |
| Mixtral-8x7B, BOS | objects (main, all) | 187 | 187 | 0.51 [0.49, 0.52] | 1.219 [1.161, 1.283] | 1.178 [1.126, 1.240] | 0.830 [0.767, 0.896] | 0.170 [0.104, 0.233] | 0.405 [0.368, 0.443] | L20 0.157 | L16 0.267 |
| Mixtral-8x7B, BOS | probe-neutral (main, all) | 86 | 86 | 0.51 [0.49, 0.53] | 1.283 [1.203, 1.369] | 1.223 [1.138, 1.321] | 0.919 [0.830, 1.006] | 0.081 [-0.006, 0.170] | 0.388 [0.335, 0.440] | L20 0.164 | L16 0.279 |
| Mixtral-8x7B, BOS | not neutral (main, all) | 102 | 102 | 0.51 [0.49, 0.53] | 1.178 [1.095, 1.269] | 1.142 [1.072, 1.223] | 0.768 [0.686, 0.853] | 0.232 [0.147, 0.314] | 0.412 [0.357, 0.465] | L20 0.150 | L16 0.260 |


**dir_f1 W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L44 | +0.444 [+0.356, +0.538] | 0.085 [0.070, 0.100] | L44E116 | 144 (≥ 90) | +0.212 [+0.154, +0.274] | +0.172 [+0.108, +0.238] | +0.447 [+0.364, +0.537] | A | +0.118 [+0.081, +0.155] |
| Qwen3-30B-A3B-Base | main | interior | L43 | +0.452 [+0.357, +0.554] | 0.086 [0.071, 0.103] | L43E114 | 128 (≥ 90) | +0.039 [+0.015, +0.063] | -0.024 [-0.055, +0.008] | +0.447 [+0.356, +0.550] | B |  |
| Mixtral-8x7B, BOS | main | two_stage | L20 | +0.704 [+0.616, +0.794] | 0.151 [0.134, 0.168] | L20E001 | 172 (≥ 94) | +0.392 [+0.319, +0.465] | +0.109 [+0.019, +0.202] | +0.705 [+0.618, +0.792] | A | -0.017 [-0.056, +0.017] |
| Mixtral-8x7B, BOS | main | interior | L20 | +0.704 [+0.616, +0.794] | 0.151 [0.134, 0.168] | L20E001 | 172 (≥ 94) | +0.392 [+0.319, +0.465] | +0.109 [+0.019, +0.202] | +0.705 [+0.618, +0.792] | A | -0.017 [-0.056, +0.017] |


#### dir_opt: direction on identical items: Phase-3 option swap (task 2)

- Qwen3 (90 validation pairs, 90 pairs; drop 8.96): attention share 0.19 [0.16, 0.24], A_direct 0.05 [0.02, 0.08], M_direct 0.95 [0.92, 0.98], all-MoE M 0.81 [0.79, 0.83]; peaks MoE L41 0.185, attention L42 0.038 of the drop; expert L41E117 (Spec +0.46 [+0.29, +0.66], pattern A); Phase-3 WinoGrande expert L41E117: rescue +0.58, Spec +0.46 (discovery-active 136); top heads L38H18 +0.46, L38H21 +0.31, L42H30 +0.21; grid: MoE at the STR site peaks L0 0.18, attention at the final position L42 0.04.
- Mixtral BOS (94 validation pairs, 94 pairs; drop 9.80): attention share 0.35 [0.33, 0.36], A_direct 0.31 [0.29, 0.33], M_direct 0.69 [0.67, 0.71], all-MoE M 0.75 [0.73, 0.78]; peaks MoE L20 0.146, attention L13 0.124 of the drop; expert L20E000 (Spec +0.90 [+0.79, +1.02], pattern A); Phase-3 WinoGrande expert L20E000: rescue +1.13, Spec +0.90 (discovery-active 187); top heads L25H9 +0.63, L19H13 +0.42, L13H4 +0.21; grid: MoE at the STR site peaks L0 0.95, attention at the final position L13 0.13.

Tables: `results/tables/ext13base_dir_opt_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`; figures `results/figures/ext13base_dir_opt_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**dir_opt: headline numbers with cluster-bootstrap 95% CIs (unit = pair; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 90 | 90 | 0.19 [0.16, 0.24] | 0.278 [0.209, 0.399] | 1.172 [1.068, 1.320] | 0.048 [0.016, 0.080] | 0.952 [0.920, 0.984] | 0.811 [0.794, 0.829] | L41 0.185 | L42 0.038 |
| Qwen3-30B-A3B-Base | main, all | 180 | 180 | 0.18 [0.16, 0.22] | 0.259 [0.218, 0.334] | 1.144 [1.063, 1.258] | 0.058 [0.033, 0.083] | 0.942 [0.917, 0.967] | 0.808 [0.794, 0.821] | L41 0.181 | L42 0.039 |
| Qwen3-30B-A3B-Base | objects (main, all) | 166 | 166 | 0.18 [0.16, 0.21] | 0.255 [0.209, 0.333] | 1.164 [1.075, 1.276] | 0.044 [0.019, 0.070] | 0.956 [0.930, 0.981] | 0.814 [0.801, 0.827] | L41 0.185 | L42 0.038 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 89 | 89 | 0.14 [0.12, 0.18] | 0.192 [0.155, 0.264] | 1.138 [1.059, 1.271] | 0.003 [-0.036, 0.043] | 0.997 [0.957, 1.036] | 0.799 [0.781, 0.816] | L41 0.180 | L42 0.042 |
| Qwen3-30B-A3B-Base | not neutral (main, all) | 91 | 91 | 0.23 [0.20, 0.27] | 0.356 [0.287, 0.475] | 1.172 [1.038, 1.348] | 0.109 [0.080, 0.139] | 0.891 [0.861, 0.920] | 0.815 [0.796, 0.834] | L41 0.182 | L38 0.049 |
| Mixtral-8x7B, BOS | main, validation | 94 | 94 | 0.35 [0.33, 0.36] | 0.678 [0.640, 0.721] | 1.273 [1.215, 1.334] | 0.307 [0.286, 0.329] | 0.693 [0.671, 0.714] | 0.754 [0.731, 0.777] | L20 0.146 | L13 0.124 |
| Mixtral-8x7B, BOS | main, all | 188 | 188 | 0.34 [0.33, 0.35] | 0.681 [0.653, 0.710] | 1.307 [1.266, 1.349] | 0.307 [0.292, 0.323] | 0.693 [0.677, 0.708] | 0.757 [0.741, 0.772] | L20 0.153 | L13 0.133 |
| Mixtral-8x7B, BOS | objects (main, all) | 187 | 187 | 0.34 [0.33, 0.35] | 0.680 [0.653, 0.709] | 1.307 [1.265, 1.351] | 0.307 [0.291, 0.323] | 0.693 [0.677, 0.709] | 0.758 [0.741, 0.773] | L20 0.153 | L13 0.133 |
| Mixtral-8x7B, BOS | probe-neutral (main, all) | 86 | 86 | 0.33 [0.32, 0.34] | 0.679 [0.642, 0.719] | 1.383 [1.330, 1.433] | 0.277 [0.257, 0.298] | 0.723 [0.702, 0.743] | 0.781 [0.760, 0.802] | L20 0.159 | L13 0.139 |
| Mixtral-8x7B, BOS | not neutral (main, all) | 102 | 102 | 0.35 [0.34, 0.37] | 0.683 [0.645, 0.724] | 1.246 [1.187, 1.306] | 0.332 [0.310, 0.354] | 0.668 [0.646, 0.690] | 0.737 [0.715, 0.759] | L20 0.149 | L13 0.127 |


**dir_opt W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L41 | +1.611 [+1.390, +1.847] | 0.185 [0.165, 0.206] | L41E117 | 136 (≥ 90) | +0.584 [+0.414, +0.779] | +0.465 [+0.290, +0.662] | +1.623 [+1.403, +1.858] | A | +0.251 [+0.167, +0.340] |
| Qwen3-30B-A3B-Base | main | interior | L41 | +1.611 [+1.390, +1.847] | 0.185 [0.165, 0.206] | L41E117 | 136 (≥ 90) | +0.584 [+0.414, +0.779] | +0.465 [+0.290, +0.662] | +1.623 [+1.403, +1.858] | A | +0.251 [+0.167, +0.340] |
| Mixtral-8x7B, BOS | main | two_stage | L20 | +1.409 [+1.305, +1.515] | 0.146 [0.135, 0.157] | L20E000 | 187 (≥ 94) | +1.130 [+1.030, +1.231] | +0.903 [+0.793, +1.018] | +1.394 [+1.291, +1.500] | A | +0.186 [+0.149, +0.226] |
| Mixtral-8x7B, BOS | main | interior | L20 | +1.409 [+1.305, +1.515] | 0.146 [0.135, 0.157] | L20E000 | 187 (≥ 94) | +1.130 [+1.030, +1.231] | +0.903 [+0.793, +1.018] | +1.394 [+1.291, +1.500] | A | +0.186 [+0.149, +0.226] |


#### cp_nat: copy prefix on the natural Phase-3 twins (task 6)

- Qwen3 (140 validation pairs, 70 Phase-3 twins; drop 6.28): attention share 0.21 [0.20, 0.24], A_direct -0.01 [-0.07, 0.04], M_direct 1.01 [0.96, 1.07], all-MoE M 0.79 [0.76, 0.81]; peaks MoE L43 0.163, attention L41 0.059 of the drop; expert L41E117 (Spec +0.42 [+0.35, +0.49], pattern A; replication selects L43E081); Phase-3 WinoGrande expert L41E117: rescue +0.48, Spec +0.42 (discovery-active 255); top heads L41H27 +0.58, L42H30 +0.14, L41H0 +0.13; share names 0.35 (22 pairs) vs objects 0.19 (264); grid: MoE at the STR site peaks L0 0.18, attention at the final position L41 0.06; copy attention of the top head L41H27: in-context answer 0.00 vs other attribute 0.00 (clean), 0.00 vs 0.00 (corrupted).
- Mixtral BOS (204 validation pairs, 102 Phase-3 twins; drop 7.13): attention share 0.37 [0.36, 0.39], A_direct 0.38 [0.34, 0.41], M_direct 0.62 [0.59, 0.66], all-MoE M 0.73 [0.70, 0.76]; peaks MoE L20 0.167, attention L13 0.153 of the drop; expert L21E006 (Spec +0.07 [-0.02, +0.15], pattern B; replication selects L20E000); Phase-3 WinoGrande expert L20E000: rescue +1.00, Spec +0.84 (discovery-active 422); top heads L25H9 +0.53, L19H13 +0.36, L13H18 +0.24; share names 0.51 (36 pairs) vs objects 0.37 (384); grid: MoE at the STR site peaks L0 0.87, attention at the final position L13 0.16; copy attention of the top head L25H9: in-context answer 0.01 vs other attribute 0.01 (clean), 0.01 vs 0.01 (corrupted).

Tables: `results/tables/ext13base_cp_nat_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`, `ext13base_cp_nat_w5_copy.md`; figures `results/figures/ext13base_cp_nat_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**cp_nat: headline numbers with cluster-bootstrap 95% CIs (unit = Phase-3 twin; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 140 | 70 | 0.21 [0.20, 0.24] | 0.326 [0.292, 0.403] | 1.207 [1.099, 1.352] | -0.013 [-0.068, 0.040] | 1.013 [0.960, 1.068] | 0.787 [0.762, 0.811] | L43 0.163 | L41 0.059 |
| Qwen3-30B-A3B-Base | main, all | 286 | 143 | 0.20 [0.19, 0.23] | 0.290 [0.262, 0.354] | 1.156 [1.072, 1.260] | -0.002 [-0.045, 0.038] | 1.002 [0.962, 1.045] | 0.773 [0.753, 0.791] | L43 0.156 | L41 0.057 |
| Qwen3-30B-A3B-Base | replication, validation | 158 | 79 | 0.23 [0.21, 0.27] | 0.336 [0.285, 0.430] | 1.120 [0.982, 1.285] | -0.018 [-0.076, 0.039] | 1.018 [0.961, 1.076] | 0.757 [0.729, 0.785] | L41 0.142 | L43 0.055 |
| Qwen3-30B-A3B-Base | names (main, all) | 22 | 11 | 0.35 [0.30, 0.41] | 0.604 [0.503, 0.820] | 1.130 [0.911, 1.515] | 0.310 [0.194, 0.426] | 0.690 [0.574, 0.806] | 0.628 [0.529, 0.718] | L42 0.155 | L38 0.077 |
| Qwen3-30B-A3B-Base | objects (main, all) | 264 | 132 | 0.19 [0.18, 0.22] | 0.278 [0.249, 0.339] | 1.164 [1.076, 1.271] | -0.032 [-0.073, 0.007] | 1.032 [0.993, 1.073] | 0.787 [0.770, 0.803] | L41 0.160 | L41 0.056 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 124 | 62 | 0.19 [0.17, 0.24] | 0.285 [0.243, 0.382] | 1.183 [1.057, 1.357] | -0.061 [-0.140, 0.014] | 1.061 [0.986, 1.140] | 0.757 [0.725, 0.783] | L43 0.165 | L43 0.056 |
| Qwen3-30B-A3B-Base | not neutral (main, all) | 162 | 81 | 0.22 [0.21, 0.25] | 0.331 [0.292, 0.405] | 1.144 [1.047, 1.266] | 0.036 [-0.011, 0.081] | 0.964 [0.919, 1.011] | 0.783 [0.757, 0.807] | L41 0.157 | L41 0.061 |
| Mixtral-8x7B, BOS | main, validation | 204 | 102 | 0.37 [0.36, 0.39] | 0.811 [0.772, 0.855] | 1.355 [1.308, 1.404] | 0.378 [0.344, 0.414] | 0.622 [0.586, 0.656] | 0.731 [0.700, 0.759] | L20 0.167 | L13 0.153 |
| Mixtral-8x7B, BOS | main, all | 420 | 210 | 0.38 [0.37, 0.39] | 0.810 [0.782, 0.841] | 1.313 [1.271, 1.354] | 0.396 [0.369, 0.424] | 0.604 [0.576, 0.631] | 0.711 [0.684, 0.733] | L20 0.158 | L13 0.158 |
| Mixtral-8x7B, BOS | replication, validation | 222 | 111 | 0.38 [0.37, 0.40] | 0.817 [0.782, 0.856] | 1.325 [1.277, 1.379] | 0.406 [0.376, 0.436] | 0.594 [0.564, 0.624] | 0.707 [0.682, 0.731] | L20 0.157 | L13 0.145 |
| Mixtral-8x7B, BOS | names (main, all) | 36 | 18 | 0.51 [0.45, 0.57] | 1.087 [0.990, 1.203] | 1.056 [0.850, 1.287] | 0.703 [0.584, 0.818] | 0.297 [0.182, 0.416] | 0.452 [0.301, 0.589] | L19 0.134 | L19 0.148 |
| Mixtral-8x7B, BOS | objects (main, all) | 384 | 192 | 0.37 [0.36, 0.38] | 0.784 [0.758, 0.812] | 1.337 [1.300, 1.376] | 0.366 [0.343, 0.391] | 0.634 [0.609, 0.657] | 0.735 [0.717, 0.753] | L21 0.164 | L13 0.159 |
| Mixtral-8x7B, BOS | probe-neutral (main, all) | 172 | 86 | 0.38 [0.36, 0.40] | 0.808 [0.762, 0.859] | 1.342 [1.275, 1.406] | 0.405 [0.359, 0.455] | 0.595 [0.545, 0.641] | 0.718 [0.672, 0.756] | L20 0.159 | L13 0.163 |
| Mixtral-8x7B, BOS | not neutral (main, all) | 248 | 124 | 0.39 [0.37, 0.40] | 0.811 [0.776, 0.850] | 1.293 [1.241, 1.348] | 0.389 [0.357, 0.423] | 0.611 [0.577, 0.643] | 0.706 [0.676, 0.734] | L20 0.158 | L13 0.154 |


**cp_nat W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L41 | +0.957 [+0.871, +1.043] | 0.160 [0.147, 0.173] | L41E117 | 255 (≥ 146) | +0.484 [+0.418, +0.553] | +0.418 [+0.351, +0.486] | +0.953 [+0.865, +1.042] | A | +0.189 [+0.159, +0.219] |
| Qwen3-30B-A3B-Base | main | interior | L41 | +0.957 [+0.871, +1.043] | 0.160 [0.147, 0.173] | L41E117 | 255 (≥ 146) | +0.484 [+0.418, +0.553] | +0.418 [+0.351, +0.486] | +0.953 [+0.865, +1.042] | A | +0.189 [+0.159, +0.219] |
| Qwen3-30B-A3B-Base | rep | two_stage | L43 | +0.828 [+0.737, +0.920] | 0.140 [0.126, 0.154] | L43E081 | 324 (≥ 162) | +0.477 [+0.409, +0.548] | +0.420 [+0.346, +0.497] | +0.828 [+0.740, +0.917] | A |  |
| Qwen3-30B-A3B-Base | rep | interior | L43 | +0.828 [+0.737, +0.920] | 0.140 [0.126, 0.154] | L43E081 | 324 (≥ 162) | +0.477 [+0.409, +0.548] | +0.420 [+0.346, +0.497] | +0.828 [+0.740, +0.917] | A |  |
| Mixtral-8x7B, BOS | main | two_stage | L21 | +1.148 [+1.075, +1.223] | 0.160 [0.151, 0.170] | L21E006 | 396 (≥ 216) | +0.566 [+0.515, +0.617] | +0.067 [-0.016, +0.150] | +1.143 [+1.070, +1.218] | B | +0.077 [+0.054, +0.100] |
| Mixtral-8x7B, BOS | main | interior | L21 | +1.148 [+1.075, +1.223] | 0.160 [0.151, 0.170] | L21E006 | 396 (≥ 216) | +0.566 [+0.515, +0.617] | +0.067 [-0.016, +0.150] | +1.143 [+1.070, +1.218] | B | +0.077 [+0.054, +0.100] |
| Mixtral-8x7B, BOS | rep | two_stage | L20 | +1.207 [+1.140, +1.271] | 0.157 [0.149, 0.165] | L20E000 | 422 (≥ 214) | +0.997 [+0.932, +1.061] | +0.835 [+0.769, +0.901] | +1.194 [+1.129, +1.256] | A |  |
| Mixtral-8x7B, BOS | rep | interior | L20 | +1.207 [+1.140, +1.271] | 0.157 [0.149, 0.165] | L20E000 | 422 (≥ 214) | +0.997 [+0.932, +1.061] | +0.835 [+0.769, +0.901] | +1.194 [+1.129, +1.256] | A |  |


#### cp_vt_orig: copy prefix on the verb templates, original verb (task 6)

- Qwen3 (128 validation pairs, 13 seeds; drop 5.90): attention share 0.31 [0.26, 0.38], A_direct 0.11 [-0.01, 0.28], M_direct 0.89 [0.72, 1.01], all-MoE M 0.69 [0.57, 0.79]; peaks MoE L42 0.130, attention L41 0.081 of the drop; expert L39E001 (Spec +0.12 [+0.07, +0.17], pattern A); Phase-3 WinoGrande expert L41E117: rescue +0.01, Spec -0.08 (discovery-active 138); top heads L41H0 +0.43, L42H30 +0.41, L41H5 +0.27; grid: MoE at the STR site peaks L37 0.06, attention at the final position L41 0.09; copy attention of the top head L41H0: in-context answer 0.00 vs other attribute 0.00 (clean), 0.00 vs 0.00 (corrupted).
- Mixtral BOS (128 validation pairs, 10 seeds; drop 5.46): attention share 0.39 [0.33, 0.47], A_direct 0.47 [0.39, 0.60], M_direct 0.53 [0.40, 0.61], all-MoE M 0.54 [0.38, 0.68]; peaks MoE L19 0.194, attention L19 0.146 of the drop; expert L19E006 (Spec +0.51 [+0.39, +0.62], pattern A); Phase-3 WinoGrande expert L20E000: rescue +0.47, Spec +0.21 (discovery-active 179); top heads L13H18 +0.41, L19H13 +0.23, L19H1 +0.19; grid: MoE at the STR site peaks L0 0.98, attention at the final position L19 0.17; copy attention of the top head L13H18: in-context answer 0.01 vs other attribute 0.00 (clean), 0.00 vs 0.01 (corrupted).

Tables: `results/tables/ext13base_cp_vt_orig_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`, `ext13base_cp_vt_orig_w5_copy.md`; figures `results/figures/ext13base_cp_vt_orig_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**cp_vt_orig: headline numbers with cluster-bootstrap 95% CIs (unit = seed; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 128 | 13 | 0.31 [0.26, 0.38] | 0.562 [0.422, 0.824] | 1.277 [1.074, 1.611] | 0.115 [-0.011, 0.282] | 0.885 [0.718, 1.011] | 0.693 [0.572, 0.789] | L42 0.130 | L41 0.081 |
| Qwen3-30B-A3B-Base | main, all | 256 | 27 | 0.29 [0.25, 0.35] | 0.539 [0.437, 0.698] | 1.311 [1.155, 1.555] | 0.103 [-0.005, 0.229] | 0.897 [0.771, 1.005] | 0.690 [0.586, 0.775] | L42 0.144 | L41 0.086 |
| Qwen3-30B-A3B-Base | names (main, all) | 256 | 27 | 0.29 [0.25, 0.35] | 0.539 [0.437, 0.698] | 1.311 [1.155, 1.555] | 0.103 [-0.005, 0.229] | 0.897 [0.771, 1.005] | 0.690 [0.586, 0.775] | L42 0.144 | L41 0.086 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 191 | 25 | 0.29 [0.24, 0.35] | 0.541 [0.443, 0.720] | 1.340 [1.160, 1.618] | 0.092 [-0.008, 0.215] | 0.908 [0.785, 1.008] | 0.706 [0.618, 0.780] | L42 0.144 | L41 0.090 |
| Qwen3-30B-A3B-Base | not neutral (main, all) | 65 | 20 | 0.31 [0.24, 0.37] | 0.564 [0.415, 0.753] | 1.272 [1.133, 1.523] | 0.132 [-0.025, 0.323] | 0.868 [0.677, 1.025] | 0.646 [0.457, 0.785] | L42 0.145 | L41 0.074 |
| Mixtral-8x7B, BOS | main, validation | 128 | 10 | 0.39 [0.33, 0.47] | 0.918 [0.802, 1.092] | 1.414 [1.157, 1.690] | 0.471 [0.390, 0.598] | 0.529 [0.402, 0.610] | 0.544 [0.379, 0.677] | L19 0.194 | L19 0.146 |
| Mixtral-8x7B, BOS | main, all | 256 | 19 | 0.42 [0.38, 0.47] | 0.954 [0.855, 1.071] | 1.308 [1.144, 1.487] | 0.532 [0.432, 0.650] | 0.468 [0.350, 0.568] | 0.522 [0.409, 0.624] | L19 0.173 | L19 0.172 |
| Mixtral-8x7B, BOS | names (main, all) | 256 | 19 | 0.42 [0.38, 0.47] | 0.954 [0.855, 1.071] | 1.308 [1.144, 1.487] | 0.532 [0.432, 0.650] | 0.468 [0.350, 0.568] | 0.522 [0.409, 0.624] | L19 0.173 | L19 0.172 |
| Mixtral-8x7B, BOS | probe-neutral (main, all) | 218 | 19 | 0.42 [0.37, 0.47] | 0.936 [0.836, 1.061] | 1.308 [1.141, 1.495] | 0.517 [0.428, 0.629] | 0.483 [0.371, 0.572] | 0.532 [0.418, 0.631] | L19 0.175 | L19 0.172 |
| Mixtral-8x7B, BOS | not neutral (main, all) | 38 | 12 | 0.45 [0.40, 0.51] | 1.071 [0.977, 1.160] | 1.310 [1.075, 1.603] | 0.628 [0.444, 0.791] | 0.372 [0.209, 0.556] | 0.457 [0.318, 0.603] | L19 0.164 | L19 0.173 |


**cp_vt_orig W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L39 | +0.543 [+0.439, +0.639] | 0.097 [0.079, 0.113] | L39E001 | 256 (≥ 128) | +0.170 [+0.128, +0.215] | +0.121 [+0.074, +0.168] | +0.560 [+0.471, +0.643] | A | +0.103 [+0.071, +0.135] |
| Qwen3-30B-A3B-Base | main | interior | L39 | +0.543 [+0.439, +0.639] | 0.097 [0.079, 0.113] | L39E001 | 256 (≥ 128) | +0.170 [+0.128, +0.215] | +0.121 [+0.074, +0.168] | +0.560 [+0.471, +0.643] | A | +0.103 [+0.071, +0.135] |
| Mixtral-8x7B, BOS | main | two_stage | L19 | +1.023 [+0.912, +1.128] | 0.194 [0.181, 0.206] | L19E006 | 218 (≥ 128) | +0.740 [+0.631, +0.846] | +0.505 [+0.392, +0.617] | +1.014 [+0.906, +1.117] | A | +0.090 [+0.058, +0.122] |
| Mixtral-8x7B, BOS | main | interior | L19 | +1.023 [+0.912, +1.128] | 0.194 [0.181, 0.206] | L19E006 | 218 (≥ 128) | +0.740 [+0.631, +0.846] | +0.505 [+0.392, +0.617] | +1.014 [+0.906, +1.117] | A | +0.090 [+0.058, +0.122] |


#### cp_vt_neu: copy prefix on the verb templates, neutral verb (task 6)

- Qwen3 (128 validation pairs, 8 seeds; drop 5.51): attention share 0.26 [0.23, 0.38], A_direct 0.09 [-0.10, 0.27], M_direct 0.91 [0.73, 1.10], all-MoE M 0.74 [0.62, 0.83]; peaks MoE L43 0.180, attention L42 0.098 of the drop; expert L42E123 (Spec -0.00 [-0.04, +0.03], pattern B); Phase-3 WinoGrande expert L41E117: rescue +0.01, Spec -0.09 (discovery-active 120); top heads L41H0 +0.57, L42H30 +0.35, L42H27 +0.23; grid: MoE at the STR site peaks L37 0.05, attention at the final position L42 0.08; copy attention of the top head L41H0: in-context answer 0.00 vs other attribute 0.00 (clean), 0.00 vs 0.00 (corrupted).
- Mixtral BOS (128 validation pairs, 8 seeds; drop 5.67): attention share 0.38 [0.33, 0.42], A_direct 0.43 [0.37, 0.52], M_direct 0.57 [0.48, 0.63], all-MoE M 0.46 [0.37, 0.63]; peaks MoE L21 0.171, attention L13 0.132 of the drop; expert L19E006 (Spec +0.05 [-0.03, +0.13], pattern B); Phase-3 WinoGrande expert L20E000: rescue +0.20, Spec -0.25 (discovery-active 173); top heads L13H18 +0.37, L19H13 +0.35, L19H12 +0.15; grid: MoE at the STR site peaks L0 0.99, attention at the final position L19 0.17; copy attention of the top head L13H18: in-context answer 0.01 vs other attribute 0.00 (clean), 0.00 vs 0.01 (corrupted).

Tables: `results/tables/ext13base_cp_vt_neu_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`, `ext13base_cp_vt_neu_w5_copy.md`; figures `results/figures/ext13base_cp_vt_neu_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**cp_vt_neu: headline numbers with cluster-bootstrap 95% CIs (unit = seed; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 128 | 8 | 0.26 [0.23, 0.38] | 0.430 [0.364, 0.695] | 1.221 [1.000, 1.669] | 0.091 [-0.100, 0.271] | 0.909 [0.729, 1.100] | 0.738 [0.625, 0.828] | L43 0.180 | L42 0.098 |
| Qwen3-30B-A3B-Base | main, all | 256 | 20 | 0.28 [0.24, 0.36] | 0.470 [0.389, 0.651] | 1.193 [1.052, 1.449] | 0.114 [-0.026, 0.278] | 0.886 [0.722, 1.026] | 0.676 [0.555, 0.767] | L43 0.154 | L42 0.080 |
| Qwen3-30B-A3B-Base | names (main, all) | 256 | 20 | 0.28 [0.24, 0.36] | 0.470 [0.389, 0.651] | 1.193 [1.052, 1.449] | 0.114 [-0.026, 0.278] | 0.886 [0.722, 1.026] | 0.676 [0.555, 0.767] | L43 0.154 | L42 0.080 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 186 | 20 | 0.27 [0.23, 0.35] | 0.443 [0.377, 0.614] | 1.203 [1.007, 1.483] | 0.098 [-0.035, 0.249] | 0.902 [0.751, 1.035] | 0.684 [0.593, 0.758] | L43 0.157 | L42 0.078 |
| Qwen3-30B-A3B-Base | not neutral (main, all) | 70 | 13 | 0.32 [0.24, 0.40] | 0.574 [0.411, 0.919] | 1.213 [1.125, 1.457] | 0.152 [-0.033, 0.389] | 0.848 [0.611, 1.033] | 0.657 [0.457, 0.802] | L41 0.152 | L42 0.084 |
| Mixtral-8x7B, BOS | main, validation | 128 | 8 | 0.38 [0.33, 0.42] | 0.823 [0.749, 1.021] | 1.358 [1.332, 1.579] | 0.434 [0.370, 0.519] | 0.566 [0.481, 0.630] | 0.463 [0.372, 0.635] | L21 0.171 | L13 0.132 |
| Mixtral-8x7B, BOS | main, all | 256 | 16 | 0.41 [0.37, 0.44] | 0.913 [0.793, 1.108] | 1.332 [1.258, 1.437] | 0.461 [0.379, 0.571] | 0.539 [0.429, 0.621] | 0.516 [0.388, 0.630] | L19 0.155 | L19 0.165 |
| Mixtral-8x7B, BOS | names (main, all) | 256 | 16 | 0.41 [0.37, 0.44] | 0.913 [0.793, 1.108] | 1.332 [1.258, 1.437] | 0.461 [0.379, 0.571] | 0.539 [0.429, 0.621] | 0.516 [0.388, 0.630] | L19 0.155 | L19 0.165 |
| Mixtral-8x7B, BOS | probe-neutral (main, all) | 234 | 15 | 0.41 [0.37, 0.44] | 0.913 [0.787, 1.107] | 1.340 [1.265, 1.447] | 0.455 [0.379, 0.563] | 0.545 [0.437, 0.621] | 0.524 [0.401, 0.637] | L19 0.156 | L19 0.168 |
| Mixtral-8x7B, BOS | not neutral (main, all) | 22 | 7 | 0.42 [0.37, 0.49] | 0.920 [0.789, 1.146] | 1.264 [1.123, 1.477] | 0.516 [0.375, 0.696] | 0.484 [0.304, 0.625] | 0.431 [0.276, 0.594] | L21 0.154 | L19 0.133 |


**cp_vt_neu W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L42 | +0.618 [+0.529, +0.710] | 0.110 [0.097, 0.124] | L42E123 | 256 (≥ 128) | +0.080 [+0.048, +0.111] | -0.002 [-0.036, +0.034] | +0.607 [+0.516, +0.703] | B | -0.057 [-0.090, -0.024] |
| Qwen3-30B-A3B-Base | main | interior | L42 | +0.618 [+0.529, +0.710] | 0.110 [0.097, 0.124] | L42E123 | 256 (≥ 128) | +0.080 [+0.048, +0.111] | -0.002 [-0.036, +0.034] | +0.607 [+0.516, +0.703] | B | -0.057 [-0.090, -0.024] |
| Mixtral-8x7B, BOS | main | two_stage | L19 | +0.810 [+0.754, +0.868] | 0.150 [0.144, 0.156] | L19E006 | 221 (≥ 128) | +0.390 [+0.331, +0.451] | +0.046 [-0.027, +0.125] | +0.812 [+0.757, +0.871] | B | -0.040 [-0.075, -0.006] |
| Mixtral-8x7B, BOS | main | interior | L19 | +0.810 [+0.754, +0.868] | 0.150 [0.144, 0.156] | L19E006 | 221 (≥ 128) | +0.390 [+0.331, +0.451] | +0.046 [-0.027, +0.125] | +0.812 [+0.757, +0.871] | B | -0.040 [-0.075, -0.006] |


#### vt_e2a_orig: verb templates, entity → attribute, original verb (task 4)

- Qwen3 (128 validation pairs, 17 seeds; drop 6.73): attention share 0.28 [0.20, 0.36], A_direct 0.13 [0.08, 0.17], M_direct 0.87 [0.83, 0.92], all-MoE M 0.82 [0.79, 0.86]; peaks MoE L43 0.168, attention L42 0.049 of the drop; expert L39E121 (Spec +0.24 [+0.14, +0.36], pattern A); Phase-3 WinoGrande expert L41E117: rescue +0.08, Spec -0.05 (discovery-active 81); top heads L42H30 +0.32, L38H3 +0.30, L41H27 +0.29; grid: MoE at the STR site peaks L13 0.04, attention at the final position L42 0.05.
- Mixtral BOS (128 validation pairs, 18 seeds; drop 6.36): attention share 0.41 [0.36, 0.46], A_direct 0.38 [0.31, 0.47], M_direct 0.62 [0.53, 0.69], all-MoE M 0.68 [0.58, 0.76]; peaks MoE L19 0.184, attention L19 0.152 of the drop; expert L19E006 (Spec +0.62 [+0.50, +0.74], pattern A); Phase-3 WinoGrande expert L20E000: rescue +0.53, Spec +0.19 (discovery-active 201); top heads L13H18 +0.44, L19H13 +0.37, L19H12 +0.30; grid: MoE at the STR site peaks L0 0.98, attention at the final position L19 0.15.

Tables: `results/tables/ext13base_vt_e2a_orig_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`; figures `results/figures/ext13base_vt_e2a_orig_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**vt_e2a_orig: headline numbers with cluster-bootstrap 95% CIs (unit = seed; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 128 | 17 | 0.28 [0.20, 0.36] | 0.519 [0.365, 0.773] | 1.335 [1.119, 1.650] | 0.128 [0.080, 0.166] | 0.872 [0.834, 0.920] | 0.823 [0.788, 0.862] | L43 0.168 | L42 0.049 |
| Qwen3-30B-A3B-Base | main, all | 256 | 33 | 0.27 [0.22, 0.33] | 0.481 [0.391, 0.613] | 1.284 [1.074, 1.527] | 0.110 [0.078, 0.141] | 0.890 [0.859, 0.922] | 0.818 [0.785, 0.850] | L42 0.143 | L42 0.056 |
| Qwen3-30B-A3B-Base | names (main, all) | 256 | 33 | 0.27 [0.22, 0.33] | 0.481 [0.391, 0.613] | 1.284 [1.074, 1.527] | 0.110 [0.078, 0.141] | 0.890 [0.859, 0.922] | 0.818 [0.785, 0.850] | L42 0.143 | L42 0.056 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 211 | 32 | 0.27 [0.22, 0.33] | 0.476 [0.385, 0.613] | 1.312 [1.093, 1.548] | 0.112 [0.080, 0.142] | 0.888 [0.858, 0.920] | 0.820 [0.790, 0.850] | L42 0.141 | L42 0.052 |
| Qwen3-30B-A3B-Base | not neutral (main, all) | 45 | 23 | 0.30 [0.22, 0.38] | 0.514 [0.374, 0.703] | 1.182 [0.999, 1.524] | 0.099 [0.051, 0.138] | 0.901 [0.862, 0.949] | 0.810 [0.752, 0.871] | L43 0.154 | L42 0.072 |
| Mixtral-8x7B, BOS | main, validation | 128 | 18 | 0.41 [0.36, 0.46] | 0.889 [0.776, 1.022] | 1.293 [1.137, 1.478] | 0.383 [0.314, 0.471] | 0.617 [0.529, 0.686] | 0.676 [0.576, 0.757] | L19 0.184 | L19 0.152 |
| Mixtral-8x7B, BOS | main, all | 256 | 32 | 0.39 [0.36, 0.43] | 0.877 [0.806, 0.953] | 1.373 [1.204, 1.547] | 0.353 [0.305, 0.411] | 0.647 [0.589, 0.695] | 0.714 [0.645, 0.774] | L19 0.189 | L19 0.150 |
| Mixtral-8x7B, BOS | names (main, all) | 256 | 32 | 0.39 [0.36, 0.43] | 0.877 [0.806, 0.953] | 1.373 [1.204, 1.547] | 0.353 [0.305, 0.411] | 0.647 [0.589, 0.695] | 0.714 [0.645, 0.774] | L19 0.189 | L19 0.150 |
| Mixtral-8x7B, BOS | probe-neutral (main, all) | 231 | 31 | 0.39 [0.36, 0.42] | 0.882 [0.815, 0.959] | 1.375 [1.214, 1.543] | 0.350 [0.306, 0.399] | 0.650 [0.601, 0.694] | 0.720 [0.658, 0.774] | L19 0.189 | L19 0.151 |
| Mixtral-8x7B, BOS | not neutral (main, all) | 25 | 15 | 0.38 [0.31, 0.46] | 0.828 [0.681, 0.974] | 1.364 [1.084, 1.662] | 0.387 [0.271, 0.543] | 0.613 [0.457, 0.729] | 0.657 [0.479, 0.798] | L19 0.182 | L19 0.140 |


**vt_e2a_orig W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L39 | +0.805 [+0.650, +0.972] | 0.124 [0.101, 0.148] | L39E121 | 206 (≥ 128) | +0.324 [+0.221, +0.441] | +0.241 [+0.138, +0.359] | +0.831 [+0.681, +0.992] | A | +0.054 [+0.018, +0.093] |
| Qwen3-30B-A3B-Base | main | interior | L39 | +0.805 [+0.650, +0.972] | 0.124 [0.101, 0.148] | L39E121 | 206 (≥ 128) | +0.324 [+0.221, +0.441] | +0.241 [+0.138, +0.359] | +0.831 [+0.681, +0.992] | A | +0.054 [+0.018, +0.093] |
| Mixtral-8x7B, BOS | main | two_stage | L19 | +1.199 [+1.066, +1.332] | 0.184 [0.170, 0.197] | L19E006 | 248 (≥ 128) | +0.880 [+0.759, +1.002] | +0.620 [+0.497, +0.744] | +1.170 [+1.043, +1.298] | A | +0.139 [+0.096, +0.183] |
| Mixtral-8x7B, BOS | main | interior | L19 | +1.199 [+1.066, +1.332] | 0.184 [0.170, 0.197] | L19E006 | 248 (≥ 128) | +0.880 [+0.759, +1.002] | +0.620 [+0.497, +0.744] | +1.170 [+1.043, +1.298] | A | +0.139 [+0.096, +0.183] |


#### vt_e2a_neu: verb templates, entity → attribute, neutral verb (task 4)

- Qwen3 (128 validation pairs, 12 seeds; drop 6.32): attention share 0.22 [0.17, 0.30], A_direct 0.13 [0.05, 0.19], M_direct 0.87 [0.81, 0.95], all-MoE M 0.82 [0.77, 0.87]; peaks MoE L43 0.170, attention L42 0.061 of the drop; expert L42E123 (Spec +0.23 [+0.16, +0.30], pattern A); Phase-3 WinoGrande expert L41E117: rescue +0.10, Spec +0.00 (discovery-active 122); top heads L38H3 +0.35, L42H30 +0.29, L41H0 +0.22; grid: MoE at the STR site peaks L0 0.05, attention at the final position L42 0.06.
- Mixtral BOS (128 validation pairs, 12 seeds; drop 5.70): attention share 0.42 [0.37, 0.48], A_direct 0.42 [0.32, 0.55], M_direct 0.58 [0.45, 0.68], all-MoE M 0.62 [0.48, 0.75]; peaks MoE L19 0.174, attention L19 0.144 of the drop; expert L19E006 (Spec +0.29 [+0.19, +0.39], pattern A); Phase-3 WinoGrande expert L20E000: rescue +0.50, Spec +0.10 (discovery-active 196); top heads L13H18 +0.40, L19H13 +0.35, L19H12 +0.21; grid: MoE at the STR site peaks L0 0.99, attention at the final position L19 0.15.

Tables: `results/tables/ext13base_vt_e2a_neu_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`; figures `results/figures/ext13base_vt_e2a_neu_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**vt_e2a_neu: headline numbers with cluster-bootstrap 95% CIs (unit = seed; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 128 | 12 | 0.22 [0.17, 0.30] | 0.383 [0.330, 0.531] | 1.381 [1.152, 1.782] | 0.130 [0.053, 0.193] | 0.870 [0.807, 0.947] | 0.817 [0.767, 0.866] | L43 0.170 | L42 0.061 |
| Qwen3-30B-A3B-Base | main, all | 256 | 28 | 0.25 [0.20, 0.31] | 0.452 [0.381, 0.554] | 1.366 [1.179, 1.610] | 0.115 [0.070, 0.156] | 0.885 [0.844, 0.930] | 0.813 [0.775, 0.850] | L42 0.147 | L42 0.059 |
| Qwen3-30B-A3B-Base | names (main, all) | 256 | 28 | 0.25 [0.20, 0.31] | 0.452 [0.381, 0.554] | 1.366 [1.179, 1.610] | 0.115 [0.070, 0.156] | 0.885 [0.844, 0.930] | 0.813 [0.775, 0.850] | L42 0.147 | L42 0.059 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 197 | 27 | 0.24 [0.20, 0.30] | 0.446 [0.385, 0.547] | 1.394 [1.220, 1.639] | 0.111 [0.066, 0.150] | 0.889 [0.850, 0.934] | 0.817 [0.783, 0.850] | L39 0.143 | L42 0.056 |
| Qwen3-30B-A3B-Base | not neutral (main, all) | 59 | 24 | 0.27 [0.19, 0.36] | 0.480 [0.342, 0.659] | 1.299 [1.079, 1.652] | 0.128 [0.064, 0.178] | 0.872 [0.822, 0.936] | 0.804 [0.748, 0.862] | L42 0.163 | L42 0.067 |
| Mixtral-8x7B, BOS | main, validation | 128 | 12 | 0.42 [0.37, 0.48] | 0.932 [0.818, 1.073] | 1.278 [1.091, 1.475] | 0.422 [0.319, 0.548] | 0.578 [0.452, 0.681] | 0.624 [0.477, 0.747] | L19 0.174 | L19 0.144 |
| Mixtral-8x7B, BOS | main, all | 256 | 23 | 0.40 [0.36, 0.45] | 0.891 [0.793, 0.999] | 1.335 [1.131, 1.567] | 0.391 [0.328, 0.469] | 0.609 [0.531, 0.672] | 0.673 [0.574, 0.752] | L19 0.178 | L19 0.152 |
| Mixtral-8x7B, BOS | names (main, all) | 256 | 23 | 0.40 [0.36, 0.45] | 0.891 [0.793, 0.999] | 1.335 [1.131, 1.567] | 0.391 [0.328, 0.469] | 0.609 [0.531, 0.672] | 0.673 [0.574, 0.752] | L19 0.178 | L19 0.152 |
| Mixtral-8x7B, BOS | probe-neutral (main, all) | 228 | 21 | 0.40 [0.36, 0.45] | 0.900 [0.788, 1.011] | 1.355 [1.144, 1.581] | 0.378 [0.324, 0.440] | 0.622 [0.560, 0.676] | 0.690 [0.608, 0.755] | L19 0.181 | L19 0.154 |
| Mixtral-8x7B, BOS | not neutral (main, all) | 28 | 14 | 0.41 [0.36, 0.47] | 0.833 [0.731, 0.944] | 1.195 [0.982, 1.479] | 0.483 [0.338, 0.643] | 0.517 [0.357, 0.662] | 0.551 [0.367, 0.730] | L20 0.173 | L19 0.131 |


**vt_e2a_neu W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L42 | +0.801 [+0.691, +0.914] | 0.132 [0.115, 0.149] | L42E123 | 256 (≥ 128) | +0.288 [+0.217, +0.363] | +0.225 [+0.156, +0.299] | +0.781 [+0.670, +0.895] | A | +0.120 [+0.081, +0.160] |
| Qwen3-30B-A3B-Base | main | interior | L42 | +0.801 [+0.691, +0.914] | 0.132 [0.115, 0.149] | L42E123 | 256 (≥ 128) | +0.288 [+0.217, +0.363] | +0.225 [+0.156, +0.299] | +0.781 [+0.670, +0.895] | A | +0.120 [+0.081, +0.160] |
| Mixtral-8x7B, BOS | main | two_stage | L19 | +1.014 [+0.906, +1.118] | 0.174 [0.162, 0.187] | L19E006 | 245 (≥ 128) | +0.609 [+0.523, +0.695] | +0.290 [+0.190, +0.390] | +0.993 [+0.890, +1.093] | A | +0.016 [-0.022, +0.051] |
| Mixtral-8x7B, BOS | main | interior | L19 | +1.014 [+0.906, +1.118] | 0.174 [0.162, 0.187] | L19E006 | 245 (≥ 128) | +0.609 [+0.523, +0.695] | +0.290 [+0.190, +0.390] | +0.993 [+0.890, +1.093] | A | +0.016 [-0.022, +0.051] |


#### vt_a2e_orig: verb templates, attribute → entity fill-in-the-blank, original verb (task 5)

- Qwen3 (37 validation pairs, 5 seeds; drop 3.38): attention share 0.79 [0.64, 0.80], A_direct 1.47 [1.34, 1.55], M_direct -0.47 [-0.55, -0.34], all-MoE M 0.21 [0.19, 0.53]; peaks MoE L37 0.080, attention L39 0.374 of the drop; expert L37E032 (Spec +0.17 [+0.13, +0.21], pattern A); Phase-3 WinoGrande expert L41E117: rescue +0.00, Spec -0.00 (discovery-active 0); top heads L42H10 +1.13, L43H24 +0.54, L39H27 +0.51; grid: MoE at the STR site peaks L2 0.29, attention at the final position L43 0.35.

Tables: `results/tables/ext13base_vt_a2e_orig_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`; figures `results/figures/ext13base_vt_a2e_orig_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**vt_a2e_orig: headline numbers with cluster-bootstrap 95% CIs (unit = seed; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 37 | 5 | 0.79 [0.64, 0.80] | 2.331 [2.230, 3.186] | 0.623 [0.568, 1.723] | 1.469 [1.342, 1.551] | -0.469 [-0.551, -0.342] | 0.212 [0.190, 0.529] | L37 0.080 | L39 0.374 |
| Qwen3-30B-A3B-Base | main, all | 97 | 13 | 0.76 [0.73, 0.80] | 2.309 [2.063, 2.509] | 0.711 [0.561, 0.927] | 1.507 [1.418, 1.592] | -0.507 [-0.592, -0.418] | 0.196 [0.164, 0.274] | L37 0.089 | L43 0.344 |
| Qwen3-30B-A3B-Base | names (main, all) | 97 | 13 | 0.76 [0.73, 0.80] | 2.309 [2.063, 2.509] | 0.711 [0.561, 0.927] | 1.507 [1.418, 1.592] | -0.507 [-0.592, -0.418] | 0.196 [0.164, 0.274] | L37 0.089 | L43 0.344 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 77 | 12 | 0.76 [0.72, 0.80] | 2.356 [2.158, 2.529] | 0.758 [0.546, 0.963] | 1.524 [1.449, 1.595] | -0.524 [-0.595, -0.449] | 0.192 [0.158, 0.263] | L37 0.093 | L43 0.334 |
| Qwen3-30B-A3B-Base | not neutral (main, all) | 20 | 5 | 0.78 [0.75, 0.80] | 2.207 [1.876, 2.482] | 0.619 [0.493, 0.815] | 1.447 [1.323, 1.572] | -0.447 [-0.572, -0.323] | 0.211 [0.144, 0.344] | L42 0.079 | L43 0.380 |


**vt_a2e_orig W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L37 | +0.294 [+0.240, +0.350] | 0.080 [0.067, 0.095] | L37E032 | 120 (≥ 60) | +0.177 [+0.132, +0.225] | +0.167 [+0.128, +0.207] | +0.252 [+0.193, +0.309] | A | +0.025 [-0.008, +0.057] |
| Qwen3-30B-A3B-Base | main | interior | L37 | +0.294 [+0.240, +0.350] | 0.080 [0.067, 0.095] | L37E032 | 120 (≥ 60) | +0.177 [+0.132, +0.225] | +0.167 [+0.128, +0.207] | +0.252 [+0.193, +0.309] | A | +0.025 [-0.008, +0.057] |


#### f2e: attribute → entity, natural sentence cut at a final blank (task 3)

- Qwen3 (32 validation pairs, 32 pairs; drop 8.22): attention share 0.69 [0.60, 0.75], A_direct 0.99 [0.87, 1.12], M_direct 0.01 [-0.12, 0.13], all-MoE M 0.21 [0.05, 0.35]; peaks MoE L37 0.072, attention L42 0.209 of the drop; expert L43E036 (Spec +0.15 [+0.07, +0.25], pattern A); Phase-3 WinoGrande expert L41E117: rescue -0.00, Spec -0.07 (discovery-active 2); top heads L42H10 +0.98, L43H28 +0.58, L42H12 +0.48; grid: MoE at the STR site peaks L0 0.25, attention at the final position L42 0.20.
- Mixtral BOS (18 validation pairs, 18 pairs; drop 7.57): attention share 0.65 [0.59, 0.70], A_direct 1.35 [1.16, 1.55], M_direct -0.35 [-0.55, -0.16], all-MoE M -0.06 [-0.34, 0.19]; peaks MoE L19 0.084, attention L19 0.247 of the drop; expert L19E004 (Spec +0.35 [+0.16, +0.55], pattern A); Phase-3 WinoGrande expert L20E000: rescue +0.03, Spec -0.17 (discovery-active 11); top heads L20H21 +0.80, L18H10 +0.70, L16H0 +0.68; grid: MoE at the STR site peaks L0 0.40, attention at the final position L19 0.25.

Tables: `results/tables/ext13base_f2e_{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}.md`; figures `results/figures/ext13base_f2e_{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}.png`.

**f2e: headline numbers with cluster-bootstrap 95% CIs (unit = pair; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 32 | 32 | 0.69 [0.60, 0.75] | 1.300 [1.109, 1.646] | 0.586 [0.394, 1.020] | 0.986 [0.865, 1.118] | 0.014 [-0.118, 0.135] | 0.211 [0.051, 0.352] | L37 0.072 | L42 0.209 |
| Qwen3-30B-A3B-Base | main, all | 64 | 64 | 0.68 [0.62, 0.72] | 1.116 [1.006, 1.309] | 0.530 [0.415, 0.752] | 0.992 [0.909, 1.083] | 0.008 [-0.083, 0.091] | 0.150 [0.053, 0.242] | L43 0.071 | L42 0.205 |
| Qwen3-30B-A3B-Base | objects (main, all) | 55 | 55 | 0.67 [0.60, 0.72] | 1.026 [0.924, 1.189] | 0.501 [0.391, 0.740] | 0.935 [0.851, 1.021] | 0.065 [-0.021, 0.149] | 0.084 [-0.006, 0.173] | L43 0.081 | L42 0.206 |
| Qwen3-30B-A3B-Base | probe-neutral (main, all) | 64 | 64 | 0.68 [0.62, 0.72] | 1.116 [1.006, 1.309] | 0.530 [0.415, 0.752] | 0.992 [0.909, 1.083] | 0.008 [-0.083, 0.091] | 0.150 [0.053, 0.242] | L43 0.071 | L42 0.205 |
| Mixtral-8x7B, BOS | main, validation | 18 | 18 | 0.65 [0.59, 0.70] | 1.356 [1.144, 1.630] | 0.738 [0.554, 0.952] | 1.347 [1.162, 1.546] | -0.347 [-0.546, -0.162] | -0.062 [-0.337, 0.189] | L19 0.084 | L19 0.247 |
| Mixtral-8x7B, BOS | main, all | 36 | 36 | 0.67 [0.64, 0.70] | 1.311 [1.188, 1.461] | 0.650 [0.550, 0.767] | 1.320 [1.216, 1.432] | -0.320 [-0.432, -0.216] | -0.069 [-0.216, 0.075] | L19 0.084 | L19 0.247 |
| Mixtral-8x7B, BOS | objects (main, all) | 31 | 31 | 0.67 [0.63, 0.70] | 1.223 [1.130, 1.329] | 0.604 [0.514, 0.708] | 1.302 [1.185, 1.429] | -0.302 [-0.429, -0.185] | -0.173 [-0.297, -0.065] | L19 0.087 | L19 0.206 |
| Mixtral-8x7B, BOS | probe-neutral (main, all) | 36 | 36 | 0.67 [0.64, 0.70] | 1.311 [1.188, 1.461] | 0.650 [0.550, 0.767] | 1.320 [1.216, 1.432] | -0.320 [-0.432, -0.216] | -0.069 [-0.216, 0.075] | L19 0.084 | L19 0.247 |


**f2e W6: two-stage expert selection (recurrence gate half of the discovery directed cases)**

| Model | Family | Rule | Layer | Layer rescue (val) | Layer / drop | Selected expert | Disc. active | Val rescue | Spec | Clean top-k coalition | Pattern | Equal-norm Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | two_stage | L43 | +0.538 [+0.334, +0.746] | 0.065 [0.043, 0.086] | L43E036 | 64 (≥ 32) | +0.202 [+0.108, +0.306] | +0.154 [+0.068, +0.250] | +0.567 [+0.352, +0.786] | A | +0.062 [-0.008, +0.134] |
| Qwen3-30B-A3B-Base | main | interior | L43 | +0.538 [+0.334, +0.746] | 0.065 [0.043, 0.086] | L43E036 | 64 (≥ 32) | +0.202 [+0.108, +0.306] | +0.154 [+0.068, +0.250] | +0.567 [+0.352, +0.786] | A | +0.062 [-0.008, +0.134] |
| Mixtral-8x7B, BOS | main | two_stage | L19 | +0.583 [+0.389, +0.795] | 0.084 [0.061, 0.106] | L19E004 | 36 (≥ 18) | +0.486 [+0.299, +0.691] | +0.351 [+0.163, +0.549] | +0.590 [+0.403, +0.795] | A | +0.028 [-0.021, +0.073] |
| Mixtral-8x7B, BOS | main | interior | L19 | +0.583 [+0.389, +0.795] | 0.084 [0.061, 0.106] | L19E004 | 36 (≥ 18) | +0.486 [+0.299, +0.691] | +0.351 [+0.163, +0.549] | +0.590 [+0.403, +0.795] | A | +0.028 [-0.021, +0.073] |


### Behaviour: restated format, memorisation, fill-in-the-blank on templates, copy prefixes

**Restated format (F2R).** It is behaviour only, as decided. Both models are right on about two thirds of the restated prompts (Qwen3 0.62, Mixtral 0.69 per prompt), but in both twins of a pair only 0.29 / 0.39 of the time, and 9 % of the pairs pass the margin. On names the models lean to the first-mentioned candidate (Qwen3 0.64 [0.60, 0.68], Mixtral 0.58 [0.53, 0.64]); on objects they do not (0.46 / 0.33, few pairs). The restated format therefore stays unsuitable for STR tracing, as the review found (38 / 16 balanced pairs).

**Memorisation check (task 7).** The 55 entity → attribute seeds with their ORIGINAL WinoGrande names behave like the same seeds filled with new IOI names. Margin rates: Qwen3 orig 0.31 vs 0.38, neutral 0.23 vs 0.26, unrelated 0.04 vs 0.05; Mixtral 0.31 vs 0.29, 0.21 vs 0.23, 0.03 vs 0.02. Mean drops also match (4.6 vs 4.8 logits orig, ≈ 0.3 vs 0.3 unrelated in Qwen3), with seed-clustered CIs overlapping everywhere. The original names give the models no advantage, and the unrelated verb removes the effect with either name set. So the name items are not solved by recall of the WinoGrande sentence: the models read the verb. Caveat: Qwen3 has 48 and Mixtral 29 of the 55 seeds, because the other original names are not single tokens or break token alignment.

**Fill-in-the-blank on the verb templates (task 5) fails.** Qwen3 passes the margin on 3.7 % (orig) and 1.9 % (neutral) of the 2,640 template items, Mixtral on 0.1 % / 0.2 %. Correct in both twins: Qwen3 0.27 / 0.22, Mixtral 0.23 / 0.20, at or below the 0.25 of an independent coin flip per prompt. The same items in the entity → attribute format pass 39 % / 27 % (Qwen3) and 33 % / 21 % (Mixtral). Natural fill-in-the-blank name items do pass (Qwen3 520 balanced name pairs, original WinoGrande sentences). So in this format the models resolve the blank on natural sentences but not on the semi-synthetic templates with new names. `vt_a2e_orig` runs for Qwen3 only (60 / 37 pairs, 8 / 5 seeds); `vt_a2e_neu` (Qwen3) and both Mixtral template a2e conditions are behaviour only, so task 5's direction contrast on identical template items does not exist at the margin rule.

**Copy prefixes (task 6).** The first prefix, `Of X and Y, one … t1 and the other … t2`, admits a 'respectively' reading (X ↔ t1, Y ↔ t2). Twin A's answer is option 1 (X) in every template and natural item, so one attribute order is always congruent with that reading, which answers both twins without the sentence, and the other order always contradicts it (`ext13base_cp_respectively`). Margin rates, congruent vs incongruent:

| Items | Qwen3 | Mixtral |
|---|---|---|
| Templates, orig verb | 0.43 vs 0.10 | 0.67 vs 0.12 |
| Templates, neutral verb | 0.37 vs 0.07 | 0.73 vs 0.04 |
| Templates, unrelated verb | 0.07 vs 0.00 | 0.43 vs 0.00 |
| Natural twins | 0.90 vs 0.63 | 0.98 vs 0.57 |

With the unrelated verb the sentence gives no answer, yet Mixtral passes 43 % in the congruent order: it binds the attribute to the name by list position. This positional effect is a finding in itself, and it makes the prefix unusable for patching: every average would mix two mechanisms of opposite sign. The prefix `Options: t1, t2.` pairs nothing with the names, and its two orders pass about equally. Natural twins: order 1 / order 2 0.73 / 0.74 (Qwen3), 0.90 / 0.89 (Mixtral). Templates, orig verb: 0.26 / 0.35 and 0.33 / 0.25. Unrelated verb: ≤ 0.03 in every cell, so no copy without the verb. Items passing both orders: natural 464 / 653 twins; templates orig 615 / 436, neutral 350 / 228 items (`ext13base_cp_options`). The copy prefix makes templates no easier in Qwen3: orig margin 0.26 / 0.35 vs 0.39 without a prefix (vt_e2a). In Mixtral it is about neutral (0.33 / 0.25 vs 0.33).

**Restated format (F2R; behaviour only): whole sentence with the blank → 'one of them', the predicate restated, entity readout (95% bootstrap CIs over pairs)**

| Model | Items | Pairs (content swaps) | Correct per prompt | Correct in both twins | Margin ≥ 1 both ways | Mean Δ toward the answer (logits) | First-mentioned candidate predicted |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | names | 370 | 0.61 [0.59, 0.64] | 0.28 [0.24, 0.33] | 0.09 [0.06, 0.12] | 0.64 [0.55, 0.74] | 0.64 [0.60, 0.68] |
| Qwen3-30B-A3B-Base | objects | 34 | 0.68 [0.60, 0.76] | 0.35 [0.21, 0.53] | 0.15 [0.03, 0.26] | 0.67 [0.42, 0.93] | 0.46 [0.32, 0.59] |
| Qwen3-30B-A3B-Base | all | 404 | 0.62 [0.59, 0.65] | 0.29 [0.25, 0.33] | 0.09 [0.07, 0.12] | 0.65 [0.56, 0.74] | 0.63 [0.59, 0.66] |
| Mixtral-8x7B, BOS | names | 160 | 0.69 [0.65, 0.73] | 0.39 [0.31, 0.46] | 0.09 [0.04, 0.13] | 0.58 [0.49, 0.67] | 0.58 [0.53, 0.64] |
| Mixtral-8x7B, BOS | objects | 21 | 0.71 [0.62, 0.83] | 0.43 [0.24, 0.67] | 0.10 [0.00, 0.24] | 0.63 [0.37, 0.94] | 0.33 [0.19, 0.48] |
| Mixtral-8x7B, BOS | all | 181 | 0.69 [0.65, 0.73] | 0.39 [0.33, 0.46] | 0.09 [0.05, 0.13] | 0.58 [0.50, 0.68] | 0.55 [0.50, 0.60] |


**Memorisation check (task 7): the 55 entity → attribute seeds with their ORIGINAL WinoGrande names vs the IOI-name fills of the same seeds, orig / neutral / unrelated verb (Phase-3 format; seed-clustered bootstrap 95% CIs)**

| Model | Verb | Names | Pairs | Seeds | Margin ≥ 1 both ways | Correct in both twins | Mean drop (logits) |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | orig | original names | 48 | 48 | 0.31 [0.19, 0.44] | 0.50 [0.35, 0.65] | 4.61 [3.90, 5.32] |
| Qwen3-30B-A3B-Base | orig | IOI names (same seeds) | 2304 | 48 | 0.38 [0.26, 0.50] | 0.51 [0.39, 0.64] | 4.76 [4.07, 5.44] |
| Qwen3-30B-A3B-Base | neutral | original names | 48 | 48 | 0.23 [0.12, 0.35] | 0.42 [0.27, 0.56] | 3.88 [3.23, 4.56] |
| Qwen3-30B-A3B-Base | neutral | IOI names (same seeds) | 2304 | 48 | 0.26 [0.16, 0.37] | 0.41 [0.30, 0.53] | 4.09 [3.39, 4.80] |
| Qwen3-30B-A3B-Base | unrelated | original names | 48 | 48 | 0.04 [0.00, 0.10] | 0.12 [0.04, 0.23] | 0.39 [0.02, 0.74] |
| Qwen3-30B-A3B-Base | unrelated | IOI names (same seeds) | 2304 | 48 | 0.05 [0.01, 0.10] | 0.16 [0.09, 0.25] | 0.31 [-0.12, 0.76] |
| Mixtral-8x7B, BOS | orig | original names | 29 | 29 | 0.31 [0.14, 0.48] | 0.45 [0.28, 0.62] | 4.50 [3.78, 5.24] |
| Mixtral-8x7B, BOS | orig | IOI names (same seeds) | 1392 | 29 | 0.29 [0.16, 0.43] | 0.50 [0.34, 0.65] | 4.64 [4.01, 5.33] |
| Mixtral-8x7B, BOS | neutral | original names | 29 | 29 | 0.21 [0.07, 0.38] | 0.31 [0.14, 0.48] | 3.83 [3.11, 4.51] |
| Mixtral-8x7B, BOS | neutral | IOI names (same seeds) | 1392 | 29 | 0.23 [0.10, 0.37] | 0.38 [0.23, 0.54] | 3.88 [3.26, 4.49] |
| Mixtral-8x7B, BOS | unrelated | original names | 29 | 29 | 0.03 [0.00, 0.10] | 0.10 [0.00, 0.21] | 0.03 [-0.60, 0.65] |
| Mixtral-8x7B, BOS | unrelated | IOI names (same seeds) | 1392 | 29 | 0.02 [0.00, 0.06] | 0.11 [0.02, 0.23] | 0.04 [-0.57, 0.61] |


**Copy prefix 'Options: t1, t2.' + newline (current; coordinator decision 2026-10-11): pass rates per attribute order (cluster bootstrap over seeds / twins)**

| Model | Build | Subset | Pairs / items | Margin ≥ 1 both ways | Correct in both twins | Mean drop | Items passing both orders |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | cp_vt | neutral, order 1 | 2640 | 0.19 [0.11, 0.27] | 0.41 [0.30, 0.52] | 3.06 [2.48, 3.63] |  |
| Qwen3-30B-A3B-Base | cp_vt | neutral, order 2 | 2640 | 0.25 [0.15, 0.35] | 0.52 [0.41, 0.63] | 3.17 [2.55, 3.77] |  |
| Qwen3-30B-A3B-Base | cp_vt | orig, order 1 | 2640 | 0.26 [0.16, 0.37] | 0.45 [0.34, 0.55] | 3.84 [3.26, 4.42] |  |
| Qwen3-30B-A3B-Base | cp_vt | orig, order 2 | 2640 | 0.35 [0.25, 0.46] | 0.62 [0.51, 0.72] | 4.05 [3.49, 4.58] |  |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, order 1 | 2640 | 0.02 [0.00, 0.06] | 0.06 [0.02, 0.11] | 0.09 [-0.23, 0.41] |  |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, order 2 | 2640 | 0.03 [0.00, 0.08] | 0.11 [0.05, 0.17] | 0.32 [-0.04, 0.70] |  |
| Qwen3-30B-A3B-Base | cp_vt | orig, both orders pass | 2640 | 0.233 of items |  |  | 615 |
| Qwen3-30B-A3B-Base | cp_vt | neutral, both orders pass | 2640 | 0.133 of items |  |  | 350 |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, both orders pass | 2640 | 0.017 of items |  |  | 45 |
| Qwen3-30B-A3B-Base | cp_nat | order 1 | 776 | 0.73 [0.70, 0.77] | 0.92 [0.90, 0.94] | 5.40 [5.24, 5.56] |  |
| Qwen3-30B-A3B-Base | cp_nat | order 2 | 776 | 0.74 [0.71, 0.77] | 0.93 [0.91, 0.95] | 5.39 [5.24, 5.54] |  |
| Qwen3-30B-A3B-Base | cp_nat | both orders pass | 776 | 0.598 of twins |  |  | 464 |
| Mixtral-8x7B, BOS | cp_vt | neutral, order 1 | 2640 | 0.24 [0.14, 0.34] | 0.43 [0.32, 0.54] | 3.16 [2.67, 3.66] |  |
| Mixtral-8x7B, BOS | cp_vt | neutral, order 2 | 2640 | 0.14 [0.06, 0.22] | 0.43 [0.32, 0.54] | 2.69 [2.19, 3.19] |  |
| Mixtral-8x7B, BOS | cp_vt | orig, order 1 | 2640 | 0.33 [0.23, 0.45] | 0.52 [0.41, 0.63] | 3.75 [3.22, 4.30] |  |
| Mixtral-8x7B, BOS | cp_vt | orig, order 2 | 2640 | 0.25 [0.16, 0.34] | 0.45 [0.33, 0.57] | 3.23 [2.71, 3.78] |  |
| Mixtral-8x7B, BOS | cp_vt | unrelated, order 1 | 2640 | 0.02 [0.00, 0.05] | 0.10 [0.04, 0.18] | 0.57 [0.16, 0.94] |  |
| Mixtral-8x7B, BOS | cp_vt | unrelated, order 2 | 2640 | 0.02 [0.00, 0.06] | 0.05 [0.01, 0.10] | -0.11 [-0.63, 0.37] |  |
| Mixtral-8x7B, BOS | cp_vt | orig, both orders pass | 2640 | 0.165 of items |  |  | 436 |
| Mixtral-8x7B, BOS | cp_vt | neutral, both orders pass | 2640 | 0.086 of items |  |  | 228 |
| Mixtral-8x7B, BOS | cp_vt | unrelated, both orders pass | 2640 | 0.009 of items |  |  | 23 |
| Mixtral-8x7B, BOS | cp_nat | order 1 | 776 | 0.90 [0.88, 0.92] | 0.98 [0.97, 0.99] | 6.91 [6.75, 7.06] |  |
| Mixtral-8x7B, BOS | cp_nat | order 2 | 776 | 0.89 [0.87, 0.91] | 0.97 [0.96, 0.98] | 6.74 [6.58, 6.91] |  |
| Mixtral-8x7B, BOS | cp_nat | both orders pass | 776 | 0.841 of twins |  |  | 653 |


**VOID copy prefix 'Of X and Y, one <frame> t1 and the other <frame> t2.' (behaviour record): twin A's answer is option 1 (X) in every item, so the order with t1 = A's attribute is 'respectively'-congruent (X ↔ t1 answers both twins without the sentence); cluster bootstrap over seeds / twins**

| Model | Build | Subset | Pairs | Margin ≥ 1 both ways | Correct in both twins | Mean drop |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | cp_vt | neutral, incongruent | 2256 | 0.07 [0.03, 0.11] | 0.30 [0.20, 0.41] | 1.04 [0.50, 1.55] |
| Qwen3-30B-A3B-Base | cp_vt | neutral, congruent (X ↔ t1) | 2256 | 0.37 [0.26, 0.48] | 0.68 [0.58, 0.78] | 3.48 [2.95, 4.00] |
| Qwen3-30B-A3B-Base | cp_vt | orig, incongruent | 2256 | 0.10 [0.05, 0.16] | 0.37 [0.26, 0.47] | 1.54 [0.96, 2.12] |
| Qwen3-30B-A3B-Base | cp_vt | orig, congruent (X ↔ t1) | 2256 | 0.43 [0.32, 0.54] | 0.73 [0.62, 0.82] | 3.78 [3.13, 4.45] |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, incongruent | 2256 | 0.00 [0.00, 0.01] | 0.03 [0.01, 0.08] | -1.17 [-1.63, -0.75] |
| Qwen3-30B-A3B-Base | cp_vt | unrelated, congruent (X ↔ t1) | 2256 | 0.07 [0.03, 0.14] | 0.37 [0.27, 0.48] | 1.21 [0.80, 1.66] |
| Qwen3-30B-A3B-Base | cp_nat | incongruent | 693 | 0.63 [0.60, 0.67] | 0.83 [0.80, 0.86] | 4.82 [4.59, 5.05] |
| Qwen3-30B-A3B-Base | cp_nat | congruent (X ↔ t1) | 693 | 0.90 [0.88, 0.92] | 0.98 [0.97, 0.99] | 6.98 [6.77, 7.17] |
| Mixtral-8x7B, BOS | cp_vt | neutral, incongruent | 2256 | 0.04 [0.01, 0.09] | 0.15 [0.07, 0.24] | -0.18 [-0.85, 0.52] |
| Mixtral-8x7B, BOS | cp_vt | neutral, congruent (X ↔ t1) | 2256 | 0.73 [0.62, 0.82] | 0.91 [0.83, 0.97] | 6.11 [5.47, 6.76] |
| Mixtral-8x7B, BOS | cp_vt | orig, incongruent | 2256 | 0.12 [0.05, 0.19] | 0.25 [0.15, 0.37] | 0.50 [-0.25, 1.25] |
| Mixtral-8x7B, BOS | cp_vt | orig, congruent (X ↔ t1) | 2256 | 0.67 [0.55, 0.77] | 0.88 [0.80, 0.95] | 6.35 [5.61, 7.12] |
| Mixtral-8x7B, BOS | cp_vt | unrelated, incongruent | 2256 | 0.00 [0.00, 0.01] | 0.02 [0.00, 0.06] | -2.87 [-3.45, -2.28] |
| Mixtral-8x7B, BOS | cp_vt | unrelated, congruent (X ↔ t1) | 2256 | 0.43 [0.31, 0.55] | 0.69 [0.57, 0.80] | 3.57 [3.04, 4.08] |
| Mixtral-8x7B, BOS | cp_nat | incongruent | 693 | 0.57 [0.53, 0.61] | 0.73 [0.70, 0.76] | 4.74 [4.42, 5.05] |
| Mixtral-8x7B, BOS | cp_nat | congruent (X ↔ t1) | 693 | 0.98 [0.97, 0.99] | 1.00 [0.99, 1.00] | 9.60 [9.40, 9.81] |


**Competence scans of the Phase-5 builds (cluster bootstrap over seeds / Phase-3 twins); verbs_ent2attr = the existing IOI-name entity → attribute scan**

| Model | Build | Subset | Pairs | Units (seeds / twins) | Margin ≥ 1 both ways | Correct in both twins | Mean drop |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | vt_a2e | variant neutral | 2640 | 55 | 0.02 [0.00, 0.05] | 0.22 [0.16, 0.28] | 0.72 [0.53, 0.92] |
| Qwen3-30B-A3B-Base | vt_a2e | variant orig | 2640 | 55 | 0.04 [0.01, 0.07] | 0.27 [0.20, 0.34] | 0.97 [0.73, 1.22] |
| Qwen3-30B-A3B-Base | vt_a2e | variant unrelated | 2640 | 55 | 0.00 [0.00, 0.00] | 0.05 [0.03, 0.08] | 0.01 [-0.14, 0.16] |
| Qwen3-30B-A3B-Base | cp_vt | variant neutral | 5280 | 55 | 0.22 [0.14, 0.30] | 0.47 [0.36, 0.57] | 3.11 [2.53, 3.68] |
| Qwen3-30B-A3B-Base | cp_vt | variant orig | 5280 | 55 | 0.31 [0.21, 0.41] | 0.53 [0.43, 0.63] | 3.95 [3.40, 4.48] |
| Qwen3-30B-A3B-Base | cp_vt | variant unrelated | 5280 | 55 | 0.03 [0.00, 0.07] | 0.08 [0.04, 0.14] | 0.20 [-0.11, 0.53] |
| Qwen3-30B-A3B-Base | cp_nat | order 1 | 776 | 776 | 0.73 [0.70, 0.77] | 0.92 [0.90, 0.94] | 5.40 [5.24, 5.56] |
| Qwen3-30B-A3B-Base | cp_nat | order 2 | 776 | 776 | 0.74 [0.71, 0.77] | 0.93 [0.91, 0.95] | 5.39 [5.24, 5.54] |
| Qwen3-30B-A3B-Base | verbs_ent2attr | variant neutral | 2640 | 55 | 0.27 [0.18, 0.37] | 0.42 [0.31, 0.53] | 4.25 [3.48, 4.99] |
| Qwen3-30B-A3B-Base | verbs_ent2attr | variant orig | 2640 | 55 | 0.39 [0.28, 0.50] | 0.53 [0.42, 0.65] | 4.97 [4.26, 5.65] |
| Qwen3-30B-A3B-Base | verbs_ent2attr | variant unrelated | 2640 | 55 | 0.05 [0.01, 0.09] | 0.16 [0.09, 0.23] | 0.22 [-0.20, 0.61] |
| Mixtral-8x7B, BOS | vt_a2e | variant neutral | 2640 | 55 | 0.00 [0.00, 0.00] | 0.20 [0.15, 0.25] | 0.37 [0.26, 0.47] |
| Mixtral-8x7B, BOS | vt_a2e | variant orig | 2640 | 55 | 0.00 [0.00, 0.00] | 0.23 [0.17, 0.30] | 0.46 [0.33, 0.61] |
| Mixtral-8x7B, BOS | vt_a2e | variant unrelated | 2640 | 55 | 0.00 [0.00, 0.00] | 0.06 [0.04, 0.09] | -0.02 [-0.10, 0.05] |
| Mixtral-8x7B, BOS | cp_vt | variant neutral | 5280 | 55 | 0.19 [0.11, 0.26] | 0.43 [0.34, 0.52] | 2.92 [2.44, 3.41] |
| Mixtral-8x7B, BOS | cp_vt | variant orig | 5280 | 55 | 0.29 [0.21, 0.38] | 0.48 [0.38, 0.59] | 3.49 [2.98, 4.02] |
| Mixtral-8x7B, BOS | cp_vt | variant unrelated | 5280 | 55 | 0.02 [0.00, 0.05] | 0.08 [0.03, 0.13] | 0.23 [-0.23, 0.64] |
| Mixtral-8x7B, BOS | cp_nat | order 1 | 776 | 776 | 0.90 [0.88, 0.92] | 0.98 [0.97, 0.99] | 6.91 [6.75, 7.06] |
| Mixtral-8x7B, BOS | cp_nat | order 2 | 776 | 776 | 0.89 [0.87, 0.91] | 0.97 [0.96, 0.98] | 6.74 [6.58, 6.91] |
| Mixtral-8x7B, BOS | verbs_ent2attr | variant neutral | 2640 | 55 | 0.21 [0.12, 0.30] | 0.39 [0.28, 0.50] | 3.91 [3.31, 4.52] |
| Mixtral-8x7B, BOS | verbs_ent2attr | variant orig | 2640 | 55 | 0.33 [0.22, 0.43] | 0.51 [0.39, 0.63] | 4.71 [4.09, 5.34] |
| Mixtral-8x7B, BOS | verbs_ent2attr | variant unrelated | 2640 | 55 | 0.02 [0.00, 0.04] | 0.08 [0.03, 0.15] | 0.00 [-0.61, 0.50] |


### Deviations and caveats

- **Copy prefix replaced** (coordinator decision 2026-10-11): `Options: t1, t2.` + newline instead of `Of X and Y, one … and the other …`, which is answered by name position. Rebuilt, rescanned (jobs ext13-p5-cpopt-scan-*) and re-announced 09:43Z as `CASES READY <cond>/<proto> (options prefix)`, before any cp_* base run. The registry flag `rebuild_pending` made the job planner hold the running chain until the new files were registered. The both-orders rule is kept. The void prefix's grammar rules C1–C6 (`moetrace/ext13_p5.frame_of`) are no longer used for tracing.
- **f1 Mixtral:** half / half (101 / 101) instead of 128 / 128 + replication, because the neutral pool has 203 pairs. **f2e** is small: 32 / 32 (Qwen3) and 18 / 18 (Mixtral).
- **Behaviour only** (< 16 pairs in a split): vt_a2e_neu (Qwen3), vt_a2e_orig and vt_a2e_neu (Mixtral). vt_a2e_orig (Qwen3) runs with 60 / 37 pairs from 8 / 5 seeds.
- **Statistics.** The Phase-3 functions bootstrap over pairs. The headline table adds a cluster bootstrap over the independent unit: seed for templates (effective n = number of seeds) and Phase-3 twin for cp_nat (its two attribute orders are dependent). Pair-level CIs of the Phase-3 tables are anti-conservative for these conditions.
- **Mention positions.** Every Phase-5 pair file carries explicit mention_a_pos / mention_b_pos: the first occurrence in the sentence before the blank, after the copy prefix for cp_*. The heads runner therefore uses the sentence mentions, not the prefix names. Copy-prefix positions are separate columns, and `ext13base_<cp>_w5_copy` reads them.
- **Runner invocation.** `python -m moetrace.ext7_controls direct` has no `--agent` flag. The first f1 / Qwen3 direct job failed (rc 2, no GPU work lost). The job planner was fixed and the chain restarted with stage markers, without redoing completed stages.
- **Probe.** v1 is flat in Mixtral (2 % responding), so Mixtral's neutral rule uses v2 only. The probes are partial-input baselines with moderate agreement (r 0.51 / 0.28); the reverse / one-sided / associative strata are descriptive.
- **Memorisation.** The original-name items cover only the seeds whose original names are single tokens with token alignment: Qwen3 48 / 55, Mixtral 29 / 55.
- **vt_a2e_orig (Qwen3)** is traced with 60 / 37 pairs from 8 / 5 seeds. Its headline CIs are seed-clustered, and five validation seeds make them wide; read it as indicative. The rest of task 5 is behaviour only (coordinator decision).
