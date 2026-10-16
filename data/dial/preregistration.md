# Phase 6 pre-registration (written 2026-10-16, before any Phase-6 GPU run)

This file and its machine-readable twin `data/dial/predictions.json` (45 predictions, IDs below) are committed before the first Stage-1 GPU job. After the runs, `results/tables/ext14_prereg_check.md` lists every prediction with its observed value and its verdict (held / failed / insufficient data). The predictions are not edited after the commit; corrections go into a dated addendum at the end of this file. The team is invited to add their own guesses in the addendum before the results are shown.

**Metric.** attn_share = AUC+ of the single-layer attention-output patch curve at the final token / (AUC+ attention + AUC+ MoE or MLP). STR pairs, validation split, LD = logit(answer) − logit(contrast) unless stated. Known values for reference (Qwen3 / Mixtral BOS):
- entity → attribute 0.16–0.31 / 0.34–0.42;
- CounterFact 0.54 / 0.58;
- attribute → entity 0.69–0.79 / 0.52–0.65;
- IOI 0.92 / 0.80.

**The dial.** Where the deciding information is and how many transformations separate it from the answer token. Copying → high attention share; transforming → low.

## Ladder v2 (different tasks, grades fixed in advance)

| Grade | Task | Qwen3 | Mixtral BOS |
|---|---|---|---|
| 1 | L-ant antonym (`The opposite of hot is` → cold) | ≤ 0.30 (L1) | ≤ 0.40 (L2) |
| 1.5 | L-prop object property (`The color of a banana is` → yellow) | 0.30–0.55 (L3) | 0.35–0.60 (L4) |
| 2 | L-e2a, L-cf (anchors) | known | known |
| 3 | L-a2e (anchor) | known | known |
| 4 | L-bind explicit binding look-up | ≥ 0.75 (L5) | ≥ 0.65 (L6) |
| 5 | L-ioi (anchor), L-ind induction on word pairs | ≥ 0.85 for L-ind (L7) | ≥ 0.75 (L8) |

L9: Spearman(grade, share) ≥ 0.8 in every model.

## Dial E (same WinoGrande items; only the evidence changes)

| Level | a2e (entity is the answer) | e2a (attribute is the answer) |
|---|---|---|
| E1 verbatim | Qwen3 ≥ 0.80 (E1), Mixtral ≥ 0.70 (E2) | Qwen3 ≥ 0.70 (E7), Mixtral ≥ 0.65 (E8) |
| E1p paraphrase | within 0.10 of E1 (E3) | at least 0.10 below E1 (E9) |
| E2 negation | at least 0.10 below E1 (E4) | below E1p (E10) |
| E3 event (original) | Qwen3 0.65–0.80 (E5), Mixtral 0.45–0.65 (E6) | Qwen3 0.10–0.30 (E11), Mixtral 0.25–0.45 (E12) |

E13, the direction gap at E1, has two competing hypotheses:
- **H-inference (our best guess):** the gap closes (|a2e − e2a| ≤ 0.10). The Phase-5 direction effect came from the inference that entity → attribute needs.
- **H-type:** the gap stays at ≥ half of the E3 gap. Entities are moved by attention and attributes are written by the MoE regardless of the evidence.

E14 (monotonic order of the levels):
- a2e: E1 ≥ E1p > E2, and E1 > E3.
- e2a: E1 > E1p > E2 ≥ E3.

## Dial A (associativity, at E1) and E0 (prior only)

- A1: share(conflict) ≥ share(neutral) > share(agree) > share(E0).
- A2: in conflict cells the MoE's direct contribution toward the stated answer is negative.
- A3 / A4: E0 share ≤ 0.35 (Qwen3) / ≤ 0.45 (Mixtral).
- A5: across five prior-strength bins, the share decreases as prior support increases.

## Set C (event + statement: which source wins)

- C1: the explicit statement wins in ≥ 60 % of margin items in every model.
- C2: on items where the statement wins, the share is within 0.10 of E1.
- C3: on items where the event wins, the share is within 0.10 of E3.

## Robustness and causal checks

- R1: the cell ranking is the same under LD, normalised LD and Δp (Spearman ≥ 0.8).
- R2: with the corruption in the evidence instead of the question (E1-a2e), the share changes by ≤ 0.15.
- R3: knocking out the top-3 mover heads lowers attention AUC+ by ≥ 20 % and the share by ≥ 0.05, beyond matched random heads.
- R4: knocking out the top-3 experts lowers MoE AUC+ by ≥ 20 %, beyond matched random experts.

## Dense vs MoE, capacity (Stage 2)

- D0 (best guess): **H-invariant** for Mistral-7B ↔ Mixtral-8x7B (prior: Li et al., ACL 2025, near-identical attention / FFN attribution).
- D1, H-invariant: |dense − MoE| ≤ 0.10 in ≥ 80 % of cells.
- D2, H-capacity: in knowledge cells (L-ant, L-prop, L-cf, E3-e2a) the MoE share is ≥ 0.10 lower than the dense share; in copy cells (L-ind, L-ioi, L-bind, E1) the difference is ≤ 0.10.
- D3, H-sparsity: the MoE share is ≥ 0.10 higher than the dense share in most cells.
- D4: the cell order is preserved between twins (Spearman ≥ 0.8).
- D5: Qwen3 4B → 8B → 14B: the per-cell trend is ≤ 0.10, and the order is the same as in Qwen3-30B-A3B.
- D6: in Qwen2-57B-A14B the always-on shared expert carries ≥ 50 % of the MoE rescue in the knowledge cells.

## Circuits

- N1: 80 % of a selected Qwen3 expert's rescue comes from ≤ 50 of its 768 neurons.
- N2: 80 % of a selected Mixtral expert's rescue comes from ≤ 200 of its 14,336 neurons.
- N3: in the concept × task knockout matrix, experts are more concept-specific than heads.

## Addendum

(empty)
