# CLAUDE.md — project memory for agents working in /home/ubuntu/MOE

Read this first. It is the entry point for any new agent session in this repository: what the project is, what has
already been done and with what result, how the codebase is used, and the rules that keep work here consistent.
Details live in the documents listed in section 8; this file is the map. Last full refresh: 2026-09-20; Phase 2 wave 1 added 2026-09-21;
Direction 6 (STR best-practices check) and 6b (STR layer × position grid) added 2026-09-28; Phase 3 2026-10-04; Phase 4 2026-10-05.

## 1. What this project is

A full-scale, independent reproduction of **Lu, Modarressi, Liu, Schütze (2026), "Expert-Aware Causal Tracing of
Factual Recall in Sparse MoE Language Models"** (arXiv:2606.03780), followed by four extension studies. The paper
released no code. We re-implemented the protocol from the text and re-ran every table and Figure 1 on the paper's exact
models, Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1, in bf16, on this machine's single 24 GB A10G.

**Status.** Base reproduction COMPLETE (2026-09-03, marker `results/DONE`, deliverable `results/REPORT.md`).
Extensions COMPLETE (2026-09-14, deliverable `results/EXTENSIONS_REPORT.md`, plan `RESEARCH_PLAN.md`).
**Phase 2 wave 1 COMPLETE (2026-09-21: F4 subject-token patching, F1 rankings/minimal sets, F2 attention heads,
F5 probability metrics; sections `results/sections/ext5_*.md`, assembled into `EXTENSIONS_REPORT.md` as Directions
5-F4/5-F1/5-F2/5-F5).** Wave 2 (F3 gradient attribution, F1.4 multi-layer sets) NOT started.
**Direction 6 COMPLETE (2026-09-28): symmetric token replacement (STR) instead of Gaussian noise, the corruption recommended
by Zhang & Nanda 2024 (arXiv:2309.16042, PDF gitignored at repo root); section `results/sections/ext6_str.md`.**
**Direction 6b COMPLETE for Qwen3 and Mixtral BOS (2026-09-28): STR layer × position grid, section
`results/sections/ext6_str_grid.md`; Mixtral without BOS awaits the user's decision.** Everything, including Direction 6b,
is committed and pushed to github.com/KenSu223/MOE (branch `main`). Open items are in section 3.
**Phase 3 COMPLETE (2026-10-04, committed and pushed 2026-10-05; plan, decisions, agent split in RESEARCH_PLAN.md "Phase 3"; sections
`results/sections/ext7_synthesis.md`, `ext7_wino.md`, `ext7_controls.md`, `ext8_addback.md`, assembled into EXTENSIONS_REPORT.md
as Directions 7-8 / 7 / 7b / 8): expert add-back curves (ext8) and WinoGrande / IOI under STR (ext7). STR only (no GN); Mixtral =
BOS for Phase 3. Headlines in section 2b.**
**Phase 4 COMPLETE (2026-10-05, committed and pushed 2026-10-06; plan in RESEARCH_PLAN.md "Phase 4"; sections `results/sections/ext9_synthesis.md`,
`ext9_knockout.md`, `ext10_circuit.md`, `ext11_writer.md`, `ext12_complete.md`, assembled as Directions 9-12 / 9 / 10 / 11 / 12):
expert knockout (necessity), head + expert add-back to full repair, writer vs computer experts, completeness checks (replication,
role vs option swap on identical items, Mixtral no BOS on WinoGrande). Engine E4 (route masks, head steps in `multi`,
contribution vectors). Headlines in section 2b.**

## 2. Results in one screen (base reproduction)

| Model | Run | Layer | Layer rescue (val) | Expert | Expert rescue | Specificity |
|---|---|---|---|---|---|---|
| Qwen3 | paper | L44 | +0.901 [+0.752, +1.053] | L44E069 | +0.463 | +0.400 |
| Qwen3 | ours (tokenizer defaults) | L44 | +0.941 | L44E069 | +0.503 | +0.450 |
| Mixtral | paper | L19 | +0.457 [+0.331, +0.579] | L19E006 | +0.099 | −0.175 |
| Mixtral | ours, BOS (tokenizer default) | L19 | +0.571 | **L19E002** | +0.352 | **+0.205** |
| Mixtral | ours, **no BOS** | L19 | +0.446 | L19E006 | +0.073 | −0.171 |

- **Qwen3 reproduces on every table** with tokenizer defaults; all validation means inside the paper's 95% CIs;
  Appendix D stability 25/25 and 5/5 relation folds as in the paper.
- **Mixtral reproduces only when prompts are tokenised WITHOUT the BOS token** (`add_special_tokens=False`). Then
  L19E006 is the sole recurrent candidate, its clean-active counts (91/128 discovery, 83/128 validation) equal the
  paper's exactly, specificity is negative, coalitions recover the layer effect (Tables 5, 6, 11, 16 all match). With
  BOS, routing at the final position shifts: E002 becomes recurrent (76/128) and positively specific and gets selected.
  Conclusion: the paper's Mixtral protocol omitted BOS (not stated in the paper). Qwen3's tokenizer adds no BOS, so
  Qwen3 is unaffected.
- Second under-specified detail: the paper most likely resolved object tokens as `tok(obj)` when that is a single
  token, else the leading-space token (`--token-rule paper_like`; 95% agreement with the paper's Qwen3 case funnel vs
  77% for our default). It does not change any conclusion; reported as a secondary run.
- Numerics: our engine's deviation from transformers equals transformers' own bf16 noise (fp32 CPU reference on
  OLMoE); big-model checks vs transformers-with-offload on 5 prompts each: max |ΔΔ| 0.25 (Qwen3), 0.06 (Mixtral).

Vocabulary used everywhere: Δ = logit(true) − logit(foil) at the final position; drop = Δ_clean − Δ_noised;
rescue = Δ_patched − Δ_noised (all in logits, validation-set means of 128 cases unless stated); Spec = rescue of the
selected expert minus the mean rescue of other clean-active experts of the same prompt and layer; "clean-active" =
routed to in the clean run at the final position; recurrence gate = clean-active in ≥ 64 of 128 discovery cases.

## 2b. Extensions (RESEARCH_PLAN.md; results in results/EXTENSIONS_REPORT.md, sections in results/sections/)

Both waves done 2026-09-14 (wave 1: Directions 1 and 3; wave 2: Directions 2, 2b and 4). GPU total for the extensions ≈ 4 h.
- **Direction 1 (joint layer×expert search, all-layer expert passes in `results/*_alllayers`)**: the two-stage
  procedure does not miss a *better* expert in Qwen3 (L44E069 is the joint argmax) but misses a *second locus*:
  L42E115 (126/128 recurrent, val +0.447 [0.363, 0.537], Spec +0.423) wins the discovery argmax on the strict and
  relaxed sets and in 30/75 grid cells. Mixtral under the paper protocol (no BOS): the joint winner is **L18E001**
  (Spec +0.098 [0.040, 0.162]) while the paper's L19E006 has Spec −0.159; E001 is the strongest expert of L17/18/21/22
  under both protocols; L19E002 is the strongest L19 expert without BOS too but fails the 64/128 recurrence gate.
- **Direction 3 (BOS mechanism)**: `<s>` is a prompt-independent attention sink whose per-layer K/V alone reproduce
  the BOS run when transplanted into no-BOS prompts (routing agreement 0.99); a key-only sink breaks the model.
  Without BOS, Mixtral forms no position-0 sink; in 63/256 prompts the final token itself becomes the sink state and
  is routed to E006 (63/63), which is the L19 "sink expert" (P(E006|`<s>`)=1.00) but an ordinary content expert on
  corpus text. Any position-0 token that forms a sink (`\n`, `,`, `:`, attached `.`, `▁`) restores the BOS-run routing;
  `the`, `▁.`, rare words do not. RoPE shift is a no-op (relative positions). Qwen3's L44E069 survives a prepended
  `<|endoftext|>`. Consequence: the paper's negative Mixtral specificity is expected once a quarter of its prompts
  route the final token like a sink. Engine gained optional diagnostics (`DiagSpec`), `PrefillSpec.pos_offset`,
  sink transplant (`sink_donor`), `prefix_ids` in `prepare_case`; verify_olmoe unchanged.
- Literature: docs/ext3_literature_review.md (Mistral is the documented exception: BOS is its sole sink driver).
- **Direction 2b (attention vs MoE patching; engine kinds `attn_layer`, `block`, `resid`, `block_diff`, runs
  `results/*_attnsweep`)**: the paper's MoE-only patch misses the largest single-sublayer locus. Qwen3: attention
  output at **L40 +1.59 [1.41, 1.79]** (block L40 +1.95) vs MoE L44 +0.93; L44 is a pure-MoE layer (attention share
  2%) but overall attention carries 51% of the positive rescue. Mixtral: attention L18 +0.99, MoE L19 +0.56, block L19
  +1.37; at L19 attention carries 62% [58, 66]. block = attn + MoE to bf16 noise (r 0.96-0.99), slight sub-additivity
  only at shared peaks. Reading: attention moves the information in a few discrete steps (Qwen3 L40/L43, Mixtral
  L15/L18/L19), MoE of the same and following layers transforms it. Verified vs transformers hooks on OLMoE.
- **Direction 2 (model zoo; runs `results/{qwen3_instruct,qwen3_coder,mixtral_instruct,olmoe,olmoe_instruct}_
  {default,nobos,chat}`, usage specs in `data/model_usage/`)**: all five new checkpoints show **pattern A** (one
  layer, one positive specific expert) under their intended protocol: Qwen3-Instruct-2507 and Qwen3-Coder keep
  **L44E069** (and L42E115 as the second locus) under raw and chat protocols with rescue/Spec at or above the base;
  Mixtral-Instruct with BOS = base (L19E002, 86% identical L19 routing); OLMoE base L13E056, OLMoE-Instruct
  L12E040/L13E056. **Pattern B** (selected expert not specific) occurs only for Mixtral base and Instruct under the
  paper's no-BOS protocol (identical 91/83 E006 activity in both; joint winner L18E001); pattern C never. Post-training
  moves neither layer nor expert. Chat wrapping raises margins and absolute rescues but not the drop-normalised layer
  share; one exception: Mixtral-Instruct chat's argmax jumps to the last layer L31 (+1.79) while the L19-L21 band
  stays. Sink-carrying final tokens: 24% of prompts in both Mixtrals without BOS, 0% in every other run.
- **Direction 4 (CodeFact; `data/codefact/items.jsonl`, 6,795 Python next-token counterfactuals in six categories from
  HumanEval, MBPP and CodeSearchNet; runs `results/codefact_{qwen3_raw,mixtral_nobos,qwen3_coder_raw,qwen3_coder_chat}`)**:
  pass rates under the paper's thresholds separate categories by how the answer is determined: S1 closing bracket 90%
  (read from the opener like a fact, drop +4.4), R1 variable recall 79%, R2/R3 57-59%, S2/S3 keywords 33-36% (redundantly
  determined, median drop +0.25/+0.12). On code the last MoE block acts as a read-out (L47/L31 selected almost
  everywhere), so an interior-layer variant (≤ L−5) is reported alongside. Localised single experts: S1 (Qwen3
  L47E025 Spec +0.97, interior L42E048 +0.87; Mixtral L31E000 +1.44), R1 in Qwen3 (L43E126 +0.32, CounterFact-like),
  S2 (Qwen3 L41E041 +0.55, Mixtral L17E003 +0.32); S3/R2/R3 weak. Categories are near-disjoint in their top experts
  (mean Jaccard 0.05); the factual experts L44E069/L42E115 rescue nothing on code; Mixtral E006 is negatively specific
  again (R1). Qwen3-Coder keeps the same code experts as the base (L47E025, L43E126, L41E041) and adds an S3 expert
  L47E014. Code prompts are 36-84 tokens median (CounterFact: 8) with a single-token "subject"; the axis that matters
  is single-token vs distributed determination, not syntax vs recall.
- **Phase 2 wave 1 (2026-09-21, ≈ 45 GPU min; plan in RESEARCH_PLAN.md "Phase 2 execution plan")**:
  - *F4 subject-token patching* (`moetrace/ext5_subject.py`, suffix wavefront rows from the last subject token p;
    runs `results/{qwen3_bos,mixtral_nobos,mixtral_bos}_subject`; verified vs transformers hooks on OLMoE, bit-identical
    to `Engine.run` at p = T−1): the MoE-output patch at p peaks in the **first ten layers** — Qwen3 **L4 +0.93**
    [0.65, 1.25] (= the final-token L44 peak +0.93), Mixtral no-BOS L6 +1.20, Mixtral BOS **L4 +2.31** (4× the L19
    peak); ≈ 0 from L9 on and at L42/L44/L18/L19. **No shared early-site expert**: recurrent candidates rescue
    ≤ +0.07 with Spec ≈ 0, the clean top-k coalition recovers the layer effect, and the per-case best expert (88–99% of
    the coalition) differs across prompts (36–38 distinct winners/128 in Qwen3). L44E069/L42E115/L19E002 patched at p
    rescue nothing. Noise on the last subject token alone = 0.47–0.52 of the whole-span drop. Reading: expert-level
    localisation is a property of the shared late read-out, not of the early subject-enrichment site.
  - *F1 rankings / minimal sets* (`moetrace/ext5_rank.py`, existing `*_alllayers` passes + exhaustive subset passes
    `results/{qwen3,mixtral_nobos}_subsets`, engine kind `coalition_set`): rankings by all-case rescue, active-only
    rescue, Spec and the discovery statistic agree wherever the effect is clear (τ 0.94–0.97 rescue vs active-only;
    L44E069 rank 1/1/1, L42E115 2/2/2, L19E002 1/1/1); disagreements are systematic: "junior partners" (rescue > 0,
    Spec < 0: Mixtral L19E006, Qwen3 L43E046) and Spec-without-rescue in layers whose block hurts. Minimal sets for
    50/80/90 % of the block rescue: Qwen3 L44 1/6/9 experts (E069 alone 53 %), L42 1/2/6 (E115 72 %); Mixtral BOS
    L19 1/3/4 (E002 63 %), L18 1/2/2 (E001 77 %). Exact subsets: one expert reaches 80 % of a case's own block in
    59–60 % of Qwen3 cases (≤ 2 in 87–88 %); pairwise interactions ≈ 0 (bf16 ulp), redundancy only between strong
    pairs; the additive approximation predicts the exact minimal size in 91–97 % of cases. Cross-layer sums over-count
    (F1.4 open).
  - *F2 attention heads* (engine kind `attn_head`, runs `results/{qwen3,mixtral_nobos,mixtral_bos}_heads`): the
    attention rescue is carried by a few **mover heads reading the last subject token**: Qwen3 L40 **head 13 +0.95**
    [0.81, 1.09] (60 % of +1.58, Spec +0.93; two heads for 80 %), L43 distributed (4 heads), L44 null; Mixtral L18
    **head 4** +0.61/+0.79 (no BOS/BOS, 76–80 %), L24 head 22 (82–86 %), L19 heads 29–31 (one GQA group), L15 heads
    1/3; identical heads under both protocols. Mover heads put 0.4–0.5 of clean attention on the last subject token,
    noise halves it (→ relation tokens without BOS, → the position-0 sink with BOS). Σ heads = attention rescue on the
    mean; per-case r 0.3–0.7, so minimal head sets are additive estimates.
  - *F5 probability metrics* (engine `--metrics`: `logp_true, logp_foil, p_true, p_foil, rank_true, kl_to_clean`;
    runs `results/{qwen3,mixtral_nobos,mixtral_bos}_metrics`): Δ = log p(true) − log p(foil) exactly (log-odds; checked
    to one bf16 ulp). Qwen3 and Mixtral-BOS selections are metric-independent (L44E069, L19E002 under all six).
    Mixtral no-BOS: Δp and rank select **L18E001** (Spec > 0) instead of L19E006 (Spec < 0 under every metric); Δlog p
    and KL pick L0, a heavy-tail artefact of the 26 most-disrupted (sink-state) cases. Dataset descriptor: the true
    object is the clean top-1 in only 29–32 % of paper cases (median clean p(true) 0.02–0.04). Recommendation adopted
    in the section: keep Δ primary, report normalised rescue, rank recovery and clean top-1 rate alongside; Δp
    descriptive only; KL/Δlog p as protocol diagnostics.
  - Incident: `run_sweep.py` crashed after a complete pass when writing `run_meta.json` (variable `meta` reused for a
    DataFrame); fixed with JSON sanitising in `run_sweep.py`/`run_expert.py`; no GPU work lost.
- **Direction 6 (2026-09-28, ≈ 58 GPU min; `moetrace/ext6_str.py`, `scripts/ext6_str_*.py`, runs
  `results/{qwen3_str,mixtral_nobos_str,mixtral_bos_str}`)**: best-practices check against Zhang & Nanda (STR over GN,
  normalised logit difference, single-layer patching, several corruption sites; the paper follows all but STR and
  normalisation). STR donor = another CounterFact subject of the same relation whose true object is the case's foil,
  same template and token positions, known by the model (logit(foil) − logit(true) ≥ 1 on the donor prompt); up to 5
  donors per case in `random.Random(2000 + case_id)` order, donor mean primary, first donor as sensitivity; paper IDs and
  split restricted to 215 / 212 / 213 cases (Qwen3 / Mixtral no BOS / BOS), recurrence = half the retained discovery.
  The paper's Δ becomes LD(r, r′). Verified vs transformers hooks on OLMoE (layer r 0.994, expert r 0.992). Results:
  **every selection of the paper survives** — Qwen3 L44 / L44E069 (Spec +0.95, per unit drop 0.082 vs GN 0.075; now
  also the all-layer joint top-1, where GN had L42E115/E069 tied), Mixtral no BOS L19 / L19E006 (Spec −0.29, coalition
  +0.80 ≈ block +0.82), L18E001 positive. STR drop ≈ 2× GN (Δ_corrupt ≈ −5 to −6.4); drop-normalised block rescue
  0.177 / 0.077 / 0.085 vs GN 0.167 / 0.089 / 0.113 (no GN inflation in Qwen3, 13–25 % in Mixtral); curves r 0.94–0.99.
  Changes: Mixtral BOS L19–L21 tie now resolves to L21E001 (two-stage and joint; first donor still L19E002); at equal
  norm Mixtral L19 experts (E006 and E002) no longer differ from their co-active partner under STR (GN: E002 +0.07–0.11).
  Mixtral no-BOS joint top-1 L21E001 on the reduced set only (59/128 < 64 on the full set).
- **Direction 6b (2026-09-28, ≈ 40 GPU min; `scripts/ext6_str_grid*.py`, rows `results/{qwen3,mixtral_bos}_str/str_grid_w{1,5}_*`)**:
  Zhang & Nanda Section 4.1 / Figure 4 for the paper's MoE-output patch under STR: every position from the first subject
  token × every layer, all donors, single layer and 5-layer centred window, Δ/drop and Δp(true). Executor additions in
  `ext5_subject.py` (`SubjectSpawn.window`, `run_subject(metrics=True)`), F4 verification unchanged (70/70 fields), new
  OLMoE check `results/verify_ext6_str_grid_olmoe.json` (r 0.991 / 0.998). Two sites: last subject token at L0–L8 (peak
  L0: Qwen3 +0.187, Mixtral BOS +0.304 of the drop) and the final position at the paper's band (Qwen3 L44 +0.176,
  Mixtral L19–L21 +0.086); positions in between ≤ 0.007. Last/middle subject-token ratio agrees across metrics with
  single layers (LD 2.9× / 4.8×, p 2.7× / 6.5×); with window 5 probability over-weights it (LD 3.2× / 5.0×, p 4.0× /
  14.9×) and windows are additive or sub-additive in LD (0.60–1.09) but super-additive in Δp (2.8–6.1): Zhang & Nanda's
  GPT-2 XL effects come from the softmax once windows are used. GN vs STR at the last subject token (F4 runs, same
  cases, normalised): GN peaks at L4 not L0; layer sum 1.06× STR (Qwen3), 1.77× (Mixtral BOS) = GN inflation at the
  early site, not at the paper's final-position site.
- **Phase 3 (2026-10-04, ≈ 3.9 GPU-h; Qwen3 and Mixtral BOS; STR only)**:
  - *WinoGrande as STR pairs* (`moetrace/ext7_wino.py`, `data/wino_str/`): fill the blank with each twin's answer and predict the
    sentence-final single-token trigger ("… but the bag was too" → " small" / "… the body was too" → " large"); rules W1–W6
    (trigger = only and last word, single token, token symmetry, option not final, dedup) + margin ≥ 1 both ways: Qwen3 2,099,
    Mixtral BOS 1,151 pairs; case set = 776 pairs shared by Qwen3 / Mixtral BOS / no BOS → 128/128 + 128/128 replication
    (`data/wino_str/case_sets.json`). GN on the option token is ineffective in Qwen3 (binary in-context choice survives noise).
  - *Attention vs MoE* (W2 single layers, all-MoE patch, direct-path split): IOI ≫ CounterFact > WinoGrande (name items >
    object items). Attention share of the positive rescue IOI 0.92 / 0.80, CounterFact 0.54 / 0.58, WinoGrande role 0.32 / 0.39,
    option 0.16 / 0.34 (Qwen3 / Mixtral; role swap = name items only — on identical items role = option swap, Direction 12 4b); direct-path MoE share WinoGrande 0.95 / 0.71. WinoGrande is NOT IOI-like at the final
    position; one late specific expert per model (Qwen3 **L41E117** Spec +0.90, Mixtral **L20E000** Spec +0.94; pattern A,
    replicated; not the CounterFact experts); attention moves the option identity gradually (Qwen3 L19–L41, heads of one KV group
    cancel within a layer) or in one step (Mixtral L13, coreference-like heads). IOI heads recovered (S2 reader Qwen3 L42H11,
    name-mover-like L42H10 / Mixtral L19H8, negative movers).
  - *Joint all-attention patch at a shared final token is degenerate* (A = 1: the MoE is per-token), so W4 = all-MoE M +
    direct-path split; for symmetric pairs noising(d) = denoising(1 − d).
  - *Add-back curves* (`moetrace/ext8_addback.py`, engine `multi` with `attn_layer` / `block` steps and noising direction):
    all-MoE ceiling CounterFact 0.53 / 0.41 of the drop (answer restored in 44 % / 37 %), WinoGrande 0.84 / 0.79 (95 % / 89 %);
    80 % of the ceiling with 4–6 experts (CounterFact) / 7–10 (WinoGrande) by adaptive greedy (random 48–320); greedy ≈ optimal
    (beam, exact top-10 optimum ≤ 0.016 better); subsets overshoot the all-MoE ceiling; patch-free DLA ranking ≈ oracle; the
    paper's layer-first order is below every effect-based ranking (oracle/DLA/pop) but above routing weight / random
    (k ≤ 15 AUC incl. greedy: `results/tables/ext8_a1_partial_auc_k15.md`). Gradient rankings need F3 (not built).
  - Fixed: `ext5_subject.py` turned `attn_layer` rows into `block` when a pass also had window > 1 rows (no result affected;
    regression identical).
- **Phase 4 (2026-10-05, ≈ 4.4 GPU-h; Qwen3 and Mixtral BOS; STR; agents ext9-knockout, ext10-circuit, ext11-writer, ext12-complete)**:
  - *Engine E4* (ext9, additive; verified `results/verify_ext9_engine_olmoe.json`, regression identical): `PrefillSpec.route_mask`
    ((layer, expert), ...) with `route_mask_pos` all|final and `route_mask_mode` reroute (router logit −inf before softmax) | zero;
    `multi` steps `(l, "attn_head", heads)` and `(l, "heads_experts", (heads, experts))`; `DiagSpec.contrib_final_vectors`.
    API: `logs/ext9_engine_api.md`.
  - *Knockout (Direction 9)*: selected experts are necessary and task-specific but small: F (fraction of clean margin lost)
    Qwen3 L41E117 0.033 on WinoGrande / −0.001 on CounterFact, L44E069 0.033 / 0.001, L42E115 0.024; Mixtral L20E000 0.036,
    L18E001 0.015, L19E002 / L21E001 0.004 (barely necessary); rank 1 in their layer; dissociation contrast +0.065 / +0.041;
    population top-10 sets 0.14–0.15 (Mixtral CounterFact top-10 0.038, contains WinoGrande / BOS-sink expert L19E006);
    random matched sets ≈ 0; zero ≈ reroute; final-position-only knockout = 0.91–0.96 of all-position. Self-repair
    (`scripts/ext9_synthesis_selfrepair.py`): a knockout loses only 0.30–0.47 (Qwen3, Mixtral L20E000) / 0.05–0.19 (Mixtral
    L21E001, L19E002) of the expert's own direct write on the same prompts; the indirect expert L18E001 is not compensated.
  - *Head + expert add-back (Direction 10)*: greedy over heads ∪ experts restores r(20) 0.97 / 1.00 (CounterFact) and 0.93 / 0.94
    (WinoGrande) of the drop (experts alone ≤ the all-MoE ceiling 0.41–0.84); 80 % with 8–10 components; heads among the first
    10 picks CounterFact 6.2 / 6.4, WinoGrande 3.6 / 2.8, IOI (Qwen3) 9.9; CounterFact picks mover heads first (Qwen3 L40H13,
    Mixtral L18H4 / L24H22), WinoGrande its experts first. Head DLA ≈ oracle on CounterFact, poor on WinoGrande (early heads act
    indirectly).
  - *Writers (Direction 11)*: direct (DLA) share of single-expert patch effects: Qwen3 L41E117 1.04, L42E115 0.96, L44E069 1.32
    (later layers undo 0.39 of its 1.59-logit write); Mixtral 0.54–0.87; share rises with depth (first-half experts ≤ 0.03 of
    the drop, indirect). Routing follows the local slot (L41E117: copulas / degree adverbs, context-free "The bag was too"
    prompts), the write needs the full context (DLA +0.54 vs +0.08); CounterFact experts route broadly ("a name follows",
    every IOI final token) but write relation-specifically. Vocabulary: L41E117 δ_e puts the right trigger at median rank 98;
    Mixtral L20E000 c_e at 10. Frozen-norm DLA error ≤ 1 %.
  - *Completeness (Direction 12)*: add-back replicates on the WinoGrande replication split and a CounterFact fold swap (all
    differences' CIs contain 0); Mixtral without BOS re-selects L20E000 (Spec +0.95) with unchanged attention/MoE balance;
    **revised Phase-3 claim**: on identical name items role swap = option swap (attention share Qwen3 0.304 vs 0.306); the
    Phase-3 role > option gap was names vs objects (role-swapped prompt = option-swapped prompt with names exchanged).
- Open method questions raised by the agents, **not yet decided by the user**: (a) how strongly to state that the
  paper's Mixtral expert claim is a search-scope artefact (L18E001); (b) last-layer read-out vs localisation
  (interior-layer rule for code / chat?); (c) select layers by block (attention + MoE) rescue rather than MoE rescue?;
  (d) flag sink-carrying prompts and the object-token rule as protocol checks; (e) recurrence threshold relative to top-k.

## 3. What is NOT done / known gaps

Base reproduction:
- Mixtral **relaxed-filter set (Tables 14, 15)** was run only under the BOS default. A no-BOS filter scan → sweep →
  expert pass (about 4 passes, 7 min GPU) would complete it: `run_filter.py mixtral --out mixtral_nobos_relaxed
  --no-special-tokens`, then `run_sweep.py` / `run_expert.py` with the same flags.
- The paper's relaxed 512-case set used a different record order (only 15 overlaps with its strict set); its IDs are
  unpublished, so Table 15 subset sizes cannot match.
- Zero-row definition (Table 3), fold assignment (Table 12), split function (Appendix D), active-random draws and the
  exact noise samples are unrecoverable; only selections and CI-level agreement are comparable.

Extensions:
- Phase 3 not done (see ext7_synthesis.md caveats): gradient rankings for add-back (needs F3); head patches at positions
  other than the final one. Done in Phase 4: Mixtral no BOS on WinoGrande, add-back replication, role vs option on identical
  items. Phase-3 results committed 2026-10-05 (≈ 300 MB).
- Phase 4 not done (ext9_synthesis.md caveats): path patching (split "indirect" into later attention vs MoE); Mixtral IOI circuit
  run (≈ 25 GPU-min); a route mask that spares position 0 (BOS-sink confound for Mixtral sets with L19E006); a second WinoGrande
  corruption site on object items; two-site add-back (subject + final position). Phase-4 results committed 2026-10-06.
- WinoGrande original direction (property → entity) not run: twins whose only difference precedes the blank, truncated at the
  blank, predicting the option (1,912 twins; single-token options + equal length: Qwen3 1,162, Mixtral 578 before margin).
- **Phase 2 wave 2 NOT started**: F3 gradient / attribution patching (reverse layer streaming), F1.4 multi-layer
  minimal sets (engine kind `multi` exists and is verified), optional F2 on Qwen3-Instruct, F4 full position × layer
  grid (the executor supports any p per row). Wave-1 loose ends: exact `attn_head_set` kind (minimal head sets are
  additive estimates); MoE-side cross-check of mover heads (patch L40 heads, read E069 routing at L44) needs wavefront
  routing; consolidate `ext5_subject._build_v_at` into `engine.py`; join ext3 per-case sink flags to the F5 L0 KL tail;
  early-site expert statistic stratified by subject token / relation instead of recurrence.
- The five open method questions in 2b (user decisions pending); the final wording of EXTENSIONS_REPORT.md follows them.
- Direction 6 (STR) not done: Qwen3 gate-matched / equal-norm control (needs `--pairs` at L44), relaxed and own strict
  sets, the zoo models, CodeFact, attention/head and residual (`resid`) patches under STR, the reverse (noising) direction.
  Direction 6b: Mixtral no BOS grid (user decision pending; `python scripts/ext6_str_grid.py mixtral --out mixtral_nobos_str
  --no-special-tokens --window 1|5`, ≈ 11 min each), expert-level analysis at the early site under STR.
- CodeFact: Mixtral S3 is partial (241 passing items → 120/121 split); The Stack was not used (gated; CodeSearchNet
  instead); R3 has a single-digit sub-category that may deserve exclusion; no HF-hook verification of the code runs
  beyond the shared engine.
- Model zoo: Qwen3-Coder has no public Base checkpoint (Instruct only); OLMoE-Instruct's strict/relaxed sets pick
  L13E056 vs L12E040 (no shared case set); Mixtral-Instruct chat's last-layer argmax is unresolved (question b).
- Direction 2b: no per-head decomposition of the attention patches (natural follow-up: mover heads at Qwen3 L40/L43,
  Mixtral L15/L18/L19/L24).
- Raw diagnostics (`/opt/dlami/nvme/moe_ext2`, `moe_ext3`, 3 GB) and all checkpoints sit on the ephemeral NVMe.

## 4. Machine and environment (verify before relying on it)

- AWS g5.4xlarge: 1× A10G 24 GB, 16 vCPU, 62 GB RAM. An unrelated user process (`repo-world-model`) holds ~9 GB RAM;
  do not kill processes you did not start.
- Python venv: `. /home/ubuntu/MOE/.venv/bin/activate` (Python 3.14, torch 2.14+cu130, transformers 5.16.1,
  safetensors, pandas, pyarrow, scipy, matplotlib, datasets). `~/.local/bin/claude` and `~/.local/bin/gh` exist.
- **Always `export HF_HOME=/opt/dlami/nvme/hf`** before any script (`scripts/gpu_queue.sh` does it for you).
  Seven checkpoints live there (≈ 407 GiB): Qwen3-30B-A3B-Base 61 G, Qwen3-30B-A3B-Instruct-2507 57 G,
  Qwen3-Coder-30B-A3B-Instruct 57 G, Mixtral-8x7B-v0.1 93 G, Mixtral-8x7B-Instruct-v0.1 87 G, OLMoE-1B-7B-0125 39 G
  (fp32 shards), OLMoE-1B-7B-0125-Instruct 13 G. NVMe free ≈ 49 GB; `/opt/dlami/nvme/offload` (65 GB, transformers
  offload scratch from the reference checks) is reclaimable. The NVMe is **ephemeral**: an instance stop wipes it;
  re-download with `scripts/download_models.sh` (base three) and `scripts/ext2_zoo_download.sh` (zoo four).
- Root disk holds code and results: `results/` is 380 MB on disk, ~110 MB tracked (gitignored: `codefact_*/
  scan_routing.parquet`, `codefact_*/expert_parts/`, `codefact_smoke/`); `.git` is 240 MB. Never write weights or
  offload folders to the root disk.
- Before launching GPU work: `pgrep -af "run_sweep|run_expert|run_filter|hf_reference|chain|ext[1-4]_|gpu_queue"`
  and `nvidia-smi`. Submit every GPU job through `bash scripts/gpu_queue.sh <job-name> -- python ...` (flock lock,
  `logs/gpu_queue.log`), even when you believe you are alone.

## 5. Codebase map and how to use it

Engine idea: one decoder layer resident on the GPU at a time (streamed from safetensors with prefetch); all runs of a
pass advance layer by layer. *Prefill rows* = full clean/noised prompts (record final-position MoE in/out, routing,
per-expert contributions c_e = w_e·E_e(x), optionally attention outputs and diagnostics). *Wavefront rows* = single
tokens implementing interventions from layer l upward, attending to the parent noised run's K/V. One pass = one read
of the checkpoint (Qwen3 ~60 s, Mixtral ~95 s, OLMoE ~4 s) regardless of how many interventions are batched; the only
limit is wavefront rows × prompt length (about 45k rows at T ≈ 15, far fewer at code lengths T ≈ 40-140: chunk by
layers and cases, see `run_expert.py --layer-chunks` and `scripts/ext4_scan.py`).

`moetrace/` (base)
- `arch.py` ArchSpec from config.json (Qwen3-MoE, Mixtral, OLMoE key templates, q/k-norm variant, top-k renorm).
- `weights.py` CheckpointStore + LayerStreamer (mmap safetensors, per-layer expert stacking, pinned double buffer).
- `engine.py` `Engine(repo)`; `PrefillSpec(ids, true_id, foil_id, noise_pos=None, noise_eps=None, pos_offset=0,
  sink_donor=-1, sink_vscale=1.0)`; `SpawnSpec(layer, parent, clean, kind, expert=-1, partner=-1)`; `DiagSpec(...)`
  (per-layer attention mass, residual norms, router logits, residuals, token log-probs, all-token routing, attention
  outputs); kinds = `zero`, `layer` (MoE output), `expert`, `expert_scaled`, `coalition_clean`, `coalition_union`,
  `attn_layer`, `block` (attention + MoE), `resid` (whole residual after the layer), `block_diff` (numerics check).
  `eng.run(prefill, spawns, record_routing=True, diag=None)` → `PassResult` (delta, logits, sp_delta, sp_vnorm,
  route_idx/route_w/route_cnorm [L, B, k], extra["diag"]).
- `data.py` CounterFact loading, seed-0 shuffle, `prepare_case(rec, tok, token_rule="space"|"paper_like",
  special_tokens=True|False, prefix_ids=None)`, filters (STRICT 1.0/0.5, RELAXED 0.5/0.25), splits.
- `noise.py` σ = mult × embed std; per-case `torch.Generator(0 + case_id)`.
- `protocol.py` `cases_by_id(model, ids, token_rule, special_tokens, prefix_ids)`, `load_case_sets`,
  `active_controls` (random.Random(1000 + case_id)).
- `stats.py` `summarize` (mean, 5,000-resample percentile bootstrap CI, positive fraction, 10,000 sign-flip p), `fmt`.
- `analysis.py` post-processing on a run dir: `load_model(run)`, `layer_analysis`, `expert_table`, `select_expert`
  (recurrence-first), `evaluate_expert`, `gate_matched_control`, `all_active_rank`, `active_pair_equal_norm`,
  `coalitions`, `stability_grid`, `relation_heldout`, `noise_table`, `funnel_check`.
- `report.py` `build()` → Tables 1–16 (md+csv) + `figure1()`; `alt_summary(suffix=...)` for secondary runs;
  `write_report.py` `write(...)` → `results/REPORT.md`.
- `models.py` `MODELS` registry: `qwen3`, `mixtral`, `olmoe`, `olmoe_instruct`, `qwen3_instruct`, `qwen3_coder`,
  `mixtral_instruct` → repo, label, n_controls (3 for top-8, 1 for top-2), paper_layer, paper_expert.

`moetrace/` (extensions, one module per direction, none edits the base modules)
- `ext1_analysis.py` per-layer best expert, joint (layer, expert) search, concentration, grid robustness.
- `ext2_attn.py` attention / MoE / block curves, peaks, shares, additivity.
- `ext2_zoo.py` chat-template prefixes, usage specs, cross-model summaries, sink fractions.
- `ext3_variants.py` position-0 substitutions, sink transplant, corpus routing helpers.
- `ext4_data.py` CodeFact item → `data.Case` (categories S1-S3, R1-R3; single-token continuation with boundary back-off).
- Phase 2 (ext5): `ext5_subject.py` (`SubjectEngine.run_subject`, `SubjectPrefill(rec_pos)`, `SubjectSpawn(pos)`: suffix
  wavefront rows patched at position p; copies engine's `_build_v` as `_build_v_at`), `ext5_rank.py` (full (layer,
  expert) rankings, Kendall τ, greedy/coverage minimal sets, exact-subset analysis), `ext5_metrics.py` (F5 metric
  columns, rescue swap so `analysis.py`/`ext1_analysis.py` run under Δp/Δlog p/rank/KL), `ext5_heads.py` +
  `ext5_heads_section.py` (per-head rankings, Spec, additivity, attention-mass classes, section writer).
  Engine additions by ext5-engine (in `engine.py`, backward compatible): `Engine.run(..., metrics=True)` →
  `PassResult.metrics_prefill/.metrics_spawn`; `PrefillSpec.clean_ref`; kinds `attn_head` (`SpawnSpec.expert` = head),
  `coalition_set` (`SpawnSpec.experts`), `multi` (`SpawnSpec.steps`); `DiagSpec.attn_heads_final`, `spawn_vectors`;
  `run_sweep.py`/`run_expert.py` flags `--metrics`, `--agent`. Verification: `results/verify_ext5_engine_olmoe.json`,
  `results/verify_ext5_subject_olmoe.json`; `verify_olmoe.json` unchanged vs `verify_olmoe_before_ext5.json`.
- Phase 3: `ext7_wino.py` (WinoGrande twins → STR pairs, rules W1–W6), `ext7_pairs.py` (generic STR-pair runner: directed cases
  2·pair_idx + d, analysis adapter), `ext7_controls.py` (role swap, IOI builder, three-task table, `direct_split_pairs`),
  `ext8_addback.py` (add-back orderings, greedy / beam / exact, curves). Engine (ext8, additive): `multi` steps `attn_layer` /
  `block`, every kind in the noising direction (parent = clean row, source = corrupted row), `SpawnSpec.kl_ref`,
  `DiagSpec.contrib_dla`; verified `results/verify_ext8_engine_olmoe.json`.
- Phase 4: `ext9_knockout.py` (items, conditions, controls, planner), `ext10_circuit.py` (head singles, mixed greedy / static
  orderings), `ext11_writer.py` (direct / indirect split, routing contexts, vocabulary projections), `ext12_complete.py` (task
  registry for the multi-task add-back driver). Engine E4 (ext9, additive): `PrefillSpec.route_mask` / `route_mask_pos` /
  `route_mask_mode`, `multi` steps `attn_head` / `heads_experts`, `DiagSpec.contrib_final_vectors` (`logs/ext9_engine_api.md`).
- Direction 6: `ext6_str.py` (STR donors: `donor_index`, `candidates` = same relation, true object == case foil, same template,
  identical token positions and continuation ids; `select` (margin 1.0, K = 5, `random.Random(2000 + case_id)`);
  `model_data(run, donors='mean'|'first')` → `analysis.ModelData` with donor-aggregated rows so every `analysis.py`
  function runs unchanged; `normalised` = mean rescue / mean drop). No engine change: the corrupted run is a plain
  prefill row, spawns use parent = donor row.

`scripts/` — base CLIs take a model key; `--out <run>` selects `results/<run>/`; `--token-rule space|paper_like`;
`--no-special-tokens` = no BOS; `--layer-chunks N` bounds wavefront rows.
- Base: `run_filter.py`, `run_sweep.py [--sets paper,strict,relaxed]`, `run_expert.py --layers ... [--no-pairs]`,
  `auto_expert.sh`, `run_noise.py`, `hf_reference_check.py`, `verify_olmoe.py`, `verify_olmoe_fp32.py`,
  `mixtral_compare.py`, `mixtral_run_section.py`, `build_final_report.py [--done]`, `chain1/2/3.sh`.
- Infrastructure: `gpu_queue.sh` (mandatory lock for GPU jobs), `build_extensions_report.py` (assembles
  `results/sections/ext*.md` into `results/EXTENSIONS_REPORT.md`), `run_agent.sh` (headless supervisor, section 7),
  `download_models.sh`, `ext2_zoo_download.sh`, `setup_env.sh`.
- Direction 1: `ext1_analyze.py` (reads `results/<run>_alllayers`), `ext1_chain*.sh`.
- Direction 2b: `ext2_attn_sweep.py <model> --out <run> --sets paper [--no-special-tokens]`, `ext2_attn_verify.py`,
  `ext2_attn_gate.py`, `ext2_attn_analyze.py`, `ext2_attn_chain.sh`.
- Direction 2: `ext2_zoo_usage.py` (writes `data/model_usage/<key>.json`), `ext2_zoo_filter.py <key> --protocol
  default|nobos|chat`, `ext2_zoo_sweep.py`, `ext2_zoo_expert.py [--resume]`, `ext2_zoo_attn.py`, `ext2_zoo_analyze.py`,
  `ext2_zoo_chain.sh <key>`.
- Direction 3: `ext3_run_variants.py <model> --variants bos,nobos,nl,dot,...,sinkfull,sinkkey`, `ext3_corpus_routing.py
  <model> --corpus wiki|code`, `ext3_verify_diag.py`, `ext3_gate.py`, `ext3_analyze.py`, `ext3_chain*.sh`.
- Direction 4: `ext4_build_codefact.py` → `data/codefact/items.jsonl`; `ext4_scan.py <model> --out <run> --protocol
  raw|nobos|chat` (calibration scan + layer patches, token-budget chunking); `ext4_select.py`; `ext4_run_expert.py
  --no-pairs`; `ext4_analyze.py`; `ext4_chain*.sh`.
- Phase 2: `ext5_subject_{verify,gate,sweep,select,expert,analyze}.py`, `ext5_subject_chain.sh` (F4);
  `ext5_engine_verify.py`, `ext5_engine_chain.sh`, `ext5_heads_sweep.py <model> --out <run> --layers ... --base-run
  <run>`, `ext5_heads_analyze.py`, `ext5_subsets_run.py <model> --out <run> --layers ... --base-run <metrics-run>`
  (F2, F1.3, F5 runs); `ext5_rank_analyze.py`, `ext5_rank_subsets.py`, `ext5_rank_text.py`, `ext5_metrics_analyze.py`,
  `ext5_metrics_text.py` (F1, F5 analysis; CPU only).
- Direction 6: `ext6_str_filter.py <model> --out <run> [--no-special-tokens]` (all symmetric candidates → Δ_donor →
  selection, case_sets.json), `ext6_str_sweep.py` (layer patch per donor row, routing), `ext6_str_expert.py --layers ...
  [--pairs] [--max-spawn N]`, `ext6_str_verify.py` (OLMoE vs transformers hooks → `results/verify_ext6_str_olmoe.json`),
  `ext6_str_analyze.py` (STR vs GN on the same cases; tables, figure, `results/ext6_str_summary.json`), `ext6_str_text.py`
  (section), `ext6_str_chain.sh`.
- Phase 3: `ext7_wino_{build,scan,funnel,casesets}.py` (pairs, competence scan, funnel, case set); generic runners
  `ext7_wino_{sweep,expert,grid,heads,joint,dla}.py --model --pairs --case-sets --out` (any STR pair parquet: pair_id, ids_a,
  ids_b, str_pos, trig_a, trig_b), `ext7_wino_{verify,analyze,text}.py`; `ext7_{cf,role,ioi}_*.py` (CounterFact STR attention
  sweep, role swap, IOI), `python -m moetrace.ext7_controls direct|section`; `ext8_{engine_verify,regress_compare,addback_run,
  addback_analyze,addback_text}.py`.
- Phase 4: `ext9_{engine_verify,merge_regress,regress_compare,knockout_run,knockout_analyze,knockout_text}.py`, `ext9_chain.sh`,
  `ext9_synthesis_selfrepair.py`; `ext10_{heads_run,heads_analyze,circuit_run,circuit_analyze,circuit_text}.py`;
  `ext11_writer_{partA,routing,routing_analyze,vocab,vocab_analyze,text}.py`; `ext12_{addback_run (multi-task ext8 driver),
  roleitems_casesets,sink_flags,analyze,complete_text}.py`; `ext8_partial_auc.py` (k ≤ 15 AUC incl. greedy);
  `ext8_plot_simple.py` (add-back figure with greedy / oracle / random only, deletion figure with oracle / random:
  `results/figures/ext8_a{1,3}_curves_simple.png`); `ext7_wg_cf_routing_overlap.py` (final-position routing overlap
  WinoGrande vs CounterFact → `results/tables/ext7_wg_cf_routing_overlap.md`). Team one-pager: `docs/team_brief_winogrande_str.md`.
- Direction 6b: `ext6_str_grid.py <model> --out <str run> [--window 1|5] [--no-special-tokens]` (layer × position grid,
  units packed by suffix length), `ext6_str_grid_verify.py`, `ext6_str_grid_analyze.py` (heatmaps, peaks, last/middle
  ratio, sliding vs adding, GN comparison with the F4 `*_subject` runs), `ext6_str_grid_text.py`, `ext6_str_grid_chain.sh`.

`results/` run directories (each has `run_meta.json`; row-level Parquet: `sweep_rows`, `sweep_routing`,
`sweep_cases`, `expert_rows`, `expert_prefill_*`; extension runs add `zoo_summary.json`, `sink_diag.json`, ...):
- base: `qwen3`, `mixtral` (tokenizer defaults, three case sets), `mixtral_nobos` (paper set, no BOS), `qwen3_alt`,
  `mixtral_alt` (paper_like token rule), `olmoe` (pilot);
- Direction 1: `{qwen3_bos,mixtral_bos,mixtral_nobos}_alllayers`;
- Direction 3: `mixtral_{bos,nobos}_diag`, `mixtral_nobos_prefix_<tok>`, `mixtral_nobos_shift1`,
  `mixtral_nobos_sink_{full,keyonly}`, `mixtral_bos_prefix_bos`, `mixtral_{bos,nobos}_corpus_{wiki,code}`,
  `qwen3_nobos_diag`, `qwen3_bos_prefix_eot`;
- Direction 2b: `{qwen3_bos,mixtral_bos,mixtral_nobos,olmoe}_attnsweep`;
- Direction 2: `<key>_{default,nobos,chat}` for the five zoo models (+ `_attnsweep`), `olmoe_default`;
- Direction 4: `codefact_{qwen3_raw,mixtral_nobos,qwen3_coder_raw,qwen3_coder_chat}`;
- Phase 2: `{qwen3_bos,mixtral_nobos,mixtral_bos}_subject` (F4; `sweep_rows` has `pos`, prefill kinds clean/noised/
  noised_lastonly/noised_exceptlast, expert kinds incl. `expert_fixed`), `{qwen3,mixtral_nobos,mixtral_bos}_metrics`
  (F5; six metric columns), `{qwen3,mixtral_nobos,mixtral_bos,olmoe}_heads` (`head_rows.parquet`, `head_attn_final.npz`),
  `{qwen3,mixtral_nobos}_subsets` (`subset_rows.parquet`, all 2^k−1 clean-active subsets + same-pass reference rows).
- Direction 6: `{qwen3,mixtral_nobos,mixtral_bos}_str` (`str_candidates.parquet` every symmetric candidate + Δ_donor +
  slot; donor-level `str_sweep_rows` (kinds clean/corrupt/layer, `slot`), `str_sweep_routing` (slot −1 = clean),
  `str_expert_rows` (all layers; Mixtral L18–21 with `expert_scaled` pairs), `sweep_cases` (delta_corrupt = donor mean));
  GN baselines = the Direction-1 `*_alllayers` runs restricted to the same cases. Direction 6b adds
  `str_grid_w{1,5}_rows.parquet` (donor level: case, pos, token group, layer, window bounds, Δ, rescue, p(true), rank) and
  `str_grid_w{1,5}_prefill.parquet` to `qwen3_str` and `mixtral_bos_str`.
- Phase 3: `wino_<qwen3|mixtral_bos|mixtral_nobos|olmoe>` (competence scans), `wino_{qwen3,mixtral_bos}_str` (W2–W6, W4, grids,
  heads), `wino_role_*`, `ioi_*`, `{qwen3,mixtral_bos}_str_attnsweep`, `*_addback` (add-back curves).
- Phase 4: `{qwen3,mixtral_bos}_knockout` (items, ko_rows_<phase>_*.parquet with condition `sig` = experts|pos|mode, controls,
  plans), `{qwen3_str,mixtral_bos_str,wino_qwen3_str,wino_mixtral_bos_str,ioi_qwen3_s2io}_circuit` (head singles, head DLA),
  `{qwen3,mixtral}_circuit` (mixed greedy / static rows), `{qwen3,mixtral_bos}_writer_{routing,vocab}`,
  `wino_*_str_addback_rep`, `*_str_addback_fold_rep`, `wino_mixtral_nobos_str[_addback]`, `wino_roleitems_*`.
Tables: `results/tables/table_01..16` (base), `ext{1,2,2_attn,2_zoo,3,4}_*`, `ext5_{subject,rank,heads,metrics}_*`, `ext6_str_*`, `ext7_*`, `ext8_*`, `ext9_*`–`ext12_*`;
figures likewise; sections in `results/sections/`; analysed numbers in `results/{summary,ext1_summary,
ext2_attn_summary,ext2_zoo_summary,ext3_numbers,ext4_summary,mixtral_compare,ext5_subject_summary,ext5_rank_summary,
ext5_subsets_summary,ext5_heads_summary,ext5_metrics_summary,ext6_str_summary,ext6_str_grid_summary,ext7_wino_summary,
ext7_controls_summary,ext8_addback_summary,ext9_knockout_summary,ext10_circuit_summary,ext11_writer_summary,
ext12_complete_summary}.json`.

Recipes:
- *Paper protocol on a new MoE model*: add a `MODELS` entry (family must be Qwen3-MoE, Mixtral or OLMoE; other
  families need `arch.py` key templates and possibly engine work), write `data/model_usage/<key>.json` with
  `ext2_zoo_usage.py`, then `ext2_zoo_filter.py` → `ext2_zoo_sweep.py` → `ext2_zoo_expert.py` → `ext2_zoo_analyze.py`.
- *New intervention*: add a kind to `KINDS` and its vector in the engine's spawn-vector builder; verify on OLMoE
  against transformers hooks as `ext2_attn_verify.py` does; re-run `verify_olmoe.py` and confirm identity with the
  previous `results/verify_olmoe*.json`.
- *New counterfactual dataset*: produce `data.Case`-compatible objects (ids, subject_pos, true_id, foil_id) as
  `ext4_data.py` does, then reuse `ext4_scan.py` / `ext4_run_expert.py` / `ext4_analyze.py`.
- *Rebuild deliverables*: `python scripts/build_final_report.py` (base) and `python scripts/build_extensions_report.py`.

## 6. Conventions and rules for work here

- The paper's Table 8 case IDs (`data/paper_case_ids.json`) are the PRIMARY case set; our own strict/relaxed sets are
  secondary. Keep the discovery/validation assignment from the paper. Instruct variants are evaluated on their base
  model's paper case set for comparability, plus their own strict set.
- Protocol labels: `default` = tokenizer defaults, raw cloze; `nobos` = no special tokens (the paper's protocol; for
  Qwen3 and OLMoE `default` ≡ `nobos`); `chat` = chat template with the cloze prompt inside the assistant turn. Keep the
  main tables on the documented default and report variants as labelled runs. For any Mixtral experiment meant to
  match the paper, use `--no-special-tokens`.
- Prefer counts and set membership (activity counts, funnel pass counts, selections) as "fingerprints" when comparing
  with the paper; means carry bf16 noise (per-case rescue ±0.1–0.6, 128-case means ±0.02–0.05).
- No quantization, ever: bf16 weights and activations are the object of study.
- Every run dir gets a `run_meta.json` (model, flags, case sets, chunking, command). Save row-level Parquet after
  every pass. Log every milestone to `logs/PROGRESS.md` (append-only, UTC timestamps).
- Parallel agents own disjoint files (base modules vs `ext*` modules, one script prefix per direction) and share the
  GPU only through `gpu_queue.sh`; never stop a running chain to edit it, append a new chain.
- Commits: trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Pushing works from this shell (the
  user's GitHub credentials are configured); commit sub-agent outputs selectively with `git add <paths>`.

## 7. Unattended / headless workflow (see PLAYBOOK.md)

Two patterns were used and both are documented in PLAYBOOK.md:
- *Headless agent* (2026-09-03): an interactive session wrote `HANDOFF.md`, the **user** started
  `scripts/run_agent.sh` in tmux (the permission classifier blocks an agent from launching another Claude process),
  the agent executed until `results/DONE`; the supervisor now parses usage-limit reset times and sleeps until then.
- *Coordinator + sub-agents* (2026-09-14): one session planned (`RESEARCH_PLAN.md`), launched one sub-agent per
  direction with a brief (goal, owned files, deliverable section, PROGRESS entries, final-report format), serialised
  the GPU with `gpu_queue.sh`, reviewed sections and committed. Sub-agents die at the account usage limit; GPU chains
  run detached and finish anyway; on reset, resume the agent with "resume from disk, do not redo GPU work".
- On any resume: read `logs/PROGRESS.md` (tail), `logs/gpu_queue.log`, `logs/chain*.log`, `logs/agent_status.txt`
  if present, then the run dirs' `run_meta.json`, to find the first unexecuted step.

## 8. Document map

- `README.md` public overview, headline table, how to reproduce, extension summary.
- `PLAN.md` feasibility analysis (why layer streaming), full implementation checklist, assumptions.
- `RESEARCH_PLAN.md` the four extension directions, experiments, agent assignment, user decisions.
- `HANDOFF.md` the operational brief the headless agent executed for the base reproduction.
- `PLAYBOOK.md` the unattended-run process, timelines, problems and fixes (both patterns).
- `results/REPORT.md` base deliverable: sections 1–8 plus 6b (Mixtral without BOS).
- `results/EXTENSIONS_REPORT.md` extension deliverable, assembled from `results/sections/ext*.md`.
- `docs/ext3_literature_review.md` attention sinks, BOS, MoE routing, tokenisation conventions (30 references).
- `data/codefact/build_stats.md`, `data/codefact/samples.md` CodeFact construction and eyeballed examples;
  `data/model_usage/*.json` per-model tokenisation / template specs.
- `logs/PROGRESS.md` timestamped log of everything that happened, including handovers and incidents.
- Paper PDF / LaTeX are gitignored (`2606.03780.pdf`, `paper.txt`, `paper_src/`); present locally on this machine.
  Also gitignored: `2309.16042.pdf` (Zhang & Nanda 2024, activation-patching best practices; basis of Direction 6).
