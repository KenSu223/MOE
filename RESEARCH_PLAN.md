# Research plan: four extensions of the expert-aware causal-tracing reproduction

Status: DRAFT for the user's decisions (2026-09-14). Execution model: one coordinating session (this one) that assigns
work to sub-agents, serialises the single GPU, reviews outputs, and asks the user at the decision points marked ❓.
Everything below builds on the existing engine (`moetrace/`), whose one-pass cost is about 60 s (Qwen3), 95 s
(Mixtral-8x7B) and a few seconds (OLMoE) regardless of how many interventions are batched. GPU time is therefore
never the bottleneck; agent implementation time and the account usage limit are.

Shared infrastructure to add first (half a day, one agent, blocks nothing else from starting):
- `scripts/gpu_queue.sh`: a lock-file job queue so several tracks can submit passes without running two on the A10G
  at once (passes are 1 to 3 min, so queueing is cheap; concurrency would split disk bandwidth and GPU memory).
- Run-directory naming: `results/<model>_<protocol>_<experiment>/` with `protocol` in {`bos`, `nobos`}; every run
  directory carries a `run_meta.json` (model, tokenisation flags, case set, sigma, engine git hash).
- `results/EXTENSIONS_REPORT.md` skeleton with one section per direction, filled by each track's section script.

---

## Direction 1 — Is "select the layer, then the expert" the right order?

**Question.** The paper picks the best layer by MoE-block rescue, then searches experts only inside that layer. A
more specific or more recurrent single expert could live in a layer whose block-level rescue is lower (e.g. Qwen3
L42/L43 at +0.62, Mixtral L20/L21 which beat L19 on validation).

**Experiment (no new engine code).**
1. Expert pass at **every** layer, not just L*: `run_expert.py --layers 0..L-1 --no-pairs` for `qwen3` (paper,
   strict and relaxed sets in one pass: 256+256+512 cases × 48 layers × ~11 rows ≈ 300k wavefront rows, chunk by
   layer range if memory requires), `mixtral` (bos) and `mixtral_nobos`. About 6 GPU passes, 10 minutes total.
2. Per layer: recurrence-first selection on discovery cases (threshold 64) → best expert, its discovery rescue,
   active-random specificity, activity counts. Produce the "best-expert curve" and "best-expert Spec curve" over
   layers next to the block-rescue curve (Figure 1a extended).
3. Global search: argmax over all (layer, expert) pairs on discovery; evaluate the winner on validation and compare
   with the paper's two-stage winner (L44E069 / L19E006). Also report the top-10 pairs with validation CIs, and the
   concentration ratio Rescue(e*)/Rescue(layer) per layer (does the signal concentrate more in some layers?).
4. Robustness of the answer: repeat with the Appendix D split-seed × threshold grid, and with relaxed thresholds
   (recurrence 32, 48) because experts at non-peak layers may be recurrent but less so.
5. Statistics caveat to state in the writeup: searching 48×128 pairs inflates the discovery maximum; only the
   validation numbers of the pre-registered winner count. Report how much the discovery-max shrinks on validation.

**Deliverable.** Section "Layer-then-expert vs joint search" with the two curves per model, the top-10 table, and a
verdict per model (does the joint winner change? does it beat L44E069/L19E006 on validation?).

**Agent.** One agent (`ext1-joint-search`), fully parallel with the others; needs the GPU queue only. Estimated
0.5 day agent time.

---

## Direction 2 — Generalisation across models, and attention vs MoE attribution

**Question A.** Does the pattern (Qwen3: one specific expert; Mixtral: coalition) hold in other MoE models, in
instruction-tuned variants of the same models, and at larger scale?

**Question B.** How much of the factual-recall rescue sits in the MoE sublayer versus the attention sublayer at each
layer? (Requires a new intervention kind.)

**Candidate models (all ungated; checked 2026-09-14).**

| Model | Arch | Engine change | bf16 size | Pass time | Why |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | Qwen3-MoE | none | 61 GB | 60 s | post-training axis, same weights lineage as Base |
| Qwen3-Coder-30B-A3B-Instruct | Qwen3-MoE | none | 61 GB | 60 s | code specialisation; also the natural model for Direction 4 |
| Mixtral-8x7B-Instruct-v0.1 | Mixtral | none | 93 GB | 95 s | post-training axis for Mixtral |
| OLMoE-1B-7B-0125 (+Instruct) | OLMoE | none (already supported) | 14 GB | 4 s | fully open training data; small |
| Mixtral-8x22B-v0.1 | Mixtral | none | 281 GB | ~3 min | scale axis within one family. ❓ needs 281 GB: NVMe has 265 GB free, so the OLMoE fp32 shards (39 GB) must be deleted first, or Mixtral-8x7B after its runs finish |
| Granite-3.1-3B-A800M-base | GraniteMoe | moderate (fused ParallelExperts, residual/attention multipliers) | 7 GB | seconds | third independent family, small |
| Qwen1.5-MoE-A2.7B | Qwen2-MoE | moderate (shared expert + gate) | 29 GB | 30 s | shared-expert design: how does the shared expert share the rescue? |
| DeepSeek-V2-Lite / Coder-V2-Lite | DeepSeek-V2 | large (MLA attention, shared experts, first layer dense) | 31 GB | 30 s | different attention; coder variant for Direction 4 |
| gpt-oss-20b | GptOss | large (MXFP4 dequant, learned attention sinks, sliding layers, biased SwiGLU) | 41 GB | 40 s | explicit learned attention sinks make it the ideal foil for Direction 3 |

Recommended core set (zero engine change, ~4 hours of runs including downloads): Qwen3-30B-A3B-Instruct-2507,
Qwen3-Coder-30B-A3B-Instruct, Mixtral-8x7B-Instruct, OLMoE base and Instruct. ❓ Mixtral-8x22B if the disk swap is
approved. ❓ Granite and Qwen1.5-MoE as second wave (moderate engine work). DeepSeek and gpt-oss only if a specific
question needs them.

**"Test each model the way it is meant to be used."** For every model a `model_usage.json` is written before any run,
verified by the agent from `tokenizer_config.json`, `generation_config.json` and the model card:
BOS/EOS behaviour (Mistral: BOS; Qwen: none; OLMoE: none; Granite: none), chat template (instruct models),
recommended dtype, any system prompt. Each model is then run under two protocols where they differ:
(i) *intended*: tokenizer defaults, and for instruct models the cloze prompt wrapped as the assistant's continuation
of a minimal chat template (❓ the user decides whether instruct models are run raw, templated, or both);
(ii) *paper*: no special tokens, raw cloze. Reporting both is what turns the BOS observation into a finding about the
method's sensitivity.

**Attention vs MoE attribution (Question B).** Engine extension: new SpawnSpec kinds `attn_layer` (replace the
final-position attention-sublayer output with the clean one) and `block` (replace the whole layer output = attention +
MoE). The wavefront mechanics are identical (the spawn starts after the sublayer instead of after the MoE block).
Verify on OLMoE against transformers hooks like the existing `verify_olmoe.py`. Then one sweep pass per model gives
three curves: attention-output rescue, MoE-output rescue, whole-layer rescue, on the same cases. Compare the areas
under the curves and the peak layers; check additivity (attention + MoE ≈ block?) which tells whether the two
sublayers carry the same or complementary information.

**Deliverables.** Per model: the paper's Table 1 row, Figure 1a curve, expert selection and Spec, under both
protocols; the attention/MoE/block curve triptych; a cross-model summary table and a short verdict on
generalisability (pattern A: specific single expert; pattern B: coalition; pattern C: no layer-level localisation).

**Agents.** `ext2-model-zoo` (usage specs, downloads, runs, per-model sections) and `ext2-attn-patch` (engine
extension + verification), parallel; the model-zoo agent uses the new kind once it is verified. Estimated 1.5 days.

---

## Direction 3 — Why does BOS change Mixtral's expert selection?

**Facts so far.** Same weights, same noise; only the `<s>` token at position 0 differs. At L19 the final-position
router changes for many prompts: E006 clean-active 71/67 (BOS) vs 91/83 (no BOS); E002 76/84 vs 59/71. Δ_clean and the
noise drop also change (e.g. case 3167: drop +1.4 vs +5.6). Qwen3 (no BOS in its tokenizer) is unaffected.

**Hypotheses to separate.**
H1 *Attention-sink relocation*: with BOS, position 0 absorbs "idle" attention mass and carries a huge-norm hidden
state; without BOS the first content token takes that role, its representation is distorted, and every later
position's attention pattern shifts, moving the final-position residual across router decision boundaries.
H2 *BOS-specific learned semantics*: the router (or upstream layers) learned features tied to the BOS embedding
itself, not to "some token at position 0".
H3 *Position-shift artefact*: removing BOS shifts every token's RoPE position by one; the effect is positional, not
about the token identity.
H4 *Expert-role*: E006 is a "default / high-entropy" expert that the router falls back to when the residual carries a
sink-like or out-of-distribution signal; E002 is content-specific. Then no-BOS routes more prompts to the default
expert, and the paper's negative specificity is partly an artefact of testing an OOD input.

**Experiments (all on Mixtral-8x7B, paper case set; each pass ~95 s).**
1. *Diagnostics recording* (small engine extension: record, at every layer, the final position's attention mass on
   position 0 and on the first content token, per head; the hidden-state norm of every position; router logits for
   all 8 experts at the final position). Run with and without BOS. Plot: sink mass vs layer, norm of position 0 vs
   position 1, per-layer routing agreement between the two protocols (fraction of prompts with identical top-2),
   and the layer at which final-position residuals start to diverge (cosine similarity).
2. *Substitution controls* (tests H1 vs H2 vs H3): position 0 = `<s>` (BOS), `</s>`, `\n`, `.`, a frequent content
   token, a rare token, two `<s>`, and no token but positions shifted by +1 (keeps RoPE positions identical to the
   BOS run while removing the token; requires a position-offset option). For each: routing agreement with the BOS run
   and with the no-BOS run at L19, E002/E006 activity counts, layer-sweep and expert-level results. If any position-0
   token restores the BOS-run routing → H1; if only `<s>` does → H2; if the shifted no-token run matches the BOS run →
   H3.
3. *Router-boundary analysis* (H4): at L19, distribution of the router logit margin between E006 and E002 across
   prompts in both protocols; corpus-level expert usage at L19 on ~50k tokens of generic text (e.g. a Wikipedia sample
   and a code sample) with and without BOS: is E006 the most-used or the most position-agnostic expert? Entropy of
   each expert's token distribution.
4. *Symmetric test on Qwen3*: prepend `<|endoftext|>` (Qwen's document separator) to every prompt and rerun the paper
   protocol. If L44E069 survives, the sensitivity is a Mixtral property (learned sink), not a general MoE property.
5. *OOD check*: mean log-likelihood of the prompts under both protocols (no-BOS should be worse for Mistral); relate
   per-prompt likelihood shift to routing change.
6. *Literature* (separate agent, no GPU): attention sinks (Xiao et al. 2023), massive activations (Sun et al. 2024),
   first-token / BOS effects on routing in MoE (Mixtral routing analysis in Jiang et al. 2024; any 2025-26 work on
   sink tokens and expert choice), ROME/CounterFact tokenisation conventions (GPT-2 has no BOS, which is probably why
   the paper's pipeline omits it). Output: 2-page synthesis with the hypotheses above mapped to prior evidence.

**Deliverable.** Section "Why BOS moves the Mixtral router", with the diagnostics figures, the substitution table,
the router-margin analysis, the Qwen3 symmetric test, and a stated conclusion on H1 to H4. Estimated 1 day agent
time, ~25 GPU minutes.

**Agents.** `ext3-bos-mechanism` (engine diagnostics + experiments) and `ext3-literature` (reading), parallel.

---

## Direction 4 — Expert-aware tracing on code: a taxonomy of counterfactuals

**Question.** Does the method transfer from factual recall to code, where the "fact" is syntactic (matching bracket,
keyword) or semantic (variable recall, API name)? Do different categories localise to the same layers and experts?

**Phase A: dataset ("CodeFact"), CPU only, the main effort.** Build CounterFact-style items from Python code.
Sources ❓: HumanEval and MBPP canonical solutions (small, clean, licence-friendly) and a sample of permissively
licensed functions from The Stack / CodeSearchNet for volume. Each item = (prompt prefix, true next token, foil next
token of the same category, "subject" span that determines the answer, category). Categories (initial taxonomy):

| Category | Prefix ends just before | True vs foil | Subject span to noise |
|---|---|---|---|
| S1 closing bracket | the closing `)` `]` `}` | matching closer vs another closer | the opening bracket token |
| S2 block keyword | `else`/`elif`/`except`/`finally` at block start | true keyword vs sibling keyword | the `if`/`try` line |
| S3 keyword completion | `in` after `for x`, `:` after `def f(args)` | true vs plausible alternative | the `for`/`def` token |
| R1 variable recall | a re-use of a variable defined earlier (`return total`) | defined name vs another in-scope name | the definition site |
| R2 attribute / API recall | `.append` after a list, `np.zeros` after `import numpy as np` | true vs same-object alternative | the object's definition / import |
| R3 literal / constant recall | reuse of a constant defined above | true vs another constant | the constant's definition |

Both true and foil must be single tokens in the model's tokenizer (as in the paper); indentation and whitespace tokens
are excluded from S categories for that reason (❓ include a whitespace category or not).

**Thresholds.** The paper's absolute filters (Δ_clean ≥ 1.0, drop ≥ 0.5) were tuned for facts. For code, syntax items
will have very large Δ_clean (10 to 20) and may show tiny drops when the "subject" is noised, because the answer is
redundantly determined by context. Phase A therefore first measures the distribution of Δ_clean and drop per category
on ~500 items each, then proposes: (a) keep the paper's absolute thresholds and report per-category pass rates (the
pass rate itself is a result: which categories are "recall-like"); (b) a relative variant (drop ≥ 25% of Δ_clean).
❓ The user chooses which is primary. Everything else (256 cases per category, 128/128 split, recurrence ≥ 64,
active-random controls, bootstrap) stays as in the paper.

**Phase B: GPU runs.** Models ❓: Qwen3-30B-A3B-Base (the paper's model, a competent coder), Qwen3-Coder-30B-A3B-Instruct
(same architecture, code-specialised; run in its intended template and raw), Mixtral-8x7B (no BOS, paper protocol).
Per model and category: filter → sweep → all-layer expert pass (Direction 1 machinery). GPU: ~5 passes per model per
category set; all categories can share passes (union of cases), so ~10 minutes per model.

**Phase C: analysis.** Per category: layer curve, selected layer and expert, Spec, coalition check. Across categories:
overlap of recurrent experts (Jaccard), whether the same layer band is selected, and whether the factual-recall expert
(L44E069) appears for any code category. Compare syntax (S) vs recall (R) categories: the hypothesis is that R
categories behave like CounterFact (localised mid-late layers, single expert for Qwen3) while S categories show
little subject-noise sensitivity and diffuse rescue.

**Deliverables.** `data/codefact/` (items + builder scripts + category stats), section "Expert-aware tracing on code"
with per-category tables and the cross-category overlap matrix, and a threshold-calibration appendix.

**Agents.** `ext4-codefact-data` (Phase A), then `ext4-codefact-runs` (Phases B and C, reuses Direction 1's all-layer
expert pass). Phase A can start immediately; Phase B waits for A and for the user's model choice. Estimated 2 days.

---

## Coordination

Parallel tracks from day 1: ext1-joint-search, ext2-attn-patch, ext2-model-zoo (downloads + usage specs first),
ext3-bos-mechanism, ext3-literature, ext4-codefact-data. GPU work goes through the queue; the coordinator reviews
each track's section before it enters `results/EXTENSIONS_REPORT.md`, keeps `logs/PROGRESS.md`, and stops for the
❓ decisions. Usage-limit exposure: six agents in parallel burn the account window several times faster than the
overnight run did (38 min for one agent). ❓ The user chooses the concurrency: all six at once (fastest, likely hits
the limit within the hour and pauses), or two waves (1+3 first, then 2+4).

Order of value if the budget forces a choice: Direction 1 (cheap, directly tests the paper's method) → Direction 3
(explains our own headline finding) → Direction 2 core set → Direction 4 (largest, most novel).

## Decisions (recorded 2026-09-14, from the user)

1. **Direction 2 models: core set only** for now: Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B-Instruct,
   Mixtral-8x7B-Instruct-v0.1, OLMoE-1B-7B-0125 base and Instruct. No Mixtral-8x22B, no second wave (Granite,
   Qwen1.5-MoE) unless the core-set results motivate it.
2. **Instruct models: both protocols**: raw cloze (comparable with the base models) and chat-template-wrapped
   (intended usage); differences reported.
3. **Direction 4: Python only; sources HumanEval + MBPP + a permissively licensed sample of The Stack; the paper's
   absolute thresholds (Δ_clean ≥ 1.0, drop ≥ 0.5) are primary**, per-category pass rates are themselves a result;
   the relative rule (drop ≥ 25% of Δ_clean) goes to an appendix. Models: Qwen3-30B-A3B-Base, Qwen3-Coder-30B-A3B-
   Instruct (both protocols), Mixtral-8x7B without BOS.
4. **Two waves**: wave 1 = Directions 1 and 3 (agents ext1-joint-search, ext3-bos-mechanism, ext3-literature);
   wave 2 = Directions 2 and 4 (ext2-model-zoo, ext2-attn-patch, ext4-codefact-data → ext4-codefact-runs).
   Sub-agents run inside the coordinating session during working hours; if the user steps away, the work is handed
   to the headless supervisor per PLAYBOOK.md.
5. Deliverable (default, not asked): `results/EXTENSIONS_REPORT.md` with one section per direction plus an artifact
   page; a paper-style writeup only if requested afterwards.

---

# Phase 2 — follow-ups from the meeting notes (planned 2026-09-20, not started)

Five items. For each: what it means concretely, what can be answered from existing data at zero GPU cost, what needs
engine work, the experiments, and the decisions. Cost estimates use the measured pass times (Qwen3 ~60 s, Mixtral
~95 s, OLMoE ~4 s).

## F1. Expert ranking and minimal sufficient sets (mostly free)

*Existing data.* `results/{qwen3_bos,mixtral_bos,mixtral_nobos}_alllayers/expert_rows.parquet` hold, for every case
and every layer, the rescue of every clean-active expert plus the clean-top-k coalition and the block patch;
`results/qwen3/expert_rows.parquet` additionally holds all 56 ordered equal-norm pairs at L44.
*Fact checked today:* single-expert rescues are additive **on average** but not per case: at Qwen3 L44 (paper
validation) Σ_e rescue_e = +0.935 vs coalition +0.916 vs block +0.941, per-case r = 0.93, mean |Σ − coalition| = 0.36;
at L40/L42/L43 r = 0.77-0.89. So population-level sets can be built from single-expert data; per-case minimal sets
need joint patches.

1. **Rankings (no GPU).** Per layer and for the joint (layer, expert) space: validation rescue, active-only rescue,
   Spec, share of block rescue, per-case rank among active experts (Table-10 style), grid stability; Kendall τ between
   the rankings; full tables `ext5_ranking_<run>.csv` (all 128 experts × 48 layers for Qwen3).
2. **Population-level minimal set (no GPU, additive approximation).** Greedy forward selection over experts at L*
   (and over (layer, expert) pairs): add the expert whose inclusion raises the mean summed rescue most; report the
   curve "fraction of block rescue vs number of experts" and the set reaching 50/80/90%; validate the additive
   estimate against the exact coalition value at the end point.
3. **Per-case minimal set (1 pass per model).** Exact joint patches of all 2^k − 1 subsets of each case's k
   clean-active experts at L* (Qwen3: 255 subsets × 256 cases ≈ 65k wavefront rows, T ≈ 8-11 → one pass; Mixtral:
   3 subsets). Engine: a `coalition_set` kind taking an explicit expert list (small extension of `coalition_clean`).
   Outputs: distribution of the smallest subset reaching 80% of the block rescue; how often {E069} alone suffices;
   interaction terms (pair rescue − sum of singles) to quantify the non-additivity seen above; the same for L42
   (E115) and for Mixtral no-BOS L18/L19.
4. **Multi-layer sets (2-3 passes).** Greedy over (layer, expert) pairs with exact joint patches at several layers in
   one wavefront row (engine: allow a spawn to carry interventions at a list of layers). Question: how many experts
   across layers are needed to recover, say, 80% of the best single-layer *block* rescue, and do L42E115 + L44E069
   together beat either alone (they covered 89% of cases jointly in ext1)?

## F2. Attention heads at the final token (engine extension, ~1 pass per model)

*Existing.* `attn_layer` patches the whole attention-sublayer output at the final position; `DiagSpec.attn_final`
records the final position's attention mass per head over positions (used in ext3).
1. **Head-level patching.** Record the pre-`o_proj` per-head outputs at the final position; new kind `attn_head`
   with head index h: v = W_o[:, slice_h] · (head_clean − head_noised). Because `o_proj` is linear, head rescues
   should sum to the `attn_layer` rescue (check, as for experts). Run at the attention peaks: Qwen3 L40, L43 (+ L44
   as a null), Mixtral L15, L18, L19, L24: 256 cases × 32 heads × 4 layers ≈ 33k rows → one pass per model.
2. **Head ranking and specificity.** Same statistics as for experts: rescue, Spec against other heads of the layer,
   recurrence is not needed (all heads are always "active"); minimal head set for 80% of the attention rescue.
3. **What the top heads attend to.** From `attn_final`: for the top heads, attention mass on the subject span vs the
   relation tokens vs position 0, clean vs noised. Expect "mover" heads reading the subject's last token; report how
   noise redistributes their attention. Cross-check with the MoE side: does the L44E069 rescue depend on the L40 heads
   (patch heads at L40 and read the routing / contribution of E069 at L44 — the diagnostics already record routing).
4. Optional: repeat on Qwen3-Instruct (does post-training keep the same mover heads as it keeps the experts?).

## F3. Input attribution: gradients w.r.t. the input (engine extension, 2 passes per model)

*Existing.* The engine is forward-only (`@torch.no_grad()`); transformers with offload gives gradients for a few
prompts at ~1 min each; OLMoE fits the GPU and supports normal autograd.
1. **Reverse layer streaming.** Forward pass stores each layer's input at all positions (256 × T × 48 × 2048 bf16 ≈
   0.5 GB); backward pass streams layers from L−1 down to 0, recomputing each layer with autograd on and passing
   ∂Δ/∂h_l backwards (layer-wise checkpointing). Yields ∂Δ/∂embedding for every input token and ∂Δ/∂h_l(final) for
   every layer. ~1 day of engine work, 2 passes per model.
2. **Gradient×input attribution per input token** (and integrated gradients along the clean→noised embedding path
   as a cheap variant): which tokens carry Δ? Compare with the paper's subject span and with a *noise-ablation
   attribution* that needs no gradients: noise each position separately (256 × T ≈ 2.5k prefill rows, one pass) and
   record the drop. Report agreement between the three maps.
3. **Attribution patching (Nanda 2023) validated against exact patching.** With ∂Δ/∂h_l(final), the first-order
   estimate of the MoE-output patch is grad · (MoEOut_clean − MoEOut_noised) and of expert e is grad · δ_e. We have the
   exact values for every layer and every expert, so we can measure how well the linear estimate ranks layers,
   experts and heads in MoE models (a methodological result in its own right; expected to fail where routing flips).
4. Decision: start with OLMoE via transformers autograd (no engine work) to fix the definitions, then port.

## F4. Patch at the subject token instead of the last token (engine extension, ~10 passes)

*Existing.* All interventions are at the final position; wavefront rows are single tokens. Patching position p
changes positions p..T−1, so the row must carry the suffix (T−p tokens) from layer l upward.
1. **Suffix wavefront rows.** Engine: a spawn with a position index p runs positions p..T−1 for layers ≥ l, attending
   to the parent's K/V for positions < p and its own for ≥ p. Cost per row ≈ (T−p) tokens instead of 1 (CounterFact
   T ≈ 8-11: 3-8×), still one pass per layer sweep. ~1 day of engine work.
2. **ROME-style grid.** Rescue for every (position, layer) with the MoE-output patch, and separately the attention
   output and the whole residual (`resid`) — the classic causal-tracing heatmap the paper never produced. 256 cases ×
   T positions × 48 layers ≈ 120k token-rows → 2-4 chunked passes per model. Positions grouped as: first subject
   token, middle subject tokens, last subject token, relation tokens, final token.
3. **Expert level at the subject site.** At the layers where the last-subject-token MoE patch peaks (expected early-
   to-mid layers, the "early site"), rank experts by rescue as in the paper; check whether they are distinct from the
   final-token experts (L44E069) and whether Mixtral's coalition/single-expert asymmetry holds there too.
4. **Design caveat to state.** Noise sits on the subject tokens, so restoring the *whole residual* of all subject
   positions at layer 0 trivially undoes the corruption; only the last-subject-token patch (or MoE/attention *output*
   patches, which do not restore the corrupted embedding itself) are informative, and the layer profile matters more
   than absolute values. Also decide whether the reference for "rescue" stays Δ_noised of the whole prompt.

## F5. Probability instead of logit (small engine extension, re-analysis)

*Fact to state first.* Δ = logit(true) − logit(foil) is exactly the log-odds log p(true) − log p(foil): the softmax
normaliser cancels. So "logit difference" already is a probability-scale quantity for the two-way contrast; what it
is not is the absolute probability of the true token.
1. **Metrics to add for every run.** (a) p(true) and p(foil) under the full softmax (needs the log-sum-exp of the
   full vocabulary for wavefront rows; the engine now computes only the two needed logits — add a chunked logsumexp,
   ~1 s per pass, no memory issue); (b) rescue on the probability scale Δp = p_patched(true) − p_noised(true);
   (c) normalised rescue = rescue / drop, aggregated as mean rescue / mean drop with bootstrap (per-case ratios blow
   up when drop is small; ext2_zoo already reports the aggregate ratio); (d) rank of the true token; (e) KL(patched ‖
   clean) at the final position. Store all in `sweep_rows` / `expert_rows` so every table can be re-derived.
2. **Re-derive the main tables under each metric** (no new GPU beyond one re-run per base run with the logsumexp on,
   ~5 passes total) and report whether layer selection, expert selection, Spec sign and the cross-model patterns
   change. Expected: rankings stable, absolute values compress for confident prompts (p ≈ 1 saturates), and
   probability-scale rescue over-weights cases where the noise flipped the top-1 token.
3. **Filters.** Try p_clean(true) ≥ 0.5 (model actually answers correctly) as an alternative funnel and report the
   overlap with the paper's Δ-based funnel.

## Suggested execution

Order by value/cost: F1.1-F1.2 (free, half a day) → F5 (small engine change, reuse everything, half a day) → F2 (one
day) → F1.3-F1.4 (uses F2's set machinery) → F4 (engine day + 10 passes) → F3 (engine day + validation study).
Total GPU ≈ 40 passes ≈ 1 h; agent time ≈ 4-5 days; parallelisable as F1+F5 (analysis-heavy), F2, F4, F3 with the
same file-ownership and gpu_queue rules as Phase 1. Decisions for the user: which of F3/F4 to prioritise (both need
a day of engine work), whether F4 should include the full position × layer grid or only the subject-last-token
column, and whether probability-scale metrics should replace or accompany Δ in the reports.

## Phase 2 execution plan (decided 2026-09-21)

User decisions: F4 before F3; F4 this round = the **subject-last-token column only** (no full position × layer grid);
F5 probability-scale metrics are reported **alongside** Δ (main tables stay on Δ). Two waves again.

**Wave 1 (three sub-agents in parallel, coordinator serialises the GPU and merges):**

| Agent | Owns (files) | Work | GPU |
|---|---|---|---|
| `ext5-engine` | `moetrace/engine.py`, `scripts/run_sweep.py`, `scripts/run_expert.py` (flag additions only), `scripts/ext5_engine_*`, `scripts/ext5_heads_*`, `scripts/ext5_subsets_*`, `moetrace/ext5_heads.py`, `moetrace/ext5_subsets.py`, section `ext5_f2_heads.md` | (1) F5 engine fields: chunked full-vocabulary log-sum-exp for prefill and wavefront rows → `logp_true`, `logp_foil`, `p_true`, `p_foil`, `rank_true`, KL(patched‖clean) stored in `sweep_rows`/`expert_rows` behind `--metrics`; (2) F2 `attn_head` kind (per-head pre-`o_proj` outputs, v = W_o[:, h]·(head_clean − head_noised)); (3) F1 `coalition_set` kind (explicit expert list) and multi-layer intervention lists on one spawn; OLMoE verification for all three (verify_olmoe identity, HF-hook check for `attn_head`); then runs: F5 re-runs of the base sweeps + L*/L42/L18 expert passes with metrics on (`results/<run>_metrics`), F2 head sweeps at Qwen3 L40/L43/L44 and Mixtral L15/L18/L19/L24 (`results/<run>_heads`), F1.3 exhaustive subset passes at Qwen3 L44/L42 and Mixtral no-BOS L18/L19 (`results/<run>_subsets`); F2 analysis + section | ~12 passes |
| `ext5-analysis` | `moetrace/ext5_rank.py`, `moetrace/ext5_metrics.py`, `scripts/ext5_rank_*`, `scripts/ext5_metrics_*`, sections `ext5_f1_rankings.md`, `ext5_f5_metrics.md` | F1.1 rankings and F1.2 population-level minimal sets from the existing all-layer data (zero GPU, start immediately); analysis code for F1.3 subsets and F5 metrics written against the agreed schemas, run when `ext5-engine` delivers the parquet files (coordinator relays); sections | 0 |
| `ext5-subject` | `moetrace/ext5_subject.py` (new executor path for suffix rows; may import helpers from engine.py but must not edit it), `scripts/ext5_subject_*`, section `ext5_f4_subject.md` | F4: suffix wavefront rows starting at the **last subject token** p; kinds `layer` (MoE output), `attn_layer`, `resid` at p for every layer; verification vs transformers hooks on OLMoE; runs on Qwen3 (paper set, defaults), Mixtral no-BOS and BOS (paper set); expert-level pass at the subject-site peak layers; comparison with the final-token curves; section | ~12 passes |

Schemas agreed up front (so analysis can be written before the data exists): new columns `logp_true`, `logp_foil`,
`p_true`, `p_foil`, `rank_true`, `kl_to_clean` on every row of `sweep_rows`/`expert_rows` in `*_metrics` runs;
head rows in `expert_rows`-like `head_rows.parquet` (case_id, layer, head, kind=`attn_head`, rescue, …); subset rows
in `subset_rows.parquet` (case_id, layer, experts as sorted comma list, n_experts, rescue); subject rows in
`sweep_rows.parquet` of `*_subject` runs with an extra `pos` column (= subject last token index) and the same kinds.

**Wave 2 (after wave 1 merges):** F3 gradient attribution (reverse layer streaming, attribution-patching validation),
F1.4 multi-layer minimal sets, optional F2 on Qwen3-Instruct. F4's full position × layer grid stays deferred.

Rules as in Phase 1: every GPU job through `scripts/gpu_queue.sh`; detached chains; `run_meta.json`; PROGRESS.md
entries; final reports under 450 words; coordinator commits selectively and rebuilds `EXTENSIONS_REPORT.md`
(sections ext5_* appended to `scripts/build_extensions_report.py`).


**Wave 1 status (2026-09-21 02:45 UTC): COMPLETE.** All three agents delivered; sections `results/sections/ext5_f4_subject.md`,
`ext5_f1_rankings.md`, `ext5_f2_heads.md`, `ext5_f5_metrics.md` are assembled in `results/EXTENSIONS_REPORT.md` (Directions
5-F4, 5-F1, 5-F2, 5-F5). GPU ≈ 15 min (F4) + 26 min (F5/F2/F1.3 incl. verifications). Headlines are summarised in
CLAUDE.md section 2b; open questions for the user are listed there and in the sections. Wave 2 (F3, F1.4, optional F2 on
Qwen3-Instruct, F4 full grid) not started.

---

# Phase 3 — expert add-back curves and WinoGrande under STR (planned 2026-10-04; only W0 run)

Source: the team's notes after Directions 6/6b were shared. (1) *Add-back*: plot the remaining gap against the number of
experts patched back (single, pairs, triples, ...), ranked over experts rather than layers; the ceiling is "all MoE outputs
patched"; how many experts restore the original answer; is greedy good enough, are there better strategies; AUC of the
saturation curve. (2) *WinoGrande*: how much of the repair is attributable to MoE vs attention? Hypothesis: WinoGrande is
like IOI (attention-driven) rather than like factual recall.

## Best-practice checklist (Zhang & Nanda 2024, arXiv:2309.16042), applied to every Phase-3 experiment

| # | Recommendation (where in the paper) | How Phase 3 applies it |
|---|---|---|
| Z1 | STR whenever possible; the corrupted prompt must be in-distribution ("identically distributed as a fresh draw of a clean prompt"), same sequence length (§2.1, §6) | CounterFact: the ext6 donors. WinoGrande: the twin, i.e. the same sentence filled with the *other* option — both prompts are crowd-written WinoGrande sentences filled with their own correct answer; rule W4 enforces token symmetry |
| Z2 | r′ = the corrupted prompt's own answer; prompts the model actually solves (PairedFacts are "known facts"; arithmetic keeps only prompts whose top-1 is correct; App. B, C) | margin ≥ 1 logit in **both** directions (primary); top-1 in both directions as a sensitivity subset |
| Z3 | Use each pair both ways (PairedFacts, App. B) | every pair is run A→B and B→A; per-pair mean = primary statistic, one direction = sensitivity (the ext6 "first donor" analogue) |
| Z4 | Prefer STR; GN only "when token alignment or lack of analogous tokens makes STR unsuitable" (§6) | **STR only (user decision 2026-10-04).** WinoGrande twins are a natural STR setting, so no GN rows, no GN filter, no GN control in Phase 3 |
| Z5 | Logit difference, normalised by LD_clean − LD_corrupt; probability can miss negative components; Stolfo's metric explodes when P_*(r) ≈ 0; KL acceptable for circuit discovery (§2.1, §4, §6, App. C) | Δ = LD primary, reported raw and drop-normalised; Δp descriptive only (F5); KL only for head discovery |
| Z6 | Single-layer patching first; sliding windows only when single layers are weak, and read as joint effects (§5, §6) | all sweeps and grids single-layer first; window 5 only as a 6b-style secondary |
| Z7 | Try several corruption sites when the task allows it (§6, App. F: S2 vs S1+IO find different IOI heads) | WinoGrande, two STR sites: (i) the filled option (primary), (ii) entity-role swap of the two candidates' earlier mentions (names) |
| Z8 | Head detection = effect ≥ 2 SD from the mean over heads (§3) | reported next to our bootstrap CIs and Spec |
| Z9 | Early-layer effects at the corrupted token largely reflect token identity (fn. 1, the MLP0 effect) | L0–L2 effects at the STR site are reported, not interpreted as computation |
| Z10 | Noising (clean → corrupt) vs denoising is named as open (§8) | add-back = denoising (sufficiency); deletion curves = noising (necessity) |

## Robustness criteria inherited from the STR protocol of Direction 6 and earlier directions

| Criterion | Origin | WinoGrande form |
|---|---|---|
| clean margin Δ_clean ≥ 1.0 | paper filter, kept by ext6 | Δ(prompt A) ≥ +1 |
| corrupted prompt known: logit(foil) − logit(true) ≥ 1.0 | ext6 donor margin | Δ(prompt B) ≤ −1 (together: STR drop ≥ 2, so the corruption is effective by construction) |
| answer and foil are single leading-space continuation tokens; true ≠ foil | `data.prepare_case` | W3: sentence-final trigger is one token after both prompts, the same id after both |
| same length, subject at the same positions, all other tokens identical, same continuation ids after the donor | `ext6_str.candidates` | W4 (+ W3) |
| subject is not the final token | `data.prepare_case` ("subject_is_final") | W5: the option is not the final prompt token |
| paper split, fixed seed; recurrence gate = half of discovery | paper / ext6 | seed-0 shuffle of `margin` pairs → 128/128 discovery/validation pairs; gate = half of discovery |
| donor mean primary, first donor sensitivity | ext6 | mean of both directions primary, direction A→B sensitivity |
| Mixtral without BOS = paper protocol, BOS as a labelled variant | base reproduction, Dir. 3 | pairs are identical under both; both protocols run |
| sink-carrying final tokens flagged (Mixtral without BOS) | Dir. 3, open question (d) | `DiagSpec` flag per prompt |
| last-layer read-out: interior-layer rule (≤ L−5) reported alongside | Dir. 4 | same |
| clean top-1 rate, normalised rescue, Δp descriptive | F5 | same |
| new intervention kinds verified against transformers hooks on OLMoE; `verify_olmoe.json` unchanged | all directions | WinoGrande OLMoE pairs exist (W0) |
| bf16, no quantization; `gpu_queue.sh`; `run_meta.json`; PROGRESS.md | CLAUDE.md §6 | same |

WinoGrande-specific rules: W1 twin integrity; W2 the trigger is the *only* difference and the *last word* of the item (user
rule 1), so everything that determines the answer precedes the prediction position; W3 the trigger is a single token for the
model's tokenizer (user rule 2); W6 one pair per normalised context (options replaced by placeholders), so no template can sit
in both discovery and validation. Recorded for stratification, not filtered (filtering would remove whole mechanisms; Bowman
& Dahl 2021): `assoc` (the context-free prompt "The bag was too" already passes the margin both ways = solvable by word
association), `names` (social vs physical items), `trigger_in_context` (copy shortcut), `debiased` (AfLite survivor),
option token count, top-1 status.

Not used: the paper's second STRICT condition (GN drop ≥ 0.5). It described how the paper's GN-corrupted case set was
selected (the ext6 cases inherited it only because they were the paper's case IDs); it is not an STR criterion. Measured
once in the W0 scan and dropped: it would select pairs by the behaviour of an off-distribution corruption, and it does so
unevenly (Qwen3 2,099 → 813 pairs with the name share rising from 0.25 to 0.37; Mixtral without BOS 1,109 → 167), because
noise on one option token does not hide which of the two in-context candidates is meant.

## W0. WinoGrande STR pairs and competence scan (DONE 2026-10-04)

Construction (`moetrace/ext7_wino.py`): in 86 % of WinoGrande twins the trigger follows the blank, so a prompt cut at the
blank is identical for both twins (accuracy exactly 50 %, drop 0). The direction is reversed instead: fill the blank with
each twin's own answer and predict the sentence-final trigger.

    prompt A: The coroner tried to put the body in the bag but the bag was too   -> " small"  (r)
    prompt B: The coroner tried to put the body in the bag but the body was too  -> " large"  (r')

Δ = logit(r) − logit(r′) = LD(r, r′); the filled option plays the CounterFact subject, the two triggers true object / foil.
Summing the two twins' lm-eval partial-scoring margins at the trigger token gives exactly Δ(A) − Δ(B), the STR drop.
Positions before the option are identical in A and B, so patches there are exactly zero (as the 6b prefix).
Scripts: `scripts/ext7_wino_build.py` (W1–W6, CPU) → `data/wino_str/pairs_<split>_<proto>.parquet`;
`scripts/ext7_wino_scan.py` (clean and context-free prompts × A / B) → `results/wino_<proto>/`. The first scans
(2026-10-04) also carried GN rows; their `gn_*` / `strict` columns are not used.
Raw data `data/winogrande_1.1/` (gitignored copy of the official v1.1 zip). STR-only funnel (`scripts/ext7_wino_funnel.py`
→ `results/ext7_wino_funnel.json`, `results/tables/ext7_wino_funnel.md`):

| Split | Model / protocol | Twins | W2 sentence-final single-word trigger | W3 single-token trigger | W4–W5 token-symmetric | W6 dedup | Correct both ways | **Margin ≥ 1 both ways (primary)** | of which names / objects | assoc | top-1 both | debiased | mean Δ_A / Δ_B | mean drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train_xl | Qwen3-30B-A3B-Base | 20,199 | 6,737 | 5,501 | 4,532 | 4,514 | 2,783 | **2,099** | 535 / 1,564 | 183 | 313 | 495 of 1,549 | +3.89 / -3.79 | 7.68 |
| train_xl | Mixtral-8x7B, no BOS | 20,199 | 6,737 | 4,037 | 2,430 | 2,421 | 1,489 | **1,109** | 127 / 982 | 113 | 273 | 255 of 845 | +3.72 / -3.55 | 7.27 |
| train_xl | Mixtral-8x7B, BOS | 20,199 | 6,737 | 4,037 | 2,430 | 2,421 | 1,524 | **1,151** | 150 / 1,001 | 132 | 295 | 279 of 845 | +3.78 / -3.65 | 7.42 |
| train_xl | OLMoE-1B-7B (verification only) | 20,199 | 6,737 | 5,115 | 3,823 | 3,809 | 1,952 | **1,325** | 218 / 1,107 | 122 | 290 | 283 of 1,341 | +3.71 / -3.48 | 7.18 |
| dev | Qwen3-30B-A3B-Base | 284 | 90 | 79 | 68 | 68 | 35 | **27** | 3 / 24 | 5 | 2 | 0 of 0 | +3.83 / -3.00 | 6.83 |
| dev | Mixtral-8x7B, no BOS | 284 | 90 | 59 | 42 | 42 | 20 | **11** | 0 / 11 | 1 | 3 | 0 of 0 | +2.95 / -3.43 | 6.39 |

`assoc` = the context-free prompt already passes the margin both ways; `debiased` = margin pairs in train_debiased, out of
the W1–W6 pairs in train_debiased. Pass rates on the AfLite-filtered items (train_xl debiased subset: Qwen3 32 %, Mixtral
30 %) match the dev twins (40 % and 26 % on 68 / 42 pairs), so memorised train_xl items do not inflate the pool. Margin pairs
shared by models: Qwen3 ∩ Mixtral no BOS 822, all three protocols 776, Mixtral no BOS ∩ BOS 1,022. Role-swap (W7)
candidates inside the margin pool: Qwen3 316, Mixtral no BOS 125 name pairs (before the margin on the swapped prompt).

## WinoGrande experiments (W1–W8), each on Qwen3, Mixtral without BOS (primary) and Mixtral with BOS (variant)

Case set (decision (g)): seed-0 shuffle of the margin pairs shared by all three protocols (776) → 128 discovery + 128
validation pairs (256 directed cases each), plus a disjoint replication set of the next 256 pairs (stability check, the
Appendix-D analogue); per-model margin pools (Qwen3 2,099; Mixtral 1,109 / 1,151) as a sensitivity set. Unit for expert recurrence and
Spec = directed case (the clean run differs between A→B and B→A); bootstrap CIs resample pairs.

- **W1 Verification (OLMoE, minutes).** Final-position `layer` / `attn_layer` / `block` patches and the option-position
  MoE patch (ext5_subject executor) on WinoGrande pairs against transformers hooks, as `ext6_str_verify.py` /
  `ext6_str_grid_verify.py` do.
- **W2 Final-position layer sweep (Z1, Z6).** Kinds `layer` (MoE), `attn_layer`, `block` at every layer. Outputs: raw and
  drop-normalised curves, peaks, attention share of the positive rescue (Direction-2b definition). First answer to "MoE or
  attention", per layer.
- **W3 Position × layer grid (Z7 sites, Z6, Z9; 6b executor).** Every position from the filled option to the final token,
  every layer, single layer first, window 5 second; kinds `layer`, `attn_layer`, `resid` at p (supported by
  `ext5_subject`). Token groups: option tokens, tokens between option and final position, final position. Positions before
  the option are exactly zero.
- **W4 Joint sublayer decomposition at the final position (engine extension, see A0).** Patch all MoE outputs (all layers),
  all attention outputs (all layers), and both (= the clean final residual, must give normalised rescue 1: sanity check).
  With A = all-attention share and M = all-MoE share, the two-player Shapley split is φ_attn = ½[A + (1 − M)], φ_MoE = 1 − φ_attn;
  A + M − 1 measures redundancy. Same in the noising direction (corrupt all attention / all MoE in the clean run: necessity).
  This is the direct answer to "how much repair is attributable to MoE vs attention".
  **Revised 2026-10-04 (found by ext8-addback / ext7-controls):** the joint game is degenerate at the final position. The
  final token is shared and the MoE is a per-token function, so patching all final-position attention outputs restores the
  clean residual exactly (A = 1 by construction). W4 is therefore reported as: A = 1 (engine sanity check only); M = all-MoE
  denoising and all-MoE noising (sufficiency / necessity of the MoE computation given the context attention delivers); and
  the direct-path split of the clean − corrupted Δ into summed attention writes and summed MoE writes at the final position
  (exact final RMSNorm; `moetrace/ext7_controls.direct_split_pairs` = `scripts/ext8_addback_run.py` direct_split), plus the
  single-layer attention share of W2. Attention is, by construction, the only channel through which the corruption reaches
  the final position; the question that remains is how much of the work the MoE does with what attention delivers.
- **W5 Attention heads (`attn_head`) at the W2 attention peaks.** Detection by Z8 (≥ 2 SD) and by Spec with CIs; attention
  mass of the top heads on the filled option, the two candidates' first mentions and the context words, clean vs corrupt
  (mover heads?). KL as an auxiliary metric for discovery only (Z5).
- **W6 Experts.** Paper two-stage selection (MoE-layer argmax → recurrence-first expert, Spec, active controls, equal-norm
  check), joint (layer, expert) search, interior-layer rule; then the add-back curves A1–A3 on WinoGrande. Pattern A/B/C as in
  Direction 2.
- **W7 Second corruption site (Z7).** Entity-role swap: exchange the two candidates' first mentions, keep the filled option
  (A: "Dennis helped Adam … since Dennis was the" → " trainer"; swap: "Adam helped Dennis … since Dennis was the" →
  " student"). Name pairs only (object swaps are often not natural), token-aligned and inside the margin pool (Qwen3 316,
  Mixtral no BOS 125), margin ≥ 1 toward r′ on the swapped prompt; Mixtral is too small for a 128/128 split, so W7 there is a
  fixed-hypothesis test of the W2/W4 results rather than a new selection. Repeat W2 and W4. The option swap tests "which entity is referred to";
  the role swap tests "which entity has which attribute" (binding), the closest analogue of IOI's S1+IO corruption.
- **W8 Calibration tasks on the same models.** IOI (Wang et al. templates, single-token names per tokenizer; STR S2→IO and
  S1+IO→random names, as Z&N) as the attention-driven reference; CounterFact STR (ext6 runs) as the other pole. W2 + W4 on
  both, so "WinoGrande is IOI-like" becomes a position on one attention–MoE axis measured with one protocol.

## Add-back curves (A0–A4): CounterFact STR first (ext6 runs exist), then WinoGrande (W6)

- **A0 Ceilings.** All MoE outputs at the final position (engine kind `multi` with `layer` steps at every layer: exists,
  verified on OLMoE for 2–3 steps, needs an all-layer check); all attention outputs (needs `attn_layer` steps in `multi`:
  small engine extension + OLMoE verification); both = 1. If the all-MoE ceiling is far below 1, "full repair from experts
  alone" is impossible for many cases, and the curves are reported against both the ceiling and the drop.
- **A1 Static orderings over (layer, expert) pairs, ignoring layers.** Exact joint patches (`multi` + `coalition_set`, one
  step per layer) of the top-k set for k = 1…all on a log-spaced grid: (i) population ranking by discovery rescue, evaluated
  on validation; (ii) per-case oracle ranking by the case's own single-expert rescue; (iii) layer-wise baseline (the paper's
  way: best layer's experts first); (iv) random orders (null curve; their average marginal gains estimate Shapley values);
  (v) patch-free proxies: routing weight, ‖δ_e‖, direct logit attribution of δ_e on W_U[r] − W_U[r′].
- **A2 Better strategies.** Adaptive greedy (re-rank the remaining candidates given the current set; candidate pool = top 32
  per case), beam search (width 4), exact optimum within the per-case top-10 pool (all 1,023 subsets) to measure greedy's
  optimality gap; gradient rankings (attribution patching, AtP*, EAP-IG) once F3 exists (pilot on OLMoE with autograd).
- **A3 Deletion curves (noising, Z10).** Start from the clean run and swap in corrupt expert outputs in the same orders.
- **Metrics.** r(k) = (Δ_k − Δ_corrupt) / (Δ_clean − Δ_corrupt) and the gap 1 − r(k); AUC over log k; k for 50 / 80 / 90 % of
  the ceiling; k at which the answer is restored (Δ_k > 0, natural under STR because the corrupted run prefers r′); k at
  which r becomes top-1 (secondary).
- **Cost.** Qwen3 384 clean-active (layer, expert) pairs per directed case, Mixtral 64. Static curves ≈ 1 pass per ordering
  per model; adaptive greedy ≈ 15 sequential passes; exact top-10 optimum ≈ 5 passes (Qwen3, one direction).

## Execution order, cost, decisions

1. W1 verification (OLMoE) → W2 sweeps (3 protocols, ≈ 5 min each) — answers "MoE or attention" per layer.
2. Engine: `attn_layer` steps in `multi` + all-layer `multi` check on OLMoE (≈ half a day) → W4 and A0 on WinoGrande and
   CounterFact (1 pass per model each).
3. W3 grid (≈ 10 min per protocol), W5 heads (1 pass), W6 experts (≈ 15 min per protocol).
4. A1–A3 on CounterFact STR, then on WinoGrande (≈ 30–40 min per model incl. adaptive greedy).
5. W7 role swap, W8 IOI (data builder + W2/W4 runs).
Total GPU ≈ 3–4 h; agent time several days; file prefix `ext7_wino_*` (WinoGrande) and `ext8_addback_*` (add-back).

Decisions (user, 2026-10-04):
- (a) STR only: no GN filter, rows or control. Primary pair set = margin both ways.
- (b) `assoc` pairs kept, reported as a stratum.
- (c) top-1 in both directions = sensitivity subset only.
- (d) W7 (role swap) and W8 (IOI) both in scope.
- (e) **Mixtral = BOS (tokenizer default) for all Phase-3 work**; Mixtral without BOS is a variant that is NOT run now.
  Phase-3 models: Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS; OLMoE for verification only.
- (f) Both the add-back study (A0–A3: CounterFact STR first, then WinoGrande) and the WinoGrande study (W1–W8) must be
  completed; order free.
- (g) Case set = margin pairs shared by Qwen3, Mixtral BOS and Mixtral no BOS (776; keeps a later no-BOS run on identical
  items): seed-0 shuffle → 128 / 128 discovery / validation + 128 / 128 replication (`data/wino_str/case_sets.json`, built by
  `scripts/ext7_wino_casesets.py`); each model's own margin pool (seed-1 sample 128 / 128) as sensitivity.

## Phase 3 execution plan (started 2026-10-04)

Coordinator = the interactive session (relays dependencies, reviews, assembles `EXTENSIONS_REPORT.md`, writes the synthesis
section `results/sections/ext7_synthesis.md`). Three sub-agents with disjoint files; GPU only through `scripts/gpu_queue.sh`.

| Agent | Owns (files) | Work |
|---|---|---|
| `ext8-addback` | `moetrace/engine.py` (additive `multi` step kinds only), `moetrace/ext8_addback.py`, `scripts/ext8_*`, `results/*_addback*`, `results/verify_ext8_*`, section `ext8_addback.md` | E: `attn_layer` / `block` steps in `multi`, noising direction, all-layer check, OLMoE verification (others wait for "ENGINE READY"); A0–A3 on CounterFact STR (Qwen3, Mixtral BOS) incl. the CounterFact W4 decomposition; then A0–A3 on WinoGrande from `ext7-wino`'s all-layer expert rows |
| `ext7-wino` | `moetrace/ext7_wino.py`, `moetrace/ext7_pairs.py` (generic STR-pair runner), `scripts/ext7_wino_*`, `results/wino_*_str*`, `results/verify_ext7_*`, section `ext7_wino.md` | W1, W2, W3, W5, W6 (all-layer expert rows in the ext6 schema for ext8), W4 after ENGINE READY; strata, replication, own-pool sensitivity |
| `ext7-controls` | `moetrace/ext7_controls.py`, `scripts/ext7_role_*`, `scripts/ext7_ioi_*`, `scripts/ext7_cf_*`, `data/wino_role/`, `data/ioi/`, `results/wino_role_*`, `results/ioi_*`, `results/*_str_attnsweep`, section `ext7_controls.md` | W7 role-swap pairs + runs; W8 IOI dataset + W2 / W4 / heads; CounterFact STR attention / block sweep (W2 for the three-task comparison) |

Contract between `ext7-wino` and `ext7-controls`: pair parquet with `pair_id`, `ids_a`, `ids_b` (JSON), `str_pos` (JSON: positions
where A and B differ), `trig_a`, `trig_b`, a case-set JSON in the format of `data/wino_str/case_sets.json`; the `ext7_wino`
run scripts take `--pairs`, `--case-sets`, `--out`.

**Phase 3 status (2026-10-04 11:30 UTC): COMPLETE.** All three agents delivered (ext8-addback, ext7-wino, ext7-controls);
sections `ext7_synthesis.md`, `ext7_wino.md`, `ext7_controls.md`, `ext8_addback.md` assembled into `results/EXTENSIONS_REPORT.md`
(Directions 7-8, 7, 7b, 8). GPU ≈ 3.9 h. Deviations: W4 revised (degenerate joint game); W3 grids for role swap / IOI on 64
validation pairs; gradient rankings not done (F3). Not committed.

# Phase 4 (started 2026-10-05): necessity, full-repair circuits, writer experts, completeness

User decision 2026-10-05: commit and push Phase 3 (done, 81377e4), then execute items 1-4 of the coordinator's
recommendation, in this order of priority. Not in scope now: item 5 (two-site add-back, F3), large random-subset sampling,
exhaustive top-12/14 subsets (the top-10 optimum already beats greedy by ≤ 0.011, at the bf16 noise floor).
Models: Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS (Phase-3 default); Mixtral without BOS only in item 4c. STR only for
every patching experiment. Zhang & Nanda checklist Z1-Z10 and the robustness criteria above still apply; knockout is the
necessity side (Z10) and uses clean prompts only, so it needs no corruption.

## Engine E4 (owner: ext9-knockout; additive, behind new fields; verified on OLMoE vs transformers hooks)

- **E4a route mask** (`PrefillSpec.route_mask` = ((layer, expert), ...), `route_mask_pos` = "all" | "final",
  `route_mask_mode` = "reroute" | "zero"). reroute (primary, MoE-native): the masked experts' router probabilities are set
  to 0 after the softmax, top-k is taken over the remaining experts and the model's own renormalisation rule applies, so the
  token still uses k experts. zero (sensitivity): routing unchanged, the masked experts' contributions c_e set to 0.
- **E4b `attn_head` steps in `multi`** (first and later steps; a step may carry heads AND experts at the same layer: heads
  patched before the MoE, the row's own MoE recomputed on the head-patched input, then the listed experts' contributions set
  to the source's). Checks: one head step = `attn_head` kind; all heads of a layer = `attn_layer` step (to o_proj bf16
  rounding); all heads at all layers = the clean Δ; random mixed sets vs hooks.
- **E4c `DiagSpec.contrib_final_vectors`** (layers): per-slot expert contribution vectors at the final position of the
  prefill rows (for vocabulary projections).
- Regression: `verify_olmoe.json`, `verify_ext5_engine_olmoe.json`, `verify_ext8_engine_olmoe.json` identical in every
  non-timing field to `*_before_ext9.json` copies. Marker: PROGRESS line "ENGINE READY (ext9)".

## Item 1 / Direction 9: expert knockout, necessity and task specificity (ext9-knockout)

Question: are the experts the patches find necessary for the behaviour, and only for their own task? Targets: Qwen3
WinoGrande L41E117, CounterFact L44E069 and L42E115; Mixtral WinoGrande L20E000, CounterFact L19E002, L21E001, L18E001.
Plus jointly the top-1/3/5/10 experts of each task's ext8 population ranking (discovery means).
Controls: same-layer experts (Mixtral all 7 others; Qwen3 the 16 most often routed at the final position on the evaluation
items, excluding targets) and, for the sets, 5 random equal-size sets matched on final-position routing frequency.
Evaluation (clean prompts; items used to select an expert are excluded): WinoGrande own margin pool (Qwen3 2,099, Mixtral BOS
1,151 pairs, minus the 128 main discovery pairs; both prompts of a pair), plus all W1-W6 pairs (4,514 / 2,421) for
baseline, targets and sets; CounterFact clean scan (`results/{qwen3,mixtral}/filter_scan.parquet`, Δ_clean ≥ 1: 867 / 941,
minus paper discovery IDs); IOI clean prompts (`data/ioi`, third task); wikitext-103 windows (per-token NLL, generic damage).
Metrics: fraction of the clean margin lost (mean Δ change / mean baseline Δ), accuracy (Δ > 0; WinoGrande pair accuracy),
p(true) and rank descriptive; effect vs the control distribution (rank, z); double-dissociation contrast
(WG expert on WG − on CF) − (CF expert on WG − on CF) with pair/item bootstrap CIs. Mask at all positions (primary), final
position only (secondary, links to the final-position patches); reroute primary, zero for the targets.

## Item 2 / Direction 10: head + expert add-back to full repair at the final position (ext10-circuit)

Experts alone cap at the all-MoE ceiling (CounterFact 0.53 / 0.41); all attention outputs restore 1.0, so "how few
components restore the answer" has a real ceiling once heads are candidates. Tasks: CounterFact STR and WinoGrande option
swap (the ext8 validation rows), IOI (i) as the attention-pole reference if the budget allows.
Step 1 (existing kinds): single-head patches (`attn_head`) at the final position for every head of every layer, plus a
patch-free head DLA (per-head outputs projected with the final norm frozen). Step 2 (after E4b): adaptive greedy, 20 steps,
over pool = top-32 heads ∪ the ext8 expert pool (Qwen3 top-32, Mixtral all 64), and head-only greedy (pool 32); static mixed
orderings (single-patch oracle, DLA). Metrics: r(k) as a fraction of the drop (ceiling 1), k for 50 / 80 / 90 %, case-level
answer restored (Δ > 0) and r ≥ 0.9 at k = 20, composition (# heads vs # experts among the first k picks), recurring
components (known mover heads: Qwen3 L40H13, WG W5 heads, IOI heads).

## Item 3 / Direction 11: writer vs computer experts (ext11-writer)

DLA ≈ oracle suggests the top experts write the answer directly. A (CPU, existing ext8 rows): per (case, expert) total =
single-expert patch, direct = DLA, indirect = total − direct; by layer, task, model, for the selected experts and the
population top-10. B (after E4c): vocabulary projection of δ_e and of c_e_clean (final norm frozen): ranks of r and r′,
top promoted / suppressed tokens, token classes (WinoGrande trigger vocabulary, CounterFact object vocabulary per relation,
other). C (routing diagnostics, existing DiagSpec): where else the target experts fire at the final position:
WinoGrande prompts, the context-free "local" prompts ("The bag was too"), CounterFact, IOI, wikitext tokens (contexts and
next tokens at which they are routed), and their DLA there.

## Item 4 / Direction 12: completeness checks (ext12-complete)

- 4a Add-back replication: WinoGrande rep_validation (population ranking from rep_discovery) and a CounterFact fold swap
  (ranking from validation, curves on discovery), static orderings + greedy, Qwen3 and Mixtral BOS.
- 4b Role swap vs option swap on identical items: the option-swap W2 sweep, all-MoE M and direct split on the role-swap
  items (all lie inside the option-swap margin pool); paired comparison of the attention share.
- 4c Mixtral without BOS on WinoGrande, same 776-pair case set: W2, W6 two-stage selection + Spec, W4 all-MoE + direct
  split, add-back static + greedy, sink-carrying flags (Direction 3).

## Phase 4 execution plan

| Agent | Owns | Work |
|---|---|---|
| `ext9-knockout` | `moetrace/engine.py` (E4 only, additive), `moetrace/ext9_knockout.py`, `scripts/ext9_*`, `results/*_knockout*`, `results/verify_ext9_*`, section `ext9_knockout.md` | E4a-c + verification first (reports, coordinator relays ENGINE READY), then item 1 |
| `ext10-circuit` | `moetrace/ext10_circuit.py`, `scripts/ext10_*`, `results/*_circuit*`, section `ext10_circuit.md` | step 1 now, step 2 after E4b |
| `ext11-writer` | `moetrace/ext11_writer.py`, `scripts/ext11_*`, `results/*_writer*`, section `ext11_writer.md` | A and C now, B after E4c |
| `ext12-complete` | `scripts/ext12_*`, `moetrace/ext12_*.py`, additive backward-compatible flags in `scripts/ext7_*` / `scripts/ext8_*`, `results/*_rep*`, `results/wino_mixtral_nobos_str*`, `results/wino_roleitems_*`, section `ext12_complete.md` | 4a, 4b, 4c (no engine change) |

`moetrace/ext8_addback.py`, `moetrace/ext7_pairs.py`, `moetrace/ext7_wino.py`, `moetrace/ext7_controls.py` are read-only for
everyone (bug fixes through the coordinator). Coordinator: this plan, CLAUDE.md, README, `scripts/build_extensions_report.py`,
synthesis section `ext9_synthesis.md` (Directions 9-12). GPU only through `scripts/gpu_queue.sh`; budget ≈ 5 GPU h.

**Phase 4 status (2026-10-05): COMPLETE.** All four agents delivered; sections `ext9_synthesis.md` (coordinator),
`ext9_knockout.md`, `ext10_circuit.md`, `ext11_writer.md`, `ext12_complete.md` assembled into `results/EXTENSIONS_REPORT.md`
(Directions 9-12, 9, 10, 11, 12). GPU ≈ 4.4 h. Deviations: reroute implemented as router logit −inf before the softmax (identical
to "probability 0 after the softmax" for Qwen3 / Mixtral, documented for OLMoE); final-only and zero-mode knockouts on the fixed
subsample; IOI circuit run for Qwen3 only; Qwen3 same-layer controls = the 16 experts most often routed at the final position on
the target task's items. Revision of a Phase-3 claim (4b): on identical items role swap = option swap; the Phase-3 gap was names
vs objects (texts in ext7_controls.md / ext7_synthesis.md / CLAUDE.md corrected). Coordinator addition: knockout vs direct write
on the same prompts (`scripts/ext9_synthesis_selfrepair.py`): 53–95 % of a writer expert's direct write is compensated.
Committed and pushed 2026-10-06.

# Phase 5 (DRAFT 2026-10-11, from the team meeting on the Phase 3/4 results; design decisions pending)

Meeting feedback, in short. The WinoGrande option swap (fill the blank, predict the sentence-final attribute) tests
entity → attribute: knowledge recall, where the entity's identity flows from the noun position to the final token and
is converted into an attribute. The original WinoGrande task is attribute → one of the two entities: reference
resolution. The team asked for the following.
- (a) Reformulate WinoGrande so that it tests reference resolution, e.g. as a two-option question.
- (b) Balance the noun–attribute association.
- (c) Control the role-swap (name) items with neutral and unrelated verbs (memorization) and with synonyms.
- (d) Build a data-centric task ladder that raises the reliance on attention step by step, and a subset that needs
  attention to be solved.
- (e) Explain why the add-back curves of CounterFact and WinoGrande differ.
- (f) Move from activation patching to attribution patching.
- (g) Discuss scale and MIB.

## P5-1 WinoGrande in the original direction: attribute → entity (core)

STR = the twin's trigger swap (dirty ↔ weak); the readout is the entity:

    Δ = logit(entity_r) − logit(entity_r′) at the final token

Both directions of every pair are used. Candidate formats (to be decided with the user):

| Format | Prompt (clean; the corrupted prompt swaps the trigger) | Pairs (single-token entities, token-aligned, before margin) |
|---|---|---|
| F1 cloze (recommended primary) | `Fill in the blank: The janitor tried to clean the mirror with the spray but the _ was too dirty.\nAnswer: the` → " mirror" | Qwen3 10,875 / Mixtral 5,213 |
| F2 natural truncation | twins whose trigger precedes the blank, cut at the blank: `The cake was sweeter than the pie because there was more sugar used in the` → " cake" | 1,162 / 578 |
| F3 letter MC (A/B) | `…\nWho or what is _?\nA. the mirror\nB. the spray\nAnswer:` → " A"; both option orders | all twins (options need not be single tokens) |
| F4 pronoun + question | `…but it was too dirty. What was too dirty? The` → " mirror" | needs per-item rewriting, and pronoun gender leaks for names |

Recommendation:
- **F1 as primary.** It is the WinoGrande task itself, with no rewriting. The answer is a context entity, as in IOI.
- **F3 as secondary.** It is the team's A/B format and links to MIB's MCQA task, but it adds a letter-binding step.
- **F2 as a naturalness check.** It is the only format with no instruction wrapper.

Filters:
- margin ≥ 1 both ways;
- W6 dedup;
- dev twins and the AfLite (train_debiased) subset as a memorization check.

Association balance:
- Per pair, a context-free association score: Δ on `Fill in the blank: Between the mirror and the spray, the _ was too dirty.`
- Main set = the balanced pairs. All association strata are reported, since association strength is the continuous
  variable of P5-2.

Analyses (the existing generic pair runners):
- W2 sweep and W4 (all-MoE + direct split);
- W3 grid at the trigger, the blank, the two candidate mentions and the final token;
- W5 heads (do heads attend to the correct candidate's mention?);
- W6 experts;
- head + expert add-back (Direction-10 machinery).

Prediction: much more attention than the option swap, towards IOI.

## P5-2 Task ladder: a data-centric attention gradient

Ladder: L1 CounterFact → L2 WinoGrande entity → attribute, objects → L3 the same with names → L4 names with a
neutral verb → L5 WinoGrande attribute → entity, objects → L6 attribute → entity, names (and neutral verb) → L7 IOI.

On every level, the same measurements:
- attention share;
- all-MoE M and the direct split;
- heads vs experts among the first greedy add-back picks.

Within levels, the attention share is regressed on the per-pair association score.

"Attention-required subset" = attribute → entity, names, neutral verb, association ≈ 0 (chance-level context-free
Δ). Open question from the meeting: does linguistic theory support such a gradient? Short literature check
(Winograd-schema analyses, reference resolution vs world knowledge).

## P5-3 Name-item controls (role-swap follow-ups)

- **Neutral verb:** replace the content verb (bought → got).
- **Unrelated verb (memorization):** e.g. bought → insulted, from WordNet, checked by hand. If the answer stays
  generous / thankful, the item is memorized or relies on a shortcut.
- **Synonym averaging:** for the attribute (as input in P5-1, as output in the entity → attribute direction).
- **Pool:** name items in the margin pools (Qwen3 535, Mixtral 150); single-token checks as before.

## P5-4 Why the add-back curves differ (mostly CPU, existing rows)

CounterFact overshoots the all-MoE ceiling and falls at the end; WinoGrande barely does. Analyses:
- the mass and layer location of negative single-expert effects;
- (curve maximum − ceiling) against the summed negative experts;
- the interior-layer ceilings (≤ L−5);
- the late-layer negative MoE in W2;
- the writer / computer shares (Direction 11);
- Shapley redundancy;
- one cheap GPU check: all experts except the negative ones.

## P5-5 Attribution patching (F3)

- **Pilot:** OLMoE with autograd (fits on the GPU). AtP, AtP* and EAP-IG against the activation-patching rankings
  of the STR pairs.
- **Then:** if the agreement is good, reverse layer streaming in the engine for Qwen3 / Mixtral (several days of
  engine work).

## P5-6 Scale (discussion)

- **Constraints:** no quantization. The same-family larger model (Qwen3-235B-A22B, about 470 GB in bf16) does not
  fit on the NVMe (49 GB free); it needs a larger disk or instance.
- **Possible now:** a confounded three-point series OLMoE-1B-7B (1.3B active) / Qwen3-30B-A3B (3B active) /
  Mixtral-8x7B (13B active). Running the ladder on OLMoE is cheap.
- **Confound noted by the team:** expert size grows with the model.

## P5-7 MIB (discussion)

MIB (Mueller et al., ICML 2025, arXiv:2504.13151): circuit localization and causal-variable localization tracks,
four tasks (including MCQA), dense models. Our possible contributions:
- MoE models with expert-level components;
- STR pairs;
- the WinoGrande reference-resolution task in an MCQA-style format (F3);
- head + expert add-back curves as a circuit-localization measure.

## Order and cost (estimates)

1. P5-1 data build and competence scans (≈ 20 GPU-min), then its analyses (≈ 1–1.5 GPU-h).
2. P5-3 controls (≈ 0.5 GPU-h).
3. P5-2 ladder (mostly re-use, plus ≈ 1 GPU-h).
4. P5-4 at any time (CPU).
5. P5-5 pilot.

P5-6 and P5-7 are for discussion.

## Phase 5 decisions (user, 2026-10-11)

- **Format.** F2 natural truncation is the PRIMARY attribute → entity format. F1 fill-in-the-blank runs as the
  comparison experiment, because F2's items share one sentence structure (the trigger precedes the blank) and F1
  covers the other question types. F3 (letter MC) and F4 (pronoun + question) are not run.
  **Superseded the same day (see "Phase 5 revision" below):** the cut at the blank drops the words after it, so F2 is
  replaced by F2R (restate, primary) and F2E (blank at the sentence end, comparison).
- **Association.** A balanced main set plus all association strata. The context-free association score is the
  ladder's continuous variable.
- **Additional experiment.** Only the name-item verb swaps (P5-3): a neutral verb (bought → got) and an unrelated verb
  (bought → insulted, memorization check). Synonym averaging is not run.
- **Models.** Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS, as in Phases 3–4.

Candidate pool before the margin filter (single-token entities, the twins token-aligned with ≤ 3 differing tokens):

| Format | Qwen3 pairs (of which names) | Mixtral BOS pairs (of which names) |
|---|---|---|
| F2 | 1,114 (734) | 555 (338) |
| F1 | 10,875 (5,985) | 5,213 (2,324) |

F2 is mostly name items, the opposite of the Phase-3 option swap (91 % objects), so names vs objects is a mandatory
stratum.

Risk: after the margin, dedup and balance filters, F2 Mixtral may have fewer than 256 pairs (128 / 128). Decide after
the competence scan: smaller splits, or a looser balance threshold.

Open details, with the coordinator's proposal:
- **Association probe:** the trigger word and the two candidates in a neutral frame that works for any part of speech,
  e.g. `Word: dirty. Which fits better, the mirror or the spray? Answer: the`, both candidate orders averaged.
- **Balanced:** context-free association drop ≤ 25 % of the full STR drop. This is the same quantity as the Phase-3
  local baseline, where objects reached 15–17 % and names 1 %.
- **Verb swaps:** the verb between the two names (`X <verb> Y …`), found by rule. Replacements must keep the token count
  so the pairs stay token-aligned. Every swapped item is reviewed by hand (a few hundred items). The swaps run on name
  items in both directions (ladder levels L3/L4 and L6).

F2 trigger types (1,538 equal-word-length twins whose only difference precedes the blank):
- 23 % swap a connective (because ↔ although / but): a discourse relation → entity, not attribute → entity;
- 20 % differ in the last word before the blank, i.e. the final prompt token, so the STR site and the readout
  position coincide (excluded by the W5 rule);
- the rest are comparatives (more / less, better / worse, easier / harder, bigger / smaller), adjectives, verbs
  (hated / loved) and prepositions (to / from).

Proposal:
- F2 main set = content-word swaps that are not at the final token;
- connective swaps as a separate stratum, or dropped.

Rough estimate after this rule and the margin: Qwen3 ≈ 350–400, Mixtral ≈ 180–200 pairs, before the balance filter.
Mixtral F2 is therefore likely below 128 / 128.

Association threshold, revised after a literature check (user request, 2026-10-11). The 25 % ratio was an ad-hoc
proposal. Related work defines association / artifacts per item with a binary partial-input decision, never a ratio:
- Trichelair et al. 2019 (EMNLP), Winograd "associative" vs "non-associative": one candidate is strongly linked to the
  pronoun clause and the other is not; three annotators had to agree (κ 0.79). Levesque's "Google-proof" design goal.
- Gururangan et al. 2018 / Poliak et al. 2018: "hard" examples = those a hypothesis-only (partial-input) model gets
  wrong.
- Poerner, Waltinger & Schütze 2020 (LAMA-UHN): facts guessable from the entity name are removed.
- Elazar et al. 2021 (EMNLP): partial-input baselines for Winograd and paired (twin) scoring.
- Gardner et al. 2021: a per-feature z-test (α = 0.01, Bonferroni), at the dataset level rather than per item.

Proposed rule, consistent with these works and with our own competence margin:
- **Associative pair:** the context-free probe gets BOTH twins right with a margin ≥ 1 logit. This is our Phase-3
  `assoc` flag and Elazar-style paired scoring.
- **Main (balanced) set:** the non-associative pairs.
- **Strict "hard" subset (sensitivity):** the probe gets neither twin right (Gururangan-style).
- **The continuous ratio** (probe drop / full drop) is kept only as the ladder's covariate, not as a threshold.

Phase 5 build (2026-10-11): `moetrace/ext13_wino_ref.py`, `scripts/ext13_build.py` → `data/wino_ref/pairs_{F2,F1}_train_xl_<proto>.parquet`.

| | F2 main (names) | F2 connective | F1 main (names) | F1 connective |
|---|---|---|---|---|
| Qwen3 | 797 (459) | 43 | 10,236 (5,472) | 446 |
| Mixtral BOS | 405 (213) | 21 | 4,883 (2,081) | 187 |

Notes:
- Most connective swaps sat right before the blank and were removed by R5.
- Verb-swap pool: `scripts/ext13_verbs.py suggest` finds only ≈ 128 loosely interpersonal name items (26 attribute →
  entity, 102 entity → attribute), and many of them are comparatives rather than "X verb Y".
- Same-token-count rule: not needed for STR alignment, because both twins get the same replacement. To be kept as a
  flag only, pending the user.

## Phase 5 revision (2026-10-11): information completeness, F2R / F2E, execution order

**Correction (raised by the user).** F2 cut the sentence at the blank and dropped everything after it. In 69 % of the
Qwen3 F2 main pairs (551 / 797; names 94 %, objects 35 %) the words after the blank decide the answer:
`Natalie was more tired than Betty because _ had been running on the treadmill all day.` cut to `… than Betty because`
leaves both names possible. With the blank at the sentence end, names and objects pass the margin equally (Qwen3 0.44 /
0.51, Mixtral 0.50 / 0.50); with words after it, 0.05 / 0.31. The earlier reading "the models ignore the trigger on
name items" is withdrawn (truncation artefact).

**Rule for every Phase-5 prompt:** the model sees the same information as in the WinoGrande sentence; only the answer
is withheld.

**User decisions (2026-10-11):** F2's primary format = F2R (restate); F2E (natural truncation, blank at the sentence end)
runs alongside as the comparison within F2; F1 stays the comparison format; execution starts with F1.

| Format | Prompt | Built (Qwen3 / Mixtral BOS) |
|---|---|---|
| F2R restate (primary F2) | `Natalie was more tired than Betty because one of them had been running on the treadmill all day. The one who had been running on the treadmill all day was` → " Natalie" | kept 646 / 291, main 405 / 181 (names 371 / 160), connective 241 / 110 |
| F2E (F2, blank at the end) | `The cake was sweeter than the pie because there was more sugar used in the` → " cake" | main 246 / 141 (names 27 / 12); scanned with F2 |
| F1 (comparison) | `Fill in the blank: <sentence>\nAnswer: the` → " mirror" | main 10,236 / 4,883 |

F2R rules (R7, `moetrace/ext13_wino_ref.restate_prompts`): F2 geometry (twins differ before the blank, identical from
it on); the blank is the subject of the following verb (optional adverbs skipped); not inside a prepositional phrase
(`the speakers on the _ were bigger`: the predicate is about the PP's head noun); not plural (`one of them were`); not
possessive; predicate ≥ 2 words; a determiner before the blank is dropped (`the _` → `one of them`); copula `is` for
present-tense predicates, else `was`; objects end in `was the`. Checked by hand on samples of names and objects. F2R and
F2E are complementary: F2R is mostly names (objects usually have the blank at the sentence end), F2E mostly objects.

**Completeness audit of every Phase-5 item:**

| Item | Prompt the model sees | Complete? |
|---|---|---|
| P5-1 F1 | `Fill in the blank: ` + the whole sentence + `Answer:` | yes (10,682 / 10,682 prompts contain the full sentence) |
| P5-1 F2R | whole sentence (blank → `one of them`) + restated predicate | yes, by construction |
| P5-1 F2E | the sentence up to the blank, nothing after it | yes (only punctuation after the blank) |
| P5-1 F2 (cut) | sentence up to the blank | **no; void** (files kept for the record) |
| Association probe | trigger word + both candidates, no sentence | deliberately partial (partial-input baseline, not a task) |
| P5-2 L1 CounterFact | fact prompt | yes |
| P5-2 L2 / L3 WinoGrande entity → attribute (Phase 3) | sentence with the blank filled, up to the final trigger | yes (prompt + trigger = sentence, 4,514 / 4,514) |
| P5-2 L4 / P5-3 verb swaps entity → attribute (55 seeds) | Phase-3 builder | yes |
| P5-2 L5 / L6 attribute → entity | F2R / F2E / F1 | yes |
| P5-3 verb swaps attribute → entity (13 seeds) | was F2 cut (all 13 affected) → rebuilt as F2R | yes after the rebuild; old files in `data/wino_ref/void_F2cut/`, old scan renamed `VOID_F2cut_*` |
| P5-2 L7 IOI | IOI prompt | yes |
| P5-4 | existing Phase-3 / 4 rows | yes |
| P5-5 pilot | pairs of the formats above | yes |

Verb-swap seeds under F2R: 138 attribute → entity seeds pass the token rules (27 under the cut), but the new ones are
almost all connective swaps (`because` ↔ `but`: "X asked Y … because / but _ could not remember"), i.e. the connective
stratum; the 68 curated seeds (55 + 13) all stay eligible, edits unchanged.

**Execution order** (GPU through `gpu_queue.sh`; each step's sample sizes reported before splits are fixed):
1. **F1** (started 2026-10-11): full-pool scan (Qwen3 balanced 1,397 pairs, names 520) → per-model case sets
   (`scripts/ext13_casesets.py`: 128 / 128 + replication 128 / 128 where the pool allows) → P5-1 analyses with the
   Phase-3 runners: W2 sweep, W4 joint + DLA + direct split, W6 experts, W5 heads (mention positions from
   `scripts/ext13_positions.py`), W3 grid; then head + expert add-back (Direction-10 machinery; task registry entry).
   Qwen3 first, Mixtral BOS second.
2. **F2R + F2E**: F2R scan (+ verb attribute → entity rescan, F2R) → sample sizes → split decision (user) → the
   analyses of step 1.
3. **P5-3 verb swaps**: entity → attribute scan done (7,920 pairs per model); seed-grouped case sets (discovery and
   validation never share a template) → W2 / W4 / DLA for orig / neutral / unrelated; attribute → entity after step 2.
4. **P5-2 ladder**: levels from steps 1–3 + the Phase-3 runs (CounterFact, WinoGrande objects / names, IOI); attention
   share regressed on the association score.
5. **P5-4** (CPU, any time while the GPU is busy).
6. **P5-5** attribution-patching pilot (OLMoE).


## Phase 5 review (2026-10-11, second session): coverage of the meeting notes, setup issues, P5-8

Numbers marked *prelim.* are quick coordinator calculations on the finished F1 runs (validation directed cases; attention
share = AUC+ of the W2 curves; direct attention = all-attention direct path / drop), not the section's final analysis.
Recomputing Phase 3 with the same code gives 0.16 / 0.34 and 0.05 / 0.29, as reported.

### Coverage of the meeting notes

| Meeting point | Plan item | Status / gap |
|---|---|---|
| Option swap ≠ original task; test reference resolution | P5-1 (F1, F2R, F2E) | covered; F1 GPU runs done for both models. *Prelim.* attention share F1 0.76 / 0.54 vs option swap 0.16 / 0.34; direct attention 1.13 / 0.93 vs 0.05 / 0.29 (IOI 1.40 / 1.17) |
| Two-option (A/B) question | F3 | not run (user decision). F1 / F2R read the entity itself (Δ is a two-way choice). The MIB-MCQA link is lost |
| Adjective equally associated with both nouns | association probe, binary rule | covered for P5-1. Not yet applied to the Phase-3 option-swap items (ladder L2): re-analyse them by their `assoc` flag (CPU) |
| Better WinoGrande filter, proper nouns | names / objects strata | covered, but the strata are unbalanced (issue 2) |
| Role swap as MC: "The person that felt generous is ___" | verb swaps attr2ent (F2R) | not viable at the margin rule (issue 1); fix 1b (F1 on the verb templates) |
| Average across synonyms | — | not run (user decision) |
| Neutral verb (bought → got) | P5-3 ent2attr | scanned: margin 715 / 542 pairs (28 / 23 seeds), mean drop 4.25 / 3.91 vs orig 4.97 / 4.71 |
| Unrelated verb (memorization) | P5-3 ent2attr unrelated | done: drop 0.22 / 0.00, so the models read the verb. Tested on new IOI names only (issue 6) |
| Data-centric ladder | P5-2 | planned. Levels change several factors at once (issue 4), hence P5-8 |
| Linguistic theory | P5-2 literature check | planned. Candidates: implicit causality of interpersonal verbs, coherence-driven coreference (both bear on bought → got), FFN as key-value memory vs attention as mover |
| Subset that needs attention | P5-2 definition | partly; P5-8 makes it operational |
| "Less looking, more thinking"; attention vs dense roles | P5-1 W3 grid, W5 heads | the decision site must be read out, not only the final token (issue 5) |
| Combined MoE + attention add-back | Direction 10 (Phase 4) | done for CounterFact / WinoGrande / IOI. To extend to F1 and P5-8 |
| Scale; capability vs share; expert-size confound | P5-6 (discussion) | add a within-model capability proxy (issue 9) |
| MIB contributions | P5-7 (discussion) | — |
| Why the add-back curves differ | P5-4 | planned. Add F1 / P5-8 curves as further shapes. Hypothesis from Phase 4: later layers cancel CounterFact writers (L44E069 direct share 1.32) |
| Activation → attribution patching | P5-5 | planned. MoE routing caveat (issue 8); frame AtP first as an add-back ordering |

### Setup issues (ranked)

**1. F2R (the primary format) and the attribute → entity verb swaps fail the margin rule.**

| Scan | Qwen3 | Mixtral BOS |
|---|---|---|
| F2R balanced pairs (names) | 38 (33) | 16 (14) |
| Verb attr2ent (F2R) margin pairs, orig | 16 (6 seeds) | 8 (4 seeds) |
| F2E balanced pairs (names) | 120 (12) | 71 (6) |
| F1 balanced pairs (names) | 1,397 (520) | 443 (27) |

Proposal (user decision):
- (a) F1 becomes the primary attribute → entity format. F2E is the no-wrapper naturalness check: Mixtral 71 pairs gives 35 / 36 with no replication. F2R is reported as descriptive only.
- (b) Build the attribute → entity verb direction in the F1 format on the 55 ent2attr seeds (blank = the entity, STR = the trigger), e.g. `Fill in the blank: Stephen carried Thomas up four flights of stairs … _ was tired.\nAnswer:` → " Stephen". All 660 test fills (orig + neutral, 6 name pairs per seed) pass the F1 builder under both tokenizers (checked 2026-10-11). This gives both directions on identical semi-synthetic items.

**2. The names / objects composition differs between models and tasks.**
- F1 validation names: Qwen3 50 / 128, Mixtral 6 / 128 (Mixtral balanced names 27 / 443).
- The Phase-3 option swap is 91 % objects.
- Cross-model and cross-task comparisons must therefore be stratified, with objects as the common stratum.
- *Prelim.* F1 attention share: Qwen3 objects 0.72, names 0.81; Mixtral objects 0.53.

**3. Items differ across models and across directions.**
- Each model has its own F1 case sets; only 184 balanced pairs are shared.
- A direction comparison on identical items is possible: re-format the 776 Phase-3 pairs as F1 (twins that end in the trigger). F1 margin: 181 / 189 of the 776.
- *Prelim.* on the overlap of the existing runs (68 / 188 pairs):
  - attention share F1 vs option swap: 0.70 vs 0.21 (Qwen3), 0.51 vs 0.34 (Mixtral);
  - direct attention: 1.12 vs 0.06, 0.83 vs 0.30.
- Action: run F1 on the Phase-3 776 set (W2 / W4 / direct; ≈ 15 GPU-min per model) for a paired comparison, as Direction 12 4b did for role vs option.

**4. The ladder levels confound factors.** Between L2 and L5–L7, the following change together:
- direction (attribute recall vs reference resolution);
- answer in context (copy) vs generated;
- the F1 instruction wrapper;
- item type and prompt length.

IOI, F1 and F2R all copy an in-context entity, so the final-token attention share may mostly index "the answer is in context". P5-8 separates the first two factors.

**5. Decision site vs final token.** In attribute → entity the final position copies the entity, so the final-token share cannot tell "look, then think" from "think, then copy".
- The W3 grid (already run for F1, layer level) is the primary readout of where the decision is made.
- Add expert-level patches at the trigger / blank positions for F1 (an F4-style site analysis, expert pass with rec_pos at the trigger).

**6. The memorization test covers new names only.** The unrelated-verb fills use IOI names; the concern is the original WinoGrande items, which are likely in the pretraining data.
- Add the 55 seeds with their original names × {orig, neutral, unrelated} (CPU build, a few GPU-min).
- Report F1 by the `debiased` (AfLite) flag.

**7. Statistics.**
- Verb-template pairs are clustered: 48 name pairs per seed; seeds with margin 33 / 32 (orig) and 28 / 23 (neutral). Bootstrap over seeds and keep the seed-grouped splits; the effective n is about 30.
- The two directed cases of a pair are dependent everywhere, so bootstrap over pairs.

**8. Attribution patching on MoE.**
- AtP linearises around the corrupted run. A patched expert output can switch top-k routing downstream, which the gradient does not see.
- Compare AtP with activation patching separately for experts whose patch changes downstream routing (recorded in the expert passes).
- Choose the pilot task by OLMoE competence (CounterFact STR, WinoGrande scan `results/wino_olmoe`).

**9. Scale.**
- Within-model capability proxy (CPU, existing rows): attention share against the clean margin and item difficulty.
- A same-family MoE series needs a larger disk (Qwen3-235B ≈ 470 GB).
- The dense Qwen3 family (0.6B–32B, same data; attention vs MLP) would need dense support in the engine.

**10. Probe scope.** The association probe uses the trigger word only. Partial-input baselines in the literature (Trichelair et al.) use the clause around the pronoun. Optional second probe: the clause containing the trigger plus the candidates. Low priority.

## P5-8 Copy vs generate control: the answer in context, direction held fixed (added 2026-10-11 at the user's request)

**Question.** Does the final-token attention share measure the reasoning direction (reference resolution) or only whether the answer token is available in context?

**Design.** Entity → attribute pairs as in Phase 3 (STR = the entity at the blank, readout = the trigger). A prefix names both candidate attributes, so the decision is the same as in the option swap but the answer can be copied:

    Of Stephen and Thomas, one was tired and the other was grateful. Stephen carried Thomas up four flights of stairs in
    the beachside condominium. Stephen was  -> " tired"     (B: "... Thomas was" -> " grateful")

- **Prefix rule.** `Of {X} and {Y}, one {frame} {t1} and the other {frame} {t2}.`, where {frame} is the template's text between the blank and the trigger.
  - Objects use `Of the mirror and the spray, one was too dirty and the other was too weak.`, with the determiner as in the sentence.
  - The prefix is identical in both twins, so the STR site, token alignment and readout are unchanged.
- **Counterbalancing.** Both attribute orders (t1, t2) are built as separate pairs and averaged.
- **Checked 2026-10-11** on the 55 ent2attr seeds × 4 name pairs × 2 orders: 440 / 440 pass the Phase-3 builder under both tokenizers, and the answer token occurs in the prefix in 440 / 440.

**Cells.** All three cells use the same items, in two sources: the semi-synthetic verb templates (names; orig and neutral verbs) and the natural Phase-3 pairs (776, names + objects).

| Cell | Direction | Answer | Format | Exists? |
|---|---|---|---|---|
| C1 | entity → attribute | generated | Phase 3 | yes (Phase-3 runs; verb ent2attr scan) |
| C2 | entity → attribute | copyable | Phase 3 + prefix | new (P5-8) |
| C3 | attribute → entity | copyable (entity mention) | F1 | Phase-3 pairs: issue 3; templates: issue 1b |

- **Contrasts.**
  - C2 − C1 = copy availability, with direction and format fixed.
  - C3 − C2 = direction, with copy fixed. The F1 wrapper still differs.
  - Optional wrapper check: C1 / C2 in the F1 wrapper (blank on the trigger, entity filled in).
- **Margin.** The margin rule applies per cell. The main comparison uses the items that pass in every cell (paired); seed-grouped splits for the templates.

**Measurements.** Per cell:
- W2 attention share; W4 all-MoE M and the direct split; W6 experts.
  - Does Phase 3's L41E117 / L20E000 stay selected in C2?
- W5 heads: attention to the in-context attribute token (copy heads) vs to the names.
- Knockout of L41E117 / L20E000 in C2 (E4 route mask): is the expert still necessary when the answer can be copied?
- Optional: head + expert add-back in C2 (Direction 10) and an add-back curve shape for P5-4.

**Readings.**
- C2 ≈ C3: the final-token share mostly indexes answer availability.
- C2 ≈ C1: the attribute is still computed by the MoE even when it is copyable, so the F1 jump is due to direction.
- In between: both factors contribute.
- The "attention-required subset" of P5-2 = C2 / C3 items with the neutral verb and association ≈ 0.

**Cost.** CPU build; scans ≈ 10 GPU-min per model; W2 / W4 / direct / W6 ≈ 25 GPU-min per model and source; W5 ≈ 5; knockout ≈ 10. Total ≈ 1.5–2 GPU-h for both models and both sources.

### Revised execution order (proposal; items 1b, 3, 6 and P5-8 await the user)

1. F1 section (CPU): stratified names / objects, W3 grid as the decision-site readout, W5, W6.
2. Paired direction runs: F1 on the Phase-3 776 set; F1 on the verb templates (orig / neutral). Scans, then W2 / W4 / direct (≈ 1 GPU-h).
3. P5-8 (≈ 1.5–2 GPU-h).
4. Verb-swap ent2attr tracing: W2 / W4 / DLA, orig vs neutral, seed-clustered CIs; original-name memorization check.
5. F2E (small); F2R descriptive.
6. P5-2 ladder assembly (CPU) and literature check.
7. P5-4 (CPU, any time); head + expert add-back on F1 / C2.
8. P5-5 pilot.

## Phase 5 task list and output standard (user decision 2026-10-11 ~05:50: every task gets the full Phase 3/4 output set)

**Full output set per task and model.**
1. Layer curves: single-layer attention / MoE patches.
2. Attention vs MoE bars: all-MoE add-back, deletion, MoE ≤ L−5, direct split.
3. Position × layer grid.
4. Per-layer DLA.
5. Heads.
6. Expert selection: two-stage, Spec, equal norm.
7. Expert add-back / deletion curves: greedy / oracle / random.
8. Expert knockout: selected expert, population top-10 set, matched random sets. Use the top-10 set if there is no specific expert.
9. Head + expert add-back.
10. Writer analysis: direct share, vocabulary projection.
11. Replication, where the pool allows.

Estimate ≈ 16–20 GPU-h.

**Tasks.** Plain names; the formats are as defined above.

| # | Task | Items |
|---|---|---|
| 1 | Attribute → entity, fill-in-the-blank format | Large per-model pool |
| 2 | Direction on identical items | The 776 Phase-3 pairs in both directions; fill-in-the-blank margin 181 / 189 |
| 3 | Attribute → entity, natural sentence | Blank at the sentence end |
| 4 | Name-item verb control, entity → attribute | Templates, orig / neutral; the unrelated verb is behavioural only |
| 5 | Name-item verb control, attribute → entity | Fill-in-the-blank format on the 55 ent2attr templates; replaces the restated version (16 / 8 pairs) |
| 6 | Copy vs generate | P5-8; templates orig / neutral, plus the natural Phase-3 pairs |
| 7 | Original-name memorization check | Behavioural only |

Analyses:
- why the add-back curves differ;
- the task ladder;
- an attribution-patching pilot on OLMoE.

The restated format is reported as behaviour only.

**Information completeness, re-checked 2026-10-11 ~05:50.**
- Natural sentence (blank at the end): the only text after the blank is "." (514 / 514 Qwen3, 288 / 288 Mixtral prompts), and the prompt equals the sentence up to the blank in every case.
- Fill-in-the-blank: the full sentence is in the prompt (10,682 / 10,682 and 5,070 / 5,070).
- Templates and copy variant: prompt + trigger = the full sentence (Phase-3 builder); the copy variant only adds a prefix.

**Association probe check (before switching the main set to "both trigger words neutral", |probe| < 1).**
- On the same 2,739 scanned object pairs, the probe `Word: X. Which fits better, ...` has SD 0.94 in Qwen3 but only 0.43 in Mixtral.
- The two models' probe values correlate at r = 0.18 / 0.23.
- Fraction of probe values with |probe| ≥ 1 on the margin pairs: Qwen3 0.30, Mixtral 0.02.

So Mixtral's "96 % neutral" mostly means that the probe does not respond in Mixtral. Proposal:
- Before rebuilding the case sets, add a second probe in the task's own fill-in-the-blank format with the sentence context removed, e.g. `Fill in the blank: Between the mirror and the spray, the _ goes with "dirty".\nAnswer: the`. This is a prefill-only scan, ≈ 5 GPU-min per model.
- Keep the neutral rule only if the probe responds in both models.
- Report the "reverse" pairs (the trigger alone points to the wrong candidate and the model still answers right) as a separate stratum: context overrides association there, which makes them candidates for the attention-required subset.

## Phase 5 execution plan (2026-10-11 ~06:10; user: this session executes everything, full output set, sequential queue)

The other Phase-5 session's two open questions are settled by the user ("this session executes"; the coordinator's
recommendations apply):
- (1) the main set is "both trigger words neutral", after the probe check below;
- (2) the restated format is behaviour only, and the natural sentence uses all usable pairs split in half.

**Probe rule.**
- v1 is the existing probe `Word: X. Which fits better, A or B?`. v2 is new, in the task's own format with the sentence removed:
  - objects: `Fill in the blank: Between the mirror and the spray, the _ goes with "dirty".\nAnswer: the`;
  - names: `Fill in the blank: Between Brett and Kevin, _ goes with "generous".\nAnswer:`.
  - Both candidate orders are averaged.
- A probe "responds" in a model if ≥ 10 % of the margin pairs' trigger probes reach |probe| ≥ 1.
- Neutral pair: |probe| < 1 for both triggers under every probe that responds in that model.
- Strata reported for every condition:
  - associative: both triggers point to their own answers by ≥ 1;
  - one-sided: one trigger does;
  - reverse: a trigger points to the other candidate by ≥ 1;
  - neutral.

**Conditions** (× qwen3 and mixtral_bos; Mixtral with BOS as in Phases 3–4). Run dirs: `results/p5_<cond>_<proto>/`.

| cond | Task | Pairs | Prompt format | Items | Split |
|---|---|---|---|---|---|
| `f1` | 1 | all WinoGrande twins | fill-in-the-blank, entity readout | margin & content swap & neutral | 128 / 128 + replication where the pool allows |
| `dir_f1` | 2 | the 776 Phase-3 twins | fill-in-the-blank | fill-in-the-blank margin (no association filter, strata only) | half / half |
| `dir_opt` | 2 | the same pairs as `dir_f1` | Phase-3 (fill the blank, predict the trigger) | same pair set | same split |
| `f2e` | 3 | twins differing before the blank, blank at the sentence end | natural truncation | margin & content & neutral | half / half |
| `vt_e2a_orig`, `vt_e2a_neu` | 4 | verb templates × 48 IOI name pairs | Phase-3 | margin per condition | seed split S, ≤ 128 pairs per split, round-robin over seeds |
| `vt_a2e_orig`, `vt_a2e_neu` | 5 | the 55 ent2attr templates | fill-in-the-blank (new build) | margin | seed split S |
| `cp_vt_orig`, `cp_vt_neu` | 6 | templates + copy prefix, both attribute orders | Phase-3 + prefix | margin | seed split S |
| `cp_nat` | 6 | 776 Phase-3 twins + copy prefix, both orders | Phase-3 + prefix | margin (strata: Phase-3 local `assoc`, names) | Phase-3 split restricted to passing pairs (+ replication) |
| `mem` | 7 | 55 templates with their ORIGINAL names × orig / neutral / unrelated | Phase-3 | none | behaviour only (scan) |

- Seed split S: the 55 ent2attr seeds shuffled with `random.Random(0)`, 27 discovery / 28 validation; the same S for tasks 4, 5 and 6 (templates).
- The restated format: behaviour table only.
- The pre-neutral fill-in-the-blank runs (`results/wino_ref_F1_*`) are kept as a record.

**Output set per condition and model** (Phase 3 / 4 format): the 11 items of "Phase 5 task list and output standard".
- Knockout targets: the condition's own selected expert, Phase 3's WinoGrande experts (Qwen3 L41E117, Mixtral L20E000) on every condition, the condition's population top-10 set, and frequency-matched random sets.

**Agents** (coordinator + sub-agents as in Phases 3–4; GPU only via `scripts/gpu_queue.sh`):

| Agent | Owns | Stages |
|---|---|---|
| `p5-data` | `moetrace/ext13_wino_ref.py`, `moetrace/ext13_templates.py`, new `moetrace/ext13_*.py` (not the prefixes below), `scripts/ext13_*` (not the prefixes below), `data/wino_ref/`, `results/wino_ref_*`, `results/p5_<cond>_<proto>/`, tables / figures `ext13base_*`, section `ext13_base.md` | v2 probe, builds, scans, case sets, `data/wino_ref/p5_conditions.json`; base runners and analysis (layer curves, all-MoE / direct, DLA, experts, heads, grid) for every condition; restated and `mem` behaviour tables |
| `p5-addback` | `moetrace/ext13ab*.py`, `scripts/ext13ab_*`, `results/p5_*_ab/`, `ext13ab_*` tables / figures, section `ext13_addback.md` | ceilings (add-back, deletion, ≤ L−5), static orderings, adaptive greedy, deletion curves, simplified figures |
| `p5-circuit` | `moetrace/ext13ci*.py`, `scripts/ext13ci_*`, `results/p5_*_ci*/`, `ext13ci_*`, section `ext13_circuit.md` | single heads + head DLA, mixed head + expert greedy |
| `p5-expert` | `moetrace/ext13ko*.py`, `moetrace/ext13wr*.py`, `scripts/ext13ko_*`, `scripts/ext13wr_*`, `results/p5_*_ko*/`, `results/p5_*_wr*/`, `ext13ko_*` / `ext13wr_*`, sections `ext13_knockout.md`, `ext13_writer.md` | knockout; writer direct share + vocabulary projection |

- Read-only for every agent: `moetrace/engine.py` and all ext7–ext12 modules and scripts. Reuse them by import or wrapper. Additive, backward-compatible flags only with the coordinator's approval and a regression check.
- Coordinator: this plan, CLAUDE.md, README, `scripts/build_extensions_report.py`, synthesis section `ext13_synthesis.md`, the ladder figure, the add-back curve analysis and the attribution-patching pilot (later agents).

**Handshake.**
- `p5-data` writes `data/wino_ref/p5_conditions.json`: per cond / proto the pair file, case-set file, run dir, families, split sizes, a status (`cases`, `base`), `selected_expert` and `pop_top10`.
- It appends the exact PROGRESS lines `[p5-data] CASES READY <cond>/<proto>` and `[p5-data] BASE READY <cond>/<proto>`.
- Downstream chains wait on those lines, detached, and never arm Monitors on the queue log.

**GPU order** (the user asked for sequential execution of all tasks): probes and scans, then the base runners in task order: `f1` → `dir_f1`, `dir_opt` → `cp_nat`, `cp_vt_orig`, `cp_vt_neu` → `vt_e2a_*`, `vt_a2e_*` → `f2e` → `mem`. Downstream stages queue per condition as soon as its base is ready.
- Jobs are kept ≤ ≈ 25 min so the queue interleaves.
- Estimate ≈ 25–30 GPU-h: base ≈ 10, add-back ≈ 6, circuit ≈ 7, knockout + writer ≈ 4.

**Team page (user request 2026-10-12 ~05:25Z; added to the wrap-up).** After the synthesis, publish a team-facing results page in the style of the Phase-3/4 team brief (`docs/team_brief_winogrande_str.md`, artifact "WinoGrande STR brief"). It is a new page; the earlier brief stays unchanged and is linked from it.
- Structure: each meeting question → what we ran (one concrete prompt pair) → one figure → a one-to-three-line takeaway.
- Content:
  - the original direction (attribute → entity);
  - association balance;
  - the verb controls and memorisation;
  - copy vs generate;
  - the task ladder;
  - head + expert add-back;
  - knockout;
  - why the add-back curves differ;
  - attribution patching;
  - open points (scale, MIB).
- English, plain wording.
- Also saved as `docs/team_brief_phase5.md`.

**Phase 5 status (2026-10-12 ~06:20Z): COMPLETE (not committed).** All 19 runnable condition × model runs have the full output set. Synthesis: `results/sections/ext13_synthesis.md`. Sections 13a–13h are assembled into EXTENSIONS_REPORT.md as Directions 13–13h. GPU ≈ 25 h (263 queued jobs).
- Deviations:
  - copy prefix changed to `Options: t1, t2.` (the "Of X and Y" prefix allowed positional mapping);
  - task 5 (attribute → entity on the templates) is Qwen3-orig only (margin failure elsewhere);
  - coordinator fix of a copy-attention indexing bug;
  - two Mixtral OOM reruns with smaller row budgets.
- Team page in progress.

Phase 5 committed and pushed 2026-10-12 (results, code, sections, team brief `docs/team_brief_phase5.md`; ≈ 1.4 GB).
