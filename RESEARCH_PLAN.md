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
