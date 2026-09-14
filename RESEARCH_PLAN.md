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
