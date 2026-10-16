# Phase 6 literature review (2026-10-16)

Background for the Phase 6 plan: the team's feedback on the Phase 5 presentation asked whether our result is novel, whether the
ICML circuit paper applies to MoE, and how dense / larger models should behave. Three questions were checked: (1) novelty of
"same instance, different component split", (2) arXiv 2601.22594, (3) dense vs MoE and scale. arXiv IDs were verified on arXiv or
ACL pages; "(abs)" = only the abstract was read. The search was not exhaustive; 2025–26 preprints may overlap.

## 1. Novelty

**Already known.**
- The extremes: IOI is attention-heavy, fact recall MLP-heavy at the subject (Meng 2022, 2202.05262), with attention extracting the
  attribute at the final token (Geva 2023, 2304.14767; consistent with our CounterFact share ≈ 0.55).
- Adding or contradicting context for the same query moves work away from the MLP: Merullo 2024 (2305.16130; the FFN
  "function application" step is used only when the answer is not in context), Ortu 2024 (2402.11655, competition of mechanisms),
  Monea 2024 (2312.02073, Fakepedia: memory answers have strong MLP effects at the last subject token, context answers are
  spread out), Yu 2023 (2310.15910, entity frequency decides context vs memory; one "memory head" flips it). Jin 2024 (2402.18154):
  both memory and context heads are attention heads, so a high attention share does not imply context use.
- Tasks differ in late-MLP share: Stolfo 2023 (2305.15054) last-token late MLP 40 % arithmetic, 8.7 % number retrieval from the
  prompt, 4.2 % facts (different templates per task).
- Same format, different mechanisms: Saynova 2025 (2410.14405, recall vs heuristics vs guessing use different MLP sites).
- Family effects: Choe 2025 (2509.08778): in Qwen2.5 (0.5B–14B) early attention outweighs MLP for fact recall, unlike GPT / LLaMA;
  size effects are not monotone. → compare within a family.
- Binding / entity tracking run through attention: Feng 2023 (2310.17191), Prakash 2024 (2402.14811), Gur-Arieh 2025
  (2510.06182, mechanism mix changes with the number of entities). ORION (Variengien 2023, 2312.10091): request processing in
  middle layers, retrieval from context late, 18 models up to 70B.
- Rewording the question shifts memory vs context reliance behaviourally (Fouilhé 2026, 2609.24238). Reversal curse (Berglund 2023,
  2309.12288; Zhu 2024, 2405.04669) has no attention-vs-MLP analysis.
- Winograd: Yamakoshi 2023 (2306.03882, abs; attention-head circuit on WSC in ALBERT-era models); Trichelair 2019 (1811.01778,
  "associative" WSC items, 13.5 %); Elazar 2021 (2104.08161, twin evaluation); WinoGrande (1907.10641, AFLite). No patching study
  on WinoGrande found. MIB (Mueller 2025, 2504.13151) has no split by component type.

**Not found (what looks new in our results).**
1. On identical sentences, changing only the question direction moves the final-token attention share by +0.52 (Qwen3) / +0.18
   (Mixtral) under matched STR corruption.
2. One measure, one protocol, four or more tasks, two models, the same order (Spearman 0.90).
3. Copyability (+0.04–0.07) and association (≤ 0.11) barely move the share while direction does: the deciding variable is not
   "is the answer in the context" (a naive reading of Merullo) but whether the final token points to a context entity or produces a
   property.
4. In MoE models, with condition-specific experts.

**What would make it a stronger claim.** Predictions written before the runs; one dial with ≥ 3 levels on identical items;
external anchors (Fakepedia / Yu's world capitals, MIB tasks); ≥ 3 model families including dense counterparts; attention at all
positions, not only the final token (our name-site attention exceeds the final-token attention on name templates); robustness to
the metric and the corruption site; a causal handle (ablating one head / expert moves the share).

## 2. Dials in the literature (expected direction of the shift)

| Dial | Expected | Datasets / sources |
|---|---|---|
| Context: none → supporting → conflicting | attention ↑ | Yu 2023 world capitals, Ortu 2024 CounterFact-with-context, Fakepedia, Minder 2024 (2411.07404, the "knob": answer from context vs prior) |
| Entity frequency / familiarity | MLP ↑ (prior pulls) | Yu 2023; Du 2024 (2404.04633) |
| Associativity | MLP ↑ | Trichelair 2019 labelled WSC subset; consistent with our small association effect |
| Memorisation (idioms) | early layers | Haviv 2023 (2210.03588) |
| Number of hops | second hop resolved late | Biran 2024 (2406.12775) |
| Number of entities in context | mechanism mix shifts, stays in attention | Gur-Arieh 2025 |
| Multiple-choice letter format | attention ↑ (mid-layer heads pick the letter) | Wiegreffe 2024 (2407.15018); Lieberum 2023 (2307.09458) |
| Computing on values from context | MLP ↑ at the final token | Stolfo 2023; Hanna 2023 (2305.00586) |
| Content vs logical form | belief-bias heads | Kim 2024 (2408.08590) |

## 3. Dense vs MoE and scale

Predictions from the literature:
- Extra MoE capacity goes to memorisation, not reasoning: Jelassi 2024 "Mixture of Parrots" (2410.19034); Nakamura 2025
  (2508.18672, memorisation tracks total parameters, reasoning saturates with active compute); Allen-Zhu 2024 Physics of LMs 3.3
  (2404.05405, MoE stores knowledge almost as efficiently as dense; attention can store facts too). → parametric conditions
  (CounterFact, entity → attribute) should show a larger FFN share in MoE than in a dense model with the same active parameters,
  not vs one with the same total parameters.
- Moving information across positions cannot be handed to a bigger FFN: Zoology (Arora 2023, 2312.04927), Repeat After Me
  (Jelassi 2024, 2402.01032). → IOI / attribute → entity should keep their attention share.
- Wang 2026 (2601.08383, abs): masking the top-10 heads costs < 10 % in an MoE model, > 50 % in a dense one (not a patching study).
- Li 2025 (2505.24593, abs): OLMoE / Mixtral / Qwen1.5-MoE vs dense (Qwen1.5-7B, Llama-7B, Mistral-7B); heads and experts
  correlate (r 0.68).
- Scale: Wei 2023 (2303.03846, larger models follow flipped in-context labels against priors); Tigges 2024 (2407.10827, circuits
  consistent across sizes); Lieberum 2023 (circuit analysis on Chinchilla 70B).

Upcycling (dense ↔ MoE pairs):
- Mixtral-8x7B: its paper (2401.04088) gives Mistral 7B's architecture with 8 FFN blocks per layer, nothing on initialisation;
  Lo 2024 (2406.18219) reports cosine ≈ 0.6 between Mixtral experts and Mistral-7B's FFN vs 0.2–0.4 between experts → strong but
  unofficial evidence of upcycling.
- Qwen1.5-MoE-A2.7B: upcycled from Qwen-1.8B (official model card). Qwen2-57B-A14B: "upscaled from Qwen2-7B" (Qwen2 report
  2407.10671; copies of the dense FFN, shuffled, 50 % re-initialised). Both have shared experts (engine support needed).
- Qwen3 (2505.09388): no upcycling mentioned; no architecture twin (30B-A3B 48 L, 32 / 4 heads; 8B 36 L, 32 / 8). OLMoE
  (2409.02060): from scratch.
- Hub availability (checked 2026-10-16): Qwen3-{0.6B,1.7B,4B,8B,14B}-Base public; Qwen3-32B-Base and Qwen3-235B-A22B-Base not
  public; Qwen3-235B-A22B (post-trained) 235 B params; Mistral-7B-v0.1 7.2 B; Mixtral-8x22B-v0.1 140.6 B; Qwen2-57B-A14B 57.4 B.

## 4. arXiv 2601.22594 (the "ICML paper")

Arora*, Wu* (Transluce / Stanford), Steinhardt, Schwettmann, "Language Model Circuits Are Sparse in the Neuron Basis", ICML 2026
Spotlight; code github.com/TransluceAI/circuits.
- Nodes: single MLP neurons (pre-down-projection activations) × position; attention heads are not nodes (frozen linear wiring),
  except in one MIB check. Edges: neuron-to-neuron attribution flow, direct effects only, among the top 10³ neurons.
- Scoring (RelP): input × gradient in a linearised replacement model (frozen RMSNorm denominator, SiLU gate and attention softmax;
  half rule on gate ⊙ up so scores add up). One backward pass. No training (no SAEs, transcoders or masks).
- Selection: top-k nodes or a threshold; faithfulness = mean-ablate everything outside the circuit; completeness = mean-ablate the
  circuit.
- Tasks / models: subject–verb agreement on Llama-3.1-8B (vs Llama Scope SAEs), Gemma-2 2B / 9B; MIB (IOI, arithmetic, MCQA, ARC);
  case studies on Llama-3.1-8B-Instruct (state capitals, addition, multilingual antonyms, user gender).
- Claims: MLP activations give circuits ≈ 100× smaller than MLP outputs; ≈ 200 neurons reach near-perfect faithfulness and
  completeness (matching SAEs); edge circuits > 80 % faithful with ≈ 10⁵ edges; "Dallas → Texas → Austin" clusters match a
  cross-layer-transcoder graph; steering one neuron flips capital → state.
- MoE: mentioned only in the discussion; no application found. Adaptation (our inference): freeze router weights and the top-k set;
  each expert is a bilinear SwiGLU, so the half rule holds; an expert's score = sum of its neurons' scores (= our c_e). Effects that
  act through routing changes become invisible (our AtP pilot: ≈ 1/3 of expert patches switch a downstream routing slot). At head /
  MLP granularity RelP is no better than edge attribution patching on MIB (0.08 for both).
- Feasibility here: needs full-model backward passes; nothing runs on Qwen3-30B / Mixtral without reverse layer streaming (several
  days of engine work). On OLMoE, adding the frozen nonlinearities and the half rule to the AtP code is ≈ 1 day, < 1 GPU-h.
- Recommendation: not needed for head + expert circuits. Cheap exact alternatives: (a) path patching at the final position
  (sender head / expert difference delivered only to a receiver's input) for edges; (b) concept scoring (AUROC of per-case exact
  effects vs concept labels) on existing rows, CPU; (c) neurons inside the 2–3 key experts by forward-only DLA plus a neuron-subset
  patch kind. Worth borrowing: the neuron basis inside experts, AUROC concept scoring, faithfulness / completeness reporting,
  filtering always-on nodes (cf. the sink expert L19E006).
