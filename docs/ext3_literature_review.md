# Direction 3 literature review: why does the BOS token move Mixtral's router?

Agent `ext3-literature`, 2026-09-14. Desk research only (no GPU, no code changes). Every reference below was opened
during this session; venue or version details that could not be confirmed are marked **[unverified]**. Context:
Mixtral-8x7B-v0.1 reproduces the paper's L19E006 result only when prompts are tokenised without `<s>`; with the
tokenizer default (`<s>` at position 0) the final-position router at L19 picks E002 for many prompts (CLAUDE.md §2,
RESEARCH_PLAN.md Direction 3).

**Summary.** The literature makes three things clear. (1) The first position of a causal transformer is special
independent of token identity: it attends only to itself, its hidden state acquires a huge-norm, input-agnostic
"massive activation" within one to two layers, and it becomes the attention sink that every later position uses to
park unwanted attention mass; a token placed there loses most of its semantic content. (2) The Mistral family is the
documented exception to "any first token will do": for Mistral-7B, `<s>` is described as the sole driver of the sink,
and without it massive activations appear at the first delimiter or at weak-semantic words such as `of` and `and`
(which Mixtral also exhibits), so removing BOS does not merely relocate the sink to the first content token, it
changes where and whether a sink forms inside a short CounterFact prompt. (3) MoE routers are linear maps of the
residual, so any change to the final-position hidden state is sufficient to flip a top-2 decision; no BOS-aware router
is needed. Taken together the literature favours H1 in a Mistral-specific form entangled with H2 ("`<s>` is the learned
sink carrier; without it the prompt has a weaker, displaced sink and over-mixes"), shows H3 cannot act on its own
(RoPE is relative, so a uniform position shift is an exact no-op), and offers only indirect support for H4. It also
explains why the paper's pipeline probably omitted BOS: ROME-lineage code inherits the tokenizer default, which is
"no BOS" for GPT-2/GPT-J and for Qwen, and the paper's reproducibility appendix never mentions special tokens.

## 1. Attention sinks, massive activations, and what happens when BOS is removed

Xiao et al. (2023) named the phenomenon: because softmax must sum to one, "the model tends to dump unnecessary
attention values to specific tokens", and initial tokens are the natural dump because they are "visible to all
subsequent tokens" (§3.1). The sink is positional rather than semantic in Llama-2: replacing the first four tokens by
`\n` retains streaming perplexity (Table 1), a learnable "Sink Token" prepended in pre-training suffices on its own
(§3.3), and SoftMax-off-by-one ("Zero Sink", Miller 2023) gives only partial relief. Sun et al. (2024) showed the
substrate: a handful of fixed feature dimensions carry values up to four orders of magnitude above the median, they
act as "fixed but important biases", and they cause the attention concentration. For Mixtral-8x7B the dimensions are
2070 and 3398, the magnitude reaches 7,100, and the carriers are "the starting token, delimiter tokens and also
certain word tokens, e.g., token 'and' and token 'of'" (§2.2). Two details matter for us: their main experiments were
run **without** BOS ("we turn off this option", App. A.3), and when BOS is prepended "for Mixtral-8x7B, some massive
activations shift to the BOS token `<s>`" whereas LLaMA2 locations are unchanged. Gu et al. (2024, ICLR 2025) add
that the first token has a much larger hidden-state norm but much smaller key and value norms, so it "acts more like
key biases" (§3.1); explicit key biases move the sink to the bias position and value biases cannot; and, decisively for
H2, if a fixed token is placed at position 2 or 3 of every pre-training sequence the sink forms on that fixed token
instead of on the first token (Table 10, right).

Barbero et al. (2025) explain the function: sinks slow "over-mixing" of information across positions, making
representations robust to prompt perturbations. Their controlled pre-training (Table 2) is the cleanest statement of
the BOS question: a model trained with `<bos>` fixed at position 0 loses the sink when `<bos>` is removed at inference
(sink metric 0.05, validation loss 2.69 to 7.56), whereas "if there is no `<bos>` during training, the sink forms at the
first token regardless, but is slightly weaker". Removing `<bos>` from Gemma 7B "causes the attention maps to become
much smoother" (§3.2). Two 2025-26 papers speak to Mistral specifically. Oh et al. (2025): "Mistral-7B has massive
activations at the first delimiter token '.', not at the starting position"; "when the bos token is placed at the
starting position, it triggers massive activations and the first delimiter token loses its massive activations"
(§2.2); "Mixtral exhibits the same behavior as Mistral" with roughly ten times larger values (App. C.3); and in the
massive-weight layer (layer 2) "the router probability between experts becomes completely skewed to one expert"
(Fig. 5). Peng et al. (2026) describe a two-block "P0-sink circuit" (position 0 attends only to itself, the MLP inflates
its norm, the representation becomes "semantically vacuous"); in Llama models removing `[BOS]` "affects only the
shallowest layers" and the position-0 sink re-forms by layer 2, but "for Mistral, `[BOS]` is the sole driver of the sink:
removing it collapses the sink rate greatly", which they attribute to sliding-window pre-training (Mistral 7B uses SWA,
Jiang et al. 2023). Sun et al. and Oh et al. disagree on whether Mixtral's starting token carries a massive activation
without BOS (Sun Fig. 3 says yes; Oh says no for Mistral and "same behavior" for Mixtral); this is input-dependent and
cheap to settle on our own prompts.

Removal and relocation methods confirm the sink is a learned bias that can be given a dedicated home: learned sink
tokens (Xiao et al.), explicit key/value biases (Sun et al. §4.3; Gu et al. Table 4), a learned per-head bias in the
softmax denominator as in gpt-oss ("Each attention head has a learned bias in the denominator of the softmax, similar
to off-by-one attention", OpenAI 2025 §2.2), register tokens in ViTs (Darcet et al. 2023), and sink-aware training
(Fu et al. 2026, who show the sink weight acts as an implicit per-head gate). Owen et al. (2025) caution that these
mitigations are model-specific. Yu et al. (2024) and Wong et al. (2025) show sinks also form on later tokens and in
middle layers via specific MLPs, so "the sink" in a no-BOS prompt need not be at position 0.

## 2. First token, sinks, and MoE routing

Jiang et al. (2024, §5) found no topic specialisation in Mixtral ("we do not observe obvious patterns in the assignment
of experts based on the topic"), but syntactic behaviour and strong positional locality: consecutive tokens share the
first-choice expert 28% of the time at layer 15 (random: 12.5%) and share either top-2 slot 62-67% (random 46%),
Table 5. Muennighoff et al. (2024) confirm that "Mixtral-8x7B exhibits little domain specialization" and hypothesise
upcycling from a dense model as the cause; they also show routing saturates very early in training. Lo et al. (2024)
report that "the router of MoE usually selects experts with larger output norms". Wang, Hayou and Nalisnick (2026)
give the mechanism we need: "the MoE router is a linear projection from token hidden states to expert logits,
therefore tokens with similar hidden states must activate similar experts" (§4.1); deeper layers collapse to
near-identical expert sets across unrelated inputs; and their Figure 1 notes that outliers in router-input space come
"from the massive activation of the first token, i.e. attention sink". Consequence for us: a BOS-induced shift of the
final-position residual is sufficient to move prompts across the E006/E002 boundary; whether that shift comes from
attention re-mixing (H1) or from the `<s>` embedding (H2) is invisible to the router.

The sink token itself is routed to specific experts. Su et al. (2025, ICLR 2026) identify "Super Experts" in the first
one to three layers whose `down_proj` outliers create the massive activations; "the router scores assigned to SEs for
the first token (which also serves as the attention sink token) are exceptionally large" (§5.1); for
Mixtral-8x7B-Instruct-v0.1 the SE is "Layer 1 Expert 3" (Table 2; the base model is not listed), and pruning SEs
destroys the sink. Oh et al.'s layer-2/expert-4 skew is likely the same object under a different index convention
**[unverified]**. Nothing in this literature examines deep-layer (L19) routing as a function of BOS. On "default"
experts: DeepSeekMoE isolates shared experts "aiming at capturing common knowledge and mitigating redundancy" (Dai et
al. 2024), but Mixtral has none, so a fallback expert at L19 would have to be emergent. Yoon et al. (2026) show the
standard router is "uninformative on the fragile tokens that drive hard reasoning", i.e. routing quality degrades
exactly where prediction is uncertain, which is the regime a noised or BOS-less prompt puts the model in.

## 3. Tokenisation conventions in causal-tracing and knowledge-editing code

ROME's `experiments/causal_trace.py` builds inputs as `token_lists = [tokenizer.encode(p) for p in prompts]`, left-pads
with a pad id, and supports only `gpt2*`, `EleutherAI/gpt-j-6B` and `gpt-neox-20b`; no special-token argument appears
anywhere (Meng et al. 2022 code). MEMIT's `compute_z.py` uses `tok([...], return_tensors="pt", padding=True)`, and
EasyEdit's ROME port (`compute_v.py`, `repr_tools.py`) does the same, computing subject indices with
`len(tok.encode(prefix))`; the only BOS-related line is commented out. The convention is therefore "trust the
tokenizer default". For GPT-2/GPT-J that default is no BOS, so the CounterFact/ROME lineage never had a position-0
special token. For Llama/Mistral the HF default is `add_bos_token: true` (Mixtral-8x7B-v0.1 `tokenizer_config.json`:
`LlamaTokenizer`, `add_bos_token true`, `add_eos_token false`), so a mechanical port **would** include BOS; omitting it
requires an explicit `add_special_tokens=False`, hand-built ids, or a shared helper designed on a model whose
tokenizer adds nothing. Qwen3 is such a model, and the target paper's reproducibility appendix (Table 2: record order,
filtering, noise, selection, statistics) says nothing about special tokens; its method text speaks only of "the
final-position MoE sublayer" and noise on "the input embeddings of the subject-token span". A Qwen3-first,
model-agnostic pipeline that disables special tokens (harmless for Qwen3) is the simplest account of the omission.
Our `moetrace/data.py` exposes exactly this switch (`add_special_tokens=special_tokens`).

## 4. Robustness of activation patching to preprocessing

The best-practice papers are silent: Zhang and Nanda (2024) discuss metric and corruption choices (recommending logit
difference and symmetric token replacement over Gaussian noise) and Heimersheim and Nanda (2024) discuss
interpretation, but neither mentions BOS or tokenisation. Tooling is not silent. TransformerLens defaults to
`prepend_bos=True` with the rationale "Even for models not explicitly trained with a prepended BOS token, heads often
use the first position as a resting position and accordingly lose information from the first token, so this
empirically seems to give better results", warns to "Pass `prepend_bos=False` when tokenizing a fragment of a larger
prompt", and carries per-model defaults (False for Bloom). A September 2026 fix (PR #1773) shows the failure mode is
live: `IOIDataset` produced two leading BOS tokens for auto-BOS tokenizers, shifting indices so that evaluation read
"logits one position too late". Google's Gemma tokenizer docs state that `<bos>` "should appear only once at the
begining of the input" and "Forgeting about those may affect the model quality significantly". Barbero et al.'s
over-mixing account yields a direct prediction for causal tracing: without a sink, perturbations spread more, so
subject-noise drops should be larger and more variable without BOS. Our data already show this (case 3167: drop +1.4
with BOS vs +5.6 without; 249 vs 233 of the paper's 256 IDs pass the drop >= 0.5 filter).

## 5. Mistral/Mixtral documentation on BOS

The HF model card runs inference with `tokenizer(text, return_tensors="pt")`, i.e. with BOS. Mistral's reference
tokenizer `mistral-common` has `encode(self, s, bos: bool, eos: bool)` with **no default**, but every request encoder
(`encode_instruct`, FIM, transcription, speech) begins with `self.start()` which returns `[self.tokenizer.bos_id]`; there
is no path that omits it. The Mistral tokenization guide states "there is always a token added to the sequence of
tokens, usually called BOS ... This token essentially starts the engine, acting as the first token". No official
statement that the base model degrades without BOS was found; the empirical statement for Mistral-7B-v0.3 comes from
Peng et al. (2026). Whether every pre-training window of Mistral/Mixtral began with `<s>` (the decisive variable in
Barbero et al.'s Table 2) is undocumented **[unverified]**.

## 6. Hypotheses H1-H4 against prior evidence

| Hyp. | Supporting evidence | Contradicting or complicating evidence | Reading |
|---|---|---|---|
| H1 sink relocation | Position 0 is special regardless of token: Xiao (`\n` substitution), Gu (first-token norm, key bias), Peng (P0 circuit, "semantically vacuous"), Barbero (no-fixed-BOS training: sink on first token anyway). Routing follows the residual (Wang et al.). Mixtral carries massive activations on weak-semantic words `of`/`and` (Sun), which CounterFact prompts contain. | Peng: for Mistral "`[BOS]` is the sole driver of the sink" and removing it collapses the sink rate, i.e. it does not simply move to the first content token. Oh: Mistral/Mixtral without BOS put massive activations on the first delimiter, absent in most prompts. | Favoured, in a Mistral-specific form: without `<s>` the prompt has a weaker, displaced or missing sink and over-mixes; final-position residual shifts; router flips. |
| H2 BOS-specific semantics | Gu Table 10 (sink follows the fixed training token), Barbero Table 2 (fixed-BOS training makes the sink BOS-dependent), Peng (Mistral), Oh (`<s>` "triggers" massive activations in Mistral), Sun A.3 (Mixtral activations partly shift to `<s>`). | Xiao (Llama-2 accepts `\n` as sink), Peng (Llama re-forms P0 sink without BOS by layer 2). All contradicting evidence is from non-Mistral models. | Not separable from H1 by the literature: `<s>` is the learned carrier of the sink in Mistral-family models. Substitution controls decide. |
| H3 RoPE shift by one | None. | RoPE makes attention depend only on relative position (Su et al. 2021); Mixtral has no absolute position embedding. A uniform shift of all positions is an exact no-op on attention logits; the only positional fact is that nothing precedes the first content token (Peng), which is H1. | Not a distinct hypothesis. The planned "no token, positions +1" control should equal the no-BOS run up to bf16 noise; keep it as a numerics check. |
| H4 default/fallback expert | Designed defaults exist (DeepSeekMoE shared experts); emergent sink experts exist in early layers (Su, Oh); deep-layer routing collapses across unrelated inputs (Wang); routers prefer large-norm experts (Lo); routers are uninformative on fragile tokens (Yoon). | No study of a default expert at Mixtral L19; Mixtral routing shows positional locality but no documented fallback (Jiang). Sink experts are at layers 1-2, not 19. | Plausible, unsupported. Needs the corpus-usage, margin and output-norm measurements below. |

## 7. Cheap experiments the engine can run (each is one or two 95-second passes)

1. **Sink map, both protocols.** Per layer, the final position's attention mass on position 0 and on every other
   position (per head), plus hidden-state norms per position and the values of dimensions 2070 and 3398 (Sun). Tests
   whether the no-BOS prompt has a sink at all, and whether it sits on the first token, on `of`/`is`/`in`, or nowhere.
   Resolves the Sun-vs-Oh disagreement on Mixtral's starting token.
2. **Early-layer routing of position 0.** Router logits at layers 0-3 for `<s>` vs the first content token. Su et al.
   predict an exceptionally large score for one expert (their L1E3 in the Instruct model; Oh's layer-2 skew) with BOS;
   if the first content token is not routed there, the sink never forms with the usual magnitude.
3. **Substitution controls with literature predictions.** `\n` or `.` at position 0 restoring BOS-run routing means the
   sink is positional (Xiao; Oh: delimiters trigger Mistral's massive activations) and supports H1; only `<s>` restoring
   it supports H2 (Peng). Double `<s>` should perturb more than a doubled content token (Peng). The shifted-position
   control must match the no-BOS run (RoPE relativity); a mismatch would indicate an engine bug.
4. **Divergence layer and router margin.** Cosine similarity of the final-position residual between protocols per
   layer, against the L19 logit margin E006 minus E002 per prompt. Wang et al.'s linear-router argument predicts a
   monotone relation; the layer where cosine first drops locates where the BOS effect enters (attention vs MoE
   sublayer, which the `attn`/`block` spawn kinds of Direction 2 can then patch).
5. **Sink-state transplant.** Patch the position-0 hidden state (or K/V) from the BOS run into the no-BOS run at layer 2
   onward, leaving all content tokens unchanged. If L19 routing reverts, the sink representation is the carrier of the
   whole effect.
6. **Subject-at-position-0 split.** Group cases by whether the subject's first token sits at position 0 without BOS.
   Peng's "semantically vacuous" P0 predicts the routing change and the larger noise drops concentrate in that group.
7. **Over-mixing check.** Distribution of subject-noise drops and clean margins per protocol (Barbero predicts larger,
   more variable drops without BOS); relate per-prompt drop change to routing change.
8. **H4 measurements.** L19 usage entropy and output norms `||E_e(x)||` for E006 and E002 on ~50k generic tokens with
   and without BOS (Lo: routers prefer large-norm experts); margin distributions; mean log-likelihood shift as the OOD
   proxy (Barbero Table 2 analogue).
9. **Qwen3 symmetric test** stays as planned: Su et al. list Super Experts at Qwen3-30B-A3B layers 1-3, so prepending
   `<|endoftext|>` may move its sink; if L44E069 survives, BOS sensitivity is a Mistral-family property.

## 8. Not verified / open questions

- Venues: Barbero et al. 2025 is described as COLM 2025 only in a third-party note; Sun et al. 2024 venue not
  checked; Oh et al. 2025 "ICML" comes from the HTML page header only; Peng et al. changed title between v1 ("How
  Attention Sinks Emerge in Large Language Models: An Interpretability Perspective") and v2.
- Sun (Fig. 3) vs Oh (§2.2, C.3) on whether Mixtral's starting token carries a massive activation without BOS.
- Super Expert index conventions (Su: L1E3, Instruct; Oh: layer 2, expert 4) and the base model's SE, unlisted by Su.
- Mistral/Mixtral pre-training packing (BOS at every window?) and the upcycling of Mixtral from Mistral-7B
  (hypothesis in Muennighoff et al., not confirmed by Mistral).
- Whether Lu et al. (2026) used `add_special_tokens=False` or an equivalent is inferred from our reproduction, not
  stated in the paper.
- lm-evaluation-harness issues on Gemma double-BOS appeared in search results but were not opened.
- No paper was found that measures BOS-dependent routing at deep MoE layers; this appears to be a gap our Direction 3
  results would fill.

## References (all opened 2026-09-14)

- Barbero, F., Arroyo, Á., Gu, X., Perivolaropoulos, C., Bronstein, M., Veličković, P., Pascanu, R. (2025). Why do LLMs attend to the first token? arXiv:2504.02732 (v4). https://arxiv.org/abs/2504.02732
- Barbero, F., Vitvitskyi, A., Perivolaropoulos, C., Pascanu, R., Veličković, P. (2024). Round and Round We Go! What makes Rotary Positional Encodings useful? arXiv:2410.06205. https://arxiv.org/abs/2410.06205
- Cancedda, N. (2024). Spectral Filters, Dark Signals, and Attention Sinks. arXiv:2402.09221. https://arxiv.org/abs/2402.09221
- Dai, D. et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. arXiv:2401.06066. https://arxiv.org/abs/2401.06066
- Darcet, T., Oquab, M., Mairal, J., Bojanowski, P. (2024). Vision Transformers Need Registers. ICLR 2024; arXiv:2309.16588. https://arxiv.org/abs/2309.16588
- Fu, Z., Zeng, W., Wang, R., Li, M. (2026). Attention Sink Forges Native MoE in Attention Layers: Sink-Aware Training to Address Head Collapse. arXiv:2602.01203 (v3). https://arxiv.org/html/2602.01203
- Gu, X., Pang, T., Du, C., Liu, Q., Zhang, F., et al. (2025). When Attention Sink Emerges in Language Models: An Empirical View. ICLR 2025 (Spotlight); arXiv:2410.10781. https://arxiv.org/abs/2410.10781
- Heimersheim, S., Nanda, N. (2024). How to use and interpret activation patching. arXiv:2404.15255. https://arxiv.org/abs/2404.15255
- Jiang, A. Q. et al. (2023). Mistral 7B. arXiv:2310.06825. https://arxiv.org/abs/2310.06825
- Jiang, A. Q. et al. (2024). Mixtral of Experts. arXiv:2401.04088. https://arxiv.org/abs/2401.04088
- Lo, K. M., Huang, Z., Qiu, Z., Wang, Z., Fu, J. (2025). A Closer Look into Mixture-of-Experts in Large Language Models. Findings of NAACL 2025; arXiv:2406.18219. https://arxiv.org/abs/2406.18219
- Lu, Modarressi, Liu, Schütze (2026). Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models. arXiv:2606.03780 (local copy `paper.txt`, Appendix A).
- Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022. Code: https://github.com/kmeng01/rome (`experiments/causal_trace.py`); MEMIT code: https://github.com/kmeng01/memit (`memit/compute_z.py`).
- Miller, E. (2023). Attention Is Off By One. Blog, 24 July 2023. https://www.evanmiller.org/attention-is-off-by-one.html
- Muennighoff, N., Soldaini, L., Groeneveld, D., et al. (2024). OLMoE: Open Mixture-of-Experts Language Models. arXiv:2409.02060. https://arxiv.org/abs/2409.02060
- Oh, J., Shin, S., Oh, D. (2025). House of Cards: Massive Weights in LLMs. arXiv:2410.01866 (v2). https://arxiv.org/abs/2410.01866
- OpenAI (2025). gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925. https://arxiv.org/abs/2508.10925
- Owen, L., Roy Chowdhury, N., Kumar, A., Güra, F. (2025). A Refined Analysis of Massive Activations in LLMs. arXiv:2503.22329. https://arxiv.org/abs/2503.22329
- Peng, R., Li, R., Chen, M., Zhou, Y., Guo, Q., Qiu, X., Lu, Y., Zhao, C. (2026). What Makes Position Zero Special? A Mechanistic Study of Position Zero Attention Sinks in LLMs. arXiv:2603.06591 (v2). https://arxiv.org/abs/2603.06591
- Sok, J., Yeom, J., Park, S., Park, J., Kim, T. (2026). Garbage Attention in Large Language Models: <BOS> Sink Heads and Sink-aware Pruning. arXiv:2601.06787. https://arxiv.org/abs/2601.06787
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864. https://arxiv.org/abs/2104.09864
- Su, Z., Li, Q., Zhang, H., Ye, W., Xue, Q., Qian, Y., Xie, Y., Wong, N., Yuan, K. (2026). Unveiling Super Experts in Mixture-of-Experts Large Language Models. ICLR 2026; arXiv:2507.23279 (v3). https://arxiv.org/abs/2507.23279
- Su, Z. et al. (2026). Attention Sink in Transformers: A Survey on Utilization, Interpretation, and Mitigation. arXiv:2604.10098 (v2). https://arxiv.org/abs/2604.10098
- Sun, M., Chen, X., Kolter, J. Z., Liu, Z. (2024). Massive Activations in Large Language Models. arXiv:2402.17762. https://arxiv.org/abs/2402.17762
- Wang, X., Hayou, S., Nalisnick, E. (2026). The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise. arXiv:2604.09780. https://arxiv.org/abs/2604.09780
- Wong, J. T. H., Zhang, C., Mahon, L., Luk, W., Isopoussu, A., Zhao, Y. (2025). On the Existence and Behavior of Secondary Attention Sinks. arXiv:2512.22213. https://arxiv.org/abs/2512.22213
- Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR 2024; arXiv:2309.17453. https://arxiv.org/abs/2309.17453
- Yoon, Y., Wang, S., Chen, W., Ok, J. (2026). When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models. arXiv:2605.07260. https://arxiv.org/abs/2605.07260
- Yu, Z., Wang, Z., Fu, Y., Shi, H., Shaikh, K., Lin, Y. C. (2024). Unveiling and Harnessing Hidden Attention Sinks. arXiv:2406.15765. https://arxiv.org/abs/2406.15765
- Zhang, F., Nanda, N. (2024). Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. ICLR 2024; arXiv:2309.16042. https://arxiv.org/abs/2309.16042
- Documentation and code: Mixtral-8x7B-v0.1 model card and `tokenizer_config.json`, https://huggingface.co/mistralai/Mixtral-8x7B-v0.1; mistral-common `sentencepiece.py`, `instruct.py`, docs `usage/tokenizers.md`, https://github.com/mistralai/mistral-common; Mistral tokenization guide (basics, control tokens), https://docs.mistral.ai/cookbooks/concept-deep-dive-tokenization-basics; TransformerLens `HookedTransformer.py` and PR #1773, https://github.com/TransformerLensOrg/TransformerLens; EasyEdit `easyeditor/models/rome/{compute_v,repr_tools}.py`, https://github.com/zjunlp/EasyEdit; Gemma tokenizer guide, https://gemma-llm.readthedocs.io/en/latest/colab_tokenizer.html.
