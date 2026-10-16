# Phase 6 literature / novelty scan (2026-10-16)

This scan was done by a sub-agent using web search and reading abstracts or pages. Papers marked "full text" were read in full. Items marked [unverified] come only from search snippets.

The coordinator spot-checked three abstracts:
- Arora et al. 2601.22594 (full HTML read by the coordinator);
- Smithline & Mascioli 2605.09403;
- Li & Subramani 2605.08348.

Li et al. ACL 2025 was also spot-checked. Its abstract does not frame its results as attention vs FFN; the attention / FFN numbers below come from the agent's reading of its Table 1. Everything else is as reported by the agent and was not independently re-checked.

## Q1. Graded tasks between factual recall and in-context copying; Winograd / coreference

**Binary context-vs-memory contrasts at component level exist:**
- **Monea et al. (ACL 2024, Fakepedia; full text).** Ungrounded (parametric) answers depend on MLPs at the last subject token. Grounded answers are distributed. A classifier on the causal-effect pattern separates the two (92.8 %).
- **Merullo, Eickhoff & Pavlick (NAACL 2024).** The FFN "vector arithmetic" update is used only when the answer must come from memory, not when it is in the local context.
- **Zhao et al. (NeurIPS 2025).** Probes, not patching. Contextual knowledge moves through attention heads and appears abruptly in early layers; parametric knowledge accumulates through MLPs.
- **Kahardipraja et al. (NeurIPS 2025).** Attribution separates in-context heads from parametric heads.
- **Ortu et al. (ACL 2024, full text).** In fact vs counterfactual, attention blocks play a larger role than MLPs.
  - Their graded variable (fact / counterfactual similarity) is measured only behaviourally.
  - Dotsinski et al. (TMLR 2025) find the head specialisation greatly reduced in Llama-3.1-8B.
  - Campregher et al. (TMLR 2025) find the "fact" heads are generic copy suppression.
- **Yu, Merullo & Pavlick (EMNLP 2023).** Frequency is a behavioural dial, and single heads control memory vs context.
  - The 2026 reproduction (Fouilhé et al.) finds the frequency effect vanishes in Qwen3-14B / 32B.
  - In the same reproduction, question phrasing alone shifts memory reliance by up to 80 pp. Keep the frame fixed across dial levels.
- **Other work:**
  - Jin et al. (Findings ACL 2024): memory vs context heads.
  - Li, Chen & Tong (ICML 2025): heads superpose both roles.
  - Minder et al. (ICLR 2025): a 1-D context-vs-prior knob subspace, not an attention / MLP split.
  - Saynova et al. (Findings ACL 2025): recall, heuristics and guesswork use different MLP sites, all parametric.
  - Haviv et al. (EACL 2023): idioms.
  - Farahani et al. (2026): copy vs recall arbitration.
  - Feng & Steinhardt (ICLR 2024) and Gur-Arieh, Geva & Geiger (ICLR 2026): binding mechanisms, number of bound entities as a dial. Neither splits attention vs MLP.

**Winograd / coreference:**
- Yamakoshi et al. (Findings ACL 2023): a head circuit in masked LMs carries the cue word to the candidate.
- Vig et al. (NeurIPS 2020): head mediation on WinoBias / Winogender.
- Trinley et al. (2026): pronoun fidelity.
- Nothing found on WinoGrande in decoder LMs with an attention vs MLP / expert split, and nothing on MoE.

**Verdict: partially done.**
- Already done: binary context-vs-memory attention / MLP contrasts in dense models; graded dials measured only behaviourally.
- Not found: one principled dial with three or more levels on the same items, with the attention vs MLP share measured along it. Nothing in MoE.
- Closest: Monea 2024, Merullo NAACL 2024, Zhao NeurIPS 2025, Saynova 2025.

## Q2. Dense vs MoE; Mixtral upcycling; division of labour vs scale

**Dense vs MoE comparisons:**
- **Li et al. (ACL 2025, full text by the agent).** The only attention-vs-FFN dense / MoE comparison found.
  - Method: a log-probability "add the neuron" score on 12 LRE relations, not patching.
  - Total FFN / attention gain: Mistral-7B 6.74 / 2.96, Mixtral 6.79 / 3.03 (nearly identical); Qwen1.5-7B 6.49 / 4.14 vs Qwen1.5-MoE 7.36 / 3.18; OLMoE 4.98 / 4.40.
  - Some internal inconsistencies.
- **Herbst, Wermter & Lee (ICML 2026).** Sparse probing over 6 dense + 6 MoE models (including Qwen3-30B-A3B and Mixtral). Expert neurons are less polysemantic than dense FFN neurons. Validated by DLA rank only; no attention analysis.
- **Xu (2606.05378).** Head circuits do not transfer between Pythia, OLMo and OLMoE.
- **Choe et al. (EMNLP 2025).** In the Qwen family, the earliest-layer attention contributes more to factual recall than MLPs. This is a possible confound for Qwen attention shares.

**Theory, toy models and behaviour:**
- **Smithline & Mascioli (2605.09403; checked).** In one-layer toy transformers, sparse MoE routing shifts computation from the FFN to attention; frozen random routing does it too. Untested in pretrained LMs.
- **Jelassi et al. (2410.19034).** More experts help memorisation, not reasoning.

**Mixtral upcycling:**
- Not documented by Mistral.
- Lo et al. (2406.18219): Mixtral experts vs the Mistral-7B FFN cosine ≈ 0.6, expert vs expert 0.2–0.4. This is strong evidence of Mistral-7B initialisation; independent training would give ≈ 0 because of neuron permutation.
- Attention weights were never compared; this is our CPU check.
- Qwen1.5-MoE-A2.7B is officially upcycled from Qwen-1.8B [model card, unverified at source].
- OLMoE is trained from scratch.

**Scale:** no systematic study of the attention share vs scale or FFN capacity in pretrained LMs was found.

**Verdict: partially done.**
- Not found: a patching-based attention share on identical items for a dense parent vs its MoE child.
- Closest: Li ACL 2025, Herbst ICML 2026, Smithline & Mascioli 2026 (toy).

## Q3. Expert-level circuit discovery in MoE

**Single experts / expert sets with causal validation:**
- Lu et al. 2026 (the paper we reproduced).
- Bandarkar, Ansell & Cohn (2603.17102): about 20 of 6,000 experts.
- Bandarkar et al. (ICLR 2026): multilingual routing.
- SteerMoE (Fayyaz et al., ICLR 2026).
- Super Experts (Su et al., ICLR 2026): pruning 3 Qwen3-30B-A3B experts collapses the model. They drive massive activations and attention sinks, which is relevant to our BOS / E006 sink finding.

**Router attribution:**
- Li, Zhang, Endo & Wahib (ICLR 2026): MoE outputs drive later routing more than attention.

**Sceptical results:**
- Salomone et al. (2606.25092): pre-registered expert ablations; modularity mostly dissolves in Command A+, with Qwen3-30B-A3B as the positive control.
- Xu (2606.09607): OLMoE head communities fail a closure test.

**Gradient methods on MoE:**
- No EAP, EAP-IG, attribution graph or transcoder application to an MoE LM was found.
- Nguyen (2605.23080): IG completeness is unreliable when routing is recomputed along the path; freezing routing lowers the residual.

**Arora, Wu, Steinhardt & Schwettmann (2601.22594, ICML 2026 spotlight):**
- Units: MLP neurons (pre-down-projection), per position.
- Score: RelP (Rezaei Jafari et al., NeurIPS 2025), Δv · ∂m/∂v in a replacement model. The RMSNorm denominator, the SiLU gate and the attention pattern are frozen, and a half rule is applied on the gated product. IG with 10 steps is the alternative.
- Edges via normalised flow.
- Validation: mean-ablation faithfulness and completeness.
  - About 200 neurons give near-perfect faithfulness on subject–verb agreement (Llama-3.1-8B).
  - Steering in city → state → capital (Llama-3.1-8B-Instruct).
  - Gemma-2 2B / 9B vs Gemma Scope SAEs.
- Code: github.com/TransluceAI/circuits.
- No MoE experiments.

**What a port to MoE needs:**
1. Frozen top-k routing and frozen gate weights; this misses rerouting.
2. A three-way product w_e · SiLU(gate) ⊙ up, which needs a generalised half rule. Experts active in only one of the two runs are a problem.
3. Scale: Qwen3-30B-A3B has 4.7 M expert neurons, of which 6,144 are active per token per layer.
4. Memory: a backward pass needs reverse layer streaming (F3).

**Verdict: partially done.**
- Open: joint head + expert circuits validated by patching, and gradient circuit discovery (EAP / RelP) on MoE.
- Closest: Lu 2026, Li ACL 2025, Li ICLR 2026.

## Q4. Concept circuits across tasks

**Prior work:**
- Merullo et al. (ICLR 2024): IOI and Colored Objects share 78 % of heads.
- Lan, Torr & Barez (2311.04131): a shared sequence-continuation sub-circuit.
- Lindsey et al. (Anthropic 2025): language-independent antonym features; attention frozen.
- Function vectors (Todd et al., ICLR 2024).
- ModCirc (He et al., ICML 2025).
- RAVEL [unverified].

**Key caveat: Li & Subramani (2605.08348; checked).** Component-level circuits are consistent but not task-specific: ablating another task's circuit hurts about as much as the task's own. Neuron-level circuits are specific but inconsistent.

**Verdict: partially done** (dense models only, heads or features).
- Not found: one concept across task formats with heads and experts jointly in an MoE, with specificity controls.
- Closest: Merullo ICLR 2024, Lindsey 2025, Li & Subramani 2026.

## Bottom line

**Defensible as new (as far as found):**
- patching-based attention vs MoE shares in MoE LMs;
- a matched-item dial with three or more levels;
- head + expert circuits with add-back and knockout validation;
- a dense parent vs (likely upcycled) MoE child comparison on identical items;
- RelP / neuron circuits on MoE.

**Not new:**
- context answers rely on attention and parametric answers on MLPs (Monea, Merullo, Zhao);
- single-expert causal localisation (Lu, Bandarkar, SteerMoE);
- cross-task head reuse (Merullo).

## References (as reported by the agent)

1. Arora, Wu, Steinhardt, Schwettmann (2026). Language Model Circuits Are Sparse in the Neuron Basis. ICML 2026. https://arxiv.org/abs/2601.22594
2. Rezaei Jafari, Eberle, Khakzar, Nanda (2025). RelP. NeurIPS 2025. https://arxiv.org/abs/2508.21258
3. Ortu et al. (2024). Competition of Mechanisms. ACL 2024. https://arxiv.org/abs/2402.11655
4. Dotsinski et al. (2025). On the Generalizability of "Competition of Mechanisms". TMLR. https://arxiv.org/abs/2506.22977
5. Campregher et al. (2025). Tracing Facts or just Copies? TMLR. https://arxiv.org/abs/2507.11809
6. Jin et al. (2024). Cutting Off the Head Ends the Conflict. Findings ACL 2024. https://arxiv.org/abs/2402.18154
7. Li, Chen, Tong (2025). Taming Knowledge Conflicts in Language Models. ICML 2025. https://arxiv.org/abs/2503.10996
8. Yu, Merullo, Pavlick (2023). Characterizing Mechanisms for Factual Recall in Language Models. EMNLP 2023. https://arxiv.org/abs/2310.15910
9. Fouilhé, Asher, Muller (2026). Memory vs. Context? https://arxiv.org/abs/2609.24238
10. Minder et al. (2025). Controllable Context Sensitivity and the Knob Behind It. ICLR 2025. https://arxiv.org/abs/2411.07404
11. Geva et al. (2023). Dissecting Recall of Factual Associations. EMNLP 2023. https://arxiv.org/abs/2304.14767
12. Feng, Steinhardt (2024). How do Language Models Bind Entities in Context? ICLR 2024. https://arxiv.org/abs/2310.17191
13. Gur-Arieh, Geva, Geiger (2026). Mixing Mechanisms. ICLR 2026. https://arxiv.org/abs/2510.06182
14. Haviv et al. (2023). Understanding Transformer Memorization Recall Through Idioms. EACL 2023. https://arxiv.org/abs/2210.03588
15. Merullo, Eickhoff, Pavlick (2024). Language Models Implement Simple Word2Vec-style Vector Arithmetic. NAACL 2024. https://arxiv.org/abs/2305.16130
16. Monea et al. (2024). A Glitch in the Matrix? Fakepedia. ACL 2024. https://arxiv.org/abs/2312.02073
17. Zhao et al. (2025). Understanding Parametric and Contextual Knowledge Reconciliation within LLMs. NeurIPS 2025. https://neurips.cc/virtual/2025/poster/119740
18. Kahardipraja et al. (2025). The Atlas of In-Context Learning. NeurIPS 2025. https://arxiv.org/abs/2505.15807
19. Saynova et al. (2025). Fact Recall, Heuristics or Pure Guesswork? Findings ACL 2025. https://arxiv.org/abs/2410.14405
20. Farahani, Penzkofer, Johansson (2026). To Copy or Not to Copy. https://arxiv.org/abs/2601.12075
21. Yamakoshi et al. (2023). Causal interventions expose implicit situation models. Findings ACL 2023. https://arxiv.org/abs/2306.03882
22. Vig et al. (2020). Investigating Gender Bias in Language Models Using Causal Mediation Analysis. NeurIPS 2020. https://arxiv.org/abs/2004.12265
23. Trinley et al. (2026). A Mechanistic Understanding of Pronoun Fidelity in LLMs. https://arxiv.org/abs/2606.16407
24. Li et al. (2025). Decoding Knowledge Attribution in Mixture-of-Experts. ACL 2025. https://arxiv.org/abs/2505.24593
25. Lo et al. (2024). A Closer Look into Mixture-of-Experts in LLMs. https://arxiv.org/abs/2406.18219
26. Jiang et al. (2024). Mixtral of Experts. https://arxiv.org/abs/2401.04088
27. Herbst, Wermter, Lee (2026). The Expert Strikes Back. ICML 2026. https://arxiv.org/abs/2604.02178
28. Choe et al. (2025). Do All Autoregressive Transformers Remember Facts the Same Way? EMNLP 2025. https://arxiv.org/abs/2509.08778
29. Smithline, Mascioli (2026). Sparsity Moves Computation. https://arxiv.org/abs/2605.09403
30. Jelassi et al. (2024). Mixture of Parrots. https://arxiv.org/abs/2410.19034
31. Xu (2026). Pattern Selectivity is Not Task-Causal Structure. https://arxiv.org/abs/2606.05378
32. Xu (2026). Closure-Validated Circuit Discovery in Attention Heads. https://arxiv.org/abs/2606.09607
33. Muennighoff et al. (2024). OLMoE. https://arxiv.org/abs/2409.02060
34. Qwen team. Qwen1.5-MoE-A2.7B model card [unverified at source]. https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B
35. Venkatesh (2026). Architecture, Not Scale: Circuit Localization in LLMs. ICML 2026 MI workshop. https://icml.cc/virtual/2026/79499
36. Lu, Modarressi, Liu, Schütze (2026). Expert-Aware Causal Tracing of Factual Recall in Sparse MoE LMs. https://arxiv.org/abs/2606.03780
37. Bandarkar, Ansell, Cohn (2026). Knowledge Localization in MoE LLMs Using Cross-Lingual Inconsistency. https://arxiv.org/abs/2603.17102
38. Bandarkar et al. (2026). Multilingual Routing in Mixture-of-Experts. ICLR 2026. https://arxiv.org/abs/2510.04694
39. Fayyaz et al. (2026). Steering MoE LLMs via Expert (De)Activation. ICLR 2026. https://arxiv.org/abs/2509.09660
40. Su et al. (2026). Unveiling Super Experts in MoE LLMs. ICLR 2026. https://arxiv.org/abs/2507.23279
41. Li, Zhang, Endo, Wahib (2026). Understanding Cross-Layer Contributions to MoE Routing in LLMs. ICLR 2026. https://mlanthology.org/iclr/2026/li2026iclr-understanding/
42. Salomone, Gandhi, Asaria (2026). How Modular Is a Frontier Mixture-of-Experts? https://arxiv.org/abs/2606.25092
43. Nguyen (2026). The Attribution Contract for Generative Language Models. https://arxiv.org/abs/2605.23080
44. Merullo, Eickhoff, Pavlick (2024). Circuit Component Reuse Across Tasks. ICLR 2024. https://arxiv.org/abs/2310.08744
45. Lan, Torr, Barez (2023/24). Towards Interpretable Sequence Continuation [venue unverified]. https://arxiv.org/abs/2311.04131
46. Lindsey et al. (2025). On the Biology of a Large Language Model. https://transformer-circuits.pub/2025/attribution-graphs/biology.html
47. Todd et al. (2024). Function Vectors in Large Language Models. ICLR 2024. https://arxiv.org/abs/2310.15213
48. He et al. (2025). Towards Global-level Mechanistic Interpretability (ModCirc). ICML 2025. https://proceedings.mlr.press/v267/he25x.html
49. Li, Subramani (2026). How Much Do Circuits Tell Us? https://arxiv.org/abs/2605.08348
50. Huang et al. (2024). RAVEL [unverified]. https://arxiv.org/abs/2402.17700
