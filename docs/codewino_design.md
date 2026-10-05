# CodeWino (working name): a WinoGrande-style single-token twin benchmark on code, built for STR activation patching

Design note, 2026-09-28. Status: proposal, nothing built. CPU feasibility probe only (section 7); no GPU used.
Inputs: WinoGrande dossier (primary sources: arXiv:1907.10641 v2, KR-2012 WSC paper, Trinh & Le 2018, Elazar et al. 2021,
Kocijan et al. 2023, Le Bras et al. 2020, lm-evaluation-harness, the official v1.1 data), Zhang & Nanda 2024
(arXiv:2309.16042, PDF at the repo root), a literature scan of code minimal pairs, and the repo's CodeFact (Direction 4) and
ext6-STR code.

## 0. Summary

Goal: pairs of code prefixes (P_a, P_b) that differ in exactly one token (the trigger, as in a Winograd twin), where both
prefixes are valid, natural code and the expected next token differs (r_a vs r_b). This is Zhang & Nanda's symmetric token
replacement (STR) with WinoGrande's twin discipline: token-aligned, in-distribution, both directions usable, metric =
logit difference LD(r_a, r_b), which is exactly our Δ with foil = the twin's answer.

Code has one advantage that WinoGrande lacked: **the twin's answer can be derived mechanically**. Four sources make this work:
- *Program equivalence:* `if c: A else: B` ≡ `if not c: B else: A`.
- *Execution:* re-run the function after the swap.
- *Syntax rules:* an opener fixes its closer.
- *Typing conventions:* `[]` → `.append`, `{}` → `[k] =`.

So crowd validation becomes automatic validation, and mining real code keeps at least one side of every pair natural.

Recommended v1 families, one per trigger-answer relation (section 4):
- BRK, bracket type (rule);
- CONT, container type (lookup);
- NEG, negated condition with swapped branches (relational, the Winograd-proper case);
- EXEC, execution-grounded asserts (computed);
- consistent renaming as the null or copy control.

Probe on the 37k functions already on disk (after aligning both tokenizers):
- BRK: about 1,650 usable pairs (6,742 before a semantic-context filter);
- CONT: 1,939;
- NEG: 253, of which 45 have a True/False answer;
- COMPL: 173;
- EXEC: 474 MBPP seed asserts.

NEG needs the CodeSearchNet Python train split, which is about 11× the probe corpus.

The engine needs no change. The corrupted run is a plain prefill row, as in ext6, and trigger-position patching exists in
`ext5_subject`.

## 1. WinoGrande: what it is and what matters for us

### 1.1 Origin: the Winograd Schema Challenge (Levesque, Davis, Morgenstern, KR-2012)
- **Definition** (quoted): "a pair of sentences differing in only one or two words and containing an ambiguity that is resolved
  in opposite ways in the two sentences and that requires the use of world knowledge and reasoning."
- **Structure.** Two parties of the same kind; a pronoun that could refer to either; a *special word* that, replaced by the
  *alternate word*, keeps the sentence sensible but flips the answer.
  Example: "The trophy doesn't fit in the brown suitcase because it's too big/small."
- **Constraints:**
  - easy for humans ("System 1");
  - not solvable by selectional restrictions;
  - **Google-proof**: no corpus statistic should disambiguate.
- **Why twins.** Contexts of "give" and "receive" are statistically alike, yet the answer must change, so word-order and
  word-feature tricks fail.
- **Size.** WSC273 has 273 sentences. The only formal competition (IJCAI-16) was won with 58%.

### 1.2 WinoGrande construction (Sakaguchi, Le Bras, Bhagavatula, Choi, AAAI-20 Outstanding Paper; CACM 2021)
- **Format.**
  - A sentence with a blank "_", plus option1, option2 and answer ∈ {1, 2}.
  - The blank replaces WSC's pronoun, which removes agreement cues and maps onto LM substitution scoring.
  - Twin example, trigger easier/harder: "Sarah was a much better surgeon than Maria so _ always got the easier/harder cases."
    → Maria / Sarah.
- **Crowdsourcing** (AMT; workers with ≥99% approval and ≥5k approved HITs; $0.40 per twin, $0.03 per validation):
  - Workers write twins of 15–30 words with ≥70% word overlap.
  - Each twin must contain an anchor word drawn at random from a random WikiHow article; the anchor must not be a function
    word. This is Stokes' "creativity from constraints"; pilots showed it greatly increases topic diversity.
  - Two domains: *social* (two same-gender people with contrasting attributes) and *physical* (two objects with contrasting
    properties).
  - 77k questions (≈ 38k twins) were collected.
- **Validation.** Three distinct workers per question. Keep a question if:
  - the majority picks the right option;
  - the workers agree it is unambiguous;
  - it cannot be answered by word association from the local context around the blank (the racecar/"going so fast"
    failure).

  68% (53k) passed.
- **AfLite** (lightweight adversarial filtering):
  - A RoBERTa-large model is fine-tuned on 6k held-out instances (then discarded) and used to precompute embeddings for the
    remaining 47k.
  - Loop: train n = 64 logistic regressions, each on a random m = 10,000 subset, and predict the held-out rest. Score each
    instance by the fraction of correct predictions. Remove the top k = 500 instances with score ≥ τ = 0.75. Repeat until
    fewer than k instances qualify.
  - About 70 rounds leave **12,282 "debiased" instances** (train 9,248 / dev 1,267 / test 1,767). The 31k removed instances
    become extra training data.
  - Bias evidence: KL divergence between label-conditional first-PC embedding distributions is 2.53 (all data), 2.51 (random
    12k), 2.42 (PMI-filtered 12k) and **0.12** (AfLite).
  - The bias AfLite caught is "structural" sentiment polarity (answer option ↔ trigger valence), which lexical PMI filtering
    misses.
- **Released sizes.**
  - Nested training sizes: XS 160 / S 640 / M 2,558 / L 10,234 / **XL 40,398**. The paper says 40,938; the file, the HF card
    and the repo's `eval.py` all say 40,398.
  - Leaderboard: test labels hidden, one model per training size, "AUC" over log training size.
- **Key numbers** (from the paper and the dossier's sources):
  - *Paper baselines (test):* human 94.0; fine-tuned RoBERTa-large 79.1; RoBERTa local-context-only baseline 50.0; BERT 64.9.
  - *RoBERTa learning curve (dev):* 51.5 (XS) → 79.3 (XL).
  - *Zero-shot, partial scoring:* GPT-3 175B 70.2 (few-shot 77.7); LLaMA-65B 77.0; Llama 2 70B 80.2; Mistral 7B 75.3.
  - *Fine-tuned:* UNICORN 91.3; ST-MoE-32B **96.1** dev, above the human baseline.
  - *5-shot:* GPT-4 87.5.
  - Status: saturated.
- **LM evaluation: partial scoring** (Trinh & Le 2018; GPT-3; lm-eval-harness `winogrande`).
  - Put each option into the blank and score only log P(shared suffix | prefix + option). The option tokens themselves are
    never scored, which cancels option-frequency priors.
  - In the harness, `doc_to_choice` returns `[sentence[:idx] + option1, sentence[:idx] + option2]`, the target is the suffix
    after "_", evaluation runs on the XL validation split, and the metric is `acc`.
  - Partial scoring is needed because the WinoGrande trigger usually comes *after* the blank. For causal next-token tracing
    the trigger must precede the predicted position; code satisfies this naturally, since definitions precede uses.
- **Known problems.**
  - *Twins broken by filtering:* >55% of debiased items are singletons whose twin was filtered out (dev: 284 intact pairs).
  - *Group scoring* (Elazar et al. 2021: a point only if both twins are right, chance 25%):
    - fine-tuned RoBERTa drops from 71.5 to 58.5 on the dev twins;
    - zero-shot masked-LM group scores are below random;
    - "progress is mostly due to supervision."
  - *Items that are not real Winograd schemas* (Kocijan et al. 2023): many items fail the criteria (not Google-proof, answer
    given away, incoherent).
  - *Label noise:* about 5% of test items are ambiguous or mislabelled (Ojastu et al. 2025).
  - *Critiques of adversarial filtering:*
    - it is mode-seeking and removes coverage of skills the adversary already solves (Bowman & Dahl 2021);
    - it over-selects contentious items (Phang et al. 2021).
  - *Contamination is low* (~1%, because the items were written from scratch).
  - *Label leak in the public files:* the official v1.1 files encode the gold label in the qID suffix. We checked dev
    1,267/1,267, and the test file keeps the suffixes.

### 1.3 Design principles we carry over (checklist)
1. Minimal edit: 1 (sometimes 2) trigger tokens. In the XL twins, 64% differ by one whitespace token and 19% by two.
2. The answer flips, and both twins stay natural.
3. Options of the same type, so surface form gives nothing away.
4. Both options plausible a priori (no selectional restriction).
5. No word-association shortcut from local context. Check it with local-context and no-candidate baselines.
6. No polarity alignment between trigger and option.
7. Trivial for a competent reader (here: for a competent program reader or the interpreter).
8. The answer must depend on the relation in the context, not on the entities alone.
9. Fresh and Google-proof: prefer data that is not memorised.
10. Keep twins together (same split) and score them as a group.

## 2. What Zhang & Nanda (arXiv:2309.16042, ICLR 2024) require

- **STR.**
  - Replace the key tokens by similar tokens with equal sequence length.
  - The corrupted prompt must be in-distribution: "X_corrupt is identically distributed as a fresh draw of a clean prompt."
  - r′ is the corrupted prompt's answer.
  - Metric: LD(r, r′) = logit(r) − logit(r′), with patching effect normalised by LD_cl − LD_*.
- **PairedFacts.** 145 pairs of CounterFact/Known1000 prompts with equal length under the GPT-2 tokenizer. Each pair is used
  **both ways** (each prompt serves as clean and as corrupt).
- **Why not Gaussian noising (GN).**
  - GN puts the model off-distribution. On IOI, name-mover heads put 0.58 of their attention on IO in clean runs; GN splits
    it 0.26/0.21, while STR preserves the pattern. Restoring the S-inhibition values recovers LD 1.04 under STR but only 0.49
    under GN.
  - On factual recall, the GN MLP peak is 2–5× the STR peak.
  - On greater-than, GN localisation is noise, while STR finds the known heads.
  - GN is acceptable "when token alignment or lack of analogous tokens makes STR unsuitable."
- **Metric.**
  - Prefer LD.
  - Probability cannot detect negative components when P_*(r) ≈ 0.
  - Stolfo's normalised metric explodes for tiny P_*(r).
  - KL is acceptable for circuit discovery.
- **Windows.** Patch single layers first; sliding windows inflate peaks by 1.4–1.75×.
- **Which tokens to corrupt matters.** Corrupting IOI's S2 versus S1+IO finds different head classes, so try several
  positions.
- **Code precedent in the paper.** The docstring circuit uses STR with the answer argument replaced by a random single-token
  word.
- **In this repo.** `moetrace/ext6_str.py` (in progress, 2026-09-28) applies STR to CounterFact:
  - donor subjects with the same template, identical token positions and a donor margin ≥ 1.0;
  - up to 5 donors per case;
  - the corrupted run is a plain prefill row.

  CodeWino reuses this machinery with pairs instead of donors.

## 3. Definition of a code twin

An item is (P_a, P_b, t, r_a, r_b, family, relation, provenance) with:

- **C1 Token symmetry.** Under each target tokenizer (Qwen3; Mixtral without BOS as primary, with BOS as a variant),
  |P_a| = |P_b| and the two prefixes differ at exactly one position t. A two-token trigger is a labelled variant, matching
  WinoGrande's 1–2 words. The trigger must not be the final token, so that the final position and the trigger position are
  distinct patch sites.
- **C2 Fixed option pair.** r_a ≠ r_b. Both are single-token continuations of *both* prefixes, so LD is defined on both sides,
  like option1/option2.
- **C3 Validity.** The full rewritten unit parses, including the family's consistent completion (for example the closer
  swapped with the opener, or the branches swapped with the negated condition). Where tests or execution are available, both
  variants run, and for equivalence families both pass the same tests.
- **C4 Ground truth by construction.** r_b follows from equivalence, execution, a syntax rule or a typing convention, with no
  human labels. A human spot check of 20 items per family goes into a samples file, as CodeFact did.
- **C5 Natural, in-distribution.** Both variants are plausible code.
  - *Measure:* surprisal of the trigger token, and the NLL gap on the tokens after t between the edited and the original
    side. Buggy lines carry higher entropy (Ray et al. 2016), so an edited side that looks like a bug is an outlier.
  - *Supporting check:* both variants are attested in the corpus (an n-gram count of the trigger line pattern).
- **C6 Competence both ways.** LD ≥ m on P_a and LD ≤ −m on P_b, with m = 1.0 (ext6's donor margin, the paper's clean margin).
  The pair-level pass rate is the behavioural score (Elazar group scoring).
- **C7 Shortcut audit, recorded rather than filtered.**
  - *local-context:* keep only the tokens from t−k onward;
  - *no-trigger-context:* delete the context line that makes the relational families work;
  - *trigger-answer PMI* in the corpus.

  Filtering on these would remove whole mechanisms (the Bowman & Dahl critique of AfLite), so they are stratification
  variables. An AfLite-style "debiased" subset is optional and never primary.
- **C8 Split hygiene.** Twins stay in one split; at most one pair per function per family. Use 128/128 discovery/validation
  at the pair level, both directions per pair, per-pair mean as the primary statistic (ext6 uses the donor mean). Recurrence
  threshold = half of the discovery pairs.

**Trigger-answer relations.** This is the axis that matters for interpretation: it says *what* the patch is tracing.

| Relation | Meaning | WinoGrande view | Examples |
|---|---|---|---|
| T1 copy | the answer is the swapped token, or copied from the trigger site | rejected (trivial) | consistent renaming, docstring argument, CounterFact-STR donors |
| T2 rule / lookup | the answer is a fixed function of the trigger | "association" | bracket → closer, container type → next operator |
| T3 relational (Winograd proper) | both options are licensed; the trigger *together with* other context decides | the target | NEG True/False (`is_even`: `== 0` → True, `!= 0` → False), `lo`/`hi`, the IOI analogue |
| T4 computed | the answer needs execution | n/a | `assert f(3) == ` → `5` vs `6` after an operator or argument swap |

Having one family per relation lets the same patching protocol compare copy, lookup, relational and computed mechanisms.
The Direction 4 lesson applies: single-token determinants localise, redundantly determined answers do not.

## 4. Families

| Family | Trigger (swap) | Answers | Relation | Ground truth | Example (last lines; ⟂ = prediction point) |
|---|---|---|---|---|---|
| **BRK** (bracket-type swap) | opener `(`↔`[` (tuple↔list literal), `[`↔`{` (list↔set) | `)` / `]` / `}` | T2 | syntax | `val = [1000, 900, 500, …, 4,1⟂` → `]` / `val = (1000, 900, …, 4,1⟂` → `)` (mbpp:958) |
| **CONT** (container-type initialisation swap) | init `x = []` ↔ `x = {}` | first later statement on x: `.` (append) / `[` (key store) | T2, long range | typing convention | `ludics = []` … `ludics⟂` → `.` / `ludics = {}` … `ludics⟂` → `[` (mbpp:603) |
| **NEG** (negated condition with swapped branches) | `==`↔`!=`, `<`↔`>=`, `>`↔`<=` | first token of the other branch | T3 | program equivalence (branches swapped) | `def is_Sum_Of_Powers_Of_Two(n):` `if (n % 2 == 1):` `return⟂` → ` False`; with `!=` → ` True` (mbpp:138) |
| **COMPL** (complement operand, IOI analogue) | left operand a → b in `a OP b` over two parameters | the other parameter | T3 (IOI analogue) | convention (weak) | `arrayType = ctypes.c_uint32 * (width *⟂` → ` height` / `(height *⟂` → ` width` |
| **EXEC** (execution-grounded output prediction) | one operator, argument digit or builtin (`max`↔`min`) in the prefix | single-digit, bool or None output after `assert f(…) == ` | T4 | execution | `return x + 2` … `assert f(3) == ⟂` → `5`; with `*` → `6` |
| **GEN** (generated twins: templates or LLM-written; v2) | template or LLM-written twins with random anchors (WinoGrande's WikiHow trick) | family-specific | T3 / T4 | execution + checks | binary search `if a[mid] < t:` `⟂` → `lo` / `>` → `hi`; ORION request swap `area(` → `K` / `circumference(` → `P`; Davies-style `x = 4` → `5` |
| **NULL** (null / answer-invariant control pairs) | consistent alpha-renaming of a non-answer variable; comment-cue flip | answer unchanged | control | construction | patch effect must be ≈ 0 (calibrates bf16 noise and Spec) |

**Notes per family.**
- **BRK.** This is CodeFact S1 (the most localised category: Qwen3 L47E025, Mixtral L31E000) turned into STR, so it gives the
  cleanest GN-vs-STR comparison on a mechanism we already know.
  - Most openers are separate tokens (`Ġ[`, `Ġ(`, `Ġ{` before digits), but they often merge with the next characters
    (`(data`, `[t`). The probe loses about half the candidates to that.
  - 41% of token-aligned candidates sit in contexts where the swap is syntactically valid but wrong at runtime:
    `except [A, B`, `"%s" % [a, b`, `VERSION >= [2, 0`, `isinstance(x, [A, B`. A context whitelist is required (assignment
    right-hand side, return, call argument, for-iterable, nested literal).
- **CONT.** The trigger sits at the init line and the answer at the first later use, so the distance is large.
  - Ambiguity: after a dict, `.update(`/`.setdefault(` compete with `[`. The competence filter decides.
  - Require the first use after the init, otherwise earlier uses leak the answer.
- **NEG.** The Winograd-proper family.
  - The rewritten program is *equivalent* to the original, so the twin's answer is well defined by intent, which the model can
    only read from context (function name, earlier code).
  - Closed-set answers (True/False, 0/1) come closest to WinoGrande's option pair. Open-set answers (`return [` vs
    `return self`) are noisier.
  - The yield is small (section 7), so this family needs the CSN train split.
  - buggy-HumanEval (1,896 operator flips on 107 problems, Apache-2.0) uses the same flips as *bugs*. It is a seed source, but
    its flipped side is not natural-correct.
- **COMPL.** The "other operand" rule is only a convention: `bin_coff: r > (r - ⟂` is not clearly `n`. Keep this family only
  as a small T3 add-on (two-parameter commutative operators), or drop it.
- **EXEC.** Seeds are MBPP asserts (2,922 total; 1,017 with a single-digit or bool output; 474 with a single-digit int
  argument), HumanEval tests and CRUXEval (800 short functions, MIT).
  - Both tokenizers split numbers into digits *and* put the space before a digit in its own token (`Ġ==`, `Ġ`, `1`, `5`). So
    prefixes end in `== ` and the answer is a single digit. That is how such text is always tokenised, so it stays
    in-distribution.
  - Execute in a sandbox (subprocess, timeout, no network).
- **GEN.** Only if the mined families leave a relation uncovered. Generated code is less clearly in-distribution, so keep it
  as a labelled tier.
- **NULL.** Borrowed from Abdelsalam et al. 2026 (cue-varied triplets with the same token count under four tokenizers,
  CC-BY-SA-4.0) and from ReCode/CCTest (semantics-preserving rewrites).

## 5. Build pipeline (CPU except steps 4–5)

0. **Corpus.**
   - CodeFact raw (HumanEval, MBPP, CSN test + validation: 36,996 usable functions, already on disk).
   - New: CSN Python train split, 412,178 functions, roughly 0.5 GB of parquet. Store on NVMe or gitignore it.
   - Optional: GREAT / CuBERT (ETH Py150 Open, per-file licence field), buggy-HumanEval and CRUXEval as seeds.
   - Record licences per unit, as in CodeFact.
1. **Mine sites** with `ast`/`tokenize`, apply the family rewrite and the context whitelist. Reuse `load_units`/`Unit` from
   `scripts/ext4_build_codefact.py`.
2. **Validate.** Parse the full rewritten unit with the consistent completion. Execute EXEC items. For NEG with tests
   (HumanEval/MBPP), run the tests on the rewritten function as an equivalence check.
3. **Token symmetry per tokenizer.** Exactly one differing position; single-token answers; whitespace back-off ≤ 4 characters,
   as in `ext4_data.resolve_item`; prefix ≤ 256 tokens. The probe used 160, which cost 4–22% of the Mixtral candidates depending on the family.
4. **Naturalness pass** (GPU, one prefill-only pass per model).
   - Per-token surprisal of both prefixes.
   - Flag a pair when the edited side's trigger surprisal is above the 95th percentile of original-side triggers, or when the
     NLL gap on tokens after t is above a threshold.
   - Measuring with the target model is circular but it is the relevant distribution; a small reference model is the
     alternative (decision 3).
5. **Competence both ways** (the same pass also yields LD on both sides). Per-family pass rates are a result in themselves,
   as they were for CodeFact.
6. **Shortcut audit.** Record the local-context and no-context baselines and PMI (C7). Also write a `samples.md` with 20
   eyeballed pairs per family.
7. **Sets.** At least 256 pairs per family, a seeded split at pair level (128/128), both directions. Write `items.jsonl`,
   `build_stats.md` and `samples.md` under `data/codewino/`.

## 6. Tracing protocol on CodeWino

- **Clean and corrupt.** Clean = P_a, corrupted = P_b as a plain prefill row (no noise), then the reverse.
  - LD = logit(r_a) − logit(r_b) = Δ with foil r_b.
  - Normalised rescue = (LD_pt − LD_*) / (LD_cl − LD_*).
  - Per-pair statistic = mean of the two directions.
  - The rest of `moetrace.analysis` applies unchanged (Spec, recurrence, active controls, bootstrap).
- **Patch sites.**
  1. Final position, per layer: MoE, attention and block (`Engine.run`, kinds `layer`/`attn_layer`/`block`), then the expert
     pass (`expert`, `coalition_*`).
  2. Trigger position t via `ext5_subject` (`SubjectPrefill(rec_pos=t)`): the F4 analogue, now with an in-distribution
     corruption.
  3. Attention heads (`attn_head`), to find heads that move trigger information to the prediction position.
- **GN control on the same items.** Noise on the trigger token embedding (`noise.py`, the paper's σ). This directly tests Zhang
  & Nanda's claim in a MoE on code: do GN and STR select the same layer and expert?
- **MoE-specific in-distribution diagnostic.** Top-k overlap between clean-run and corrupted-run routing at the final
  position. Prefix positions before t are identical by causality. Under STR the corrupted run should route like a normal
  prompt; under GN it may shift toward the sink-like routing found in Direction 3.
- **Layer rule.** Report the interior-layer rule (≤ L−5) alongside the paper rule, because on code the last MoE block acts as
  a read-out (Direction 4).
- **Engine work.** None. New code: a data builder (`scripts/ext7_codewino_build.py`, module `moetrace/ext7_codewino.py`)
  and thin runners modelled on `scripts/ext6_str_*`, with pairs in place of donors.

## 7. Feasibility probe (CPU, 2026-09-28)

Script and output live in the session scratchpad, not in the repo.
- *Corpus:* 36,996 functions (CSN test + validation 35,858, MBPP 974, HumanEval 164).
- *Cap:* 2 candidates per function per family.
- *Kept:* candidates where both tokenizers give equal length, exactly one differing token and single-token answers, with
  prefix ≤ 160 tokens.

| Family | Raw candidates | After cap | Qwen3 aligned | Mixtral aligned | Both | Main losses | Notes |
|---|---|---|---|---|---|---|---|
| BRK | 19,864 | 13,166 | 7,877 | 6,945 | **6,742** | multi-token merges (`(data`), length | 2,770 of the 6,742 in runtime-invalid contexts; tuple↔list in whitelisted contexts ≈ **1,657**; set variants add more |
| CONT | 2,576 | 2,531 | 2,254 | 1,939 | **1,939** | Mixtral too_long (546) | 1,379 list-origin, 560 dict-origin |
| NEG | 732 | 730 | 308 | 258 | **253** | answer not single-token after `return`/indent (≈ 55%) | True/False 45, other closed sets 9, open-set 199; subtypes: `==` 159, `>` 36, `<` 23, `>=` 17, `!=` 14, `<=` 4 |
| COMPL | 656 | 553 | 199 | 228 | **173** | length mismatch, multi-token names | two-parameter commutative operators only 21 |
| EXEC seeds | 2,922 MBPP asserts | | | | 474 with a single-digit argument and a single-digit or bool output | | CRUXEval / HumanEval not yet counted |

**Reading of the probe.**
- BRK and CONT already reach far above 256 pairs per family, before the competence filter. CodeFact pass rates under the
  paper filter were 84–91% for S1.
- NEG and COMPL do not. At the probe's rate, the CSN train split (≈ 11× more functions) would give roughly 2,700 NEG pairs and
  ≈ 500 with True/False. This is an estimate; yields per function may differ between splits.
- EXEC depends on how many swaps keep a single-token output, which needs execution (not run in the probe).

## 8. Risks and how the design handles them

- **Real vs edited side.** Only one side is attested real code, and HumanEval/MBPP originals are likely memorised while
  twins are novel.
  - Use both directions, balance which side is the original, prefer CSN.
  - Report the surprisal gap per family; drop families where the gap dominates.
- **Code redundancy.** In CodeFact S2/S3 the subject-noise drop was only 0.12–0.25: many next tokens are determined by
  several cues, so a single swap does not flip them. C6 removes those items; yields per family will drop accordingly.
- **Syntactically valid, semantically wrong twins** (BRK contexts): handled by the whitelist, execution and tests.
- **Tokenizer differences.** Qwen3 byte-BPE and Mixtral SentencePiece merge brackets, operators and indentation differently,
  and about 50% of BRK/NEG candidates fail alignment. Keep one item set per model, or the intersection for cross-model
  comparisons (CodeFact did both).
- **Intent-dependent answers** (NEG, COMPL): the twin's answer is correct *given the original intent*. The competence filter
  plus the spot check guard this; the "correct" label is weaker than for BRK/EXEC.
- **Low foil probability.** If r_b is improbable under P_a, probability-type metrics fail (Zhang & Nanda); LD does not. Report
  the clean top-1 rate as F5 recommends.
- **Last-layer read-out on code:** handled by the interior-layer rule (section 6).

## 9. Decisions for the user

1. **v1 family set.** Recommended: BRK, CONT, NEG, EXEC, plus NULL controls; GEN and COMPL deferred.
2. **Corpus.** Download the CSN Python train split (≈ 0.5 GB, NVMe or gitignored) to make NEG viable? Also use CRUXEval /
   buggy-HumanEval as EXEC and NEG seeds?
3. **Naturalness judge.** The target model itself (recommended: one pass, same distribution we trace) or a separate small
   code model?
4. **Competence rule.** Both directions ≥ 1.0 (recommended; group scoring) or the clean side only?
5. **GN-vs-STR on the same items as a deliverable.** Recommended: it is Zhang & Nanda's question on a MoE, it costs one extra
   noised row per item, and it connects to ext6.
6. **Two-token triggers** as a labelled variant (WinoGrande allows 1–2 words)? Recommended: allowed but kept out of primary
   tables.
7. **Models.** Qwen3-30B-A3B-Base and Mixtral no-BOS (primary), Mixtral BOS and Qwen3-Coder as variants?

## 10. Effort estimate

- **Builder** (mining, validation, execution sandbox, token checks, stats, samples): about one agent-day, CPU only.
- **GPU:**
  - naturalness + competence scan: one prefill pass per model and chunk (minutes per family set);
  - tracing: like CodeFact (≈ 10 min per model per family set for sweep + expert pass), plus the GN control and
    trigger-position patching.

  Total ≈ 1–1.5 h GPU for two models. It shares the GPU queue with ext6; nothing starts until the user decides.
