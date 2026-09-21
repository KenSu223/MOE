"""Interpretive text for results/sections/ext5_f1_rankings.md (ext5-analysis). Numbers quoted here were read from
results/ext5_rank_summary.json and the ext5_rank_* tables of 2026-09-22; regenerate the section with
scripts/ext5_rank_analyze.py --section-only after editing."""

SUMMARY = (
    "**Summary.** (1) *Rankings agree where it matters.* In all three runs the same two or three (layer, expert) pairs lead under all-case "
    "rescue, active-only rescue, Spec and the paper's discovery statistic (Kendall tau rescue vs active-only 0.94-0.97; rescue vs Spec "
    "0.35-0.53 among pairs with a clearly positive rescue, 0.64-0.73 within the selected layer). Over the long tail of near-zero pairs the "
    "orderings are uncorrelated (rescue vs Spec 0.01-0.03 in Mixtral): noise, not disagreement. The real disagreements are systematic: "
    "*junior partners* (positive rescue, negative Spec: Qwen3 L43E046, Mixtral L19E006 and its kin) and *Spec without rescue* in layers "
    "whose block patch hurts (Qwen3 L47, Mixtral L29-L31); Spec is only interpretable next to a positive rescue, and block share or per-case "
    "percentile measure concentration, not size. L44E069 is rank 1 under rescue, active-only, Spec and the discovery statistic; L42E115 rank 2. "
    "(2) *Minimal sets.* The sum of single-expert rescues equals the exact coalition and the block on average at every layer (Qwen3 L44 "
    "+0.935/+0.916/+0.941), so additive population-level sets are trustworthy: 50/80/90% of the block rescue take 1/6/9 experts at Qwen3 L44 "
    "(E069 alone 53%), 1/2/6 at L42 (E115 72%), 1/3/4 at Mixtral L19 with BOS (E002 63%), 1/2/2 at L18, 1/4/5 at L19 without BOS (E002 51%, "
    "fails recurrence). Per case the approximation is loose in Qwen3 (mean |sum - coalition| 0.36); exact per-case sets are in F1.3 below. "
    "Across layers L44E069 + L42E115 is 80-101% of the L44 block (F1.4).\n"
)

OVERVIEW = (
    "**Questions (RESEARCH_PLAN.md F1.1, F1.2).** (1) Does the ranking of experts depend on the statistic used to rank them? The paper "
    "ranks by all-case rescue (rescue where the expert is clean-active, zero elsewhere) after a recurrence gate and reports Spec as a "
    "second number; other natural choices are the active-only rescue, Spec itself, the expert's share of its layer's block rescue and "
    "the expert's per-case rank among the case's active experts (Table 10 style). (2) How many experts of a layer are needed to recover "
    "50/80/90% of the layer's MoE-block rescue, as a fixed set over cases?\n\n"
    "**Data and definitions.** Existing all-layer expert passes on the paper case set (`results/{qwen3_bos,mixtral_bos,mixtral_nobos}_alllayers`, "
    "no new GPU work). *Block* = the MoE-block output patch of the layer (kind `layer`, same pass). Every ordering is computed on the "
    "discovery split where it is a selection statistic and on the validation split otherwise; all CIs are 5,000-resample percentile "
    "bootstraps over validation cases with the same resampling indices as `stats.summarize`, vectorised over experts. *Spec* is the "
    "all-case active-random specificity of `analysis.evaluate_expert` (3 controls for Qwen3, 1 for Mixtral). *Block share* = mean all-case "
    "rescue / mean block rescue on validation and is left undefined in layers whose block rescue CI includes zero (a share of a null "
    "effect is noise; the raw value is kept in the CSV as `block_share_raw`). *Mean percentile* = mean over the expert's validation-active "
    "cases of (n_active - rank)/(n_active - 1), rank 1 = the case's best expert. Recurrence = clean-active in >= 64 of 128 discovery cases. "
    "Kendall tau-b (scipy) between the orderings is reported over all recurrent pairs, over those in layers with a clearly positive block "
    "rescue, over those whose own rescue CI excludes zero, and within the two-stage layer.\n\n"
    "**Minimal sets (F1.2).** Under the additive approximation the value of a fixed set S is mean_c sum_{e in S ∩ active(c)} rescue_e, "
    "which is linear in S, so greedy forward selection is exactly the descending order of all-case mean rescue on discovery and the curve "
    "is its cumulative sum; we report it on validation as a fraction of the validation block rescue, and the in-sample (validation-ordered) "
    "curve as an optimistic bound. Because a fixed set that recovers 80% of the *mean* block rescue need not recover 80% in most cases, we "
    "add a genuinely non-linear coverage variant: greedy on the number of discovery cases whose additive sum reaches 80% of *their own* "
    "block rescue (cases with block > 0), evaluated as the covered fraction of validation cases. The additive end point is checked against "
    "the exact `coalition_clean` rows (all clean-active experts patched jointly). Cross-layer sets are reported both as the additive sum "
    "(upper bound, over-counts information that several layers restore) and as the per-case maximum over the pairs in S (lower bound, "
    "full redundancy); exact multi-layer patches are F1.4. Code: `moetrace/ext5_rank.py`, `scripts/ext5_rank_analyze.py`.\n"
)

INTERPRETATION = {
    "qwen3_bos_alllayers": {
        "f11": (
            "**Reading.** Wherever the effect is clear the orderings agree: L44E069 and L42E115 are ranks 1 and 2 under all-case rescue, "
            "active-only rescue, Spec and the discovery statistic, and the next tier (L43E005, L40E127, L41E001, L40E030) is the same under all "
            "four (tau rescue vs active-only 0.94; rescue vs Spec 0.53 over the 37 recurrent pairs with a clearly positive rescue and 0.73 within "
            "L44). The low tau over all 234 recurrent pairs (0.38 rescue vs Spec, 0.27 discovery vs validation) is the ordering of near-zero "
            "effects, i.e. noise: among the 37 clear effects the discovery statistic predicts validation rescue with tau 0.74. The disagreements are "
            "systematic and of three kinds. (a) *Junior partners*: L43E046 is rank 3 on discovery and 8 on validation rescue but 29 on Spec "
            "(+0.017) and 168 on percentile: its L43 partners rescue as much as it does, so L43's +0.61 block rescue is spread (no L43 expert "
            "exceeds 24% of the block); L43E104 is the same case. (b) *Spec without rescue*: L47E032 and L47E034 rank 9-10 on Spec (+0.06) with "
            "rescues of +0.03 in a layer whose block patch *hurts* (-0.13): their controls are negative, so Spec is positive although the expert does "
            "nothing. Spec is a within-case contrast and is only interpretable together with a positive rescue. (c) *Concentration measures reward "
            "weak layers*: block share puts L33E076 first (81% of a +0.10 block) and L44E069 seventh (53% of +0.94), and the per-case percentile "
            "puts L22E091, L23E058 and L26E101 (rescue <= +0.02) in its top 7 because they beat their equally ineffective peers. L42E115 is the "
            "expert that every metric likes: rank 1-2 everywhere, 72% of its block, top-1 among the 8 active experts in 71% of its cases (E069: 53%)."),
        "f12": (
            "**Reading.** At L44 one expert (E069) is 53% of the block rescue, six experts are 80% and nine are 90%, out of 80 experts that are "
            "ever active (every case has exactly 8). The second and third experts in the greedy order, E098 and E108, are rare (16 and 11 of 128 "
            "discovery cases) but large when active, so the curve is flat between |S| = 1 and 3 on validation. L42 is more concentrated: E115 alone "
            "is 72%, two experts are 83%. L43 has no dominant expert (three for 50%, seven for 80%) and L40 is intermediate (two for 50%, six for "
            "80%). The curves saturate at 97-104%: the sum of all single-expert rescues equals the exact coalition and the block on average "
            "(differences within [-0.12, +0.10], CIs include zero at every layer), which is the additivity fact from RESEARCH_PLAN.md re-derived "
            "here (L44: sum +0.935, coalition +0.916, block +0.941, r = 0.93, mean |sum - coalition| 0.36; L40/L42/L43 r 0.77-0.89). Per case the "
            "approximation is loose: only 50-62% of cases are within 0.25 of the coalition and 75-84% within 0.5, and this bounds the coverage "
            "variant: even with every active expert in S, the additive sum reaches 80% of the case's own block in at most 66% (L44) to 72% (L42) "
            "of cases. {E069} alone covers 30% of the L44 cases, four experts 51%; {E115} alone covers 52% of the L42 cases. Whether the exact "
            "joint patches behave better per case is F1.3. Across layers, the additive sum of L44E069 and L42E115 (+0.946) already equals the L44 "
            "block rescue and keeps growing to 3.4x the block with 59 pairs, which no joint patch can do; the per-case maximum, the other extreme, "
            "gives 80% for the two and 91-99% for 5-9 pairs. The truth lies between and needs the multi-layer patches of F1.4; what the data say "
            "already is that a *second* expert from L42 adds more than any further expert of L44 (+0.25 by the max reading vs +0.03 for E098)."),
    },
    "mixtral_bos_alllayers": {
        "f11": (
            "**Reading.** With 8 experts and top-2 routing every case has one partner, so Spec = rescue(e) - rescue(partner) and the orderings "
            "separate into 'senior' and 'junior' partners. The top three under rescue, active-only, Spec and the discovery statistic are the same "
            "(L19E002, L21E001, L18E001; tau rescue vs discovery 0.88 over the 18 clear effects), and L19E002 is first under every metric except "
            "block share (8th: 63% of the largest block). The disagreements are the two failure modes seen in Qwen3, sharper here. (a) Junior partners: "
            "L20E005 (rescue rank 5, Spec rank 36, -0.07), L19E006 (rank 8 vs 38, Spec -0.25), L25E003, L17E005 and L18E006 all rescue when active "
            "but are out-rescued by their partner (E002, E001). (b) Spec without rescue: L29E003, L30E004 and L31E007 are ranks 4-6 on Spec (+0.10) with "
            "zero or negative rescue in layers whose block patch hurts (-0.12 to -0.20). Over all 38 recurrent pairs rescue and Spec are uncorrelated "
            "(tau 0.03); within L19 tau is 0.64, and Spec agrees best with block share (0.72 in block-positive layers) because both measure how much "
            "of a layer's effect one expert carries. The per-case percentile is again dominated by early layers with no effect (L0E004, L6E007, L2E006)."),
        "f12": (
            "**Reading.** Additivity is tight in Mixtral (one pairwise interaction per case): the sum of the two singles is within 0.25 of the exact "
            "coalition in 95-99% of cases (r 0.91-0.98); at L19 the sum exceeds the coalition by +0.028 [+0.007, +0.049], a small sub-additive "
            "interaction between E002 and its partners. Population-level sets are small because there are only 8 experts: E002 is 63% of the L19 "
            "block, three experts (E002, E006, E004) are 80% and four are 90%; L18E001 alone is 77% of its block and two experts are 90%; L21 needs "
            "three for 80%; L20 is the most spread (E005 31%, four for 80%). Coverage is correspondingly better than in Qwen3 (80% of cases at |S| = "
            "6-8 in L19-L22) but one expert still covers only 21-51% of cases. Across layers the per-case maximum of L19E002 and L21E001 is 88% of "
            "the L19 block and adding L18E001 gives 105%; the additive sum of the same three is 152%, an over-count. L19E002 + L18E001: sum "
            "+0.607 (105% of the L19 block), per-case max +0.490 (84%); positive together in 38% of cases, either in 70%."),
    },
    "mixtral_nobos_alllayers": {
        "f11": (
            "**Reading.** Under the paper's protocol the two-stage winner L19E006 is rank 4 by validation rescue and by the discovery statistic but "
            "28th of 30 recurrent pairs by Spec (-0.16), 29th by percentile (it is the *worse* of the two active experts in 51% of its cases) and 18th "
            "by block share (15%). It is one of five junior partners in the L17-L22 band, with L20E005 (rescue rank 2, active-only rank 1, Spec rank "
            "27), L21E000, L20E000 and L18E006: in every one of these layers the senior expert is E001 (L17, L18, L21, L22), E002 (L19) or E005 "
            "(L20), and the junior one rescues only when its partner is not there to rescue more. L18E001 is rank 1 under rescue, Spec and the "
            "discovery statistic (2 under active-only, 6 under share, 8 under percentile) and is the only pair whose Spec CI excludes zero (ext1). "
            "Spec and rescue are uncorrelated over the 30 recurrent pairs (tau 0.01) and only moderately related among the 7 clear effects (0.43) "
            "and within L19 (0.71); L30E004 and L31E002 are ranks 2-3 on Spec with zero rescue in layers whose block hurts (-0.17, -0.07), the "
            "same artefact as with BOS. The metrics therefore agree that L19E006 is not a locus and disagree only on how to say so: rescue ranks it "
            "as an ordinary fourth-best expert, Spec and percentile as the model's clearest example of an expert that rescues *less* than its partner."),
        "f12": (
            "**Reading.** The greedy order at L19 starts with E002 (51% of the block alone, four experts for 80%), then E004 and E006: the paper's "
            "recurrence gate is what removes E002 (clean-active in 59/128 discovery cases) and promotes E006 (91/128), not the rescue. L18E001 is "
            "75% of its block alone, L21E001 56% (three experts for 80%), L22 is spread (two for 50%, never reaches 90% because two of its experts "
            "have negative rescue). Additivity holds as with BOS (92-98% of cases within 0.25; no significant sum - coalition difference). Across "
            "layers the per-case maximum of L21E001 and L19E002 is 94% of the L19 block (sum 119%), with L18E001 108%; L19E006 + L18E001 together "
            "reach only 47% (sum) to 62% (max) of the L19 block because E006 rescues in 12% of the cases where E001 also does. The band structure "
            "is the same as with BOS (the E001 seniors at L17/L18/L21/L22), and the BOS-induced change is confined to L19, where E002's activity "
            "crosses the recurrence threshold."),
    },
}

F12_READING = (
    "### Summary across models\n\n"
    "- **Do the rankings agree?** Yes at the top and wherever the effect is clear; no over the long tail. In all three runs the same 2-3 pairs "
    "lead under all-case rescue, active-only rescue, Spec and the discovery statistic (Kendall tau between rescue and active-only 0.94-0.97 "
    "everywhere; rescue vs Spec 0.53 / 0.35 / 0.43 among the recurrent pairs with a clearly positive rescue and 0.64-0.73 within the selected "
    "layer). Over all recurrent pairs rescue and Spec are nearly uncorrelated in Mixtral (tau 0.01-0.03) and weakly correlated in Qwen3 (0.38), "
    "and the discovery statistic predicts validation rescue only among clear effects (tau 0.74-0.88 vs 0.27-0.64 overall).\n"
    "- **Where they disagree, and why.** (a) *Junior partners* (positive rescue, negative Spec): experts that rescue when active but are "
    "out-rescued by a co-active expert. Qwen3 L43E046/L43E104; Mixtral L19E006, L20E005, L21E000, L20E000, L18E006. The paper's Mixtral finding "
    "(recurrent but non-specific E006) is this pattern. (b) *Spec without rescue* (positive Spec, zero or negative rescue): experts in layers whose "
    "block patch hurts (Qwen3 L47, Mixtral L29-L31); their controls are negative. Spec must be read together with rescue; the joint criterion "
    "'rescue CI > 0 and Spec CI > 0' (which ext1 used implicitly) has no such artefacts. (c) *Concentration and relative measures* (block share, "
    "per-case percentile) reward experts of layers with little or no effect and should be reported only for layers whose block rescue CI excludes "
    "zero, as done here.\n"
    "- **Recommendation for the protocol.** Keep all-case rescue as the primary statistic (it and active-only rescue order experts identically "
    "wherever it matters), report Spec alongside it and interpret Spec only where the rescue CI excludes zero, and add the percentile / top-1 "
    "count among active experts as the descriptive complement (Table 10) rather than as a ranking metric.\n"
    "- **Minimal sets.** Fixed sets recovering 50 / 80 / 90% of the mean block rescue: Qwen3 L44 1 / 6 / 9 experts (E069 alone 53%), L42 1 / 2 / 6 "
    "(E115 alone 72%), L43 3 / 7 / 8, L40 2 / 6 / 10; Mixtral with BOS L19 1 / 3 / 4 (E002 63%), L18 1 / 2 / 2 (E001 77%), L21 1 / 3 / 5; without "
    "BOS L19 1 / 4 / 5 (E002 51%, not recurrent), L18 1 / 3 / 3, L21 1 / 3 / 6. The additive end points equal the exact coalition on average at "
    "every layer (Mixtral per case as well: 92-99% of cases within 0.25; Qwen3 only 50-62%), so these population-level sizes are trustworthy but "
    "the per-case coverage is not: even the full active set's additive sum reaches 80% of the case's own block in only 60-72% of Qwen3 cases. "
    "Per-case minimal sets and pairwise interactions require the exact subset patches (F1.3, below when available).\n"
    "- **Two loci.** L44E069 + L42E115 is 80% (per-case max) to 101% (additive sum) of the L44 block rescue against 53% / 47% alone; the two "
    "Mixtral seniors L19E002 + L21E001 (BOS) are 88-110% of the L19 block. Which end of the interval is right is the F1.4 question.\n"
)

F5_NOTE_ADDITIVE = ""

F13_READING = (
    "**Reading (F1.3).** The subset passes are exhaustive (Qwen3: 255 subsets x 256 cases at L44 and at L42, 130,560 rows; Mixtral no-BOS: 3 "
    "subsets at L18 and L19) and self-consistent: the patch of the full clean set equals the same pass's `coalition_clean` row to 0.004 and "
    "recovers 94-98% of the block (the rest is what noised-only experts contribute). *Per-case minimal sets are small.* For 80% of the case's "
    "own block rescue one expert suffices in 59-60% of the eligible Qwen3 cases (block > 0) and at most two in 87-88%; the median is 1 at 80% "
    "and 2 at 90%; only 2 cases (80%) and 5-9 (90%) are not reached even by all eight experts. Mixtral: one of the two active experts reaches "
    "80% in 53-54% of cases, and in 13/150 (L18) and 24/185 (L19) cases the full pair does not. *Which single expert?* At L42 the size-1 "
    "minimal set is E115 in 105 of 129 cases (81%) and {E115} alone reaches 80% in 110/210 = 52% of the cases where it is active; at L44 the "
    "size-1 set is E069 in only 61 of 126 (48%) and {E069} alone reaches 80% in 64/200 = 32% (50% of its cases at the 50% target). Mixtral no-"
    "BOS: {L18E001} 49/107 = 46%, {L19E002} 44/103 = 43%, {L19E006} 35/114 = 31%. So the population-level statement 'one expert carries half "
    "the block' translates per case into 'one expert carries most of it in a third to a half of the prompts, and which expert varies'; E115 is "
    "the more often sufficient of the two Qwen3 loci, as its higher recurrence and block share suggested. *Additive prediction vs exact.* The "
    "additive prediction (accumulate the sorted single rescues) matches the exact minimal size in 97% (L44) and 91% (L42) of cases at 80% and "
    "picks the identical set in 90% / 84%; its set reaches the target when patched exactly in 95% / 90% (Mixtral: 100% and 97%). *Interactions "
    "are small and cancel.* Over the 7,168 pair-cases per Qwen3 layer the mean pairwise interaction is +0.004 [+0.001, +0.006] (L42) and +0.007 "
    "[+0.004, +0.009] (L44), 26-28% negative; E069 and E115 interact with their co-active experts by +0.005 [+0.000, +0.011] and +0.003 [-0.002, "
    "+0.008] on average, i.e. additively. Individual pairs deviate: the only strong-strong pair, L44 E060 + E069, is redundant (-0.10 over 11 "
    "cases: +0.53 and +0.70 alone, +1.13 together), E059 + E115 is synergistic (+0.06 over 35 cases). The per-case non-additivity rescue(full) "
    "- sum(singles) is +0.02 on average but +/-0.33 per case; the sum of the 28 pairwise terms tracks it (r 0.85-0.90) but overshoots "
    "(+0.10 / +0.19) and is cancelled by the higher-order remainder (-0.08 / -0.16), each pairwise term carrying bf16 noise of the size of the "
    "effect, so the decomposition beyond 'small, mostly cancelling' is not resolvable at this precision. Mixtral shows the two regimes cleanly: "
    "L18 E001 + E006 are synergistic (+0.029 [+0.007, +0.052]; +0.36 and +0.09 alone, +0.48 together) and L19 E002 + E006 redundant (-0.034 "
    "over 62 cases; +0.45 and +0.22 alone, +0.64 together; E002 + E004 -0.26). *Bottom line for F1.* The additive approximation used in F1.2 is "
    "validated per case as well: exact per-case minimal sets are as small as the singles predict, the strong experts add up with their "
    "partners, and the one systematic non-additivity is redundancy between two strong experts of the same layer (E060/E069, E002/E006, "
    "E002/E004)."
)
