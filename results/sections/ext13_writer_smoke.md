**Summary.** Writer vs computer (Phase 4 Direction 11) per Phase-5 condition: the single-expert STR patch rescue T (p5-data's expert rows) split into the expert's direct write D (DLA of δ_e = c_e(clean) − c_e(corrupted twin), final norm frozen at the corrupted twin; one prefill pass per condition with `DiagSpec.contrib_dla`) and the indirect rest I = T − D; vocabulary projections of the targets' writes (`DiagSpec.contrib_final_vectors`). Qwen3-30B-A3B-Base (validation cases) — f1pre: L37E032 D/drop -0.002, D/T -0.09, δ_e rank of r 85292, top-10 D/T 1.96.

### What was run

For each condition × model one prefill pass over both twins of every pair of the case sets (discovery, validation, replication) records, at the final position, every routed expert's DLA numerator (c_e ⊙ γ)·(W_U[r] − W_U[r']) and the final RMS (`DiagSpec.contrib_dla`), the contribution vectors of the experts of interest (targets + the condition's population top-10; `contrib_final_vectors`) and the final residual. For directed case (X clean, Y = twin corrupted) D = (cdla_X + cdla_Y) / rms_Y, since the twin's readout direction is the negative of X's; T comes from p5-data's single-expert patch rows of the same case (a different pass: (case, expert) pairs whose routing differs between the two passes are dropped; routing match in the band table). Statistics as `moetrace/ext11_writer.py` (cluster bootstrap over the condition's units). Vocabulary: projections ((v ⊙ γ) W_Uᵀ) / rms with the norm frozen at the corrupted twin (δ_e) or at the prompt (c_e); classes = the condition's own attribute words (trigger words; readout class of entity → attribute conditions), its entities (answers; readout class of attribute → entity conditions), Phase 3's trigger vocabulary, IOI names, CounterFact objects.


### Part A. Total vs direct vs indirect, by expert

| model | condition | expert | role | active cases | total T / drop | direct D / drop | direct share D/T | all-case T / drop |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | L37E032 | selected+pop1 | 8/8 | +0.022 [-0.010, +0.073] | -0.002 [-0.003, -0.000] | -0.09 [+nan, +0.09] | 0.0218 |
| Qwen3-30B-A3B-Base | f1pre | L41E117 | WG | 0/8 | n/a | n/a | n/a |  |
| Qwen3-30B-A3B-Base | f1pre | L44E069 | CF | 2/8 | n/a | n/a | n/a |  |
| Qwen3-30B-A3B-Base | f1pre | L42E115 | CF | 1/8 | n/a | n/a | n/a |  |
| Qwen3-30B-A3B-Base | f1pre | L44E116 | pop2 | 6/8 | +0.101 [+0.015, +0.212] | +0.260 [+0.027, +0.541] | +2.58 [+1.79, +2.73] | 0.0817 |
| Qwen3-30B-A3B-Base | f1pre | L43E070 | pop3 | 0/8 | n/a | n/a | n/a |  |
| Qwen3-30B-A3B-Base | f1pre | L44E022 | pop4 | 5/8 | -0.004 [-0.015, +0.010] | +0.003 [-0.015, +0.028] | -0.71 [+nan, +nan] | -0.0027 |
| Qwen3-30B-A3B-Base | f1pre | L37E107 | pop5 | 0/8 | n/a | n/a | n/a |  |
| Qwen3-30B-A3B-Base | f1pre | L39E103 | pop6 | 2/8 | n/a | n/a | n/a |  |
| Qwen3-30B-A3B-Base | f1pre | L43E036 | pop7 | 5/8 | +0.019 [+0.000, +0.023] | +0.049 [+0.002, +0.090] | +2.56 [+0.45, +nan] | 0.0136 |
| Qwen3-30B-A3B-Base | f1pre | L39E004 | pop8 | 4/8 | +0.005 [+0.000, +0.019] | +0.001 [-0.001, +0.004] | +0.27 [-0.04, +nan] | 0.0027 |
| Qwen3-30B-A3B-Base | f1pre | L43E114 | pop9 | 6/8 | +0.020 [+0.000, +0.050] | +0.046 [+0.015, +0.107] | +2.29 [+0.95, +nan] | 0.0163 |
| Qwen3-30B-A3B-Base | f1pre | L43E087 | pop10 | 6/8 | +0.003 [-0.015, +0.020] | +0.004 [-0.002, +0.009] | +1.22 [-2.61, +nan] | 0.0027 |
| Qwen3-30B-A3B-Base | f1pre | top-10 set | pop |  | +0.147 [+0.041, +0.255] | +0.288 [+0.083, +0.534] | +1.96 [+0.76, +3.98] |  |


### Part A. Summed writes by depth band

| model | condition | routing match | first half: T / D / share | L/2..L-9: T / D / share | L-8..L-5: T / D / share | last 4: T / D / share | r(D, T) all |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | 0.9697 | -0.346 / +0.001 / -0.0 | +0.528 / +0.055 / 0.1 | +0.210 / +0.385 / 1.83 | -0.466 / -0.442 / 0.95 | 0.493 |


### Part B. Vocabulary projections

| model | condition | expert | vectors | n | median rank of r (of V) | r in top-10 / top-100 (%) | median rank in own class (size) | z attr_own | z ent_own | z wg_trigger | z names | z cf_any | top-50 attr / ent (%) | most promoted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | L37E032 | delta | 32 | 85292 | 0 / 0 | 18 (32) | +0.00 | -0.00 | +0.00 | -0.00 | -0.00 | 0.0 / 0.0 | '��' 9%; 'hti' 6%; '的机会' 6%; 'chart' 6%; ' nextPage' 6%; '抵' 6% |
| Qwen3-30B-A3B-Base | f1pre | L37E032 | clean | 32 | 62147 | 0 / 0 | 15 (32) | -0.61 | -0.06 | -0.39 | -0.17 | +0.09 | 0.0 / 0.0 | '或者其他' 47%; 'escaped' 28%; '/Branch' 25%; ' searchData' 25%; '也需要' 25%; 'IService' 22% |
| Qwen3-30B-A3B-Base | f1pre | L37E107 | delta | 8 | 90140 | 0 / 0 | 18 (32) | -0.00 | -0.00 | -0.00 | +0.00 | -0.00 | 0.0 / 0.0 | 'abouts' 12%; '糊' 12%; 'ufacturer' 12%; '|{\n' 12%; '��' 12%; '折' 12% |
| Qwen3-30B-A3B-Base | f1pre | L39E004 | delta | 20 | 50969 | 0 / 0 | 10 (32) | -0.00 | +0.01 | +0.01 | +0.01 | +0.07 | 0.0 / 0.0 | '保驾' 10%; ' superhero' 10%; '时常' 10%; '把这些' 10%; ' hel' 5%; ' helm' 5% |
| Qwen3-30B-A3B-Base | f1pre | L39E103 | delta | 8 | 69828 | 0 / 0 | 16 (32) | +0.00 | +0.00 | -0.00 | +0.00 | +0.00 | 0.0 / 0.0 | ' one' 12%; ' solitary' 12%; ' only' 12%; ' constructs' 12%; 'one' 12%; ' одного' 12% |
| Qwen3-30B-A3B-Base | f1pre | L42E115 | delta | 7 | 16025 | 14 / 14 | 7 (32) | -0.15 | +0.02 | -0.03 | -0.02 | -0.16 | 0.0 / 0.3 | ' Werner' 14%; 'unal' 14%; ' Holland' 14%; '荷兰' 14%; '葡萄牙' 14%; '届时' 14% |
| Qwen3-30B-A3B-Base | f1pre | L42E115 | clean | 7 | 20778 | 14 / 14 | 11 (32) | -0.22 | +0.01 | -0.00 | -0.43 | -0.64 | 0.0 / 0.3 | ' unravel' 29%; '重要因素' 29%; ' disadv' 29%; ' Werner' 14%; 'unal' 14%; ' Holland' 14% |
| Qwen3-30B-A3B-Base | f1pre | L43E036 | delta | 24 | 26964 | 4 / 8 | 7 (32) | -0.03 | +0.03 | +0.00 | -0.02 | +0.00 | 0.1 / 0.4 | 'city' 8%; ' winter' 8%; ' summer' 8%; ' winters' 8%; 'BufferData' 4%; 'Cities' 4% |
| Qwen3-30B-A3B-Base | f1pre | L43E070 | delta | 8 | 32023 | 25 / 38 | 7 (32) | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | 0.0 / 0.7 | '��' 25%; ' meat' 12%; ' meats' 12%; ' pizzas' 12%; ' baker' 12%; ' Meat' 12% |
| Qwen3-30B-A3B-Base | f1pre | L43E087 | delta | 24 | 52281 | 4 / 8 | 10 (32) | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | 0.0 / 0.3 | ' selection' 17%; 'selection' 17%; '-selection' 17%; 'Selection' 17%; ' selections' 17%; ' fine' 12% |
| Qwen3-30B-A3B-Base | f1pre | L43E114 | delta | 14 | 14280 | 7 / 7 | 2 (32) | +0.00 | +0.00 | +0.00 | -0.00 | +0.00 | 0.3 / 0.1 | ' opening' 14%; ' Opening' 14%; 'opening' 14%; 'Opening' 14%; '开口' 14%; ' weight' 14% |
| Qwen3-30B-A3B-Base | f1pre | L44E022 | delta | 21 | 47125 | 0 / 5 | 11 (32) | +0.01 | -0.02 | +0.00 | +0.02 | -0.00 | 0.0 / 0.3 | ' cloak' 10%; 'cloak' 10%; ' aisle' 10%; ' home' 10%; ' observation' 5%; '观察' 5% |
| Qwen3-30B-A3B-Base | f1pre | L44E069 | delta | 5 | 75221 | 0 / 0 | 19 (32) | +0.01 | -0.04 | +0.03 | -0.09 | +0.06 | 0.0 / 0.4 | ' Manila' 20%; '苏州' 20%; ' Lisbon' 20%; '杭州' 20%; ' Madrid' 20%; ' Denver' 20% |
| Qwen3-30B-A3B-Base | f1pre | L44E069 | clean | 5 | 75221 | 0 / 0 | 15 (32) | -0.11 | -0.08 | +0.04 | +0.05 | -0.22 | 0.0 / 0.4 | ' Panama' 40%; ' Nicar' 40%; ' Barb' 40%; 'Pu' 20%; ' Pu' 20%; ' Puerto' 20% |
| Qwen3-30B-A3B-Base | f1pre | L44E116 | delta | 22 | 13392 | 14 / 14 | 8 (32) | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | 0.2 / 0.4 | ' ice' 9%; ' onion' 9%; ' fruit' 9%; ' hard' 9%; '材料' 4%; 'Materials' 4% |


### Caveats

- D freezes the final norm (Phase 4: single-expert exact / frozen 0.996-0.999; the exact-norm check is in `vec_rows.parquet` per vector).
- T and D come from different passes (bf16 batch noise in T, Phase 4: per-row SD ≈ 0.1-0.3 logits); shares are ratios of sums over many cases.
- A direct share above 1 means later layers partly undo the expert's write; a share near 0 marks a computer (acts through later layers).
- Figures: `results/figures/ext13wr_layers_<proto>_<cond>_smoke.png`, `results/figures/ext13wr_overview_smoke.png`.
