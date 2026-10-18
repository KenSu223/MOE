| model, task | expert | neurons | cases (active rows) | expert rescue T [95% CI] | T / drop | direct share (sum DLA / T) | k50 DLA | k80 DLA [95% CI] | k80 fixed set | k80 magnitude | k80 random | per-case k80 median (IQR; n) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B, CounterFact STR | L44E069 | 768 | 3 (5) | +3.625 [+2.000, +4.875] | 0.287 | 1.39 | 1.3 | 3.0 [3, 12] | 542 | 16 | 534 | 2.7 (2.6-7.4; 3) |
| Qwen3-30B-A3B, CounterFact STR | L42E115 | 768 | 3 (5) | +1.333 [+0.625, +2.000] | 0.106 | 0.65 | 1.5 | 4.1 [1, 66] | 552 | 17 | 542 | 2.5 (1.9-34; 3) |

k50 / k80 = the number of neurons whose joint patch (set to the clean values inside the corrupted run, final position, downstream recomputed) restores 50 / 80 % of the expert's own single-expert patch (ratio of case means, validation split; log-k interpolation; 'never' = not reached before all neurons). DLA = the row's own neuron DLA ranking; fixed set = one neuron order from the discovery-split mean DLA; magnitude = |w a change| x |W_down column|.
Per-case k80 on cases with T >= 0.25 logits.
