| model / task | pool | STR denoise frac. of drop at N = 10 / 100 / 1000 / 10000 | N for 80 % (denoise) | mean-ablation faithfulness at N = 10 / 100 / 1000 / 10000 | N for F = 0.8 | completeness at N = 100 / 1000 |
|---|---|---|---|---|---|---|
| olmoe / wino | all | 0.36 / 0.76 / 0.92 / 0.99 | 158 | -0.03 / -0.05 / 0.28 / 0.73 | 26837 | 0.22 / 0.15 |
| olmoe / wino | final_swap | 0.36 / 0.75 / 0.86 / 0.97 | 242 | -0.05 / -0.04 / 0.16 / 0.41 | never | 0.30 / 0.20 |
| olmoe / wino | final | 0.36 / 0.73 / 0.86 / 0.90 | 265 | -0.05 / -0.04 / 0.17 / 0.40 | never | 0.32 / 0.22 |
