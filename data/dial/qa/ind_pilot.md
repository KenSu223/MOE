# IND PILOT (p6-data; l_ind formats, behaviour only)

200 items (4 key / value pairs of 8 distinct random words, single tokens in both families; the same for every frame and both models), both query orders (pairs = items x 2); STR = the query key (k1 <-> k2). I0 'Pairs: a = b, c = d. a ='; I1 'a: b\\nc: d\\na:'; I2 'a: b\\nc: d\\ne: f\\ng: h\\na:'; I3 'List: a b c d e f a' (behaviour reference only: the query key is the final token = STR at the readout position, which R2 forbids for patching); I1 / I2 start with a newline so that every key, also the first, is the same line-start token; I4 'The code for a is b. The code for c is d. The code for a is' (reference). First-mention bias = mean logit(value of the first-listed key) - logit(value of the second) over all prompts.

| Model | Frame | Margin both ways | Correct both directions | Items both orders margin | Order gap (o1 - o2) | First-mention bias | Median drop |
|---|---|---|---|---|---|---|---|
| qwen3 | I0 | 0.35 | 0.77 | 0.15 | +0.03 | +0.58 | 2.75 |
| qwen3 | I1 | 0.03 | 0.18 | 0.01 | +0.00 | +0.17 | -0.25 |
| qwen3 | I2 | 0.12 | 0.47 | 0.04 | -0.00 | +0.26 | 1.25 |
| qwen3 | I3 | 0.34 | 0.69 | 0.14 | -0.03 | +0.77 | 3.25 |
| qwen3 | I4 | 0.00 | 0.05 | 0.00 | +0.00 | -0.11 | -0.62 |
| mixtral_bos | I0 | 0.10 | 0.43 | 0.01 | -0.01 | +0.74 | 1.38 |
| mixtral_bos | I1 | 0.35 | 0.77 | 0.20 | -0.03 | +0.24 | 2.25 |
| mixtral_bos | I2 | 0.34 | 0.77 | 0.14 | -0.02 | +0.55 | 2.50 |
| mixtral_bos | I3 | 0.33 | 0.68 | 0.14 | +0.02 | +0.61 | 2.75 |
| mixtral_bos | I4 | 0.55 | 0.86 | 0.32 | -0.06 | +0.03 | 3.12 |

Decision (pre-set rule): **I2** (no I1-I3 frame reaches 0.8 in both models: best I1-I3 frame anyway (highest minimum over models, |gap| <= 0.10 where possible)).
