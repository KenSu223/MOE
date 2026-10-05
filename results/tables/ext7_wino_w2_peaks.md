**W2: final-position single-layer patches under STR (pairs; discovery argmax, validation value with pair-bootstrap CI)**

| Model | Pair set | Patched output | L* (discovery) | Discovery mean at L* | Validation rescue at L* [95% CI] | Normalised (rescue / drop) | Validation argmax | Discovery top 4 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | MoE output | L41 | +1.75 (gap +0.38 to L43) | +1.778 [+1.594, +1.972] | 0.218 [0.200, 0.238] | L41 +1.778 | L41 +1.75, L43 +1.37, L39 +1.25, L44 +0.80 |
| Qwen3-30B-A3B-Base | main | attention output | L38 | +0.33 (gap +0.04 to L42) | +0.076 [-0.042, +0.199] | 0.009 [-0.005, 0.025] | L42 +0.242 | L38 +0.33, L42 +0.29, L40 +0.24, L39 +0.22 |
| Qwen3-30B-A3B-Base | main | attention + MoE (block) | L41 | +1.93 (gap +0.42 to L43) | +1.940 [+1.753, +2.132] | 0.238 [0.219, 0.258] | L41 +1.940 | L41 +1.93, L43 +1.51, L39 +1.47, L42 +0.99 |
| Qwen3-30B-A3B-Base | replication | MoE output | L41 | +1.59 (gap +0.10 to L43) | +1.715 [+1.526, +1.911] | 0.200 [0.180, 0.219] | L41 +1.715 | L41 +1.59, L43 +1.49, L39 +1.21, L44 +0.83 |
| Qwen3-30B-A3B-Base | replication | attention output | L42 | +0.33 (gap +0.13 to L38) | +0.293 [+0.229, +0.360] | 0.034 [0.027, 0.042] | L42 +0.293 | L42 +0.33, L38 +0.20, L41 +0.19, L39 +0.17 |
| Qwen3-30B-A3B-Base | replication | attention + MoE (block) | L41 | +1.76 (gap +0.16 to L43) | +1.963 [+1.764, +2.177] | 0.228 [0.209, 0.248] | L41 +1.963 | L41 +1.76, L43 +1.60, L39 +1.37, L42 +0.98 |
| Qwen3-30B-A3B-Base | own pool | MoE output | L41 | +1.26 (gap +0.01 to L43) | +1.462 [+1.241, +1.695] | 0.184 [0.161, 0.207] | L41 +1.462 | L41 +1.26, L43 +1.25, L39 +0.97, L42 +0.91 |
| Qwen3-30B-A3B-Base | own pool | attention output | L38 | +0.37 (gap +0.01 to L41) | +0.397 [+0.248, +0.553] | 0.050 [0.031, 0.071] | L38 +0.397 | L38 +0.37, L41 +0.37, L42 +0.31, L40 +0.28 |
| Qwen3-30B-A3B-Base | own pool | attention + MoE (block) | L41 | +1.57 (gap +0.15 to L43) | +1.610 [+1.383, +1.842] | 0.203 [0.180, 0.225] | L41 +1.610 | L41 +1.57, L43 +1.43, L42 +1.20, L39 +1.09 |
| Qwen3-30B-A3B-Base | main, L ≤ L−5 | MoE output | L41 | +1.75 (gap +0.38 to L43) | +1.778 [+1.594, +1.972] | 0.218 [0.200, 0.238] | L41 +1.778 | L41 +1.75, L43 +1.37, L39 +1.25, L42 +0.73 |
| Qwen3-30B-A3B-Base | main, L ≤ L−5 | attention output | L38 | +0.33 (gap +0.04 to L42) | +0.076 [-0.042, +0.199] | 0.009 [-0.005, 0.025] | L42 +0.242 | L38 +0.33, L42 +0.29, L40 +0.24, L39 +0.22 |
| Qwen3-30B-A3B-Base | main, L ≤ L−5 | attention + MoE (block) | L41 | +1.93 (gap +0.42 to L43) | +1.940 [+1.753, +2.132] | 0.238 [0.219, 0.258] | L41 +1.940 | L41 +1.93, L43 +1.51, L39 +1.47, L42 +0.99 |
| Mixtral-8x7B, BOS | main | MoE output | L20 | +1.25 (gap +0.04 to L21) | +1.324 [+1.222, +1.429] | 0.172 [0.162, 0.183] | L20 +1.324 | L20 +1.25, L21 +1.21, L19 +1.18, L22 +0.84 |
| Mixtral-8x7B, BOS | main | attention output | L13 | +1.19 (gap +0.35 to L19) | +1.087 [+0.971, +1.209] | 0.141 [0.127, 0.157] | L13 +1.087 | L13 +1.20, L19 +0.84, L25 +0.57, L15 +0.56 |
| Mixtral-8x7B, BOS | main | attention + MoE (block) | L19 | +1.93 (gap +0.42 to L20) | +1.927 [+1.801, +2.058] | 0.251 [0.238, 0.264] | L19 +1.927 | L19 +1.93, L20 +1.51, L13 +1.49, L21 +1.25 |
| Mixtral-8x7B, BOS | replication | MoE output | L20 | +1.23 (gap +0.08 to L19) | +1.396 [+1.296, +1.502] | 0.165 [0.155, 0.175] | L20 +1.396 | L20 +1.23, L19 +1.16, L21 +1.09, L26 +0.87 |
| Mixtral-8x7B, BOS | replication | attention output | L13 | +1.10 (gap +0.32 to L19) | +1.132 [+1.004, +1.268] | 0.134 [0.119, 0.149] | L13 +1.132 | L13 +1.10, L19 +0.78, L25 +0.58, L15 +0.48 |
| Mixtral-8x7B, BOS | replication | attention + MoE (block) | L19 | +1.86 (gap +0.40 to L20) | +2.118 [+1.979, +2.269] | 0.250 [0.236, 0.264] | L19 +2.118 | L19 +1.86, L20 +1.46, L13 +1.39, L25 +1.21 |
| Mixtral-8x7B, BOS | own pool | MoE output | L20 | +1.15 (gap +0.03 to L21) | +1.163 [+1.059, +1.268] | 0.157 [0.146, 0.168] | L20 +1.163 | L20 +1.15, L21 +1.11, L19 +1.07, L26 +0.74 |
| Mixtral-8x7B, BOS | own pool | attention output | L13 | +1.01 (gap +0.24 to L19) | +0.972 [+0.837, +1.106] | 0.131 [0.114, 0.149] | L13 +0.972 | L13 +1.01, L19 +0.76, L25 +0.51, L15 +0.47 |
| Mixtral-8x7B, BOS | own pool | attention + MoE (block) | L19 | +1.76 (gap +0.41 to L20) | +1.708 [+1.576, +1.843] | 0.231 [0.216, 0.245] | L19 +1.708 | L19 +1.76, L20 +1.36, L13 +1.21, L21 +1.15 |
| Mixtral-8x7B, BOS | main, L ≤ L−5 | MoE output | L20 | +1.25 (gap +0.04 to L21) | +1.324 [+1.222, +1.429] | 0.172 [0.162, 0.183] | L20 +1.324 | L20 +1.25, L21 +1.21, L19 +1.18, L22 +0.84 |
| Mixtral-8x7B, BOS | main, L ≤ L−5 | attention output | L13 | +1.19 (gap +0.35 to L19) | +1.087 [+0.971, +1.209] | 0.141 [0.127, 0.157] | L13 +1.087 | L13 +1.20, L19 +0.84, L25 +0.57, L15 +0.56 |
| Mixtral-8x7B, BOS | main, L ≤ L−5 | attention + MoE (block) | L19 | +1.93 (gap +0.42 to L20) | +1.927 [+1.801, +2.058] | 0.251 [0.238, 0.264] | L19 +1.927 | L19 +1.93, L20 +1.51, L13 +1.49, L21 +1.25 |
