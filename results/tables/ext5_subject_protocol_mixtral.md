**Mixtral-8x7B-v0.1: subject-site curves under the BOS (tokenizer default) and no-BOS (paper) protocols**

| Patched component (at the last subject token) | Corr. of validation curves | BOS: L* and val. rescue [CI] | no BOS: L* and val. rescue [CI] | Paired BOS − no BOS at the BOS L* [CI] |
|---|---|---|---|---|
| MoE output | 0.948 | L4 +2.312 [+1.870, +2.779] | L6 +1.197 [+0.940, +1.469] | L4: +0.963 [+0.598, +1.348] (p=0.000) |
| attention output | 0.941 | L1 +0.777 [+0.514, +1.062] | L1 +0.730 [+0.511, +0.959] | L1: +0.047 [-0.205, +0.300] (p=0.728) |
| residual after layer (hidden state) | 0.957 | L5 +3.749 [+3.230, +4.278] | L6 +2.851 [+2.452, +3.242] | L5: +0.921 [+0.472, +1.386] (p=0.000) |
