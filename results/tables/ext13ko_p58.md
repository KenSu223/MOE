**P5-8 and direction contrasts of knockout F (eval items, independent bootstrap per condition; different item sets unless stated). wg = Phase 3's WinoGrande expert; selected_b = the second condition's own selected expert on both conditions; own top-10 = each condition's own population top-10 set**

| model | contrast | expert(s) | a | b | F a | F b | a - b [95% CI] | paired: source items (units) | paired F a / F b | paired a - b [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | copy availability (templates, orig verb): C2 - C1 | L41E117 | cp_vt_orig | vt_e2a_orig | -0.0013 | 0.0014 | -0.003 [-0.016, +0.010] | 28 (8) | +0.002 / -0.010 | +0.012 [-0.005, +0.038] |
| Qwen3-30B-A3B-Base | copy availability (templates, orig verb): C2 - C1 | L39E121 | cp_vt_orig | vt_e2a_orig | -0.0005 | 0.005 | -0.005 [-0.039, +0.032] | 28 (8) | -0.021 / -0.019 | -0.002 [-0.022, +0.043] |
| Qwen3-30B-A3B-Base | copy availability (templates, orig verb): C2 - C1 | own top-10 | cp_vt_orig | vt_e2a_orig | 0.0969 | 0.056 | +0.041 [-0.097, +0.177] | 28 (8) | +0.067 / +0.043 | +0.023 [-0.119, +0.207] |
| Qwen3-30B-A3B-Base | copy availability (templates, neutral verb): C2 - C1 | L41E117 | cp_vt_neu | vt_e2a_neu | 0.0002 | 0.0011 | -0.001 [-0.015, +0.013] | 25 (4) | -0.006 / -0.005 | -0.001 [-0.030, +0.033] |
| Qwen3-30B-A3B-Base | copy availability (templates, neutral verb): C2 - C1 | L42E123 | cp_vt_neu | vt_e2a_neu | -0.0067 | 0.0201 | -0.027 [-0.056, +0.004] | 25 (4) | -0.005 / +0.025 | -0.031 [-0.058, -0.008] |
| Qwen3-30B-A3B-Base | copy availability (templates, neutral verb): C2 - C1 | own top-10 | cp_vt_neu | vt_e2a_neu | 0.0583 | 0.1613 | -0.103 [-0.192, -0.017] | 25 (4) | +0.048 / +0.129 | -0.081 [-0.126, -0.029] |
| Qwen3-30B-A3B-Base | copy availability (natural 776 pairs): C2 - C1 | L41E117 | cp_nat | dir_opt | 0.0267 | 0.0254 | +0.001 [-0.016, +0.017] | 29 (29) | +0.023 / +0.033 | -0.010 [-0.041, +0.020] |
| Qwen3-30B-A3B-Base | copy availability (natural 776 pairs): C2 - C1 | own top-10 | cp_nat | dir_opt | 0.2035 | 0.1849 | +0.018 [-0.019, +0.056] | 29 (29) | +0.181 / +0.147 | +0.034 [-0.012, +0.084] |
| Qwen3-30B-A3B-Base | direction (templates, orig; copy fixed): C3 - C2 | L41E117 | vt_a2e_orig | cp_vt_orig | -0.0093 | -0.0013 | -0.008 [-0.030, +0.000] |  |  |  |
| Qwen3-30B-A3B-Base | direction (templates, orig; copy fixed): C3 - C2 | L39E001 | vt_a2e_orig | cp_vt_orig | -0.0177 | 0.0084 | -0.026 [-0.062, +0.024] |  |  |  |
| Qwen3-30B-A3B-Base | direction (templates, orig; copy fixed): C3 - C2 | own top-10 | vt_a2e_orig | cp_vt_orig | 0.2881 | 0.0969 | +0.191 [+0.084, +0.366] |  |  |  |
| Qwen3-30B-A3B-Base | direction (natural 776 pairs; copy fixed): C3 - C2 | L41E117 | dir_f1 | cp_nat | 0.0011 | 0.0267 | -0.026 [-0.038, -0.013] | 29 (29) | -0.001 / +0.023 | -0.024 [-0.052, +0.004] |
| Qwen3-30B-A3B-Base | direction (natural 776 pairs; copy fixed): C3 - C2 | own top-10 | dir_f1 | cp_nat | 0.1818 | 0.2035 | -0.022 [-0.064, +0.021] | 29 (29) | +0.097 / +0.181 | -0.084 [-0.149, -0.019] |
| Qwen3-30B-A3B-Base | direction on identical items (fill-in a→e - option swap e→a) | L41E117 | dir_f1 | dir_opt | 0.0011 | 0.0254 | -0.024 [-0.042, -0.008] | 90 (90) | +0.001 / +0.025 | -0.024 [-0.042, -0.007] |
| Qwen3-30B-A3B-Base | direction on identical items (fill-in a→e - option swap e→a) | own top-10 | dir_f1 | dir_opt | 0.1818 | 0.1849 | -0.003 [-0.040, +0.034] | 90 (90) | +0.182 / +0.185 | -0.003 [-0.040, +0.032] |
| Qwen3-30B-A3B-Base | direction (templates, orig): a→e - e→a | L41E117 | vt_a2e_orig | vt_e2a_orig | -0.0093 | 0.0014 | -0.011 [-0.034, -0.000] |  |  |  |
| Qwen3-30B-A3B-Base | direction (templates, orig): a→e - e→a | L39E121 | vt_a2e_orig | vt_e2a_orig | -0.0548 | 0.005 | -0.060 [-0.085, +0.053] |  |  |  |
| Qwen3-30B-A3B-Base | direction (templates, orig): a→e - e→a | own top-10 | vt_a2e_orig | vt_e2a_orig | 0.2881 | 0.056 | +0.232 [+0.153, +0.390] |  |  |  |
| Qwen3-30B-A3B-Base | verb: neutral - orig (e→a) | L41E117 | vt_e2a_neu | vt_e2a_orig | 0.0011 | 0.0014 | -0.000 [-0.015, +0.015] |  |  |  |
| Qwen3-30B-A3B-Base | verb: neutral - orig (e→a) | L39E121 | vt_e2a_neu | vt_e2a_orig | 0.0113 | 0.005 | +0.006 [-0.036, +0.057] |  |  |  |
| Qwen3-30B-A3B-Base | verb: neutral - orig (e→a) | own top-10 | vt_e2a_neu | vt_e2a_orig | 0.1613 | 0.056 | +0.105 [-0.011, +0.220] |  |  |  |
| Mixtral-8x7B (BOS) | copy availability (templates, orig verb): C2 - C1 | L20E000 | cp_vt_orig | vt_e2a_orig | 0.0202 | 0.0117 | +0.009 [-0.037, +0.059] | 35 (8) | +0.025 / -0.004 | +0.029 [-0.000, +0.057] |
| Mixtral-8x7B (BOS) | copy availability (templates, orig verb): C2 - C1 | L19E006 | cp_vt_orig | vt_e2a_orig | 0.0167 | 0.0422 | -0.025 [-0.053, +0.001] | 35 (8) | +0.020 / +0.043 | -0.022 [-0.055, +0.004] |
| Mixtral-8x7B (BOS) | copy availability (templates, orig verb): C2 - C1 | own top-10 | cp_vt_orig | vt_e2a_orig | 0.0871 | 0.0753 | +0.012 [-0.096, +0.136] | 35 (8) | +0.108 / +0.060 | +0.048 [-0.035, +0.140] |
| Mixtral-8x7B (BOS) | copy availability (templates, neutral verb): C2 - C1 | L20E000 | cp_vt_neu | vt_e2a_neu | -0.0009 | 0.0231 | -0.024 [-0.060, +0.023] | 31 (5) | -0.000 / +0.004 | -0.004 [-0.008, +0.015] |
| Mixtral-8x7B (BOS) | copy availability (templates, neutral verb): C2 - C1 | L19E006 | cp_vt_neu | vt_e2a_neu | 0.0345 | 0.0368 | -0.002 [-0.039, +0.042] | 31 (5) | +0.032 / +0.035 | -0.003 [-0.028, +0.011] |
| Mixtral-8x7B (BOS) | copy availability (templates, neutral verb): C2 - C1 | own top-10 | cp_vt_neu | vt_e2a_neu | 0.0448 | 0.0581 | -0.013 [-0.129, +0.099] | 31 (5) | +0.048 / +0.044 | +0.004 [-0.074, +0.066] |
| Mixtral-8x7B (BOS) | copy availability (natural 776 pairs): C2 - C1 | L20E000 | cp_nat | dir_opt | 0.0446 | 0.0251 | +0.019 [+0.009, +0.030] | 30 (30) | +0.032 / +0.036 | -0.003 [-0.014, +0.007] |
| Mixtral-8x7B (BOS) | copy availability (natural 776 pairs): C2 - C1 | own top-10 | cp_nat | dir_opt | 0.1846 | 0.1711 | +0.013 [-0.013, +0.040] | 30 (30) | +0.149 / +0.162 | -0.013 [-0.036, +0.009] |
| Mixtral-8x7B (BOS) | direction (natural 776 pairs; copy fixed): C3 - C2 | L20E000 | dir_f1 | cp_nat | 0.002 | 0.0446 | -0.043 [-0.051, -0.035] | 30 (30) | +0.008 / +0.032 | -0.024 [-0.047, -0.003] |
| Mixtral-8x7B (BOS) | direction (natural 776 pairs; copy fixed): C3 - C2 | L21E006 | dir_f1 | cp_nat | 0.0114 | 0.023 | -0.012 [-0.023, +0.000] | 30 (30) | -0.005 / +0.015 | -0.020 [-0.040, -0.001] |
| Mixtral-8x7B (BOS) | direction (natural 776 pairs; copy fixed): C3 - C2 | own top-10 | dir_f1 | cp_nat | 0.243 | 0.1846 | +0.058 [+0.012, +0.104] | 30 (30) | +0.255 / +0.149 | +0.106 [+0.036, +0.173] |
| Mixtral-8x7B (BOS) | direction on identical items (fill-in a→e - option swap e→a) | L20E000 | dir_f1 | dir_opt | 0.002 | 0.0251 | -0.023 [-0.035, -0.012] | 94 (94) | +0.002 / +0.025 | -0.023 [-0.034, -0.012] |
| Mixtral-8x7B (BOS) | direction on identical items (fill-in a→e - option swap e→a) | own top-10 | dir_f1 | dir_opt | 0.243 | 0.1711 | +0.072 [+0.028, +0.115] | 94 (94) | +0.243 / +0.171 | +0.072 [+0.029, +0.114] |
| Mixtral-8x7B (BOS) | verb: neutral - orig (e→a) | L20E000 | vt_e2a_neu | vt_e2a_orig | 0.0231 | 0.0117 | +0.011 [-0.031, +0.054] |  |  |  |
| Mixtral-8x7B (BOS) | verb: neutral - orig (e→a) | L19E006 | vt_e2a_neu | vt_e2a_orig | 0.0368 | 0.0422 | -0.005 [-0.036, +0.027] |  |  |  |
| Mixtral-8x7B (BOS) | verb: neutral - orig (e→a) | own top-10 | vt_e2a_neu | vt_e2a_orig | 0.0581 | 0.0753 | -0.017 [-0.088, +0.055] |  |  |  |
