**Strata: names vs objects, association-probe neutral vs not (CIs only with >= 20 pairs)**

| Model | Condition | Stratum | Cases | Pairs | Ceiling | Deletion ceiling | k80 greedy / oracle | Oracle overshoot | Greedy AUC k ≤ 15 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | WinoGrande option swap (Phase 3) | names | 20 | 10 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | WinoGrande option swap (Phase 3) | objects | 236 | 118 | 0.853 [0.837, 0.867] | 0.853 [0.837, 0.868] | 10 [9, 12] / 48 [32, 64] | +0.000 [-0.009, +0.010] | 0.490 [0.470, 0.510] |
| Mixtral-8x7B (BOS) | WinoGrande option swap (Phase 3) | names | 20 | 10 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | WinoGrande option swap (Phase 3) | objects | 236 | 118 | 0.801 [0.781, 0.819] | 0.801 [0.781, 0.820] | 7 [7, 8] / 9 [8, 9] | +0.019 [+0.009, +0.030] | 0.499 [0.486, 0.512] |
| Qwen3-30B-A3B-Base | attr->entity, fill-in-the-blank (F1) | names | 140 | 70 | 0.339 [0.268, 0.405] | 0.339 [0.268, 0.405] | 3 [2, 5] / 16 [7, 32] | +0.202 [+0.164, +0.245] | 0.288 [0.256, 0.320] |
| Qwen3-30B-A3B-Base | attr->entity, fill-in-the-blank (F1) | objects | 116 | 58 | 0.140 [0.048, 0.230] | 0.140 [0.048, 0.230] | 2 [1, 3] / 2 [1, 6] | +0.346 [+0.277, +0.418] | 0.208 [0.184, 0.234] |
| Mixtral-8x7B (BOS) | attr->entity, fill-in-the-blank (F1) | names | 24 | 12 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | attr->entity, fill-in-the-blank (F1) | objects | 178 | 89 | 0.266 [0.169, 0.353] | 0.262 [0.166, 0.350] | 2 [1, 3] / 2 [1, 3] | +0.374 [+0.312, +0.443] | 0.401 [0.376, 0.427] |
| Qwen3-30B-A3B-Base | attr->entity, fill-in-the-blank (F1) [replication] | names | 120 | 60 | 0.257 [0.178, 0.333] | 0.259 [0.185, 0.332] | 2 [1, 3] / 5 [1, 24] | +0.230 [+0.186, +0.281] | 0.275 [0.243, 0.309] |
| Qwen3-30B-A3B-Base | attr->entity, fill-in-the-blank (F1) [replication] | objects | 136 | 68 | 0.109 [0.019, 0.194] | 0.104 [0.016, 0.188] | 1 [1, 2] / 1 [1, 3] | +0.358 [+0.302, +0.420] | 0.223 [0.199, 0.248] |
| Qwen3-30B-A3B-Base | direction: F1 on the Phase-3 twins | names | 12 | 6 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | direction: F1 on the Phase-3 twins | objects | 168 | 84 | 0.083 [0.017, 0.146] | 0.088 [0.022, 0.151] | 1 [1, 2] / 1 [1, 2] | +0.385 [+0.338, +0.436] | 0.216 [0.198, 0.235] |
| Qwen3-30B-A3B-Base | direction: F1 on the Phase-3 twins | assoc neutral | 100 | 50 | 0.013 [-0.066, 0.091] | 0.018 [-0.060, 0.096] | 1 [1, 1] / 1 [1, 1] | +0.437 [+0.384, +0.492] | 0.213 [0.188, 0.240] |
| Qwen3-30B-A3B-Base | direction: F1 on the Phase-3 twins | assoc not neutral | 80 | 40 | 0.191 [0.093, 0.279] | 0.196 [0.099, 0.283] | 2 [1, 3] / 4 [1, 10] | +0.305 [+0.240, +0.385] | 0.229 [0.203, 0.257] |
| Mixtral-8x7B (BOS) | direction: F1 on the Phase-3 twins | names | 2 | 1 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | direction: F1 on the Phase-3 twins | objects | 186 | 93 | 0.416 [0.365, 0.462] | 0.416 [0.367, 0.463] | 3 [3, 4] / 4 [3, 4] | +0.268 [+0.231, +0.308] | 0.419 [0.402, 0.437] |
| Mixtral-8x7B (BOS) | direction: F1 on the Phase-3 twins | assoc neutral | 90 | 45 | 0.377 [0.306, 0.446] | 0.373 [0.305, 0.442] | 3 [2, 3] / 3 [2, 4] | +0.309 [+0.260, +0.359] | 0.430 [0.404, 0.459] |
| Mixtral-8x7B (BOS) | direction: F1 on the Phase-3 twins | assoc not neutral | 98 | 49 | 0.435 [0.360, 0.501] | 0.439 [0.363, 0.505] | 4 [3, 5] / 4 [3, 5] | +0.242 [+0.189, +0.305] | 0.408 [0.386, 0.430] |
| Qwen3-30B-A3B-Base | direction: option swap on the same twins | names | 12 | 6 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | direction: option swap on the same twins | objects | 168 | 84 | 0.820 [0.803, 0.838] | 0.820 [0.803, 0.839] | 11 [10, 14] / 48 [32, 48] | -0.002 [-0.011, +0.007] | 0.459 [0.438, 0.482] |
| Qwen3-30B-A3B-Base | direction: option swap on the same twins | assoc neutral | 100 | 50 | 0.805 [0.784, 0.826] | 0.805 [0.784, 0.826] | 11 [10, 14] / 48 [24, 48] | +0.006 [-0.005, +0.020] | 0.454 [0.434, 0.476] |
| Qwen3-30B-A3B-Base | direction: option swap on the same twins | assoc not neutral | 80 | 40 | 0.825 [0.793, 0.855] | 0.825 [0.794, 0.855] | 11 [8, never] / 48 [24, 64] | -0.007 [-0.018, +0.011] | 0.461 [0.422, 0.503] |
| Mixtral-8x7B (BOS) | direction: option swap on the same twins | names | 2 | 1 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | direction: option swap on the same twins | objects | 186 | 93 | 0.755 [0.732, 0.777] | 0.755 [0.731, 0.776] | 8 [7, 8] / 9 [8, 10] | +0.033 [+0.021, +0.047] | 0.467 [0.452, 0.481] |
| Mixtral-8x7B (BOS) | direction: option swap on the same twins | assoc neutral | 90 | 45 | 0.780 [0.751, 0.809] | 0.780 [0.751, 0.808] | 8 [7, 9] / 9 [8, 10] | +0.023 [+0.009, +0.039] | 0.475 [0.453, 0.496] |
| Mixtral-8x7B (BOS) | direction: option swap on the same twins | assoc not neutral | 98 | 49 | 0.731 [0.696, 0.762] | 0.729 [0.695, 0.761] | 7 [7, 8] / 9 [8, 10] | +0.043 [+0.024, +0.065] | 0.459 [0.439, 0.478] |
| Qwen3-30B-A3B-Base | copy prefix, Phase-3 twins | names | 20 | 10 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | copy prefix, Phase-3 twins | objects | 260 | 130 | 0.800 [0.781, 0.818] | 0.800 [0.781, 0.818] | never [never, never] / 96 [64, 96] | -0.003 [-0.014, +0.009] | 0.410 [0.394, 0.429] |
| Qwen3-30B-A3B-Base | copy prefix, Phase-3 twins | assoc neutral | 120 | 60 | 0.773 [0.748, 0.798] | 0.773 [0.748, 0.798] | never [10, never] / 64 [48, 96] | +0.003 [-0.015, +0.027] | 0.419 [0.389, 0.452] |
| Qwen3-30B-A3B-Base | copy prefix, Phase-3 twins | assoc not neutral | 160 | 80 | 0.797 [0.769, 0.822] | 0.797 [0.769, 0.822] | never [never, never] / 96 [96, 96] | -0.004 [-0.016, +0.010] | 0.399 [0.381, 0.416] |
| Mixtral-8x7B (BOS) | copy prefix, Phase-3 twins | names | 32 | 16 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | copy prefix, Phase-3 twins | objects | 376 | 188 | 0.753 [0.734, 0.771] | 0.751 [0.732, 0.768] | 7 [6, 7] / 8 [7, 8] | +0.058 [+0.048, +0.068] | 0.484 [0.474, 0.493] |
| Mixtral-8x7B (BOS) | copy prefix, Phase-3 twins | assoc neutral | 156 | 78 | 0.711 [0.670, 0.748] | 0.710 [0.668, 0.746] | 7 [6, 7] / 8 [7, 8] | +0.072 [+0.054, +0.093] | 0.455 [0.438, 0.472] |
| Mixtral-8x7B (BOS) | copy prefix, Phase-3 twins | assoc not neutral | 252 | 126 | 0.747 [0.722, 0.771] | 0.744 [0.719, 0.768] | 6 [6, 7] / 7 [7, 8] | +0.067 [+0.053, +0.081] | 0.488 [0.476, 0.500] |
| Qwen3-30B-A3B-Base | copy prefix, Phase-3 twins [replication] | names | 28 | 14 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | copy prefix, Phase-3 twins [replication] | objects | 288 | 144 | 0.764 [0.744, 0.783] | 0.765 [0.745, 0.784] | 12 [11, never] / 64 [48, 96] | +0.008 [-0.005, +0.021] | 0.419 [0.401, 0.438] |
| Qwen3-30B-A3B-Base | copy prefix, Phase-3 twins [replication] | assoc neutral | 128 | 64 | 0.751 [0.716, 0.783] | 0.753 [0.717, 0.787] | never [11, never] / 96 [64, 96] | +0.009 [-0.005, +0.025] | 0.402 [0.378, 0.428] |
| Qwen3-30B-A3B-Base | copy prefix, Phase-3 twins [replication] | assoc not neutral | 188 | 94 | 0.760 [0.734, 0.784] | 0.759 [0.733, 0.784] | 11 [9, never] / 48 [48, 96] | +0.012 [-0.006, +0.031] | 0.426 [0.402, 0.449] |
| Mixtral-8x7B (BOS) | copy prefix, Phase-3 twins [replication] | names | 40 | 20 | 0.577 [0.490, 0.671] | 0.583 [0.496, 0.678] | 5 [4, 6] / 6 [5, 6] | +0.130 [+0.085, +0.176] | 0.435 [0.387, 0.487] |
| Mixtral-8x7B (BOS) | copy prefix, Phase-3 twins [replication] | objects | 404 | 202 | 0.717 [0.699, 0.735] | 0.717 [0.699, 0.735] | 7 [6, 7] / 8 [7, 8] | +0.067 [+0.058, +0.077] | 0.465 [0.455, 0.475] |
| Mixtral-8x7B (BOS) | copy prefix, Phase-3 twins [replication] | assoc neutral | 212 | 106 | 0.696 [0.668, 0.722] | 0.697 [0.670, 0.722] | 6 [6, 7] / 7 [7, 8] | +0.078 [+0.064, +0.094] | 0.458 [0.444, 0.473] |
| Mixtral-8x7B (BOS) | copy prefix, Phase-3 twins [replication] | assoc not neutral | 232 | 116 | 0.715 [0.688, 0.741] | 0.715 [0.688, 0.741] | 7 [6, 7] / 8 [7, 8] | +0.067 [+0.054, +0.082] | 0.466 [0.453, 0.480] |
| Qwen3-30B-A3B-Base | copy prefix, templates, orig verb | names | 256 | 128 | 0.699 [0.664, 0.733] | 0.699 [0.664, 0.733] | 8 [7, 9] / 24 [16, 32] | +0.065 [+0.045, +0.087] | 0.422 [0.405, 0.441] |
| Qwen3-30B-A3B-Base | copy prefix, templates, orig verb | objects | 0 | 0 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | copy prefix, templates, orig verb | assoc neutral | 202 | 101 | 0.689 [0.649, 0.728] | 0.689 [0.649, 0.729] | 7 [6, 9] / 24 [16, 32] | +0.075 [+0.051, +0.101] | 0.426 [0.405, 0.448] |
| Qwen3-30B-A3B-Base | copy prefix, templates, orig verb | assoc not neutral | 54 | 27 | 0.729 [0.655, 0.792] | 0.729 [0.656, 0.793] | 10 [7, 15] / 24 [12, 48] | +0.034 [-0.004, +0.080] | 0.409 [0.381, 0.444] |
| Mixtral-8x7B (BOS) | copy prefix, templates, orig verb | names | 256 | 128 | 0.543 [0.496, 0.584] | 0.545 [0.498, 0.586] | 4 [4, 4] / 4 [4, 5] | +0.197 [+0.167, +0.232] | 0.463 [0.443, 0.481] |
| Mixtral-8x7B (BOS) | copy prefix, templates, orig verb | objects | 0 | 0 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | copy prefix, templates, orig verb | assoc neutral | 232 | 116 | 0.550 [0.503, 0.592] | 0.552 [0.505, 0.594] | 4 [4, 4] / 4 [4, 5] | +0.190 [+0.159, +0.226] | 0.462 [0.442, 0.482] |
| Mixtral-8x7B (BOS) | copy prefix, templates, orig verb | assoc not neutral | 24 | 12 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | copy prefix, templates, neutral verb | names | 256 | 128 | 0.737 [0.706, 0.767] | 0.731 [0.699, 0.762] | 9 [8, 12] / 32 [24, 48] | +0.033 [+0.017, +0.050] | 0.426 [0.408, 0.446] |
| Qwen3-30B-A3B-Base | copy prefix, templates, neutral verb | objects | 0 | 0 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | copy prefix, templates, neutral verb | assoc neutral | 184 | 92 | 0.741 [0.703, 0.778] | 0.736 [0.697, 0.773] | 8 [7, 11] / 32 [24, 48] | +0.033 [+0.013, +0.055] | 0.439 [0.416, 0.465] |
| Qwen3-30B-A3B-Base | copy prefix, templates, neutral verb | assoc not neutral | 72 | 36 | 0.727 [0.666, 0.780] | 0.719 [0.654, 0.775] | 12 [8, never] / 48 [24, 64] | +0.031 [+0.005, +0.065] | 0.395 [0.366, 0.430] |
| Mixtral-8x7B (BOS) | copy prefix, templates, neutral verb | names | 256 | 128 | 0.461 [0.436, 0.487] | 0.463 [0.437, 0.490] | 4 [3, 4] / 4 [3, 4] | +0.269 [+0.251, +0.288] | 0.431 [0.422, 0.441] |
| Mixtral-8x7B (BOS) | copy prefix, templates, neutral verb | objects | 0 | 0 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | copy prefix, templates, neutral verb | assoc neutral | 228 | 114 | 0.467 [0.441, 0.493] | 0.468 [0.441, 0.496] | 4 [3, 4] / 4 [3, 4] | +0.267 [+0.248, +0.286] | 0.433 [0.423, 0.443] |
| Mixtral-8x7B (BOS) | copy prefix, templates, neutral verb | assoc not neutral | 28 | 14 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | verb templates ent->attr, orig verb | names | 256 | 128 | 0.824 [0.811, 0.837] | 0.823 [0.811, 0.837] | 10 [8, 11] / 24 [24, 32] | -0.005 [-0.016, +0.007] | 0.483 [0.466, 0.503] |
| Qwen3-30B-A3B-Base | verb templates ent->attr, orig verb | objects | 0 | 0 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | verb templates ent->attr, orig verb | assoc neutral | 218 | 109 | 0.819 [0.805, 0.833] | 0.818 [0.804, 0.832] | 9 [8, 10] / 24 [16, 24] | +0.000 [-0.012, +0.013] | 0.487 [0.468, 0.507] |
| Qwen3-30B-A3B-Base | verb templates ent->attr, orig verb | assoc not neutral | 38 | 19 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | verb templates ent->attr, orig verb | names | 256 | 128 | 0.677 [0.643, 0.708] | 0.677 [0.643, 0.707] | 6 [6, 7] / 7 [6, 8] | +0.069 [+0.046, +0.093] | 0.463 [0.452, 0.475] |
| Mixtral-8x7B (BOS) | verb templates ent->attr, orig verb | objects | 0 | 0 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | verb templates ent->attr, orig verb | assoc neutral | 234 | 117 | 0.687 [0.655, 0.717] | 0.687 [0.655, 0.716] | 6 [6, 7] / 7 [7, 8] | +0.060 [+0.038, +0.083] | 0.462 [0.451, 0.474] |
| Mixtral-8x7B (BOS) | verb templates ent->attr, orig verb | assoc not neutral | 22 | 11 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | verb templates ent->attr, neutral verb | names | 256 | 128 | 0.816 [0.800, 0.831] | 0.817 [0.802, 0.832] | 10 [9, 11] / 32 [24, 48] | +0.004 [-0.006, +0.014] | 0.482 [0.463, 0.503] |
| Qwen3-30B-A3B-Base | verb templates ent->attr, neutral verb | objects | 0 | 0 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | verb templates ent->attr, neutral verb | assoc neutral | 200 | 100 | 0.817 [0.799, 0.834] | 0.818 [0.801, 0.834] | 9 [9, 11] / 32 [24, 48] | +0.005 [-0.006, +0.016] | 0.482 [0.460, 0.505] |
| Qwen3-30B-A3B-Base | verb templates ent->attr, neutral verb | assoc not neutral | 56 | 28 | 0.812 [0.783, 0.841] | 0.815 [0.786, 0.844] | 10 [7, 15] / 32 [16, 48] | -0.002 [-0.019, +0.017] | 0.482 [0.440, 0.527] |
| Mixtral-8x7B (BOS) | verb templates ent->attr, neutral verb | names | 256 | 128 | 0.626 [0.587, 0.664] | 0.627 [0.588, 0.665] | 6 [5, 6] / 6 [5, 7] | +0.104 [+0.075, +0.134] | 0.445 [0.433, 0.456] |
| Mixtral-8x7B (BOS) | verb templates ent->attr, neutral verb | objects | 0 | 0 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | verb templates ent->attr, neutral verb | assoc neutral | 222 | 111 | 0.652 [0.616, 0.688] | 0.653 [0.616, 0.689] | 6 [5, 7] / 7 [6, 8] | +0.081 [+0.055, +0.110] | 0.445 [0.433, 0.458] |
| Mixtral-8x7B (BOS) | verb templates ent->attr, neutral verb | assoc not neutral | 34 | 17 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | verb templates attr->entity (F1), orig verb | names | 74 | 37 | 0.197 [0.151, 0.242] | 0.197 [0.151, 0.242] | 4 [3, 7] / 24 [10, 32] | +0.283 [+0.243, +0.327] | 0.151 [0.136, 0.166] |
| Qwen3-30B-A3B-Base | verb templates attr->entity (F1), orig verb | objects | 0 | 0 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | verb templates attr->entity (F1), orig verb | assoc neutral | 52 | 26 | 0.195 [0.143, 0.250] | 0.195 [0.143, 0.250] | 4 [2, 7] / 24 [8, 32] | +0.277 [+0.229, +0.327] | 0.154 [0.136, 0.174] |
| Qwen3-30B-A3B-Base | verb templates attr->entity (F1), orig verb | assoc not neutral | 22 | 11 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | attr->entity, natural sentence (F2E) | names | 12 | 6 | – | – | – | – | – |
| Qwen3-30B-A3B-Base | attr->entity, natural sentence (F2E) | objects | 52 | 26 | 0.108 [-0.051, 0.250] | 0.108 [-0.051, 0.250] | 1 [1, 2] / 1 [1, 3] | +0.324 [+0.225, +0.454] | 0.290 [0.244, 0.335] |
| Mixtral-8x7B (BOS) | attr->entity, natural sentence (F2E) | names | 4 | 2 | – | – | – | – | – |
| Mixtral-8x7B (BOS) | attr->entity, natural sentence (F2E) | objects | 32 | 16 | – | – | – | – | – |
