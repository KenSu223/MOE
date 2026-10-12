**GPU check (Qwen3, validation rows; same-pass values): add back all experts except the cross-fitted negative ones; late-MoE reaction test**

| Condition | Row | Value / drop | Kind |
|---|---|---|---|
| dir_opt_qwen3 | all_moe | +0.813 [+0.796, +0.831] | variant |
| dir_opt_qwen3 | int_moe | +0.769 [+0.748, +0.791] | variant |
| dir_opt_qwen3 | intattn | +0.957 [+0.949, +0.965] | variant |
| dir_opt_qwen3 | intattn_latemoe | +0.949 [+0.940, +0.958] | variant |
| dir_opt_qwen3 | intblock | +0.962 [+0.955, +0.968] | variant |
| dir_opt_qwen3 | intblock_latemoe | +0.954 [+0.946, +0.962] | variant |
| dir_opt_qwen3 | lateonly | -0.041 [-0.086, +0.004] | variant |
| dir_opt_qwen3 | negonly | -0.226 [-0.296, -0.158] | variant |
| dir_opt_qwen3 | noneg | +0.797 [+0.776, +0.817] | variant |
| dir_opt_qwen3 | noneg_early | +0.796 [+0.777, +0.815] | variant |
| dir_opt_qwen3 | noneg_late | +0.815 [+0.796, +0.834] | variant |
| dir_opt_qwen3 | noneg_row | +0.822 [+0.807, +0.838] | variant |
| dir_opt_qwen3 | noneg - all_moe | -0.016 [-0.031, -0.003] | difference |
| dir_opt_qwen3 | noneg_late - all_moe | +0.002 [-0.010, +0.014] | difference |
| dir_opt_qwen3 | noneg_early - all_moe | -0.017 [-0.022, -0.013] | difference |
| dir_opt_qwen3 | noneg_row - all_moe | +0.010 [+0.001, +0.020] | difference |
| dir_opt_qwen3 | late mismatch | MoE upstream (int_moe - all_moe) | -0.044 [-0.056, -0.032] | difference |
| dir_opt_qwen3 | late mismatch | attention upstream (intattn - intattn_latemoe) | +0.008 [+0.004, +0.013] | difference |
| dir_opt_qwen3 | late mismatch | both upstream (intblock - intblock_latemoe) | +0.008 [+0.004, +0.012] | difference |
| dir_opt_qwen3 | negonly + noneg - all_moe (additivity) | -0.242 [-0.302, -0.183] | difference |
| dir_opt_qwen3 | oracle overshoot (add-back run) | +0.000 [-0.009, +0.010] | reference |
| dir_f1_qwen3 | all_moe | +0.104 [+0.041, +0.165] | variant |
| dir_f1_qwen3 | int_moe | +0.428 [+0.405, +0.452] | variant |
| dir_f1_qwen3 | intattn | +0.922 [+0.910, +0.932] | variant |
| dir_f1_qwen3 | intattn_latemoe | +0.895 [+0.879, +0.911] | variant |
| dir_f1_qwen3 | intblock | +0.928 [+0.917, +0.938] | variant |
| dir_f1_qwen3 | intblock_latemoe | +0.895 [+0.880, +0.909] | variant |
| dir_f1_qwen3 | lateonly | -0.540 [-0.609, -0.476] | variant |
| dir_f1_qwen3 | negonly | -0.693 [-0.764, -0.625] | variant |
| dir_f1_qwen3 | noneg | +0.470 [+0.449, +0.492] | variant |
| dir_f1_qwen3 | noneg_early | +0.094 [+0.032, +0.153] | variant |
| dir_f1_qwen3 | noneg_late | +0.476 [+0.453, +0.499] | variant |
| dir_f1_qwen3 | noneg_row | +0.491 [+0.467, +0.514] | variant |
| dir_f1_qwen3 | noneg - all_moe | +0.366 [+0.319, +0.416] | difference |
| dir_f1_qwen3 | noneg_late - all_moe | +0.371 [+0.326, +0.420] | difference |
| dir_f1_qwen3 | noneg_early - all_moe | -0.011 [-0.019, -0.002] | difference |
| dir_f1_qwen3 | noneg_row - all_moe | +0.386 [+0.343, +0.432] | difference |
| dir_f1_qwen3 | late mismatch | MoE upstream (int_moe - all_moe) | +0.324 [+0.275, +0.375] | difference |
| dir_f1_qwen3 | late mismatch | attention upstream (intattn - intattn_latemoe) | +0.027 [+0.020, +0.034] | difference |
| dir_f1_qwen3 | late mismatch | both upstream (intblock - intblock_latemoe) | +0.033 [+0.026, +0.039] | difference |
| dir_f1_qwen3 | negonly + noneg - all_moe (additivity) | -0.328 [-0.364, -0.290] | difference |
| dir_f1_qwen3 | oracle overshoot (add-back run) | +0.375 [+0.331, +0.423] | reference |
| f1_qwen3 | all_moe | +0.238 [+0.177, +0.296] | variant |
| f1_qwen3 | int_moe | +0.466 [+0.432, +0.501] | variant |
| f1_qwen3 | intattn | +0.960 [+0.952, +0.969] | variant |
| f1_qwen3 | intattn_latemoe | +0.946 [+0.931, +0.960] | variant |
| f1_qwen3 | intblock | +0.959 [+0.950, +0.967] | variant |
| f1_qwen3 | intblock_latemoe | +0.941 [+0.927, +0.954] | variant |
| f1_qwen3 | lateonly | -0.398 [-0.460, -0.339] | variant |
| f1_qwen3 | negonly | -0.515 [-0.575, -0.460] | variant |
| f1_qwen3 | noneg | +0.492 [+0.459, +0.525] | variant |
| f1_qwen3 | noneg_early | +0.229 [+0.170, +0.287] | variant |
| f1_qwen3 | noneg_late | +0.496 [+0.462, +0.530] | variant |
| f1_qwen3 | noneg_row | +0.517 [+0.487, +0.548] | variant |
| f1_qwen3 | noneg - all_moe | +0.254 [+0.217, +0.294] | difference |
| f1_qwen3 | noneg_late - all_moe | +0.258 [+0.222, +0.297] | difference |
| f1_qwen3 | noneg_early - all_moe | -0.009 [-0.021, +0.003] | difference |
| f1_qwen3 | noneg_row - all_moe | +0.279 [+0.242, +0.319] | difference |
| f1_qwen3 | late mismatch | MoE upstream (int_moe - all_moe) | +0.228 [+0.189, +0.269] | difference |
| f1_qwen3 | late mismatch | attention upstream (intattn - intattn_latemoe) | +0.014 [+0.007, +0.023] | difference |
| f1_qwen3 | late mismatch | both upstream (intblock - intblock_latemoe) | +0.018 [+0.011, +0.025] | difference |
| f1_qwen3 | negonly + noneg - all_moe (additivity) | -0.261 [-0.302, -0.225] | difference |
| f1_qwen3 | oracle overshoot (add-back run) | +0.276 [+0.235, +0.322] | reference |
