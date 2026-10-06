**4c: Mixtral no BOS vs BOS on identical pairs (paired pair bootstrap, 95% CI)**

| Pairs | Quantity | no BOS | BOS | no BOS − BOS (paired) |
|---|---|---|---|---|
| main validation | attention share | 0.349 [0.337, 0.362] | 0.344 [0.331, 0.357] | +0.005 [-0.002, +0.013] |
| main validation | AUC+ attention / drop | 0.767 [0.729, 0.808] | 0.728 [0.692, 0.765] | +0.039 [+0.002, +0.074] |
| main validation | AUC+ MoE / drop | 1.432 [1.378, 1.488] | 1.389 [1.346, 1.439] | +0.042 [+0.003, +0.075] |
| main validation | direct-path attention A_dir | 0.288 [0.266, 0.311] | 0.288 [0.265, 0.313] | -0.001 [-0.007, +0.006] |
| main validation | direct-path MoE M_dir | 0.712 [0.689, 0.734] | 0.712 [0.687, 0.735] | +0.001 [-0.006, +0.007] |
| main validation | all-MoE M (denoise) | 0.795 [0.773, 0.815] | 0.788 [0.765, 0.810] | +0.007 [+0.001, +0.013] |
| main validation | mean drop (logits) | 7.562 [7.136, 7.988] | 7.685 [7.251, 8.138] | -0.123 [-0.273, +0.023] |
| main all | attention share | 0.356 [0.347, 0.366] | 0.350 [0.340, 0.360] | +0.006 [+0.001, +0.012] |
| main all | AUC+ attention / drop | 0.767 [0.739, 0.797] | 0.730 [0.705, 0.759] | +0.037 [+0.011, +0.062] |
| main all | AUC+ MoE / drop | 1.386 [1.346, 1.428] | 1.356 [1.321, 1.393] | +0.030 [+0.004, +0.057] |
| main all | direct-path attention A_dir | 0.300 [0.283, 0.318] | 0.300 [0.282, 0.318] | +0.001 [-0.004, +0.006] |
| main all | direct-path MoE M_dir | 0.700 [0.682, 0.717] | 0.700 [0.682, 0.718] | -0.001 [-0.006, +0.004] |
| main all | all-MoE M (denoise) | 0.778 [0.761, 0.794] | 0.774 [0.756, 0.791] | +0.004 [-0.001, +0.009] |
| main all | mean drop (logits) | 7.643 [7.347, 7.936] | 7.782 [7.466, 8.087] | -0.138 [-0.256, -0.021] |
