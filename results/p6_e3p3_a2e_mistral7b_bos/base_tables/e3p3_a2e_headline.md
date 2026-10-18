**e3p3_a2e: headline numbers with cluster-bootstrap 95% CIs (unit = item; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main, validation | 106 | 106 | 0.56 [0.54, 0.57] | 1.476 [1.415, 1.542] | 1.182 [1.119, 1.261] | 0.921 [0.856, 0.987] | 0.079 [0.013, 0.144] | 0.517 [0.483, 0.550] | L17 0.145 | L16 0.307 |
| mistral7b_bos | main, all | 106 | 106 | 0.56 [0.54, 0.57] | 1.476 [1.415, 1.542] | 1.182 [1.119, 1.261] | 0.921 [0.856, 0.987] | 0.079 [0.013, 0.144] | 0.517 [0.483, 0.550] | L17 0.145 | L16 0.307 |
| mistral7b_bos | objects (main, all) | 97 | 97 | 0.54 [0.53, 0.56] | 1.435 [1.378, 1.504] | 1.198 [1.130, 1.278] | 0.890 [0.825, 0.955] | 0.110 [0.045, 0.175] | 0.518 [0.483, 0.554] | L17 0.145 | L16 0.293 |
| mistral7b_bos | probe-neutral (main, all) | 106 | 106 | 0.56 [0.54, 0.57] | 1.476 [1.415, 1.542] | 1.182 [1.119, 1.261] | 0.921 [0.856, 0.987] | 0.079 [0.013, 0.144] | 0.517 [0.483, 0.550] | L17 0.145 | L16 0.307 |
