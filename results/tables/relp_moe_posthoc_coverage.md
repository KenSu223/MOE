| model/task | tokens (A) | patched positions | pairs | patched units |
|---|---|---|---|---|
| qwen3/paper/simple | The Ġboys | Ġboys | 380 | 421,080 |
| qwen3/paper/nounpp | The Ġman Ġnear Ġthe Ġchairs | Ġman, Ġnear, Ġthe, Ġchairs | 380 | 1,564,852 |
| qwen3/paper/rc | The Ġlawyer Ġthat Ġthe Ġarchitects Ġlike | Ġlawyer, Ġthat, Ġthe, Ġarchitects, Ġlike | 380 | 1,955,638 |
| qwen3/paper/within_rc | The Ġfarmer Ġthat Ġthe Ġminister | Ġminister | 380 | 396,732 |
| mixtral/paper/simple | <s> ▁The ▁lawyers | ▁lawyers | 316 | 70,980 |
| mixtral/paper/nounpp | <s> ▁The ▁teachers ▁near ▁the ▁chairs | ▁teachers, ▁near, ▁the, ▁chairs | 380 | 323,258 |
| mixtral/paper/rc | <s> ▁The ▁victim ▁that ▁the ▁chef ▁visits | ▁victim, ▁that, ▁the, ▁chef, ▁visits | 380 | 400,540 |
| mixtral/paper/within_rc | <s> ▁The ▁poet ▁that ▁the ▁parents | ▁parents | 380 | 85,496 |

| model | RelP top-20 expert-positions on test pairs: routed at both | clean only (injected, no renormalisation) | corrupt only (removed) | neither (no-op) |
|---|---|---|---|---|
| qwen3 | 74% | 10% | 12% | 5% |
| mixtral | 73% | 8% | 8% | 11% |
