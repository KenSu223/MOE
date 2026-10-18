# Random-order experts needed to match greedy's / the oracle's first k (Phase-5 add-back, validation)

k_rand(k) = smallest k' with r_random(k') >= r_greedy(k) (log-k interpolation); 'never' = random order never reaches it
(greedy above the all-MoE ceiling, where random ends). Written by scripts/ext13ab_plot.py (Phase 6, P6-0).

| Condition | Family | Model | K | greedy 1 | greedy 5 | greedy 10 | greedy 15 | oracle 1 | oracle 5 | oracle 10 | oracle 15 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attr->entity, fill-in-the-blank (F1) | main | Qwen3-30B-A3B-Base | 384 | 240 | never | never | never | 240 | 344 | never | never |
| attr->entity, fill-in-the-blank (F1) | main | Mixtral-8x7B (BOS) | 64 | 37 | never | never | never | 38 | never | never | never |
| attr->entity, fill-in-the-blank (F1) [replication] | rep | Qwen3-30B-A3B-Base | 384 | 335 | never | never | never | 338 | never | never | never |
| direction: F1 on the Phase-3 twins | main | Qwen3-30B-A3B-Base | 384 | never | never | never | never | never | never | never | never |
| direction: F1 on the Phase-3 twins | main | Mixtral-8x7B (BOS) | 64 | 22 | never | never | never | 22 | never | never | never |
| direction: option swap on the same twins | main | Qwen3-30B-A3B-Base | 384 | 80 | 212 | 276 | 300 | 81 | 183 | 228 | 251 |
| direction: option swap on the same twins | main | Mixtral-8x7B (BOS) | 64 | 13 | 38 | 54 | never | 13 | 36 | 49 | 57 |
| attr->entity, natural sentence (F2E) | main | Qwen3-30B-A3B-Base | 384 | 279 | never | never | never | 282 | never | never | never |
| attr->entity, natural sentence (F2E) | main | Mixtral-8x7B (BOS) | 64 | never | never | never | never | never | never | never | never |
| verb templates ent->attr, orig verb | main | Qwen3-30B-A3B-Base | 384 | 82 | 216 | 290 | 318 | 82 | 188 | 236 | 263 |
| verb templates ent->attr, orig verb | main | Mixtral-8x7B (BOS) | 64 | 14 | 42 | 60 | never | 14 | 39 | 53 | 62 |
| verb templates ent->attr, neutral verb | main | Qwen3-30B-A3B-Base | 384 | 80 | 218 | 291 | 316 | 79 | 189 | 228 | 250 |
| verb templates ent->attr, neutral verb | main | Mixtral-8x7B (BOS) | 64 | 14 | 43 | never | never | 14 | 40 | 59 | never |
| verb templates attr->entity (F1), orig verb | main | Qwen3-30B-A3B-Base | 384 | 249 | 382 | never | never | 248 | 320 | 346 | 360 |
| copy prefix, templates, orig verb | main | Qwen3-30B-A3B-Base | 384 | 83 | 258 | 331 | 361 | 83 | 202 | 258 | 284 |
| copy prefix, templates, orig verb | main | Mixtral-8x7B (BOS) | 64 | 13 | 59 | never | never | 13 | 48 | never | never |
| copy prefix, templates, neutral verb | main | Qwen3-30B-A3B-Base | 384 | 86 | 244 | 318 | 346 | 86 | 189 | 242 | 267 |
| copy prefix, templates, neutral verb | main | Mixtral-8x7B (BOS) | 64 | 12 | never | never | never | 12 | never | never | never |
| copy prefix, Phase-3 twins | main | Qwen3-30B-A3B-Base | 384 | 69 | 199 | 250 | 265 | 70 | 158 | 192 | 209 |
| copy prefix, Phase-3 twins | main | Mixtral-8x7B (BOS) | 64 | 13 | 41 | 59 | never | 13 | 39 | 54 | 62 |
| copy prefix, Phase-3 twins [replication] | rep | Qwen3-30B-A3B-Base | 384 | 73 | 210 | 267 | 288 | 73 | 164 | 195 | 214 |
| copy prefix, Phase-3 twins [replication] | rep | Mixtral-8x7B (BOS) | 64 | 14 | 41 | 60 | never | 14 | 39 | 54 | 62 |
