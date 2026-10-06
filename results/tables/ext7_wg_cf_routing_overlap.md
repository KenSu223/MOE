Expected fraction of the top-k experts that two clean prompts share at the final position (same layer): two WinoGrande prompts, two CounterFact prompts, or one of each.

**Qwen3-30B-A3B-Base** (1384 WinoGrande / 215 CounterFact clean prompts; top-8 of 128; chance 0.062)

| Layers | WG–WG | CF–CF | WG–CF |
|---|---|---|---|
| first third L0–L15 | 0.50 | 0.36 | 0.14 |
| middle third L16–L31 | 0.48 | 0.40 | 0.10 |
| last third L32–L47 | 0.56 | 0.46 | 0.15 |
| expert band L39–L44 | 0.59 | 0.45 | 0.15 |

| Layer | experts in ≥ 50 % of WinoGrande prompts | experts in ≥ 50 % of CounterFact prompts |
|---|---|---|
| L39 | E007 E008 E069 E071 E082 E095 E116 | E003 E006 E034 E040 E089 |
| L40 | E007 E037 E040 E079 E115 E117 | E020 E030 E080 E105 E121 E127 |
| L41 | E052 E053 E062 E085 E106 E117 | E001 E014 E045 E097 |
| L42 | E022 E037 E055 E059 E073 E123 E125 | E023 E065 E101 E115 |
| L43 | E008 E036 E049 E081 E097 E114 | E004 E005 E036 E046 E104 |
| L44 | E022 E033 E037 E054 E080 E116 E122 | E006 E027 E054 E056 E069 E071 |

**Mixtral-8x7B (BOS)** (1316 WinoGrande / 213 CounterFact clean prompts; top-2 of 8; chance 0.250)

| Layers | WG–WG | CF–CF | WG–CF |
|---|---|---|---|
| first third L0–L10 | 0.48 | 0.40 | 0.25 |
| middle third L11–L21 | 0.67 | 0.47 | 0.34 |
| last third L22–L31 | 0.62 | 0.41 | 0.26 |
| expert band L18–L21 | 0.67 | 0.42 | 0.32 |

| Layer | experts in ≥ 50 % of WinoGrande prompts | experts in ≥ 50 % of CounterFact prompts |
|---|---|---|
| L18 | E003 E005 | E001 E006 |
| L19 | E004 E006 | E002 E006 |
| L20 | E000 | E005 |
| L21 | E004 E006 | E001 |
