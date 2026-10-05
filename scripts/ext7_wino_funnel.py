"""ext7: STR-only funnel of the WinoGrande pairs (CPU). Combines data/wino_str/funnel_<split>.json (rules W1-W6) with
results/wino_<proto>[_dev]/scan_pairs.parquet (model margins; GN columns of the first scans are ignored).
Writes results/ext7_wino_funnel.json and results/tables/ext7_wino_funnel.md"""
import json
import os

import pandas as pd

R = "/home/ubuntu/MOE/results"
D = "/home/ubuntu/MOE/data/wino_str"
LABEL = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral_nobos": "Mixtral-8x7B, no BOS", "mixtral_bos": "Mixtral-8x7B, BOS",
         "olmoe": "OLMoE-1B-7B (verification only)"}
out, lines = {}, []
for split, sfx in (("train_xl", ""), ("dev", "_dev")):
    fun = json.load(open(os.path.join(D, f"funnel_{split}.json")))
    for proto in LABEL:
        run = os.path.join(R, f"wino_{proto}{sfx}", "scan_pairs.parquet")
        if proto not in fun or not os.path.exists(run):
            continue
        f, p = fun[proto], pd.read_parquet(run)
        st = {s["rule"]: s["remaining"] for s in f["steps"]}
        m = p[p.margin]
        row = {"split": split, "proto": proto, "twins": f["W1_twins"],
               "final_word_trigger": st["W2_blank_position"], "single_token_trigger": st["W3_same_trigger_token"],
               "token_symmetric": st["W5_option_is_final"], "dedup": f["kept"],
               "correct_both": int(p.correct_both.sum()), "margin": int(len(m)),
               "margin_names": int(m.names.sum()), "margin_objects": int((~m.names).sum()),
               "margin_assoc": int(m.assoc.sum()), "margin_top1_both": int(m.top1_both.sum()),
               "margin_one_token_option": int((m.n_opt_tokens == 1).sum()), "margin_debiased": int(m.debiased.sum()),
               "kept_debiased": int(p.debiased.sum()),
               "mean_delta_a": float(m.d_clean_a.mean()), "mean_delta_b": float(m.d_clean_b.mean()),
               "mean_drop": float(m.drop_str.mean())}
        out[f"{split}/{proto}"] = row
json.dump(out, open(os.path.join(R, "ext7_wino_funnel.json"), "w"), indent=1)
hdr = ("| Split | Model / protocol | Twins | W2 sentence-final single-word trigger | W3 single-token trigger | W4–W5 token-symmetric | "
       "W6 dedup | Correct both ways | **Margin ≥ 1 both ways (primary)** | of which names / objects | assoc | top-1 both | "
       "debiased | mean Δ_A / Δ_B | mean drop |")
lines += [hdr, "|" + "---|" * 15]
for k, r in out.items():
    lines.append(f"| {r['split']} | {LABEL[r['proto']]} | {r['twins']:,} | {r['final_word_trigger']:,} | {r['single_token_trigger']:,} | "
                 f"{r['token_symmetric']:,} | {r['dedup']:,} | {r['correct_both']:,} | **{r['margin']:,}** | "
                 f"{r['margin_names']:,} / {r['margin_objects']:,} | {r['margin_assoc']:,} | {r['margin_top1_both']:,} | "
                 f"{r['margin_debiased']:,} of {r['kept_debiased']:,} | {r['mean_delta_a']:+.2f} / {r['mean_delta_b']:+.2f} | {r['mean_drop']:.2f} |")
os.makedirs(os.path.join(R, "tables"), exist_ok=True)
open(os.path.join(R, "tables", "ext7_wino_funnel.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
