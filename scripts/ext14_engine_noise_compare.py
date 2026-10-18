"""ext14 (p6-engine): is a model's bf16 engine-vs-transformers deviation transformers' own bf16 noise?

Inputs: two full-mode JSONs of scripts/ext14_engine_verify.py on the SAME units and plan (same --units, both --light or
both not), one bf16 (transformers bf16 vs engine bf16) and one fp32 (transformers fp32 vs engine fp32, e.g. --offload).
Rows are matched by (tag, unit); rescue = Delta(patched) - Delta(parent). Reports correlations and mean |differences| of
the rescues of transformers bf16, engine bf16 and engine fp32 against transformers fp32 (the reference), and
transformers bf16 vs engine bf16.
Verdict "noise" if corr(engine bf16, fp32) >= corr(transformers bf16, fp32) - 0.01 and mean |engine bf16 - fp32| <=
1.5 x mean |transformers bf16 - fp32| + 0.02, and engine fp32 = transformers fp32 (r >= 0.999).
Usage: python scripts/ext14_engine_noise_compare.py <bf16.json> <fp32.json> [--out results/verify_ext14_<key>_noise.json]
"""
import json, sys
import numpy as np


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out in args:
        args.remove(out)
    rb, rf = json.load(open(args[0])), json.load(open(args[1]))
    assert rb["dtype"] == "bf16" and rf["dtype"] == "fp32", (rb["dtype"], rf["dtype"])
    fb = {(r["tag"], r["unit"]): r for r in rb["rows"]}
    ff = {(r["tag"], r["unit"]): r for r in rf["rows"]}
    keys = sorted(set(fb) & set(ff))
    hb = np.array([fb[k]["hf_delta"] - fb[k]["hf_parent"] for k in keys])
    eb = np.array([fb[k]["eng_delta"] - fb[k]["eng_parent"] for k in keys])
    hf = np.array([ff[k]["hf_delta"] - ff[k]["hf_parent"] for k in keys])
    ef = np.array([ff[k]["eng_delta"] - ff[k]["eng_parent"] for k in keys])
    rep = {"bf16_json": args[0], "fp32_json": args[1], "n_matched": len(keys),
           "corr": {"hf_bf16_vs_fp32": corr(hb, hf), "eng_bf16_vs_fp32": corr(eb, hf), "eng_bf16_vs_hf_bf16": corr(eb, hb),
                    "eng_fp32_vs_hf_fp32": corr(ef, hf)},
           "mean_absdiff": {"hf_bf16_vs_fp32": float(np.abs(hb - hf).mean()), "eng_bf16_vs_fp32": float(np.abs(eb - hf).mean()),
                            "eng_bf16_vs_hf_bf16": float(np.abs(eb - hb).mean()), "eng_fp32_vs_hf_fp32": float(np.abs(ef - hf).mean())},
           "fp32_rescue_meanabs": float(np.abs(hf).mean())}
    c, m = rep["corr"], rep["mean_absdiff"]
    rep["verdict"] = "noise" if (c["eng_fp32_vs_hf_fp32"] is not None and c["eng_fp32_vs_hf_fp32"] >= 0.999
                                 and c["eng_bf16_vs_fp32"] >= c["hf_bf16_vs_fp32"] - 0.01
                                 and m["eng_bf16_vs_fp32"] <= 1.5 * m["hf_bf16_vs_fp32"] + 0.02) else "engine"
    print(json.dumps(rep, indent=1))
    if out:
        json.dump(rep, open(out, "w"), indent=1)
    sys.exit(0 if rep["verdict"] == "noise" else 1)


if __name__ == "__main__":
    main()
