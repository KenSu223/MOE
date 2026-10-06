"""Final-position routing overlap between WinoGrande (option swap) and CounterFact STR clean prompts.

For every layer: the expected fraction of the top-k experts that two clean prompts share at the final position,
for two WinoGrande prompts (WG-WG), two CounterFact prompts (CF-CF) and one of each (WG-CF). With p_A(e) = fraction of
set A's prompts that route the final token to expert e, E|S_a ∩ S_b| = sum_e p_A(e) p_B(e) for a in A, b in B
(within a set: the unbiased version over different prompts); divided by k. Chance level (uniform routing) = k / E.
Also lists the experts routed in >= 50 % of each task's prompts in the expert band.
Data: clean rows (slot -1) of results/{wino_qwen3_str,qwen3_str,wino_mixtral_bos_str,mixtral_bos_str}/str_sweep_routing.parquet.

Usage: python scripts/ext7_wg_cf_routing_overlap.py
Outputs results/tables/ext7_wg_cf_routing_overlap.{md,csv}
"""
import numpy as np
import pandas as pd

R = "/home/ubuntu/MOE/results"
MODELS = [("Qwen3-30B-A3B-Base", "wino_qwen3_str", "qwen3_str", 8, 128, (39, 44)),
          ("Mixtral-8x7B (BOS)", "wino_mixtral_bos_str", "mixtral_bos_str", 2, 8, (18, 21))]


def freq(d, layer, n, E):
    return np.bincount(d[d.layer == layer].expert, minlength=E) / n


def main():
    rows, md = [], []
    for label, wf, cf, K, E, band in MODELS:
        W = pd.read_parquet(f"{R}/{wf}/str_sweep_routing.parquet")
        C = pd.read_parquet(f"{R}/{cf}/str_sweep_routing.parquet")
        W, C = W[W.slot == -1], C[C.slot == -1]
        nW, nC = W.case_id.nunique(), C.case_id.nunique()
        L = int(W.layer.max()) + 1
        per = []
        for l in range(L):
            pw, pc = freq(W, l, nW, E), freq(C, l, nC, E)
            ww = (np.sum((pw * nW) ** 2) - nW * K) / (nW * (nW - 1)) / K
            cc = (np.sum((pc * nC) ** 2) - nC * K) / (nC * (nC - 1)) / K
            wc = np.sum(pw * pc) / K
            per.append(dict(model=label, layer=l, wg_wg=ww, cf_cf=cc, wg_cf=wc, chance=K / E,
                            wg_common=" ".join(f"E{e:03d}" for e in np.where(pw >= 0.5)[0]),
                            cf_common=" ".join(f"E{e:03d}" for e in np.where(pc >= 0.5)[0])))
        df = pd.DataFrame(per)
        rows.append(df)
        md.append(f"**{label}** ({nW} WinoGrande / {nC} CounterFact clean prompts; top-{K} of {E}; chance {K / E:.3f})\n")
        md.append("| Layers | WG–WG | CF–CF | WG–CF |\n|---|---|---|---|")
        for name, idx in zip(("first third", "middle third", "last third"), np.array_split(np.arange(L), 3)):
            s = df.iloc[idx]
            md.append(f"| {name} L{idx[0]}–L{idx[-1]} | {s.wg_wg.mean():.2f} | {s.cf_cf.mean():.2f} | {s.wg_cf.mean():.2f} |")
        s = df[(df.layer >= band[0]) & (df.layer <= band[1])]
        md.append(f"| expert band L{band[0]}–L{band[1]} | {s.wg_wg.mean():.2f} | {s.cf_cf.mean():.2f} | {s.wg_cf.mean():.2f} |\n")
        md.append("| Layer | experts in ≥ 50 % of WinoGrande prompts | experts in ≥ 50 % of CounterFact prompts |\n|---|---|---|")
        for _, r in s.iterrows():
            md.append(f"| L{r.layer} | {r.wg_common or '–'} | {r.cf_common or '–'} |")
        md.append("")
    pd.concat(rows).to_csv(f"{R}/tables/ext7_wg_cf_routing_overlap.csv", index=False)
    head = ("Expected fraction of the top-k experts that two clean prompts share at the final position (same layer): "
            "two WinoGrande prompts, two CounterFact prompts, or one of each.\n\n")
    open(f"{R}/tables/ext7_wg_cf_routing_overlap.md", "w").write(head + "\n".join(md))
    print(head + "\n".join(md))


if __name__ == "__main__":
    main()
