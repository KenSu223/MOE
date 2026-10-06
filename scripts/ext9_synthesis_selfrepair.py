"""Phase-4 synthesis (coordinator): knockout effect vs the expert's own direct write on the same clean prompts.

For each target expert and its own task: on clean prompts whose final position routes to the expert (ext11-writer Part C,
results/<model>_writer_routing/final_routing.parquet; DLA of the expert's clean contribution c_e on W_U[r] - W_U[r'], final
norm frozen at the prompt's own scale), compare
    direct write     W = mean DLA(c_e)                                   (what removing c_e would cost if nothing reacted)
    knockout effect  K = -(mean Delta_knockout - mean Delta_baseline)    (ext9-knockout, results/<model>_knockout rows)
ratio K / W = share of the direct write that is actually lost; 1 - K / W = compensation by the rest of the network
(self-repair). Prompts are matched by token ids + true / foil ids. zero mode (slot dropped, routing unchanged) is the clean
comparison; reroute (next-best expert takes the slot) adds the replacement's own write. Bootstrap over prompts (2,000).
Caveats: the two runs are different passes (bf16 batch noise, per-item SD ~0.2-0.4 logits; means over hundreds of prompts);
DLA freezes the final norm (single-expert error <= 1 %, ext11 Part B); for experts that act mostly indirectly (ext11's
"computers") DLA understates the expert's effect, so K / W is not a compensation measure there.

Usage: python scripts/ext9_synthesis_selfrepair.py
Outputs results/tables/ext9_synthesis_selfrepair.{md,csv}
"""
import glob
import json

import numpy as np
import pandas as pd

R = "/home/ubuntu/MOE/results"
RUNS = [("Qwen3-30B-A3B-Base", "qwen3_knockout", "qwen3_writer_routing", [(41, 117, "wg"), (44, 69, "cf"), (42, 115, "cf")]),
        ("Mixtral-8x7B (BOS)", "mixtral_bos_knockout", "mixtral_bos_writer_routing",
         [(20, 0, "wg"), (18, 1, "cf"), (19, 2, "cf"), (21, 1, "cf")])]
TASK = {"wg": "WinoGrande", "cf": "CounterFact"}


def key(ids, t, f):
    ids = ids if isinstance(ids, str) else json.dumps([int(x) for x in ids])
    return json.dumps(json.loads(ids)) + f"|{int(t)}|{int(f)}"


def main():
    out = []
    for label, ko, wr, targets in RUNS:
        rows = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{R}/{ko}/ko_rows_*short*.parquet")])
        items = pd.read_parquet(f"{R}/{ko}/items.parquet")
        pi = pd.read_parquet(f"{R}/{wr}/prompts_input.parquet")
        fr = pd.read_parquet(f"{R}/{wr}/final_routing.parquet")
        items["k"] = [key(a, b, c) for a, b, c in zip(items.ids_json, items.true_id, items.foil_id)]
        pi["k"] = [key(a, b, c) for a, b, c in zip(pi.ids, pi.true_id, pi.foil_id)]
        m = items.merge(pi[["idx", "k"]].drop_duplicates("k"), on="k", how="inner")
        base = rows[rows.sig == "none|all|reroute"].groupby("item").delta.mean()
        for l, e, task in targets:
            name = f"L{l}E{e:03d}"
            for mode in ("zero", "reroute"):
                ko_ = rows[rows.sig == f"{name}|all|{mode}"].groupby("item").delta.mean()
                if ko_.empty:
                    continue
                d = (ko_ - base.reindex(ko_.index)).rename("dko").reset_index()
                j = d.merge(m[["item", "idx", "task"]], on="item")
                j = j[j.task == task].merge(fr[(fr.layer == l) & (fr.expert == e)][["idx", "dla"]], on="idx", how="inner")
                rng = np.random.default_rng(l * 1000 + e)
                bs = []
                for _ in range(2000):
                    s = j.iloc[rng.integers(0, len(j), len(j))]
                    bs.append(-s.dko.mean() / s.dla.mean())
                ratio = -j.dko.mean() / j.dla.mean()
                out.append({"model": label, "expert": name, "task": TASK[task], "mode": mode, "n_prompts": len(j),
                            "direct_write": j.dla.mean(), "knockout_dDelta": j.dko.mean(), "lost_share": ratio,
                            "lost_lo": np.percentile(bs, 2.5), "lost_hi": np.percentile(bs, 97.5)})
    df = pd.DataFrame(out)
    df.to_csv(f"{R}/tables/ext9_synthesis_selfrepair.csv", index=False)
    md = ["**Knockout effect vs the expert's direct write on the same clean prompts (final position routed to the expert; "
          "own task; all-position knockout). Lost share = knockout change of Δ / DLA of the expert's clean output; "
          "1 − lost share = compensation by the rest of the network.**", "",
          "| Model | Expert | Task | Mode | Prompts | Direct write DLA(c_e) (logits) | Knockout ΔΔ (logits) | Lost share [95% CI] | Compensation |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        md.append(f"| {r['model']} | {r['expert']} | {r['task']} | {r['mode']} | {r['n_prompts']} | {r['direct_write']:+.3f} | "
                  f"{r['knockout_dDelta']:+.3f} | {r['lost_share']:.2f} [{r['lost_lo']:.2f}, {r['lost_hi']:.2f}] | "
                  f"{1 - r['lost_share']:.2f} |")
    open(f"{R}/tables/ext9_synthesis_selfrepair.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
