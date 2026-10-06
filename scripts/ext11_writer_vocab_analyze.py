"""ext11 Part B analysis (CPU): vocabulary projections of the target experts' final-position writes.

Reads results/<run>/{vec_rows,vec_top,prompts}.parquet (scripts/ext11_writer_vocab.py); writes results/tables/ext11_B_*.md|csv
and key "B" of results/ext11_writer_summary.json.

Usage: python scripts/ext11_writer_vocab_analyze.py [--runs qwen3:qwen3_writer_vocab,mixtral:mixtral_bos_writer_vocab]
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
import numpy as np, pandas as pd
from moetrace.models import MODELS, RESULTS
from moetrace import ext11_writer as W

TAB = os.path.join(RESULTS, "tables")
MLAB = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral": "Mixtral-8x7B (BOS)"}
GROUPS = [("delta", "cf", "δ_e, CounterFact STR rows"), ("delta", "wino", "δ_e, WinoGrande STR rows"),
          ("clean", "cf_clean", "c_e, CounterFact clean prompts"), ("clean", "wino", "c_e, WinoGrande prompts"),
          ("clean", "wino_local", "c_e, WinoGrande local prompts"), ("clean", "ioi", "c_e, IOI prompts")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def analyse(model, run):
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    od = os.path.join(RESULTS, run)
    vr = pd.read_parquet(os.path.join(od, "vec_rows.parquet"))
    vt = pd.read_parquet(os.path.join(od, "vec_top.parquet"))
    pr = pd.read_parquet(os.path.join(od, "prompts.parquet"))
    meta = json.load(open(os.path.join(od, "run_meta.json")))
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model]["repo"]))
    classes = W.vocab_classes(model, tok)
    from moetrace.arch import load_spec
    Vn = int(load_spec(MODELS[model]["repo"])[0].vocab)  # rows of W_U = the ranked vocabulary
    vr["expert_name"] = [W.ename((a, b)) for a, b in zip(vr.layer, vr.expert)]
    vr["split"] = pr.split.reindex(vr.prompt.values).values if "split" in pr.columns else None
    vr["unit"] = pr.unit.reindex(vr.prompt.values).values if "unit" in pr.columns else vr.prompt.values
    vr["unit"] = np.where(pd.isna(vr["unit"]), vr.prompt.values, vr["unit"])
    vr["relation"] = pr.relation.reindex(vr.prompt.values).values if "relation" in pr.columns else ""
    vr["vec"] = np.arange(len(vr))
    targets = [W.ename(x) for x in W.model_targets(model)]
    own = {"wino": [W.ename(x) for x in W.TARGETS[model]["wino"]], "cf": [W.ename(x) for x in W.TARGETS[model]["cf"]]}
    A = json.load(open(W.SUMMARY))["A"]
    pops = {r["task"]: r["pop_top10"] for k, r in A.items() if r["model"] == model}
    piece = lambda i: tok.convert_tokens_to_string([tok.convert_ids_to_tokens(int(i))]) if model == "qwen3" else tok.convert_ids_to_tokens(int(i)).replace("▁", " ")
    out = {"model": model, "run": run, "n_vectors": int(len(vr)), "class_sizes": {k: len(v) for k, v in classes.items()}, "V": Vn,
           "experts": {}}
    rows_r, rows_c, rows_t, rows_n = [], [], [], []
    exps = list(dict.fromkeys(targets + pops.get("wino", []) + pops.get("cf", [])))
    for e in exps:
        out["experts"][e] = {}
        for kind, task, lab in GROUPS:
            s = vr[(vr.expert_name == e) & (vr.kind == kind) & (vr.task == task)]
            if len(s) < 5:
                continue
            # CounterFact delta rows: donor level; per-case aggregation for the means, row level for ranks
            st = {"n": int(len(s)), "n_units": int(s.unit.nunique()),
                  "median_rank_true": float(s.rank_true.median()), "frac_true_top10": float((s.rank_true <= 10).mean()),
                  "frac_true_top100": float((s.rank_true <= 100).mean()),
                  "median_rank_low_foil": float(s.rank_low_foil.median()), "frac_foil_bottom100": float((s.rank_low_foil <= 100).mean()),
                  "median_rank_true_in_class": float(s.rank_true_in_class.median()) if s.rank_true_in_class.notna().any() else None,
                  "median_n_class": float(s.n_class.median()) if s.n_class.notna().any() else None,
                  "median_rank_foil_in_class": float(s.rank_foil_in_class.median()) if s.rank_foil_in_class.notna().any() else None,
                  "mean_val_true_minus_foil": float((s.val_true - s.val_foil).mean()),
                  "exact_mean": float(s.exact.mean()), "frozen_mean": float(s.frozen.mean()),
                  "exact_over_frozen": float(s.exact.sum() / s.frozen.sum()) if abs(s.frozen.sum()) > 1e-9 else None,
                  "r_exact_frozen": float(np.corrcoef(s.exact, s.frozen)[0, 1]) if len(s) > 3 else None,
                  "mean_abs_err": float((s.exact - s.frozen).abs().mean()),
                  "rel_abs_err": float((s.exact - s.frozen).abs().sum() / s.frozen.abs().sum()) if s.frozen.abs().sum() > 0 else None}
            u, Wm = W.boot_matrix(s.unit.values, 1000)
            for k in classes:
                col = f"clsz_{k}"
                if col in s:
                    S = W.unit_sums(s.assign(one=1.0), [col, "one"], u)
                    st[f"z_{k}"] = W.ratio_ci(S[:, 0], S[:, 1], Wm)
                    st[f"top50_{k}"] = float(s[f"top50_{k}"].mean())
                    st[f"base_{k}"] = len(classes[k]) / Vn
            # most frequently promoted / suppressed tokens across vectors
            tp = vt.iloc[s.vec.values]
            ct = collections.Counter(int(i) for ids in tp.top_ids for i in ids[:10])
            cb = collections.Counter(int(i) for ids in tp.bot_ids for i in ids[:10])
            st["top_promoted"] = [(piece(i), c / len(s)) for i, c in ct.most_common(12)]
            st["top_suppressed"] = [(piece(i), c / len(s)) for i, c in cb.most_common(12)]
            # how often r / r' themselves are among the promoted / suppressed top-10
            st["frac_true_in_top10_list"] = float(np.mean([int(t) in set(ids[:10]) for t, ids in zip(s.true_id, tp.top_ids)]))
            out["experts"][e][f"{kind}:{task}"] = st
            role = ("target" if e in targets else "") + (" pop" if e in pops.get("wino", []) + pops.get("cf", []) else "")
            own_cls = "wg_trigger" if task.startswith("wino") else ("cf_any" if task.startswith("cf") else "ioi_names")
            rows_r.append({"model": MLAB[model], "expert": e, "vectors": lab, "n": st["n"],
                           "median rank of r (of V)": f"{st['median_rank_true']:.0f}", "r in top-10 / top-100 (%)":
                               f"{100 * st['frac_true_top10']:.0f} / {100 * st['frac_true_top100']:.0f}",
                           "median rank of r' from the bottom": f"{st['median_rank_low_foil']:.0f}",
                           "median rank of r in its class (class size)": (f"{st['median_rank_true_in_class']:.0f} ({st['median_n_class']:.0f})"
                                                                          if st["median_rank_true_in_class"] is not None else ""),
                           f"class z (own class)": f"{own_cls}: {st[f'z_{own_cls}']['v']:+.2f}" if f"z_{own_cls}" in st else "",
                           "most promoted tokens (share of vectors with it in the top-10)": "; ".join(f"{repr(a)} {100 * b:.0f}%" for a, b in st["top_promoted"][:6]),
                           "most suppressed tokens": "; ".join(f"{repr(a)} {100 * b:.0f}%" for a, b in st["top_suppressed"][:5])})
            rows_c.append({"model": MLAB[model], "expert": e, "vectors": lab, "n": st["n"],
                           **{f"z {k}": f"{st[f'z_{k}']['v']:+.2f}" for k in ("wg_trigger", "cf_place", "cf_language", "cf_organisation",
                                                                                 "cf_occupation / field", "ioi_names") if f"z_{k}" in st},
                           **{f"top-50 share {k} (base)": f"{100 * st[f'top50_{k}']:.1f} ({100 * st[f'base_{k}']:.2f})"
                              for k in ("wg_trigger", "cf_any") if f"top50_{k}" in st}})
            rows_n.append({"model": MLAB[model], "expert": e, "vectors": lab, "n": st["n"], "exact-norm mean (logits)": f"{st['exact_mean']:+.3f}",
                           "frozen-norm mean (logits)": f"{st['frozen_mean']:+.3f}",
                           "exact / frozen": f"{st['exact_over_frozen']:.3f}" if st["exact_over_frozen"] is not None else "",
                           "r": f"{st['r_exact_frozen']:.3f}" if st["r_exact_frozen"] is not None else "",
                           "mean |exact - frozen| (logits)": f"{st['mean_abs_err']:.3f}",
                           "sum |err| / sum |frozen|": f"{st['rel_abs_err']:.3f}" if st["rel_abs_err"] is not None else ""})
    # pooled exact vs frozen over every delta vector (all interest experts)
    d = vr[vr.kind == "delta"]
    out["norm_pooled"] = {t: {"n": int((d.task == t).sum()), "exact_over_frozen": float(d.exact[d.task == t].sum() / d.frozen[d.task == t].sum()),
                              "r": float(np.corrcoef(d.exact[d.task == t], d.frozen[d.task == t])[0, 1]),
                              "mean_abs_err": float((d.exact - d.frozen)[d.task == t].abs().mean()),
                              "p95_abs_err": float((d.exact - d.frozen)[d.task == t].abs().quantile(0.95)),
                              "rel_abs_err": float((d.exact - d.frozen)[d.task == t].abs().sum() / d.frozen[d.task == t].abs().sum())}
                          for t in ("cf", "wino") if (d.task == t).any()}
    out["meta_check"] = meta.get("check_frozen_vs_contrib_dla_max_abs")
    # cross-check: delta DLA of this pass vs ext8's DLA (pass 0 of the add-back runs) for the same (row, expert)
    xc = {}
    ck, wk = ("cf_qwen3", "wino_qwen3") if model == "qwen3" else ("cf_mixtral", "wino_mixtral")
    dcf = vr[(vr.kind == "delta") & (vr.task == "cf")].assign(row_id=pr.row_id.reindex(vr[(vr.kind == "delta") & (vr.task == "cf")].ref.values).values)
    e8 = pd.read_parquet(os.path.join(RESULTS, W.RUNS[ck]["out"], "addback_dla.parquet"))
    m1 = dcf.merge(e8, on=["row_id", "layer", "expert"])
    dw = vr[(vr.kind == "delta") & (vr.task == "wino")].assign(case_id=pr.case_id.reindex(vr[(vr.kind == "delta") & (vr.task == "wino")].prompt.values).values)
    e8w = pd.read_parquet(os.path.join(RESULTS, W.RUNS[wk]["out"], "addback_dla.parquet")).merge(
        pd.read_parquet(os.path.join(RESULTS, W.RUNS[wk]["out"], "addback_direct.parquet"))[["row_id", "case_id"]], on="row_id")
    m2 = dw.merge(e8w, on=["case_id", "layer", "expert"])
    for t, mm in (("cf", m1), ("wino", m2)):
        xc[t] = {"n": int(len(mm)), "r": float(np.corrcoef(mm.frozen, mm.dla)[0, 1]), "mean_abs_diff": float((mm.frozen - mm.dla).abs().mean())}
    out["xcheck_ext8"] = xc
    return out, pd.DataFrame(rows_r), pd.DataFrame(rows_c), pd.DataFrame(rows_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="qwen3:qwen3_writer_vocab,mixtral:mixtral_bos_writer_vocab")
    args = ap.parse_args()
    res, R, Cc, N = {}, [], [], []
    for spec in args.runs.split(","):
        model, run = spec.split(":")
        mp = os.path.join(RESULTS, run, "run_meta.json")
        if not os.path.exists(mp) or not json.load(open(mp)).get("complete"):
            log(f"{run} not complete, skipped")
            continue
        res[model], r, c, n = analyse(model, run)
        R.append(r), Cc.append(c), N.append(n)
    if not res:
        return
    notes = {
        "ext11_B_ranks": "Vocabulary projection of the expert's write at the final position, ((v ⊙ γ) @ W_U^T) / rms(h_final) with the norm frozen at the "
                         "corrupted run (δ_e = c_e(clean) − c_e(corrupt)) or at the prompt's own run (c_e). Ranks over the full vocabulary "
                         "(1 = most promoted; r' from the bottom: 1 = most suppressed); class = WinoGrande trigger vocabulary (WinoGrande rows) or "
                         "the CounterFact objects of the case's relation group (CounterFact rows).",
        "ext11_B_classes": "Mean projection over the class tokens minus the vocabulary mean, in SD units of the vector's projection (z), and the share of the "
                           "50 most promoted tokens that belong to the class (base = class size / vocabulary size).",
        "ext11_B_norm": "Exact final RMSNorm vs frozen norm for the single expert's direct effect: δ_e: Δ(norm(h_corrupt + δ_e)) − Δ(norm(h_corrupt)) vs "
                        "δ_e·u / rms(h_corrupt); c_e: Δ(norm(h_clean)) − Δ(norm(h_clean − c_e)) vs c_e·u / rms(h_clean) (fp32, same path).",
    }
    for name, parts in (("ext11_B_ranks", R), ("ext11_B_classes", Cc), ("ext11_B_norm", N)):
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
        with open(os.path.join(TAB, name + ".md"), "w") as f:
            f.write(W.md_table(df) + "\n\n" + notes[name] + "\n")
    W.update_summary("B", res)
    log("written key B, tables ext11_B_*")


if __name__ == "__main__":
    main()
