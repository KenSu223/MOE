"""ext3: analyse the BOS-mechanism runs and write tables, figures and the section results/sections/ext3_bos_mechanism.md.

Inputs (whatever exists): results/<run_dir>/ for every variant in moetrace.ext3_variants (Mixtral + Qwen3), the raw
diagnostics under /opt/dlami/nvme/moe_ext3/<run_dir>/, the corpus statistics results/mixtral_*_corpus_*/corpus_stats.json,
and an optional hand-written results/sections/ext3_conclusion.md that is appended verbatim.
Usage: python scripts/ext3_analyze.py
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moetrace import analysis as A
from moetrace.models import MODELS, RESULTS
from moetrace.stats import summarize
from moetrace.ext3_variants import MIXTRAL_VARIANTS, QWEN3_VARIANTS, load_diag, DIAG_ROOT

TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")
SECTIONS = os.path.join(RESULTS, "sections")
for d in (TABLES, FIGS, SECTIONS):
    os.makedirs(d, exist_ok=True)
NUM = {}


def fmt(s):
    return f"{s['mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"


def md_table(header, rows, path_stem=None):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    s = "\n".join(lines) + "\n"
    if path_stem:
        with open(path_stem + ".md", "w") as f:
            f.write(s)
        pd.DataFrame(rows, columns=header).to_csv(path_stem + ".csv", index=False)
    return s


def have(run_dir):
    return os.path.exists(os.path.join(RESULTS, run_dir, "sweep_rows.parquet"))


def clean_sets(md, layer):
    r = md.routing
    r = r[(r.run == "clean") & (r.layer == layer)]
    return {int(c): frozenset(int(e) for e in g.expert) for c, g in r.groupby("case_id")}


def clean_top1(md, layer):
    r = md.routing
    r = r[(r.run == "clean") & (r.layer == layer) & (r.slot == 0)]
    return dict(zip(r.case_id.astype(int), r.expert.astype(int)))


def agreement_curve(md_a, md_b, ids, L):
    ra = md_a.routing[md_a.routing.run == "clean"]
    rb = md_b.routing[md_b.routing.run == "clean"]
    sa = {(int(c), int(l)): frozenset(g.expert.astype(int)) for (c, l), g in ra.groupby(["case_id", "layer"])}
    sb = {(int(c), int(l)): frozenset(g.expert.astype(int)) for (c, l), g in rb.groupby(["case_id", "layer"])}
    return np.array([np.mean([sa[(c, l)] == sb[(c, l)] for c in ids]) for l in range(L)])


# ---------------------------------------------------------------------------------------------------------------
def substitution_table(model, variants, layer, paper_e, n_controls, ref_keys=("bos", "nobos")):
    mds = {k: A.load_model(v.run_dir) for k, v in variants.items() if have(v.run_dir)}
    if not mds:
        return None, {}
    ref = {k: mds[k] for k in ref_keys if k in mds}
    L = int(mds[next(iter(mds))].routing.layer.max()) + 1
    any_md = next(iter(mds.values()))
    disc, val = any_md.ids("paper", "discovery"), any_md.ids("paper", "validation")
    ids = disc + val
    ref_sets = {k: clean_sets(m, layer) for k, m in ref.items()}
    ref_top1 = {k: clean_top1(m, layer) for k, m in ref.items()}
    rows, curves = [], {}
    E = MODELS[model]["paper_expert"]
    alt_e = 2 if model == "mixtral" else None
    for k, v in variants.items():
        if k not in mds:
            continue
        md = mds[k]
        cs = clean_sets(md, layer)
        t1 = clean_top1(md, layer)
        la = A.layer_analysis(md, "paper")
        ct = md.cases
        agree = {r: np.mean([cs[c] == ref_sets[r][c] for c in ids]) for r in ref_sets}
        agree1 = {r: np.mean([t1[c] == ref_top1[r][c] for c in ids]) for r in ref_top1}
        for r in ref:
            curves[(k, r)] = agreement_curve(md, ref[r], ids, L)
        act = lambda e, sub: sum(e in cs[c] for c in sub)
        row = {"variant": k, "position 0": v.label, "RoPE pos of 1st content token": v.pos_offset + len(v.prefix_ids) + (1 if (v.special_tokens and model == "mixtral") else 0),
               "strict pass": f"{int(ct.strict.sum())}/{len(ct)}", "mean Δ_clean": f"{ct.delta_clean.mean():+.2f}",
               "L* (disc)": f"L{la['L_star']}", f"val rescue @L{layer}": f"{md.R.loc[val, layer].mean():+.3f}",
               f"agree@L{layer} w/ {ref_keys[0]} (set/top1)": (f"{agree[ref_keys[0]]:.2f}/{agree1[ref_keys[0]]:.2f}" if ref_keys[0] in agree else "n/a"),
               f"agree@L{layer} w/ {ref_keys[1]} (set/top1)": (f"{agree[ref_keys[1]]:.2f}/{agree1[ref_keys[1]]:.2f}" if ref_keys[1] in agree else "n/a")}
        if alt_e is not None:
            row[f"E{alt_e:03d} active disc/val"] = f"{act(alt_e, disc)}/{act(alt_e, val)}"
        row[f"E{E:03d} active disc/val"] = f"{act(E, disc)}/{act(E, val)}"
        if md.expert_rows is not None and layer in set(md.expert_rows.layer):
            et = A.expert_table(md, layer)
            sel = A.select_expert(et, disc, len(disc) // 2)
            e_star = sel["e_star"]
            if e_star is not None:
                ev = A.evaluate_expert(md, layer, e_star, val, n_controls)
                row["selected e* (cands)"] = f"E{e_star:03d} ({sel['n_candidates']})"
                row["e* val rescue"] = fmt(ev["rescue_all"])
                row["e* spec"] = fmt(ev["spec_all"])
            else:
                row["selected e* (cands)"] = f"none (max act {sel['max_activity']})"
                row["e* val rescue"] = row["e* spec"] = ""
            evp = A.evaluate_expert(md, layer, E, val, n_controls)
            row[f"E{E:03d} val rescue / spec"] = f"{evp['rescue_all']['mean']:+.3f} / {evp['spec_all']['mean']:+.3f}"
            co = A.coalitions(md, layer, val)
            row["coalition top-k / union"] = f"{co['coalition_clean']['mean']:+.3f} / {co['coalition_union']['mean']:+.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, {"mds": mds, "curves": curves, "ids": ids, "disc": disc, "val": val, "L": L}


def fig_agreement(curves, model, layer, ref_keys, stem):
    fig, axes = plt.subplots(1, len(ref_keys), figsize=(6.2 * len(ref_keys), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, ref_keys):
        for (k, rr), cur in curves.items():
            if rr != r or k == r:
                continue
            ax.plot(cur, marker="o", ms=3, label=k)
        ax.axvline(layer, color="k", ls=":", lw=1)
        ax.set_xlabel("layer")
        ax.set_title(f"{model}: L-wise top-k set agreement with the '{r}' run (clean, final position)", fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("fraction of 256 prompts with identical top-k set")
    fig.tight_layout()
    fig.savefig(stem, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def diagnostics(model, variants, layer, paper_e, alt_e, ids_ctx):
    out = {}
    dg = {k: load_diag(v.run_dir) for k, v in variants.items() if os.path.exists(os.path.join(DIAG_ROOT, v.run_dir, "diag.npz"))}
    if not dg:
        return out
    n = len(ids_ctx["ids"])
    L = next(iter(dg.values()))["attn_final"].shape[0]
    keys = [k for k in ("bos", "nobos", "shift1", "eos", "bosbos", "nl", "dot", "dot2", "comma", "colon", "the", "of", "space", "unk", "rare", "sinkfull", "sinkkey", "default", "eot") if k in dg]
    # ---- sink mass and norms per layer (clean rows)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for k in keys:
        d = dg[k]
        att = d["attn_final"][:, :n].astype(np.float32)  # [L, n, nH, T]
        lens = d["lens"][:n]
        m0 = att[:, :, :, 0].mean((1, 2))
        m1 = att[:, :, :, 1].mean((1, 2))
        mself = np.array([att[:, i, :, lens[i] - 1].mean(-1) for i in range(n)]).mean(0)
        if "attn_sink" in d:  # transplanted slot: report its mass in place of position 0
            msink = d["attn_sink"][:, :n].astype(np.float32).mean((1, 2))
            axes[0].plot(msink, marker="s", ms=3, label=f"{k}: sink slot")
            out[f"{k}_sink_slot_mass"] = msink.round(4).tolist()
        axes[0].plot(m0, marker="o", ms=3, label=f"{k}: pos 0", alpha=0.9 if "attn_sink" not in d else 0.5)
        if k in ("bos", "nobos", "eot", "default"):
            axes[0].plot(m1, ls="--", label=f"{k}: pos 1")
        rn = d["resid_norms"][:, :n]
        axes[1].plot(rn[:, :, 0].mean(1), marker="o", ms=3, label=f"{k}: pos 0")
        if k in ("bos", "nobos", "eot", "default"):
            axes[1].plot(rn[:, :, 1].mean(1), ls="--", label=f"{k}: pos 1")
        out[f"{k}_sink_mass_pos0"] = m0.round(4).tolist()
        out[f"{k}_sink_mass_pos1"] = m1.round(4).tolist()
        out[f"{k}_self_mass"] = mself.round(4).tolist()
        out[f"{k}_norm_pos0"] = rn[:, :, 0].mean(1).round(2).tolist()
        out[f"{k}_norm_pos1"] = rn[:, :, 1].mean(1).round(2).tolist()
        out[f"{k}_norm_final"] = np.array([rn[:, i, lens[i] - 1] for i in range(n)]).mean(0).round(2).tolist()
    axes[0].set_title(f"{model}: final position's attention on pos 0 (squares: transplanted slot)", fontsize=10)
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("attention probability"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title("residual norm after the layer, position 0 (dashed: position 1)", fontsize=10)
    axes[1].set_xlabel("layer"); axes[1].set_yscale("log"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7, ncol=2)
    # ---- cosine divergence of final-position residuals vs the two reference protocols
    refs = [r for r in ("bos", "nobos", "default") if r in dg and "resid_final" in dg[r]]
    for r in refs:
        R = dg[r]["resid_final"][:, :n].float()
        for k in keys:
            if k == r or "resid_final" not in dg[k]:
                continue
            X = dg[k]["resid_final"][:, :n].float()
            cos = (R * X).sum(-1) / (R.norm(dim=-1) * X.norm(dim=-1)).clamp_min(1e-6)  # [L, n]
            cm = cos.mean(1).numpy()
            out[f"cos_{k}_vs_{r}_mean"] = cm.round(4).tolist()
            out[f"cos_{k}_vs_{r}_median"] = cos.median(1).values.numpy().round(4).tolist()
            if r in ("bos", "default"):
                axes[2].plot(cm, marker="o", ms=3, label=f"{k} vs {r}")
    axes[2].set_title("cosine of the final-position residual with the BOS run", fontsize=10)
    axes[2].set_xlabel("layer"); axes[2].set_ylabel("cosine"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"ext3_{model}_diagnostics.png"), dpi=150)
    plt.close(fig)
    # ---- per-head sink mass at layer `layer` and a couple of other layers, bos vs nobos
    # ---- router margins at the paper layer
    if "bos" in dg and "nobos" in dg and alt_e is not None:
        rl_b = dg["bos"]["router_logits_final"][layer, :n]
        rl_n = dg["nobos"]["router_logits_final"][layer, :n]
        mb, mn = rl_b[:, paper_e] - rl_b[:, alt_e], rl_n[:, paper_e] - rl_n[:, alt_e]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        bins = np.linspace(min(mb.min(), mn.min()), max(mb.max(), mn.max()), 40)
        axes[0].hist(mb, bins=bins, alpha=0.6, label=f"BOS (mean {mb.mean():+.2f})")
        axes[0].hist(mn, bins=bins, alpha=0.6, label=f"no BOS (mean {mn.mean():+.2f})")
        axes[0].axvline(0, color="k", lw=1)
        axes[0].set_title(f"L{layer} router logit margin E{paper_e:03d} - E{alt_e:03d} at the final position (clean)")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].scatter(mb, mn, s=10, alpha=0.6)
        lim = [bins[0], bins[-1]]
        axes[1].plot(lim, lim, "k:", lw=1); axes[1].axhline(0, color="gray", lw=0.5); axes[1].axvline(0, color="gray", lw=0.5)
        axes[1].set_xlabel("margin with BOS"); axes[1].set_ylabel("margin without BOS"); axes[1].set_title("per-prompt margin, BOS vs no BOS"); axes[1].grid(alpha=0.3)
        # distance to the routing boundary: 2nd - 3rd largest router logit
        def gap(rl):
            s = np.sort(rl, axis=-1)
            return s[:, -2] - s[:, -3]
        gb, gn = gap(rl_b), gap(rl_n)
        axes[2].hist(gb, bins=30, alpha=0.6, label=f"BOS (median {np.median(gb):.2f})")
        axes[2].hist(gn, bins=30, alpha=0.6, label=f"no BOS (median {np.median(gn):.2f})")
        axes[2].set_title(f"L{layer} top-2 boundary gap (2nd - 3rd router logit)"); axes[2].legend(); axes[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"ext3_{model}_margins_L{layer}.png"), dpi=150)
        plt.close(fig)
        mean_shift = (rl_n - rl_b).mean(0)
        out["margin_bos"] = summarize(mb, with_p=False)
        out["margin_nobos"] = summarize(mn, with_p=False)
        out["margin_shift_nobos_minus_bos"] = summarize(mn - mb)
        out["frac_margin_positive_bos"] = float((mb > 0).mean())
        out["frac_margin_positive_nobos"] = float((mn > 0).mean())
        out["boundary_gap_median_bos"] = float(np.median(gb))
        out["boundary_gap_median_nobos"] = float(np.median(gn))
        out["router_logit_mean_shift_nobos_minus_bos"] = {f"E{e:03d}": round(float(mean_shift[e]), 3) for e in range(rl_b.shape[1])}
        out["router_logit_mean_bos"] = {f"E{e:03d}": round(float(rl_b[:, e].mean()), 3) for e in range(rl_b.shape[1])}
        out["router_logit_mean_nobos"] = {f"E{e:03d}": round(float(rl_n[:, e].mean()), 3) for e in range(rl_b.shape[1])}
        # margin distribution for the substitution variants (does the margin move back towards the BOS value?)
        for k in keys:
            rl = dg[k]["router_logits_final"][layer, :n]
            mk = rl[:, paper_e] - rl[:, alt_e]
            out[f"margin_{k}_mean"] = float(mk.mean())
            out[f"margin_{k}_corr_with_bos"] = float(np.corrcoef(mk, mb)[0, 1])
            out[f"margin_{k}_corr_with_nobos"] = float(np.corrcoef(mk, mn)[0, 1])
            out[f"margin_{k}_mad_to_bos"] = float(np.abs(mk - mb).mean())
            out[f"margin_{k}_mad_to_nobos"] = float(np.abs(mk - mn).mean())
    # ---- OOD check: per-prompt mean log-likelihood of the content tokens (aligned), relation to routing change
    if "bos" in dg and "nobos" in dg:
        lp_b, lp_n = dg["bos"]["token_logprobs"][:n], dg["nobos"]["token_logprobs"][:n]
        lens_b, lens_n = dg["bos"]["lens"][:n], dg["nobos"]["lens"][:n]
        pb = int(dg["bos"]["prefix_len"]); pn = int(dg["nobos"]["prefix_len"])
        ll_b = np.array([np.nanmean(lp_b[i, pb : lens_b[i] - 1]) for i in range(n)])  # predictions of content tokens 2..end
        ll_n = np.array([np.nanmean(lp_n[i, pn : lens_n[i] - 1]) for i in range(n)])
        first_b = lp_b[:, pb - 1]  # log p(first content token | <s>)
        mds = ids_ctx["mds"]
        cs_b, cs_n = clean_sets(mds["bos"], layer), clean_sets(mds["nobos"], layer)
        ids = ids_ctx["ids"]
        changed = np.array([cs_b[c] != cs_n[c] for c in ids])
        shift = ll_n - ll_b
        ct = mds["nobos"].cases.set_index("case_id").loc[ids]
        subj0 = np.array([json.loads(s)[0] == 0 for s in ct.subject_pos])
        from scipy import stats as st
        out["ood"] = {"mean_ll_bos": float(ll_b.mean()), "mean_ll_nobos": float(ll_n.mean()), "mean_shift": float(shift.mean()),
                      "frac_prompts_worse_without_bos": float((shift < 0).mean()),
                      "mean_logp_first_content_token_given_bos": float(np.nanmean(first_b)),
                      "n_changed_routing": int(changed.sum()), "shift_changed": float(shift[changed].mean()), "shift_unchanged": float(shift[~changed].mean()),
                      "mannwhitney_p": float(st.mannwhitneyu(shift[changed], shift[~changed]).pvalue),
                      "pointbiserial_r": float(st.pointbiserialr(changed.astype(int), shift)[0]),
                      "auc_shift_predicts_change": float(st.mannwhitneyu(-shift[changed], -shift[~changed]).statistic / (changed.sum() * (~changed).sum())),
                      "frac_subject_at_pos0_nobos": float(subj0.mean()),
                      "routing_change_rate_subject_pos0": float(changed[subj0].mean()), "routing_change_rate_subject_later": float(changed[~subj0].mean()) if (~subj0).any() else None,
                      "fisher_p_subject_pos0_vs_change": float(st.fisher_exact([[int((changed & subj0).sum()), int((~changed & subj0).sum())],
                                                                                 [int((changed & ~subj0).sum()), int((~changed & ~subj0).sum())]])[1]),
                      "logprob_true_bos": float(dg["bos"]["logprob_true"][:n].mean()), "logprob_true_nobos": float(dg["nobos"]["logprob_true"][:n].mean())}
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.scatter(ll_b, ll_n, c=np.where(changed, "C3", "C0"), s=12, alpha=0.7)
        lim = [min(ll_b.min(), ll_n.min()), max(ll_b.max(), ll_n.max())]
        ax.plot(lim, lim, "k:", lw=1)
        ax.set_xlabel("mean log p(content token) with BOS"); ax.set_ylabel("mean log p(content token) without BOS")
        ax.set_title(f"{model}: prompt likelihood, red = L{layer} routing changed ({changed.sum()}/{n})")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"ext3_{model}_ood.png"), dpi=150); plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------------------------
DELIM = {",", ".", ";", ":", "(", ")", "\"", "'", "-", "–", "—", "!", "?", "/", "&"}


def token_pieces(model):
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    return AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model]["repo"]))


def piece_class(p: str) -> str:
    q = p.replace("▁", " ")
    core = q.strip()
    if core == "":
        return "space"
    if p.startswith("<0x"):
        return "byte"
    if core in DELIM or all(not ch.isalnum() for ch in core):
        return "delim"
    if core.lower() in ("of", "is", "in", "the", "a", "an", "and", "to", "by", "at", "on", "for", "was", "as", "from", "with", "that", "s"):
        return "function"
    return "content"


def sink_and_strata(model, variants, layer, paper_e, alt_e, ctx):
    """Where does the final position park its attention and where are the massive norms, per protocol; strata by
    delimiter presence and by subject-at-position-0; over-mixing (noise drop) per protocol; early-layer routing of
    position 0; contribution norms of E006/E002 at the paper layer."""
    out, parts = {}, []
    dg = {k: load_diag(v.run_dir) for k, v in variants.items() if k in ("bos", "nobos") and os.path.exists(os.path.join(DIAG_ROOT, v.run_dir, "diag.npz"))}
    if len(dg) < 2:
        return out, ""
    mds, ids = ctx["mds"], ctx["ids"]
    n = len(ids)
    tok = token_pieces(model)
    ct_n = mds["nobos"].cases.set_index("case_id").loc[ids]
    ct_b = mds["bos"].cases.set_index("case_id").loc[ids]
    ids_n = [json.loads(x) for x in ct_n.ids]
    pieces = [tok.convert_ids_to_tokens(x) for x in ids_n]
    classes = [[piece_class(p) for p in pp] for pp in pieces]
    has_delim = np.array([any(c == "delim" for c in cl[:-1]) for cl in classes])
    subj0 = np.array([json.loads(x)[0] == 0 for x in ct_n.subject_pos])
    L = dg["bos"]["attn_final"].shape[0]
    # ---- sink location in the no-BOS run: argmax over positions of the head-mean final-position attention, per layer
    loc_rows = []
    for k in ("bos", "nobos"):
        att = dg[k]["attn_final"][:, :n].astype(np.float32).mean(2)  # [L, n, T] head-mean
        rn = dg[k]["resid_norms"][:, :n]
        lens = dg[k]["lens"][:n]
        pl = int(dg[k]["prefix_len"])
        for l in (1, 2, 3, 5, 10, layer, L - 1):
            arg_a = att[l].argmax(1)
            arg_n = rn[l].argmax(1)
            def cls_at(i, pos):
                pos = int(pos)
                if pos < pl:
                    return "prefix(<s>)" if k == "bos" else "prefix"
                if pos == lens[i] - 1:
                    return "final"
                return classes[i][pos - pl]
            ca = [cls_at(i, arg_a[i]) for i in range(n)]
            cn = [cls_at(i, arg_n[i]) for i in range(n)]
            mass_at_argmax = att[l][np.arange(n), arg_a].mean()
            row = {"run": k, "layer": l, "mean max attention mass (final pos.)": f"{mass_at_argmax:.3f}",
                   "attention argmax at pos 0": f"{(arg_a == 0).mean():.2f}", "attention argmax class": pd.Series(ca).value_counts().to_dict(),
                   "norm argmax at pos 0": f"{(arg_n == 0).mean():.2f}", "norm argmax class": pd.Series(cn).value_counts().to_dict(),
                   "norm at argmax / median norm": f"{np.mean([rn[l, i, arg_n[i]] / np.median(rn[l, i, : lens[i]]) for i in range(n)]):.1f}"}
            loc_rows.append(row)
        out[f"{k}_attn_pos0_by_layer_delim"] = att[:, has_delim, 0].mean(1).round(4).tolist()
        out[f"{k}_attn_pos0_by_layer_nodelim"] = att[:, ~has_delim, 0].mean(1).round(4).tolist()
        # mass on the first delimiter (no-BOS content coordinates) when present
        fd = []
        for i in range(n):
            cl = classes[i][:-1]
            if "delim" in cl:
                fd.append(att[:, i, cl.index("delim") + pl])
        if fd:
            out[f"{k}_attn_first_delim_by_layer"] = np.mean(fd, 0).round(4).tolist()
    parts.append("**Where the final position parks its attention and where the largest residual norm sits (clean prompts; class of the argmax position: "
                 "prefix = the prepended token, delim = punctuation, function = of/is/in/the..., content = other prompt tokens, final = the final position itself)**\n")
    parts.append(md_table(list(loc_rows[0].keys()), [list(r.values()) for r in loc_rows], os.path.join(TABLES, "ext3_sink_location")))
    # ---- strata: delimiter present / subject at position 0
    cs_b, cs_n = clean_sets(mds["bos"], layer), clean_sets(mds["nobos"], layer)
    changed = np.array([cs_b[c] != cs_n[c] for c in ids])
    drop_b = (ct_b.delta_clean - ct_b.delta_noised).values
    drop_n = (ct_n.delta_clean - ct_n.delta_noised).values
    act = lambda cs, e, mask: int(sum(e in cs[c] for c, m in zip(ids, mask) if m))
    # does the final position itself carry the maximal residual norm without BOS (checked at layer 5 and at L-1 before the paper layer)?
    rn_n, rn_b = dg["nobos"]["resid_norms"][:, :n], dg["bos"]["resid_norms"][:, :n]
    lens_n, lens_b = dg["nobos"]["lens"][:n], dg["bos"]["lens"][:n]
    fin_is_max = np.array([rn_n[5, i].argmax() == lens_n[i] - 1 for i in range(n)])
    fin_ratio = np.array([rn_n[layer - 1, i, lens_n[i] - 1] / rn_b[layer - 1, i, lens_b[i] - 1] for i in range(n)])  # router-input norm, no BOS / BOS
    out["final_norm_ratio_nobos_over_bos_Lm1"] = {"mean": float(fin_ratio.mean()), "median": float(np.median(fin_ratio)),
                                                   "frac_gt_1.5": float((fin_ratio > 1.5).mean()), "frac_gt_3": float((fin_ratio > 3).mean())}
    rows = []
    for name, mask in (("all", np.ones(n, bool)), ("delimiter before final position", has_delim), ("no delimiter", ~has_delim),
                       ("subject starts at position 0 (no BOS)", subj0), ("subject starts later", ~subj0),
                       ("final position carries the max norm at L5 (no BOS)", fin_is_max), ("max norm elsewhere", ~fin_is_max),
                       (f"final-position norm ratio no-BOS/BOS > 1.5 (L{layer - 1})", fin_ratio > 1.5), ("ratio <= 1.5", fin_ratio <= 1.5)):
        m = mask.sum()
        if m == 0:
            continue
        rows.append([name, int(m), f"{1 - changed[mask].mean():.2f}", f"{act(cs_b, paper_e, mask)}/{act(cs_n, paper_e, mask)}", f"{act(cs_b, alt_e, mask)}/{act(cs_n, alt_e, mask)}",
                     f"{drop_b[mask].mean():+.2f} / {drop_n[mask].mean():+.2f}", f"{np.std(drop_b[mask]):.2f} / {np.std(drop_n[mask]):.2f}",
                     f"{ct_b.delta_clean.values[mask].mean():+.2f} / {ct_n.delta_clean.values[mask].mean():+.2f}",
                     f"{int(ct_b.strict.values[mask].sum())} / {int(ct_n.strict.values[mask].sum())}"])
    has_func = np.array([any(c == "function" for c in cl[:-1]) for cl in classes])
    has_trigger = has_delim | has_func
    xt = pd.crosstab(pd.Series(np.where(fin_is_max, "final position is the sink", "sink elsewhere"), name="no-BOS run"),
                     pd.Series(np.where(has_delim, "delimiter before final", "no delimiter"), name=""), margins=True)
    xt2 = pd.crosstab(pd.Series(np.where(fin_is_max, "final position is the sink", "sink elsewhere"), name="no-BOS run"),
                      pd.Series(np.where(has_trigger, "delimiter or function word before final", "neither"), name=""), margins=True)
    out["crosstab_sink_delim"] = {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in xt.to_dict().items()}
    out["crosstab_sink_trigger"] = {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in xt2.to_dict().items()}
    pm_cls = pd.Series([classes[i][int(p) - int(dg["nobos"]["prefix_len"])] if int(p) < lens_n[i] - 1 else "final" for i, p in
                       enumerate([rn_n[5, i].argmax() for i in range(n)])]).value_counts().to_dict()
    out["max_norm_position_class_L5_nobos"] = pm_cls
    parts.append(f"\n**Which prompts hand the sink to their final token without BOS (L5 max-norm position vs the tokens before the final position)**\n")
    parts.append(md_table([""] + [str(c) for c in xt.columns], [[str(i)] + [int(v) for v in r] for i, r in zip(xt.index, xt.values)], os.path.join(TABLES, "ext3_sink_x_delim")))
    parts.append(md_table([""] + [str(c) for c in xt2.columns], [[str(i)] + [int(v) for v in r] for i, r in zip(xt2.index, xt2.values)], os.path.join(TABLES, "ext3_sink_x_trigger")))
    parts.append(f"Class of the max-norm position at L5 without BOS: {pm_cls} (the first token is a content token in "
                 f"{sum(cl[0] == 'content' for cl in classes)}/256 prompts, a function word in {sum(cl[0] == 'function' for cl in classes)}).\n")
    parts.append(f"\n**Strata (L{layer}, clean final-position routing; counts are BOS / no BOS)**\n")
    parts.append(md_table(["stratum", "n", f"routing agreement @L{layer}", f"E{paper_e:03d} active", f"E{alt_e:03d} active", "mean noise drop", "sd of drop", "mean Δ_clean", "strict pass"],
                          rows, os.path.join(TABLES, "ext3_strata")))
    from scipy import stats as st
    out["strata"] = {"n_delim": int(has_delim.sum()), "change_rate_delim": float(changed[has_delim].mean()), "change_rate_nodelim": float(changed[~has_delim].mean()),
                     "fisher_p_delim": float(st.fisher_exact([[int((changed & has_delim).sum()), int((~changed & has_delim).sum())],
                                                              [int((changed & ~has_delim).sum()), int((~changed & ~has_delim).sum())]])[1]),
                     "drop_mean_bos": float(drop_b.mean()), "drop_mean_nobos": float(drop_n.mean()), "drop_sd_bos": float(drop_b.std()), "drop_sd_nobos": float(drop_n.std()),
                     "drop_change_vs_routing_change_r": float(st.pointbiserialr(changed.astype(int), drop_n - drop_b)[0]),
                     "drop_change_changed": float((drop_n - drop_b)[changed].mean()), "drop_change_unchanged": float((drop_n - drop_b)[~changed].mean()),
                     "n_final_is_max_norm_L5": int(fin_is_max.sum()), "change_rate_final_is_max": float(changed[fin_is_max].mean()) if fin_is_max.any() else None,
                     "change_rate_final_not_max": float(changed[~fin_is_max].mean()),
                     "fisher_p_final_is_max": float(st.fisher_exact([[int((changed & fin_is_max).sum()), int((~changed & fin_is_max).sum())],
                                                                     [int((changed & ~fin_is_max).sum()), int((~changed & ~fin_is_max).sum())]])[1]),
                     "pointbiserial_r_log_final_norm_ratio_vs_change": float(st.pointbiserialr(changed.astype(int), np.log(fin_ratio))[0]),
                     "change_rate_ratio_gt_1.5": float(changed[fin_ratio > 1.5].mean()) if (fin_ratio > 1.5).any() else None,
                     "change_rate_ratio_le_1.5": float(changed[fin_ratio <= 1.5].mean())}
    # ---- early-layer routing of position 0 (<s> vs first content token) and of the first content token in both runs
    er = []
    for l in (0, 1, 2, 3):
        kb, kn = f"route_all_L{l}_topi", f"route_all_L{l}_logits"
        if kb not in dg["bos"] or kb not in dg["nobos"]:
            continue
        tb, tn = dg["bos"][kb][:n], dg["nobos"][kb][:n]  # [n, T, k]
        lb, ln = dg["bos"][kn][:n].astype(np.float32), dg["nobos"][kn][:n].astype(np.float32)
        plb = int(dg["bos"]["prefix_len"])
        top_bos = pd.Series(tb[:, 0, 0]).value_counts().to_dict()  # top-1 expert of <s>
        top_first_b = pd.Series(tb[:, plb, 0]).value_counts().to_dict()  # first content token in the BOS run
        top_first_n = pd.Series(tn[:, 0, 0]).value_counts().to_dict()  # first content token in the no-BOS run
        pmax = lambda lg: float(np.mean(np.exp(lg).max(-1) / np.exp(lg).sum(-1)))
        er.append([l, {f"E{int(e)}": int(c) for e, c in top_bos.items()}, f"{pmax(lb[:, 0]):.3f}", {f"E{int(e)}": int(c) for e, c in top_first_b.items()},
                   {f"E{int(e)}": int(c) for e, c in top_first_n.items()}, f"{pmax(ln[:, 0]):.3f}",
                   f"{np.mean([sorted(tb[i, plb].tolist()) == sorted(tn[i, 0].tolist()) for i in range(n)]):.2f}"])
    if er:
        parts.append("\n**Early-layer routing of position 0 (top-1 expert counts over 256 clean prompts; max router prob = mean softmax probability of the top expert)**\n")
        parts.append(md_table(["layer", "<s> top-1 (BOS run)", "<s> max router prob", "1st content token top-1 (BOS run)", "1st content token top-1 (no-BOS run)",
                               "1st content token max router prob (no BOS)", "1st content token: same top-2 set in both runs"], er, os.path.join(TABLES, "ext3_early_routing")))
    # ---- the sink state's own routing: <s> at every recorded layer (BOS run) vs the final position of the no-BOS prompts whose
    #      final position carries the massive norm (their router logits are nearly prompt-independent)
    from collections import Counter
    if f"route_all_L{layer}_topi" in dg["bos"] and f"route_all_L{layer}_topi" in dg["nobos"]:
        rows = []
        for l in sorted(int(k.split("_L")[1].split("_")[0]) for k in dg["bos"] if k.startswith("route_all_L") and k.endswith("_topi")):
            tb = dg["bos"][f"route_all_L{l}_topi"][:n]
            lgb = dg["bos"][f"route_all_L{l}_logits"][:n].astype(np.float32)
            sets = Counter(tuple(sorted(tb[i, 0].tolist())) for i in range(n))
            rows.append([f"L{l}", "<s> (BOS run, position 0)", ", ".join(f"{{{a},{b}}}: {c}" for (a, b), c in sets.most_common(3)),
                         " ".join(f"{x:+.2f}" for x in lgb[:, 0].mean(0)), f"{lgb[:, 0].std(0).max():.3f}"])
        tn = dg["nobos"][f"route_all_L{layer}_topi"][:n]
        rl_n = dg["nobos"]["router_logits_final"][layer, :n]
        for name, mask in ((f"no-BOS final position, final carries the max norm (n={int(fin_is_max.sum())})", fin_is_max),
                           (f"no-BOS final position, max norm elsewhere (n={int((~fin_is_max).sum())})", ~fin_is_max),
                           (f"BOS final position (n={n})", np.ones(n, bool))):
            src = tn if name.startswith("no-BOS") else dg["bos"][f"route_all_L{layer}_topi"][:n]
            ln_ = lens_n if name.startswith("no-BOS") else lens_b
            rl = rl_n if name.startswith("no-BOS") else dg["bos"]["router_logits_final"][layer, :n]
            sets = Counter(tuple(sorted(src[i, ln_[i] - 1].tolist())) for i in range(n) if mask[i])
            rows.append([f"L{layer}", name, ", ".join(f"{{{a},{b}}}: {c}" for (a, b), c in sets.most_common(4)),
                         " ".join(f"{x:+.2f}" for x in rl[mask].mean(0)), f"{rl[mask].std(0).max():.3f}"])
        parts.append(f"\n**The sink state's own routing (top-2 set counts; mean router logits E000..E{rl_n.shape[1] - 1:03d}; largest across-prompt sd of any logit)**\n")
        parts.append(md_table(["layer", "token / group", "top-2 sets", "mean router logits", "max sd"], rows, os.path.join(TABLES, "ext3_sink_expert")))
        out["sink_expert"] = {"s_top2_L": Counter(tuple(sorted(dg["bos"][f"route_all_L{layer}_topi"][i, 0].tolist())) for i in range(n)).most_common(2),
                              "final_is_sink_top2": Counter(tuple(sorted(tn[i, lens_n[i] - 1].tolist())) for i in range(n) if fin_is_max[i]).most_common(4),
                              "final_is_sink_contains_E": int(sum(paper_e in tn[i, lens_n[i] - 1].tolist() for i in range(n) if fin_is_max[i])),
                              "final_is_sink_logit_sd_max": float(rl_n[fin_is_max].std(0).max()) if fin_is_max.any() else None,
                              "E_active_rate_final_is_sink_nobos": float(np.mean([paper_e in cs_n[c] for c, m in zip(ids, fin_is_max) if m])) if fin_is_max.any() else None,
                              "E_active_rate_others_nobos": float(np.mean([paper_e in cs_n[c] for c, m in zip(ids, fin_is_max) if not m])),
                              "E_active_rate_others_bos": float(np.mean([paper_e in cs_b[c] for c, m in zip(ids, fin_is_max) if not m])),
                              "E_active_rate_final_is_sink_bos": float(np.mean([paper_e in cs_b[c] for c, m in zip(ids, fin_is_max) if m])) if fin_is_max.any() else None}
        # which final tokens become the sink
        fin_tok = Counter(pieces[i][-1] for i in range(n) if fin_is_max[i])
        oth_tok = Counter(pieces[i][-1] for i in range(n) if not fin_is_max[i])
        out["final_token_when_sink"] = fin_tok.most_common(8)
        out["final_token_otherwise"] = oth_tok.most_common(8)
        parts.append(f"\nFinal tokens of the prompts whose final position becomes the sink without BOS: {dict(fin_tok.most_common(8))}; "
                     f"final tokens of the other prompts: {dict(oth_tok.most_common(8))}. In the BOS run the final position carries the maximal norm in "
                     f"{int(sum(rn_b[5, i].argmax() == lens_b[i] - 1 for i in range(n)))} prompts.\n")
    # ---- contribution norms of the two experts at the paper layer (final position, clean), both protocols
    cn_rows = []
    for k in ("bos", "nobos"):
        r = mds[k].routing
        r = r[(r.run == "clean") & (r.layer == layer)]
        for e in (paper_e, alt_e):
            sub = r[r.expert == e]
            cn_rows.append([k, f"E{e:03d}", len(sub), f"{sub.cnorm.mean():.2f}", f"{sub.weight.mean():.3f}", f"{(sub.slot == 0).mean():.2f}"])
    parts.append(f"\n**Contribution norms ||w_e E_e(x)|| and routing weights of E{paper_e:03d} / E{alt_e:03d} at L{layer} when active (clean final position)**\n")
    parts.append(md_table(["run", "expert", "n active", "mean ||c_e||", "mean routing weight", "fraction top-1"], cn_rows, os.path.join(TABLES, f"ext3_cnorm_L{layer}")))
    return out, "\n".join(parts)


# ---------------------------------------------------------------------------------------------------------------
def transition_table(mds, ids, layer, stem):
    from collections import Counter
    cs_b, cs_n = clean_sets(mds["bos"], layer), clean_sets(mds["nobos"], layer)
    tr = Counter((tuple(sorted(cs_b[c])), tuple(sorted(cs_n[c]))) for c in ids)
    rows = [[f"{{{a[0]},{a[1]}}}", f"{{{b[0]},{b[1]}}}", n, "same" if a == b else "changed"] for (a, b), n in tr.most_common(14)]
    return md_table(["top-2 with BOS", "top-2 without BOS", "prompts", ""], rows, stem)


def corpus_section(layer, paper_e, alt_e):
    parts, nums = [], {}
    rows_u = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ci, corpus in enumerate(("wiki", "code")):
        p = os.path.join(RESULTS, f"mixtral_nobos_corpus_{corpus}", "corpus_stats.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        E = s["experts"]
        ub, un = np.array(s["usage_topk_bos"])[layer], np.array(s["usage_topk_nobos"])[layer]
        for e in range(E):
            rows_u.append([corpus, f"E{e:03d}", f"{ub[e]:.3f}", f"{un[e]:.3f}", f"{un[e] - ub[e]:+.3f}",
                           s[f"class_entropy_bits_paper_layer_nobos"][f"E{e:03d}"], s[f"tokenid_entropy_bits_paper_layer_nobos"][f"E{e:03d}"],
                           s[f"position_entropy_normalised_paper_layer_nobos"][f"E{e:03d}"],
                           f"{s['first_content_token_routing_nobos'][f'E{e:03d}']:.2f}", f"{s['first_token_routing_bos_row_pos0'][f'E{e:03d}']:.2f}",
                           " / ".join(f"{x:.2f}" for x in s[f"class_distribution_paper_layer_nobos"][f"E{e:03d}"])])
        nums[corpus] = {"agreement_set_L": s["agreement_set_per_layer"][layer], "agreement_by_position_L_first8": s["agreement_set_by_position_paper_layer"][:8],
                        "agreement_by_position_L_last8": s["agreement_set_by_position_paper_layer"][-8:],
                        "usage_rank_nobos": [f"E{e:03d}" for e in np.argsort(-un)], "usage_rank_bos": [f"E{e:03d}" for e in np.argsort(-ub)],
                        "mean_logprob_bos": s["mean_logprob_bos"], "mean_logprob_nobos": s["mean_logprob_nobos"],
                        "logprob_by_pos_first4_bos": s["mean_logprob_by_content_position_bos"][:4], "logprob_by_pos_first4_nobos": s["mean_logprob_by_content_position_nobos"][:4],
                        "resid_norm_pos0_L2_bos": s["resid_norm_pos0_per_layer_bos"][2], "resid_norm_pos0_L2_nobos": s["resid_norm_pos0_per_layer_nobos"][2],
                        "resid_norm_median_content_L2_nobos": s["resid_norm_median_content_per_layer_nobos"][2],
                        "classes": s["classes"], "n_tokens": s["n_content_tokens"]}
        ax = axes[ci]
        ax.plot(s["agreement_set_by_position_paper_layer"], label=f"L{layer} set agreement", lw=1)
        ax.plot(s["agreement_set_by_position_all_layers_mean"], label="mean over layers", lw=1)
        ax.plot(s["usage_by_position_paper_layer_nobos"][f"E{paper_e:03d}"], label=f"E{paper_e:03d} usage, no BOS", lw=1)
        ax.plot(s["usage_by_position_paper_layer_bos"][f"E{paper_e:03d}"], label=f"E{paper_e:03d} usage, BOS", lw=1, ls="--")
        ax.plot(s["usage_by_position_paper_layer_nobos"][f"E{alt_e:03d}"], label=f"E{alt_e:03d} usage, no BOS", lw=1)
        ax.plot(s["usage_by_position_paper_layer_bos"][f"E{alt_e:03d}"], label=f"E{alt_e:03d} usage, BOS", lw=1, ls="--")
        ax.set_xlabel("content-token position"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"{corpus}: per-position routing agreement BOS vs no BOS and E{paper_e:03d}/E{alt_e:03d} usage at L{layer}")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "ext3_mixtral_corpus_positions.png"), dpi=150); plt.close(fig)
    # ---- Direction-1 cross-check: E001 at layers 17-22 (strongest single expert of L17/18/21/22 in ext1) next to E002/E006 at L19,
    #      from the raw per-token routing; and E006's L19 usage against the presence of a sink
    rows_e, sink_rows = [], []
    tok = None
    for corpus in ("wiki", "code"):
        raw_p = os.path.join(DIAG_ROOT, f"mixtral_corpus_{corpus}", "route_all.npz")
        if not os.path.exists(raw_p):
            continue
        raw = np.load(raw_p)
        topi, content, rn = raw["topi"], raw["content"], raw["resid_norms"]
        nW, W = content.shape
        tb, tn = topi[:, :nW, 1 : W + 1], topi[:, nW:, 0:W]  # content-aligned [L, n, W, k]
        if tok is None:
            tok = token_pieces("mixtral")
        uniq = np.unique(content)
        cls_of = dict(zip(uniq.tolist(), [piece_class(p) for p in tok.convert_ids_to_tokens(uniq.tolist())]))
        cls_names = sorted(set(cls_of.values()))
        cls_arr = np.vectorize(lambda x: cls_names.index(cls_of[int(x)]))(content)
        def ent(counts):
            pr = counts[counts > 0] / counts.sum()
            return float(-(pr * np.log2(pr)).sum())
        for (l, e) in [(17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (19, 2), (19, 6)]:
            mb, mn = (tb[l] == e).any(-1), (tn[l] == e).any(-1)  # [n, W]
            cc_n = np.bincount(cls_arr[mn], minlength=len(cls_names)).astype(float)
            cc_b = np.bincount(cls_arr[mb], minlength=len(cls_names)).astype(float)
            pos_n = mn.sum(0).astype(float)
            rows_e.append([corpus, f"L{l}", f"E{e:03d}", f"{mb.mean():.3f}", f"{mn.mean():.3f}", f"{mn.mean() - mb.mean():+.3f}",
                           f"{ent(cc_b):.3f} / {ent(cc_n):.3f}", f"{ent(pos_n) / np.log2(W):.3f}",
                           f"{(tn[l] == e).any(-1)[:, 0].mean():.2f}", f"{(topi[l, :nW, 0] == e).any(-1).mean():.2f}",
                           " / ".join(f"{x:.2f}" for x in cc_n / max(cc_n.sum(), 1))])
        # E006 at L19 vs sink presence: (a) protocol-level; (b) per window: no-BOS windows whose position 0 carries the max norm at L5
        #  (a position-0 sink formed) vs windows where it did not; (c) rank correlation of the window's position-0 norm ratio with E006 usage
        from scipy import stats as st
        e6b, e6n = (tb[19] == 6).any(-1), (tn[19] == 6).any(-1)  # [n, W]
        pos0_norm = rn[5, nW:, 0]
        med_norm = np.median(rn[5, nW:, 1:W], axis=1)
        ratio = pos0_norm / med_norm
        sink0 = np.array([rn[5, nW + i, :W].argmax() == 0 for i in range(nW)])
        rho = st.spearmanr(np.log(ratio), e6n.mean(1))
        sink_rows.append([corpus, f"{e6b.mean():.3f}", f"{e6n.mean():.3f}", f"{e6n[:, 1:].mean() - e6b[:, 1:].mean():+.4f}",
                          f"{int(sink0.sum())}/{nW}", f"{e6n[sink0].mean():.3f}" if sink0.any() else "n/a", f"{e6n[~sink0].mean():.3f}" if (~sink0).any() else "n/a",
                          f"{rho.correlation:+.3f} (p={rho.pvalue:.3f})", f"{np.median(ratio):.1f}",
                          f"{(topi[19, :nW, 0] == 6).any(-1).mean():.2f}", f"{e6n[:, 0].mean():.2f}", f"{e6b[:, 0].mean():.2f}"])
        nums.setdefault(corpus, {})["E006_L19_usage_bos"] = float(e6b.mean())
        nums[corpus]["E006_L19_usage_nobos"] = float(e6n.mean())
        nums[corpus]["E006_L19_usage_excl_pos0_diff"] = float(e6n[:, 1:].mean() - e6b[:, 1:].mean())
        nums[corpus]["E006_L19_on_<s>"] = float((topi[19, :nW, 0] == 6).any(-1).mean())
        nums[corpus]["E006_L19_on_first_content_nobos"] = float(e6n[:, 0].mean())
        nums[corpus]["E006_L19_on_first_content_bos"] = float(e6b[:, 0].mean())
        nums[corpus]["n_windows_pos0_sink_nobos"] = int(sink0.sum())
        nums[corpus]["spearman_log_pos0_norm_ratio_vs_E006_usage"] = [float(rho.correlation), float(rho.pvalue)]
        nums[corpus]["E001_usage_by_layer_bos_nobos"] = {f"L{l}": [float((tb[l] == 1).any(-1).mean()), float((tn[l] == 1).any(-1).mean())] for l in range(17, 23)}
    if rows_e:
        parts.append("\n**Direction-1 cross-check: E001 at layers 17-22 next to E002 / E006 at L19 (content tokens; usage = top-2 membership; class entropy in bits, BOS / no BOS; "
                     "position entropy normalised, no BOS; P(routed | first content token) and P(routed | `<s>`) at that layer; class distribution without BOS in the order "
                     + ", ".join(cls_names) + ")**\n")
        parts.append(md_table(["corpus", "layer", "expert", "usage BOS", "usage no BOS", "Δ", "class entropy", "position entropy", "P(1st content tok)", "P(<s>)", "class distribution (no BOS)"],
                              rows_e, os.path.join(TABLES, "ext3_corpus_E001_L17-22")))
        parts.append("\n**E006 at L19 against the presence of a sink (corpus windows)**\n")
        parts.append(md_table(["corpus", "E006 usage BOS", "E006 usage no BOS", "Δ excluding the first content token", "no-BOS windows with a position-0 sink (max norm at L5)",
                               "E006 usage in those", "E006 usage in the others", "Spearman(log pos-0 norm ratio, E006 usage per window)", "median pos-0 norm / median content norm (no BOS, L5)",
                               "P(E006 | <s>)", "P(E006 | 1st content token, no BOS)", "P(E006 | 1st content token, BOS)"], sink_rows, os.path.join(TABLES, "ext3_corpus_E006_sink")))
    if rows_u:
        parts.append(md_table(["corpus", "expert", f"L{layer} usage BOS", f"L{layer} usage no BOS", "Δ usage", "token-class entropy (bits)", "token-id entropy (bits)",
                               "position entropy (norm.)", "P(routed | 1st content token, no BOS)", "P(routed | <s>)", "class distribution (no BOS)"],
                              rows_u, os.path.join(TABLES, f"ext3_corpus_L{layer}")))
        cls = nums[next(iter(nums))]["classes"]
        parts.append(f"Token classes in the class-distribution column, in order: {', '.join(cls)}. Usage = fraction of content tokens whose top-2 set contains the expert "
                     f"(sums to 2 over experts). Position entropy is the entropy of an expert's usage over the {'/'.join(str(nums[c]['n_tokens']) for c in nums)} content positions, "
                     f"normalised by log2(window); 1.0 = perfectly position-agnostic.\n")
    return "\n".join(parts), nums


# ---------------------------------------------------------------------------------------------------------------
def main():
    sec = ["## Direction 3: why does the BOS token move Mixtral's L19 router?\n",
           "Agent `ext3-bos-mechanism`. Model Mixtral-8x7B-v0.1 (bf16), the paper's 256 case IDs (128 discovery / 128 validation), "
           "sigma = 3.0 x embedding std, the same noise draws in every variant. Every variant is a full layer sweep plus an expert pass at L19 "
           "in which every expert is patched (so recurrence-first selection, validation rescue and specificity with the active-random control are "
           "computed exactly as in the main runs; no equal-norm pairs). Run directories `results/mixtral_<bos|nobos>_<experiment>/` with `run_meta.json`; "
           "raw diagnostics on the NVMe scratch `/opt/dlami/nvme/moe_ext3/`.\n"]
    # ----- Mixtral substitution table + agreement curves
    df, ctx = substitution_table("mixtral", MIXTRAL_VARIANTS, 19, 6, MODELS["mixtral"]["n_controls"])
    if df is not None:
        sec.append("### Experiment 2: what sits at position 0 (substitution controls)\n")
        sec.append(md_table(list(df.columns), df.values.tolist(), os.path.join(TABLES, "ext3_substitution")))
        sec.append("`agree@L19 w/ bos` = fraction of the 256 clean prompts whose final-position top-2 expert set (top-1 expert) at L19 is identical to the "
                   "BOS run's; likewise for the no-BOS run. `RoPE pos of 1st content token` = absolute position of the first prompt token. "
                   "Paper: L19E006 active 91/83, rescue +0.099, spec -0.175.\n")
        fig_agreement(ctx["curves"], "mixtral", 19, ("bos", "nobos"), os.path.join(FIGS, "ext3_mixtral_agreement.png"))
        sec.append("![per-layer routing agreement](../figures/ext3_mixtral_agreement.png)\n")
        NUM["substitution"] = df.to_dict(orient="records")
        NUM["agreement_curves"] = {f"{k}_vs_{r}": c.round(3).tolist() for (k, r), c in ctx["curves"].items()}
        if "bos" in ctx["mds"] and "nobos" in ctx["mds"]:
            sec.append("**L19 top-2 set transitions BOS -> no BOS (clean prompts, most frequent):**\n")
            sec.append(transition_table(ctx["mds"], ctx["ids"], 19, os.path.join(TABLES, "ext3_transitions_L19")))
        # ----- sink location, strata, over-mixing, early routing
        try:
            sst, sst_md = sink_and_strata("mixtral", MIXTRAL_VARIANTS, 19, 6, 2, ctx)
            if sst_md:
                sec.append("### Experiment 1b: where the sink lands, and strata by delimiter / subject position\n")
                sec.append(sst_md)
                NUM["sink_strata"] = sst
        except Exception as ex:  # keep the rest of the report if a diagnostic is missing
            sec.append(f"(sink/strata analysis unavailable: {ex!r})\n")
        # ----- diagnostics
        dgn = diagnostics("mixtral", MIXTRAL_VARIANTS, 19, 6, 2, ctx)
        NUM["diagnostics"] = dgn
        if dgn:
            sec.append("### Experiment 1: diagnostics (attention sink, norms, residual divergence, router margins)\n")
            sec.append("![diagnostics](../figures/ext3_mixtral_diagnostics.png)\n")
            if "margin_bos" in dgn:
                sec.append("![margins](../figures/ext3_mixtral_margins_L19.png)\n")
                rows = []
                for k in MIXTRAL_VARIANTS:
                    if f"margin_{k}_mean" in dgn:
                        rows.append([k, f"{dgn[f'margin_{k}_mean']:+.3f}", f"{dgn[f'margin_{k}_corr_with_bos']:.3f}", f"{dgn[f'margin_{k}_corr_with_nobos']:.3f}",
                                     f"{dgn[f'margin_{k}_mad_to_bos']:.3f}", f"{dgn[f'margin_{k}_mad_to_nobos']:.3f}"])
                sec.append("**L19 router margin E006 - E002 at the final position (clean prompts): per variant, and how close it is to the two reference runs**\n")
                sec.append(md_table(["variant", "mean margin", "corr with BOS", "corr with no BOS", "mean |diff| to BOS", "mean |diff| to no BOS"], rows,
                                    os.path.join(TABLES, "ext3_margins_L19")))
            if "ood" in dgn:
                o = dgn["ood"]
                sec.append("### Experiment 5: prompt likelihood (OOD check)\n")
                sec.append(f"Mean log-probability per content token (tokens 2..end of the prompt, same tokens in both protocols): with BOS {o['mean_ll_bos']:+.3f}, "
                           f"without BOS {o['mean_ll_nobos']:+.3f} (shift {o['mean_shift']:+.3f}; {o['frac_prompts_worse_without_bos']:.0%} of prompts are less likely without BOS). "
                           f"Log-probability of the object token at the final position: {o['logprob_true_bos']:+.3f} (BOS) vs {o['logprob_true_nobos']:+.3f} (no BOS). "
                           f"Prompts whose L19 top-2 set changed ({o['n_changed_routing']}/256) have a mean shift of {o['shift_changed']:+.3f} vs {o['shift_unchanged']:+.3f} for unchanged prompts "
                           f"(Mann-Whitney p = {o['mannwhitney_p']:.3f}, point-biserial r = {o['pointbiserial_r']:+.3f}, AUC = {o['auc_shift_predicts_change']:.3f}). "
                           f"Subject starts at position 0 in {o['frac_subject_at_pos0_nobos']:.0%} of the no-BOS prompts; routing change rate {o['routing_change_rate_subject_pos0']:.2f} for those vs "
                           f"{o['routing_change_rate_subject_later']:.2f} when the subject comes later (Fisher p = {o['fisher_p_subject_pos0_vs_change']:.3f}).\n")
                sec.append("![ood](../figures/ext3_mixtral_ood.png)\n")
    # ----- corpus
    cs, cnums = corpus_section(19, 6, 2)
    if cs:
        sec.append("### Experiment 3: corpus-level routing at L19 (H4)\n")
        sec.append(cs)
        sec.append("![corpus positions](../figures/ext3_mixtral_corpus_positions.png)\n")
        NUM["corpus"] = cnums
    # ----- Qwen3 symmetric test
    dfq, ctxq = substitution_table("qwen3", QWEN3_VARIANTS, 44, 69, MODELS["qwen3"]["n_controls"], ref_keys=("default", "eot"))
    if dfq is not None:
        sec.append("### Experiment 4: symmetric test on Qwen3-30B-A3B-Base (prepend `<|endoftext|>`)\n")
        sec.append(md_table(list(dfq.columns), dfq.values.tolist(), os.path.join(TABLES, "ext3_qwen3_symmetric")))
        fig_agreement(ctxq["curves"], "qwen3", 44, ("default", "eot"), os.path.join(FIGS, "ext3_qwen3_agreement.png"))
        sec.append("![qwen3 agreement](../figures/ext3_qwen3_agreement.png)\n")
        NUM["qwen3"] = dfq.to_dict(orient="records")
        NUM["qwen3_agreement"] = {f"{k}_vs_{r}": c.round(3).tolist() for (k, r), c in ctxq["curves"].items()}
        dq = diagnostics("qwen3", QWEN3_VARIANTS, 44, 69, None, ctxq)
        NUM["qwen3_diagnostics"] = dq
        if dq:
            sec.append("![qwen3 diagnostics](../figures/ext3_qwen3_diagnostics.png)\n")
    concl = os.path.join(SECTIONS, "ext3_conclusion.md")
    if os.path.exists(concl):
        sec.append(open(concl).read())
    with open(os.path.join(SECTIONS, "ext3_bos_mechanism.md"), "w") as f:
        f.write("\n".join(sec))
    with open(os.path.join(RESULTS, "ext3_numbers.json"), "w") as f:
        json.dump(NUM, f, indent=1, default=str)
    print("\n".join(sec))


if __name__ == "__main__":
    main()
