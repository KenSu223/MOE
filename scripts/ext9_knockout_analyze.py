"""ext9 knockout analysis (CPU): effects per condition and task, controls, sets vs random sets, double dissociation,
routing change, final-only and zero mode. Reads results/{qwen3,mixtral_bos}_knockout; writes results/ext9_knockout_summary.json,
results/tables/ext9_*.{md,csv}, results/figures/ext9_*.png.

Effect of a knockout on a task subset S (items evaluated under the condition's scope):
  F = (sum_i Delta_base(i) - sum_i Delta_ko(i)) / sum_i Delta_base(i)   (fraction of the clean margin lost, ratio of means)
  with a percentile bootstrap over units (WinoGrande pairs, CounterFact cases, IOI items, wiki windows; 5,000 resamples,
  the same resample matrix for every condition of a subset, so differences between conditions are paired);
  accuracy (Delta > 0), WinoGrande pair accuracy (both prompts), mean p(true), median rank, top-1 rate; wiki: mean
  per-token NLL change (nats) and its relative size.
Subsets: wg_margin (own margin pool minus discovery; primary), wg_all (every W1-W6 pair minus discovery), cf (Delta_clean >= 1
minus paper discovery), ioi (baseline Delta >= 1; primary) / ioi_all, wiki.
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext9_knockout as K
from moetrace.models import RESULTS

N_BOOT = 5000
TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")
SUBSETS = ("wg_margin", "wg_all", "cf", "ioi", "ioi_all", "wiki")
BASE_SIG = "none|all|reroute"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


class Model:
    def __init__(self, key):
        self.key = key
        self.cfg = K.KO[key]
        self.od = K.run_dir(key)
        self.it = K.build_items(key).set_index("item", drop=False)
        self.rows = K.load_rows(key)
        self.conds = {}
        for ph in K.PHASES:
            p = os.path.join(self.od, f"plan_{ph}.json")
            if os.path.exists(p):
                for c in json.load(open(p))["conditions"]:
                    c["mask"] = [tuple(m) for m in c["mask"]]
                    self.conds[c["cid"]] = c
        self.by_sig = {s: g.set_index("item") for s, g in self.rows.groupby("sig")}
        self.base = self.by_sig[BASE_SIG]
        it = self.it
        b = self.base
        ioi_ok = set(b.index[(b.task == "ioi") & (b.delta >= 1.0)])
        self.subset_items = {
            "wg_margin": set(it.item[(it.task == "wg") & it.in_margin]),
            "wg_all": set(it.item[it.task == "wg"]),
            "cf": set(it.item[it.task == "cf"]),
            "ioi": set(it.item[it.task == "ioi"]) & ioi_ok,
            "ioi_all": set(it.item[it.task == "ioi"]),
            "wiki": set(it.item[it.task == "wiki"]),
        }
        self.scope_items = {sc: set(it.item[K.scope_mask(it.reset_index(drop=True), sc)]) for sc in ("full", "sub", "margin", "margin_wiki", "sub_nowiki")}
        self._boot = {}
        # wiki per-token NLL: (sig, item) -> [126]
        self.wiki_tok = {}
        for f in sorted(os.listdir(self.od)):
            if f.startswith("ko_wiki_") and f.endswith(".npz"):
                z = np.load(os.path.join(self.od, f))
                for s_, x, v in zip(z["sig"], z["item"], z["nll"]):
                    self.wiki_tok[(str(s_), int(x))] = v

    def items(self, subset, scope):
        return sorted(self.subset_items[subset] & self.scope_items[scope])

    def units(self, subset, scope):
        k = (subset, scope)
        if k not in self._boot:
            its = self.items(subset, scope)
            u = self.it.unit.loc[its].values
            uniq, inv = np.unique(u, return_inverse=True)
            bi = np.random.default_rng(0).integers(0, len(uniq), size=(N_BOOT, len(uniq)), dtype=np.int32)
            self._boot[k] = (its, inv, len(uniq), bi)
        return self._boot[k]

    def effect(self, sig, subset, scope, keep_boot=False):
        """Effect of the intervention `sig` on a task subset, evaluated on the items of `scope`."""
        its, inv, nu, bi = self.units(subset, scope)
        if sig not in self.by_sig or len(its) == 0:
            return None
        ko = self.by_sig[sig]
        miss = [x for x in its if x not in ko.index]
        if miss:
            return None
        b = self.base.loc[its]
        k_ = ko.loc[its]
        out = {"n_items": len(its), "n_units": int(nu)}
        if subset == "wiki":
            d = (k_.nll_mean.values - b.nll_mean.values).astype(np.float64)
            us = np.bincount(inv, weights=d, minlength=nu)
            cnt = np.bincount(inv, minlength=nu).astype(np.float64)
            bs = us[bi].sum(1) / cnt[bi].sum(1)
            nb = float(b.nll_mean.mean())
            out.update(dnll=float(d.mean()), dnll_lo=float(np.percentile(bs, 2.5)), dnll_hi=float(np.percentile(bs, 97.5)),
                       nll_base=nb, dnll_rel=float(d.mean() / nb))
            tok = [self.wiki_tok.get((sig, x)) for x in its]
            tb = [self.wiki_tok.get((BASE_SIG, x)) for x in its]
            if tok and all(t is not None for t in tok) and all(t is not None for t in tb):
                dt = np.concatenate([a - c for a, c in zip(tok, tb)])
                out.update(dnll_tok_p95=float(np.percentile(dt, 95)), frac_tok_worse_0p1=float((dt > 0.1).mean()))
            if keep_boot:
                out["_boot"] = bs
            return out
        d0 = b.delta.values.astype(np.float64)
        d1 = k_.delta.values.astype(np.float64)
        s0 = np.bincount(inv, weights=d0, minlength=nu)
        s1 = np.bincount(inv, weights=d1, minlength=nu)
        S0, S1 = s0[bi].sum(1), s1[bi].sum(1)
        Fb = (S0 - S1) / S0
        out.update(F=float((d0.sum() - d1.sum()) / d0.sum()), F_lo=float(np.percentile(Fb, 2.5)), F_hi=float(np.percentile(Fb, 97.5)),
                   mean_delta_base=float(d0.mean()), mean_delta_ko=float(d1.mean()), mean_ddelta=float((d1 - d0).mean()),
                   acc_base=float((d0 > 0).mean()), acc_ko=float((d1 > 0).mean()),
                   flipped_frac=float(((d0 > 0) & (d1 <= 0)).mean()))
        if "p_true" in k_.columns and k_.p_true.notna().all():
            out.update(p_true_base=float(b.p_true.mean()), p_true_ko=float(k_.p_true.mean()),
                       rank_med_base=float(np.median(b.rank_true)), rank_med_ko=float(np.median(k_.rank_true)),
                       top1_base=float((b.top1.values == self.it.true_id.loc[its].values).mean()),
                       top1_ko=float((k_.top1.values == self.it.true_id.loc[its].values).mean()))
        if subset.startswith("wg"):
            pa0 = pd.Series(d0 > 0).groupby(inv).all()
            pa1 = pd.Series(d1 > 0).groupby(inv).all()
            out.update(pair_acc_base=float(pa0.mean()), pair_acc_ko=float(pa1.mean()))
        if keep_boot:
            out["_boot"] = Fb
        return out


def sig_of(c):
    return c["sig"]


def table_md(df: pd.DataFrame, path: str, title: str):
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(path + ".csv", index=False)
    with open(path + ".md", "w") as f:
        f.write(f"**{title}**\n\n")
        f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for r in df.itertuples(index=False):
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")


def fmtF(e, key="F"):
    if e is None:
        return "n/a"
    return f"{e[key]:+.3f} [{e[key + '_lo']:+.3f}, {e[key + '_hi']:+.3f}]"


def fmtN(e):
    if e is None:
        return "n/a"
    return f"{e['dnll']:+.4f} [{e['dnll_lo']:+.4f}, {e['dnll_hi']:+.4f}]"


def analyze_model(M: Model) -> dict:
    key, cfg = M.key, M.cfg
    out = {"label": K.MODELS[cfg["model"]]["label"], "run": cfg["run"], "n_rows": int(len(M.rows)),
           "n_items": {s: len(M.subset_items[s]) for s in SUBSETS}}
    # ---- baseline
    base = {}
    for s in SUBSETS:
        its = M.items(s, "full")
        b = M.base.loc[its]
        if s == "wiki":
            base[s] = {"n": len(its), "nll_mean": float(b.nll_mean.mean())}
        else:
            e = {"n": len(its), "mean_delta": float(b.delta.mean()), "acc": float((b.delta > 0).mean()),
                 "top1": float((b.top1.values == M.it.true_id.loc[its].values).mean()), "p_true": float(b.p_true.mean())}
            if s.startswith("wg"):
                inv = M.units(s, "full")[1]
                e["pair_acc"] = float(pd.Series(b.delta.values > 0).groupby(inv).all().mean())
                e["n_pairs"] = int(M.units(s, "full")[2])
            base[s] = e
    out["baseline"] = base
    # ---- every condition x subset on its own scope
    eff = {}
    for cid, c in M.conds.items():
        if cid == "base":
            continue
        eff[cid] = {s: M.effect(c["sig"], s, c["scope"]) for s in SUBSETS}
        eff[cid] = {s: v for s, v in eff[cid].items() if v is not None}
    out["effects"] = {cid: {s: {k: r3(v) for k, v in e.items()} for s, e in d.items()} for cid, d in eff.items()}
    out["conditions"] = {cid: {"sig": c["sig"], "scope": c["scope"], "phase": c["phase"], "pos": c["pos"], "mode": c["mode"],
                               "mask": [K.le(*m) for m in c["mask"]], "meta": c.get("meta", {})} for cid, c in M.conds.items()}
    targets = [K.le(*t) for t in K.all_targets(key)]
    task_of = {K.le(*t): task for task in ("wg", "cf") for t in cfg["targets"][task]}
    # ---- targets vs same-layer controls (sub scope)
    cs = json.load(open(os.path.join(M.od, "controls.json"))) if os.path.exists(os.path.join(M.od, "controls.json")) else None
    ctl = {}
    if cs:
        for t, v in cs["same_layer"].items():
            l = v["layer"]
            tsig = f"{t}|all|reroute"
            rec = {"task": v["task"], "layer": l, "controls": [K.le(l, x) for x in v["controls"]],
                   "freq_target": r3(v["freq_target"]), "freq_controls": [r3(x) for x in v["freq_controls"]], "by_subset": {}}
            for s in SUBSETS:
                et = M.effect(tsig, s, "sub")
                ec = [M.effect(f"{K.le(l, x)}|all|reroute", s, "sub") for x in v["controls"]]
                if et is None or any(e is None for e in ec):
                    continue
                m = "dnll" if s == "wiki" else "F"
                vals = np.array([e[m] for e in ec])
                tv = et[m]
                rec["by_subset"][s] = {"target": r3(tv), "controls_mean": r3(vals.mean()), "controls_sd": r3(vals.std(ddof=1)),
                                       "controls_max": r3(vals.max()), "controls_min": r3(vals.min()),
                                       "rank": int(1 + (vals > tv).sum()), "n": int(len(vals) + 1),
                                       "z": r3((tv - vals.mean()) / vals.std(ddof=1)) if vals.std(ddof=1) > 0 else None,
                                       "max_control": K.le(l, v["controls"][int(np.argmax(vals))]),
                                       "controls": {K.le(l, x): r3(e[m]) for x, e in zip(v["controls"], ec)}}
            ctl[t] = rec
    out["same_layer"] = ctl
    # ---- population sets vs frequency-matched random sets (sub scope)
    sets = {}
    if cs:
        ps = K.pop_sets(key)
        for task in ("wg", "cf"):
            for k in K.SET_SIZES:
                pop = ps[task][k]
                psig = f"{K.mask_str(sorted(pop))}|all|reroute"
                rs = cs["random"][f"{task}:k{k}"]["sets"]
                rsigs = [f"{K.mask_str(sorted(tuple(m) for m in s_))}|all|reroute" for s_ in rs]
                rec = {"members": [K.le(*m) for m in pop], "random_sets": [[K.le(*m) for m in s_] for s_ in rs], "by_subset": {}}
                for s in SUBSETS:
                    ep = M.effect(psig, s, "sub")
                    er = [M.effect(x, s, "sub") for x in rsigs]
                    if ep is None or any(e is None for e in er):
                        continue
                    m = "dnll" if s == "wiki" else "F"
                    vals = np.array([e[m] for e in er])
                    rec["by_subset"][s] = {"pop": r3(ep[m]), "pop_lo": r3(ep[m + "_lo"]), "pop_hi": r3(ep[m + "_hi"]),
                                           "rand_mean": r3(vals.mean()), "rand_max": r3(vals.max()), "rand_min": r3(vals.min()),
                                           "rand_sd": r3(vals.std(ddof=1)), "rank": int(1 + (vals > ep[m]).sum()),
                                           "z": r3((ep[m] - vals.mean()) / vals.std(ddof=1)) if vals.std(ddof=1) > 0 else None,
                                           "rand": [r3(x) for x in vals]}
                    # full-scope effect of the population set (or target) for the headline table
                    ef = M.effect(psig, s, "full")
                    if ef is not None:
                        rec["by_subset"][s]["pop_full"] = r3(ef[m])
                        rec["by_subset"][s]["pop_full_lo"] = r3(ef[m + "_lo"])
                        rec["by_subset"][s]["pop_full_hi"] = r3(ef[m + "_hi"])
                sets[f"{task}:k{k}"] = rec
    out["sets"] = sets
    # ---- double dissociation (full scope; WG = own margin pool, CF = pool), paired bootstrap over pairs and cases
    dis = {}
    wg_t = K.le(*cfg["primary"]["wg"])

    def boot_F(sig, subset):
        e = M.effect(sig, subset, "full", keep_boot=True)
        return e

    pairs = [(wg_t, K.le(*c)) for c in cfg["targets"]["cf"]]
    ps = K.pop_sets(key)
    for k in (3, 5, 10):
        pairs.append((f"pop:wg:k{k}", f"pop:cf:k{k}"))
    for a, c in pairs:
        sa = f"{a}|all|reroute" if not a.startswith("pop") else f"{K.mask_str(sorted(ps['wg'][int(a.split('k')[-1])]))}|all|reroute"
        sc_ = f"{c}|all|reroute" if not c.startswith("pop") else f"{K.mask_str(sorted(ps['cf'][int(c.split('k')[-1])]))}|all|reroute"
        ea_wg, ea_cf = boot_F(sa, "wg_margin"), boot_F(sa, "cf")
        ec_wg, ec_cf = boot_F(sc_, "wg_margin"), boot_F(sc_, "cf")
        if None in (ea_wg, ea_cf, ec_wg, ec_cf):
            continue
        cb = (ea_wg["_boot"] - ea_cf["_boot"]) - (ec_wg["_boot"] - ec_cf["_boot"])
        cv = (ea_wg["F"] - ea_cf["F"]) - (ec_wg["F"] - ec_cf["F"])
        rec = {"wg_expert": a, "cf_expert": c, "F_wgexp_on_wg": r3(ea_wg["F"]), "F_wgexp_on_cf": r3(ea_cf["F"]),
               "F_cfexp_on_wg": r3(ec_wg["F"]), "F_cfexp_on_cf": r3(ec_cf["F"]),
               "contrast": r3(cv), "contrast_lo": r3(np.percentile(cb, 2.5)), "contrast_hi": r3(np.percentile(cb, 97.5)),
               "wgexp_own_minus_other": r3(ea_wg["F"] - ea_cf["F"]),
               "wgexp_own_minus_other_ci": [r3(np.percentile(ea_wg["_boot"] - ea_cf["_boot"], q)) for q in (2.5, 97.5)],
               "cfexp_own_minus_other": r3(ec_cf["F"] - ec_wg["F"]),
               "cfexp_own_minus_other_ci": [r3(np.percentile(ec_cf["_boot"] - ec_wg["_boot"], q)) for q in (2.5, 97.5)]}
        dis[f"{a}__{c}"] = rec
    out["dissociation"] = dis
    # ---- routing change at the masked layer (single-expert reroute conditions; baseline rows routed to the expert)
    items_b, ri_b, NE = K.base_route_final(key)
    pos_of = {int(x): j for j, x in enumerate(items_b)}
    rt = {}
    single = sorted({c["sig"] for c in M.conds.values() if len(c["mask"]) == 1 and c["mode"] == "reroute"})
    for sig in single:
        mask, pos, mode = K.parse_sig(sig)
        (l, e), = mask
        ko = M.by_sig.get(sig)
        if ko is None:
            continue
        rec = {}
        for task in ("wg", "cf", "ioi"):
            sub = ko[ko.task == task]
            n_tot, n_hit, repl, other = len(sub), 0, {}, 0
            for x, rf in zip(sub.index, sub.route_final):
                b_set = set(int(v) for v in ri_b[l, pos_of[int(x)]])
                if e not in b_set:
                    continue
                n_hit += 1
                k_set = set(json.loads(rf)[str(l)][0])
                new = k_set - b_set
                if len(new) == 1 and (b_set - k_set) == {e}:
                    r_ = new.pop()
                    repl[r_] = repl.get(r_, 0) + 1
                else:
                    other += 1
            top = sorted(repl.items(), key=lambda kv: -kv[1])[:3]
            rec[task] = {"n_items": n_tot, "freq_final": r3(n_hit / max(n_tot, 1)), "n_routed": n_hit,
                         "frac_clean_swap": r3(sum(repl.values()) / max(n_hit, 1)),
                         "top_replacements": [[K.le(l, a), r3(c / max(n_hit, 1))] for a, c in top]}
        rt[sig] = rec
    out["routing_change"] = rt
    # knockout damage split by whether the expert was routed at the FINAL position in the baseline (targets, full scope)
    rs = {}
    for t in targets:
        l, e = K.parse_le(t)
        ko = M.by_sig.get(f"{t}|all|reroute")
        if ko is None:
            continue
        rec = {}
        for s in ("wg_margin", "cf", "ioi"):
            its = M.items(s, "full")
            routed = np.array([e in set(int(v) for v in ri_b[l, pos_of[int(x)]]) for x in its])
            d0 = M.base.loc[its].delta.values.astype(np.float64)
            d1 = ko.loc[its].delta.values.astype(np.float64)
            r_ = {}
            for nm, m_ in (("routed", routed), ("not_routed", ~routed)):
                if m_.sum() >= 10:
                    r_[nm] = {"n": int(m_.sum()), "F": r3((d0[m_].sum() - d1[m_].sum()) / d0[m_].sum()),
                              "mean_ddelta": r3((d1[m_] - d0[m_]).mean())}
            rec[s] = r_
        rs[t] = rec
    out["routing_split"] = rs
    # all-position routing frequency of the watched experts in the baseline (fraction of positions, per task)
    fs = sorted(f for f in os.listdir(M.od) if f.startswith("ko_baseallpos_full_"))
    if fs:
        zs = [np.load(os.path.join(M.od, f)) for f in fs]
        cnt = sum(z["counts"] for z in zs)
        tot = sum(z["totals"] for z in zs)
        tasks = [str(t) for t in zs[0]["tasks"]]
        wl = [int(l) for l in zs[0]["watch_layers"]]
        out["allpos_freq"] = {K.le(l, e): {t: r3(cnt[ti, wl.index(l), e] / max(tot[ti], 1)) for ti, t in enumerate(tasks)}
                              for l, e in [tuple(int(v) for v in p_) for p_ in zs[0]["watch_pairs"]]}
    # ---- final-only and zero mode vs all-position reroute (margin scope)
    fz = {}
    for t in targets:
        rec = {}
        for nm, sig in (("all_reroute", f"{t}|all|reroute"), ("final_reroute", f"{t}|final|reroute"), ("all_zero", f"{t}|all|zero")):
            rec[nm] = {}
            for s in ("wg_margin", "cf", "ioi", "wiki"):
                e = M.effect(sig, s, "sub")
                if e is not None:
                    rec[nm][s] = {k: r3(v) for k, v in e.items() if not k.startswith("_")}
        fz[t] = rec
    for task in ("wg", "cf"):
        for k in (3, 5, 10):
            mstr = K.mask_str(sorted(ps[task][k]))
            rec = {}
            for nm, sig in (("all_reroute", f"{mstr}|all|reroute"), ("final_reroute", f"{mstr}|final|reroute")):
                rec[nm] = {}
                for s in ("wg_margin", "cf", "ioi"):
                    e = M.effect(sig, s, "sub")
                    if e is not None:
                        rec[nm][s] = {k_: r3(v) for k_, v in e.items() if not k_.startswith("_")}
            fz[f"pop:{task}:k{k}"] = rec
    out["final_zero"] = fz
    out["targets"] = targets
    out["target_task"] = task_of
    out["primary"] = {t: K.le(*v) for t, v in cfg["primary"].items()}
    out["pop_sets"] = {t: {str(k): [K.le(*m) for m in v] for k, v in d.items()} for t, d in K.pop_sets(key).items()}
    out["pop_overlap"] = {str(k): sorted(set(out["pop_sets"]["wg"][str(k)]) & set(out["pop_sets"]["cf"][str(k)])) for k in K.SET_SIZES}
    return out


def tables_and_figures(S: dict):
    rows = []
    for key, d in S["models"].items():
        for t in d["targets"]:
            e = d["effects"].get(f"tgt:{t}", {})
            rows.append({"model": d["label"], "expert": t, "selected on": d["target_task"][t].upper(),
                         "WG F": fmtF(e.get("wg_margin")), "WG pair acc": f"{d['baseline']['wg_margin']['pair_acc']:.3f} -> {e['wg_margin']['pair_acc_ko']:.3f}" if e.get("wg_margin") else "n/a",
                         "CF F": fmtF(e.get("cf")), "CF acc": f"{d['baseline']['cf']['acc']:.3f} -> {e['cf']['acc_ko']:.3f}" if e.get("cf") else "n/a",
                         "IOI F": fmtF(e.get("ioi")), "wiki dNLL": fmtN(e.get("wiki")),
                         "WG all F": fmtF(e.get("wg_all"))})
        for task in ("wg", "cf"):
            for k in (3, 5, 10):
                e = d["effects"].get(f"pop:{task}:k{k}", {})
                if not e:
                    continue
                rows.append({"model": d["label"], "expert": f"{task.upper()} top-{k}", "selected on": task.upper(),
                             "WG F": fmtF(e.get("wg_margin")), "WG pair acc": f"{d['baseline']['wg_margin']['pair_acc']:.3f} -> {e['wg_margin']['pair_acc_ko']:.3f}" if e.get("wg_margin") else "n/a",
                             "CF F": fmtF(e.get("cf")), "CF acc": f"{d['baseline']['cf']['acc']:.3f} -> {e['cf']['acc_ko']:.3f}" if e.get("cf") else "n/a",
                             "IOI F": fmtF(e.get("ioi")), "wiki dNLL": fmtN(e.get("wiki")), "WG all F": fmtF(e.get("wg_all"))})
    table_md(pd.DataFrame(rows), os.path.join(TABLES, "ext9_knockout_targets"),
             "Knockout (all positions, reroute), full scope: fraction of the clean margin lost F [95% CI]; accuracy baseline -> knockout")
    rows = []
    for key, d in S["models"].items():
        for t, v in d["same_layer"].items():
            for s in ("wg_margin", "cf", "ioi", "wiki"):
                x = v["by_subset"].get(s)
                if x is None:
                    continue
                rows.append({"model": d["label"], "target": t, "task of target": v["task"].upper(), "subset": s,
                             "target effect": x["target"], "controls mean": x["controls_mean"], "controls max": x["controls_max"],
                             "max control": x["max_control"], "rank": f"{x['rank']}/{x['n']}", "z": x["z"]})
    table_md(pd.DataFrame(rows), os.path.join(TABLES, "ext9_knockout_controls"),
             "Targets vs same-layer controls (sub scope; F, wiki: dNLL in nats/token); rank 1 = most damaging")
    rows = []
    for key, d in S["models"].items():
        for name, v in d["sets"].items():
            for s in ("wg_margin", "cf", "ioi", "wiki"):
                x = v["by_subset"].get(s)
                if x is None:
                    continue
                rows.append({"model": d["label"], "set": name, "subset": s, "population set": x["pop"],
                             "random mean": x["rand_mean"], "random max": x["rand_max"], "rank": f"{x['rank']}/6", "z": x["z"],
                             "population set (full scope)": x.get("pop_full")})
    table_md(pd.DataFrame(rows), os.path.join(TABLES, "ext9_knockout_sets"),
             "Population top-k sets vs 5 frequency-matched random sets (sub scope); rank 1 = most damaging")
    rows = []
    for key, d in S["models"].items():
        for name, v in d["dissociation"].items():
            rows.append({"model": d["label"], "WG expert(s)": v["wg_expert"], "CF expert(s)": v["cf_expert"],
                         "F WGexp on WG": v["F_wgexp_on_wg"], "F WGexp on CF": v["F_wgexp_on_cf"],
                         "F CFexp on WG": v["F_cfexp_on_wg"], "F CFexp on CF": v["F_cfexp_on_cf"],
                         "contrast [95% CI]": f"{v['contrast']:+.3f} [{v['contrast_lo']:+.3f}, {v['contrast_hi']:+.3f}]"})
    table_md(pd.DataFrame(rows), os.path.join(TABLES, "ext9_knockout_dissociation"),
             "Double dissociation (full scope): (F_WGexp(WG) - F_WGexp(CF)) - (F_CFexp(WG) - F_CFexp(CF))")
    rows = []
    for key, d in S["models"].items():
        for t in d["targets"]:
            r = d["routing_change"].get(f"{t}|all|reroute", {})
            ap = d.get("allpos_freq", {}).get(t, {})
            for task in ("wg", "cf", "ioi"):
                x = r.get(task)
                if not x:
                    continue
                rows.append({"model": d["label"], "expert": t, "task": task, "routed at final (baseline)": x["freq_final"],
                             "routed, all positions": ap.get(task), "clean one-for-one swap": x["frac_clean_swap"],
                             "top replacements (share)": ", ".join(f"{a} ({b})" for a, b in x["top_replacements"])})
    table_md(pd.DataFrame(rows), os.path.join(TABLES, "ext9_knockout_routing"),
             "Which expert takes the slot (all-position reroute; final position at the masked layer, items where the expert was routed)")
    rows = []
    for key, d in S["models"].items():
        for name, v in d["final_zero"].items():
            for s in ("wg_margin", "cf", "ioi", "wiki"):
                cells = {}
                for nm in ("all_reroute", "final_reroute", "all_zero"):
                    x = v.get(nm, {}).get(s)
                    if x:
                        cells[nm] = x.get("F", x.get("dnll"))
                if cells:
                    rows.append({"model": d["label"], "condition": name, "subset": s, **{k: cells.get(k) for k in ("all_reroute", "final_reroute", "all_zero")}})
    table_md(pd.DataFrame(rows), os.path.join(TABLES, "ext9_knockout_final_zero"),
             "All positions vs final position only (reroute) vs zero mode (sub scope; F, wiki dNLL)")
    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIGS, exist_ok=True)
        ms = list(S["models"].items())
        fig, axes = plt.subplots(1, len(ms), figsize=(6.5 * len(ms), 4.2), squeeze=False)
        for ax, (key, d) in zip(axes[0], ms):
            xs = []
            for i, t in enumerate(d["targets"]):
                v = d["same_layer"].get(t, {}).get("by_subset", {})
                for j, (s, col) in enumerate((("wg_margin", "#3b6fb6"), ("cf", "#c8553d"))):
                    x = v.get(s)
                    if not x:
                        continue
                    xpos = i * 3 + j
                    ax.scatter([xpos] * len(x["controls"]), list(x["controls"].values()), s=10, color="#999999", zorder=2)
                    ax.scatter([xpos], [x["target"]], s=60, color=col, zorder=3, marker="D")
                xs.append((i * 3 + 0.5, t + f"\n({d['target_task'][t].upper()})"))
            ax.set_xticks([a for a, _ in xs])
            ax.set_xticklabels([b for _, b in xs], fontsize=8)
            ax.axhline(0, color="k", lw=0.5)
            nl = d["effects"].get("null", {})
            for s, col in (("wg_margin", "#3b6fb6"), ("cf", "#c8553d")):
                if nl.get(s):
                    ax.axhspan(min(0, nl[s]["F_lo"]), max(0, nl[s]["F_hi"]), color=col, alpha=0.08, lw=0)
            from matplotlib.lines import Line2D
            if ax is axes[0][0]:
                ax.legend(handles=[Line2D([], [], marker="D", ls="", color="#3b6fb6", label="target expert, effect on WinoGrande"),
                               Line2D([], [], marker="D", ls="", color="#c8553d", label="target expert, effect on CounterFact"),
                               Line2D([], [], marker="o", ls="", color="#999999", label="same-layer controls (same task)")],
                          fontsize=7, loc="upper center")
            ax.set_ylabel("fraction of clean margin lost F (sub scope)")
            ax.set_title(f"{d['label']}: knockout of each target vs its same-layer controls (shaded: null 95% CI)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "ext9_knockout_controls.png"), dpi=130)
        plt.close(fig)
        fig, axes = plt.subplots(1, len(ms), figsize=(6.5 * len(ms), 4.2), squeeze=False)
        for ax, (key, d) in zip(axes[0], ms):
            for task, col in (("wg", "#3b6fb6"), ("cf", "#c8553d")):
                for s, ls in (("wg_margin", "-"), ("cf", "--")):
                    ks, pv, rv = [], [], []
                    for k in K.SET_SIZES:
                        x = d["sets"].get(f"{task}:k{k}", {}).get("by_subset", {}).get(s)
                        if x:
                            ks.append(k)
                            pv.append(x["pop"])
                            rv.append(x["rand_mean"])
                    if ks:
                        ax.plot(ks, pv, ls, color=col, marker="o", label=f"{task.upper()} top-k on {s}")
                        ax.plot(ks, rv, ls, color=col, alpha=0.35, marker="x", label=f"random (matched) on {s}")
            ax.set_xscale("log")
            ax.set_xticks(list(K.SET_SIZES))
            ax.set_xticklabels([str(k) for k in K.SET_SIZES])
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("set size k")
            ax.set_ylabel("fraction of clean margin lost (sub scope)")
            ax.set_title(d["label"], fontsize=9)
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "ext9_knockout_sets.png"), dpi=130)
        plt.close(fig)
    except Exception as ex:  # figures are optional
        log(f"figure error: {ex}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3,mixtral")
    ap.add_argument("--out", default=os.path.join(RESULTS, "ext9_knockout_summary.json"))
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--tables-dir", default=None, help="write tables / figures here instead of results/tables, results/figures")
    args = ap.parse_args()
    global TABLES, FIGS
    if args.tables_dir:
        TABLES = FIGS = args.tables_dir
    S = {"models": {}, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_boot": N_BOOT}
    for key in args.models.split(","):
        if not os.path.exists(os.path.join(K.run_dir(key), "plan_full.json")):
            continue
        log(f"analysing {key}")
        M = Model(key)
        S["models"][key] = analyze_model(M)
        meta = json.load(open(os.path.join(M.od, "run_meta.json")))
        S["models"][key]["gpu_s"] = round(sum(p.get("pass_s", 0) for ph in meta.get("phases", {}).values() for p in ph.get("passes", [])), 1)
        S["models"][key]["phases_complete"] = {ph: bool(v.get("complete")) for ph, v in meta.get("phases", {}).items()}
    with open(args.out, "w") as f:
        json.dump(S, f, indent=1)
    if not args.no_tables:
        tables_and_figures(S)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
