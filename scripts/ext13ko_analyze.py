"""ext13ko analysis (CPU): knockout effects per Phase-5 condition and model, controls, sets vs random sets, final-only
masks, routing change, self-repair, the P5-8 contrasts across conditions and the cross-condition matrix.

Reads results/p5_<proto>_ko<suffix>/<cond>/; writes results/ext13ko_summary<suffix>.json, results/tables/ext13ko_*.md|csv,
results/figures/ext13ko_*.png.

    python scripts/ext13ko_analyze.py [--protos qwen3,mixtral_bos] [--conds ...] [--suffix _smoke] [--no-figs]

Effect on the evaluated items S of an intervention (its own scope: full = eval items = validation + replication pairs,
sub = main validation pairs): F = (sum Delta_base - sum Delta_ko) / sum Delta_base (ratio of means; bootstrap over the
condition's units = template seeds or pairs, 5,000 resamples, the same resample matrix for every intervention of a scope,
so within-condition differences are paired). Differences between conditions use independent resamples.
Self-repair (items whose final position routes the target in the baseline): W = the expert's own direct write in the
baseline (DLA of c_e, final norm frozen at the prompt's scale), N = the net change of the masked layer's MoE direct write
(baseline minus knockout row; includes the replacement expert under reroute), K = -(mean Delta_ko - mean Delta_base);
lost share K / W (ext9_synthesis_selfrepair convention) and K / N.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext13ko as K
from moetrace.models import RESULTS

N_BOOT = K.N_BOOT
TABLES = os.path.join(RESULTS, "tables")
FIGS = os.path.join(RESULTS, "figures")
BASE = "none|all|reroute"
MIN_W = 0.1  # logits: below this mean direct write the K / W ratio is not interpretable
NULL = "none|all|null"
CLAB = {"f1": "F1 a→e (fill-in)", "dir_f1": "776 set, a→e (fill-in)", "dir_opt": "776 set, e→a (option swap)",
        "f2e": "a→e natural (blank at end)", "vt_e2a_orig": "templates e→a, orig verb", "vt_e2a_neu": "templates e→a, neutral verb",
        "vt_a2e_orig": "templates a→e, orig verb", "vt_a2e_neu": "templates a→e, neutral verb",
        "cp_vt_orig": "copy: templates e→a + prefix, orig", "cp_vt_neu": "copy: templates e→a + prefix, neutral",
        "cp_nat": "copy: 776 set e→a + prefix", "f1pre": "F1 pre-neutral (smoke)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def r4(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return round(x, 4) if np.isfinite(x) else None


class Cond:
    def __init__(self, cond, proto, suffix=""):
        self.e = K.entry(cond, proto)
        self.cond, self.proto = cond, proto
        self.od = K.out_dir(self.e, suffix)
        self.it = pd.read_parquet(os.path.join(self.od, "items.parquet")).set_index("item", drop=False)
        self.rows = K.load_rows(self.od)
        self.conds = {}
        for ph in ("main", "cross"):
            p = os.path.join(self.od, f"plan_{ph}.json")
            if os.path.exists(p):
                pl = json.load(open(p))
                if ph == "main":
                    self.plan = pl
                for c in pl["conditions"]:
                    c["mask"] = [tuple(m) for m in c["mask"]]
                    self.conds[c["cid"]] = c
        self.by_sig = {s: g.drop_duplicates("item").set_index("item") for s, g in self.rows.groupby("sig")}
        self.base = self.by_sig[BASE]
        self.scope_items = {sc: sorted(self.it.item[K.scope_mask(self.it.reset_index(drop=True), sc)]) for sc in ("full", "sub")}
        self._boot = {}
        self.tg = self.plan["targets"]
        self.bf = K.load_basefinal(self.od)
        self.bf_pos = {int(x): j for j, x in enumerate(self.bf["items"])} if self.bf else {}

    def units(self, scope, items=None):
        key = (scope, None if items is None else tuple(items))
        if key not in self._boot:
            its = self.scope_items[scope] if items is None else list(items)
            u = self.it.unit.loc[its].values
            uniq, inv = np.unique(u, return_inverse=True)
            seed = abs(hash((self.cond, self.proto, scope))) % (2**31) if items is not None else 0
            bi = np.random.default_rng(seed).integers(0, len(uniq), size=(N_BOOT, len(uniq)), dtype=np.int32)
            self._boot[key] = (its, inv, len(uniq), bi)
        return self._boot[key]

    def effect(self, sig, scope, items=None, keep_boot=False):
        its, inv, nu, bi = self.units(scope, items)
        if sig not in self.by_sig or not len(its):
            return None
        ko = self.by_sig[sig]
        if any(x not in ko.index for x in its):
            return None
        b = self.base.loc[its]
        k_ = ko.loc[its]
        d0 = b.delta.values.astype(np.float64)
        d1 = k_.delta.values.astype(np.float64)
        s0 = np.bincount(inv, weights=d0, minlength=nu)
        s1 = np.bincount(inv, weights=d1, minlength=nu)
        S0, S1 = s0[bi].sum(1), s1[bi].sum(1)
        Fb = (S0 - S1) / S0
        out = {"n_items": len(its), "n_units": int(nu), "F": float((d0.sum() - d1.sum()) / d0.sum()),
               "F_lo": float(np.percentile(Fb, 2.5)), "F_hi": float(np.percentile(Fb, 97.5)),
               "mean_delta_base": float(d0.mean()), "mean_delta_ko": float(d1.mean()), "mean_ddelta": float((d1 - d0).mean()),
               "acc_base": float((d0 > 0).mean()), "acc_ko": float((d1 > 0).mean()),
               "flipped_frac": float(((d0 > 0) & (d1 <= 0)).mean())}
        pid = self.it.pair_idx.loc[its].values
        pa0 = pd.Series(d0 > 0).groupby(pid).all()
        pa1 = pd.Series(d1 > 0).groupby(pid).all()
        out.update(pair_acc_base=float(pa0.mean()), pair_acc_ko=float(pa1.mean()))
        if "p_true" in k_.columns:
            out.update(p_true_base=float(b.p_true.mean()), p_true_ko=float(k_.p_true.mean()))
        if keep_boot:
            out["_boot"] = Fb
        return out

    def routed(self, l, x, its):
        """Was (l, x) routed at the final position of the baseline row (bool array over its)."""
        ri = self.bf["route_idx"]
        return np.array([x in set(int(v) for v in ri[l, self.bf_pos[int(i)]]) for i in its])


def sig1(name, pos="all"):
    return f"{name}|{pos}|reroute"


def analyse(C: Cond) -> dict:
    tg = C.tg
    out = {"cond": C.cond, "proto": C.proto, "label": K.PROTO[C.proto]["label"], "direction": C.e["direction"],
           "targets": tg, "n_rows": int(len(C.rows)), "unit_col": json.load(open(os.path.join(C.od, "items_meta.json")))["unit_col"]}
    # baseline
    base = {}
    for sc in ("full", "sub"):
        its = C.scope_items[sc]
        b = C.base.loc[its]
        pid = C.it.pair_idx.loc[its].values
        base[sc] = {"n_items": len(its), "n_pairs": int(len(set(pid))), "n_units": int(C.it.unit.loc[its].nunique()),
                    "mean_delta": float(b.delta.mean()), "acc": float((b.delta > 0).mean()),
                    "pair_acc": float(pd.Series(b.delta.values > 0).groupby(pid).all().mean()),
                    "p_true": float(b.p_true.mean()) if "p_true" in b else None}
    out["baseline"] = base
    # every intervention on its own scope (and the full-scope ones also on sub)
    eff = {}
    for cid, c in C.conds.items():
        if cid == "base":
            continue
        d = {c["scope"]: C.effect(c["sig"], c["scope"])}
        if c["scope"] == "full":
            d["sub"] = C.effect(c["sig"], "sub")
        eff[cid] = {k: {kk: r4(vv) for kk, vv in v.items()} for k, v in d.items() if v is not None}
    out["effects"] = eff
    out["conditions"] = {cid: {"sig": c["sig"], "scope": c["scope"], "pos": c["pos"], "mask": [K.le(*m) for m in c["mask"]],
                               "meta": c.get("meta", {})} for cid, c in C.conds.items()}
    # targets vs same-layer controls (sub scope)
    cs = json.load(open(os.path.join(C.od, "controls.json")))
    ctl = {}
    for t, v in cs["same_layer"].items():
        l = v["layer"]
        et = C.effect(sig1(t), "sub")
        ec = [C.effect(sig1(K.le(l, y)), "sub") for y in v["controls"]]
        if et is None or any(x is None for x in ec):
            continue
        vals = np.array([x["F"] for x in ec])
        ctl[t] = {"layer": l, "target": r4(et["F"]), "controls_mean": r4(vals.mean()), "controls_sd": r4(vals.std(ddof=1)),
                  "controls_max": r4(vals.max()), "rank": int(1 + (vals > et["F"]).sum()), "n": int(len(vals) + 1),
                  "z": r4((et["F"] - vals.mean()) / vals.std(ddof=1)) if vals.std(ddof=1) > 0 else None,
                  "max_control": K.le(l, v["controls"][int(np.argmax(vals))]),
                  "controls": {K.le(l, y): r4(x["F"]) for y, x in zip(v["controls"], ec)},
                  "freq_target": r4(v["freq_target"])}
    out["same_layer"] = ctl
    # population sets vs frequency-matched random sets (sub scope)
    rk = json.load(open(os.path.join(C.od, "pop_rank.json")))["ranking"]
    sets = {}
    for kk, v in cs["random"].items():
        k = int(kk[1:])
        members = [tuple(m) for m in v["members"]]
        psig = f"{K.mask_str(sorted(members))}|all|reroute"
        ep = C.effect(psig, "sub")
        er = [C.effect(f"{K.mask_str(sorted(tuple(m) for m in s_))}|all|reroute", "sub") for s_ in v["sets"]]
        if ep is None or any(x is None for x in er):
            continue
        vals = np.array([x["F"] for x in er])
        ef = C.effect(psig, "full")
        sets[kk] = {"members": [K.le(*m) for m in members], "pop": r4(ep["F"]), "pop_lo": r4(ep["F_lo"]), "pop_hi": r4(ep["F_hi"]),
                    "rand_mean": r4(vals.mean()), "rand_max": r4(vals.max()), "rand_min": r4(vals.min()),
                    "rand_sd": r4(vals.std(ddof=1)), "rank": int(1 + (vals > ep["F"]).sum()), "rand": [r4(x) for x in vals],
                    "pop_full": r4(ef["F"]) if ef else None, "pop_full_lo": r4(ef["F_lo"]) if ef else None,
                    "pop_full_hi": r4(ef["F_hi"]) if ef else None,
                    "pair_acc_ko_full": r4(ef["pair_acc_ko"]) if ef else None}
    out["sets"] = sets
    out["pop_top10"] = [K.le(l, x) for l, x, _ in rk[:10]]
    # final-only vs all positions (sub scope)
    fz = {}
    for t in tg["all"]:
        a, f = C.effect(sig1(t), "sub"), C.effect(sig1(t, "final"), "sub")
        if a and f:
            fz[t] = {"all": r4(a["F"]), "final": r4(f["F"]), "final_over_all": r4(f["F"] / a["F"]) if abs(a["F"]) > 1e-9 else None,
                     "final_lo": r4(f["F_lo"]), "final_hi": r4(f["F_hi"])}
    if "10" in [k[1:] for k in cs["random"]]:
        m10 = sorted(tuple(m) for m in cs["random"]["k10"]["members"])
        a, f = C.effect(f"{K.mask_str(m10)}|all|reroute", "sub"), C.effect(f"{K.mask_str(m10)}|final|reroute", "sub")
        if a and f:
            fz["pop:k10"] = {"all": r4(a["F"]), "final": r4(f["F"]), "final_over_all": r4(f["F"] / a["F"]) if abs(a["F"]) > 1e-9 else None,
                             "final_lo": r4(f["F_lo"]), "final_hi": r4(f["F_hi"])}
    out["final_vs_all"] = fz
    # routing frequency at the final position (baseline, eval items), routing split, replacements, self-repair
    its = C.scope_items["full"]
    rs, rc, sr = {}, {}, {}
    for t in tg["all"]:
        l, x = K.parse_le(t)
        ko = C.by_sig.get(sig1(t))
        if ko is None or not C.bf:
            continue
        routed = C.routed(l, x, its)
        d0 = C.base.loc[its].delta.values.astype(np.float64)
        d1 = ko.loc[its].delta.values.astype(np.float64)
        rec = {"freq_final": r4(routed.mean()), "n_routed": int(routed.sum())}
        for nm, m_ in (("routed", routed), ("not_routed", ~routed)):
            if m_.sum() >= 5:
                rec[nm] = {"n": int(m_.sum()), "F": r4((d0[m_].sum() - d1[m_].sum()) / d0[m_].sum()), "mean_ddelta": r4((d1[m_] - d0[m_]).mean())}
        rs[t] = rec
        # replacements
        repl, other = {}, 0
        ri = C.bf["route_idx"]
        for i, hit in zip(its, routed):
            if not hit:
                continue
            bset = set(int(v) for v in ri[l, C.bf_pos[int(i)]])
            kset = set(json.loads(ko.route_final.loc[i])[str(l)][0])
            new = kset - bset
            if len(new) == 1 and (bset - kset) == {x}:
                r_ = new.pop()
                repl[r_] = repl.get(r_, 0) + 1
            else:
                other += 1
        n_hit = max(int(routed.sum()), 1)
        rc[t] = {"frac_clean_swap": r4(sum(repl.values()) / n_hit),
                 "top_replacements": [[K.le(l, a), r4(c / n_hit)] for a, c in sorted(repl.items(), key=lambda kv: -kv[1])[:3]]}
        # self-repair on routed items
        if routed.sum() >= 5:
            ii = [i for i, h in zip(its, routed) if h]
            jj = np.array([C.bf_pos[int(i)] for i in ii])
            slot = np.array([int(np.nonzero(ri[l, j] == x)[0][0]) for j in jj])
            W = C.bf["contrib_dla"][l, jj, slot] / C.bf["final_rms"][jj]
            layer_base = C.bf["contrib_dla"][l, jj].sum(-1) / C.bf["final_rms"][jj]
            layer_ko = np.array([json.loads(ko.moe_dla_masked.loc[i])[str(l)] for i in ii])
            N = layer_base - layer_ko
            Kd = -(ko.delta.loc[ii].values.astype(np.float64) - C.base.delta.loc[ii].values.astype(np.float64))
            u = C.it.unit.loc[ii].values
            uniq, inv = np.unique(u, return_inverse=True)
            bi = np.random.default_rng(l * 1000 + x).integers(0, len(uniq), size=(2000, len(uniq)))
            sK, sW, sN = (np.bincount(inv, weights=a, minlength=len(uniq)) for a in (Kd, W, N))
            bK, bW, bN = sK[bi].sum(1), sW[bi].sum(1), sN[bi].sum(1)
            sr[t] = {"n_items": len(ii), "direct_write_W": r4(W.mean()), "net_layer_direct_N": r4(N.mean()), "knockout_K": r4(Kd.mean()),
                     "lost_K_over_W": r4(Kd.sum() / W.sum()), "lost_K_over_W_ci": [r4(np.percentile(bK / bW, q)) for q in (2.5, 97.5)],
                     "lost_K_over_N": r4(Kd.sum() / N.sum()), "lost_K_over_N_ci": [r4(np.percentile(bK / bN, q)) for q in (2.5, 97.5)],
                     "N_over_W": r4(N.sum() / W.sum())}
    out["routing_split"] = rs
    out["routing_change"] = rc
    out["selfrepair"] = sr
    # strata (names / objects; association strata when present), full scope, main targets + top-10
    strata = {}
    for col in ("names", "stratum_assoc", "stratum_local_assoc", "local_assoc", "stratum", "assoc_stratum", "variant", "order"):
        if col not in C.it.columns:
            continue
        vals = C.it[col].loc[its]
        if vals.nunique() < 2:
            continue
        for val in sorted(vals.dropna().unique(), key=str):
            sub_its = [i for i, v in zip(its, vals.values) if v == val]
            if len(set(C.it.unit.loc[sub_its])) < 5:
                continue
            rec = {"n_items": len(sub_its)}
            for nm, sig in [(t, sig1(t)) for t in tg["all"]] + ([("pop:k10", f"{K.mask_str(sorted(K.parse_le(m) for m in sets['k10']['members']))}|all|reroute")] if "k10" in sets else []):
                ef = C.effect(sig, "full", items=sub_its)
                if ef:
                    rec[nm] = {"F": r4(ef["F"]), "F_lo": r4(ef["F_lo"]), "F_hi": r4(ef["F_hi"])}
            strata[f"{col}={val}"] = rec
    out["strata"] = strata
    # cross phase
    cross = {}
    for cid, c in C.conds.items():
        if cid.startswith("cross:"):
            ef = C.effect(c["sig"], "sub")
            if ef:
                cross[cid[6:]] = {"F": r4(ef["F"]), "F_lo": r4(ef["F_lo"]), "F_hi": r4(ef["F_hi"]), "from": c["meta"].get("from_cond")}
    out["cross"] = cross
    # GPU time
    meta = json.load(open(os.path.join(C.od, "run_meta.json")))
    out["gpu_s"] = round(sum(p.get("pass_s", 0) for ph in meta.get("phases", {}).values() for p in ph.get("passes", [])), 1)
    out["phases_complete"] = {ph: bool(v.get("complete")) for ph, v in meta.get("phases", {}).items()}
    return out


def boot_of(C: Cond, sig, scope="full"):
    e = C.effect(sig, scope, keep_boot=True)
    return e


def contrasts(models: dict, summ: dict) -> dict:
    """P5-8 and direction contrasts of the WinoGrande expert's F (full scope), independent bootstrap per condition."""
    pairs = [("cp_vt_orig", "vt_e2a_orig", "copy availability (templates, orig verb): C2 - C1"),
             ("cp_vt_neu", "vt_e2a_neu", "copy availability (templates, neutral verb): C2 - C1"),
             ("cp_nat", "dir_opt", "copy availability (natural 776 pairs): C2 - C1"),
             ("vt_a2e_orig", "cp_vt_orig", "direction (templates, orig; copy fixed): C3 - C2"),
             ("vt_a2e_neu", "cp_vt_neu", "direction (templates, neutral; copy fixed): C3 - C2"),
             ("dir_f1", "cp_nat", "direction (natural 776 pairs; copy fixed): C3 - C2"),
             ("dir_f1", "dir_opt", "direction on identical items (fill-in a→e - option swap e→a)"),
             ("vt_a2e_orig", "vt_e2a_orig", "direction (templates, orig): a→e - e→a"),
             ("vt_e2a_neu", "vt_e2a_orig", "verb: neutral - orig (e→a)")]
    out = {}
    for proto, conds in models.items():
        wg = K.le(*K.PROTO[proto]["wg"])
        for a, b, lab in pairs:
            if a not in conds or b not in conds:
                continue
            for what in ("wg", "selected_b", "pop10"):
                Ca, Cb = conds[a], conds[b]
                if what == "wg":
                    sa = sb = sig1(wg)
                    name = wg
                elif what == "selected_b":  # the e->a / reference condition's own selected expert on both
                    sel = Cb.tg.get("selected")
                    if not sel or sel == wg:
                        continue
                    sa = sb = sig1(sel)
                    name = sel
                else:
                    ka = summ[proto][a]["sets"].get("k10")
                    kb = summ[proto][b]["sets"].get("k10")
                    if not ka or not kb:
                        continue
                    sa = f"{K.mask_str(sorted(K.parse_le(m) for m in ka['members']))}|all|reroute"
                    sb = f"{K.mask_str(sorted(K.parse_le(m) for m in kb['members']))}|all|reroute"
                    name = "own top-10"
                ea, eb = boot_of(Ca, sa), boot_of(Cb, sb)
                if ea is None or eb is None:
                    continue
                d = ea["_boot"] - eb["_boot"]
                rec = {"proto": proto, "a": a, "b": b, "label": lab, "expert": name,
                       "F_a": r4(ea["F"]), "F_b": r4(eb["F"]), "diff": r4(ea["F"] - eb["F"]),
                       "diff_lo": r4(np.percentile(d, 2.5)), "diff_hi": r4(np.percentile(d, 97.5))}
                pr = paired(Ca, sa, Cb, sb)
                if pr:
                    rec["paired"] = pr
                out[f"{proto}:{a}-{b}:{what}"] = rec
    return out


def match_key(C: Cond) -> pd.Series:
    """Source-item key of each item: base_pair_id (copy-prefix conditions) else pair_id (templates: 'seed|variant|X|Y';
    natural pairs: the WinoGrande twin id, shared by the fill-in and option-swap formats of the 776 set)."""
    if "base_pair_id" in C.it.columns and C.it.base_pair_id.notna().all():
        return C.it.base_pair_id.astype(str)
    return C.it.pair_id.astype(str)


def paired(Ca: Cond, sa: str, Cb: Cond, sb: str, n_boot: int = N_BOOT):
    """F_a - F_b on the source items present in BOTH conditions' eval sets (both twins; both prefix orders on the copy
    side), joint bootstrap over the shared units (template seeds / source pairs)."""
    if sa not in Ca.by_sig or sb not in Cb.by_sig:
        return None
    ka, kb = match_key(Ca), match_key(Cb)
    ia = [i for i in Ca.scope_items["full"]]
    ib = [i for i in Cb.scope_items["full"]]
    common = set(ka.loc[ia]) & set(kb.loc[ib])
    if len(common) < 10:
        return None
    ia = [i for i in ia if ka.loc[i] in common and i in Ca.by_sig[sa].index]
    ib = [i for i in ib if kb.loc[i] in common and i in Cb.by_sig[sb].index]
    ua, ub = Ca.it.unit.loc[ia].astype(str).values, Cb.it.unit.loc[ib].astype(str).values
    U = np.unique(np.concatenate([ua, ub]))
    pos = {u: j for j, u in enumerate(U)}
    rng = np.random.default_rng(7)
    bi = rng.integers(0, len(U), size=(n_boot, len(U)))
    res = {}
    for nm, C, its, us, sig in (("a", Ca, ia, ua, sa), ("b", Cb, ib, ub, sb)):
        inv = np.array([pos[u] for u in us])
        d0 = C.base.delta.loc[its].values.astype(np.float64)
        d1 = C.by_sig[sig].delta.loc[its].values.astype(np.float64)
        s0, s1 = np.bincount(inv, weights=d0, minlength=len(U)), np.bincount(inv, weights=d1, minlength=len(U))
        res[nm] = ((d0.sum() - d1.sum()) / d0.sum(), (s0[bi].sum(1) - s1[bi].sum(1)) / s0[bi].sum(1), len(its))
    d = res["a"][1] - res["b"][1]
    return {"n_source_items": len(common), "n_units": int(len(U)), "n_items_a": res["a"][2], "n_items_b": res["b"][2],
            "F_a": r4(res["a"][0]), "F_b": r4(res["b"][0]), "diff": r4(res["a"][0] - res["b"][0]),
            "diff_lo": r4(np.percentile(d, 2.5)), "diff_hi": r4(np.percentile(d, 97.5))}


def table_md(df: pd.DataFrame, name: str, title: str, note: str = ""):
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, name + ".csv"), index=False)
    with open(os.path.join(TABLES, name + ".md"), "w") as f:
        f.write(f"**{title}**\n\n")
        f.write("| " + " | ".join(str(c) for c in df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for r in df.itertuples(index=False):
            f.write("| " + " | ".join("" if (isinstance(x, float) and not np.isfinite(x)) else str(x) for x in r) + " |\n")
        if note:
            f.write("\n" + note + "\n")


def fF(e, k="F"):
    if not e or e.get(k) is None:
        return "n/a"
    return f"{e[k]:+.3f} [{e[k + '_lo']:+.3f}, {e[k + '_hi']:+.3f}]"


def tables(S: dict, sfx: str):
    rows = []
    for proto, cd in S["models"].items():
        for cond, d in cd.items():
            tg = d["targets"]
            null = d["effects"].get("null", {}).get("full")
            for t in tg["all"]:
                e = d["effects"].get(f"tgt:{t}", {}).get("full")
                role = d["conditions"].get(f"tgt:{t}", {}).get("meta", {}).get("role", "")
                rows.append({"model": d["label"], "condition": cond, "knockout": t, "role": role,
                             "routed at final": d["routing_split"].get(t, {}).get("freq_final"),
                             "F [95% CI]": fF(e), "pair acc base -> ko": f"{d['baseline']['full']['pair_acc']:.3f} -> {e['pair_acc_ko']:.3f}" if e else "n/a",
                             "F routed items": d["routing_split"].get(t, {}).get("routed", {}).get("F"),
                             "null F": f"{null['F']:+.3f}" if null else "n/a"})
            for k in ("k3", "k5", "k10"):
                s = d["sets"].get(k)
                if s:
                    rows.append({"model": d["label"], "condition": cond, "knockout": f"top-{k[1:]}", "role": "pop",
                                 "routed at final": None,
                                 "F [95% CI]": f"{s['pop_full']:+.3f} [{s['pop_full_lo']:+.3f}, {s['pop_full_hi']:+.3f}]" if s.get("pop_full") is not None else "n/a",
                                 "pair acc base -> ko": f"{d['baseline']['full']['pair_acc']:.3f} -> {s['pair_acc_ko_full']:.3f}" if s.get("pair_acc_ko_full") is not None else "n/a",
                                 "F routed items": None, "null F": f"{null['F']:+.3f}" if null else "n/a"})
    table_md(pd.DataFrame(rows), f"ext13ko_targets{sfx}",
             "Phase 5 knockout (route mask at all positions, reroute), eval items (validation + replication pairs, both twins): "
             "fraction of the clean margin lost F [95% CI]; pair accuracy baseline -> knockout; null = baseline recomputed in other passes")
    rows = []
    for proto, cd in S["models"].items():
        for cond, d in cd.items():
            for t, v in d["same_layer"].items():
                rows.append({"model": d["label"], "condition": cond, "target": t, "target F": v["target"],
                             "controls mean": v["controls_mean"], "controls max": v["controls_max"], "max control": v["max_control"],
                             "rank": f"{v['rank']}/{v['n']}", "z": v["z"]})
    table_md(pd.DataFrame(rows), f"ext13ko_controls{sfx}", "Main targets vs same-layer controls (validation pairs; F); rank 1 = most damaging")
    rows = []
    for proto, cd in S["models"].items():
        for cond, d in cd.items():
            for k, v in d["sets"].items():
                rows.append({"model": d["label"], "condition": cond, "set": f"top-{k[1:]}", "members": " ".join(v["members"]),
                             "population set F": v["pop"], "random mean": v["rand_mean"], "random max": v["rand_max"],
                             "rank": f"{v['rank']}/6"})
    table_md(pd.DataFrame(rows), f"ext13ko_sets{sfx}", "Population top-k sets vs 5 frequency-matched random sets (validation pairs; F)")
    rows = []
    for proto, cd in S["models"].items():
        for cond, d in cd.items():
            for t, v in d["final_vs_all"].items():
                rows.append({"model": d["label"], "condition": cond, "knockout": t, "all positions": v["all"],
                             "final position only": v["final"], "final / all": v["final_over_all"]})
    table_md(pd.DataFrame(rows), f"ext13ko_final{sfx}", "All-position vs final-position-only route masks (reroute; validation pairs; F)")
    rows = []
    for proto, cd in S["models"].items():
        for cond, d in cd.items():
            for t, v in d["selfrepair"].items():
                rows.append({"model": d["label"], "condition": cond, "expert": t, "routed items": v["n_items"],
                             "direct write W": v["direct_write_W"], "net layer direct change N": v["net_layer_direct_N"],
                             "knockout K": v["knockout_K"],
                             "K / W [95% CI]": (f"{v['lost_K_over_W']:.2f} [{v['lost_K_over_W_ci'][0]:.2f}, {v['lost_K_over_W_ci'][1]:.2f}]"
                                                if v["lost_K_over_W"] is not None and abs(v["direct_write_W"]) >= MIN_W else "n/a (W < 0.1: acts indirectly)"),
                             "K / N [95% CI]": (f"{v['lost_K_over_N']:.2f} [{v['lost_K_over_N_ci'][0]:.2f}, {v['lost_K_over_N_ci'][1]:.2f}]"
                                                if v["lost_K_over_N"] is not None and abs(v["net_layer_direct_N"]) >= MIN_W else "n/a")})
    table_md(pd.DataFrame(rows), f"ext13ko_selfrepair{sfx}",
             "Knockout loss vs the expert's own direct write on the same clean prompts (eval items whose final position routes the "
             "expert; all-position reroute). W = DLA of c_e (baseline), N = net change of the layer's MoE direct write (baseline - "
             "knockout; includes the replacement expert), K = knockout loss of Delta (logits). 1 - K/W = compensation in the "
             "ext9 synthesis convention; K/N > 1 means the downstream network amplifies the loss.")
    rows = []
    for proto, cd in S["models"].items():
        for cond, d in cd.items():
            for t, v in d["routing_change"].items():
                rows.append({"model": d["label"], "condition": cond, "expert": t, "routed at final": d["routing_split"].get(t, {}).get("freq_final"),
                             "clean one-for-one swap": v["frac_clean_swap"],
                             "top replacements (share)": ", ".join(f"{a} ({b})" for a, b in v["top_replacements"])})
    table_md(pd.DataFrame(rows), f"ext13ko_routing{sfx}", "Which expert takes the slot (all-position reroute; final position, items routed to the expert)")
    rows = []
    for k, v in S.get("contrasts", {}).items():
        pv = v.get("paired")
        rows.append({"model": K.PROTO[v["proto"]]["label"], "contrast": v["label"], "expert(s)": v["expert"], "a": v["a"], "b": v["b"],
                     "F a": v["F_a"], "F b": v["F_b"], "a - b [95% CI]": f"{v['diff']:+.3f} [{v['diff_lo']:+.3f}, {v['diff_hi']:+.3f}]",
                     "paired: source items (units)": f"{pv['n_source_items']} ({pv['n_units']})" if pv else "",
                     "paired F a / F b": f"{pv['F_a']:+.3f} / {pv['F_b']:+.3f}" if pv else "",
                     "paired a - b [95% CI]": f"{pv['diff']:+.3f} [{pv['diff_lo']:+.3f}, {pv['diff_hi']:+.3f}]" if pv else ""})
    if rows:
        table_md(pd.DataFrame(rows), f"ext13ko_p58{sfx}",
                 "P5-8 and direction contrasts of knockout F (eval items, independent bootstrap per condition; different item sets "
                 "unless stated). wg = Phase 3's WinoGrande expert; selected_b = the second condition's own selected expert on both "
                 "conditions; own top-10 = each condition's own population top-10 set")
    rows = []
    for proto, cd in S["models"].items():
        for cond, d in cd.items():
            for st, v in d["strata"].items():
                for nm, x in v.items():
                    if nm == "n_items":
                        continue
                    rows.append({"model": d["label"], "condition": cond, "stratum": st, "items": v["n_items"], "knockout": nm,
                                 "F [95% CI]": fF(x)})
    if rows:
        table_md(pd.DataFrame(rows), f"ext13ko_strata{sfx}", "Knockout F by stratum (eval items)")
    rows = []
    for proto, cd in S["models"].items():
        for cond, d in cd.items():
            for ex, v in d["cross"].items():
                rows.append({"model": d["label"], "evaluated on": cond, "knockout": ex, "selected on": v["from"], "F [95% CI]": fF(v)})
    if rows:
        table_md(pd.DataFrame(rows), f"ext13ko_cross{sfx}", "Cross-condition knockout: other conditions' selected experts (validation pairs)")


def figures(S: dict, sfx: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGS, exist_ok=True)
    # per condition: controls + sets
    for proto, cd in S["models"].items():
        for cond, d in cd.items():
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            ax = axes[0]
            xt = []
            for i, (t, v) in enumerate(d["same_layer"].items()):
                ax.scatter([i] * len(v["controls"]), list(v["controls"].values()), s=10, color="#999999", zorder=2)
                ax.scatter([i], [v["target"]], s=60, color="#3b6fb6", marker="D", zorder=3)
                xt.append(t)
            for j, t in enumerate(d["targets"]["cf"]):
                e = d["effects"].get(f"tgt:{t}", {}).get("sub")
                if e:
                    ax.scatter([len(xt)], [e["F"]], s=40, color="#c8553d", marker="s", zorder=3)
                    ax.annotate(t, (len(xt), e["F"]), fontsize=6, xytext=(4, 0), textcoords="offset points")
            ax.set_xticks(list(range(len(xt))) + [len(xt)])
            ax.set_xticklabels(xt + ["CF experts"], fontsize=7)
            nl = d["effects"].get("null", {}).get("sub")
            if nl:
                ax.axhspan(min(0, nl["F_lo"]), max(0, nl["F_hi"]), color="#3b6fb6", alpha=0.08, lw=0)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel("fraction of clean margin lost F")
            ax.set_title("targets (diamonds) vs same-layer controls (grey); shaded: null 95% CI", fontsize=8)
            ax = axes[1]
            ks, pv, rv, rmax = [], [], [], []
            for k in K.SET_SIZES:
                s = d["sets"].get(f"k{k}")
                if s:
                    ks.append(k)
                    pv.append(s["pop"])
                    rv.append(s["rand_mean"])
                    rmax.append(s["rand_max"])
            ax.plot(ks, pv, "-o", color="#3b6fb6", label="population top-k")
            ax.plot(ks, rv, "-x", color="#999999", label="random matched (mean)")
            ax.plot(ks, rmax, ":", color="#999999", label="random matched (max)")
            ax.set_xscale("log")
            ax.set_xticks(K.SET_SIZES)
            ax.set_xticklabels([str(k) for k in K.SET_SIZES])
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("set size k")
            ax.legend(fontsize=7)
            ax.set_title("population sets vs frequency-matched random sets", fontsize=8)
            fig.suptitle(f"{d['label']} — {CLAB.get(cond, cond)}: expert knockout (validation pairs)", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(FIGS, f"ext13ko_{proto}_{cond}{sfx}.png"), dpi=130)
            plt.close(fig)
    # cross-condition overview
    ms = list(S["models"].items())
    if not ms:
        return
    fig, axes = plt.subplots(len(ms), 1, figsize=(12, 3.6 * len(ms)), squeeze=False)
    for ax, (proto, cd) in zip(axes[:, 0], ms):
        conds = [c for c in list(K.CONDITIONS) + ["f1pre"] if c in cd]
        wg = K.le(*K.PROTO[proto]["wg"])
        series = [("selected", "#3b6fb6"), ("wg", "#2e9e5b"), ("cf", "#c8553d"), ("top10", "#7a4fa3"), ("rand10", "#999999")]
        w = 0.16
        for si, (nm, col) in enumerate(series):
            ys, lo, hi = [], [], []
            for c in conds:
                d = cd[c]
                if nm == "selected":
                    e = d["effects"].get(f"tgt:{d['targets']['selected']}", {}).get("full") if d["targets"]["selected"] else None
                elif nm == "wg":
                    e = d["effects"].get(f"tgt:{wg}", {}).get("full")
                elif nm == "cf":
                    vals = [d["effects"].get(f"tgt:{t}", {}).get("full") for t in d["targets"]["cf"]]
                    vals = [v for v in vals if v]
                    e = max(vals, key=lambda v: v["F"]) if vals else None
                elif nm == "top10":
                    s = d["sets"].get("k10")
                    e = {"F": s["pop_full"], "F_lo": s["pop_full_lo"], "F_hi": s["pop_full_hi"]} if s and s.get("pop_full") is not None else None
                else:
                    s = d["sets"].get("k10")
                    e = {"F": s["rand_mean"], "F_lo": s["rand_min"], "F_hi": s["rand_max"]} if s else None
                ys.append(e["F"] if e else np.nan)
                lo.append(e["F"] - e["F_lo"] if e else 0)
                hi.append(e["F_hi"] - e["F"] if e else 0)
            x = np.arange(len(conds)) + (si - 2) * w
            ax.bar(x, ys, width=w, color=col, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.6},
                   label={"selected": "own selected expert", "wg": f"WinoGrande expert {wg}", "cf": "strongest CounterFact expert",
                          "top10": "own population top-10", "rand10": "matched random 10-sets (mean, min-max)"}[nm])
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([CLAB.get(c, c) for c in conds], rotation=25, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("F (fraction of clean margin lost)")
        ax.set_title(K.PROTO[proto]["label"], fontsize=9)
        ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"ext13ko_overview{sfx}.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protos", default="qwen3,mixtral_bos")
    ap.add_argument("--conds", default=",".join(K.CONDITIONS))
    ap.add_argument("--suffix", default="")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    sfx = args.suffix
    S = {"models": {}, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_boot": N_BOOT}
    objs = {}
    for proto in args.protos.split(","):
        for cond in args.conds.split(","):
            od = os.path.join(RESULTS, f"p5_{proto}_ko{sfx}", cond)
            if not os.path.exists(os.path.join(od, "plan_main.json")):
                continue
            meta = json.load(open(os.path.join(od, "run_meta.json"))) if os.path.exists(os.path.join(od, "run_meta.json")) else {}
            if not meta.get("phases", {}).get("main", {}).get("complete"):
                log(f"{cond}/{proto}: main phase incomplete, skipped")
                continue
            log(f"analysing {cond}/{proto}")
            C = Cond(cond, proto, sfx)
            objs.setdefault(proto, {})[cond] = C
            S["models"].setdefault(proto, {})[cond] = analyse(C)
    S["contrasts"] = contrasts(objs, S["models"])
    path = os.path.join(RESULTS, f"ext13ko_summary{sfx}.json")
    json.dump(S, open(path, "w"), indent=1)
    tables(S, sfx)
    if not args.no_figs:
        figures(S, sfx)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
