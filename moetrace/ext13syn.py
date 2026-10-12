"""ext13syn (Phase 5, agent p5-synth), Task A: the task ladder (RESEARCH_PLAN.md "P5-2 Task ladder", review issue 4).

One table across tasks and both models (Qwen3-30B-A3B-Base, Mixtral-8x7B with BOS; STR, final position), recomputed from
the run rows with ONE code path wherever the runs are pair runs (moetrace/ext7_pairs.PairRun schema):
  * attention share = AUC+(attention) / (AUC+(attention) + AUC+(MoE)) over the single-layer final-position curves
    (Direction-2b definition, as scripts/ext13_p5_analyze.cluster_stats);
  * direct attention share A_direct = Delta(h_corrupt + sum dAttn) - Delta_corrupt over the drop (exact final norm,
    direct_split.parquet);
  * all-MoE M = every final-position MoE output patched at once (denoising, joint_rows.parquet) over the drop;
  * heads among the first 10 picks of the mixed head + expert greedy (p5-circuit's circuit state; Phase-4 runs for the
    Phase-3 option swap), per row;
  * own selected expert and its knockout F, Phase 3's WinoGrande expert's knockout F (p5-expert's knockout table;
    Phase 4 ext9 for CounterFact / Phase-3 option swap / IOI).
CounterFact is a donor-level run (ext6 schema), so its row comes from the Phase-3 three-task table
(results/ext7_controls_summary.json); everything else is recomputed.

Statistics: cluster bootstrap over the condition's independent unit (5,000 resamples, seed 0): templates = seed template,
copy prefix on natural twins = WinoGrande twin (both attribute orders together), otherwise the pair. The unit is the first
'|' field of the pair id in every run (HIT id of the twin, or of the seed template). Paired contrasts restrict both
conditions to the items they share (natural twins: twin id; templates: seed x name pair, attribute orders pooled) and
apply the same unit resamples to both, so the CI of the difference is a paired cluster bootstrap.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from . import ext7_pairs as P
from .models import RESULTS

ROOT = "/home/ubuntu/MOE"
NB = 5000
PROTOS = ("qwen3", "mixtral_bos")
MODEL_OF = {"qwen3": "qwen3", "mixtral_bos": "mixtral"}
MLABEL = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral_bos": "Mixtral-8x7B (BOS)"}
MSHORT = {"qwen3": "Qwen3", "mixtral_bos": "Mixtral"}
COND_JSON = os.path.join(ROOT, "data/wino_ref/p5_conditions.json")
R_PROBES = {"qwen3": ("v1", "v2"), "mixtral_bos": ("v2",)}  # responding probes (ext13_base: pre-registered rule)
WG_EXPERT = {"qwen3": "L41E117", "mixtral_bos": "L20E000"}
CF_EXPERT = {"qwen3": "L44E069", "mixtral_bos": "L21E001"}  # Direction 6 STR selections (Mixtral BOS: L21E001)

# level metadata: the factors the ladder changes together (RESEARCH_PLAN.md Phase 5 review, issue 4)
LEVELS = {
    "cf": dict(label="CounterFact STR (subject swap)", src="json", direction="fact recall", fmt="plain cloze",
               answer="object noun (generated)", ctx="no", items="natural facts"),
    "p3_opt": dict(label="WinoGrande option swap (Phase 3)", run="wino_{proto}_str", direction="entity -> attribute",
                   fmt="sentence, blank filled", answer="attribute word (generated)", ctx="no", items="natural twins"),
    "dir_opt": dict(label="dir_opt: option swap on the F1-margin twins", run="p5_dir_opt_{proto}", direction="entity -> attribute",
                    fmt="sentence, blank filled", answer="attribute word (generated)", ctx="no", items="natural twins"),
    "vt_e2a_orig": dict(label="vt_e2a_orig: verb templates, orig verb", run="p5_vt_e2a_orig_{proto}", direction="entity -> attribute",
                        fmt="sentence, blank filled", answer="attribute word (generated)", ctx="no", items="templates x names"),
    "vt_e2a_neu": dict(label="vt_e2a_neu: verb templates, neutral verb", run="p5_vt_e2a_neu_{proto}", direction="entity -> attribute",
                       fmt="sentence, blank filled", answer="attribute word (generated)", ctx="no", items="templates x names"),
    "cp_vt_orig": dict(label="cp_vt_orig: + copy prefix, templates, orig verb", run="p5_cp_vt_orig_{proto}", direction="entity -> attribute",
                       fmt="'Options:' prefix + sentence", answer="attribute word (in prefix)", ctx="yes", items="templates x names"),
    "cp_vt_neu": dict(label="cp_vt_neu: + copy prefix, templates, neutral verb", run="p5_cp_vt_neu_{proto}", direction="entity -> attribute",
                      fmt="'Options:' prefix + sentence", answer="attribute word (in prefix)", ctx="yes", items="templates x names"),
    "cp_nat": dict(label="cp_nat: + copy prefix, natural twins", run="p5_cp_nat_{proto}", direction="entity -> attribute",
                   fmt="'Options:' prefix + sentence", answer="attribute word (in prefix)", ctx="yes", items="natural twins"),
    "dir_f1": dict(label="dir_f1: fill-in-the-blank on the same twins", run="p5_dir_f1_{proto}", direction="attribute -> entity",
                   fmt="'Fill in the blank' wrapper", answer="entity (mention)", ctx="yes", items="natural twins"),
    "f1": dict(label="f1: fill-in-the-blank, probe-neutral pool", run="p5_f1_{proto}", direction="attribute -> entity",
               fmt="'Fill in the blank' wrapper", answer="entity (mention)", ctx="yes", items="natural twins"),
    "vt_a2e_orig": dict(label="vt_a2e_orig: fill-in-the-blank on the templates", run="p5_vt_a2e_orig_{proto}", direction="attribute -> entity",
                        fmt="'Fill in the blank' wrapper", answer="name (mention)", ctx="yes", items="templates x names"),
    "f2e": dict(label="f2e: natural sentence, blank at the end", run="p5_f2e_{proto}", direction="attribute -> entity",
                fmt="natural truncation", answer="entity (mention)", ctx="yes", items="natural twins"),
    "ioi": dict(label="IOI (S2 -> IO)", run="ioi_{proto}_s2io", direction="indirect object", fmt="IOI template",
                answer="name (mention)", ctx="yes", items="IOI templates"),
}
P5_CONDS = ("dir_opt", "vt_e2a_orig", "vt_e2a_neu", "cp_vt_orig", "cp_vt_neu", "cp_nat", "dir_f1", "f1", "vt_a2e_orig", "f2e")
STRATA_ROWS = {"p3_opt": "names", "f1": "names", "dir_f1": "names", "dir_opt": "names", "cp_nat": "names"}

# paired contrasts on identical items (a - b)
CONTRASTS = [
    ("dir_f1", "dir_opt", "direction on identical twins (attribute -> entity minus entity -> attribute)", "direction"),
    ("dir_f1", "cp_nat", "direction with the answer in context on both sides (C3 - C2; wrapper differs)", "direction"),
    ("vt_a2e_orig", "vt_e2a_orig", "direction on identical templates (Qwen3; fill-in-the-blank minus option swap)", "direction"),
    ("cp_nat", "p3_opt", "answer copyable, natural twins (C2 - C1: prefix minus Phase-3 option swap)", "copy"),
    ("cp_nat", "dir_opt", "answer copyable, natural twins (C2 - C1 on the F1-margin twins)", "copy"),
    ("cp_vt_orig", "vt_e2a_orig", "answer copyable, templates, orig verb (C2 - C1)", "copy"),
    ("cp_vt_neu", "vt_e2a_neu", "answer copyable, templates, neutral verb (C2 - C1)", "copy"),
    ("vt_e2a_neu", "vt_e2a_orig", "neutral minus original verb, entity -> attribute", "verb"),
    ("cp_vt_neu", "cp_vt_orig", "neutral minus original verb, copy prefix", "verb"),
]


# ------------------------------------------------------------------------------------------------------------------
# registry helpers
# ------------------------------------------------------------------------------------------------------------------
def registry() -> dict:
    if not os.path.exists(COND_JSON):
        return {}
    return json.load(open(COND_JSON)).get("conditions", {})


def base_ready(cond: str, proto: str, reg: Optional[dict] = None) -> bool:
    reg = registry() if reg is None else reg
    e = reg.get(cond, {}).get(proto)
    if not e or e.get("behaviour_only"):
        return False
    if not (e.get("status") or {}).get("base"):
        return False
    od = os.path.join(RESULTS, f"p5_{cond}_{proto}")
    return all(os.path.exists(os.path.join(od, f)) for f in ("str_sweep_rows.parquet", "direct_split.parquet", "joint_rows.parquet"))


def run_of(level: str, proto: str) -> Optional[str]:
    m = LEVELS[level]
    return m["run"].format(proto=proto) if "run" in m else None


def pairs_file(level: str, proto: str) -> Optional[str]:
    if level in P5_CONDS:
        e = registry().get(level, {}).get(proto, {})
        p = e.get("pair_file")
        return os.path.join(ROOT, p) if p else None
    return None


# ------------------------------------------------------------------------------------------------------------------
# pair-level data
# ------------------------------------------------------------------------------------------------------------------
def item_key(pid: str) -> str:
    """Item identity across conditions: natural twins -> twin id; copy-prefix order suffix dropped; templates -> seed id |
    name_x | name_y (verb variant and attribute order dropped)."""
    f = str(pid).split("|")
    if len(f) >= 4:
        return f"{f[0]}|{f[2]}|{f[3]}"
    return f[0]


def unit_key(pid: str) -> str:
    return str(pid).split("|")[0]


class PairData:
    """Pair-level arrays of one pair run restricted to (family, splits): attention / MoE single-layer rescue curves
    (pair means over the two directed cases), drop, direct-path attention numerator / denominator, all-MoE joint
    numerator / denominator, pair ids, units, items, strata."""

    def __init__(self, run: str, family: str = "main", splits=("validation",), pairs_path: Optional[str] = None, proto: str = "qwen3"):
        pr = P.PairRun(run)
        self.run, self.proto, self.L = run, proto, pr.L
        ids = []
        for f in (family if isinstance(family, (list, tuple)) else (family,)):
            if f not in pr.fam:
                continue
            for s in splits:
                ids += [int(c) for c in pr.fam[f].get(s, [])]
        ids = sorted(set(ids))
        self.ids = ids
        c = pr.cases.loc[ids]
        pidx = np.array(sorted({i // 2 for i in ids}))
        self.pidx = pidx
        g = c.groupby(c.case_id // 2)
        self.pair_id = g.pair_id.first().reindex(pidx).astype(str).to_numpy()
        self.unit = np.array([unit_key(p) for p in self.pair_id])
        self.item = np.array([item_key(p) for p in self.pair_id])
        self.A = P.matrix_pairs(pr.mat("attn_layer", ids)).reindex(pidx).to_numpy(np.float64)
        self.M = P.matrix_pairs(pr.mat("layer", ids)).reindex(pidx).to_numpy(np.float64)
        self.drop = P.pair_means(pr.drop.loc[ids]).reindex(pidx).to_numpy(np.float64)
        self.T = g["T"].first().reindex(pidx).to_numpy(np.float64)
        self.names = g["names"].first().reindex(pidx).astype(bool).to_numpy() if "names" in c else np.zeros(len(pidx), bool)
        self.dnum = self.dden = self.jnum = self.jden = None
        dp = os.path.join(pr.dir, "direct_split.parquet")
        if os.path.exists(dp):
            x = pd.read_parquet(dp).drop_duplicates("case_id").set_index("case_id")
            if set(ids) <= set(x.index):
                x = x.loc[ids]
                self.dnum = P.pair_means(x.d_attn_direct - x.d_corrupt_fp32).reindex(pidx).to_numpy(np.float64)
                self.dden = P.pair_means(x.d_clean_fp32 - x.d_corrupt_fp32).reindex(pidx).to_numpy(np.float64)
        jp = os.path.join(pr.dir, "joint_rows.parquet")
        if os.path.exists(jp):
            j = pd.read_parquet(jp)
            j = j[(j.span == "all") & (j.direction == "denoise") & (j.kind == "all_moe")].drop_duplicates("case_id").set_index("case_id")
            if set(ids) <= set(j.index):
                self.jnum = P.pair_means(j.effect.loc[ids]).reindex(pidx).to_numpy(np.float64)
                self.jden = P.pair_means(j["drop"].loc[ids]).reindex(pidx).to_numpy(np.float64)
        self.heads10 = None
        # strata / probe scores
        self.stratum = np.array(["n/a"] * len(pidx), dtype=object)
        self.local_assoc = None
        self.score = np.full(len(pidx), np.nan)
        if pairs_path and os.path.exists(pairs_path):
            pf = pd.read_parquet(pairs_path)
            key = "pair_idx" if "pair_idx" in pf.columns and not run.startswith("wino_ref_F1") else "pair_id"
            pf = pf.drop_duplicates(key).set_index(key)
            look = pidx if key == "pair_idx" else self.pair_id
            sub = pf.reindex(look)
            if "stratum_assoc" in sub:
                self.stratum = sub.stratum_assoc.fillna("no_probe").astype(str).to_numpy()
            if "stratum_local_assoc" in sub:
                self.local_assoc = sub.stratum_local_assoc.astype("boolean").fillna(False).astype(bool).to_numpy()
            R = R_PROBES[proto]
            cols = [(f"probe_{v}_a", f"probe_{v}_b") for v in R]
            if all(a in sub.columns and b in sub.columns for a, b in cols):
                sc = np.nanmean(np.stack([(sub[a].to_numpy(float) - sub[b].to_numpy(float)) / 2.0 for a, b in cols]), axis=0)
                self.score = sc

    @property
    def n(self) -> int:
        return len(self.pidx)

    def set_local_assoc_from_scan(self, scan_run: str):
        """Phase-3 local-context association (results/wino_<proto>/scan_pairs.parquet): flag + continuous own-answer score."""
        p = os.path.join(RESULTS, scan_run, "scan_pairs.parquet")
        if not os.path.exists(p):
            return
        s = pd.read_parquet(p).drop_duplicates("pair_id").set_index("pair_id").reindex(self.pair_id)
        self.local_assoc = s.assoc.astype("boolean").fillna(False).astype(bool).to_numpy()
        self.score = ((s.d_local_a - s.d_local_b) / 2.0).to_numpy(float)
        self.stratum = np.where(self.local_assoc, "local assoc", "not local assoc").astype(object)

    def attach_heads(self, rows: Optional[pd.DataFrame]):
        """rows: per directed row heads10 (ext13ci_analyze.row_table) -> pair means (NaN where absent)."""
        if rows is None or rows.empty:
            return
        h = rows.groupby(rows.case_id.astype(int) // 2).heads10.mean()
        self.heads10 = h.reindex(self.pidx).to_numpy(np.float64)


# ------------------------------------------------------------------------------------------------------------------
# bootstrap
# ------------------------------------------------------------------------------------------------------------------
def unit_weights(units, nb: int = NB, seed: int = 0):
    """Unit-cluster bootstrap: (unique units, W_unit [nb + 1, n_units]); row 0 = point estimate (all ones)."""
    u = np.unique(np.asarray(units))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(nb, len(u)))
    cnt = np.zeros((nb, len(u)))
    np.add.at(cnt, (np.repeat(np.arange(nb), len(u)), idx.ravel()), 1.0)
    return u, np.vstack([np.ones(len(u)), cnt])


def pair_weights(d: PairData, u: np.ndarray, Wu: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    pos = {x: i for i, x in enumerate(u)}
    cols = np.array([pos.get(x, -1) for x in d.unit])
    W = np.where(cols[None, :] >= 0, Wu[:, np.clip(cols, 0, None)], 0.0)
    if mask is not None:
        W = W * mask[None, :].astype(float)
    return W


def stats(d: PairData, W: np.ndarray) -> dict:
    """Weighted pair statistics, one value per bootstrap row (row 0 = point estimate)."""
    ws = W.sum(1)
    ok = ws > 0
    ws = np.where(ok, ws, np.nan)
    a = np.clip((W @ d.A) / ws[:, None], 0, None).sum(1)
    m = np.clip((W @ d.M) / ws[:, None], 0, None).sum(1)
    dr = (W @ d.drop) / ws
    out = {"share": a / (a + m), "auc_attn": a / dr, "auc_moe": m / dr, "drop": dr}
    if d.dnum is not None:
        out["A_direct"] = (W @ d.dnum) / (W @ d.dden)
    if d.jnum is not None:
        out["M_all"] = (W @ d.jnum) / (W @ d.jden)
    if d.heads10 is not None:
        hm = ~np.isnan(d.heads10)
        if hm.any():
            Wh = W * hm[None, :]
            out["heads10"] = (Wh @ np.nan_to_num(d.heads10)) / np.where(Wh.sum(1) > 0, Wh.sum(1), np.nan)
    return out


def ci(x) -> list:
    x = np.asarray(x, dtype=float)
    b = x[1:][np.isfinite(x[1:])]
    if not np.isfinite(x[0]) or len(b) < 10:
        return [float(x[0]), float("nan"), float("nan")]
    return [float(x[0]), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def summarise(d: PairData, mask: Optional[np.ndarray] = None, nb: int = NB, seed: int = 0) -> dict:
    units = d.unit if mask is None else d.unit[mask]
    u, Wu = unit_weights(units, nb, seed)
    W = pair_weights(d, u, Wu, mask)
    s = stats(d, W)
    sel = np.ones(d.n, bool) if mask is None else mask
    out = {k: ci(v) for k, v in s.items()}
    out.update(n_pairs=int(sel.sum()), n_units=int(len(u)), T_median=float(np.median(d.T[sel])) if sel.any() else float("nan"),
               names_frac=float(d.names[sel].mean()) if sel.any() else float("nan"))
    if d.heads10 is not None:
        out["n_pairs_heads"] = int((~np.isnan(d.heads10[sel])).sum())
    return out


def paired(da: PairData, db: PairData, nb: int = NB, seed: int = 0) -> Optional[dict]:
    """Paired contrast a - b on the shared items; same unit resamples for both conditions."""
    common = sorted(set(da.item) & set(db.item))
    if not common:
        return None
    cs = set(common)
    ma = np.array([x in cs for x in da.item])
    mb = np.array([x in cs for x in db.item])
    units = sorted(set(da.unit[ma]) | set(db.unit[mb]))
    u, Wu = unit_weights(np.array(units), nb, seed)
    sa = stats(da, pair_weights(da, u, Wu, ma))
    sb = stats(db, pair_weights(db, u, Wu, mb))
    out = {"n_items": len(common), "n_units": len(u), "n_pairs_a": int(ma.sum()), "n_pairs_b": int(mb.sum())}
    for k in ("share", "auc_attn", "A_direct", "M_all", "heads10", "drop"):
        if k in sa and k in sb:
            out[k] = {"a": ci(sa[k]), "b": ci(sb[k]), "diff": ci(sa[k] - sb[k])}
    return out


def strata_compare(d: PairData, labels: np.ndarray, ref: str, nb: int = NB, seed: int = 0, min_pairs: int = 8) -> dict:
    """Per stratum label: share / A_direct / M with CI, and the difference to the reference stratum, all on the same unit
    resamples (strata are subsets of one run's pairs)."""
    u, Wu = unit_weights(d.unit, nb, seed)
    Wall = pair_weights(d, u, Wu)
    res = {}
    st = {}
    for lab in sorted(set(labels)):
        m = labels == lab
        if m.sum() < min_pairs:
            continue
        st[lab] = stats(d, Wall * m[None, :])
        res[lab] = {k: ci(v) for k, v in st[lab].items()}
        res[lab]["n_pairs"] = int(m.sum())
        res[lab]["n_units"] = int(len(set(d.unit[m])))
    if ref in st:
        for lab in st:
            if lab == ref:
                continue
            res[lab]["diff_vs_ref"] = {k: ci(st[lab][k] - st[ref][k]) for k in ("share", "A_direct", "M_all") if k in st[lab]}
    return res


def probe_slope(d: PairData, nb: int = NB, seed: int = 0) -> Optional[dict]:
    """Per-pair direct attention share y = A_direct / drop regressed on the own-answer probe score s (signed: the trigger
    alone points to the clean answer) and on |s| (association strength); weighted least squares on the unit resamples."""
    if d.dnum is None:
        return None
    x = d.score
    ok = np.isfinite(x) & (np.abs(d.dden) > 1e-6)
    if ok.sum() < 20:
        return None
    y = d.dnum[ok] / d.dden[ok]
    xs, units = x[ok], d.unit[ok]
    u, Wu = unit_weights(units, nb, seed)
    pos = {k: i for i, k in enumerate(u)}
    W = Wu[:, [pos[k] for k in units]]
    out = {"n_pairs": int(ok.sum()), "n_units": int(len(u)), "score_sd": float(np.std(xs)), "abs_score_mean": float(np.mean(np.abs(xs))),
           "frac_abs_ge1": float(np.mean(np.abs(xs) >= 1))}
    ws = W.sum(1)
    for name, xx in (("signed", xs), ("abs", np.abs(xs))):
        mx, my = (W @ xx) / ws, (W @ y) / ws
        sxy = (W @ (xx * y)) / ws - mx * my
        sxx = (W @ (xx * xx)) / ws - mx * mx
        out[f"slope_{name}"] = ci(sxy / sxx)
        syy = (W @ (y * y)) / ws - my * my
        out[f"r_{name}"] = ci(sxy / np.sqrt(sxx * syy))
    return out


# ------------------------------------------------------------------------------------------------------------------
# reference numbers that are not pair runs (CounterFact) and knockout tables
# ------------------------------------------------------------------------------------------------------------------
def _js(x):
    return json.loads(x) if isinstance(x, str) else x


def cf_reference(proto: str) -> Optional[dict]:
    p = os.path.join(RESULTS, "ext7_controls_summary.json")
    if not os.path.exists(p):
        return None
    tt = json.load(open(p)).get("three_task", [])
    lab = MLABEL[proto].split(" (")[0]
    for e in tt:
        if e["task"].startswith("CounterFact") and e["model"].startswith(lab.split("-")[0]):
            dr = _js(e["direct"])
            ad = dr.get("a_dir") or {}
            m = _js(e["w4_M"])
            return {"share": [e["attn_share"], e["attn_share_lo"], e["attn_share_hi"]],
                    "A_direct": [ad.get("ratio"), ad.get("lo"), ad.get("hi")],
                    "M_all": [m["ratio"], m["lo"], m["hi"]], "drop": [e["drop"], np.nan, np.nan],
                    "n_pairs": int(str(e["n_val"]).split()[0]), "n_units": int(str(e["n_val"]).split()[0]), "unit": "case (donor mean)",
                    "auc_attn": [np.nan] * 3, "T_median": np.nan, "names_frac": 0.0, "source": "ext7_controls three-task table"}
    return None


def ci_reference(key: str) -> Optional[float]:
    p = os.path.join(RESULTS, "ext13ci_summary.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p)).get("reference", {}).get(key)
    return float(r["heads_first10"]) if r else None


def ko_table() -> pd.DataFrame:
    p = os.path.join(RESULTS, "tables", "ext13ko_targets.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


def _parse_F(s) -> list:
    try:
        a, rest = str(s).split(" [")
        lo, hi = rest.rstrip("]").split(", ")
        return [float(a), float(lo), float(hi)]
    except Exception:
        return [float("nan")] * 3


def ko_lookup(kt: pd.DataFrame, cond: str, proto: str) -> dict:
    out = {}
    if kt.empty:
        return out
    ml = MLABEL[proto]
    sub = kt[(kt.model == ml) & (kt.condition == cond)]
    for _, r in sub.iterrows():
        role = str(r.role)
        if "selected" in role:
            out["own"] = {"expert": r.knockout, "F": _parse_F(r["F [95% CI]"])}
        if "wg" in role:
            out["wg"] = {"expert": r.knockout, "F": _parse_F(r["F [95% CI]"])}
    return out


def ko_phase4(proto: str, task: str, expert: str) -> Optional[list]:
    p = os.path.join(RESULTS, "ext9_knockout_summary.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))["models"].get(MODEL_OF[proto], {})
    e = m.get("effects", {}).get(f"tgt:{expert}", {}).get(task)
    if not e or "F" not in e:
        return None
    return [e["F"], e["F_lo"], e["F_hi"]]
