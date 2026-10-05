"""ext7: generic symmetric-token-replacement (STR) PAIR datasets for the final-position / position x layer / head /
expert runners (Phase 3; WinoGrande twins, entity-role swaps, IOI).

Contract (RESEARCH_PLAN Phase 3, "pair-file contract").
  pair parquet   one row per pair: pair_id, ids_a, ids_b (JSON token lists of equal length; tokenizer defaults of the
                 model), str_pos (JSON list of the positions where A and B differ; WinoGrande files carry opt_pos instead,
                 used as fallback), trig_a, trig_b (single continuation token ids: the answer after A / after B), optional
                 strata columns (copied into sweep_cases.parquet).
  case-set JSON  format of data/wino_str/case_sets.json: "pair_idx_of" {pair_id: pair_idx}, "directed" {split: [directed
                 ids]}, optional "own_pool_directed" {proto: {split: [...]}}.

Directed cases. id = 2 * pair_idx + d.
    d = 0: clean = A, corrupted = B, true = trig_a, foil = trig_b
    d = 1: clean = B, corrupted = A, true = trig_b, foil = trig_a
Delta = logit(true) - logit(foil) = LD(r, r') (Zhang & Nanda 2024). The corrupted run is a plain prefill row; every spawn
uses parent = corrupted row and clean = clean row (the ext6 STR protocol with one donor). Positions before min(str_pos)
are identical in A and B, so patches there are exactly zero.

Statistics. Per-pair value = mean over the two directions (primary); bootstrap CIs resample PAIRS (5,000 resamples,
seed 0, as moetrace.stats). Expert recurrence / Spec are defined on directed cases (the clean run differs between the
two directions); the recurrence gate is half the discovery directed cases.

Case-set families (resolved per run and written to <run>/case_sets.json in the analysis.ModelData format
{family: {"discovery": [...], "validation": [...]}}): split keys "discovery"/"validation" -> family "main"; keys
"rep_<split>" -> family "rep"; own_pool_directed[<proto>] -> family "own"; any other key k -> family k (split "all").
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .models import RESULTS
from .stats import N_BOOT, SEED, signflip_p

STRATA_COLS = ("names", "trigger_in_context", "debiased", "n_opt_tokens", "n_opt_diff_tokens", "n_tokens", "word_a", "word_b",
               "ans_a", "ans_b", "prompt_a", "prompt_b")


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class DCase:
    case_id: int  # directed id = 2 * pair_idx + d
    pair_idx: int
    d: int
    pair_id: str
    clean_ids: list
    corrupt_ids: list
    true_id: int
    foil_id: int
    str_pos: list
    T: int

    @property
    def first_str(self) -> int:
        return int(min(self.str_pos))


def _jl(x):
    return json.loads(x) if isinstance(x, str) else [int(v) for v in x]


def load_pairs(pairs_path: str, case_sets: dict) -> pd.DataFrame:
    """Pair table indexed by pair_idx with parsed ids_a_l / ids_b_l / str_pos_l (only pairs listed in pair_idx_of)."""
    p = pd.read_parquet(pairs_path)
    pio = case_sets["pair_idx_of"]
    p = p[p.pair_id.isin(pio)].copy()
    p["pair_idx"] = p.pair_id.map(pio).astype(int)
    p["ids_a_l"] = p.ids_a.map(_jl)
    p["ids_b_l"] = p.ids_b.map(_jl)
    col = "str_pos" if "str_pos" in p.columns else "opt_pos"
    p["str_pos_l"] = p[col].map(_jl)
    bad = p[p.ids_a_l.map(len) != p.ids_b_l.map(len)]
    assert bad.empty, f"pairs with unequal length: {bad.pair_id.tolist()[:5]}"
    for r in p.itertuples():  # contract check: A and B differ exactly at str_pos (subset allowed: identical option tokens)
        diff = [i for i, (x, y) in enumerate(zip(r.ids_a_l, r.ids_b_l)) if x != y]
        assert set(diff) <= set(r.str_pos_l) and diff, (r.pair_id, diff, r.str_pos_l)
        assert int(r.trig_a) != int(r.trig_b), r.pair_id
    return p.set_index("pair_idx", drop=False).sort_index()


def resolve_sets(case_sets: dict, splits: Optional[list[str]] = None, own_pool: Optional[str] = None) -> dict:
    """{family: {split: [directed ids]}} from the case-set JSON (see module docstring)."""
    out: dict = {}
    for k, v in case_sets.get("directed", {}).items():
        if k == "shared_unused" or (splits and k not in splits):
            continue
        if k in ("discovery", "validation"):
            out.setdefault("main", {})[k] = [int(c) for c in v]
        elif k.startswith("rep_"):
            out.setdefault("rep", {})[k[4:]] = [int(c) for c in v]
        else:
            out.setdefault(k, {})["all"] = [int(c) for c in v]
    if own_pool:
        own = case_sets.get("own_pool_directed", {}).get(own_pool)
        assert own is not None, f"no own_pool_directed[{own_pool}] in the case-set JSON"
        out["own"] = {k: [int(c) for c in v] for k, v in own.items()}
    return out


def all_ids(fam: dict) -> list[int]:
    seen, out = set(), []
    for f in fam.values():
        for v in f.values():
            for c in v:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
    return out


def directed_cases(pairs: pd.DataFrame, ids: list[int]) -> dict[int, DCase]:
    out = {}
    for c in ids:
        pi, d = divmod(int(c), 2)
        assert pi in pairs.index, f"pair_idx {pi} (directed id {c}) not in the pair parquet"
        r = pairs.loc[pi]
        a, b = list(r.ids_a_l), list(r.ids_b_l)
        ta, tb = int(r.trig_a), int(r.trig_b)
        if d == 0:
            out[c] = DCase(c, pi, 0, r.pair_id, a, b, ta, tb, list(r.str_pos_l), len(a))
        else:
            out[c] = DCase(c, pi, 1, r.pair_id, b, a, tb, ta, list(r.str_pos_l), len(a))
    return out


def split_of(fam: dict) -> dict[int, str]:
    """directed id -> 'family:split' of its first occurrence."""
    out = {}
    for f, sp in fam.items():
        for s, v in sp.items():
            for c in v:
                out.setdefault(c, f"{f}:{s}")
    return out


def run_dir(run: str) -> str:
    d = run if os.path.isabs(run) else os.path.join(RESULTS, run)
    os.makedirs(d, exist_ok=True)
    return d


def write_meta(run: str, key: str, upd: dict):
    p = os.path.join(run_dir(run), "run_meta.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.setdefault(key, {}).update(upd)
    with open(p, "w") as f:
        json.dump(cur, f, indent=1, default=str)


def setup_run(args, stage: str) -> tuple[dict, pd.DataFrame, dict, list[int], dict]:
    """Common CLI handling of the runners: resolves the case sets, writes/validates <run>/case_sets.json.
    Returns (case_sets_json, pairs, families, ids, directed cases)."""
    cs = json.load(open(args.case_sets))
    fam = resolve_sets(cs, args.splits.split(",") if args.splits else None, args.own_pool)
    od = run_dir(args.out)
    p = os.path.join(od, "case_sets.json")
    if os.path.exists(p):  # later stages reuse the families of the first stage unless they add new ones
        old = json.load(open(p))
        for f, sp in fam.items():
            if f in old and old[f] != sp:
                raise SystemExit(f"case-set family {f} differs from {p}; use a new --out")
        old.update(fam)
        fam_all = {k: v for k, v in old.items() if not k.startswith("_")}
        old_meta = {k: v for k, v in old.items() if k.startswith("_")}
    else:
        fam_all, old_meta = fam, {}
    with open(p, "w") as f:
        json.dump({**fam_all, "_source": {"case_sets": os.path.abspath(args.case_sets), "pairs": os.path.abspath(args.pairs),
                                           **old_meta.get("_source", {})}}, f, indent=1)
    pairs = load_pairs(args.pairs, cs)
    ids = all_ids(fam)
    dc = directed_cases(pairs, ids)
    return cs, pairs, fam, ids, dc


def add_common_args(ap):
    ap.add_argument("--model", required=True, help="model key (qwen3 | mixtral | olmoe | ...)")
    ap.add_argument("--pairs", required=True, help="pair parquet (ids_a, ids_b, str_pos|opt_pos, trig_a, trig_b)")
    ap.add_argument("--case-sets", required=True, help="case-set JSON (format of data/wino_str/case_sets.json)")
    ap.add_argument("--out", required=True, help="run dir under results/")
    ap.add_argument("--splits", default=None, help="comma list of keys of 'directed' to use (default: all but shared_unused)")
    ap.add_argument("--own-pool", default=None, help="also run own_pool_directed[<proto>] as family 'own'")
    ap.add_argument("--agent", default="ext7-wino")
    ap.add_argument("--dry-run", action="store_true")


def load_families(run: str) -> dict:
    sets = json.load(open(os.path.join(run_dir(run), "case_sets.json")))
    return {k: v for k, v in sets.items() if not k.startswith("_")}


def case_table(dc: dict[int, DCase], pairs: pd.DataFrame, fam: dict) -> pd.DataFrame:
    """Per directed case: geometry, ids, split, strata copied from the pair parquet."""
    so = split_of(fam)
    rows = []
    for c, x in dc.items():
        r = pairs.loc[x.pair_idx]
        row = dict(case_id=c, pair_idx=x.pair_idx, d=x.d, pair_id=x.pair_id, T=x.T, str_pos=json.dumps(x.str_pos),
                   first_str=x.first_str, last_str=int(max(x.str_pos)), n_str=len(x.str_pos), true_id=x.true_id, foil_id=x.foil_id,
                   split=so.get(c, ""))
        for k in STRATA_COLS:
            if k in pairs.columns:
                row[k] = r[k]
        for k in pairs.columns:
            if k.startswith("stratum_"):
                row[k] = r[k]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# position groups (Meng et al. / Zhang & Nanda token groups, generalised to non-contiguous STR sites)
# ---------------------------------------------------------------------------------------------------------------
POS_CATS = ("first STR token", "middle STR tokens", "between STR tokens", "last STR token", "first subsequent token",
            "further tokens", "last token")


def position_category(p: int, str_pos: list[int], T: int) -> str:
    """Priority: final position > last STR token > first STR token > other STR tokens > non-STR positions between STR
    tokens > first subsequent token > further tokens. A one-token STR site counts as 'last STR token'."""
    first, last = min(str_pos), max(str_pos)
    if p == T - 1:
        return "last token"
    if p == last:
        return "last STR token"
    if p == first:
        return "first STR token"
    if p in str_pos:
        return "middle STR tokens"
    if first < p < last:
        return "between STR tokens"
    if p == last + 1:
        return "first subsequent token"
    return "further tokens" if p > last else "prefix"


# ---------------------------------------------------------------------------------------------------------------
# pair-level statistics
# ---------------------------------------------------------------------------------------------------------------
def pair_means(x: pd.Series) -> pd.Series:
    """Directed-case series -> pair series (mean over the directions present; pairs indexed by pair_idx)."""
    x = pd.Series(x, dtype=float)
    return x.groupby(np.asarray(x.index) // 2).mean()


def _boot_idx(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def summ(x: pd.Series, with_p: bool = True) -> dict:
    """moetrace.stats.summarize on pair means (bootstrap over pairs); n = pairs; pos_frac on pairs; dir_pos_frac on
    directed cases."""
    x = pd.Series(x, dtype=float).dropna()
    pm = pair_means(x).to_numpy()
    if len(pm) == 0:
        return {"n": 0, "n_dir": 0, "mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "pos_frac": float("nan"), "p": float("nan")}
    idx = _boot_idx(len(pm))
    bm = pm[idx].mean(1)
    out = {"n": int(len(pm)), "n_dir": int(len(x)), "mean": float(pm.mean()), "ci_lo": float(np.percentile(bm, 2.5)),
           "ci_hi": float(np.percentile(bm, 97.5)), "pos_frac": float((pm > 0).mean()), "dir_pos_frac": float((x.to_numpy() > 0).mean()),
           "zero_frac": float((x.to_numpy() == 0).mean())}
    if with_p:
        out["p"] = signflip_p(pm)
    return out


def ratio(num: pd.Series, den: pd.Series) -> tuple[float, float, float]:
    """Ratio of means with a paired PAIR-bootstrap CI (num, den: directed-case series on the same ids)."""
    n_, d_ = pair_means(num), pair_means(den)
    d_ = d_.loc[n_.index]
    a, b = n_.to_numpy(), d_.to_numpy()
    idx = _boot_idx(len(a))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def matrix_pairs(M: pd.DataFrame) -> pd.DataFrame:
    """case x column matrix -> pair x column matrix (mean over directions)."""
    return M.groupby(np.asarray(M.index) // 2).mean()


def curve_ci(M: pd.DataFrame) -> pd.DataFrame:
    """Per column of a directed case x layer matrix: pair-mean mean, pair-bootstrap CI, positive fraction (pairs)."""
    P = matrix_pairs(M)
    A = P.to_numpy(dtype=np.float64)
    idx = _boot_idx(A.shape[0])
    bm = A[idx].mean(1)  # [n_boot, L]
    return pd.DataFrame({"layer": list(P.columns), "mean": A.mean(0), "ci_lo": np.percentile(bm, 2.5, axis=0),
                         "ci_hi": np.percentile(bm, 97.5, axis=0), "pos_frac": (A > 0).mean(0), "n": A.shape[0]})


def fmt(s: dict, d: int = 3) -> str:
    if not s or s.get("n", 0) == 0:
        return "n/a"
    return f"{s['mean']:+.{d}f} [{s['ci_lo']:+.{d}f}, {s['ci_hi']:+.{d}f}]"


def fmt_r(r: tuple, d: int = 3) -> str:
    return f"{r[0]:.{d}f} [{r[1]:.{d}f}, {r[2]:.{d}f}]"


# ---------------------------------------------------------------------------------------------------------------
# analysis.ModelData adapter (so that moetrace.analysis runs unchanged on a pair run)
# ---------------------------------------------------------------------------------------------------------------
def model_data(run: str):
    """analysis.ModelData for a pair run: sets = case-set families, sweep rows of kind 'layer' (final-position MoE
    patch), cases (delta_noised = delta_corrupt alias), all-layer expert rows (kind expert_corrupt_only dropped)."""
    from .analysis import ModelData
    od = run_dir(run)
    sets = load_families(run)
    cases = pd.read_parquet(os.path.join(od, "sweep_cases.parquet"))
    sw = pd.read_parquet(os.path.join(od, "str_sweep_rows.parquet"))
    lay = sw[sw.kind == "layer"][["case_id", "layer", "rescue", "delta", "delta_corrupt"]].copy()
    lay["kind"] = "layer"
    er = None
    p = os.path.join(od, "str_expert_rows.parquet")
    if os.path.exists(p):
        e = pd.read_parquet(p)
        er = e[e.kind != "expert_corrupt_only"].copy()
        er["clean_active"] = er.clean_active.astype(bool)
    return ModelData(run, sets, lay, None, cases, er, None)


# ---------------------------------------------------------------------------------------------------------------
# post-processing of a pair run (generic; WinoGrande-specific strata live in scripts/ext7_wino_analyze.py)
# ---------------------------------------------------------------------------------------------------------------
class PairRun:
    """Final-position sweep of a pair run: per kind a directed case x layer rescue matrix, the case table, families."""

    def __init__(self, run: str):
        self.run = run
        self.dir = run_dir(run)
        self.fam = load_families(run)
        self.cases = pd.read_parquet(os.path.join(self.dir, "sweep_cases.parquet")).set_index("case_id", drop=False)
        sw = pd.read_parquet(os.path.join(self.dir, "str_sweep_rows.parquet"))
        self.sw = sw
        self.kinds = [k for k in sw.kind.unique() if k not in ("clean", "corrupt")]
        self.R = {k: sw[sw.kind == k].pivot(index="case_id", columns="layer", values="rescue") for k in self.kinds}
        self.V = {k: sw[sw.kind == k].pivot(index="case_id", columns="layer", values="vnorm") for k in self.kinds}
        self.layers = [int(l) for l in self.R[self.kinds[0]].columns]
        self.L = len(self.layers)
        self.drop = self.cases["drop"]
        meta_p = os.path.join(self.dir, "run_meta.json")
        self.meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}

    def ids(self, family: str, split: str) -> list[int]:
        return list(self.fam[family][split])

    def mat(self, kind: str, ids: list[int]) -> pd.DataFrame:
        return self.R[kind].loc[ids]


def norm_curve(M: pd.DataFrame, drop: pd.Series) -> pd.DataFrame:
    """Per layer: mean rescue / mean drop (Zhang & Nanda's normalised logit difference), pair-bootstrap CI."""
    P_ = matrix_pairs(M).to_numpy(dtype=np.float64)
    d = pair_means(drop.loc[M.index]).to_numpy()
    idx = _boot_idx(P_.shape[0])
    r = P_[idx].mean(1) / d[idx].mean(1)[:, None]
    return pd.DataFrame({"layer": list(M.columns), "norm": P_.mean(0) / d.mean(), "norm_lo": np.percentile(r, 2.5, axis=0),
                         "norm_hi": np.percentile(r, 97.5, axis=0)})


def sweep_curves(run: PairRun, ids: list[int], kinds=None) -> pd.DataFrame:
    kinds = kinds or run.kinds
    out = []
    for k in kinds:
        M = run.mat(k, ids)
        c = curve_ci(M).merge(norm_curve(M, run.drop), on="layer")
        c.insert(0, "kind", k)
        out.append(c)
    return pd.concat(out, ignore_index=True)


def sweep_peaks(run: PairRun, family: str = "main", kinds=None, max_layer: Optional[int] = None) -> list[dict]:
    """Per kind: discovery argmax L_disc (optionally restricted to layers <= max_layer: interior-layer rule), its
    validation rescue (pair-level summary) and normalised value, validation argmax and maximum."""
    kinds = kinds or run.kinds
    disc, val = run.ids(family, "discovery"), run.ids(family, "validation")
    out = []
    for k in kinds:
        Md, Mv = run.mat(k, disc), run.mat(k, val)
        md, mv = Md.mean(0), Mv.mean(0)
        if max_layer is not None:
            md, mv = md[md.index <= max_layer], mv[mv.index <= max_layer]
        ld, lv = int(md.idxmax()), int(mv.idxmax())
        top = md.sort_values(ascending=False)
        out.append({"kind": k, "family": family, "max_layer": max_layer, "L_disc": ld, "disc_mean": float(md[ld]),
                    "disc_gap_to_2nd": float(top.iloc[0] - top.iloc[1]), "disc_2nd": int(top.index[1]),
                    "val_at_L_disc": summ(Mv[ld]), "norm_at_L_disc": ratio(Mv[ld], run.drop.loc[val]),
                    "L_val": lv, "val_max": summ(Mv[lv], with_p=False), "norm_at_L_val": ratio(Mv[lv], run.drop.loc[val]),
                    "disc_top5": [(int(l), round(float(v), 3)) for l, v in top.head(5).items()],
                    "n_disc_pairs": len(disc) // 2, "n_val_pairs": len(val) // 2})
    return out


def attention_share(run: PairRun, ids: list[int], attn: str = "attn_layer", moe: str = "layer") -> dict:
    """Direction-2b definition: AUC+(attention) / (AUC+(attention) + AUC+(MoE)) over the mean layer curves (sum over layers
    of the positive part of the mean rescue), pair-bootstrap CI; plus the AUC+ of block and of the drop-normalised curves."""
    A, M = matrix_pairs(run.mat(attn, ids)).to_numpy(dtype=np.float64), matrix_pairs(run.mat(moe, ids)).to_numpy(dtype=np.float64)
    idx = _boot_idx(A.shape[0])
    aa = np.clip(A[idx].mean(1), 0, None).sum(1)
    mm = np.clip(M[idx].mean(1), 0, None).sum(1)
    r = aa / (aa + mm)
    a0, m0 = float(np.clip(A.mean(0), 0, None).sum()), float(np.clip(M.mean(0), 0, None).sum())
    d = pair_means(run.drop.loc[ids]).mean()
    out = {"auc_attn": a0, "auc_moe": m0, "auc_attn_norm": a0 / d, "auc_moe_norm": m0 / d, "share": a0 / (a0 + m0),
           "share_lo": float(np.percentile(r, 2.5)), "share_hi": float(np.percentile(r, 97.5)), "n_pairs": int(A.shape[0])}
    if "block" in run.kinds:
        B = matrix_pairs(run.mat("block", ids)).to_numpy(dtype=np.float64)
        out["auc_block"] = float(np.clip(B.mean(0), 0, None).sum())
        out["auc_block_norm"] = out["auc_block"] / d
    return out


def share_at_layer(run: PairRun, layer: int, ids: list[int]) -> dict:
    a = run.mat("attn_layer", ids)[layer]
    m = run.mat("layer", ids)[layer]
    r = ratio(a, a + m)
    return {"layer": int(layer), "attn": summ(a, with_p=False), "moe": summ(m, with_p=False), "share": r}


def additivity(run: PairRun, ids: list[int]) -> pd.DataFrame:
    """Per layer: block vs attn + MoE (pair means): mean gap with pair-bootstrap CI, per-pair Pearson r."""
    A, M, B = (matrix_pairs(run.mat(k, ids)) for k in ("attn_layer", "layer", "block"))
    rows = []
    for l in A.columns:
        s = A[l] + M[l]
        gap = B[l] - s
        g = gap.to_numpy()
        idx = _boot_idx(len(g))
        bm = g[idx].mean(1)
        r = float(np.corrcoef(B[l], s)[0, 1]) if B[l].std() > 0 and s.std() > 0 else float("nan")
        rows.append({"layer": int(l), "attn": float(A[l].mean()), "moe": float(M[l].mean()), "sum": float(s.mean()), "block": float(B[l].mean()),
                     "gap": float(g.mean()), "gap_lo": float(np.percentile(bm, 2.5)), "gap_hi": float(np.percentile(bm, 97.5)), "r_pair": r})
    return pd.DataFrame(rows)


# ---- expert level (on top of moetrace.analysis via model_data) ------------------------------------------------
def evaluate_expert_pairs(md, layer: int, e: int, val: list[int], n_controls: int) -> dict:
    """analysis.evaluate_expert with pair-level summaries (rescue, Spec, control; all-case and active-only)."""
    from . import analysis as A
    ev = A.evaluate_expert(md, layer, e, val, n_controls)
    pc = ev["per_case"].set_index("case_id")
    drop = md.cases.set_index("case_id")["drop"].loc[val]
    out = {"layer": int(layer), "expert": int(e), "n_val_pairs": len(val) // 2, "val_active": int(pc.active.sum()),
           "rescue": summ(pc.rescue), "spec": summ(pc.spec.dropna()), "control": summ(pc.control_mean.dropna(), with_p=False),
           "rescue_active": summ(pc[pc.active].rescue, with_p=False), "spec_active": summ(pc[pc.active].spec.dropna(), with_p=False),
           "rescue_norm": ratio(pc.rescue.loc[val], drop), "per_case": pc}
    sp = pc.spec.dropna()
    out["spec_norm"] = ratio(sp, drop.loc[sp.index])
    return out


def coalitions_pairs(md, layer: int, val: list[int]) -> dict:
    e = md.expert_rows
    e = e[(e.layer == layer) & e.case_id.isin(val)]
    out = {}
    for kind in ("coalition_clean", "coalition_union", "layer"):
        s = e[e.kind == kind].set_index("case_id").rescue.reindex(val)
        out[kind] = summ(s, with_p=False)
    return out


def pattern(layer_val: dict, spec: Optional[dict], rescue: Optional[dict]) -> str:
    """Direction 2: A one positive, specific expert; B layer localised but no specific expert; C no layer effect."""
    if layer_val is None or not (layer_val["ci_lo"] > 0):
        return "C"
    if spec is not None and rescue is not None and spec.get("n", 0) > 0 and spec["ci_lo"] > 0 and rescue["ci_lo"] > 0:
        return "A"
    return "B"


# ---- heads ------------------------------------------------------------------------------------------------------
class HeadRun:
    def __init__(self, run: str):
        self.dir = run_dir(run)
        self.rows = pd.read_parquet(os.path.join(self.dir, "head_rows.parquet"))
        hr = self.rows[self.rows.kind == "attn_head"]
        self.layers = sorted(int(l) for l in hr.layer.unique())
        self.nH = int(hr["head"].max()) + 1
        self.R = {l: hr[hr.layer == l].pivot(index="case_id", columns="head", values="rescue") for l in self.layers}
        self.ref = {l: {k: self.rows[(self.rows.kind == k) & (self.rows.layer == l)].set_index("case_id").rescue for k in ("attn_layer", "layer", "block")}
                    for l in self.layers}
        self.pos = pd.read_parquet(os.path.join(self.dir, "head_positions.parquet")).set_index("case_id")
        z = np.load(os.path.join(self.dir, "head_attn_final.npz"))
        self.attn = {int(l): z["attn"][i] for i, l in enumerate(z["layers"])}  # [2n, nH, T]
        self.attn_ids = [int(c) for c in z["case_ids"]]
        self.lens = z["lens"]


HEAD_CLASSES = ("final", "str", "ment_filled", "ment_other", "pos0", "other")


def head_table(hr: HeadRun, disc: list[int], val: list[int]) -> pd.DataFrame:
    """One row per (layer, head): validation pair-mean rescue with CI, discovery mean, Spec (rescue minus the mean of the
    other heads of the layer, per directed case), share of the attn_layer rescue, z-score of the validation (and
    discovery) mean over ALL scanned heads (Zhang & Nanda: detection at |z| >= 2)."""
    rows = []
    for l in hr.layers:
        Mv, Md = hr.R[l].loc[val], hr.R[l].loc[disc]
        attn_v = pair_means(hr.ref[l]["attn_layer"].loc[val]).mean()
        tot = Mv.sum(1)
        for h in range(hr.nH):
            s = summ(Mv[h])
            others = (tot - Mv[h]) / (hr.nH - 1)
            sp = summ(Mv[h] - others, with_p=False)
            rows.append(dict(layer=l, head=h, val_mean=s["mean"], val_lo=s["ci_lo"], val_hi=s["ci_hi"], val_p=s["p"], val_pos_frac=s["pos_frac"],
                             disc_mean=float(pair_means(Md[h]).mean()), spec=sp["mean"], spec_lo=sp["ci_lo"], spec_hi=sp["ci_hi"],
                             share_attn_layer=float(s["mean"] / attn_v) if attn_v else np.nan))
    df = pd.DataFrame(rows)
    for col, z in (("val_mean", "z_val"), ("disc_mean", "z_disc")):
        df[z] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    df["detected_2sd"] = (df.z_val.abs() >= 2) & (df.z_disc.abs() >= 2)
    return df.sort_values("val_mean", ascending=False).reset_index(drop=True)


def head_layer_summary(hr: HeadRun, l: int, disc: list[int], val: list[int], fracs=(0.5, 0.8)) -> dict:
    """Additivity (sum of heads vs attn_layer, pair level) and greedy additive minimal head sets (order by discovery
    mean, cumulative validation mean vs frac x the attn_layer validation mean)."""
    Mv = hr.R[l].loc[val]
    ssum = Mv.sum(1)
    attn = hr.ref[l]["attn_layer"].loc[val]
    sp, ap = pair_means(ssum), pair_means(attn)
    out = {"layer": l, "attn_layer": summ(attn, with_p=False), "moe_layer": summ(hr.ref[l]["layer"].loc[val], with_p=False),
           "block": summ(hr.ref[l]["block"].loc[val], with_p=False), "sum_heads": summ(ssum, with_p=False),
           "r_sum_vs_attn_pair": float(np.corrcoef(sp, ap)[0, 1]) if sp.std() > 0 and ap.std() > 0 else float("nan")}
    md = pair_means_matrix(hr.R[l].loc[disc]).mean(0)
    mv = pair_means_matrix(Mv).mean(0)
    order = list(md.sort_values(ascending=False).index)
    cum = np.cumsum(mv.loc[order].to_numpy())
    a = float(ap.mean())
    for f in fracs:
        hit = np.nonzero(cum >= f * a)[0] if a > 0 else []
        out[f"k_{int(f * 100)}"] = int(hit[0]) + 1 if len(hit) else None
        out[f"set_{int(f * 100)}"] = [int(h) for h in order[: out[f"k_{int(f * 100)}"]]] if out[f"k_{int(f * 100)}"] else None
    out["top1_head_disc"] = int(order[0])
    out["top3_share_of_attn"] = float(mv.loc[order[:3]].sum() / a) if a > 0 else float("nan")
    return out


def pair_means_matrix(M: pd.DataFrame) -> pd.DataFrame:
    return matrix_pairs(M)


def position_classes(hr: HeadRun, cid: int) -> np.ndarray:
    """Class index per position (HEAD_CLASSES order). Priority final > str > ment_filled > ment_other > pos0 > other."""
    r = hr.pos.loc[cid]
    T = int(r["T"])
    cls = np.full(T, HEAD_CLASSES.index("other"), dtype=np.int64)
    cls[0] = HEAD_CLASSES.index("pos0")
    for k in ("ment_other", "ment_filled", "str_pos"):
        for p in json.loads(r[k]):
            cls[p] = HEAD_CLASSES.index("str" if k == "str_pos" else k)
    cls[T - 1] = HEAD_CLASSES.index("final")
    return cls


def attention_mass(hr: HeadRun, l: int, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """[n, nH, n_classes] final-position attention mass per position class, clean and corrupted runs."""
    A = hr.attn[l].astype(np.float32)
    n = len(hr.attn_ids)
    pos = {c: i for i, c in enumerate(hr.attn_ids)}
    oc = np.zeros((len(ids), A.shape[1], len(HEAD_CLASSES)), dtype=np.float32)
    on = np.zeros_like(oc)
    for j, c in enumerate(ids):
        i = pos[c]
        cls = position_classes(hr, c)
        T = len(cls)
        for k in range(len(HEAD_CLASSES)):
            msk = cls == k
            if msk.any():
                oc[j, :, k] = A[i, :, :T][:, msk].sum(-1)
                on[j, :, k] = A[n + i, :, :T][:, msk].sum(-1)
    return oc, on
