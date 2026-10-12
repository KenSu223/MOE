"""ext13ko (Phase 5, agent p5-expert): Direction-9 expert knockout for every Phase-5 condition.

Question (Direction 9 carried to Phase 5, plus P5-8). Are the experts that the final-position STR patches select for a
Phase-5 condition NECESSARY for the behaviour on clean prompts? Is Phase 3's WinoGrande expert (Qwen3 L41E117, Mixtral
L20E000) still necessary when the answer can be copied from a prefix (cp_* conditions) or when the direction is reversed
(attribute -> entity, fill-in-the-blank)? The machinery is Phase 4's (moetrace/ext9_knockout.py, engine E4 route masks);
this module only re-targets it to the Phase-5 condition registry data/wino_ref/p5_conditions.json (agent p5-data).

Per condition x model (protocols qwen3 and mixtral_bos; Mixtral with BOS as in Phases 3-4):
  items      clean prompts of BOTH twins of every pair of the condition's case sets. eval = every pair except the main
             discovery pairs (validation + replication splits); sub = the main validation pairs. Delta = logit(own answer)
             - logit(twin's answer) (prompt A: trig_a - trig_b; prompt B: the reverse). Bootstrap unit = the condition's
             cluster (template seed for the semi-synthetic template conditions, source pair otherwise; see unit_col).
  targets    the condition's own selected expert (p5-data two-stage selection), Phase 3's WinoGrande expert, the
             CounterFact experts (Qwen3 L44E069, L42E115; Mixtral L19E002, L21E001, L18E001) as cross-task controls.
  sets       the condition's population top-1/3/5/10 (p5-data pop_top10 = discovery all-case single-expert STR rescue).
  controls   same-layer experts of the MAIN targets (own selected + WinoGrande expert; Mixtral: the 7 other experts of the
             layer, Qwen3: the 16 experts of the layer most often routed at the final position on the condition's eval
             prompts, excluding every target); 5 frequency-matched random sets per set size (ext9 rule: each member
             replaced by a random one of the 4 (Qwen3) / 3 (Mixtral) experts of the same layer with the nearest
             final-position routing frequency on the eval prompts, excluding the condition's top-10 and every target).
  masks      route mask at all positions, mode reroute (Phase 4 primary); final-position-only masks for the targets and
             the top-10 set (secondary). Zero mode is not run: Phase 4 found zero ~ reroute (ext9 final_zero table).
  null       the unmasked baseline recomputed in passes of its own (other batch composition) = noise floor.
  cross      (optional phase) the other conditions' selected experts knocked out on this condition (sub scope).
Scopes: full = eval items (baseline, targets, sets, null), sub = main validation items (controls, random sets,
final-only masks, cross).
F = (sum Delta_base - sum Delta_ko) / sum Delta_base over the evaluated items (ratio of means), bootstrap over units.
Self-repair: the engine's contrib_dla is recorded on every row, so the knockout's loss can be compared with (i) the
expert's own direct write in the baseline (ext9_synthesis_selfrepair convention) and (ii) the net direct-write change of
the whole MoE output of the masked layer (which includes the replacement expert's write under reroute).
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Optional

import numpy as np
import pandas as pd

from .models import MODELS, RESULTS

ROOT = "/home/ubuntu/MOE"
CONDS_JSON = os.path.join(ROOT, "data/wino_ref/p5_conditions.json")
N_RAND = 5
SET_SIZES = (1, 3, 5, 10)
N_BOOT = 5000
CONDITIONS = ("f1", "dir_f1", "dir_opt", "f2e", "vt_e2a_orig", "vt_e2a_neu", "vt_a2e_orig", "vt_a2e_neu",
              "cp_vt_orig", "cp_vt_neu", "cp_nat")
A2E = {"f1", "dir_f1", "f2e", "vt_a2e_orig", "vt_a2e_neu", "f1pre"}  # readout = entity; else readout = attribute

PROTO = {
    "qwen3": dict(model="qwen3", bos=None, n_same=16, rand_near=4, wg=(41, 117), cf=[(44, 69), (42, 115)],
                  budget=160_000, max_rows=6_000, label="Qwen3-30B-A3B-Base"),
    "mixtral_bos": dict(model="mixtral", bos=1, n_same=7, rand_near=3, wg=(20, 0), cf=[(19, 2), (21, 1), (18, 1)],
                        budget=140_000, max_rows=6_000, label="Mixtral-8x7B (BOS)"),
}

# smoke-test entry: the pre-neutral F1 run of the other Phase-5 session (selected expert / pop computed here)
SMOKE = {
    "f1pre": {"qwen3": dict(pairs="data/wino_ref/pairs_F1_train_xl_qwen3.parquet",
                            case_sets="data/wino_ref/case_sets_F1_qwen3.json", run="wino_ref_F1_qwen3",
                            selected_expert=None, pop_top10=None, status={"base": True})},
}


def le(l: int, e: int) -> str:
    return f"L{int(l)}E{int(e):03d}"


def parse_le(s) -> tuple:
    if isinstance(s, (list, tuple)):
        return int(s[0]), int(s[1])
    m = re.match(r"L(\d+)E(\d+)$", str(s).strip())
    return int(m.group(1)), int(m.group(2))


# ------------------------------------------------------------------------------------------------------------------
# condition registry (p5-data's data/wino_ref/p5_conditions.json; tolerant of the exact layout)
# ------------------------------------------------------------------------------------------------------------------
def _registry() -> dict:
    if not os.path.exists(CONDS_JSON):
        return {}
    J = json.load(open(CONDS_JSON))
    if isinstance(J, dict) and "conditions" in J:
        J = J["conditions"]
    out = {}
    if isinstance(J, list):
        for e in J:
            out.setdefault(e["cond"], {})[e.get("proto")] = e
    else:
        for k, v in J.items():
            if not isinstance(v, dict):
                continue
            if "/" in k:
                c, p = k.split("/", 1)
                out.setdefault(c, {})[p] = v
            elif any(p in v for p in PROTO):
                for p in PROTO:
                    if p in v and isinstance(v[p], dict):
                        out.setdefault(k, {})[p] = v[p]
    return out


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def entry(cond: str, proto: str) -> dict:
    """Normalised condition entry: cond, proto, model, pairs, case_sets, run, selected (tuple|None), pop_top10 (list of
    tuples|None), status (dict), direction ('a2e'|'e2a'), unit_col, raw (the registry entry)."""
    if cond in SMOKE:
        e = dict(SMOKE[cond][proto])
    else:
        reg = _registry()
        assert cond in reg and proto in reg[cond], f"{cond}/{proto} not in {CONDS_JSON}"
        e = reg[cond][proto]
    pairs = _first(e, "pairs", "pair_file", "pairs_path", "pairs_parquet")
    cs = _first(e, "case_sets", "case_set_file", "case_sets_path", "case_sets_json")
    run = _first(e, "run", "run_dir", "out")
    if run and os.path.isabs(run):
        run = os.path.relpath(run, RESULTS)
    if run and run.startswith("results/"):
        run = run[len("results/"):]
    sel = _first(e, "selected_expert", "selected")
    if isinstance(sel, dict):
        sel = (sel["layer"], sel["expert"]) if "layer" in sel and "expert" in sel else _first(sel, "name", "le")
    sel = parse_le(sel) if sel not in (None, "", "none") else None
    pop = _first(e, "pop_top10", "pop_top", "pop")
    if pop is not None:
        pop = [parse_le(x if not (isinstance(x, (list, tuple)) and len(x) == 3) else x[:2]) if not isinstance(x, dict)
               else parse_le(_first(x, "expert", "name", "le")) for x in pop]
    st = e.get("status", {})
    if not isinstance(st, dict):
        st = {"base": bool(st)}
    direction = {"attr2ent": "a2e", "ent2attr": "e2a"}.get(_first(e, "direction", default=None), _first(e, "direction", default=None))
    if direction not in ("a2e", "e2a"):
        direction = "a2e" if cond in A2E else "e2a"
    return dict(cond=cond, proto=proto, model=PROTO[proto]["model"], pairs=pairs, case_sets=cs, run=run, selected=sel,
                pop_top10=pop, status=st, direction=direction, unit_col=e.get("unit_col"), raw=e,
                behaviour_only=bool(_first(e, "behaviour_only", "behavior_only", default=False)))


def ready_conditions(proto: str) -> list:
    """Conditions of the registry with status.base true and not behaviour-only, in CONDITIONS order."""
    reg = _registry()
    out = []
    for c in list(CONDITIONS) + sorted(set(reg) - set(CONDITIONS)):
        if c in reg and proto in reg[c]:
            e = entry(c, proto)
            if e["status"].get("base") and not e["behaviour_only"]:
                out.append(c)
    return out


def out_dir(e: dict, suffix: str = "") -> str:
    return os.path.join(RESULTS, f"p5_{e['proto']}_ko{suffix}", e["cond"])


# ------------------------------------------------------------------------------------------------------------------
# items
# ------------------------------------------------------------------------------------------------------------------
def _jl(x) -> list:
    return [int(v) for v in (json.loads(x) if isinstance(x, str) else x)]


UNIT_CANDIDATES = ("seed_id", "seed", "template_id", "template", "src_pair_id", "source_pair_id", "orig_pair_id",
                   "base_pair_id", "item_id")
STRATA = ("names", "assoc", "stratum", "assoc_stratum", "stratum_assoc", "stratum_neutral", "local_assoc", "stratum_local_assoc", "debiased", "swap_type", "order", "verb", "variant", "seed_id",
          "seed", "src_pair_id", "source_pair_id", "base_pair_id", "phase3_pair_idx", "n_tokens", "word_a", "word_b", "ans_a",
          "ans_b", "name_x", "name_y")


def families(e: dict) -> dict:
    """{split_name: [directed ids]} with split names discovery / validation / rep_discovery / rep_validation / <other>."""
    cs = json.load(open(os.path.join(ROOT, e["case_sets"])))
    out = {}
    for k, v in cs.get("directed", {}).items():
        if k == "shared_unused":
            continue
        out[k] = [int(c) for c in v]
    return out


def unit_column(e: dict, p: pd.DataFrame) -> str:
    if e.get("unit_col") and e["unit_col"] in p.columns:
        return e["unit_col"]
    if e["cond"].startswith(("vt_", "cp_vt")):
        for c in ("seed_id", "seed", "template_id", "template"):
            if c in p.columns:
                return c
    for c in ("src_pair_id", "source_pair_id", "orig_pair_id", "base_pair_id"):
        if c in p.columns:
            return c
    return "pair_id"


def build_items(e: dict, od: str, limit: Optional[int] = None) -> pd.DataFrame:
    """One row per clean prompt (both twins of every pair of the case sets): item, key (pair_id:a|b), pair_idx, side,
    case_id (directed id whose clean prompt this is), split, unit, ids, true_id, foil_id, T, in_eval, in_sub, in_disc,
    strata. Cached in <od>/items.parquet."""
    path = os.path.join(od, "items.parquet")
    if os.path.exists(path):
        it = pd.read_parquet(path)
        it["ids"] = it.ids_json.map(json.loads)
        return it
    from . import ext7_pairs as P
    cs = json.load(open(os.path.join(ROOT, e["case_sets"])))
    p = P.load_pairs(os.path.join(ROOT, e["pairs"]), cs)
    fam = families(e)
    split_of = {}
    for s, ids in fam.items():
        for c in ids:
            split_of.setdefault(int(c) // 2, s)
    ucol = unit_column(e, p)
    rows = []
    per_split = {}
    for pi in sorted(split_of):
        s = split_of[pi]
        if limit is not None:
            per_split[s] = per_split.get(s, 0) + 1
            if per_split[s] > limit:
                continue
        r = p.loc[pi]
        u = str(r[ucol]) if ucol in p.columns else str(r.pair_id)
        extra = {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in STRATA if c in p.columns}
        for d, (side, ids, t, f) in enumerate((("a", r.ids_a_l, r.trig_a, r.trig_b), ("b", r.ids_b_l, r.trig_b, r.trig_a))):
            rows.append(dict(key=f"{r.pair_id}:{side}", pair_id=str(r.pair_id), pair_idx=int(pi), side=side,
                             case_id=2 * int(pi) + d, split=s, unit=u, ids=[int(x) for x in ids], true_id=int(t),
                             foil_id=int(f), **extra))
    it = pd.DataFrame(rows)
    it.insert(0, "item", np.arange(len(it)))
    it["T"] = it.ids.map(len)
    it["in_disc"] = it.split == "discovery"
    it["in_eval"] = ~it.in_disc
    it["in_sub"] = it.split == "validation"
    it.attrs["unit_col"] = ucol
    os.makedirs(od, exist_ok=True)
    o = it.drop(columns=["ids"]).copy()
    o["ids_json"] = it.ids.map(json.dumps)
    o.to_parquet(path, index=False)
    json.dump({"unit_col": ucol, "n_items": int(len(it)), "n_pairs": int(it.pair_idx.nunique()),
               "n_units": int(it.unit.nunique()), "splits": it.groupby("split").pair_idx.nunique().to_dict()},
              open(os.path.join(od, "items_meta.json"), "w"), indent=1)
    return it


def scope_mask(it: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "full":
        return it.in_eval.values.astype(bool)
    if scope == "sub":
        return it.in_sub.values.astype(bool)
    if scope == "all":
        return np.ones(len(it), dtype=bool)
    raise ValueError(scope)


# ------------------------------------------------------------------------------------------------------------------
# targets, sets, controls
# ------------------------------------------------------------------------------------------------------------------
def pop_ranking(e: dict, od: str) -> list:
    """The condition's population ranking [(l, e, value)], top 32: p5-data's pop_top10 when given (value = rank order),
    else computed here from the run's expert rows (ext8 convention: discovery all-case single-expert STR rescue)."""
    path = os.path.join(od, "pop_rank.json")
    if os.path.exists(path):
        return [tuple(x) for x in json.load(open(path))["ranking"]]
    from . import ext8_addback as A
    src = "p5_conditions.json"
    rk = None
    try:
        t = A.load_wino(e["model"], e["run"], os.path.join(ROOT, e["pairs"]), os.path.join(ROOT, e["case_sets"]))
        r = sorted(t.pop.items(), key=lambda kv: (-kv[1], kv[0]))[:32]
        rk = [(int(l), int(x), float(v)) for (l, x), v in r]
    except Exception as ex:  # noqa: BLE001
        if not e.get("pop_top10"):
            raise
        print(f"pop ranking from expert rows failed ({ex}); using p5_conditions pop_top10")
    if e.get("pop_top10"):
        given = [tuple(x) for x in e["pop_top10"]]
        vals = {(l, x): v for l, x, v in (rk or [])}
        if rk is not None and [tuple(x[:2]) for x in rk[:len(given)]] != given:
            print(f"note: own pop ranking {[le(*x[:2]) for x in rk[:10]]} differs from p5-data's {[le(*g) for g in given]}; "
                  "using p5-data's")
        rk_out = [(l, x, float(vals.get((l, x), np.nan))) for l, x in given]
        rk_out += [x for x in (rk or []) if tuple(x[:2]) not in set(given)]
    else:
        src = "computed (ext8 load_wino pop)"
        rk_out = rk
    os.makedirs(od, exist_ok=True)
    json.dump({"source": src, "ranking": [list(x) for x in rk_out]}, open(path, "w"), indent=1)
    return rk_out


def selected_expert(e: dict, od: str) -> Optional[tuple]:
    if e.get("selected"):
        return tuple(e["selected"])
    if e["cond"] not in SMOKE:
        return None  # no specific expert: the top-10 set carries the condition (sets are always run)
    rk = pop_ranking(e, od)  # smoke: the population top-1
    return (int(rk[0][0]), int(rk[0][1])) if rk else None


def targets(e: dict, od: str) -> dict:
    """{'selected': (l, e)|None, 'wg': (l, e), 'cf': [...], 'main': [...], 'all': [...]} (unique, order kept)."""
    cfg = PROTO[e["proto"]]
    sel = selected_expert(e, od)
    main = list(dict.fromkeys([x for x in (sel, cfg["wg"]) if x is not None]))
    allt = list(dict.fromkeys(main + list(cfg["cf"])))
    return {"selected": sel, "wg": cfg["wg"], "cf": list(cfg["cf"]), "main": main, "all": allt}


def pop_sets(e: dict, od: str) -> dict:
    rk = pop_ranking(e, od)
    return {k: [(int(l), int(x)) for l, x, _ in rk[:k]] for k in SET_SIZES if len(rk) >= k}


def final_freq(e: dict, it: pd.DataFrame, n_experts: int, n_layers: int) -> np.ndarray:
    """[L, E] final-position routing frequency on the condition's eval prompts (clean routing of the run's sweep,
    slot -1, one row per directed case = one clean prompt)."""
    rt = pd.read_parquet(os.path.join(RESULTS, e["run"], "str_sweep_routing.parquet"), columns=["case_id", "slot", "layer", "expert"])
    rt = rt[rt.slot == -1]
    ev = set(it.case_id[it.in_eval].astype(int))
    have = set(rt.case_id.unique()) & ev
    if len(have) < 0.5 * len(ev):  # fall back to every case of the sweep
        have = set(rt.case_id.unique())
    rt = rt[rt.case_id.isin(have)]
    cnt = np.zeros((n_layers, n_experts))
    np.add.at(cnt, (rt.layer.values.astype(int), rt.expert.values.astype(int)), 1.0)
    return cnt / max(len(have), 1)


def control_sets(e: dict, od: str, it: pd.DataFrame) -> dict:
    path = os.path.join(od, "controls.json")
    if os.path.exists(path):
        return json.load(open(path))
    from .arch import load_spec
    cfg = PROTO[e["proto"]]
    spec, _ = load_spec(MODELS[cfg["model"]]["repo"])
    NE, L = spec.n_experts, spec.n_layers
    fq = final_freq(e, it, NE, L)
    tg = targets(e, od)
    tset = set(tg["all"])
    ps = pop_sets(e, od)
    out = {"same_layer": {}, "random": {}, "note": "same_layer: per main target the control experts of its layer; random: "
           "per set size, 5 sets matched member-wise on final-position routing frequency (eval prompts) within the member's layer"}
    for (l, x) in tg["main"]:
        if NE <= 8:
            ctl = [y for y in range(NE) if (l, y) not in tset]
        else:
            cand = sorted([y for y in range(NE) if (l, y) not in tset], key=lambda y: (-fq[l, y], y))
            ctl = cand[: cfg["n_same"]]
        out["same_layer"][le(l, x)] = {"layer": l, "controls": ctl, "freq_target": float(fq[l, x]),
                                       "freq_controls": [float(fq[l, y]) for y in ctl]}
    excl = tset | set(ps.get(10, ps[max(ps)]))
    for k, members in ps.items():
        sets = []
        for r in range(N_RAND):
            rng = random.Random(9000 + 100 * k + r)
            used, chosen = set(), []
            for (l, x) in members:
                cand = [y for y in range(NE) if (l, y) not in excl and (l, y) not in used]
                cand = sorted(cand, key=lambda y: (abs(fq[l, y] - fq[l, x]), y))[: cfg["rand_near"]]
                pick = rng.choice(cand)
                used.add((l, pick))
                chosen.append((l, pick))
            sets.append(sorted(chosen))
        out["random"][f"k{k}"] = {"members": [list(m) for m in members], "member_freq": [float(fq[m]) for m in members],
                                  "sets": [[list(m) for m in s_] for s_ in sets],
                                  "set_freq": [[float(fq[tuple(m)]) for m in s_] for s_ in sets]}
    json.dump(out, open(path, "w"), indent=1)
    return out


# ------------------------------------------------------------------------------------------------------------------
# conditions (interventions) and pass plan
# ------------------------------------------------------------------------------------------------------------------
def _cond(cid, mask, scope, phase, pos="all", mode="reroute", group=None, **meta) -> dict:
    """group = pass group: 'full' (full-scope rows: baseline, targets, sets) or 'ctl' (sub-scope rows and the null rows,
    so that the null baseline is recomputed in passes of another composition than the baseline rows)."""
    if group is None:
        group = "full" if scope == "full" else "ctl"
    return dict(cid=cid, mask=tuple(sorted((int(l), int(x)) for l, x in mask)), scope=scope, phase=phase, pos=pos, mode=mode,
                group=group, meta=meta)


def mask_str(mask) -> str:
    return "+".join(le(l, x) for l, x in mask) if mask else "none"


def cond_sig(c: dict) -> str:
    if not c["mask"]:
        return "none|all|null" if c.get("meta", {}).get("kind") == "null" else "none|all|reroute"
    return f"{mask_str(c['mask'])}|{c['pos']}|{c['mode']}"


def parse_sig(sig: str) -> tuple:
    m, pos, mode = sig.split("|")
    mask = () if m == "none" else tuple(parse_le(x) for x in m.split("+"))
    return mask, pos, mode


def conditions(e: dict, od: str, it: pd.DataFrame, phase: str = "main", cross: Optional[list] = None) -> list:
    tg = targets(e, od)
    ps = pop_sets(e, od)
    out = []
    if phase == "main":
        out.append(_cond("base", (), "full", phase, kind="base"))
        out.append(_cond("null", (), "full", phase, group="ctl", kind="null"))
        role = {}
        for x in tg["all"]:
            r = []
            if x == tg["selected"]:
                r.append("selected")
            if x == tg["wg"]:
                r.append("wg")
            if x in tg["cf"]:
                r.append("cf")
            role[x] = "+".join(r)
            out.append(_cond(f"tgt:{le(*x)}", [x], "full", phase, kind="target", role=role[x]))
        tsig = {cond_sig(c) for c in out}
        for k, s_ in ps.items():
            c = _cond(f"pop:k{k}", s_, "full", phase, kind="pop", k=k)
            if cond_sig(c) not in tsig:
                out.append(c)
        cs = control_sets(e, od, it)
        for k in ps:
            for r, s_ in enumerate(cs["random"][f"k{k}"]["sets"]):
                out.append(_cond(f"rand:k{k}:r{r}", [tuple(m) for m in s_], "sub", phase, kind="rand", k=k))
        seen = set()
        for tname, v in cs["same_layer"].items():
            for y in v["controls"]:
                c = le(v["layer"], y)
                if c in seen:
                    continue
                seen.add(c)
                out.append(_cond(f"same:{c}", [(v["layer"], y)], "sub", phase, kind="same", layer=v["layer"]))
        for x in tg["all"]:
            out.append(_cond(f"fin:{le(*x)}", [x], "sub", phase, pos="final", kind="target_final", role=role[x]))
        if 10 in ps:
            out.append(_cond("fin:pop:k10", ps[10], "sub", phase, pos="final", kind="pop_final", k=10))
    elif phase == "cross":
        have = set(tuple(x) for x in tg["all"])
        for name, x in (cross or []):
            x = tuple(x)
            if x in have:
                continue
            have.add(x)
            out.append(_cond(f"cross:{le(*x)}", [x], "sub", phase, kind="cross", from_cond=name))
    else:
        raise ValueError(phase)
    return out


def plan_passes(it: pd.DataFrame, conds: list, budget: int, max_rows: int, done: Optional[set] = None) -> list:
    """Deterministic pass plan [{name, group, rows: [(sig, item)]}]: unique (sig, item) rows per pass group, sorted by
    prompt length, padded tokens <= budget and rows <= max_rows per pass, pass sizes balanced."""
    done = done or set()
    T = it.set_index("item")["T"]
    groups, seen = {}, set()
    for c in conds:
        sig = cond_sig(c)
        for x in it.item.values[scope_mask(it, c["scope"])]:
            r = (sig, int(x))
            if r in seen or r in done:
                continue
            seen.add(r)
            groups.setdefault(c.get("group", "full"), []).append(r)

    def fill(rows, bud):
        out, cur, tmax = [], [], 0
        for r in rows:
            t = int(T[r[1]])
            if cur and (max(tmax, t) * (len(cur) + 1) > bud or len(cur) + 1 > max_rows):
                out.append(cur)
                cur, tmax = [], 0
            cur.append(r)
            tmax = max(tmax, t)
        if cur:
            out.append(cur)
        return out

    passes = []
    for g in sorted(groups, key=lambda z: (z != "full", z)):
        rows = sorted(groups[g], key=lambda r: (int(T[r[1]]), r[0], r[1]))
        n_min = len(fill(rows, budget))
        lo, hi = budget // 4, budget
        while hi - lo > 1000:
            mid = (lo + hi) // 2
            if len(fill(rows, mid)) <= n_min:
                hi = mid
            else:
                lo = mid
        for k, cur in enumerate(fill(rows, hi)):
            passes.append({"name": f"{g}_p{k:03d}", "group": g, "rows": [[s_, int(x)] for s_, x in cur]})
    return passes


# ------------------------------------------------------------------------------------------------------------------
# analysis helpers
# ------------------------------------------------------------------------------------------------------------------
def load_rows(od: str) -> pd.DataFrame:
    fs = sorted(f for f in os.listdir(od) if f.startswith("ko_rows_") and f.endswith(".parquet"))
    return pd.concat([pd.read_parquet(os.path.join(od, f)) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def load_basefinal(od: str) -> dict:
    """Baseline rows' final-position routing and per-slot DLA: items, route_idx [L, n, k], contrib_dla [L, n, k],
    final_rms [n] (sorted by item)."""
    fs = sorted(f for f in os.listdir(od) if f.startswith("ko_basefinal_") and f.endswith(".npz"))
    zs = [np.load(os.path.join(od, f)) for f in fs]
    if not zs:
        return {}
    items = np.concatenate([z["items"] for z in zs])
    o = np.argsort(items, kind="stable")
    out = {"items": items[o]}
    for k in ("route_idx", "contrib_dla"):
        out[k] = np.concatenate([z[k] for z in zs], axis=1)[:, o]
    out["final_rms"] = np.concatenate([z["final_rms"] for z in zs])[o]
    return out
