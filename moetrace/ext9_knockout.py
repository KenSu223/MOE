"""ext9 (Phase 4, Item 1 / Direction 9): expert knockout = necessity and task specificity of the patching-selected experts.

Question. The final-position patches of Directions 6-8 show that a few experts are SUFFICIENT to move the answer under STR.
Are they also NECESSARY for the behaviour on clean prompts, and only for their own task? A knockout removes an expert
from the router's menu (engine E4a route mask, PrefillSpec.route_mask) and measures how much of the clean margin is lost.

Models (Phase-3 default): Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 WITH BOS (tokenizer defaults), bf16.
Targets: Qwen3 WinoGrande L41E117, CounterFact L44E069 and L42E115; Mixtral WinoGrande L20E000, CounterFact L19E002,
L21E001, L18E001; plus joint knockout of the top-1/3/5/10 experts of each task's ext8 population ranking (discovery means
of the single-expert STR rescue; moetrace.ext8_addback.load_cf / load_wino, read-only).
Controls: same-layer experts (Mixtral: the 7 other experts of the layer; Qwen3: the 16 experts of the layer most often
routed at the final position on the target task's evaluation items, excluding every target) and, for the sets, 5 random
equal-size sets matched on final-position routing frequency (each member replaced by a random one of the 4 (Qwen3) / 3
(Mixtral) experts OF THE SAME LAYER with the nearest final-position routing frequency on the task's evaluation items,
excluding the task's population top-10 and every target; sampled without replacement).

Evaluation items (clean prompts only; items used to select an expert are excluded):
  wg    WinoGrande STR pairs (data/wino_str/pairs_train_xl_<proto>.parquet), both prompts of a pair, Delta toward each
        prompt's own trigger (prompt A: logit(trig_a) - logit(trig_b); prompt B: the reverse). Minus the 128 main discovery
        pairs (data/wino_str/case_sets.json). "margin" = the model's own margin pool (results/wino_<proto>/scan_pairs.parquet
        `margin`), "all" = every W1-W6 pair.
  cf    CounterFact clean scan (results/{qwen3,mixtral}/filter_scan.parquet, tokenizer defaults = Mixtral BOS), Delta_clean
        >= 1, minus the paper discovery IDs (data/paper_case_ids.json).
  ioi   IOI clean prompts (data/ioi/items.parquet, <proto>_ids_clean), Delta = logit(IO) - logit(S); primary subset =
        baseline Delta >= 1.
  wiki  wikitext-103 test windows of 127 content tokens (ext3 detokenisation; Mixtral: BOS prepended), per-token NLL of
        content tokens 1..126 (engine DiagSpec.token_logprobs), generic-damage control.
Scopes: full = wg all + cf + ioi all + wiki 128 windows (baseline, targets, population sets); sub = fixed subsample 512 wg
margin pairs / 512 cf / 256 ioi / 64 wiki windows (same-layer controls, random sets, zero mode); sub_nowiki = sub without the
wiki windows (final-position-only masks: a final-only mask cannot change the NLL of the window's tokens).
Interventions: route mask at all positions + reroute (primary); final position only (secondary); zero mode (targets).
Noise floor: condition `null` (signature none|all|null) = the unmasked baseline recomputed in the random phase's passes (other
batch composition; bf16 results depend on it at the 0.1-logit level), evaluated like a knockout.
Metrics: fraction of the clean margin lost F = (mean Delta_base - mean Delta_ko) / mean Delta_base (ratio of means over
items; bootstrap over pairs / cases / items / windows), accuracy (Delta > 0; WinoGrande pair accuracy = both prompts),
p(true), rank and top-1 rate (descriptive), NLL change on wikitext; rank / z among the controls; double-dissociation
contrast (F_wg-expert(WG) - F_wg-expert(CF)) - (F_cf-expert(WG) - F_cf-expert(CF)) with a paired bootstrap that resamples
WinoGrande pairs and CounterFact cases independently; routing change at the masked layer (which expert took the slot).
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
WIKI = "/opt/dlami/nvme/moe_ext3/corpus/wikitext103_test.parquet"
WIKI_WINDOW = 127
N_WIKI_FULL, N_WIKI_SUB = 128, 64
N_SUB_WG, N_SUB_CF, N_SUB_IOI = 512, 512, 256
SUB_SEED = 9
N_RAND = 5
SET_SIZES = (1, 3, 5, 10)
N_BOOT = 5000

KO = {
    "qwen3": dict(model="qwen3", run="qwen3_knockout", proto="qwen3", cf_scan="qwen3", paper_key="Qwen3", bos=None,
                  wg_run="wino_qwen3_str", cf_run="qwen3_str", n_same=16, rand_near=4,
                  targets={"wg": [(41, 117)], "cf": [(44, 69), (42, 115)]}, primary={"wg": (41, 117), "cf": (44, 69)},
                  budget=200_000, max_rows=12_000, wiki_budget=152_000),  # 240k: 17.5 GB peak, some passes OOM (q-norm fp32 copy)
    "mixtral": dict(model="mixtral", run="mixtral_bos_knockout", proto="mixtral_bos", cf_scan="mixtral", paper_key="Mixtral", bos=1,
                    wg_run="wino_mixtral_bos_str", cf_run="mixtral_bos_str", n_same=7, rand_near=3,
                    targets={"wg": [(20, 0)], "cf": [(19, 2), (21, 1), (18, 1)]}, primary={"wg": (20, 0), "cf": (21, 1)},
                    budget=140_000, max_rows=16_000, wiki_budget=120_000),
    # OLMoE: driver / analysis smoke test only (tiny item sets, arbitrary expert sets; not a result)
    "olmoe": dict(model="olmoe", run="olmoe_knockout_smoke", proto="olmoe", cf_scan="olmoe", paper_key=None, bos=None,
                  wg_run=None, cf_run=None, n_same=6, rand_near=4,
                  targets={"wg": [(12, 40)], "cf": [(13, 56)]}, primary={"wg": (12, 40), "cf": (13, 56)},
                  budget=60_000, max_rows=4_000, wiki_budget=30_000, limit={"wg": 96, "cf": 96, "ioi": 64, "wiki": 24},
                  pop_override={"wg": [(12, 40), (11, 3), (10, 7), (9, 1), (8, 2), (12, 5), (13, 9), (14, 11), (7, 4), (6, 8)],
                                "cf": [(13, 56), (13, 1), (12, 2), (11, 4), (10, 5), (9, 6), (14, 7), (15, 8), (8, 9), (7, 10)]}),
}
PHASES = ("full", "random", "samelayer", "final")


def run_dir(key: str) -> str:
    return os.path.join(RESULTS, KO[key]["run"])


def le(l: int, e: int) -> str:
    return f"L{int(l)}E{int(e):03d}"


def parse_le(s: str) -> tuple:
    m = re.match(r"L(\d+)E(\d+)$", s)
    return int(m.group(1)), int(m.group(2))


def all_targets(key: str) -> list:
    t = KO[key]["targets"]
    return sorted(set(t["wg"]) | set(t["cf"]))


# ------------------------------------------------------------------------------------------------------------------
# items
# ------------------------------------------------------------------------------------------------------------------
def wiki_text() -> str:
    """wikitext-103 test split, lightly detokenised (identical to scripts/ext3_corpus_routing.py:wiki_text)."""
    d = pd.read_parquet(WIKI)
    lines = [t for t in d.text.tolist() if t.strip() and not t.strip().startswith("=")]
    t = "".join(lines)
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r" ([,.;:!?')])", r"\1", t)
    t = re.sub(r"([(\[]) ", r"\1", t)
    t = t.replace(" n't", "n't").replace(" 's", "'s").replace('" ', '"').replace(" \"", "\"")
    return t


def _ids(x) -> list:
    return [int(v) for v in (json.loads(x) if isinstance(x, str) else x)]


def build_items(key: str) -> pd.DataFrame:
    """One row per evaluation prompt: item, task, ids (list), true_id, foil_id, T, unit (bootstrap unit), and scope flags
    in_full / in_margin / in_sub. Written to <run>/items.parquet (ids as JSON) on first call."""
    cfg = KO[key]
    path = os.path.join(run_dir(key), "items.parquet")
    if os.path.exists(path):
        it = pd.read_parquet(path)
        it["ids"] = it.ids_json.map(json.loads)
        return it
    rows = []
    # ---- WinoGrande
    cs = json.load(open(os.path.join(ROOT, "data/wino_str/case_sets.json")))
    pio = cs["pair_idx_of"]
    disc = set(int(x) for x in cs["pairs"]["discovery"])
    p = pd.read_parquet(os.path.join(ROOT, f"data/wino_str/pairs_train_xl_{cfg['proto']}.parquet"))
    sc = pd.read_parquet(os.path.join(RESULTS, f"wino_{cfg['proto']}", "scan_pairs.parquet"))[["pair_id", "margin", "names", "assoc"]]
    p = p.drop(columns=[c for c in ("margin", "names", "assoc") if c in p.columns]).merge(sc, on="pair_id", how="left", validate="one_to_one")
    p["pair_idx"] = p.pair_id.map(pio)
    p["disc"] = p.pair_idx.isin(disc)
    p = p[~p.disc].sort_values("pair_id").reset_index(drop=True)
    lim = cfg.get("limit", {})
    if "wg" in lim:
        p = p.iloc[: lim["wg"]].reset_index(drop=True)
    marg = p[p.margin.astype(bool)]
    rs = random.Random(SUB_SEED)
    sub_wg = set(rs.sample(sorted(marg.pair_id), min(N_SUB_WG, len(marg))))
    for r in p.itertuples():
        a, b = json.loads(r.ids_a), json.loads(r.ids_b)
        for side, ids, t, f in (("a", a, r.trig_a, r.trig_b), ("b", b, r.trig_b, r.trig_a)):
            rows.append(dict(task="wg", key=f"{r.pair_id}:{side}", unit=r.pair_id, ids=ids, true_id=int(t), foil_id=int(f),
                             in_full=True, in_margin=bool(r.margin), in_sub=r.pair_id in sub_wg,
                             names=bool(r.names), assoc=bool(r.assoc)))
    # ---- CounterFact
    fs = pd.read_parquet(os.path.join(RESULTS, cfg["cf_scan"], "filter_scan.parquet"))
    pdisc = set(int(c) for c in json.load(open(os.path.join(ROOT, "data/paper_case_ids.json")))[cfg["paper_key"]]["discovery"]) \
        if cfg["paper_key"] else set()
    fs = fs[(fs.delta_clean >= 1.0) & ~fs.case_id.isin(pdisc)].sort_values("case_id").reset_index(drop=True)
    if "cf" in lim:
        fs = fs.iloc[: lim["cf"]].reset_index(drop=True)
    sub_cf = set(random.Random(SUB_SEED + 1).sample(sorted(fs.case_id.astype(int)), min(N_SUB_CF, len(fs))))
    for r in fs.itertuples():
        rows.append(dict(task="cf", key=str(int(r.case_id)), unit=str(int(r.case_id)), ids=_ids(r.ids),
                         true_id=int(r.true_id), foil_id=int(r.foil_id), in_full=True, in_margin=True,
                         in_sub=int(r.case_id) in sub_cf, names=False, assoc=False))
    # ---- IOI
    io = pd.read_parquet(os.path.join(ROOT, "data/ioi/items.parquet")).sort_values("pair_id").reset_index(drop=True)
    if "ioi" in lim:
        io = io.iloc[: lim["ioi"]].reset_index(drop=True)
    px = cfg["proto"]
    from transformers import AutoTokenizer
    from .arch import snapshot_dir
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[cfg["model"]]["repo"]))
    sub_ioi = set(random.Random(SUB_SEED + 2).sample(sorted(io.pair_id), min(N_SUB_IOI, len(io))))
    for r in io.itertuples():
        if f"{px}_ids_clean" in io.columns:
            ids, ti, fi = _ids(getattr(r, f"{px}_ids_clean")), int(getattr(r, f"{px}_trig_io")), int(getattr(r, f"{px}_trig_s"))
        else:  # smoke model without IOI ids: tokenise the clean prompt, first token of the leading-space names
            ids = tok(r.prompt_clean, add_special_tokens=False)["input_ids"]
            ti, fi = tok(" " + r.io, add_special_tokens=False)["input_ids"][0], tok(" " + r.s, add_special_tokens=False)["input_ids"][0]
        rows.append(dict(task="ioi", key=r.pair_id, unit=r.pair_id, ids=ids, true_id=int(ti), foil_id=int(fi), in_full=True,
                         in_margin=r.pair_id in sub_ioi, in_sub=r.pair_id in sub_ioi, names=True, assoc=False))
    # ---- wikitext windows
    ids_all = tok(wiki_text(), add_special_tokens=False)["input_ids"]
    W = WIKI_WINDOW
    n_full = min(N_WIKI_FULL, lim.get("wiki", N_WIKI_FULL))
    for w in range(n_full):
        win = [int(x) for x in ids_all[w * W : (w + 1) * W]]
        ids = ([cfg["bos"]] if cfg["bos"] is not None else []) + win
        rows.append(dict(task="wiki", key=f"w{w:03d}", unit=f"w{w:03d}", ids=ids, true_id=0, foil_id=1, in_full=True,
                         in_margin=False, in_sub=w < (N_WIKI_SUB if n_full == N_WIKI_FULL else n_full // 2), names=False, assoc=False))
    it = pd.DataFrame(rows)
    it.insert(0, "item", np.arange(len(it)))
    it["T"] = it.ids.map(len)
    it["in_wiki64"] = (it.task == "wiki") & it.in_sub
    os.makedirs(run_dir(key), exist_ok=True)
    out = it.drop(columns=["ids"]).copy()
    out["ids_json"] = it.ids.map(json.dumps)
    out.to_parquet(path, index=False)
    return it


def scope_mask(it: pd.DataFrame, scope: str) -> np.ndarray:
    t = it.task.values
    if scope == "full":
        return it.in_full.values.astype(bool)
    if scope == "sub":
        return it.in_sub.values.astype(bool)
    if scope == "margin":
        return ((t == "wg") & it.in_margin.values) | (t == "cf") | ((t == "ioi") & it.in_sub.values)
    if scope == "margin_wiki":
        return scope_mask(it, "margin") | it.in_wiki64.values
    if scope == "sub_nowiki":
        return it.in_sub.values.astype(bool) & (t != "wiki")
    raise ValueError(scope)


# ------------------------------------------------------------------------------------------------------------------
# expert sets and conditions
# ------------------------------------------------------------------------------------------------------------------
def population_ranking(key: str) -> dict:
    """{'wg': [(l, e, pop), ...], 'cf': [...]} = ext8 population ranking (discovery all-case single-expert STR rescue),
    cached in <run>/pop_rank.json."""
    path = os.path.join(run_dir(key), "pop_rank.json")
    if os.path.exists(path):
        d = json.load(open(path))
        return {k: [tuple(x) for x in v] for k, v in d.items()}
    cfg = KO[key]
    if "pop_override" in cfg:  # smoke config
        out = {t: [(int(l), int(e), float(10 - i)) for i, (l, e) in enumerate(v)] for t, v in cfg["pop_override"].items()}
        os.makedirs(run_dir(key), exist_ok=True)
        json.dump(out, open(path, "w"), indent=1)
        return out
    from . import ext8_addback as A
    tc = A.load_cf(cfg["model"], cfg["cf_run"])
    tw = A.load_wino(cfg["model"], cfg["wg_run"], os.path.join(ROOT, f"data/wino_str/pairs_train_xl_{cfg['proto']}.parquet"),
                     os.path.join(ROOT, "data/wino_str/case_sets.json"))
    out = {}
    for name, t in (("cf", tc), ("wg", tw)):
        r = sorted(t.pop.items(), key=lambda kv: (-kv[1], kv[0]))[:32]
        out[name] = [(int(l), int(e), float(v)) for (l, e), v in r]
    os.makedirs(run_dir(key), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def _cond(cid, mask, scope, phase, pos="all", mode="reroute", **meta) -> dict:
    return dict(cid=cid, mask=tuple(sorted((int(l), int(e)) for l, e in mask)), scope=scope, phase=phase, pos=pos, mode=mode, meta=meta)


def pop_sets(key: str) -> dict:
    """{'wg': {k: [(l, e), ...]}, 'cf': {...}} top-k of the population ranking."""
    pr = population_ranking(key)
    return {t: {k: [(l, e) for l, e, _ in pr[t][:k]] for k in SET_SIZES} for t in ("wg", "cf")}


def base_route_final(key: str):
    """(items, route_idx [L, n, k], n_experts) of the baseline rows (final position, every layer), all full-phase passes."""
    d = run_dir(key)
    fs = sorted(f for f in os.listdir(d) if f.startswith("ko_basefinal_full_") and f.endswith(".npz"))
    assert fs, "no baseline routing yet (run the full phase first)"
    zs = [np.load(os.path.join(d, f)) for f in fs]
    items = np.concatenate([z["items"] for z in zs])
    ri = np.concatenate([z["route_idx"] for z in zs], axis=1)
    o = np.argsort(items)
    return items[o], ri[:, o], int(zs[0]["n_experts"])


def final_freq(key: str, task: str, scope_items: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Final-position routing frequency per (layer, expert) on a task's evaluation items in the baseline run
    (WinoGrande: own margin pool; CounterFact: the pool), from the full phase's <run>/ko_basefinal_*.npz. Columns layer,
    expert, freq."""
    items, ri, E = base_route_final(key)
    it = build_items(key)
    sel = it.set_index("item").loc[items]
    keep = (sel.task.values == task) & ((task != "wg") | sel.in_margin.values.astype(bool))
    ri = ri[:, keep]  # [L, n, k]
    L, n, k = ri.shape
    cnt = np.zeros((L, E))
    for l in range(L):
        cnt[l] = np.bincount(ri[l].reshape(-1).astype(np.int64), minlength=E)
    rows = [(l, e, cnt[l, e] / n) for l in range(L) for e in range(E)]
    return pd.DataFrame(rows, columns=["layer", "expert", "freq"])


def control_sets(key: str) -> dict:
    """Same-layer controls per target and frequency-matched random sets per (task, k); cached in <run>/controls.json."""
    path = os.path.join(run_dir(key), "controls.json")
    if os.path.exists(path):
        return json.load(open(path))
    cfg = KO[key]
    targets = all_targets(key)
    tset = set(targets)
    ps = pop_sets(key)
    out = {"same_layer": {}, "random": {}, "note": "same_layer: per target (layer, expert) the control experts of its layer; "
           "random: per task and set size, 5 sets matched member-wise on final-position routing frequency within the member's layer"}
    from .arch import load_spec
    spec, _ = load_spec(MODELS[cfg["model"]]["repo"])
    for task in ("wg", "cf"):
        fq = final_freq(key, task)
        fd = {(int(r.layer), int(r.expert)): float(r.freq) for r in fq.itertuples()}
        for (l, e) in cfg["targets"][task]:
            if spec.n_experts <= 8:
                ctl = [x for x in range(spec.n_experts) if (l, x) not in tset]
            else:
                cand = sorted([x for x in range(spec.n_experts) if (l, x) not in tset], key=lambda x: (-fd[(l, x)], x))
                ctl = cand[: cfg["n_same"]]
            out["same_layer"][le(l, e)] = {"task": task, "layer": l, "controls": ctl, "freq_target": fd[(l, e)],
                                           "freq_controls": [fd[(l, x)] for x in ctl]}
        excl = tset | set(ps[task][10])
        for k in SET_SIZES:
            members = ps[task][k]
            sets = []
            for r in range(N_RAND):
                rng = random.Random(9000 + 100 * k + r + (0 if task == "wg" else 50000))
                used = set()
                chosen = []
                for (l, e) in members:
                    cand = [x for x in range(spec.n_experts) if (l, x) not in excl and (l, x) not in used]
                    cand = sorted(cand, key=lambda x: (abs(fd[(l, x)] - fd[(l, e)]), x))[: cfg["rand_near"]]
                    pick = rng.choice(cand)
                    used.add((l, pick))
                    chosen.append((l, pick))
                sets.append(sorted(chosen))
            out["random"][f"{task}:k{k}"] = {"members": members, "member_freq": [fd[m] for m in members],
                                              "sets": sets, "set_freq": [[fd[m] for m in s_] for s_ in sets]}
    json.dump(out, open(path, "w"), indent=1)
    return out


def conditions(key: str, phase: str) -> list:
    """Condition list of a phase (see module docstring)."""
    cfg = KO[key]
    targets = all_targets(key)
    ps = pop_sets(key)
    tnames = {le(l, e) for l, e in targets}
    out = []
    if phase == "full":
        out.append(_cond("base", (), "full", phase))
        for (l, e) in targets:
            out.append(_cond(f"tgt:{le(l, e)}", [(l, e)], "full", phase, kind="target"))
        for task in ("wg", "cf"):
            for k in SET_SIZES:
                s_ = ps[task][k]
                if k == 1 and le(*s_[0]) in tnames:
                    continue  # top-1 = a target: reused in analysis
                out.append(_cond(f"pop:{task}:k{k}", s_, "full", phase, kind="pop", task=task, k=k))
    elif phase == "random":
        cs = control_sets(key)
        out.append(_cond("null", (), "sub", phase, kind="null"))  # baseline re-run in another pass composition (noise floor)
        for task in ("wg", "cf"):
            for k in SET_SIZES:
                for r, s_ in enumerate(cs["random"][f"{task}:k{k}"]["sets"]):
                    out.append(_cond(f"rand:{task}:k{k}:r{r}", s_, "sub", phase, kind="rand", task=task, k=k))
    elif phase == "samelayer":
        cs = control_sets(key)
        seen = set()
        for tname, v in cs["same_layer"].items():
            for x in v["controls"]:
                c = le(v["layer"], x)
                if c in seen:
                    continue
                seen.add(c)
                out.append(_cond(f"same:{c}", [(v["layer"], x)], "sub", phase, kind="same", layer=v["layer"]))
    elif phase == "final":  # (sub scope: lowest priority; a final-only mask cannot change the wiki NLL -> no wiki rows)
        for (l, e) in targets:
            out.append(_cond(f"fin:{le(l, e)}", [(l, e)], "sub_nowiki", phase, pos="final", kind="target_final"))
        for task in ("wg", "cf"):
            for k in SET_SIZES[1:]:
                out.append(_cond(f"fin:pop:{task}:k{k}", ps[task][k], "sub_nowiki", phase, pos="final", kind="pop_final", task=task, k=k))
        for (l, e) in targets:
            out.append(_cond(f"zero:{le(l, e)}", [(l, e)], "sub", phase, mode="zero", kind="target_zero"))
    else:
        raise ValueError(phase)
    return out


def mask_str(mask) -> str:
    return "+".join(le(l, e) for l, e in mask) if mask else "none"


def cond_sig(c: dict) -> str:
    """Intervention signature (identical signatures give identical rows, whatever the condition id)."""
    if not c["mask"]:
        return "none|all|null" if c.get("meta", {}).get("kind") == "null" else "none|all|reroute"
    return f"{mask_str(c['mask'])}|{c['pos']}|{c['mode']}"


def parse_sig(sig: str) -> tuple:
    m, pos, mode = sig.split("|")
    mask = () if m == "none" else tuple(parse_le(x) for x in m.split("+"))
    return mask, pos, mode


def plan_passes(key: str, it: pd.DataFrame, conds: list, done: Optional[set] = None, budget: Optional[int] = None,
                max_rows: Optional[int] = None, wiki_budget: Optional[int] = None) -> list:
    """Deterministic pass plan: list of dicts {name, wiki (bool), base (bool), rows: [(sig, item)]}. Rows are unique per
    (intervention signature, item); rows already in `done` (computed in earlier phases) are skipped. Short prompts are
    sorted by length (padded tokens <= budget, rows <= max_rows, pass sizes balanced); wiki windows get their own passes."""
    cfg = KO[key]
    budget = budget or cfg["budget"]
    max_rows = max_rows or cfg["max_rows"]
    wiki_budget = wiki_budget or cfg["wiki_budget"]
    done = done or set()
    T = it.set_index("item")["T"]
    task = it.set_index("item").task
    passes = []
    groups, seen = {}, set()
    for c in conds:
        sig = cond_sig(c)
        items = it.item.values[scope_mask(it, c["scope"])]
        g = "main"  # (baseline rows share the passes; their routing diagnostics are taken from the pass-wide records)
        for x in items:
            r = (sig, int(x))
            if r in seen or r in done:
                continue
            seen.add(r)
            groups.setdefault((g, bool(task[x] == "wiki")), []).append(r)
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

    for (g, wk) in sorted(groups, key=lambda z: (z[0] != "base", z[1])):
        rows = sorted(groups[(g, wk)], key=lambda r: (int(T[r[1]]), r[0], r[1]))
        bud = wiki_budget if wk else budget
        n_min = len(fill(rows, bud))
        lo, hi = bud // 4, bud  # balance: the smallest budget that still needs only n_min passes
        while hi - lo > 1000:
            mid = (lo + hi) // 2
            if len(fill(rows, mid)) <= n_min:
                hi = mid
            else:
                lo = mid
        for k, cur in enumerate(fill(rows, hi)):
            passes.append({"name": f"{g}_{'wiki' if wk else 'short'}_p{k:03d}", "wiki": wk, "base": g == "base", "rows": cur})
    return passes


def wiki_nll_slice(key: str) -> slice:
    """Token-logprob positions predicting content tokens 1..126 (Qwen3: no BOS; Mixtral: BOS at position 0)."""
    return slice(1, WIKI_WINDOW) if KO[key]["bos"] is not None else slice(0, WIKI_WINDOW - 1)


# ------------------------------------------------------------------------------------------------------------------
# analysis helpers
# ------------------------------------------------------------------------------------------------------------------
def load_rows(key: str) -> pd.DataFrame:
    d = run_dir(key)
    fs = sorted(f for f in os.listdir(d) if f.startswith("ko_rows_") and f.endswith(".parquet"))
    return pd.concat([pd.read_parquet(os.path.join(d, f)) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def boot_idx(n_units: int, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_units, size=(n_boot, n_units))


def unit_sums(df: pd.DataFrame, cols: list, unit: str = "unit") -> pd.DataFrame:
    """Per bootstrap unit: sums of the columns and the item count (ratio-of-means bootstrap over units)."""
    g = df.groupby(unit)
    out = g[cols].sum()
    out["_n"] = g.size()
    return out


def ratio_boot(us: pd.DataFrame, num: str, den: str, n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """(estimate, lo, hi, boot array) of sum(num) / sum(den) with a bootstrap over the rows of us (units)."""
    a, b = us[num].values.astype(np.float64), us[den].values.astype(np.float64)
    bi = boot_idx(len(a), n_boot, seed)
    r = a[bi].sum(1) / b[bi].sum(1)
    return float(a.sum() / b.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), r


def mean_boot(us: pd.DataFrame, col: str, n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """item-weighted mean of col via unit sums: sum(col) / sum(_n)."""
    return ratio_boot(us, col, "_n", n_boot, seed)
