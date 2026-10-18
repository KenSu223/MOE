"""ext14 Phase 6 Dial A / E0 builds (CPU, agent p6-data).

  --stage specs  association-probe specs (Phase-5 v1 / v2 prompts, moetrace/ext13_p5.probe_specs) per tokenizer family for
                 (a) the controlled strong-prior items (data/dial/curated_dialA_nouns.json: noun pair x / y, attribute pair p / q,
                 designed prior x <-> p, y <-> q) and (b) the Dial-E WinoGrande items without Phase-5 probe values
                 -> data/dial/build/probe_specs_<fam>.parquet, data/dial/build/ctrl_items.json
  --stage build  after the probe scans (results/dial_scan_<proto>/probe_values.parquet): candidate items = controlled items +
                 WinoGrande items whose prior is coherent (the two triggers point to different entities) with strength >= 1 in a
                 model of the family; per family
                   dA_e1_a2e / dA_e1_e2a  E1 frame with the statements of the item's nominal binding (stmt = bind) and swapped
                                          (stmt = swap), both orders;
                   e0_a2e / e0_e2a        'Of the X and the Y, <blank clause>' with the nominal binding as answer orientation;
                 -> data/dial/build/<cell>_<fam>.parquet. scripts/ext14_casesets.py derives aagr_* / acon_* / e0_* per model from
                 the model's own prior (agree = statements match the prior; E0 orientation = the prior's binding).
Prior per model: Phase-5 rule R (Qwen3 v1 + v2 averaged, Mixtral v2); trigger support s_a = probe(trig_a) (> 0: trig_a -> ent_a),
s_b = probe(trig_b) (< 0: trig_b -> ent_b); coherent associative if s_a > 0 > s_b, reverse if s_a < 0 < s_b; strength = min(|s_a|, |s_b|).
Usage: python scripts/ext14_build_dialA.py --stage specs|build [--fams qwen,mistral]
"""
import argparse
import collections
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
from moetrace import ext14_data as X
from moetrace import ext13_p5 as P5
from moetrace.models import RESULTS

FAM_PROTOS = {"qwen": ["qwen3"], "mistral": ["mixtral_bos"]}
CTRL_OFFSET = 50000


def ctrl_items():
    d = json.load(open(f"{X.DIAL}/curated_dialA_nouns.json"))
    out, seen = [], set()
    for g in d["groups"]:
        p, q = g["attr"]
        for x, y in g["pairs"]:
            k = (x, y, p, q)
            if k in seen or " " in x or " " in y:
                continue
            seen.add(k)
            i = len(out)
            out.append(X.Item(f"ctrl|{x}|{y}|{p}|{q}", CTRL_OFFSET + i, "ctrl", False, "the", "was", "", x, y, p, q,
                              extra={"group": f"{p}/{q}"}))
    return out


def key_of(it):
    return X.probe_key(it.names, it.trig_a.lower(), it.trig_b.lower(), it.ent_a, it.ent_b)


def p5_values(proto):
    p = f"{RESULTS}/wino_ref_{proto}/p5_probe_values.parquet"
    if not os.path.exists(p):
        return {}
    v = pd.read_parquet(p)
    return {r.key: (r.v1_a, r.v1_b, r.v2_a, r.v2_b) for r in v.itertuples()}


def stage_specs(fams):
    items, _ = X.dial_items()
    ctrl = ctrl_items()
    json.dump([{"src_pair_id": it.src_pair_id, "twin_idx": it.twin_idx, "x": it.ent_a, "y": it.ent_b, "p": it.trig_a, "q": it.trig_b,
                "group": it.extra["group"]} for it in ctrl], open(f"{X.BUILD}/ctrl_items.json", "w"), indent=0)
    for fam in fams:
        tok, st = X.family_tokenizer(fam)
        have = [p5_values(pr) for pr in FAM_PROTOS[fam]]
        keys = collections.OrderedDict()
        for it in ctrl:
            keys[key_of(it)] = "ctrl"
        for it in items:
            k = key_of(it)
            if any(k not in h or not np.isfinite(h[k][2]) for h in have):
                keys.setdefault(k, "wino")
        rows, fail = [], collections.Counter()
        for k, src in keys.items():
            for v in ("v1", "v2"):
                sp = P5.probe_specs(tok, st, k, v)
                if sp is None:
                    fail[f"{src}:{v}"] += 1
                    continue
                for slot, (q, qi, ea, eb) in sp.items():
                    rows.append({"key": k, "src": src, "version": v, "slot": slot, "prompt": q, "ids": json.dumps(qi), "ea": ea, "eb": eb})
        df = pd.DataFrame(rows)
        df.to_parquet(f"{X.BUILD}/probe_specs_{fam}.parquet", index=False)
        print(fam, "probe keys", len(keys), "ctrl", sum(v == "ctrl" for v in keys.values()), "rows", len(df), "fail", dict(fail), flush=True)


def stage_specs_full(fams):
    """Stage 2 (models without Phase-5 probes): v1 / v2 probe specs for EVERY Dial-E item built in the family (union over the
    WinoGrande cells) and the controlled items -> data/dial/build/probe_specs_full_<fam>.parquet."""
    items, _ = X.dial_items()
    byid = {it.src_pair_id: it for it in items}
    for fam in fams:
        tok, st = X.family_tokenizer(fam)
        ids = set()
        for c in ("e1_a2e", "e1_e2a", "e1p_a2e", "e1p_e2a", "e2_a2e", "e2_e2a", "e3_a2e", "e3_e2a", "e3p3_a2e", "sc_agree_a2e", "rsite_e1_a2e"):
            p = f"{X.BUILD}/{c}_{fam}.parquet"
            if os.path.exists(p):
                ids |= set(pd.read_parquet(p, columns=["src_pair_id"]).src_pair_id)
        keys = collections.OrderedDict((key_of(it), "ctrl") for it in ctrl_items())
        for k in sorted(ids):
            if k in byid:
                keys.setdefault(key_of(byid[k]), "wino")
        rows, fail = [], collections.Counter()
        for k, src in keys.items():
            for v in ("v1", "v2"):
                sp = P5.probe_specs(tok, st, k, v)
                if sp is None:
                    fail[f"{src}:{v}"] += 1
                    continue
                for slot, (q, qi, ea, eb) in sp.items():
                    rows.append({"key": k, "src": src, "version": v, "slot": slot, "prompt": q, "ids": json.dumps(qi), "ea": ea, "eb": eb})
        df = pd.DataFrame(rows)
        df.to_parquet(f"{X.BUILD}/probe_specs_full_{fam}.parquet", index=False)
        print(fam, "full probe keys", len(keys), "rows", len(df), "fail", dict(fail), flush=True)


def prior_table(proto):
    """key -> (v1_a, v1_b, v2_a, v2_b) merged Phase-5 + Phase-6 probe values, and the prior (sign, strength) of the model."""
    vals = p5_values(proto)
    p6 = f"{RESULTS}/dial_scan_{proto}/probe_values.parquet"
    if os.path.exists(p6):
        for r in pd.read_parquet(p6).itertuples():
            new = tuple(getattr(r, c, np.nan) for c in ("v1_a", "v1_b", "v2_a", "v2_b"))
            old = vals.get(r.key)
            vals[r.key] = new if old is None else tuple(o if np.isfinite(o) else n for o, n in zip(old, new))
    return vals


def prior_of(vals, proto):
    """(sign, strength, s_a, s_b): sign +1 = trig_a -> ent_a / trig_b -> ent_b, -1 = reversed, 0 = incoherent / missing."""
    if vals is None:
        return 0, 0.0, np.nan, np.nan
    v1a, v1b, v2a, v2b = vals
    if proto == "qwen3":
        sa, sb = np.nanmean([v1a, v2a]), np.nanmean([v1b, v2b])
    else:
        sa, sb = v2a, v2b
    if not (np.isfinite(sa) and np.isfinite(sb)):
        return 0, 0.0, sa, sb
    if sa > 0 > sb:
        return 1, float(min(abs(sa), abs(sb))), sa, sb
    if sa < 0 < sb:
        return -1, float(min(abs(sa), abs(sb))), sa, sb
    return 0, 0.0, sa, sb


def stage_build(fams, min_strength, frames):
    items, _ = X.dial_items()
    ctrl = ctrl_items()
    sp = json.load(open(f"{X.DIAL}/twin_split.json"))["twins"]
    rng = random.Random(16)
    cids = sorted(it.src_pair_id for it in ctrl)
    rng.shuffle(cids)
    csplit = {k: ("discovery" if i < len(cids) // 2 else "validation") for i, k in enumerate(cids)}
    crank = {k: 100000 + i for i, k in enumerate(cids)}
    import importlib.util
    spec = importlib.util.spec_from_file_location("bd", "/home/ubuntu/MOE/scripts/ext14_build_dial.py")
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    for fam in fams:
        tok, st = X.family_tokenizer(fam)
        tables = {pr: prior_table(pr) for pr in FAM_PROTOS[fam]}
        cand = list(ctrl)
        for it in items:
            k = key_of(it)
            if any(prior_of(t.get(k), pr)[0] != 0 and prior_of(t.get(k), pr)[1] >= min_strength for pr, t in tables.items()):
                cand.append(it)
        print(fam, "candidates: ctrl", len(ctrl), "wino", len(cand) - len(ctrl), flush=True)
        funnel = {}
        for cell in ("dA_e1_a2e", "dA_e1_e2a", "e0_a2e", "e0_e2a"):
            direction = cell.split("_")[-1]
            level = "e0" if cell.startswith("e0") else "e1"
            rows, why = [], collections.Counter()
            for it in cand:
                variants = ["bind", "swap"] if level == "e1" else ["bind"]
                for var in variants:
                    got = []
                    for o in (1, 2):
                        r = X.build_level(it, level, direction, o, conflict=(var == "swap"), frame=frames[fam] if direction == "a2e" else "P0")
                        if isinstance(r, str):
                            why[r] += 1
                            break
                        ta, tb, aa, ab, n = r
                        p, w = X.make_pair(tok, st, ta, tb, aa, ab, n)
                        why[w] += 1
                        if p is None:
                            break
                        pos = p.pop("positions")
                        row = {"pair_id": f"{it.src_pair_id}|{cell}|{var}|o{o}", "cell": cell, "level": level, "direction": direction, "order": o,
                               "stmt": var, "src_pair_id": it.src_pair_id, "twin_idx": it.twin_idx, "pair_idx": 2 * it.twin_idx + o - 1,
                               "source": it.source, "core776": it.core776, "names": it.names, "det": it.det, "cop": it.cop, "adv": it.adv,
                               "ans_a": it.ent_a, "ans_b": it.ent_b, "word_a": it.trig_a, "word_b": it.trig_b, "answer_a": aa, "answer_b": ab,
                               "group": it.extra.get("group", ""), "frame": frames[fam] if direction == "a2e" else "opt", **p, **X.pos_columns(pos), "norm_key": X.Item.norm_text(p["prompt_a"]),
                               "trig_key": "|".join(sorted((it.trig_a.lower(), it.trig_b.lower()))),
                               "antonym_pair": X.antonym_pair(it.trig_a, it.trig_b)}
                        if it.source == "wino":
                            chk = X.completeness(f"{level}_{direction}", it, row)
                            row["completeness"] = json.dumps(chk)
                            row["complete"] = all(chk.values()) if chk else True
                        else:
                            row["completeness"], row["complete"] = "{}", True
                        got.append(row)
                    if len(got) == 2 and all(g["complete"] for g in got):
                        rows.extend(got)
                    elif len(got) == 2:
                        why["item:incomplete_dropped"] += 1
            df = pd.DataFrame(rows)
            df["split"] = [csplit[k] if s == "ctrl" else sp[k]["split"] for k, s in zip(df.src_pair_id, df.source)]
            df["rank"] = [crank[k] if s == "ctrl" else sp[k]["rank"] for k, s in zip(df.src_pair_id, df.source)]
            df.to_parquet(f"{X.BUILD}/{cell}_{fam}.parquet", index=False)
            funnel[cell] = {"reasons": dict(sorted(why.items())), "pairs": int(len(df)), "items": int(df.src_pair_id.nunique()),
                            "items_ctrl": int(df[df.source == "ctrl"].src_pair_id.nunique())}
            print(fam, cell, json.dumps({k: v for k, v in funnel[cell].items() if k != "reasons"}), flush=True)
            if fam == "qwen":
                bd.samples_md(cell, df.assign(para_a=None, para_b=None))
        json.dump(funnel, open(f"{X.BUILD}/funnel_dialA_{fam}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["specs", "specs_full", "build"])
    ap.add_argument("--fams", default="qwen,mistral")
    ap.add_argument("--min-strength", type=float, default=1.0)
    ap.add_argument("--frame-qwen", default="P0")
    ap.add_argument("--frame-mistral", default="P0")
    a = ap.parse_args()
    os.makedirs(X.BUILD, exist_ok=True)
    if a.stage == "specs":
        stage_specs(a.fams.split(","))
    elif a.stage == "specs_full":
        stage_specs_full(a.fams.split(","))
    else:
        stage_build(a.fams.split(","), a.min_strength, {"qwen": a.frame_qwen, "mistral": a.frame_mistral})


if __name__ == "__main__":
    main()
