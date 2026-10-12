"""ext13 Phase 5 builds (CPU, agent p5-data; module moetrace/ext13_p5.py). Per model protocol (qwen3, mixtral_bos):

  probes   association-probe keys of every scanned attribute -> entity pair (F1, F2, F2R pair files), of the 776 Phase-3
           twins and of the verb templates (55 ent2attr seeds x 48 IOI name pairs); token specs of probe v2 for every key
           and of probe v1 where no earlier scan has it -> data/wino_ref/p5_probe_specs_<proto>.parquet,
           data/wino_ref/p5_probe_keys_<proto>.parquet
  vt_a2e   fill-in-the-blank (F1) format on the 55 ent2attr seeds, SAME name pairs as verbswap_ent2attr (ext13_templates.
           name_pairs, rng "ext13-verbs-<seed>"), variants orig / neutral / unrelated (ext13_wino_ref.build_ref_pair "F1")
  cp_vt    copy variant on the same template items (prefix "Of X and Y, one <frame> t1 and the other <frame> t2. ", both
           attribute orders; ext7_wino.build_pair on the prefixed twin)
  cp_nat   copy variant on the 776 Phase-3 twins (data/wino_str/case_sets.json pairs + shared_unused), both orders
  mem      the 55 seeds with their ORIGINAL WinoGrande names x orig / neutral / unrelated (Phase-3 builder)
Outputs data/wino_ref/p5build_<cond>_<proto>.parquet (pair contract + strata columns + token positions),
data/wino_ref/p5build_funnel.json, data/wino_ref/p5_cp_nat_review.tsv (prefix review sample).
Usage: python scripts/ext13_p5_build.py [--what probes,vt_a2e,cp_vt,cp_nat,mem] [--protos qwen3,mixtral_bos]
"""
import argparse
import collections
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
from moetrace import ext13_p5 as X
from moetrace.ext7_wino import NAME_RE, build_pair, load_twins
from moetrace.ext13_templates import fill, name_pairs
from moetrace.ext13_wino_ref import build_ref_pair, word_diff

D = "/home/ubuntu/MOE/data/wino_ref"
RES = "/home/ubuntu/MOE/results"
PROTOS = {"qwen3": ("qwen3", True), "mixtral_bos": ("mixtral", True)}
K = 48
VARIANTS = ("orig", "neutral", "unrelated")


def tokenizer(proto):
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    from moetrace.models import MODELS
    key, st = PROTOS[proto]
    return AutoTokenizer.from_pretrained(snapshot_dir(MODELS[key]["repo"])), st


def ent2attr_seeds():
    vt = json.load(open(f"{D}/verb_templates.json"))
    sd = pd.read_parquet(f"{D}/verb_seeds.parquet").set_index("seed")
    out = []
    for sid, spec in vt["seeds"].items():
        if spec["dir"] != "ent2attr":
            continue
        s = dict(sd.loc[sid]) | {"seed": sid}
        out.append((sid, s, spec))
    assert len(out) == 55, len(out)
    return out


def template_items():
    """(sid, seed, spec, j, x, y) for the 55 ent2attr seeds x 48 name pairs (exactly as scripts/ext13_verbs.py expand)."""
    for sid, s, spec in ent2attr_seeds():
        for j, (x, y) in enumerate(name_pairs(s, K, random.Random(f"ext13-verbs-{sid}"))):
            yield sid, s, spec, j, x, y


def edits_of(spec, variant):
    return spec.get(variant, []) if variant != "orig" else []


def phase3_twins():
    cs = json.load(open("/home/ubuntu/MOE/data/wino_str/case_sets.json"))
    twins, _ = load_twins("train_xl")
    byid = {t.pair_id: t for t in twins}
    inv = {v: k for k, v in cs["pair_idx_of"].items()}
    out = []
    for split, v in cs["pairs"].items():
        for i in v:
            out.append((byid[inv[i]], int(i), split))
    assert len(out) == 776
    return out


# ---------------------------------------------------------------------------------------------------------------
def build_probes(proto, funnel):
    tok, st = tokenizer(proto)
    keys = collections.OrderedDict()  # key -> set of sources

    def add(k, src):
        keys.setdefault(k, set()).add(src)
    known_v1 = {}
    for fmt in ("F1", "F2R", "F2"):
        p = pd.read_parquet(f"{D}/pairs_{fmt}_train_xl_{proto}.parquet", columns=["names", "word_a", "word_b", "ans_a", "ans_b"])
        for r in p.itertuples():
            add(X.probe_key(r.names, r.word_a, r.word_b, r.ans_a, r.ans_b), fmt)
        sp = f"{RES}/wino_ref_{proto}/scan_pairs_{fmt}.parquet"
        if os.path.exists(sp):
            s = pd.read_parquet(sp, columns=["names", "word_a", "word_b", "ans_a", "ans_b", "probe_a", "probe_b"])
            for r in s.itertuples():
                known_v1.setdefault(X.probe_key(r.names, r.word_a, r.word_b, r.ans_a, r.ans_b), (float(r.probe_a), float(r.probe_b), fmt))
    for t, _, _ in phase3_twins():
        wa, wb, ok = word_diff(t.sent_a, t.sent_b)
        names = bool(NAME_RE.match(t.opt1) and NAME_RE.match(t.opt2))
        if ok and wa and wb:
            add(X.probe_key(names, wa, wb, t.ans_a, t.ans_b), "phase3")
    for sid, s, spec, j, x, y in template_items():
        t = fill(s, x, y, [], "orig")
        wa, wb, ok = word_diff(t.sent_a, t.sent_b)
        add(X.probe_key(True, wa, wb, t.ans_a, t.ans_b), "templates")
    rows, krows, fail = [], [], collections.Counter()
    for k, src in keys.items():
        kv1 = known_v1.get(k)
        ent = {"key": k, "src": ",".join(sorted(src)), "v1_known_a": kv1[0] if kv1 else None, "v1_known_b": kv1[1] if kv1 else None,
               "v1_known_src": kv1[2] if kv1 else None}
        for v in ("v1", "v2"):
            if v == "v1" and kv1 is not None:
                ent["v1_spec"] = "known"
                continue
            sp = X.probe_specs(tok, st, k, v)
            if sp is None:
                fail[v] += 1
                ent[f"{v}_spec"] = "fail"
                continue
            ent[f"{v}_spec"] = "new"
            for slot, (q, qi, ea, eb) in sp.items():
                rows.append({"key": k, "version": v, "slot": slot, "prompt": q, "ids": json.dumps(qi), "ea": ea, "eb": eb})
        krows.append(ent)
    pd.DataFrame(rows).to_parquet(f"{D}/p5_probe_specs_{proto}.parquet", index=False)
    kd = pd.DataFrame(krows)
    kd.to_parquet(f"{D}/p5_probe_keys_{proto}.parquet", index=False)
    funnel.setdefault(proto, {})["probes"] = {"keys": len(kd), "rows": len(rows), "fail": dict(fail),
                                              "v1_known": int((kd.v1_spec == "known").sum()), "v1_new": int((kd.v1_spec == "new").sum()),
                                              "by_src": kd.src.value_counts().to_dict()}
    print(proto, "probes", json.dumps(funnel[proto]["probes"]))


def build_vt_a2e(proto, funnel):
    tok, st = tokenizer(proto)
    out, f = [], collections.Counter()
    for sid, s, spec, j, x, y in template_items():
        for v in VARIANTS:
            t = fill(s, x, y, edits_of(spec, v), v)
            if t is None:
                f[f"{v}:edit_not_found"] += 1
                continue
            p, why = build_ref_pair(t, "F1", tok, st, set())
            f[f"{v}:{why}"] += 1
            if p is None:
                continue
            r = p.to_row()
            r.update(seed=sid, variant=v, name_x=x, name_y=y, name_j=j, gender=s["gender"])
            r.update(X.positions_f1(tok, st, r))
            r["probe_key"] = X.probe_key(True, r["word_a"], r["word_b"], r["ans_a"], r["ans_b"])
            out.append(r)
    df = pd.DataFrame(out)
    df.to_parquet(f"{D}/p5build_vt_a2e_{proto}.parquet", index=False)
    funnel.setdefault(proto, {})["vt_a2e"] = dict(sorted(f.items())) | {"kept": len(df)}
    print(proto, "vt_a2e", len(df), df.groupby("variant").size().to_dict())


def build_cp_vt(proto, funnel):
    tok, st = tokenizer(proto)
    out, f = [], collections.Counter()
    for sid, s, spec, j, x, y in template_items():
        for v in VARIANTS:
            t = fill(s, x, y, edits_of(spec, v), v)
            if t is None:
                f[f"{v}:edit_not_found"] += 1
                continue
            for o in (1, 2):
                r, why = X.build_copy(t, o, tok, st)
                f[f"{v}:{why}"] += 1
                if r is None:
                    continue
                r.update(seed=sid, variant=v, name_x=x, name_y=y, name_j=j, gender=s["gender"], base_pair_id=t.pair_id)
                r["probe_key"] = X.probe_key(True, r["word_a"].lower(), r["word_b"].lower(), r["ans_a"], r["ans_b"])
                out.append(r)
    df = pd.DataFrame(out)
    df.to_parquet(f"{D}/p5build_cp_vt_{proto}.parquet", index=False)
    funnel.setdefault(proto, {})["cp_vt"] = dict(sorted(f.items())) | {"kept": len(df)}
    print(proto, "cp_vt", len(df), df.groupby("variant").size().to_dict(), "seeds", df.seed.nunique())


def build_cp_nat(proto, funnel):
    tok, st = tokenizer(proto)
    out, f = [], collections.Counter()
    for t, pidx, split in phase3_twins():
        for o in (1, 2):
            r, why = X.build_copy(t, o, tok, st)
            f[why] += 1
            if r is None:
                continue
            wa, wb, _ = word_diff(t.sent_a, t.sent_b)
            r.update(phase3_pair_idx=pidx, phase3_split=split, base_pair_id=t.pair_id,
                     probe_key=X.probe_key(r["names"], wa, wb, t.ans_a, t.ans_b))
            out.append(r)
    df = pd.DataFrame(out)
    df.to_parquet(f"{D}/p5build_cp_nat_{proto}.parquet", index=False)
    funnel.setdefault(proto, {})["cp_nat"] = dict(sorted(f.items())) | {"kept": len(df), "twins": int(df.base_pair_id.nunique())}
    print(proto, "cp_nat", len(df), "twins", df.base_pair_id.nunique(), dict(f))
    if proto == "qwen3":  # review sample (both models share the prefix text)
        smp = df[df.order == 1].sample(n=min(40, int((df.order == 1).sum())), random_state=0)
        smp[["pair_id", "names", "frame", "prefix", "prompt_a", "word_a", "word_b"]].to_csv(f"{D}/p5_cp_nat_review.tsv", sep="\t", index=False)


def build_mem(proto, funnel):
    tok, st = tokenizer(proto)
    twins, _ = load_twins("train_xl")
    byid = {t.pair_id: t for t in twins}
    out, f = [], collections.Counter()
    for sid, s, spec in ent2attr_seeds():
        tw = byid[sid]
        for v in VARIANTS:
            t = fill(s, tw.opt1, tw.opt2, edits_of(spec, v), v)
            if t is None:
                f[f"{v}:edit_not_found"] += 1
                continue
            p, why = build_pair(t, tok, st)
            f[f"{v}:{why}"] += 1
            if p is None:
                continue
            r = X.ensure_pair_row(p)
            r.update(seed=sid, variant=v, name_x=tw.opt1, name_y=tw.opt2, gender=s["gender"], original_names=True)
            out.append(r)
    df = pd.DataFrame(out)
    df.to_parquet(f"{D}/p5build_mem_{proto}.parquet", index=False)
    funnel.setdefault(proto, {})["mem"] = dict(sorted(f.items())) | {"kept": len(df)}
    print(proto, "mem", len(df), df.groupby("variant").size().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="probes,vt_a2e,cp_vt,cp_nat,mem")
    ap.add_argument("--protos", default="qwen3,mixtral_bos")
    a = ap.parse_args()
    fp = f"{D}/p5build_funnel.json"
    funnel = json.load(open(fp)) if os.path.exists(fp) else {}
    fns = {"probes": build_probes, "vt_a2e": build_vt_a2e, "cp_vt": build_cp_vt, "cp_nat": build_cp_nat, "mem": build_mem}
    for proto in a.protos.split(","):
        for w in a.what.split(","):
            fns[w](proto, funnel)
            json.dump(funnel, open(fp, "w"), indent=1)


if __name__ == "__main__":
    main()
