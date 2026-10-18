"""ext14 Phase 6 FRAME PILOT for the E1-family attribute -> entity cells (CPU builds + analysis; agent p6-data; coordinator request
2026-10-16 after the Qwen3 E1-a2e scan: objects at chance in the current cloze, first-mention bias ~ 1 logit).

Items: 200 core-776 twins (the same for every frame and both tokenizer families), both statement orders, E1 evidence; association-
neutral twins first (Phase-5 probes: |v1|, |v2| < 1 for Qwen3, |v2| < 1 for Mixtral, both triggers), then the twin rank.
Frames (STR = the trigger in the final clause; the blank clause and its distance to the final token are kept in P0-P3):
  P0 (current) 'Fill in the blank: {A} and {B}, so the _ was X.\\nAnswer: the'
  P1           'Fill in the blank: {A}. {B}. The _ was X.\\nAnswer: the'
  P2           'Fill in the blank: {A} and {B}. The _ was X.\\nAnswer: the'
  P3           '{A} and {B}. Fill in the blank: The _ was X.\\nAnswer: the'
  P4 (no wrapper, behaviour reference) '{A} and {B}. The one that was X was the'  (names: 'The one who was X was'; plural copula: 'The ones that were X were the')
  --stage build    -> data/dial/build/pilot_frames_<fam>.parquet (scanned by scripts/ext14_scan.py <proto> --part cells --subset all --cells pilot_frames)
  --stage analyze  -> data/dial/qa/frame_pilot.md, data/dial/qa/frame_pilot.json, PROGRESS line "[p6-data] FRAME PILOT ..."
Metrics per frame x model x names / objects (all items and association-neutral items): margin both ways (pair), correct in both
directions (pair), order gap (pair margin rate order 1 - order 2), first-mention bias = mean logit(first-mentioned) -
logit(second-mentioned) over all prompts (answers are balanced between first and second, so 0 = no bias).
Decision rule (coordinator, pre-registered): the frame among P0-P3 with the highest neutral-objects correct-both rate, provided its
order gap <= 10 pp; if no frame exceeds 60 % neutral-objects correct-both in a model, keep P0 for that model.
Usage: python scripts/ext14_build_pilot.py --stage build|analyze
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
from moetrace import ext14_data as X
from moetrace import ext13_p5 as P5
from moetrace.models import RESULTS

FRAMES = ["P0", "P1", "P2", "P3", "P4"]
N_ITEMS = 200
QA = f"{X.DIAL}/qa"


def cap_bc(it):
    bc = it.bc_text()
    return bc[0].upper() + bc[1:] if bc[:1].isalpha() else bc


def frame_texts(it, frame, order, trig):
    """Text for one prompt (question trigger trig) in a frame; E1 statements with the twin binding in `order`."""
    bind = {it.ent_a: it.trig_a, it.ent_b: it.trig_b}
    t = X.Text()
    head = X.F1_HEAD if frame in ("P0", "P1", "P2") else ""
    t.add(head)
    if frame == "P0":
        X.evidence(t, it, order, bind)  # ', so '
    elif frame == "P1":
        X.evidence(t, it, order, bind, sep=". ")
        # second statement as its own sentence: rebuild instead
        t = X.Text().add(head)
        (e1, w1), (e2, w2) = X._ents_in_order(it, order, bind)
        it.stmt(t, e1, w1, *X._lab(it, e1), cap=True)
        t.add(". ")
        it.stmt(t, e2, w2, *X._lab(it, e2), cap=True)
        t.add(". ")
    else:
        X.evidence(t, it, order, bind, sep=". ")
    if frame == "P3":
        t.add("Fill in the blank: ")
    if frame == "P4":
        many = it.cop in ("were", "are")
        rel = "who" if it.names else "that"
        one = "The ones" if many else "The one"
        t.add(f"{one} {rel} {it.cop}" + (f" {it.adv}" if it.adv else "") + " ").add(trig, "q").add(f" {it.cop}" + ("" if it.names else " the"))
        return t
    bc = it.bc_text() if frame == "P0" else cap_bc(it)
    i = bc.index("_")
    t.add(bc[:i]).add("_", "blank").add(bc[i + 1:] + " ").add(trig, "q").add(".")
    t.add("\nAnswer:" + (f" {it.det}" if it.det else ""))
    return t


def neutral_flags(items):
    out = {}
    for proto, R in (("qwen3", ["v1", "v2"]), ("mixtral_bos", ["v2"])):
        pv = pd.read_parquet(f"{RESULTS}/wino_ref_{proto}/p5_probe_values.parquet").set_index("key")
        f = {}
        for it in items:
            k = X.probe_key(it.names, it.trig_a.lower(), it.trig_b.lower(), it.ent_a, it.ent_b)
            if k in pv.index:
                r = pv.loc[k]
                f[it.src_pair_id] = P5.assoc_stratum({"v1": (r.v1_a, r.v1_b), "v2": (r.v2_a, r.v2_b)}, R)
            else:
                f[it.src_pair_id] = "no_probe"
        out[proto] = f
    return out


def stage_build():
    items, _ = X.dial_items()
    items = [it for it in items if it.core776]
    toks = {fam: X.family_tokenizer(fam) for fam in ("qwen", "mistral")}
    split = json.load(open(f"{X.DIAL}/twin_split.json"))["twins"]
    neu = neutral_flags(items)
    rows = {fam: [] for fam in toks}
    ok_items = []
    for it in sorted(items, key=lambda i: split[i.src_pair_id]["rank"]):
        got = {fam: [] for fam in toks}
        good = True
        for fam, (tok, st) in toks.items():
            for fr in FRAMES:
                for o in (1, 2):
                    ta, tb = frame_texts(it, fr, o, it.trig_a), frame_texts(it, fr, o, it.trig_b)
                    p, w = X.make_pair(tok, st, ta, tb, it.ent_a, it.ent_b)
                    if p is None:
                        good = False
                        break
                    pos = p.pop("positions")
                    got[fam].append({"pair_id": f"{it.src_pair_id}|{fr}|o{o}", "frame": fr, "src_pair_id": it.src_pair_id, "order": o,
                                     "twin_idx": it.twin_idx, "pair_idx": 2 * it.twin_idx + o - 1, "names": it.names, "source": "wino",
                                     "core776": True, "ans_a": it.ent_a, "ans_b": it.ent_b, "word_a": it.trig_a, "word_b": it.trig_b,
                                     "answer_a": it.ent_a, "answer_b": it.ent_b, "adv": it.adv, "rank": split[it.src_pair_id]["rank"],
                                     "assoc_qwen3": neu["qwen3"][it.src_pair_id], "assoc_mixtral_bos": neu["mixtral_bos"][it.src_pair_id],
                                     **p, **X.pos_columns(pos)})
                if not good:
                    break
            if not good:
                break
        if good:
            ok_items.append((it, got))
    # neutral in both models first, then rank
    ok_items.sort(key=lambda x: (not (neu["qwen3"][x[0].src_pair_id] == "neutral" and neu["mixtral_bos"][x[0].src_pair_id] == "neutral"),
                                 split[x[0].src_pair_id]["rank"]))
    sel = ok_items[:N_ITEMS]
    for fam in toks:
        df = pd.DataFrame([r for _, g in sel for r in g[fam]])
        df.to_parquet(f"{X.BUILD}/pilot_frames_{fam}.parquet", index=False)
        print(fam, "pilot pairs", len(df), "items", df.src_pair_id.nunique(), "names", int(df.drop_duplicates("src_pair_id").names.sum()),
              "neutral qwen3", int((df.drop_duplicates("src_pair_id").assoc_qwen3 == "neutral").sum()),
              "neutral mixtral", int((df.drop_duplicates("src_pair_id").assoc_mixtral_bos == "neutral").sum()), flush=True)
    print("eligible core twins (all five frames, both families):", len(ok_items))
    ex = sel[0][1]["qwen"]
    for r in ex:
        if r["order"] == 1:
            print(r["frame"], repr(r["prompt_a"]), "->", r["answer_a"])


def metrics(d):
    if len(d) == 0:
        return None
    fm = np.where(d.order == 1, 1.0, -1.0)
    bias = float(np.mean(np.concatenate([fm * d.d_clean_a.to_numpy(), fm * d.d_clean_b.to_numpy()])))
    mo = d.groupby("order").margin.mean()
    return {"pairs": int(len(d)), "items": int(d.src_pair_id.nunique()), "margin": float(d.margin.mean()), "correct_both": float(d.correct_both.mean()),
            "order_gap": float(mo.get(1, np.nan) - mo.get(2, np.nan)), "first_mention_bias": bias,
            "items_both_orders_margin": float(d.groupby("src_pair_id").margin.all().mean()), "drop_median": float(d.drop_str.median())}


def stage_analyze():
    res, lines = {}, []
    for proto in ("qwen3", "mixtral_bos"):
        p = f"{RESULTS}/dial_scan_{proto}/scan_pilot_frames_all.parquet"
        if not os.path.exists(p):
            print("missing", p)
            continue
        s = pd.read_parquet(p)
        s["neutral"] = s[f"assoc_{proto}"] == "neutral"
        for fr in FRAMES:
            for grp, sub in (("objects", s[~s.names]), ("names", s[s.names]), ("all", s)):
                for nset, ss in (("all", sub), ("neutral", sub[sub.neutral])):
                    res[f"{proto}|{fr}|{grp}|{nset}"] = metrics(ss[ss.frame == fr])
    json.dump(res, open(f"{QA}/frame_pilot.json", "w"), indent=1)
    hdr = "| Model | Frame | Stratum | Items | Margin both ways | Correct both directions | Items both orders margin | Order gap (o1 - o2) | First-mention bias | Median drop |"
    lines = ["# FRAME PILOT (p6-data, E1 evidence, attribute -> entity; behaviour only)", "",
             "200 core twins, the same for every frame and both models, both statement orders (pairs = items x 2). Neutral = Phase-5 association "
             "probes |v| < 1 for both triggers (Qwen3 v1 and v2, Mixtral v2). Frames: P0 'Fill in the blank: {A} and {B}, so the _ was X.\\nAnswer: the'; "
             "P1 'Fill in the blank: {A}. {B}. The _ was X.\\nAnswer: the'; P2 'Fill in the blank: {A} and {B}. The _ was X.\\nAnswer: the'; "
             "P3 '{A} and {B}. Fill in the blank: The _ was X.\\nAnswer: the'; P4 '{A} and {B}. The one that was X was the' (no wrapper, reference). "
             "First-mention bias = mean logit(first-mentioned entity) - logit(second) over all prompts (answers balanced).", "", hdr,
             "|" + "---|" * hdr.count("|")[:-1] if False else "|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in res.items():
        if m is None:
            continue
        proto, fr, grp, nset = k.split("|")
        lines.append(f"| {proto} | {fr} | {grp}{' (neutral)' if nset == 'neutral' else ''} | {m['items']} | {m['margin']:.2f} | {m['correct_both']:.2f} | "
                     f"{m['items_both_orders_margin']:.2f} | {m['order_gap']:+.2f} | {m['first_mention_bias']:+.2f} | {m['drop_median']:.2f} |")
    # decision rule
    dec = {}
    for proto in ("qwen3", "mixtral_bos"):
        cand = []
        for fr in ("P0", "P1", "P2", "P3"):
            m = res.get(f"{proto}|{fr}|objects|neutral")
            if m:
                cand.append((m["correct_both"], fr, abs(m["order_gap"]), m))
        ok = [c for c in cand if c[2] <= 0.10]
        best = max(ok) if ok else None
        if best and best[0] > 0.60:
            dec[proto] = {"frame": best[1], "neutral_objects_correct_both": best[0], "order_gap": best[2]}
        else:
            dec[proto] = {"frame": "P0", "note": "no frame among P0-P3 above 60 % neutral-objects correct-both with order gap <= 10 pp; keep P0 "
                                                   "(behavioural finding: explicit look-up fails in the cloze)",
                          "best": {"frame": best[1], "correct_both": best[0], "order_gap": best[2]} if best else None}
    lines += ["", "## Decision rule (coordinator, pre-registered)", "",
              "Highest neutral-objects correct-both among P0-P3 with |order gap| <= 10 pp; if none > 60 % in a model, keep P0.", ""]
    for proto, d in dec.items():
        lines.append(f"- {proto}: {json.dumps(d)}")
    open(f"{QA}/frame_pilot.md", "w").write("\n".join(lines) + "\n")
    json.dump({"metrics": res, "decision": dec}, open(f"{QA}/frame_pilot.json", "w"), indent=1)
    summ = []
    for proto in ("qwen3", "mixtral_bos"):
        parts = []
        for fr in FRAMES:
            m = res.get(f"{proto}|{fr}|objects|neutral")
            if m:
                parts.append(f"{fr} {m['correct_both']:.2f}/{m['margin']:.2f}/gap {m['order_gap']:+.2f}/bias {m['first_mention_bias']:+.1f}")
        summ.append(f"{proto} neutral objects correct-both/margin/order gap/first-mention bias: " + "; ".join(parts) + f" -> {dec[proto]['frame']}")
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [p6-data] FRAME PILOT " + " || ".join(summ) + " (table data/dial/qa/frame_pilot.md)"
    with open(f"{X.ROOT}/logs/PROGRESS.md", "a") as f:
        f.write(line + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "analyze"])
    a = ap.parse_args()
    if a.stage == "build":
        stage_build()
    else:
        stage_analyze()


if __name__ == "__main__":
    main()
