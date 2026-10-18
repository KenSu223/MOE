"""ext14 Phase 6 IND PILOT (agent p6-data; coordinator option (b) after REVIEW l_ind/mixtral_bos): behaviour-only format pilot for
l_ind, both models. 200 items (the same random unrelated single-token words for every frame and both tokenizer families: 4 key /
value pairs of 8 distinct words from the shared single-token list), both query orders; STR = the query key (k1 <-> k2), answer = its
value. Frames I0-I4: moetrace/ext14_data.ind_text.
  --stage build    -> data/dial/build/pilot_ind_<fam>.parquet (scan: scripts/ext14_scan.py <proto> --part cells --subset all --cells pilot_ind)
  --stage analyze  -> data/dial/qa/ind_pilot.md / .json, PROGRESS "[p6-data] IND PILOT ..."
Rule (coordinator): among I1-I3 the frame with the highest correct-both rate with |order gap| <= 10 pp and correct-both >= 0.8 in BOTH
models; if none reaches 0.8 in a model, the best I1-I3 frame anyway.
Usage: python scripts/ext14_build_indpilot.py --stage build|analyze
"""
import argparse, json, os, random, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
from moetrace import ext14_data as X
from moetrace.models import RESULTS

FRAMES = ["I0", "I1", "I2", "I3", "I4"]
QA = f"{X.DIAL}/qa"


def build():
    toks = {fam: X.family_tokenizer(fam) for fam in ("qwen", "mistral")}
    words = X.single_token_words(X.IND_WORDS, toks.values())
    keys = X.line_start_words(words, toks.values())
    print("values", len(words), "keys (line-start single tokens in both families)", len(keys), flush=True)
    rng = random.Random(9)
    rows = {f: [] for f in toks}
    n, tries, seen = 0, 0, set()
    while n < 200 and tries < 20000:
        tries += 1
        ks = rng.sample(keys, 4)
        vs = rng.sample([x for x in words if x not in ks], 4)
        w = [ks[0], vs[0], ks[1], vs[1], ks[2], vs[2], ks[3], vs[3]]
        pairs = [(w[0], w[1]), (w[2], w[3]), (w[4], w[5]), (w[6], w[7])]
        key = tuple(w)
        if key in seen:
            continue
        seen.add(key)
        got = {f: [] for f in toks}
        ok = True
        for fam, (tok, st) in toks.items():
            for fr in FRAMES:
                for o in (1, 2):
                    ta, tb = X.ind_text(fr, pairs, o, pairs[0][0]), X.ind_text(fr, pairs, o, pairs[1][0])
                    p, why = X.make_pair(tok, st, ta, tb, pairs[0][1], pairs[1][1], allow_final=(fr == "I3"))
                    if p is None:
                        ok = False
                        break
                    pos = p.pop("positions")
                    got[fam].append({"pair_id": f"indp|{'|'.join(w)}|{fr}|o{o}", "frame": fr, "src_pair_id": f"indp|{'|'.join(w)}", "order": o,
                                     "pair_idx": 2 * n + o - 1, "names": False, "source": "synthetic", "answer_a": pairs[0][1],
                                     "answer_b": pairs[1][1], "word_a": pairs[0][0], "word_b": pairs[1][0], **p, **X.pos_columns(pos)})
                if not ok:
                    break
            if not ok:
                break
        if ok:
            for f in toks:
                rows[f] += got[f]
            n += 1
    for f in toks:
        df = pd.DataFrame(rows[f])
        df.to_parquet(f"{X.BUILD}/pilot_ind_{f}.parquet", index=False)
        print(f, "items", df.src_pair_id.nunique(), "pairs", len(df), flush=True)
    ex = pd.DataFrame(rows["qwen"])
    for fr in FRAMES:
        r = ex[(ex.frame == fr) & (ex.order == 1)].iloc[0]
        print(fr, repr(r.prompt_a), "->", r.answer_a, "|", repr(r.prompt_b), "->", r.answer_b)


def metrics(d):
    fm = np.where(d.order == 1, 1.0, -1.0)
    mo = d.groupby("order").margin.mean()
    return {"pairs": int(len(d)), "items": int(d.src_pair_id.nunique()), "margin": float(d.margin.mean()), "correct_both": float(d.correct_both.mean()),
            "items_both_orders_margin": float(d.groupby("src_pair_id").margin.all().mean()), "order_gap": float(mo.get(1, np.nan) - mo.get(2, np.nan)),
            "first_mention_bias": float(np.mean(np.concatenate([fm * d.d_clean_a.to_numpy(), fm * d.d_clean_b.to_numpy()]))),
            "drop_median": float(d.drop_str.median())}


def analyze():
    res = {}
    for proto in ("qwen3", "mixtral_bos"):
        s = pd.read_parquet(f"{RESULTS}/dial_scan_{proto}/scan_pilot_ind_all.parquet")
        for fr in FRAMES:
            res[f"{proto}|{fr}"] = metrics(s[s.frame == fr])
    lines = ["# IND PILOT (p6-data; l_ind formats, behaviour only)", "",
             "200 items (4 key / value pairs of 8 distinct random words, single tokens in both families; the same for every frame and both "
             "models), both query orders (pairs = items x 2); STR = the query key (k1 <-> k2). I0 'Pairs: a = b, c = d. a ='; I1 'a: b\\\\nc: d\\\\na:'; "
             "I2 'a: b\\\\nc: d\\\\ne: f\\\\ng: h\\\\na:'; I3 'List: a b c d e f a' (behaviour reference only: the query key is the final token = STR at the readout position, which R2 forbids "
             "for patching); I1 / I2 start with a newline so that every key, also the first, is the same line-start token; I4 'The code for a is b. The code for c is d. The code for a is' "
             "(reference). First-mention bias = mean logit(value of the first-listed key) - logit(value of the second) over all prompts.", "",
             "| Model | Frame | Margin both ways | Correct both directions | Items both orders margin | Order gap (o1 - o2) | First-mention bias | Median drop |",
             "|---|---|---|---|---|---|---|---|"]
    for k, m in res.items():
        proto, fr = k.split("|")
        lines.append(f"| {proto} | {fr} | {m['margin']:.2f} | {m['correct_both']:.2f} | {m['items_both_orders_margin']:.2f} | {m['order_gap']:+.2f} | "
                     f"{m['first_mention_bias']:+.2f} | {m['drop_median']:.2f} |")
    cand = []
    for fr in ("I1", "I2"):  # I3 is not eligible for tier 1 (R2: STR = final token)
        q, m = res[f"qwen3|{fr}"], res[f"mixtral_bos|{fr}"]
        ok = abs(q["order_gap"]) <= 0.10 and abs(m["order_gap"]) <= 0.10 and q["correct_both"] >= 0.8 and m["correct_both"] >= 0.8
        cand.append((ok, min(q["correct_both"], m["correct_both"]), q["correct_both"] + m["correct_both"], fr))
    good = [c for c in cand if c[0]]
    if good:
        best = max(good, key=lambda c: (c[2], c[1]))
        dec = {"frame": best[3], "rule": "highest correct-both (sum over models) among I1-I3 with |gap| <= 0.10 and >= 0.8 in both models"}
    else:
        gap_ok = [c for c in cand if abs(res[f"qwen3|{c[3]}"]["order_gap"]) <= 0.10 and abs(res[f"mixtral_bos|{c[3]}"]["order_gap"]) <= 0.10] or cand
        best = max(gap_ok, key=lambda c: (c[1], c[2]))
        dec = {"frame": best[3], "rule": "no I1-I3 frame reaches 0.8 in both models: best I1-I3 frame anyway (highest minimum over models, |gap| <= 0.10 where possible)"}
    lines += ["", f"Decision (pre-set rule): **{dec['frame']}** ({dec['rule']})."]
    open(f"{QA}/ind_pilot.md", "w").write("\n".join(lines) + "\n")
    json.dump({"metrics": res, "decision": dec}, open(f"{QA}/ind_pilot.json", "w"), indent=1)
    parts = []
    for proto in ("qwen3", "mixtral_bos"):
        parts.append(proto + " correct-both/margin/gap: " + "; ".join(
            f"{fr} {res[f'{proto}|{fr}']['correct_both']:.2f}/{res[f'{proto}|{fr}']['margin']:.2f}/{res[f'{proto}|{fr}']['order_gap']:+.2f}" for fr in FRAMES))
    with open(f"{X.ROOT}/logs/PROGRESS.md", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [p6-data] IND PILOT " + " || ".join(parts) +
                f" -> rule gives {dec['frame']} (data/dial/qa/ind_pilot.md); l_ind is rebuilt in it after '[coordinator] IND <frame>' or in 30 min\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "analyze"])
    a = ap.parse_args()
    build() if a.stage == "build" else analyze()
