"""ext7-controls W8 step 1 (CPU): IOI STR pairs for Qwen3 and Mixtral (BOS) on identical items.

Templates, names, places and objects: Wang et al. (2023) via Easy-Transformer `easy_transformer/ioi_dataset.py`
(15 BABA templates + their ABBA versions; fetched 2026-10-04, copied into moetrace/ext7_controls.py). Words are kept
when they are ONE leading-space token in context and as a continuation under both tokenizers. Items (seed 0): template
uniform over the 30, IO and S two distinct names, C and D two further distinct names, place, object. Per item three
prompts (the final ' [A]' removed): clean, (i) S2 -> IO, (ii) S1 -> C and IO -> D. Kept when, under BOTH tokenizers,
every name is one token, (i) differs from clean only at S2 and (ii) only at S1 and IO (same length), and ' IO' and ' S'
are single-token continuations with the same ids after all three prompts.

Usage: python scripts/ext7_ioi_build.py [--n 1600] [--seed 0]
Outputs data/ioi/items.parquet (texts, names, char spans), data/ioi/pairs_{s2io,s1io}_{qwen3,mixtral_bos}.parquet
(contract format; pair_idx = item index, identical across models and corruptions), data/ioi/build.json
"""
import argparse, collections, json, os, random, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd
from moetrace import ext7_controls as C
from moetrace.data import _single_token_continuation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(C.IOI_DIR, exist_ok=True)
    toks = {p: (C.load_tok(k), st) for p, (k, st) in C.PROTOS.items()}
    names = C.single_token_words(toks, C.IOI_NAMES, "Then, Mary and {} went to the store")
    names = [n for n in names if n != "Mary"] + (["Mary"] if "Mary" in C.single_token_words(toks, ["Mary"], "Then, John and {} went to the store") else [])
    places = C.single_token_words(toks, C.IOI_PLACES, "Then, they went to the {} together")
    objects = C.single_token_words(toks, C.IOI_OBJECTS, "Then, John gave a {} to")
    print(f"single-token words under both tokenizers: names {len(names)}/{len(C.IOI_NAMES)}, places {places}, objects {objects}")
    rng = random.Random(args.seed)
    items, seen, why = [], set(), collections.Counter()
    tries = 0
    while len(items) < args.n and tries < 50 * args.n:
        tries += 1
        tt, ti, tpl = C.IOI_TEMPLATES[rng.randrange(len(C.IOI_TEMPLATES))]
        io, s, c, d = rng.sample(names, 4)
        place, obj = rng.choice(places), rng.choice(objects)
        key = (tt, ti, io, s, place, obj)
        if key in seen:
            why["duplicate"] += 1
            continue
        seen.add(key)
        pr = C.ioi_prompts(tpl, io, s, place, obj, c, d)
        rec = {"template_type": tt, "template_idx": ti, "template": tpl, "io": io, "s": s, "c": c, "d": d, "place": place, "obj": obj}
        ok = True
        for proto, (tok, st) in toks.items():
            enc, ids, pos = {}, {}, {}
            for k, (txt, spans) in pr.items():
                e = C._enc(tok, txt, st)
                enc[k], ids[k] = e, list(e["input_ids"])
                pos[k] = {lab: C._span_positions(e, a, b) for lab, (a, b) in spans.items()}
                if any(len(v) != 1 for v in pos[k].values()):
                    ok = False
                    why[f"{proto}:name_not_one_token"] += 1
                    break
            if not ok:
                break
            if not (len(ids["clean"]) == len(ids["s2io"]) == len(ids["s1io"])) or pos["clean"] != pos["s2io"] or pos["clean"] != pos["s1io"]:
                ok = False
                why[f"{proto}:length_or_position_mismatch"] += 1
                break
            P = {lab: v[0] for lab, v in pos["clean"].items()}
            d1 = [t for t in range(len(ids["clean"])) if ids["clean"][t] != ids["s2io"][t]]
            d2 = [t for t in range(len(ids["clean"])) if ids["clean"][t] != ids["s1io"][t]]
            if d1 != [P["S2"]] or d2 != sorted([P["S1"], P["IO"]]):
                ok = False
                why[f"{proto}:diff_outside_names"] += 1
                break
            trig = {}
            for nm, w in (("io", io), ("s", s)):
                t = {k: _single_token_continuation(tok, ids[k], pr[k][0], w, st) for k in pr}
                if None in t.values() or len(set(t.values())) != 1:
                    ok = False
                    why[f"{proto}:answer_not_single_token"] += 1
                    break
                trig[nm] = t["clean"]
            if not ok:
                break
            rec.update({f"{proto}_ids_clean": json.dumps(ids["clean"]), f"{proto}_ids_s2io": json.dumps(ids["s2io"]),
                        f"{proto}_ids_s1io": json.dumps(ids["s1io"]), f"{proto}_trig_io": int(trig["io"]), f"{proto}_trig_s": int(trig["s"]),
                        f"{proto}_s1_pos": P["S1"], f"{proto}_io_pos": P["IO"], f"{proto}_s2_pos": P["S2"], f"{proto}_n_tokens": len(ids["clean"])})
        if not ok:
            continue
        rec.update({"prompt_clean": pr["clean"][0], "prompt_s2io": pr["s2io"][0], "prompt_s1io": pr["s1io"][0]})
        items.append(rec)
    it = pd.DataFrame(items)
    it.insert(0, "pair_id", [f"ioi{i:04d}" for i in range(len(it))])
    it.to_parquet(os.path.join(C.IOI_DIR, "items.parquet"), index=False)
    for proto in toks:
        for corr in ("s2io", "s1io"):
            ids_a, ids_b = it[f"{proto}_ids_clean"], it[f"{proto}_ids_{corr}"]
            str_pos = [json.dumps([int(r[f"{proto}_s2_pos"])]) if corr == "s2io" else json.dumps(sorted([int(r[f"{proto}_s1_pos"]), int(r[f"{proto}_io_pos"])]))
                       for _, r in it.iterrows()]
            p = pd.DataFrame({"pair_id": it.pair_id, "ids_a": ids_a, "ids_b": ids_b, "str_pos": str_pos,
                              "trig_a": it[f"{proto}_trig_io"], "trig_b": it[f"{proto}_trig_s"],
                              "prompt_a": it.prompt_clean, "prompt_b": it[f"prompt_{corr}"],
                              "word_a": it.io, "word_b": it.s, "template_type": it.template_type, "template_idx": it.template_idx,
                              "io": it.io, "s": it.s, "c": it.c, "d": it.d, "place": it.place, "obj": it.obj,
                              "s1_pos": it[f"{proto}_s1_pos"], "io_pos": it[f"{proto}_io_pos"], "s2_pos": it[f"{proto}_s2_pos"],
                              "n_tokens": it[f"{proto}_n_tokens"], "corruption": corr,
                              # first mention of the answer of A (IO) and of B (S, first mentioned at S1); read by the
                              # ext7 heads runner as ment_filled / ment_other
                              "mention_a_pos": it[f"{proto}_io_pos"].map(lambda x: json.dumps([int(x)])),
                              "mention_b_pos": it[f"{proto}_s1_pos"].map(lambda x: json.dumps([int(x)]))})
            p.to_parquet(os.path.join(C.IOI_DIR, f"pairs_{corr}_{proto}.parquet"), index=False)
    info = {"source": "redwoodresearch/Easy-Transformer easy_transformer/ioi_dataset.py (BABA_TEMPLATES, ABBA via its own rule, "
                      "NAMES, PLACES, OBJECTS), fetched 2026-10-04", "n_requested": args.n, "seed": args.seed, "n_items": len(it),
            "tries": tries, "rejections": dict(why), "names": names, "places": places, "objects": objects,
            "template_counts": it.groupby("template_type").size().to_dict()}
    json.dump(info, open(os.path.join(C.IOI_DIR, "build.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "names"}, indent=1))
    for r in it.head(3).itertuples():
        print(repr(r.prompt_clean), "->", r.io, "|", repr(r.prompt_s2io), "|", repr(r.prompt_s1io))


if __name__ == "__main__":
    main()
