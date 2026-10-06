"""ext11 Part C (GPU, prefill only, existing DiagSpec): where else do the target experts fire?

Prompt sets (one model, its Phase-3 protocol: Qwen3 tokenizer default, Mixtral with BOS):
  wino_main    clean prompts of the 512 directed cases of the main family (discovery + validation pairs); true = own
               trigger, foil = the twin's trigger (= the ext7 STR rows)
  wino_local   the context-free "local" prompts of the main pairs ("The bag was too"; local_prompt() of
               scripts/ext7_wino_scan.py), both answers; true = own trigger, foil = the twin's trigger
  wino_pool    the remaining margin-pool pairs (both prompts) of the model's ext7 competence scan
  wino_pool_local  their local prompts
  cf_str       CounterFact STR clean prompts (Direction 6 cases, paper IDs) and every selected donor prompt (scored with
               the case's true / foil)
  cf_scan      the 1,024 clean CounterFact prompts of the base filter scan (results/<qwen3|mixtral>/filter_scan.parquet)
  ioi          the 1,600 IOI clean prompts (data/ioi/items.parquet), true = IO, foil = S
  wiki         wikitext-103 test windows of 127 content tokens (Mixtral: BOS prepended), consecutive slices
Recorded per prompt: Delta, top-1, final-position routing at every layer (route_idx / route_w), per routed expert the
final-position DLA (DiagSpec.contrib_dla / final_rms; c_e of the expert itself, final norm frozen at the prompt's own
scale), and at the "interest" layers (targets + both tasks' ext8 population top-10) the routing of EVERY token
(DiagSpec.route_all_layers): top-k ids / weights and, for the experts of interest, router probability and router rank.

Raw arrays: /opt/dlami/nvme/moe_ext11/<out>/chunk_NN.npz (resumable per chunk); prompt table and compact tables in
results/<out>/ (prompts.parquet, final_routing.parquet = interest layers, all slots; token_events.parquet = (prompt,
position, layer, expert) for every routing of an expert of interest; run_meta.json).

Usage: python scripts/ext11_writer_routing.py qwen3|mixtral --out <run> [--budget 140000] [--wiki-windows 1100] [--dry-run]
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd

from moetrace.models import MODELS, RESULTS
from moetrace import ext11_writer as W

RAW = "/opt/dlami/nvme/moe_ext11"
WIKI = "/opt/dlami/nvme/moe_ext3/corpus/wikitext103_test.parquet"
PROTO = {"qwen3": ("qwen3", True, "qwen3"), "mixtral": ("mixtral_bos", True, "mixtral_bos")}  # model -> (wino proto, special tokens, ioi prefix)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def interest(model: str) -> tuple[list, list]:
    """Experts of interest (targets + both tasks' ext8 population top-10) and their layers."""
    A = json.load(open(W.SUMMARY))["A"]
    ex = [W.ename(x) for x in W.model_targets(model)]
    for k, r in A.items():
        if r["model"] == model:
            ex += r["pop_top10"]
    ex = list(dict.fromkeys(ex))
    layers = sorted({W.parse_ename(e)[0] for e in ex})
    return ex, layers


def wiki_windows(tok, W_: int, n: int):
    import importlib.util
    spec = importlib.util.spec_from_file_location("e3c", "/home/ubuntu/MOE/scripts/ext3_corpus_routing.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    text = m.wiki_text()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n = min(n, len(ids) // W_ - 1)
    win = [ids[i * W_:(i + 1) * W_] for i in range(n)]
    nxt = [ids[(i + 1) * W_] for i in range(n)]  # the token after each window (next-token of the last position)
    return win, nxt, len(ids)


def build(model: str, args):
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    from moetrace import ext7_pairs as P, ext8_addback as X, ext6_str as S6
    wp, st, ip = PROTO[model]
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model]["repo"]))
    bos = [tok.bos_token_id] if (model == "mixtral") else []
    sys.path.insert(0, "/home/ubuntu/MOE/scripts")
    from ext7_wino_scan import local_prompt
    rows = []  # dicts: set, ids, true_id, foil_id, meta...

    def add(set_, ids, t, f, **meta):
        rows.append(dict(set=set_, ids=[int(x) for x in ids], true_id=int(t), foil_id=int(f), **meta))
    # ---- WinoGrande main (directed cases of the main family)
    key = "wino_qwen3" if model == "qwen3" else "wino_mixtral"
    t = W.load_task(key)
    pairs_path = os.path.join(W.ROOT, W.RUNS[key]["pairs"])
    praw = pd.read_parquet(pairs_path)
    cs_json = json.load(open(os.path.join(W.ROOT, W.RUNS[key]["case_sets"])))
    pairs = P.load_pairs(pairs_path, cs_json)
    main_pids = set()
    for c in t.cases.itertuples():
        pi, d = divmod(int(c.case_id), 2)
        r = pairs.loc[pi]
        main_pids.add(r.pair_id)
        fw = r.context.split()
        add("wino_main", c.clean_ids, c.true_id, c.foil_id, case_id=int(c.case_id), pair_idx=int(pi), d=int(d), split=c.split,
            pair_id=r.pair_id, final_word=fw[-1], final2=" ".join(fw[-2:]), names=bool(r.names),
            option=(r.ans_a if d == 0 else r.ans_b), trigger=(r.word_a if d == 0 else r.word_b))
    # local prompts of the main pairs (true = own trigger)
    for pi in sorted({int(c) // 2 for c in t.cases.case_id}):
        r = pairs.loc[pi]
        fw = r.context.split()
        for d, (ans, tt, ff, w) in enumerate(((r.ans_a, r.trig_a, r.trig_b, r.word_a), (r.ans_b, r.trig_b, r.trig_a, r.word_b))):
            ids = tok(local_prompt(r.context, ans), add_special_tokens=st)["input_ids"]
            add("wino_local", ids, tt, ff, case_id=2 * pi + d, pair_idx=int(pi), d=d, pair_id=r.pair_id, final_word=fw[-1],
                final2=" ".join(fw[-2:]), names=bool(r.names), option=ans, trigger=w, prompt=local_prompt(r.context, ans))
    # pool pairs (margin under this protocol), minus the main pairs
    scan = pd.read_parquet(os.path.join(RESULTS, f"wino_{wp}", "scan_pairs.parquet"))
    pool = scan[scan.margin & ~scan.pair_id.isin(main_pids)]
    pr = praw.set_index("pair_id")
    for r in pool.itertuples():
        q = pr.loc[r.pair_id]
        fw = r.context.split()
        ia, ib = json.loads(q.ids_a), json.loads(q.ids_b)
        for d, (ids, tt, ff, ans, w) in enumerate(((ia, q.trig_a, q.trig_b, r.ans_a, r.word_a), (ib, q.trig_b, q.trig_a, r.ans_b, r.word_b))):
            add("wino_pool", ids, tt, ff, pair_id=r.pair_id, d=d, final_word=fw[-1], final2=" ".join(fw[-2:]), names=bool(r.names),
                option=ans, trigger=w)
            lp = local_prompt(r.context, ans)
            add("wino_pool_local", tok(lp, add_special_tokens=st)["input_ids"], tt, ff, pair_id=r.pair_id, d=d, final_word=fw[-1],
                final2=" ".join(fw[-2:]), names=bool(r.names), option=ans, trigger=w, prompt=lp)
    # ---- CounterFact STR (clean + donors) and the base scan
    ck = "cf_qwen3" if model == "qwen3" else "cf_mixtral"
    tc = W.load_task(ck)
    sc = pd.read_parquet(os.path.join(RESULTS, W.RUNS[ck]["src"], "sweep_cases.parquet")).set_index("case_id")
    for c in tc.cases.itertuples():
        add("cf_str", c.clean_ids, c.true_id, c.foil_id, case_id=int(c.case_id), slot=-1, split=c.split,
            relation=sc.relation.get(int(c.case_id), ""), kind="clean")
    cmap = tc.cases.set_index("case_id")
    for r in tc.rows.itertuples():
        c = cmap.loc[int(r.case_id)]
        add("cf_str", r.corrupt_ids, c.true_id, c.foil_id, case_id=int(r.case_id), slot=int(r.slot), split=c.split,
            relation=sc.relation.get(int(r.case_id), ""), kind="donor")
    fs = pd.read_parquet(os.path.join(RESULTS, "qwen3" if model == "qwen3" else "mixtral", "filter_scan.parquet"))
    for r in fs.itertuples():
        add("cf_scan", json.loads(r.ids), r.true_id, r.foil_id, case_id=int(r.case_id), relation=r.relation,
            true_str=r.true_str, prompt=r.prompt)
    # ---- IOI clean prompts
    it = pd.read_parquet(os.path.join(W.ROOT, "data/ioi/items.parquet"))
    for r in it.itertuples():
        add("ioi", json.loads(getattr(r, f"{ip}_ids_clean")), int(getattr(r, f"{ip}_trig_io")), int(getattr(r, f"{ip}_trig_s")),
            pair_id=r.pair_id, template_type=r.template_type, prompt=r.prompt_clean)
    # ---- wikitext windows
    win, nxt, n_tok = wiki_windows(tok, args.window, args.wiki_windows)
    for i, (w, nx) in enumerate(zip(win, nxt)):
        add("wiki", bos + w, 0, 1, window=i, next_after=int(nx))
    df = pd.DataFrame(rows)
    df["n_tok"] = df.ids.map(len)
    df.insert(0, "idx", np.arange(len(df)))
    return df, {"wiki_corpus_tokens": int(n_tok)}


def chunks_of(df: pd.DataFrame, budget: int):
    order = df.sort_values("n_tok", kind="stable").idx.values
    out, cur, cmax = [], [], 0
    for j in order:
        T = int(df.n_tok.iloc[j])
        if cur and max(cmax, T) * (len(cur) + 1) > budget:
            out.append(cur)
            cur, cmax = [], 0
        cur.append(int(j))
        cmax = max(cmax, T)
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(PROTO))
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=141000, help="padded prefill tokens per engine pass")
    ap.add_argument("--window", type=int, default=127)
    ap.add_argument("--wiki-windows", type=int, default=1100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    od = os.path.join(RESULTS, args.out)
    rd = os.path.join(RAW, args.out)
    os.makedirs(od, exist_ok=True)
    os.makedirs(rd, exist_ok=True)
    ex, layers = interest(args.model)
    ex_le = [W.parse_ename(e) for e in ex]
    pp = os.path.join(od, "prompts_input.parquet")
    if os.path.exists(pp):
        df = pd.read_parquet(pp)
        extra = json.load(open(os.path.join(od, "run_meta.json"))).get("build", {})
    else:
        df, extra = build(args.model, args)
        df.assign(ids=df.ids.map(json.dumps)).to_parquet(pp, index=False)
    if isinstance(df.ids.iloc[0], str):
        df["ids"] = df.ids.map(json.loads)
    ch = chunks_of(df, args.budget)
    meta = {"model": args.model, "repo": MODELS[args.model]["repo"], "protocol": "tokenizer default" + (" (BOS)" if args.model == "mixtral" else ""),
            "experts_of_interest": ex, "route_all_layers": layers, "sets": df.set.value_counts().to_dict(), "n_prompts": int(len(df)),
            "n_tokens": int(df.n_tok.sum()), "budget": args.budget, "n_chunks": len(ch), "window": args.window,
            "wiki_windows": int((df.set == "wiki").sum()), "raw_dir": rd, "build": extra,
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    log(f"{args.model}: {len(df)} prompts ({meta['sets']}), {meta['n_tokens']} tokens, {len(ch)} chunks; interest {ex}; layers {layers}")
    if args.dry_run:
        for i, c in enumerate(ch):
            log(f"  chunk {i}: {len(c)} rows, T <= {df.n_tok.iloc[c].max()}, sets {df.set.iloc[c].value_counts().to_dict()}")
        return
    import torch
    from moetrace.engine import Engine, PrefillSpec, DiagSpec
    eng = Engine(MODELS[args.model]["repo"])
    L, E, k = eng.spec.n_layers, eng.spec.n_experts, eng.spec.top_k
    times = []
    for ci, c in enumerate(ch):
        fp = os.path.join(rd, f"chunk_{ci:02d}.npz")
        if os.path.exists(fp):
            log(f"chunk {ci} exists, skipped")
            continue
        sub = df.iloc[c]
        pre = [PrefillSpec(list(r.ids), int(r.true_id), int(r.foil_id)) for r in sub.itertuples()]
        diag = DiagSpec(route_all_layers=tuple(layers), contrib_dla=True)
        log(f"chunk {ci}/{len(ch)}: {len(pre)} rows, T <= {sub.n_tok.max()}")
        torch.cuda.reset_peak_memory_stats()
        res = eng.run(pre, [], record_routing=True, log=log if ci == 0 else None, diag=diag)
        peak = torch.cuda.max_memory_allocated() / 2**30
        dg = res.extra["diag"]
        out = dict(idx=np.array(c), lens=res.lens, delta=res.delta, top1=res.top1, logit_true=res.logit_true, logit_foil=res.logit_foil,
                   route_idx=res.route_idx.astype(np.int16), route_w=res.route_w.astype(np.float32),
                   contrib_dla=dg["contrib_dla"], final_rms=dg["final_rms"], layers=np.array(layers), ex=np.array(ex_le))
        for l in layers:
            topi, topv, rlog = dg["route_all"][l]
            out[f"topi_{l}"] = topi.astype(np.int16)
            out[f"topv_{l}"] = topv.astype(np.float16)
            cols = [e for (ll, e) in ex_le if ll == l]
            lg = torch.from_numpy(rlog)
            pr = torch.softmax(lg.float(), -1)
            out[f"prob_{l}"] = pr[..., cols].numpy().astype(np.float16)
            # router rank of each expert of interest (0 = highest logit): number of experts with a strictly larger logit
            out[f"rank_{l}"] = (lg.unsqueeze(-2) > lg[..., cols].unsqueeze(-1)).sum(-1).numpy().astype(np.int16)
            out[f"cols_{l}"] = np.array(cols)
        np.savez(fp, **out)
        times.append(res.extra["total_s"])
        log(f"chunk {ci} done in {res.extra['total_s']:.0f}s, peak {peak:.1f} GB")
        del res, dg, out
        torch.cuda.empty_cache()
    meta.update({"complete": True, "pass_times_s": times, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    log("ALL CHUNKS COMPLETE")


if __name__ == "__main__":
    main()
