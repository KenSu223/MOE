"""ext7 W5: per-head attention patching at the final position on STR pairs (as scripts/ext5_heads_sweep.py, STR
instead of GN; generic pair files, see moetrace/ext7_pairs.py).

Per directed case and requested layer (parent = corrupted row, clean = clean row): one `attn_head` spawn per head
(v_h = W_o[:, h] (H_h_clean - H_h_corrupt) added to the corrupted attention output before the MoE of that layer), plus
reference rows attn_layer, layer, block. Prefill diagnostics: the final position's attention distribution over
positions at the requested layers (DiagSpec.attn_final) for the clean and corrupted rows. Position classes of every
directed case (head_positions.parquet): STR site (the filled option for WinoGrande), first mention of the candidate
filled in the CLEAN prompt (ment_filled) and of the other candidate (ment_other), position 0, final, other. Mentions come
from pair columns mention_a_pos / mention_b_pos (JSON) when present, else (WinoGrande) from the first occurrence of
ans_a / ans_b in the shared context before the blank (tokenizer offsets).

--layers auto = the top-4 layers of the discovery attention-output (attn_layer) curve of the run's final-position sweep
plus one null layer (smallest |discovery mean| among the layers of the second half that are not in the top 4).

Usage: python scripts/ext7_wino_heads.py --model qwen3 --pairs <parquet> --case-sets <json> --out wino_qwen3_str
           [--layers auto|l1,l2] [--families main] [--max-rows 50000]
Outputs results/<out>/head_rows.parquet, head_prefill.parquet, head_positions.parquet, head_attn_final.npz,
        run_meta.json ("heads")
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import MODELS
from moetrace import ext7_pairs as P

REF_KINDS = ("attn_layer", "layer", "block")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def auto_layers(od, fam, L):
    sw = pd.read_parquet(os.path.join(od, "str_sweep_rows.parquet"), columns=["case_id", "kind", "layer", "rescue"])
    R = sw[sw.kind == "attn_layer"].pivot(index="case_id", columns="layer", values="rescue")
    f = "main" if "main" in fam else next(k for k, v in fam.items() if "discovery" in v)
    md = R.loc[fam[f]["discovery"]].mean(0)
    top = [int(l) for l in md.sort_values(ascending=False).index[:4]]
    rest = md[[l for l in md.index if l >= L // 2 and int(l) not in top]]
    null = int(rest.abs().idxmin())
    return sorted(top), null


def _span_tokens(offsets, special_mask, start, end):
    return [i for i, ((a, b), sp) in enumerate(zip(offsets, special_mask)) if not sp and b > a and a < end and b > start]


def mention_positions(pairs, model_repo):
    """pair_idx -> (mention_a_pos, mention_b_pos) lists (may be empty)."""
    out = {}
    if "mention_a_pos" in pairs.columns and "mention_b_pos" in pairs.columns:
        for r in pairs.itertuples():
            out[r.pair_idx] = (P._jl(r.mention_a_pos), P._jl(r.mention_b_pos))
        return out, "pair columns mention_a_pos / mention_b_pos"
    if not {"prompt_a", "ans_a", "ans_b", "context"} <= set(pairs.columns):
        return {r.pair_idx: ([], []) for r in pairs.itertuples()}, "none (no mention columns, no WinoGrande text)"
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    tok = AutoTokenizer.from_pretrained(snapshot_dir(model_repo))
    n_found = 0
    for r in pairs.itertuples():
        blank = r.context.index("_")
        res = []
        for st in (True, False):
            enc = tok(r.prompt_a, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=st)
            if list(enc["input_ids"]) == list(r.ids_a_l):
                break
        else:
            raise SystemExit(f"cannot reproduce ids_a of {r.pair_id}")
        for ans in (r.ans_a, r.ans_b):
            pre = r.prompt_a[:blank]
            j = -1
            for cand in (ans, ans.lower(), ans[:1].upper() + ans[1:]):
                mt = re.search(r"\b" + re.escape(cand) + r"\b", pre)
                if mt:
                    j, ans = mt.start(), cand
                    break
            res.append(_span_tokens(enc["offset_mapping"], enc["special_tokens_mask"], j, j + len(ans)) if j >= 0 else [])
        n_found += bool(res[0]) + bool(res[1])
        out[r.pair_idx] = (res[0], res[1])
    return out, f"first occurrence of ans_a / ans_b before the blank (found {n_found} of {2 * len(pairs)})"


def main():
    ap = argparse.ArgumentParser()
    P.add_common_args(ap)
    ap.add_argument("--layers", default="auto")
    ap.add_argument("--families", default="main")
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--wf-chunk", type=int, default=2048, help="wavefront attention row chunk (memory)")
    args = ap.parse_args()
    m = MODELS[args.model]
    from moetrace.arch import load_spec
    spec = load_spec(m["repo"])[0]
    L, nH = spec.n_layers, spec.n_heads
    cs, pairs, _, _, _ = P.setup_run(args, "heads")
    od = P.run_dir(args.out)
    fam_all = P.load_families(args.out)
    fam = {k: v for k, v in fam_all.items() if k in args.families.split(",")}
    ids = P.all_ids(fam)
    dc = P.directed_cases(pairs, ids)
    n = len(ids)
    null = None
    if args.layers == "auto":
        top, null = auto_layers(od, fam_all, L)
        layers = sorted(set(top) | {null})
    else:
        layers = sorted(set(int(x) for x in args.layers.split(",")))
    per_layer = n * (nH + len(REF_KINDS))
    chunks, cur = [], []
    for l in layers:
        if cur and (len(cur) + 1) * per_layer > args.max_rows:
            chunks.append(cur); cur = []
        cur.append(l)
    chunks.append(cur)
    ment, ment_src = mention_positions(pairs.loc[sorted({c // 2 for c in ids})], m["repo"])
    prow = []
    for c in ids:
        x = dc[c]
        ma, mb = ment[x.pair_idx]
        mf, mo = (ma, mb) if x.d == 0 else (mb, ma)
        prow.append(dict(case_id=c, T=x.T, str_pos=json.dumps(x.str_pos), ment_filled=json.dumps(mf), ment_other=json.dumps(mo)))
    pd.DataFrame(prow).to_parquet(os.path.join(od, "head_positions.parquet"), index=False)
    log(f"{args.model} -> {args.out}: {n} directed cases, layers {layers} (null layer {null}), {nH} heads, {len(chunks)} passes, "
        f"{per_layer * len(layers)} spawn rows; mentions: {ment_src}")
    if args.dry_run:
        return
    import torch
    from moetrace.engine import Engine, PrefillSpec, SpawnSpec, DiagSpec
    meta = {"model": args.model, "repo": m["repo"], "families": list(fam), "layers": layers, "null_layer": null, "n_heads": nH, "n_cases": n,
            "chunks": chunks, "mentions": ment_src, "agent": args.agent, "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    P.write_meta(args.out, "heads", meta)
    eng = Engine(m["repo"])
    pre = [PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    pre += [PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    rows, times = [], []
    for ci, ch in enumerate(chunks):
        spawns, tags = [], []
        for i, c in enumerate(ids):
            for l in ch:
                for hh in range(nH):
                    spawns.append(SpawnSpec(l, n + i, i, "attn_head", expert=hh)); tags.append((i, l, "attn_head", hh))
                for k in REF_KINDS:
                    spawns.append(SpawnSpec(l, n + i, i, k)); tags.append((i, l, k, -1))
        log(f"pass {ci + 1}/{len(chunks)}: layers {ch}, {len(pre)} prefill rows, {len(spawns)} spawn rows")
        res = eng.run(pre, spawns, record_routing=False, log=log if ci == 0 else None, diag=DiagSpec(attn_final=(ci == 0)), wf_chunk=args.wf_chunk)
        times.append(res.extra["total_s"])
        d, sd = res.delta, res.sp_delta
        for jj, (i, l, k, hh) in enumerate(tags):
            rows.append(dict(case_id=ids[i], layer=l, head=hh, kind=k, delta=float(sd[jj]), rescue=float(sd[jj] - d[n + i]),
                             vnorm=float(res.sp_vnorm[jj]), delta_clean=float(d[i]), delta_corrupt=float(d[n + i])))
        pd.DataFrame(rows).to_parquet(os.path.join(od, "head_rows.parquet"), index=False)
        if ci == 0:
            pd.DataFrame(dict(case_id=ids, delta_clean=d[:n], delta_corrupt=d[n:], top1_clean=res.top1[:n], top1_corrupt=res.top1[n:])
                         ).to_parquet(os.path.join(od, "head_prefill.parquet"), index=False)
            attn = res.extra["diag"]["attn_final"]  # [L, 2n, nH, T] fp16
            np.savez_compressed(os.path.join(od, "head_attn_final.npz"), attn=attn[layers], layers=np.array(layers), case_ids=np.array(ids),
                                lens=res.lens, T=res.extra["T"])
        log(f"pass {ci + 1}: {res.extra['total_s']:.0f}s; {len(rows)} rows")
        del res
        torch.cuda.empty_cache()
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": times, "n_rows": len(rows)})
    P.write_meta(args.out, "heads", meta)


if __name__ == "__main__":
    main()
