"""ext3 experiment 3 (H4): corpus-level routing statistics with and without BOS.

Generic text (wikitext-103 test split, lightly detokenised) or Python code (CPython standard library sources) is
tokenised without special tokens and cut into non-overlapping windows of `--window` content tokens. Every window is run
twice in the same pass: with `<s>` prepended (BOS protocol, T = window + 1) and bare (no-BOS protocol, T = window).
Recorded for every token at every layer: top-k routing, routing weights and router logits; next-token log-probs;
per-position residual norms.

Outputs: raw arrays in /opt/dlami/nvme/moe_ext3/<model>_corpus_<corpus>/ (route_all.npz);
results/<model>_bos_corpus_<corpus>/ and results/<model>_nobos_corpus_<corpus>/ with run_meta.json and
corpus_stats.json (per-layer per-expert usage, per-position agreement between the protocols, per-expert token-class
entropy at the paper layer, per-position mean log-prob).

Usage: python scripts/ext3_corpus_routing.py mixtral --corpus wiki|code [--window 127 --n-windows 400]
"""
import argparse, glob, json, math, os, re, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS, RESULTS
from moetrace.engine import Engine, PrefillSpec, DiagSpec
from moetrace.arch import snapshot_dir
from moetrace.ext3_variants import DIAG_ROOT, write_run_meta

WIKI = "/opt/dlami/nvme/moe_ext3/corpus/wikitext103_test.parquet"
PY_DIR = "/usr/lib/python3.13"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wiki_text() -> str:
    d = pd.read_parquet(WIKI)
    lines = [t for t in d.text.tolist() if t.strip() and not t.strip().startswith("=")]
    t = "".join(lines)
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r" ([,.;:!?')])", r"\1", t)
    t = re.sub(r"([(\[]) ", r"\1", t)
    t = t.replace(" n't", "n't").replace(" 's", "'s").replace('" ', '"').replace(" \"", "\"")
    return t


def code_text(min_chars: int = 2_000_000) -> str:
    files = sorted(glob.glob(os.path.join(PY_DIR, "*.py")))
    out, n = [], 0
    for f in files:
        try:
            s = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if len(s) < 2000:
            continue
        out.append(s)
        n += len(s)
        if n >= min_chars:
            break
    return "\n\n".join(out)


def token_class(piece: str) -> str:
    p = piece.replace("▁", " ")
    if p.strip() == "":
        return "space"
    if p.startswith("<0x"):
        return "byte"
    core = p.strip()
    if core.isdigit():
        return "digit"
    if core.isalpha():
        return "word_start" if p.startswith(" ") else "word_cont"
    if all(not ch.isalnum() for ch in core):
        return "punct"
    return "mixed"


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--corpus", required=True, choices=["wiki", "code"])
    ap.add_argument("--window", type=int, default=127)
    ap.add_argument("--n-windows", type=int, default=400)
    args = ap.parse_args()
    m = MODELS[args.model]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_dir(m["repo"]))
    bos_id = tok.bos_token_id
    assert bos_id is not None, "corpus experiment needs a BOS token"
    text = wiki_text() if args.corpus == "wiki" else code_text()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    W = args.window
    n_win = min(args.n_windows, len(ids) // W)
    windows = [ids[i * W : (i + 1) * W] for i in range(n_win)]
    log(f"{args.corpus}: {len(text)} chars, {len(ids)} tokens, {n_win} windows of {W}")
    eng = Engine(m["repo"])
    L, E, k = eng.spec.n_layers, eng.spec.n_experts, eng.spec.top_k
    pre = [PrefillSpec([bos_id] + w, 0, 1) for w in windows] + [PrefillSpec(list(w), 0, 1) for w in windows]
    diag = DiagSpec(resid_norms=True, token_logprobs=True, route_all_layers=tuple(range(L)))
    t0 = time.time()
    res = eng.run(pre, [], record_routing=False, log=log, diag=diag)
    log(f"pass time {time.time() - t0:.1f}s  T={res.extra['T']} row_chunk={res.extra['row_chunk']}")
    dg = res.extra["diag"]
    topi = np.stack([dg["route_all"][l][0] for l in range(L)])  # [L, 2n, T, k] int16
    rlog = np.stack([dg["route_all"][l][2] for l in range(L)])  # [L, 2n, T, E]
    n = n_win
    # content-aligned views: bos rows positions 1..W, nobos rows positions 0..W-1
    tb = topi[:, :n, 1 : W + 1]  # [L, n, W, k]
    tn = topi[:, n:, 0:W]
    lb = rlog[:, :n, 1 : W + 1]
    ln = rlog[:, n:, 0:W]
    content = np.array(windows)  # [n, W]
    raw_dir = os.path.join(DIAG_ROOT, f"{args.model}_corpus_{args.corpus}")
    os.makedirs(raw_dir, exist_ok=True)
    np.savez_compressed(os.path.join(raw_dir, "route_all.npz"), topi=topi, router_logits=rlog, content=content,
                        token_logprobs=dg["token_logprobs"], resid_norms=dg["resid_norms"], lens=res.lens, bos_id=bos_id, window=W)
    # ---- statistics
    def usage(t):  # [L, n, W, k] -> [L, E] fraction of content tokens routing to e (top-k membership)
        u = np.zeros((L, E))
        for e in range(E):
            u[:, e] = (t == e).any(-1).reshape(L, -1).mean(-1)
        return u

    def top1_usage(t):
        u = np.zeros((L, E))
        for e in range(E):
            u[:, e] = (t[..., 0] == e).reshape(L, -1).mean(-1)
        return u

    same_set = (np.sort(tb, -1) == np.sort(tn, -1)).all(-1)  # [L, n, W]
    same_top1 = tb[..., 0] == tn[..., 0]
    pl = m["paper_layer"]
    pieces = tok.convert_ids_to_tokens([int(x) for x in np.unique(content)])
    cls_of = dict(zip([int(x) for x in np.unique(content)], [token_class(p) for p in pieces]))
    classes = sorted(set(cls_of.values()))
    cls_arr = np.vectorize(lambda x: classes.index(cls_of[int(x)]))(content)  # [n, W]
    stats = {"corpus": args.corpus, "n_windows": n, "window": W, "n_content_tokens": int(n * W), "layers": L, "experts": E, "top_k": k,
             "paper_layer": pl, "pass_time_s": res.extra["total_s"], "classes": classes,
             "class_counts": {c: int((cls_arr == i).sum()) for i, c in enumerate(classes)},
             "usage_topk_bos": usage(tb).round(4).tolist(), "usage_topk_nobos": usage(tn).round(4).tolist(),
             "usage_top1_bos": top1_usage(tb).round(4).tolist(), "usage_top1_nobos": top1_usage(tn).round(4).tolist(),
             "agreement_set_per_layer": same_set.reshape(L, -1).mean(-1).round(4).tolist(),
             "agreement_top1_per_layer": same_top1.reshape(L, -1).mean(-1).round(4).tolist(),
             "agreement_set_by_position_paper_layer": same_set[pl].mean(0).round(4).tolist(),
             "agreement_set_by_position_all_layers_mean": same_set.mean((0, 1)).round(4).tolist(),
             "first_token_routing_bos_row_pos0": {f"E{e:03d}": float((topi[pl, :n, 0] == e).any(-1).mean()) for e in range(E)},
             "first_content_token_routing_nobos": {f"E{e:03d}": float((tn[pl, :, 0] == e).any(-1).mean()) for e in range(E)},
             "first_content_token_routing_bos": {f"E{e:03d}": float((tb[pl, :, 0] == e).any(-1).mean()) for e in range(E)},
             "mean_router_logit_shift_nobos_minus_bos_paper_layer": (ln[pl] - lb[pl]).mean((0, 1)).round(4).tolist(),
             "mean_router_logit_shift_by_position_E_paper_layer": {f"E{e:03d}": (ln[pl, :, :, e] - lb[pl, :, :, e]).mean(0).round(3).tolist() for e in range(E)},
             }
    # per-expert usage by content position (paper layer), both protocols
    stats["usage_by_position_paper_layer_bos"] = {f"E{e:03d}": (tb[pl] == e).any(-1).mean(0).round(4).tolist() for e in range(E)}
    stats["usage_by_position_paper_layer_nobos"] = {f"E{e:03d}": (tn[pl] == e).any(-1).mean(0).round(4).tolist() for e in range(E)}
    # per-expert token-class entropy and token-id entropy at the paper layer
    for name, t in (("bos", tb), ("nobos", tn)):
        ent_cls, ent_id, pos_ent, n_tok = {}, {}, {}, {}
        for e in range(E):
            mask = (t[pl] == e).any(-1)  # [n, W]
            cc = np.bincount(cls_arr[mask], minlength=len(classes)).astype(float)
            ic = np.bincount(content[mask], minlength=int(content.max()) + 1).astype(float)
            pc = mask.sum(0).astype(float)  # usage by position
            ent_cls[f"E{e:03d}"] = round(entropy(cc), 4) if cc.sum() else None
            ent_id[f"E{e:03d}"] = round(entropy(ic), 4) if ic.sum() else None
            pos_ent[f"E{e:03d}"] = round(entropy(pc) / math.log2(W), 4) if pc.sum() else None  # 1 = perfectly position-agnostic
            n_tok[f"E{e:03d}"] = int(mask.sum())
        stats[f"class_entropy_bits_paper_layer_{name}"] = ent_cls
        stats[f"tokenid_entropy_bits_paper_layer_{name}"] = ent_id
        stats[f"position_entropy_normalised_paper_layer_{name}"] = pos_ent
        stats[f"n_tokens_paper_layer_{name}"] = n_tok
        cls_dist = {}
        for e in range(E):
            mask = (t[pl] == e).any(-1)
            cc = np.bincount(cls_arr[mask], minlength=len(classes)).astype(float)
            cls_dist[f"E{e:03d}"] = (cc / max(cc.sum(), 1)).round(3).tolist()
        stats[f"class_distribution_paper_layer_{name}"] = cls_dist
    # log-probs by content position: bos rows predict content[j] at position j (token j+1), nobos rows at position j-1
    lp = dg["token_logprobs"]
    lp_b = lp[:n, 1 : W]  # predictions of content tokens 1..W-1 given BOS + content[:j]
    lp_n = lp[n:, 0 : W - 1]  # predictions of content tokens 1..W-1 given content[:j]
    stats["mean_logprob_by_content_position_bos"] = np.nanmean(lp_b, 0).round(3).tolist()
    stats["mean_logprob_by_content_position_nobos"] = np.nanmean(lp_n, 0).round(3).tolist()
    stats["mean_logprob_bos"] = float(np.nanmean(lp_b))
    stats["mean_logprob_nobos"] = float(np.nanmean(lp_n))
    stats["logprob_first_content_token_given_bos"] = float(np.nanmean(lp[:n, 0]))
    rn = dg["resid_norms"]
    stats["resid_norm_pos0_per_layer_bos"] = rn[:, :n, 0].mean(1).round(1).tolist()
    stats["resid_norm_pos0_per_layer_nobos"] = rn[:, n:, 0].mean(1).round(1).tolist()
    stats["resid_norm_content1_per_layer_bos"] = rn[:, :n, 1].mean(1).round(1).tolist()
    stats["resid_norm_median_content_per_layer_nobos"] = np.median(rn[:, n:, 1:W], axis=(1, 2)).round(1).tolist()
    for proto in ("bos", "nobos"):
        rd = f"{args.model}_{proto}_corpus_{args.corpus}"
        os.makedirs(os.path.join(RESULTS, rd), exist_ok=True)
        with open(os.path.join(RESULTS, rd, "corpus_stats.json"), "w") as f:
            json.dump(stats, f, indent=1)
        write_run_meta(rd, {"model": args.model, "repo": m["repo"], "experiment": "ext3 corpus-level routing statistics (H4)", "agent": "ext3-bos-mechanism",
                            "protocol": proto, "corpus": args.corpus, "source": WIKI if args.corpus == "wiki" else PY_DIR, "window": W, "n_windows": n,
                            "pass_shared_with": f"{args.model}_{'nobos' if proto == 'bos' else 'bos'}_corpus_{args.corpus}",
                            "raw": os.path.join(raw_dir, "route_all.npz"), "pass_time_s": res.extra["total_s"],
                            "command": "python scripts/ext3_corpus_routing.py " + " ".join(sys.argv[1:]),
                            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "outputs": ["corpus_stats.json (identical copy in both protocol dirs; per-layer/per-expert usage, agreement, entropies)"]})
    u_b, u_n = usage(tb)[pl], usage(tn)[pl]
    log(f"L{pl} usage (top-k membership) bos:   " + " ".join(f"E{e}={u_b[e]:.3f}" for e in range(E)))
    log(f"L{pl} usage (top-k membership) nobos: " + " ".join(f"E{e}={u_n[e]:.3f}" for e in range(E)))
    log(f"L{pl} set agreement bos vs nobos: {same_set[pl].mean():.3f}; by position (first 8): {same_set[pl].mean(0)[:8].round(3).tolist()}")
    log(f"mean logprob bos {stats['mean_logprob_bos']:.3f} nobos {stats['mean_logprob_nobos']:.3f}")
    log("done")


if __name__ == "__main__":
    main()
