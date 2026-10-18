"""Phase 6 (coordinator): "how many random-order experts equal greedy's first k" for add-back curves.

random_equivalent(k_ref, r_ref, k_rand, r_rand, ks) -> {k: k_rand_equivalent or None}
    For each k in ks, the smallest k' with r_rand(k') >= r_ref(k), linearly interpolated in log k between the two grid
    points that bracket the crossing. None = random order never reaches that value (e.g. greedy overshoots the all-MoE
    ceiling, where random ends). k_ref / r_ref is the greedy (or oracle) curve.
draw_random_equivalent(ax, k_ref, r_ref, k_rand, r_rand, ks, color)
    Dotted horizontal connector from greedy's point at k to the random curve at the same r, a small label with the
    random-order k ("never" with an arrow to the right edge if it is not reached). Returns the dict above.
fmt_equiv(d) -> "1→3 · 5→40 · 10→120 · 15→never" (for subtitles and tables).
"""
import numpy as np

KS = (1, 5, 10, 15)


def _at(k, r, kq):
    """Value of a curve at k = kq (exact grid point, else log-k interpolation; None outside the grid)."""
    k = np.asarray(k, float)
    r = np.asarray(r, float)
    ok = np.isfinite(k) & np.isfinite(r)
    k, r = k[ok], r[ok]
    if len(k) == 0 or kq < k.min() or kq > k.max():
        return None
    return float(np.interp(np.log(kq), np.log(k), r))


def random_equivalent(k_ref, r_ref, k_rand, r_rand, ks=KS):
    kr = np.asarray(k_rand, float)
    rr = np.asarray(r_rand, float)
    ok = np.isfinite(kr) & np.isfinite(rr)
    kr, rr = kr[ok], rr[ok]
    order = np.argsort(kr)
    kr, rr = kr[order], rr[order]
    out = {}
    for kq in ks:
        target = _at(k_ref, r_ref, kq)
        if target is None:
            continue
        hit = np.nonzero(rr >= target)[0]
        if len(hit) == 0:
            out[kq] = None
            continue
        j = hit[0]
        if j == 0:
            out[kq] = float(kr[0])
            continue
        r0, r1 = rr[j - 1], rr[j]
        f = 0.0 if r1 == r0 else (target - r0) / (r1 - r0)
        out[kq] = float(np.exp(np.log(kr[j - 1]) + f * (np.log(kr[j]) - np.log(kr[j - 1]))))
    return out


def fmt_k(v):
    if v is None:
        return "never"
    return str(int(round(v))) if v >= 10 else (f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}")


def fmt_equiv(d):
    return " · ".join(f"{k}→{fmt_k(v)}" for k, v in d.items())


def draw_random_equivalent(ax, k_ref, r_ref, k_rand, r_rand, ks=KS, color="#52514e", kmax=None):
    d = random_equivalent(k_ref, r_ref, k_rand, r_rand, ks)
    xmax = kmax if kmax is not None else float(np.nanmax(np.asarray(k_rand, float)))
    for kq, kv in d.items():
        y = _at(k_ref, r_ref, kq)
        x1 = kv if kv is not None else xmax
        ax.plot([kq, x1], [y, y], color=color, linewidth=0.9, linestyle=(0, (1.2, 2.0)), zorder=1.5)
        ax.plot([kq], [y], marker="o", markersize=3.2, color=color, zorder=4)
        if kv is not None:
            ax.plot([kv], [y], marker="|", markersize=7, color=color, zorder=4)
            ax.annotate(f"{kq}→{fmt_k(kv)}", (kv, y), xytext=(3, 2), textcoords="offset points", fontsize=7.2, color=color,
                        va="bottom", ha="left", zorder=5)
        else:
            ax.annotate(f"{kq}→never", (x1, y), xytext=(-2, 2), textcoords="offset points", fontsize=7.2, color=color,
                        va="bottom", ha="right", zorder=5)
    return d
