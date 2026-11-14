#!/usr/bin/env python3
"""Slack-tolerant 2-pocket and 3-pocket staircase DP (K = 16..31).

Finds top-3 distinct m in {2, 3} staircase seams for every (K, D) cell with:
    sum w_i <= K
    sum h_i <= K
    sum w_i + sum h_i + m - 1 <= 2 * K
    sum g_i >= D
    dL_rigid = (K + max_i f_i) - L_inc(K^2 + D) <= slack_max  (default +6.0e-3)
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np


def items_from_gadlib(path: str) -> list[dict]:
    lib = json.loads(Path(path).read_text())
    best: dict[tuple[int, int, int], tuple[float, str, int]] = {}
    for g in lib:
        fx = float(g.get("fx", g.get("f", 0.0)))
        fy = float(g.get("fy", g.get("f", 0.0)))
        f = max(fx, fy)
        for w, h in ((int(g["w"]), int(g["h"])), (int(g["h"]), int(g["w"]))):
            k = (w, h, int(g["g"]))
            if k not in best or f < best[k][0] - 1e-12:
                best[k] = (f, f"n={g['src']}", int(g["src"]))
    return [
        dict(w=w, h=h, g=gg, f=v[0], src=v[2], tag=v[1], transposed=False)
        for (w, h, gg), v in best.items()
    ]


def items_from_pockets(paths: list[str]) -> list[dict]:
    best: dict[tuple[int, int, int], dict] = {}
    for path in paths:
        if not Path(path).exists():
            continue
        lib = json.loads(Path(path).read_text())
        for g in lib:
            gw, gh, gg, gf, gsrc = int(g["w"]), int(g["h"]), int(g["g"]), float(g["f"]), int(g["src"])
            for w, h in ((gw, gh), (gh, gw)):
                k = (w, h, gg)
                if k not in best or gf < best[k]["f"] - 1e-12:
                    tr = (w, h) != (gw, gh)
                    best[k] = dict(
                        w=w,
                        h=h,
                        g=gg,
                        f=gf,
                        src=gsrc,
                        pocket=g,
                        tag=f"n={gsrc}{'T' if tr else ''}",
                        transposed=tr,
                    )
    return list(best.values())


def pareto(items: list[dict]) -> list[dict]:
    """Drop items dominated by w'<=w, h'<=h, g'>=g, f'<=f."""
    it = sorted(items, key=lambda r: (r["f"], -r["g"], r["w"] + r["h"]))
    keep: list[dict] = []
    for a in it:
        dom = False
        for b in keep:
            if b["w"] <= a["w"] and b["h"] <= a["h"] and b["g"] >= a["g"]:
                dom = True
                break
        if not dom:
            keep.append(a)
    return keep


def _seam_key(indices: tuple[int, ...], items: list[dict]) -> tuple:
    return tuple(sorted((items[i]["w"], items[i]["h"], items[i]["g"], items[i]["src"]) for i in indices))


def solve_for_K(
    K: int,
    items: list[dict],
    L_inc: dict[int, float],
    slack_max: float = 6.0e-3,
    top_k: int = 3,
    nmax: int = 1000,
) -> list[dict]:
    """Find top-k distinct m in {2, 3} seams for every D in 1..2K+1 with dL_rigid <= slack_max."""
    D_list = [D for D in range(1, 2 * K + 2) if K * K + D in L_inc and K * K + D <= nmax]
    if not D_list:
        return []
    f_limits = {D: L_inc[K * K + D] - K + slack_max for D in D_list}
    f_global_max = max(f_limits.values())

    # Filter items eligible for this K and global f_limit
    # For m >= 2, at least one other item has w>=1, h>=1, so item must have w <= K-1, h <= K-1, w+h <= 2K-3
    elig = [
        it for it in items
        if it["f"] <= f_global_max + 1e-12
        and 1 <= it["w"] <= K - 1
        and 1 <= it["h"] <= K - 1
        and it["w"] + it["h"] <= 2 * K - 3
    ]
    elig.sort(key=lambda r: (r["f"], -r["g"], r["w"] + r["h"]))
    M = len(elig)
    if M == 0:
        return []

    ws = np.array([it["w"] for it in elig], dtype=np.int32)
    hs = np.array([it["h"] for it in elig], dtype=np.int32)
    gs = np.array([it["g"] for it in elig], dtype=np.int32)
    fs = np.array([it["f"] for it in elig], dtype=np.float64)

    # 1) Enumerate valid 2-pocket pairs (i <= j) within K bounds
    # Keep per-D top_k for m=2, AND build compact 2-pocket Pareto/top-k table for m=3
    # For m=3, third item has w>=1, h>=1, so 2-pocket prefix has x <= K-1, y <= K-1, x+y <= 2K-4
    # Store for each (x, y, g_capped) up to top_k distinct pairs sorted by f2
    max_g_cap = 2 * K + 1
    pair_buckets: dict[tuple[int, int, int], list[tuple[float, int, int]]] = {}
    d_heaps: dict[int, list[tuple[float, int, tuple[int, ...]]]] = {D: [] for D in D_list}
    d_seen: dict[int, set[tuple]] = {D: set() for D in D_list}

    def push_cand(D: int, f_val: float, idxs: tuple[int, ...]) -> None:
        if f_val > f_limits[D] + 1e-12:
            return
        sk = _seam_key(idxs, elig)
        if sk in d_seen[D]:
            return
        h = d_heaps[D]
        if len(h) < top_k:
            d_seen[D].add(sk)
            heapq.heappush(h, (-f_val, len(idxs), idxs))
        elif f_val < -h[0][0] - 1e-12:
            d_seen[D].add(sk)
            _, _, old_idxs = heapq.heapreplace(h, (-f_val, len(idxs), idxs))
            d_seen[D].discard(_seam_key(old_idxs, elig))

    # Vectorized m=2 generation
    S2_max = 2 * K - 1  # w1+h1+w2+h2+1 <= 2K  <=> sum(w+h) <= 2K-1
    for i in range(M):
        wi, hi, gi, fi = int(ws[i]), int(hs[i]), int(gs[i]), float(fs[i])
        # j >= i
        w_sum = wi + ws[i:]
        h_sum = hi + hs[i:]
        ok = (w_sum <= K) & (h_sum <= K) & (w_sum + h_sum <= S2_max)
        js = np.nonzero(ok)[0] + i
        for j in js:
            j = int(j)
            x2 = wi + int(ws[j])
            y2 = hi + int(hs[j])
            g2 = gi + int(gs[j])
            f2 = max(fi, float(fs[j]))
            # Record for D <= g2
            for D in D_list:
                if D <= g2 and f2 <= f_limits[D] + 1e-12:
                    push_cand(D, f2, (i, j))
            # Record for m=3 extension if x2 <= K-1 and y2 <= K-1 and x2+y2 <= 2K-4
            if x2 <= K - 1 and y2 <= K - 1 and x2 + y2 <= 2 * K - 4:
                gc = min(g2, max_g_cap)
                bk = (x2, y2, gc)
                lst = pair_buckets.setdefault(bk, [])
                if len(lst) < top_k:
                    lst.append((f2, i, j))
                    if len(lst) == top_k:
                        lst.sort(key=lambda t: t[0])
                elif f2 < lst[-1][0] - 1e-12:
                    lst[-1] = (f2, i, j)
                    lst.sort(key=lambda t: t[0])

    # 2) Combine m=2 buckets with 3rd item k for m=3
    # To prune dominated (x2, y2, gc) buckets: for fixed (x2, y2), a larger gc' >= gc with f2' <= f2 dominates!
    # We keep only non-dominated buckets per (x2, y2) or sort buckets by f2.
    by_xy: dict[tuple[int, int], list[tuple[int, list[tuple[float, int, int]]]]] = {}
    for (x2, y2, gc), lst in pair_buckets.items():
        lst.sort(key=lambda t: t[0])
        by_xy.setdefault((x2, y2), []).append((gc, lst))

    # Build flat arrays of m=2 states: for each (x2, y2), keep Pareto-undominated (gc, best_f2, lst)
    p2_x, p2_y, p2_g, p2_f, p2_lst = [], [], [], [], []
    for (x2, y2), g_list in by_xy.items():
        # Sort by gc descending; maintain running top-k best f2s from >= gc
        g_list.sort(key=lambda t: -t[0])
        best_f_so_far = math.inf
        # Actually, since we want top_k distinct seams for each (K, D), keep bucket if its best f2 < best_f_of_larger_g
        # Wait: what if larger g has the SAME f2 (same pair)? Keeping top_k lowest f2 across >= gc could have duplicates,
        # so simply keeping any bucket whose lst[0][0] <= min_prev_f2 + 1e-12 (or keeping top-2 per g) is safe and tiny!
        # Let's keep a bucket if its top-3rd f2 improves on the 3rd-best f2 seen for larger g.
        top_seen: list[float] = []
        for gc, lst in g_list:
            # Check if lst adds anything better than top_seen[top_k-1]
            kth = top_seen[top_k - 1] if len(top_seen) >= top_k else math.inf
            if lst[0][0] < kth - 1e-12:
                p2_x.append(x2)
                p2_y.append(y2)
                p2_g.append(gc)
                p2_f.append(lst[0][0])
                p2_lst.append(lst)
                top_seen.extend(t[0] for t in lst)
                top_seen.sort()
                top_seen = top_seen[:top_k]

    if p2_x:
        ax = np.array(p2_x, dtype=np.int32)
        ay = np.array(p2_y, dtype=np.int32)
        ag = np.array(p2_g, dtype=np.int32)
        af = np.array(p2_f, dtype=np.float64)
        S3_max = 2 * K - 2  # sum(w+h) + 3 - 1 <= 2K <=> sum(w+h) <= 2K-2

        for k in range(M):
            wk, hk, gk, fk = int(ws[k]), int(hs[k]), int(gs[k]), float(fs[k])
            ok = (ax + wk <= K) & (ay + hk <= K) & (ax + ay + wk + hk <= S3_max) & (af <= f_global_max + 1e-12)
            # Only consider pairs where max(f2, fk) <= f_global_max
            idxs = np.nonzero(ok)[0]
            if idxs.size == 0:
                continue
            g3_arr = ag[idxs] + gk
            f3_arr = np.maximum(af[idxs], fk)
            for D in D_list:
                flim = f_limits[D]
                # Prune against worst in heap if heap is full
                if len(d_heaps[D]) >= top_k:
                    flim = min(flim, -d_heaps[D][0][0] - 1e-12)
                if fk > flim + 1e-12:
                    continue
                m_ok = (g3_arr >= D) & (f3_arr <= flim + 1e-12)
                for p_idx in idxs[m_ok]:
                    for f2, i, j in p2_lst[int(p_idx)]:
                        f3 = max(f2, fk)
                        if f3 > flim + 1e-12:
                            break
                        push_cand(D, f3, (i, j, k))

    results = []
    for D in D_list:
        h = d_heaps[D]
        if not h:
            continue
        n = K * K + D
        inc = L_inc[n]
        cands = sorted(h, key=lambda t: (-t[0], t[1]))
        seams = []
        for neg_f, m_cnt, idxs in cands:
            f_val = -neg_f
            dL_rigid = (K + f_val) - inc
            comp = [
                {
                    "w": elig[idx]["w"],
                    "h": elig[idx]["h"],
                    "g": elig[idx]["g"],
                    "f": elig[idx]["f"],
                    "src": elig[idx]["src"],
                    "tag": elig[idx]["tag"],
                    "transposed": bool(elig[idx].get("transposed", False)),
                }
                for idx in idxs
            ]
            seams.append({
                "m": len(comp),
                "f": f_val,
                "L_rigid": K + f_val,
                "dL_rigid": dL_rigid,
                "comp": comp,
            })
        results.append({
            "n": n,
            "K": K,
            "D": D,
            "L_inc": inc,
            "best_dL_rigid": seams[0]["dL_rigid"],
            "best_L_rigid": seams[0]["L_rigid"],
            "best_m": seams[0]["m"],
            "seams": seams,
        })
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--pockets", nargs="+", required=True)
    ap.add_argument("--gadlib")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kmin", type=int, default=16)
    ap.add_argument("--kmax", type=int, default=31)
    ap.add_argument("--slack-max", type=float, default=6.0e-3)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    rep = json.loads(Path(args.report).read_text())["results"]
    L_inc = {int(r["n"]): float(r["L"]) for r in rep}

    raw_items: list[dict] = []
    if args.gadlib and Path(args.gadlib).exists():
        raw_items.extend(items_from_gadlib(args.gadlib))
    raw_items.extend(items_from_pockets(args.pockets))

    # Keep smallest f per (w, h, g)
    by_whg: dict[tuple[int, int, int], dict] = {}
    for it in raw_items:
        k = (it["w"], it["h"], it["g"])
        if k not in by_whg or it["f"] < by_whg[k]["f"] - 1e-12:
            by_whg[k] = it
    items = pareto(list(by_whg.values()))
    print(f"Loaded {len(by_whg)} distinct (w,h,g) items -> {len(items)} Pareto items", flush=True)

    all_cells = []
    for K in range(args.kmin, args.kmax + 1):
        res = solve_for_K(K, items, L_inc, slack_max=args.slack_max, top_k=args.top_k)
        all_cells.extend(res)
        print(f"  K={K}: {len(res)} cells with dL_rigid <= {args.slack_max:+.1e}", flush=True)

    all_cells.sort(key=lambda r: (r["best_dL_rigid"], -r["n"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"cells": all_cells, "slack_max": args.slack_max}, indent=1))
    print(f"Total qualifying (K, D) cells: {len(all_cells)} (elapsed {time.time() - t0:.2f}s)", flush=True)
    for c in all_cells[:25]:
        s0 = c["seams"][0]
        desc = " + ".join(f"({p['w']},{p['h']},+{p['g']})<{p['src']}>" for p in s0["comp"])
        print(f"  n={c['n']} (K={c['K']},D={c['D']}) dL_rigid={c['best_dL_rigid']:+.6f} m={s0['m']} L_inc={c['L_inc']:.6f}  {desc}")


if __name__ == "__main__":
    main()
