#!/usr/bin/env python3
"""H114 (vstrip852) Scale-Up across D = 11..15, K = 26..31:
2-pocket staircase DP (zero-overlap) + 1-corner melted pinwheels + lpslack + pslp + mechwalk.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vstrip_lib as vl  # noqa: E402
import gstep_lib as gl  # noqa: E402
import build_and_polish as bp  # noqa: E402
import build_cert as bc  # noqa: E402
import homocasc2_lib as hlib  # noqa: E402
import lpslack  # noqa: E402
import union as auto_union  # noqa: E402
import utils as hu  # noqa: E402

sys.path.insert(0, str(vl.REPO_ROOT / "poc" / "row_ratchet_980" / "poc"))
import pslp  # noqa: E402

WINS_DIR = vl.DATA_DIR / "wins"


def build_union_incumbent(target_ns: set[int]) -> dict[int, dict]:
    """Load exact origin/main frontier (merged.csv + all poc/*/final.csv excluding swarm_vstrip852)."""
    return gl.fast_load_true_frontier(target_ns, certify=True)


def load_all_items() -> tuple[list[dict], dict[tuple[int, int, int, str], dict]]:
    """Load vstrip852 bricks, gstep bricks, and origin/main Pareto pockets (with transposes)."""
    items: list[dict] = []
    by_tag: dict[tuple[int, int, int, str], dict] = {}

    def _add_brick(b: dict, source_group: str) -> None:
        for op, suf in (("id", ""), ("T", "_T")):
            bo = b if op == "id" else gl.bl.transform_brick(b, "T")
            tag = f"{source_group}:{b['tag']}{suf}"
            bo = dict(bo)
            bo["tag"] = tag
            bo["is_vstrip"] = (source_group == "vstrip852")
            items.append(bo)
            by_tag[(bo["w"], bo["h"], bo["g"], tag)] = bo

    for p in sorted(vl.DATA_DIR.glob("brick_*.json")):
        nm = p.stem.replace("brick_", "")
        b = vl.load_brick(p, nm)
        _add_brick(b, "vstrip852")

    for p in sorted((vl.REPO_ROOT / "poc" / "swarm_gstep" / "data").glob("brick_*.json")):
        nm = p.stem.replace("brick_", "")
        b = gl.load_brick_json(p, nm)
        _add_brick(b, "gstep")

    # Also load pre-existing pockets from pockets_pareto_e3_sane.json & pockets_partb_sane.json
    # Resolve their squares only on demand when selected!
    rows_cache: dict[int, list[dict]] | None = None
    pocket_paths = [
        vl.REPO_ROOT / "poc" / "pocket_lift_scan3" / "data" / "pockets_pareto_e3_sane.json",
        vl.REPO_ROOT / "poc" / "swarm_liftpolish3" / "data" / "pockets_partb_sane.json",
    ]
    seen_pk: dict[tuple[int, int, int], dict] = {}
    for pp in pocket_paths:
        if not pp.exists():
            continue
        for g in json.loads(pp.read_text(encoding="utf-8")):
            src = int(g["src"])
            if src >= 2000:
                continue  # already loaded from gstep/data
            gw, gh, gg, gf = int(g["w"]), int(g["h"]), int(g["g"]), float(g["f"])
            if gf > 0.76 or max(gw, gh) > 25:
                continue
            for tr, (w, h) in ((False, (gw, gh)), (True, (gh, gw))):
                k = (w, h, gg)
                if k not in seen_pk or gf < seen_pk[k]["f"] - 1e-12:
                    seen_pk[k] = {
                        "w": w, "h": h, "g": gg, "f": gf, "fx": gf, "fy": gf,
                        "src": src, "tr": tr, "pocket": g,
                        "tag": f"pk:n{src}{'T' if tr else ''}",
                        "is_vstrip": False,
                    }

    # Keep Pareto-optimal items per (w, h, g) (top-2 per shape from bricks + best pocket)
    by_shape: dict[tuple[int, int, int], list[dict]] = collections.defaultdict(list)
    for it in items:
        by_shape[(it["w"], it["h"], it["g"])].append(it)
    for k, pk in seen_pk.items():
        by_shape[k].append(pk)

    kept: list[dict] = []
    for k, lst in sorted(by_shape.items()):
        lst.sort(key=lambda r: (r["f"], 0 if r["is_vstrip"] else 1))
        # Keep up to 2 best, ensuring at least 1 vstrip852 brick if within +0.008 of best
        top = lst[:2]
        top_tags = {r["tag"] for r in top}
        vs_only = [r for r in lst if r["is_vstrip"]]
        if vs_only and vs_only[0]["tag"] not in top_tags and vs_only[0]["f"] <= top[0]["f"] + 0.008:
            top.append(vs_only[0])
        kept.extend(top)
    return kept, by_tag


def materialize_brick(it: dict, donor_rows: dict[int, list[dict]]) -> dict | None:
    if "x" in it:
        return it
    src = int(it["src"])
    if src not in donor_rows:
        return None
    sq, W, H = bc.pocket_setup(donor_rows[src], it["pocket"], bool(it["tr"]))
    xs = np.array([float(s["x"]) for s in sq], dtype=np.float64)
    ys = np.array([float(s["y"]) for s in sq], dtype=np.float64)
    ths = np.array([float(s["angle"]) for s in sq], dtype=np.float64)
    return gl.make_brick(it["w"], it["h"], it["g"], xs, ys, ths, tag=it["tag"])


def build_pinwheel_layout(
    m: int, n: int, b1: dict, b2: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Build 2-brick layout inside m x m grid (supports g1+g2 >= n - m^2 by dropping extra outer grid cells)."""
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    if w1 + w2 > m or h1 + h2 > m or w1 + w2 + h1 + h2 > 2 * m:
        return None
    f = max(b1["f"], b2["f"])
    L = m + f
    dx1 = 0.0
    dy1 = (m - h1) + (f - b1["fy"])
    dx2 = (m - w2) + (f - b2["fx"])
    dy2 = 0.0

    gx, gy = [], []
    for i in range(m):
        for j in range(m):
            if i < w1 and j >= m - h1:
                continue
            if i >= m - w2 and j < h2:
                continue
            if i < m - w2 and j < m - h1:
                gx.append(i + 0.5)
                gy.append(j + 0.5)
            else:
                gx.append(i + f + 0.5)
                gy.append(j + f + 0.5)

    n_b = len(b1["x"]) + len(b2["x"])
    need_g = n - n_b
    if need_g < 0 or need_g > len(gx):
        return None
    if need_g < len(gx):
        # Drop cells closest to the boundary corner or farthest from origin (same as softseamdp)
        order = sorted(range(len(gx)), key=lambda idx: -(gx[idx] + gy[idx]))
        drop = set(order[: len(gx) - need_g])
        gx = [gx[idx] for idx in range(len(gx)) if idx not in drop]
        gy = [gy[idx] for idx in range(len(gy)) if idx not in drop]

    ax = np.concatenate([b1["x"] + dx1, b2["x"] + dx2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([b1["y"] + dy1, b2["y"] + dy2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([b1["th"], b2["th"], np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != n:
        return None
    seam = (
        ((ax >= w1 - 2.2) & (ax <= w1 + f + 2.2))
        | ((ay >= m - h1 - 2.2) & (ay <= m - h1 + 2.2))
        | ((ax >= m - w2 - 2.2) & (ax <= m - w2 + 2.2))
        | ((ay >= h2 - 2.2) & (ay <= h2 + f + 2.2))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def safe_certify(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, inc_dec: Decimal, label: str = "") -> dict | None:
    x, y, th, _ = hlib.norm_pack(x, y, th)
    for eps in (0.0, 1e-12, 1e-10, 1e-8, 1e-7):
        sc = 1.0 + eps
        c = hlib.certify_pack_against_frontier(n, x * sc, y * sc, th, inc_dec)
        if c is not None:
            c["label"] = label
            return c
    return None


def polish_candidate(n: int, K: int, cert: dict, inc_dec: Decimal, rng: np.random.Generator, label: str) -> dict:
    best = cert
    best_L = Decimal(max(best["L_lit"], best["L_pd"]))
    bx, by, bt, _ = hlib._rows_to_arrays(best["rows"])

    # 1. lpslack
    c_lp = bp._run_lpslack_fast(
        n, bx, by, bt, inc_dec, cycles=2, outer_iters=10, fixed_rounds=8, slp_modes=("dual_non90", "non90"), time_limit=12.0
    )
    if c_lp is not None and Decimal(max(c_lp["L_lit"], c_lp["L_pd"])) < best_L:
        best = c_lp
        best_L = Decimal(max(best["L_lit"], best["L_pd"]))
        best["label"] = label + "+lp"
        bx, by, bt, _ = hlib._rows_to_arrays(best["rows"])

    # 2. pslp full-row kink-safe squeeze
    try:
        x0, x1, y0, y1 = pslp.box_of(bx, by, bt)
        X, Y = bx - x0, by - y0
        f0 = max(x1 - x0 - K, y1 - y0 - K)
        fn, Xn, Yn, Tn, vn, _ = pslp.squeeze(X, Y, bt, K, K, f0)
        if vn <= 1e-12 and K + fn < float(best_L):
            c_ps = safe_certify(n, Xn, Yn, Tn, inc_dec, label + "+pslp")
            if c_ps is not None and Decimal(max(c_ps["L_lit"], c_ps["L_pd"])) < best_L:
                best = c_ps
                best_L = Decimal(max(best["L_lit"], best["L_pd"]))
                bx, by, bt, _ = hlib._rows_to_arrays(best["rows"])
    except Exception:
        pass

    # 3. mechwalk + lpslack only if promising (< +1.5e-3 from incumbent)
    if best["dL"] < 1.5e-3:
        mw_cert, _ = hlib.mechwalk_compress(n, bx, by, bt, inc_dec, rng, rounds=2, threads=1)
        if mw_cert is not None and Decimal(max(mw_cert["L_lit"], mw_cert["L_pd"])) < best_L:
            best = mw_cert
            best_L = Decimal(max(best["L_lit"], best["L_pd"]))
            best["label"] = label + "+mw"
            bx, by, bt, _ = hlib._rows_to_arrays(best["rows"])
            c_lp2 = bp._run_lpslack_fast(
                n, bx, by, bt, inc_dec, cycles=2, outer_iters=10, fixed_rounds=8, slp_modes=("dual_non90", "non90"), time_limit=12.0
            )
            if c_lp2 is not None and Decimal(max(c_lp2["L_lit"], c_lp2["L_pd"])) < best_L:
                best = c_lp2
                best["label"] = label + "+mw+lp"
    return best


def solve_cell_task(args_t: tuple) -> dict:
    n, K, D, inc_str, pairs_spec, donor_rows_sub = args_t
    t0 = time.time()
    inc_dec = Decimal(inc_str)
    rng = np.random.default_rng(70001 + n * 31)
    best_cert: dict | None = None
    tried = 0

    for pidx, (dL_pred, kind, b1_raw, b2_raw) in enumerate(pairs_spec):
        b1_base = materialize_brick(b1_raw, donor_rows_sub)
        b2_base = materialize_brick(b2_raw, donor_rows_sub)
        if b1_base is None or b2_base is None:
            continue
        tried += 1
        tag = f"n{n}_{kind}_{b1_base['w']}x{b1_base['h']}+{b2_base['w']}x{b2_base['h']}_p{pidx}"

        if kind == "zero":
            # Try orientations directly; if small pen, melt
            cand_cert = None
            for m1 in ("id", "flipx", "flipy", "rot180"):
                b1 = gl.bl.transform_brick(b1_base, m1)
                for m2 in ("id", "flipx", "flipy", "rot180"):
                    b2 = gl.bl.transform_brick(b2_base, m2)
                    built = build_pinwheel_layout(K, n, b1, b2)
                    if built is None:
                        continue
                    ax, ay, at, _, _ = built
                    pen, _ = gl.cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-8:
                        c = safe_certify(n, ax, ay, at, inc_dec, f"{tag}_{m1}_{m2}")
                        if c is not None:
                            cand_cert = c
                            break
                if cand_cert is not None:
                    break
            if cand_cert is None:
                built = build_pinwheel_layout(K, n, b1_base, b2_base)
                if built is not None:
                    ax, ay, at, _, _ = built
                    melted = bp.melt_overlapping_seam(n, ax, ay, at, len(b1_base["x"]) + len(b2_base["x"]), inc_dec, rng)
                    if melted is not None:
                        cand_cert = safe_certify(n, melted[0], melted[1], melted[2], inc_dec, f"{tag}_melt")
            if cand_cert is not None:
                cand_cert = polish_candidate(n, K, cand_cert, inc_dec, rng, cand_cert["label"])
                if best_cert is None or Decimal(max(cand_cert["L_lit"], cand_cert["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                    best_cert = cand_cert
        else:
            # 1-corner overlap: C++ batch melt -> bp.melt_overlapping_seam -> polish_candidate
            seeds, masks, scheds, labels = [], [], [], []
            for m1 in ("id", "flipx", "flipy", "rot180"):
                b1 = gl.bl.transform_brick(b1_base, m1)
                for m2 in ("id", "flipx", "flipy", "rot180"):
                    if (m1, m2) not in (("id", "id"), ("flipx", "flipy"), ("id", "rot180"), ("flipy", "rot180"), ("rot180", "rot180"), ("flipx", "rot180")):
                        continue
                    b2 = gl.bl.transform_brick(b2_base, m2)
                    built = build_pinwheel_layout(K, n, b1, b2)
                    if built is None:
                        continue
                    ax, ay, at, L_tgt, sm = built
                    seeds.append((ax, ay, at))
                    masks.append(sm)
                    scheds.append([L_tgt + 0.024, L_tgt + 0.008, L_tgt + 0.002])
                    labels.append(f"{tag}_{m1}_{m2}")
            if not seeds:
                continue
            relaxed = gl.bl.su.cpp_batch_melt(
                seeds, scheds, sym_mode=1, a_start=0.955, n_melt_stages=3,
                seam_masks=masks, seam_iters=180, full_iters=240, n_ruin_iters=0, n_threads=1, base_seed=7000 + n + pidx,
            )
            pool = []
            for bi, row in enumerate(relaxed):
                nx, ny, nth, Lm = gl.bl.normalize(row["x"], row["y"], row["th"])
                if row["best_L_feas"] is not None:
                    pool.append((0.0, Lm, nx, ny, nth, labels[bi]))
                elif row["pen"] < 0.05:
                    pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, labels[bi]))
            pool.sort(key=lambda t: (t[0], t[1]))
            for _, _, cx, cy, cth, lbl in pool[:2]:
                pen, _ = gl.cpp_eval_sat_pen_bbox(cx, cy, cth)
                if pen > 1e-8:
                    # Relax via lbfgs + joint SLP
                    xr, yr, tr, pr = hu.relax_lbfgs(cx, cy, cth, max(float(inc_dec) + 0.006, gl.bl.rect_bbox(cx, cy, cth)[2] + 2.0 * pen), free_mask=None, maxiter=250, w_wall=4.0)
                    if pr < 0.02:
                        xs, ys, ts, _, ok = bp.joint_xytheta_slp(xr, yr, tr, rounds=10, tr_th0=0.02, tr_xy0=0.08, margin=1e-8, tol=1e-9, patience=2)
                        if ok:
                            cx, cy, cth = xs, ys, ts
                c = safe_certify(n, cx, cy, cth, inc_dec, lbl)
                if c is not None:
                    c = polish_candidate(n, K, c, inc_dec, rng, lbl)
                    if best_cert is None or Decimal(max(c["L_lit"], c["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                        best_cert = c

        if best_cert is not None and best_cert["is_real_win"]:
            WINS_DIR.mkdir(parents=True, exist_ok=True)
            win_p = WINS_DIR / f"win_n{n}.json"
            prev_ok = False
            if win_p.exists():
                prev = json.loads(win_p.read_text(encoding="utf-8"))
                if Decimal(max(prev["L_lit"], prev["L_pd"])) <= Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                    prev_ok = True
            if not prev_ok:
                win_p.write_text(json.dumps({
                    "n": n, "K": K, "D": D, "label": best_cert["label"],
                    "L_lit": best_cert["L_lit"], "L_pd": best_cert["L_pd"],
                    "dL": best_cert["dL"], "rows": best_cert["rows"],
                }, indent=1), encoding="utf-8")
            if best_cert["dL"] <= -5e-4:
                break

    res = {
        "n": n, "K": K, "D": D, "L_inc": inc_str,
        "tried": tried, "elapsed_s": round(time.time() - t0, 1),
        "win": bool(best_cert is not None and best_cert["is_real_win"]),
        "dL": best_cert["dL"] if best_cert is not None else None,
        "L_lit": best_cert["L_lit"] if best_cert is not None else None,
        "label": best_cert.get("label") if best_cert is not None else None,
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min", type=int, default=26)
    ap.add_argument("--k-max", type=int, default=31)
    ap.add_argument("--d-min", type=int, default=11)
    ap.add_argument("--d-max", type=int, default=15)
    ap.add_argument("--n-list", type=str, default="")
    ap.add_argument("--slack-zero", type=float, default=0.008)
    ap.add_argument("--slack-corner", type=float, default=0.005)
    ap.add_argument("--max-pairs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    if args.n_list:
        target_ns = sorted({int(x) for x in args.n_list.split(",") if x.strip()})
    else:
        target_ns = sorted({
            K * K + D
            for K in range(args.k_min, args.k_max + 1)
            for D in range(args.d_min, args.d_max + 1)
            if K * K + D <= 1000
        })

    print(f"Loading true frontier for {len(target_ns)} target ns...", flush=True)
    frontier = build_union_incumbent(set(target_ns))
    items, _ = load_all_items()
    print(f"Loaded {len(items)} items ({sum(1 for it in items if it['is_vstrip'])} from vstrip852)", flush=True)

    # Collect needed donor srcs for pocket items
    need_srcs = {int(it["src"]) for it in items if "src" in it}
    donor_rows: dict[int, list[dict]] = {}
    if need_srcs:
        src_front = gl.fast_load_true_frontier(need_srcs, certify=False)
        for src, rec in src_front.items():
            x, y, th = rec["x"], rec["y"], rec["th"]
            donor_rows[src] = [
                {"square_id": str(i), "x": repr(float(x[i])), "y": repr(float(y[i])), "angle": repr(float(th[i]))}
                for i in range(len(x))
            ]

    tasks = []
    for n in target_ns:
        K = int(math.isqrt(n))
        while (K + 1) * (K + 1) <= n:
            K += 1
        D = n - K * K
        inc_dec = frontier[n]["L_dec"]
        f_inc = float(inc_dec) - K

        cands = []
        seen_pairs = set()
        for i, b1 in enumerate(items):
            for j in range(i, len(items)):
                b2 = items[j]
                if not (b1["is_vstrip"] or b2["is_vstrip"]):
                    continue
                if b1["w"] + b2["w"] > K or b1["h"] + b2["h"] > K:
                    continue
                swh = b1["w"] + b2["w"] + b1["h"] + b2["h"]
                if swh > 2 * K:
                    continue
                if b1["g"] + b2["g"] < D:
                    continue
                kind = "zero" if swh <= 2 * K - 1 else "corner"
                if kind == "corner" and b1["g"] + b2["g"] != D:
                    continue
                fmax = max(b1["f"], b2["f"])
                dL_pred = fmax - f_inc
                lim = args.slack_zero if kind == "zero" else args.slack_corner
                if dL_pred > lim:
                    continue
                pkey = tuple(sorted([(b1["w"], b1["h"], b1["g"]), (b2["w"], b2["h"], b2["g"])])) + (kind,)
                if pkey in seen_pairs:
                    continue
                seen_pairs.add(pkey)
                cands.append((dL_pred, 0 if kind == "zero" else 1, b1, b2, kind))

        cands.sort(key=lambda t: (t[0], t[1]))
        if not cands:
            continue
        pairs_spec = [(dLp, knd, b1, b2) for dLp, _, b1, b2, knd in cands[: args.max_pairs]]
        sub_srcs = {int(b["src"]) for _, _, b1, b2 in pairs_spec for b in (b1, b2) if "src" in b}
        sub_rows = {s: donor_rows[s] for s in sub_srcs if s in donor_rows}
        print(
            f"  Plan n={n} (K={K},D={D}, f_inc={f_inc:.6f}): {len(cands)} cands -> top {len(pairs_spec)}: "
            + ", ".join(f"{knd}:{b1['w']}x{b1['h']}+{b2['w']}x{b2['h']}({dLp:+.4f})" for dLp, knd, b1, b2 in pairs_spec),
            flush=True,
        )
        tasks.append((n, K, D, str(inc_dec), pairs_spec, sub_rows))

    print(f"\nRunning {len(tasks)} cells with max_workers={min(2, args.workers)}...", flush=True)
    results = []
    if min(2, args.workers) <= 1 or len(tasks) <= 1:
        for t in tasks:
            r = solve_cell_task(t)
            results.append(r)
            print(f"  [CELL] n={r['n']}: win={r['win']} dL={r['dL']} ({r['elapsed_s']}s) {r['label']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(2, args.workers)) as pool:
            futs = [pool.submit(solve_cell_task, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                print(f"  [CELL] n={r['n']}: win={r['win']} dL={r['dL']} ({r['elapsed_s']}s) {r['label']}", flush=True)

    results.sort(key=lambda r: r["n"])
    vl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (vl.DOCS_DIR / "scale_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    wins = [r for r in results if r["win"]]
    print(f"\nDONE in {time.time()-t0:.1f}s: {len(wins)}/{len(results)} certified wins!", flush=True)


if __name__ == "__main__":
    main()
