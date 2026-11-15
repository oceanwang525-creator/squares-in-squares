#!/usr/bin/env python3
"""H107 Scale-Up Sweep across top qualifying diagonal (m+1, r+k) -> (m, r) pairs in m=12..30."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_diagonal_pairs as adp  # noqa: E402
import cascade_engine as ce  # noqa: E402
import corrcasc_lib as cc  # noqa: E402


def diagonal_seeds(nt, nd, xd, yd, td, Ld, L_inc, step=1.0, max_seeds=8):
    """Seeds from exact dn = nd - nt diagonal L-border cuts (bbox / cent / armtip rules)."""
    hd = cc.hx_of(td)
    tilt = cc.dominant_tilt(td)
    Lt = Ld - step
    Lt_use = min(L_inc - 3e-4, Lt)
    dn = nd - nt
    corners = {
        "BL": (0.0, 0.0, (xd + hd) - Lt, (yd + hd) - Lt, xd - Lt, yd - Lt),
        "TL": (0.0, -step, (xd + hd) - Lt, step - (yd - hd), xd - Lt, step - yd),
        "BR": (-step, 0.0, step - (xd - hd), (yd + hd) - Lt, step - xd, yd - Lt),
        "TR": (-step, -step, step - (xd - hd), step - (yd - hd), step - xd, step - yd),
    }
    deg = np.degrees(td) % 90.0
    tilted = (deg > 1.0) & (deg < 89.0)
    cuts = adp.audit_l_borders(xd, yd, td, Ld, step, dn)
    cuts_rt = sorted([c for c in cuts if c["rem_tilt"] <= 3 and c["n_prot"] <= 2], key=lambda c: (c["n_prot"], c["score"]))
    picked = []
    seen_types = set()
    for c in cuts_rt[:3] + cuts[:max_seeds]:
        if c["type"] in seen_types or c["score"] > 0.35:
            continue
        seen_types.add(c["type"])
        picked.append(c)

    seeds = []
    for c in picked:
        dx, dy, px, py, cx, cy = corners[c["corner"]]
        ws_bbox = np.maximum(px, py)
        if c["rule"] == "bbox":
            sv = ws_bbox
        elif c["rule"] == "cent":
            sv = np.maximum(cx, cy)
        else:
            bonus = np.where((ws_bbox > -0.6) & tilted, 0.55, 0.0)
            sv = np.where(ws_bbox > 0.01, ws_bbox + 10.0, ws_bbox + bonus)
        rem_idx = np.argsort(-sv)[:dn]
        keep = np.ones(len(xd), dtype=bool)
        keep[rem_idx] = False
        xk = xd[keep] + dx
        yk = yd[keep] + dy
        tk = td[keep].copy()
        hk = hd[keep]
        prot = np.maximum.reduce([-(xk - hk), (xk + hk) - Lt, -(yk - hk), (yk + hk) - Lt])
        p_idx = np.where(prot > 1e-6)[0]
        tag = f"diag_{nd}_{c['type']}"
        if len(p_idx) == 0:
            seeds.append((f"{tag}_clean0", xk, yk, tk, Lt_use, np.array([[0.5 * Lt, 0.5 * Lt]])))
            continue
        pts = np.c_[xk[p_idx], yk[p_idx]]
        seeds.append((f"{tag}_raw", xk.copy(), yk.copy(), tk.copy(), Lt_use, pts))
        xc, yc = xk.copy(), yk.copy()
        xc[p_idx] = np.clip(xc[p_idx], hk[p_idx] + 1e-4, Lt - hk[p_idx] - 1e-4)
        yc[p_idx] = np.clip(yc[p_idx], hk[p_idx] + 1e-4, Lt - hk[p_idx] - 1e-4)
        d2 = ((xc[:, None] - pts[None, :, 0]) ** 2 + (yc[:, None] - pts[None, :, 1]) ** 2).min(axis=1)
        L_rep = min(L_inc + 0.002, max(Lt + 0.015, L_inc - 0.004))
        xr, yr, tr, _ = cc.repair_overlaps(xc, yc, tk.copy(), L_rep, free_mask=(d2 <= 4.5 ** 2), maxiter=240)
        seeds.append((f"{tag}_clampRep", xr, yr, tr, Lt_use, pts))
        if len(p_idx) <= 3:
            non_p = np.ones(len(xk), dtype=bool)
            non_p[p_idx] = False
            vp = cc.clearance_points(xk[non_p], yk[non_p], tk[non_p], Lt, len(p_idx) + 3, step=0.10, min_sep=0.80)
            if len(vp) >= len(p_idx):
                for atag, aval in (("u0", 0.0), ("tilt", tilt)):
                    xv, yv, tv = xk.copy(), yk.copy(), tk.copy()
                    vpts = []
                    for ki, si in enumerate(p_idx):
                        vx, vy, _ = vp[ki]
                        xv[si], yv[si], tv[si] = vx, vy, aval
                        vpts.append([vx, vy])
                    ctrs = np.vstack([pts, np.array(vpts)])
                    seeds.append((f"{tag}_reloc_{atag}", xv, yv, tv, Lt_use, ctrs))
    return seeds


def _prioritize(seeds, max_keep: int = 12):
    if len(seeds) <= max_keep:
        return seeds

    def score(s):
        name, x, y, th, _, _ = s
        pen, _ = cc.cpp_eval_sat_pen_bbox(x, y, th)
        tier = (
            0 if "dropProt" in name else
            1 if name.startswith("diag_") else
            2 if "clean" in name else
            3 if "relocRep" in name else
            4
        )
        return (tier, pen)

    return sorted(seeds, key=score)[:max_keep]


def select_scale_pairs(limit: int = 28) -> list[tuple[int, int, int]]:
    """Select top qualifying 1-step/2-step diagonal pairs (m+s, r+k) -> (m, r) excluding tested targets."""
    merged = cc.load_merged_fast()
    for wp in sorted(cc.DATA_DIR.glob("win_n*.json")):
        try:
            d = json.loads(wp.read_text())
            n = int(d["n"])
            x, y, th, L = cc._rows_to_arrays(d["rows"])
            if n not in merged or L < merged[n]["L"]:
                merged[n] = {"x": x, "y": y, "th": th, "L": L, "rows": d["rows"], "source": wp.name}
        except Exception:
            continue
    donors = targets = merged

    already_tested_nt = {
        758, 420, 650, 687, 602, 931, 383, 876, 936, 934,
        874, 873, 875, 870, 929, 814,
        703, 857, 816, 556, 597, 759, 690, 811, 423, 762,
        757, 603, 755, 604, 555,
        818, 930, 381, 465, 601, 308, 506, 505, 866, 274,
        815, 596, 241, 810, 463, 552, 553, 460, 868, 812,
        935, 211, 809, 705, 752,
    }
    by_nt: dict[int, tuple[tuple, tuple[int, int, int]]] = {}

    for m in range(12, 31):
        for r in range(1, 2 * m + 1):
            nt = m * m + r
            if nt in already_tested_nt or nt not in targets:
                continue
            L_inc = targets[nt]["L"]
            for step, ks in ((1, (1, 2, 3)), (2, (2, 3, 4))):
                for k in ks:
                    nd = (m + step) ** 2 + r + k
                    if nd > 1000 or nd not in donors:
                        continue
                    Ld = donors[nd]["L"]
                    dL_raw = (Ld - float(step)) - L_inc
                    if dL_raw > -5e-4:
                        continue
                    res = adp.audit_l_borders(donors[nd]["x"], donors[nd]["y"], donors[nd]["th"], Ld, float(step), nd - nt)
                    best = res[0]
                    if best["score"] > 0.08:
                        continue
                    rank_key = (best["score"] > 0.05, dL_raw + 0.05 * best["score"])
                    if nt not in by_nt or rank_key < by_nt[nt][0]:
                        by_nt[nt] = (rank_key, (nt, nd, step))

    ranked = sorted(by_nt.values(), key=lambda item: item[0])
    return [pair for _, pair in ranked[:limit]]


def _run_scale_pair(args: tuple) -> dict:
    nt, nd, step, xd, yd, td, Ld, L_inc_dec, seed_int = args
    rng = np.random.default_rng(seed_int)
    L_inc = float(L_inc_dec)
    t0 = time.time()

    seeds = diagonal_seeds(nt, nd, xd, yd, td, Ld, L_inc, step=float(step), max_seeds=5)
    if step == 1:
        seeds += ce.generate_peel_seeds(nt, nd, xd, yd, td, Ld, L_inc, rng, layers=1, max_prot=20, max_rem_tilt=20)
        seeds += ce.generate_interior_cross_seeds(nt, nd, xd, yd, td, Ld, L_inc, max_seeds=4, max_rem_tilt=8)
    seeds = _prioritize(seeds, max_keep=10)
    print(f"[SCALE {nt}<-{nd} s={step}] L_inc={L_inc:.9f} Lt={Ld-step:.9f} dL_raw={(Ld-step)-L_inc:+.6f} seeds={len(seeds)} prep={time.time()-t0:.1f}s", flush=True)
    if not seeds:
        return {"n": nt, "donor": nd, "step": step, "L_inc": L_inc, "win": False, "n_seeds": 0}

    res = ce.solve_target(
        nt,
        seeds,
        L_inc_dec,
        threads=1,
        rng=rng,
        n_seeds_per_spec=3,
        masked_iters=140,
        full_iters=180,
        top_polish=5,
        run_mechwalk=False,
    )
    best_pack = res.get("best_pack")
    best_seen_L = res.get("best_seen_L")
    if best_pack is not None and best_seen_L is not None and best_seen_L < L_inc + 0.0075:
        bx, by, bt, bL = best_pack
        mw_rounds = 3 if best_seen_L < L_inc + 0.004 else 2
        print(f"[{nt}] Running {mw_rounds}-round mechwalk_compress + lpslack from best_seen_L={bL:.12f}...", flush=True)
        mw_cert, mw_pack = cc.mechwalk_compress(nt, bx, by, bt, L_inc_dec, rng, rounds=mw_rounds, threads=1)
        if mw_pack is not None and mw_pack[3] < best_seen_L:
            best_seen_L = mw_pack[3]
            res["best_seen_L"] = best_seen_L
            res["dL"] = best_seen_L - L_inc
        if mw_cert is not None and mw_cert.get("is_real_win"):
            if res["L_win"] is None or float(mw_cert["L_lit"]) < float(res["L_win"]):
                lbl = f"{res.get('best_label') or 'basin'}_mechwalk_lpslack"
                cc.save_win_and_rebuild_final(nt, mw_cert, lbl)
                res["win"] = True
                res["best_label"] = lbl
                res["L_win"] = mw_cert["L_lit"]
                res["dL"] = mw_cert["dL"]
                print(f"[{nt}] *** SCALE WIN (mechwalk+lpslack) *** {lbl} L={mw_cert['L_lit'][:20]} dL={mw_cert['dL']:+.6e}", flush=True)

    res["donor"] = nd
    res["step"] = step
    res["dL_raw"] = (Ld - step) - L_inc
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=28)
    ap.add_argument("--out", type=str, default="scale_results.json")
    args = ap.parse_args()

    cc.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    cc.DATA_DIR.mkdir(parents=True, exist_ok=True)

    pairs = select_scale_pairs(limit=args.limit)
    print(f"Selected {len(pairs)} scale pairs: {pairs}", flush=True)

    need = {n for nt, nd, _ in pairs for n in (nt, nd)}
    loaded = cc.load_portfolio(need)
    targets, donors = loaded["targets"], dict(loaded["donors"])
    for wp in sorted(cc.DATA_DIR.glob("win_n*.json")):
        try:
            d = json.loads(wp.read_text())
            dn = int(d["n"])
            if dn in need:
                dx, dy, dth, dL = cc._rows_to_arrays(d["rows"])
                if dn not in donors or dL < donors[dn]["L"]:
                    donors[dn] = {"x": dx, "y": dy, "th": dth, "L": dL, "rows": d["rows"], "source": wp.name}
        except Exception:
            continue

    specs = [
        (nt, nd, step, donors[nd]["x"], donors[nd]["y"], donors[nd]["th"], donors[nd]["L"], targets[nt]["L_dec"], 20281070 + i)
        for i, (nt, nd, step) in enumerate(pairs)
    ]

    t_start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for res in pool.map(_run_scale_pair, specs):
            results.append({k: v for k, v in res.items() if k != "best_pack"})
            (cc.DOCS_DIR / args.out).write_text(json.dumps(results, indent=2, default=str))
            r = results[-1]
            print(f"### SCALE DONE n={r['n']}<-{r['donor']}: win={r['win']} dL={r.get('dL')} ({r.get('elapsed_s')}s)", flush=True)

    wins = [r for r in results if r.get("win")]
    print(f"\n=== H107 SCALE COMPLETE in {time.time()-t_start:.1f}s | WINS={len(wins)}/{len(results)} ===", flush=True)
    for r in wins:
        print(f"  WIN n={r['n']}<-{r['donor']}: dL={r['dL']:+.6e} L={str(r['L_win'])[:22]}", flush=True)


if __name__ == "__main__":
    main()
