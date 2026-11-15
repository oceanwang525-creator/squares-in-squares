#!/usr/bin/env python3
"""Audit H107 Diagonal Peel Cascade pairs (m+1, r+k)->(m, r) and (m+2, r+k)->(m, r) across m=12..30."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corrcasc_lib as cc  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def eval_cut_metrics(
    xk: np.ndarray, yk: np.ndarray, tk: np.ndarray, Lt: float
) -> tuple[float, float, int, int, float]:
    """Return (max_prot, max_ov, n_prot, n_ov, score = max(max_prot, max_ov))."""
    hk = cc.hx_of(tk)
    prot_arr = np.maximum.reduce([-(xk - hk), (xk + hk) - Lt, -(yk - hk), (yk + hk) - Lt])
    n_prot = int((prot_arr > 1e-6).sum())
    max_prot = float(max(0.0, float(prot_arr.max()))) if len(prot_arr) else 0.0
    ov = cc.overlap_pairs(xk, yk, tk, tol=1e-6)
    n_ov = len(ov)
    max_ov = float(max((p for p, _, _ in ov), default=0.0))
    return max_prot, max_ov, n_prot, n_ov, max(max_prot, max_ov)


def fast_seam_overlap(
    xk: np.ndarray, yk: np.ndarray, tk: np.ndarray, xc: float, yc: float, step: float
) -> tuple[float, int]:
    """Check overlaps only on squares near the shifted xcross seam lines."""
    near = (np.abs(xk - (xc - 0.5 * step)) < 1.8) | (np.abs(yk - (yc - 0.5 * step)) < 1.8)
    if near.sum() < 2:
        return 0.0, 0
    ov = cc.overlap_pairs(xk[near], yk[near], tk[near], tol=1e-6)
    return float(max((p for p, _, _ in ov), default=0.0)), len(ov)


def audit_l_borders(
    xd: np.ndarray, yd: np.ndarray, td: np.ndarray, Ld: float, step: float, dn: int
) -> list[dict]:
    hd = cc.hx_of(td)
    Lt = Ld - step
    deg = np.degrees(td) % 90.0
    tilted = (deg > 1.0) & (deg < 89.0)
    results = []

    corners = [
        ("BL", 0.0, 0.0, (xd + hd) - Lt, (yd + hd) - Lt, xd - Lt, yd - Lt),
        ("TL", 0.0, -step, (xd + hd) - Lt, step - (yd - hd), xd - Lt, step - yd),
        ("BR", -step, 0.0, step - (xd - hd), (yd + hd) - Lt, step - xd, yd - Lt),
        ("TR", -step, -step, step - (xd - hd), step - (yd - hd), step - xd, step - yd),
    ]

    seen = set()
    for cname, dx, dy, px, py, cx, cy in corners:
        ws_bbox = np.maximum(px, py)
        ws_cent = np.maximum(cx, cy)
        tilted_bonus = np.where((ws_bbox > -0.6) & tilted, 0.55, 0.0)
        ws_arm = np.where(ws_bbox > 0.01, ws_bbox + 10.0, ws_bbox + tilted_bonus)

        for rname, score_vec in (("bbox", ws_bbox), ("cent", ws_cent), ("armtip", ws_arm)):
            rem_idx = np.argsort(-score_vec)[:dn]
            keep = np.ones(len(xd), dtype=bool)
            keep[rem_idx] = False
            sig = (cname, keep.tobytes())
            if sig in seen:
                continue
            seen.add(sig)
            xk = xd[keep] + dx
            yk = yd[keep] + dy
            tk = td[keep]
            rem_t = int(tilted[~keep].sum())
            max_prot, max_ov, n_prot, n_ov, score = eval_cut_metrics(xk, yk, tk, Lt)
            results.append(
                {
                    "type": f"L_{cname}_{rname}",
                    "corner": cname,
                    "rule": rname,
                    "rem_tilt": rem_t,
                    "max_prot": round(max_prot, 6),
                    "max_ov": round(max_ov, 6),
                    "n_prot": n_prot,
                    "n_ov": n_ov,
                    "score": round(score, 6),
                }
            )
    results.sort(key=lambda r: (r["score"], r["n_prot"] + r["n_ov"]))
    return results


def audit_xcross(
    xd: np.ndarray, yd: np.ndarray, td: np.ndarray, Ld: float, step: float, dn: int, n_target: int
) -> list[dict]:
    Lt = Ld - step
    deg = np.degrees(td) % 90.0
    tilted = (deg > 1.0) & (deg < 89.0)
    hd = cc.hx_of(td)
    half = 0.5 * step
    cuts = np.arange(0.75 * step + 0.25, Ld - 0.75 * step - 0.25, 0.50)

    # Precompute 1D masks for x and y
    x_masks = [(float(xc), np.abs(xd - xc) <= half) for xc in cuts]
    x_masks = [(xc, m) for xc, m in x_masks if int(m.sum()) <= dn + 4 and int((m & tilted).sum()) <= 10]
    y_masks = [(float(yc), np.abs(yd - yc) <= half) for yc in cuts]
    y_masks = [(yc, m) for yc, m in y_masks if int(m.sum()) <= dn + 4 and int((m & tilted).sum()) <= 10]

    raw_cands = []
    seen = set()
    for xc, in_x in x_masks:
        for yc, in_y in y_masks:
            rem0 = in_x | in_y
            c0 = int(rem0.sum())
            if c0 < dn - 4 or c0 > dn + 4:
                continue
            rem_t0 = int((rem0 & tilted).sum())
            if rem_t0 > 12:
                continue
            raw_cands.append((rem_t0, abs(c0 - dn), xc, yc, rem0, c0))

    raw_cands.sort(key=lambda t: (t[0], t[1]))
    cands = []
    for _, _, xc, yc, rem0, c0 in raw_cands[:30]:
        if c0 == dn:
            rem = rem0
        elif c0 < dn:
            need = dn - c0
            dist_strip = np.minimum(np.abs(xd - xc) - hd, np.abs(yd - yc) - hd)
            avail = np.where(~rem0)[0]
            add_idx = avail[np.argsort(dist_strip[avail])[:need]]
            rem = rem0.copy()
            rem[add_idx] = True
        else:
            drop_n = c0 - dn
            dist_cl = np.minimum(np.abs(xd - xc), np.abs(yd - yc))
            rem_idx = np.where(rem0)[0]
            keep_back = rem_idx[np.argsort(-dist_cl[rem_idx])[:drop_n]]
            rem = rem0.copy()
            rem[keep_back] = False

        keep = ~rem
        if int(keep.sum()) != n_target:
            continue
        sig = keep.tobytes()
        if sig in seen:
            continue
        seen.add(sig)

        xk = xd[keep].copy()
        yk = yd[keep].copy()
        tk = td[keep].copy()
        hk = hd[keep]
        xk[xk > xc] -= step
        yk[yk > yc] -= step
        rem_t = int(tilted[rem].sum())
        prot_arr = np.maximum.reduce([-(xk - hk), (xk + hk) - Lt, -(yk - hk), (yk + hk) - Lt])
        n_prot = int((prot_arr > 1e-6).sum())
        max_prot = float(max(0.0, float(prot_arr.max()))) if len(prot_arr) else 0.0
        max_ov, n_ov = fast_seam_overlap(xk, yk, tk, xc, yc, step)
        score = max(max_prot, max_ov)
        cands.append(
            {
                "type": f"xc({xc:.2f},{yc:.2f})",
                "xc": float(xc),
                "yc": float(yc),
                "rem_tilt": rem_t,
                "max_prot": round(max_prot, 6),
                "max_ov": round(max_ov, 6),
                "n_prot": n_prot,
                "n_ov": n_ov,
                "score": round(score, 6),
            }
        )
    cands.sort(key=lambda r: (r["score"], r["rem_tilt"], r["n_prot"] + r["n_ov"]))
    return cands


def main() -> None:
    t0 = time.time()
    print("Loading merged.csv (origin/main aea8dc0)...", flush=True)
    merged = cc.load_merged_fast()
    print(f"Loaded {len(merged)} packings in {time.time() - t0:.2f}s", flush=True)

    all_pairs = []
    # 1-step diagonal pairs: (m+1, r+k) -> (m, r), k in {1, 2, 3}
    for m in range(12, 31):
        for r in range(1, 2 * m + 1):
            nt = m * m + r
            for k in (1, 2, 3):
                nd = (m + 1) ** 2 + r + k
                if nd <= 1000:
                    all_pairs.append((1, m, r, k, nt, nd))

    # 2-step diagonal pairs: (m+2, r+k) -> (m, r), k in {2, 3, 4}
    for m in range(12, 30):
        for r in range(1, 2 * m + 1):
            nt = m * m + r
            for k in (2, 3, 4):
                nd = (m + 2) ** 2 + r + k
                if nd <= 1000:
                    all_pairs.append((2, m, r, k, nt, nd))

    print(f"Scanning {len(all_pairs)} candidate diagonal pairs...", flush=True)
    cond_a = []
    for step, m, r, k, nt, nd in all_pairs:
        Ld = merged[nd]["L"]
        L_inc = merged[nt]["L"]
        Lt = Ld - float(step)
        dL_raw = Lt - L_inc
        if dL_raw <= 0.005:
            cond_a.append((step, m, r, k, nt, nd, Ld, L_inc, Lt, dL_raw))

    print(f"Pairs meeting Condition (a) dL_raw <= +0.005: {len(cond_a)}", flush=True)
    cond_a.sort(key=lambda x: x[9])

    qualifying = []
    near_miss = []
    for idx, (step, m, r, k, nt, nd, Ld, L_inc, Lt, dL_raw) in enumerate(cond_a):
        d = merged[nd]
        xd, yd, td = d["x"], d["y"], d["th"]
        dn = nd - nt
        l_cuts = audit_l_borders(xd, yd, td, Ld, float(step), dn)
        xc_cuts = audit_xcross(xd, yd, td, Ld, float(step), dn, nt)
        best_l = l_cuts[0] if l_cuts else None
        best_x = xc_cuts[0] if xc_cuts else None
        candidates = [c for c in (best_l, best_x) if c is not None]
        best = min(candidates, key=lambda c: c["score"])
        rec = {
            "step": step,
            "m": m,
            "r": r,
            "k": k,
            "nt": nt,
            "nd": nd,
            "dn": dn,
            "Ld": round(Ld, 9),
            "Lt": round(Lt, 9),
            "L_inc": round(L_inc, 9),
            "dL_raw": round(dL_raw, 9),
            "best_cut": best,
            "best_L_cut": best_l,
            "best_xc_cut": best_x,
        }
        if best["score"] <= 0.12:
            qualifying.append(rec)
        else:
            near_miss.append(rec)
        if (idx + 1) % 500 == 0:
            print(f"  audited {idx+1}/{len(cond_a)} pairs ({time.time()-t0:.1f}s), qualifying={len(qualifying)}", flush=True)

    qualifying.sort(key=lambda x: (x["best_cut"]["score"], x["dL_raw"]))

    cc.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cc.DOCS_DIR / "audit_diagonal_pairs.json"
    out_path.write_text(
        json.dumps(
            {
                "total_scanned": len(all_pairs),
                "cond_a_count": len(cond_a),
                "qualifying_count": len(qualifying),
                "qualifying": qualifying,
                "near_miss_top30": sorted(near_miss, key=lambda x: (x["best_cut"]["score"], x["dL_raw"]))[:30],
            },
            indent=2,
        )
    )

    print(f"\nDone in {time.time()-t0:.1f}s. Qualifying pairs (dL_raw <= +0.005 AND score <= 0.12): {len(qualifying)}", flush=True)
    print(f"{'step':>4} {'t<-d':<10} {'(m,r,k)':<12} {'L_inc':>11} {'L_d-step':>11} {'dL_raw':>10} | {'best_cut':<22} {'score':>7} {'prot':>7} {'ov':>7} {'np':>3} {'no':>3} {'rt':>3}")
    for rec in qualifying[:40]:
        bc = rec["best_cut"]
        mrk = f"({rec['m']},{rec['r']},{rec['k']})"
        print(
            f"{rec['step']:>4} {rec['nt']}<-{rec['nd']:<5} {mrk:<12} {rec['L_inc']:>11.6f} {rec['Lt']:>11.6f} {rec['dL_raw']*1e3:>+9.2f}m | "
            f"{bc['type']:<22} {bc['score']:>7.4f} {bc['max_prot']:>7.4f} {bc['max_ov']:>7.4f} {bc['n_prot']:>3} {bc['n_ov']:>3} {bc['rem_tilt']:>3}"
        )

    # Also print top 20 qualifying with dL_raw < -1e-4 sorted by dL_raw
    neg_qual = [r for r in qualifying if r["dL_raw"] < -1e-4]
    neg_qual.sort(key=lambda x: (x["dL_raw"], x["best_cut"]["score"]))
    print(f"\nQualifying pairs with dL_raw < -0.1e-3 ({len(neg_qual)} total), top 25 by dL_raw:")
    for rec in neg_qual[:25]:
        bc = rec["best_cut"]
        mrk = f"({rec['m']},{rec['r']},{rec['k']})"
        print(
            f"{rec['step']:>4} {rec['nt']}<-{rec['nd']:<5} {mrk:<12} {rec['L_inc']:>11.6f} {rec['Lt']:>11.6f} {rec['dL_raw']*1e3:>+9.2f}m | "
            f"{bc['type']:<22} {bc['score']:>7.4f} {bc['max_prot']:>7.4f} {bc['max_ov']:>7.4f} {bc['n_prot']:>3} {bc['n_ov']:>3} {bc['rem_tilt']:>3}"
        )

    if len(qualifying) < 10:
        print("\nTop 20 near-miss pairs (dL_raw <= +0.005, score > 0.12):")
        for rec in sorted(near_miss, key=lambda x: (x["best_cut"]["score"], x["dL_raw"]))[:20]:
            bc = rec["best_cut"]
            mrk = f"({rec['m']},{rec['r']},{rec['k']})"
            print(
                f"{rec['step']:>4} {rec['nt']}<-{rec['nd']:<5} {mrk:<12} {rec['L_inc']:>11.6f} {rec['Lt']:>11.6f} {rec['dL_raw']*1e3:>+9.2f}m | "
                f"{bc['type']:<22} {bc['score']:>7.4f} {bc['max_prot']:>7.4f} {bc['max_ov']:>7.4f} {bc['n_prot']:>3} {bc['n_ov']:>3} {bc['rem_tilt']:>3}"
            )


if __name__ == "__main__":
    main()
