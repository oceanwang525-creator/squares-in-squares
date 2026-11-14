#!/usr/bin/env python3
"""H107 30-min gate: diagonal (m+1, r+k) -> (m, r) arm-tip + L-border peel, C++ masked melt, SLP, mechwalk, lpslack."""
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

DEFAULT_PAIRS = [
    (758, 814),  # (27,29,1) dL_raw=-14.21m, BL rt=1 np=1 sc=0.252, BR sc=0.061
    (420, 462),  # (20,20,1) dL_raw=-12.52m, BL rt=1 np=1 sc=0.256
    (650, 702),  # (25,25,1) dL_raw=-12.45m, BL rt=1 np=1 sc=0.266, BR sc=0.053
    (687, 741),  # (26,11,1) dL_raw=-10.15m, TR rt=1 np=1 sc=0.241, TL sc=0.063
    (602, 652),  # (24,26,1) dL_raw=-10.10m, TL sc=0.041, BL rt=1 np=1 sc=0.254
    (931, 993),  # (30,31,1) dL_raw=-6.74m, BR sc=0.031, TR rt=1 np=1 sc=0.234
    (383, 423),  # (19,22,1) dL_raw=-6.27m, BR sc=0.037
    (876, 936),  # (29,35,1) dL_raw=-2.04m, BR sc=0.015
]


def diagonal_seeds(nt, nd, xd, yd, td, Ld, L_inc, step=1.0, max_seeds=10):
    """Seeds from the audit's exact dn = nd - nt diagonal L-border cuts (bbox / cent / armtip rules)."""
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
    # Also include lowest rem_tilt cuts even if score up to 0.32
    cuts_rt = sorted([c for c in cuts if c["rem_tilt"] <= 3 and c["n_prot"] <= 2], key=lambda c: (c["n_prot"], c["score"]))
    picked = []
    seen_types = set()
    for c in cuts_rt[:4] + cuts[:max_seeds]:
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


def _prioritize(seeds, max_keep=24):
    if len(seeds) <= max_keep:
        return seeds

    def score(s):
        name, x, y, th, _, _ = s
        pen, _ = cc.cpp_eval_sat_pen_bbox(x, y, th)
        tier = (
            0 if name.startswith("diag_") else
            1 if "clean" in name else
            2 if "relocRep" in name else
            3 if "dropProt" in name else
            4 if "xcross" in name else
            5
        )
        return (tier, pen)

    return sorted(seeds, key=score)[:max_keep]


def _run_one_pair(args):
    nt, nd, xd, yd, td, Ld, L_inc_dec, seed_int = args
    rng = np.random.default_rng(seed_int)
    L_inc = float(L_inc_dec)
    t0 = time.time()
    seeds = diagonal_seeds(nt, nd, xd, yd, td, Ld, L_inc)
    seeds += ce.generate_peel_seeds(nt, nd, xd, yd, td, Ld, L_inc, rng, layers=1, max_prot=20, max_rem_tilt=20)
    seeds += ce.generate_interior_cross_seeds(nt, nd, xd, yd, td, Ld, L_inc, max_seeds=4, max_rem_tilt=12)
    seeds = _prioritize(seeds, max_keep=20)
    print(
        f"[{nt}<-{nd}] L_inc={L_inc:.9f} Lt={Ld-1.0:.9f} dL_raw={(Ld-1.0)-L_inc:+.6f} seeds={len(seeds)} "
        f"({', '.join(s[0] for s in seeds[:5])}...) prep={time.time()-t0:.1f}s",
        flush=True,
    )
    if not seeds:
        return {"n": nt, "donor": nd, "L_inc": L_inc, "win": False, "n_seeds": 0}
    res = ce.solve_target(
        nt, seeds, L_inc_dec, threads=1, rng=rng, n_seeds_per_spec=4,
        masked_iters=180, full_iters=220, top_polish=8, run_mechwalk=True,
    )
    res["donor"] = nd
    res["dL_raw"] = (Ld - 1.0) - L_inc
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default="")
    ap.add_argument("--out", type=str, default="gate_results.json")
    args = ap.parse_args()
    pairs = DEFAULT_PAIRS
    if args.pairs:
        pairs = [tuple(int(v) for v in p.split(":")) for p in args.pairs.split(",")]

    cc.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    cc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cc.rebuild_final_csv()

    need = {n for p in pairs for n in p}
    loaded = cc.load_portfolio(need)
    targets, donors = loaded["targets"], loaded["donors"]
    for n in sorted(need):
        print(f"Frontier n={n:4d}: L={targets[n]['L']:.12f} ({targets[n]['source']})", flush=True)

    specs = [
        (nt, nd, donors[nd]["x"], donors[nd]["y"], donors[nd]["th"], donors[nd]["L"], targets[nt]["L_dec"], 20271070 + i)
        for i, (nt, nd) in enumerate(pairs)
    ]
    t_start = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for res in pool.map(_run_one_pair, specs):
            results.append({k: v for k, v in res.items() if k != "best_pack"})
            (cc.DOCS_DIR / args.out).write_text(json.dumps(results, indent=2, default=str))
            r = results[-1]
            print(f"### DONE n={r['n']}<-{r['donor']}: win={r['win']} dL={r.get('dL')} feas={r.get('n_feas')}/{r.get('n_jobs')} ({r.get('elapsed_s')}s)", flush=True)

    print(f"\n=== H107 gate COMPLETE in {time.time()-t_start:.1f}s ===", flush=True)
    for r in results:
        print(f"n={r['n']}<-{r['donor']}: win={r['win']} dL_raw={r.get('dL_raw')} best_seen_L={r.get('best_seen_L')} dL={r.get('dL')}", flush=True)


if __name__ == "__main__":
    main()
