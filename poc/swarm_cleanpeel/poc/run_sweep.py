#!/usr/bin/env python3
"""H16 clean-peel sweep across cascade (362, 485) and prot=1..2 tilt=0 pairs using soft melt targeting L_inc-2e-4."""
from __future__ import annotations

import argparse
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cleanpeel_lib as cp  # noqa: E402
import peel_engine as pe  # noqa: E402
from cpp_wrap import cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402

DEFAULT_PAIRS = [
    # Cascade & small gaps
    (362, 401),
    (485, 530),
    # prot = 1 list + small-n top gaps
    (83, 102),
    (85, 104),
    (105, 126),
    (123, 146),
    (126, 149),
    (175, 202),
    (199, 228),
    (200, 229),
    (260, 293),
    (292, 327),
    (450, 493),
    (488, 533),
    # prot = 2 list
    (109, 130),
    (110, 131),
    (125, 148),
    (234, 265),
    (539, 586),
    (550, 597),
    (583, 632),
    (789, 846),
    (801, 858),
    (854, 913),
]


def run_target_soft(
    n_t: int,
    n_d: int,
    packs: dict,
    best_L: dict,
    rng: np.random.Generator,
    n_jobs_per_seed: int = 4,
    top_polish: int = 4,
) -> dict:
    xd, yd, td, Ld = packs[n_d]
    L_inc_dec = best_L[n_t]
    L_inc = float(L_inc_dec)
    t0 = time.time()

    seeds = pe.generate_cleanpeel_seeds(n_t, n_d, xd, yd, td, Ld, L_inc, rng, max_prot=3, allow_tilted_border=False)
    if not seeds:
        return {"n": n_t, "donor": n_d, "L_inc": L_inc, "status": "no_seeds", "win": False}

    # Cap seeds to top 12 to keep runtime crisp
    seeds = seeds[:12]
    allq, allm, alls, alll = [], [], [], []
    L0 = L_inc + 0.032
    sched = [L0, L_inc + 0.008, L_inc + 0.0015, L_inc - 2e-4]

    best_cert = None
    best_label = None
    best_seen_L = 1e18

    for sname, X, Y, T, Lt, ctrs in seeds:
        d2 = ((X[:, None] - ctrs[None, :, 0]) ** 2 + (Y[:, None] - ctrs[None, :, 1]) ** 2).min(axis=1)
        xr, yr, tr, pen_r = cp.repair_overlaps(X, Y, T, L_inc + 0.004, free_mask=(d2 <= 5.5 ** 2), maxiter=200)
        if pen_r < 1e-6:
            cert0 = cp.polish_candidate(n_t, xr, yr, tr, L_inc_dec)
            if cert0 is not None:
                best_seen_L = min(best_seen_L, cert0["L_float"])
                if cert0["is_real_win"] and (best_cert is None or Decimal(cert0["L_lit"]) < Decimal(best_cert["L_lit"])):
                    best_cert = cert0
                    best_label = f"{sname}_direct"
                    cp.save_win_and_rebuild_final(n_t, best_cert, best_label)
                    print(f"[{n_t}<-{n_d}] *** REAL WIN (direct) *** {best_label} L={best_cert['L_lit'][:20]} dL={best_cert['dL']:+.6e}", flush=True)

        off = 0.5 * (L0 - (L_inc + 0.004))
        xb, yb = xr + off, yr + off
        h = cp.hx_of(tr)
        xb = np.clip(xb, h + 1e-4, L0 - h - 1e-4)
        yb = np.clip(yb, h + 1e-4, L0 - h - 1e-4)
        for s in range(n_jobs_per_seed):
            R = 100.0 if s == n_jobs_per_seed - 1 else float(rng.uniform(5.8, 11.5))
            d2b = ((xb[:, None] - (ctrs[None, :, 0] + off)) ** 2 + (yb[:, None] - (ctrs[None, :, 1] + off)) ** 2).min(axis=1)
            near = (d2b <= R * R) if R < 50 else np.ones(n_t, dtype=bool)
            x2, y2, t2 = xb.copy(), yb.copy(), tr.copy()
            if s > 0:
                jit = 0.04 if R > 50 else 0.09
                k = int(near.sum())
                x2[near] += rng.uniform(-jit, jit, k)
                y2[near] += rng.uniform(-jit, jit, k)
                t2[near] += rng.uniform(-0.22, 0.22, k)
            allq.append(full_to_free_sym(x2, y2, t2, L0))
            mm = near.astype(np.uint8)
            allm.append(np.concatenate([mm, mm, mm]))
            alls.append(sched)
            alll.append(f"{sname}_R{R:.0f}_s{s}")

    qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
        n_t, 1, allq, allm, alls, [6, 10, 14, 18],
        a_start=0.72, n_melt_stages=4, masked_iters=200, full_iters=240,
        n_threads=4, base_seed=int(rng.integers(10000, 999999))
    )
    n_feas = int((lf < 1e9).sum())
    print(f"[{n_t}<-{n_d}] ({len(alll)} jobs, {time.time()-t0:.1f}s) feas={n_feas}/{len(lf)} best_Lf={float(lf.min()):.9f} min_pen={float(po.min()):.2e}", flush=True)
    order = sorted(range(len(alll)), key=lambda i: (0 if lf[i] < 1e9 else 1, lf[i] if lf[i] < 1e9 else po[i]))[:top_polish]
    for i in order:
        use_f = lf[i] < 1e9
        Lr = float(lf[i]) if use_f else alls[i][-1]
        x, y, th = cpp_expand_sym(n_t, 1, Lr, qf[i] if use_f else qb[i])
        cert = cp.polish_candidate(n_t, x, y, th, L_inc_dec)
        if cert is None:
            continue
        best_seen_L = min(best_seen_L, cert["L_float"])
        print(f"  {alll[i]}: L={cert['L_float']:.12f} dL={cert['dL']:+.6e}", flush=True)
        if cert["is_real_win"] and (best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"])):
            best_cert = cert
            best_label = alll[i]
            cp.save_win_and_rebuild_final(n_t, best_cert, best_label)
            print(f"[{n_t}<-{n_d}] *** REAL WIN *** {best_label} L={best_cert['L_lit'][:20]} dL={best_cert['dL']:+.6e}", flush=True)

    return {
        "n": n_t,
        "donor": n_d,
        "L_inc": L_inc,
        "n_jobs": len(alll),
        "n_feas": n_feas,
        "best_seen_L": best_seen_L if best_seen_L < 1e17 else None,
        "win": best_cert is not None,
        "best_label": best_label,
        "L_win": best_cert["L_lit"] if best_cert else None,
        "dL": best_cert["dL"] if best_cert else (best_seen_L - L_inc if best_seen_L < 1e17 else None),
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    args = ap.parse_args()

    pairs = DEFAULT_PAIRS[: args.limit]
    need_ns = {n for p in pairs for n in p}
    packs, best_L, best_src = cp.load_frontier(target_ns=need_ns)
    # Include our saved win for n=401 as donor if present!
    win401 = cp.DATA_DIR / "win_n401.json"
    if win401.exists():
        d = json.loads(win401.read_text())
        rows = d["rows"]
        packs[401] = cp.norm_pack(
            np.array([float(r["x"]) for r in rows]),
            np.array([float(r["y"]) for r in rows]),
            np.array([float(r["angle"]) for r in rows]),
        )
        best_L[401] = Decimal(d["L_lit"])

    rng = np.random.default_rng(20261117)
    results = []
    for n_t, n_d in pairs:
        res = run_target_soft(n_t, n_d, packs, best_L, rng)
        results.append(res)
        (cp.DOCS_DIR / "sweep_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
