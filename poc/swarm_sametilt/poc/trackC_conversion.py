#!/usr/bin/env python3
"""Track C: Synthetic same-tilt conversion on two-arm opposite-tilt incumbents (~25+~65 -> ~65+~65 or ~25+~25)."""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sametilt_lib as st  # noqa: E402
from trackB_homology import run_seed_batch  # noqa: E402

DOCS = st.DOCS_DIR


def build_conversion_seeds(
    n_target: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    packs_910: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """Convert opposite-tilt (t25 + t65) incumbent into same-tilt candidates."""
    d = np.degrees(th) % 90.0
    m25 = (d >= 15.0) & (d <= 35.0)
    m65 = (d >= 55.0) & (d <= 75.0)
    seeds = []

    # 1. Reflect the ~25 deg arm -> ~65 deg (and also try reflecting ~65 deg -> ~25 deg)
    for tag, mask in (("flip25to65", m25), ("flip65to25", m65)):
        if mask.sum() < 20:
            continue
        xm = float(0.5 * (x[mask].min() + x[mask].max()))
        ym = float(0.5 * (y[mask].min() + y[mask].max()))

        # Vertical mirror across x=xm: x -> 2*xm - x, th -> -th
        for shift_x in (0.0, -0.35, 0.35):
            xr, yr, tr = x.copy(), y.copy(), th.copy()
            xr[mask] = 2.0 * (xm + shift_x) - x[mask]
            tr[mask] = -th[mask]
            ctrs = np.array([[xm, ym], [float(x[mask].min()), ym], [float(x[mask].max()), ym]])
            seeds.append((f"{tag}_vx{shift_x:+.2f}", xr, yr, tr, L - 1e-3, ctrs))

        # Horizontal mirror across y=ym: y -> 2*ym - y, th -> -th
        xr2, yr2, tr2 = x.copy(), y.copy(), th.copy()
        yr2[mask] = 2.0 * ym - y[mask]
        tr2[mask] = -th[mask]
        ctrs2 = np.array([[xm, ym], [xm, float(y[mask].min())], [xm, float(y[mask].max())]])
        seeds.append((f"{tag}_hy", xr2, yr2, tr2, L - 1e-3, ctrs2))

        # Diagonal transpose of the arm onto itself? Better: rotate arm by 180 deg (preserves angle mod 90? Wait: 180 deg preserves th mod 90, doesn't flip 25->65).
        # Wait: 90-deg swap of axis around arm center changes x,y shape while flipping 25->65 via reflection across y-ym = x-xm!
        xr3, yr3, tr3 = x.copy(), y.copy(), th.copy()
        xr3[mask] = xm + (y[mask] - ym)
        yr3[mask] = ym + (x[mask] - xm)
        tr3[mask] = 0.5 * np.pi - th[mask]
        # Only keep if inside [0.5, L-0.5] roughly
        if xr3[mask].min() > 0.3 and xr3[mask].max() < L - 0.3 and yr3[mask].min() > 0.3 and yr3[mask].max() < L - 0.3:
            seeds.append((f"{tag}_diag", xr3, yr3, tr3, L - 1e-3, np.array([[xm, ym]])))

    # 2. If m is close to 30 (e.g. 26..31) and n has ~25+~65, also try transplanting n=910's BOT_65+BR+BL scaled by TR peel
    # (Already well covered by Track B, so Track C focuses on pure arm conversion.)
    # Sort by lowest initial overlap penetration
    def score_seed(item):
        _, sx, sy, st_a, _, _ = item
        pen, bb = st.cpp_eval_sat_pen_bbox(sx, sy, st_a)
        return (pen, bb)

    seeds.sort(key=score_seed)
    return seeds[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    st.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    st.DATA_DIR.mkdir(parents=True, exist_ok=True)
    packs, best_L = st.load_frontier()
    rng = np.random.default_rng(20261115)

    if args.targets:
        targets = args.targets
    else:
        priority = [463, 639, 686, 691, 701, 739, 794, 851, 861, 908, 911, 912, 971, 973]
        # Also find all n in 350..1000 with both t25 >= 35 and t65 >= 35
        two_arm = []
        for n in range(350, 1001):
            if n not in packs:
                continue
            _, _, th, L = packs[n]
            d = np.degrees(th) % 90.0
            t25 = int(np.sum((d >= 15.0) & (d <= 35.0)))
            t65 = int(np.sum((d >= 55.0) & (d <= 75.0)))
            if t25 >= 35 and t65 >= 35:
                m = int(np.floor(L))
                frac = L - m
                two_arm.append((frac, n))
        # Prioritize higher fractional excess frac (more room to improve) after priority list
        two_arm.sort(reverse=True)
        targets = priority + [n for _, n in two_arm if n not in priority][:30]

    summary = {}
    for n_t in targets:
        if n_t not in packs:
            continue
        x, y, th, L = packs[n_t]
        L_inc_dec = best_L[n_t]
        print(f"\n=== Track C Target n={n_t} (frontier L={float(L_inc_dec):.12f}) ===", flush=True)
        seeds = build_conversion_seeds(n_t, x, y, th, L, packs[910])
        if not seeds:
            print(f"[{n_t}] No opposite-tilt arms to convert.", flush=True)
            continue

        cert = run_seed_batch(
            n_t,
            seeds,
            L_inc_dec,
            args.threads,
            rng,
            n_seeds_per_spec=6,
            masked_iters=args.iters,
            full_iters=args.iters + 30,
            top_polish=3,
            R_range=(7.0, 14.0),
            allow_full=True,
        )
        if cert is not None and cert["is_real_win"]:
            best_L[n_t] = Decimal(cert["L_lit"])
            summary[n_t] = {"L": cert["L_lit"], "dL": cert["dL"]}
        else:
            summary[n_t] = None
        (DOCS / "trackC_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
