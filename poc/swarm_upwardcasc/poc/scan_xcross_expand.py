#!/usr/bin/env python3
"""Detailed diagnostic scan of (m-1, r) -> (m, r) plateau-tied & superior-donor pairs + xcross_expand cut quality."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scan_pairs as sp

REPO = Path(__file__).resolve().parents[3]


def best_xcross_cut(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> tuple[int, float, float, float]:
    """Find best interior split (x0, y0) where shifting >=x0 by +1 and >=y0 by +1 cuts fewest tilted squares."""
    deg = np.degrees(th) % 90.0
    tilted = (deg > 1.0) & (deg < 89.0)
    h = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    # Normalize to [0, L]
    x = x - float((x - h).min())
    y = y - float((y - h).min())

    xs = np.linspace(1.5, L - 1.5, int((L - 3.0) * 4) + 1)
    ys = np.linspace(1.5, L - 1.5, int((L - 3.0) * 4) + 1)

    best = (9999, 9999.0, L * 0.5, L * 0.5)
    # Precompute per-line cut penalties
    x_stats = []
    for xc in xs:
        cut_all = (x - h < xc) & (x + h > xc)
        cut_tilt = int((cut_all & tilted).sum())
        ov_amt = float(np.minimum(xc - (x - h), (x + h) - xc)[cut_all].sum()) if cut_all.any() else 0.0
        x_stats.append((cut_tilt, ov_amt, float(xc)))
    y_stats = []
    for yc in ys:
        cut_all = (y - h < yc) & (y + h > yc)
        cut_tilt = int((cut_all & tilted).sum())
        ov_amt = float(np.minimum(yc - (y - h), (y + h) - yc)[cut_all].sum()) if cut_all.any() else 0.0
        y_stats.append((cut_tilt, ov_amt, float(yc)))

    x_stats.sort()
    y_stats.sort()
    xt, xo, xc = x_stats[0]
    yt, yo, yc = y_stats[0]
    # Also compute total cut (note: center square cut by both is still counted once)
    cut_x = (x - h < xc) & (x + h > xc)
    cut_y = (y - h < yc) & (y + h > yc)
    cut_tilt_tot = int(((cut_x | cut_y) & tilted).sum())
    return cut_tilt_tot, xo + yo, xc, yc


def main() -> None:
    best = sp.load_merged(REPO / "poc/pell_corrugation/data/border-scan/merged.csv")
    for p in sorted((REPO / "poc").glob("*/final.csv")):
        if p.stat().st_size <= 20:
            continue
        parsed = sp.parse_final_csv_text(p.read_text(encoding="utf-8", errors="ignore"), str(p.relative_to(REPO)))
        for n, entry in parsed.items():
            if n not in best or entry[0] < best[n][0] - 1e-13:
                best[n] = entry

    by_mr: dict[tuple[int, int], int] = {}
    for n in range(1, 1001):
        m = math.isqrt(n)
        by_mr[(m, n - m * m)] = n

    # Compute min f for m' > m at same r
    rows = []
    for (m, r), n_tgt in sorted(by_mr.items()):
        if m < 8 or (m - 1, r) not in by_mr:
            continue
        n_don = by_mr[(m - 1, r)]
        L_tgt, xt, yt, tt, src_tgt = best[n_tgt]
        L_don, xd, yd, td, src_don = best[n_don]
        f_tgt = L_tgt - m
        f_don = L_don - (m - 1)
        df_up = f_tgt - f_don
        if df_up < -1e-3:
            continue

        # Best higher-m f at same r
        higher_fs = [best[by_mr[(m2, r)]][0] - m2 for m2 in range(m + 1, 32) if (m2, r) in by_mr]
        f_min_hi = min(higher_fs) if higher_fs else f_tgt
        gap_hi = f_tgt - f_min_hi
        if gap_hi < 5e-3 and df_up < 1e-5:
            continue

        deg_d = np.degrees(td) % 90.0
        tilt_don = int(((deg_d > 1.0) & (deg_d < 89.0)).sum())
        xc_tilt, xc_ov, xc_x, xc_y = best_xcross_cut(xd, yd, td, L_don)
        rows.append(
            {
                "n_tgt": n_tgt,
                "n_don": n_don,
                "m": m,
                "r": r,
                "f_tgt": f_tgt,
                "f_don": f_don,
                "df_up": df_up,
                "gap_hi": gap_hi,
                "ds_gap": gap_hi / math.sqrt(n_tgt),
                "tilt_don": tilt_don,
                "xc_tilt": xc_tilt,
                "xc_ov": xc_ov,
                "xc_x": xc_x,
                "xc_y": xc_y,
            }
        )

    rows.sort(key=lambda r: (-r["df_up"] if r["df_up"] > 1e-6 else 0.0, -r["ds_gap"]))
    print(f"Found {len(rows)} upward pairs with df_up >= -1e-3 and (gap_hi >= 5e-3 or df_up >= 1e-5)")
    print(f"{'tgt<-don':>10} {'(m,r)':>9} {'f_tgt':>9} {'f_don':>9} {'df_up':>10} {'gap_hi':>9} {'ds_gap':>9} {'tdon':>5} {'xc_t':>5} {'xc_ov':>7}")
    for r in rows[:45]:
        print(
            f"{r['n_tgt']:>4}<-{r['n_don']:<4} ({r['m']:>2},{r['r']:>+3}) "
            f"{r['f_tgt']:>9.6f} {r['f_don']:>9.6f} {r['df_up']:>+10.6f} "
            f"{r['gap_hi']:>9.6f} {r['ds_gap']:>9.6f} {r['tilt_don']:>5} {r['xc_tilt']:>5} {r['xc_ov']:>7.3f}"
        )


if __name__ == "__main__":
    main()
