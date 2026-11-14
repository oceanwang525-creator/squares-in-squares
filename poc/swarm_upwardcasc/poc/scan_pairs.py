#!/usr/bin/env python3
"""2-Minute Scan for Hypothesis H98: Upward (m-k, r) -> (m, r) pairs across merged.csv + poc/*/final.csv + PR branches."""
from __future__ import annotations

import csv
import io
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def _strip_s(val: str) -> float:
    return float(val[1:] if val.startswith("s") else val)


def _norm_L(x: np.ndarray, y: np.ndarray, th_rad: np.ndarray) -> float:
    h = 0.5 * (np.abs(np.cos(th_rad)) + np.abs(np.sin(th_rad)))
    x0 = float((x - h).min())
    y0 = float((y - h).min())
    return float(max((x + h - x0).max(), (y + h - y0).max()))


def load_merged(path: Path) -> dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray, str]]:
    df = pd.read_csv(path, usecols=["n", "square_id", "x", "y", "angle"])
    out = {}
    for n_val, grp in df.groupby("n"):
        n = int(n_val)
        x = grp["x"].to_numpy(dtype=np.float64)
        y = grp["y"].to_numpy(dtype=np.float64)
        th = grp["angle"].to_numpy(dtype=np.float64)
        out[n] = (_norm_L(x, y, th), x, y, th, "merged.csv")
    return out


def parse_final_csv_text(text: str, label: str) -> dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray, str]]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "id,x,y,deg" or len(lines) <= 1:
        return {}
    by_n: dict[int, list[tuple[float, float, float]]] = {}
    reader = csv.reader(io.StringIO(text))
    next(reader)
    for row in reader:
        if len(row) != 4:
            continue
        n = int(row[0].split("_")[0])
        by_n.setdefault(n, []).append((_strip_s(row[1]), _strip_s(row[2]), math.radians(_strip_s(row[3]))))
    out = {}
    for n, pts in by_n.items():
        if len(pts) != n:
            continue
        arr = np.array(pts, dtype=np.float64)
        L = _norm_L(arr[:, 0], arr[:, 1], arr[:, 2])
        out[n] = (L, arr[:, 0], arr[:, 1], arr[:, 2], label)
    return out


def main() -> None:
    best = load_merged(REPO / "poc/pell_corrugation/data/border-scan/merged.csv")
    for p in sorted((REPO / "poc").glob("*/final.csv")):
        if p.stat().st_size <= 20:
            continue
        parsed = parse_final_csv_text(p.read_text(encoding="utf-8", errors="ignore"), str(p.relative_to(REPO)))
        for n, entry in parsed.items():
            if n not in best or entry[0] < best[n][0] - 1e-13:
                best[n] = entry

    # Also check pending PR branches mentioned in prompt/README
    for spec in [
        "origin/swarm-homocasc2-pr1:poc/swarm_homocasc2/final.csv",
        "origin/swarm-lpslack-pr3:poc/swarm_lpslack/final.csv",
    ]:
        try:
            txt = subprocess.check_output(["git", "show", spec], text=True, stderr=subprocess.DEVNULL)
            parsed = parse_final_csv_text(txt, spec.split(":")[0])
            for n, entry in parsed.items():
                if n not in best or entry[0] < best[n][0] - 1e-13:
                    best[n] = entry
        except subprocess.CalledProcessError:
            pass

    # Map (m, r) where m = isqrt(n), r = n - m^2 (note: GAP_AUDIT uses m = isqrt(n), r = n - m^2)
    by_mr: dict[tuple[int, int], int] = {}
    for n in range(1, 1001):
        m = math.isqrt(n)
        r = n - m * m
        by_mr[(m, r)] = n

    # Check both (m-1, r) -> (m, r) and (m-2, r) -> (m, r)
    cands = []
    for (m, r), n_tgt in sorted(by_mr.items()):
        L_tgt, xt, yt, tt, src_tgt = best[n_tgt]
        f_tgt = L_tgt - m
        deg_t = np.degrees(tt) % 90.0
        tilt_tgt = int(((deg_t > 1.0) & (deg_t < 89.0)).sum())
        for k in (1, 2):
            m_don = m - k
            if (m_don, r) not in by_mr:
                continue
            n_don = by_mr[(m_don, r)]
            L_don, xd, yd, td, src_don = best[n_don]
            f_don = L_don - m_don
            df_up = f_tgt - f_don  # positive means donor f is STRICTLY smaller -> raw L_don + k < L_tgt!
            if df_up >= -5e-3 and m >= 6:
                deg_d = np.degrees(td) % 90.0
                tilt_don = int(((deg_d > 1.0) & (deg_d < 89.0)).sum())
                cands.append(
                    {
                        "n_tgt": n_tgt,
                        "n_don": n_don,
                        "k": k,
                        "m": m,
                        "r": r,
                        "L_tgt": L_tgt,
                        "f_tgt": f_tgt,
                        "L_don": L_don,
                        "f_don": f_don,
                        "df_up": df_up,
                        "ds_raw": -df_up / math.sqrt(n_tgt),
                        "tilt_tgt": tilt_tgt,
                        "tilt_don": tilt_don,
                        "src_tgt": src_tgt,
                        "src_don": src_don,
                    }
                )

    cands.sort(key=lambda c: -c["df_up"])
    print(f"Total pairs with df_up >= -5e-3 (m>=6): {len(cands)}")
    print("\n--- TOP 40 by df_up (f_tgt - f_don) ---")
    print(f"{'tgt<-don':>10} {'k':>1} {'(m,r)':>9} {'f_tgt':>9} {'f_don':>9} {'df_up':>10} {'ds_raw':>10} {'tilt t/d':>9} {'src_don'}")
    for c in cands[:40]:
        print(
            f"{c['n_tgt']:>4}<-{c['n_don']:<4} {c['k']:>1} ({c['m']:>2},{c['r']:>+3}) "
            f"{c['f_tgt']:>9.6f} {c['f_don']:>9.6f} {c['df_up']:>+10.6f} {c['ds_raw']:>+10.7f} "
            f"{c['tilt_tgt']:>4}/{c['tilt_don']:<4} {c['src_don']}"
        )

    # Also print top m >= 16 pairs specifically
    cands16 = [c for c in cands if c["m"] >= 16]
    print(f"\n--- TOP 30 for m >= 16 ({len(cands16)} total) ---")
    for c in cands16[:30]:
        print(
            f"{c['n_tgt']:>4}<-{c['n_don']:<4} {c['k']:>1} ({c['m']:>2},{c['r']:>+3}) "
            f"{c['f_tgt']:>9.6f} {c['f_don']:>9.6f} {c['df_up']:>+10.6f} {c['ds_raw']:>+10.7f} "
            f"{c['tilt_tgt']:>4}/{c['tilt_don']:<4} {c['src_don']}"
        )


if __name__ == "__main__":
    main()
