#!/usr/bin/env python3
"""Scan the 88 newly landed origin/main rows (public_watch_nov14 + row_ratchet_980 + swarm_corrcasc)."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SCAN3_POC = REPO / "poc" / "pocket_lift_scan3" / "poc"
AUTO_UNION_POC = REPO / "poc" / "auto_union" / "poc"
sys.path.insert(0, str(SCAN3_POC))
sys.path.insert(0, str(AUTO_UNION_POC))

import scan_pockets as sp  # noqa: E402
import union as auto_union  # noqa: E402

OUT_JSON = REPO / "poc" / "swarm_interlock2" / "data" / "scanned_watch89.json"


def _norm_L(rows: list[tuple[str, str, str, str]]) -> float:
    a = np.array([[float(x), float(y), float(t)] for _, x, y, t in rows], dtype=np.float64)
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    return float(max((a[:, 0] + e).max() - (a[:, 0] - e).min(), (a[:, 1] + e).max() - (a[:, 1] - e).min()))


def load_new_donor_rows() -> dict[int, list[tuple[str, str, str, str]]]:
    srcs = [
        REPO / "poc" / "public_watch_nov14" / "final.csv",
        REPO / "poc" / "row_ratchet_980" / "final.csv",
        REPO / "poc" / "swarm_corrcasc" / "final.csv",
    ]
    best_rows: dict[int, list[tuple[str, str, str, str]]] = {}
    best_L: dict[int, float] = {}
    for fp in srcs:
        if not fp.exists():
            continue
        grps = auto_union.parse_text(fp.read_text())
        for n, rows in grps.items():
            rs = [(str(r["square_id"]), r["x"], r["y"], r["angle"]) for r in rows]
            rs.sort(key=lambda z: int(z[0]))
            L = _norm_L(rs)
            if n not in best_L or L < best_L[n] - 1e-12:
                best_L[n] = L
                best_rows[n] = rs
    return best_rows


def _scan_one(item):
    n, rows = item
    t0 = time.time()
    arr = np.array([[float(x), float(y), float(t)] for _, x, y, t in rows], dtype=np.float64)
    res, _, _ = sp.scan_row(
        n, arr, step=0.0625, fmax=1.15, nmax=31, cap=2000, off=(0.015625, 0.015625), verbose=False
    )
    row = sp.Row(n, arr)
    tol = 1e-9
    by_whg: dict[tuple[int, int, int], dict] = {}
    for g in res:
        m = ((row.lx >= g["x0"] - tol) & (row.hx <= g["x1"] + tol) &
             (row.ly >= g["y0"] - tol) & (row.hy <= g["y1"] + tol))
        k = int(m.sum())
        if k < 2:
            continue
        X0, X1 = float(row.lx[m].min()), float(row.hx[m].max())
        Y0, Y1 = float(row.ly[m].min()), float(row.hy[m].max())
        W, H = X1 - X0, Y1 - Y0
        w, h = int(W + 1e-6), int(H + 1e-6)
        gg = k - w * h
        if gg < 1 or w < 1 or h < 1 or max(w, h) >= 31:
            continue
        f = max(W - w, H - h)
        if f >= 0.999:
            continue
        key = (w, h, gg)
        if key not in by_whg or f < by_whg[key]["f"]:
            by_whg[key] = dict(src=n, w=w, h=h, g=gg, f=f, cnt=k, x0=X0, x1=X1, y0=Y0, y1=Y1)
    print(f"Scanned n={n:4d}: clean_whg={len(by_whg):4d} in {time.time()-t0:.1f}s", flush=True)
    return list(by_whg.values())


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    rows_by_n = load_new_donor_rows()
    items = sorted(rows_by_n.items(), reverse=True)
    print(f"Scanning {len(items)} new origin/main rows with max_workers={args.workers}...", flush=True)

    with mp.get_context("fork").Pool(processes=min(2, args.workers)) as pool:
        per_row = pool.map(_scan_one, items)

    # Keep top-2 min-f per (w, h, g) across all 88 rows so output stays compact (<500 KB)
    global_by_whg: dict[tuple[int, int, int], list[dict]] = {}
    for lst in per_row:
        for pk in lst:
            k = (pk["w"], pk["h"], pk["g"])
            global_by_whg.setdefault(k, []).append(pk)

    kept = []
    for k in sorted(global_by_whg.keys()):
        lst = sorted(global_by_whg[k], key=lambda r: r["f"])
        kept.extend(lst[:2])
    OUT_JSON.write_text(json.dumps(kept, indent=1))
    print(f"Saved {len(kept)} compact clean pockets to {OUT_JSON} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
