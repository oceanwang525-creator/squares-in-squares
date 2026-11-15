#!/usr/bin/env python3
"""Scan n=492, 852, 911, 974, 975 from current best portfolio rows at step=0.0625."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SCAN3_POC = REPO / "poc" / "pocket_lift_scan3" / "poc"
sys.path.insert(0, str(SCAN3_POC))
import scan_pockets as sp  # noqa: E402

TARGETS = [492, 852, 911, 974, 975]
OUT_JSON = REPO / "poc" / "swarm_interlock2" / "data" / "scanned_5rows.json"
DONORS_JSON = REPO / "poc" / "swarm_interlock2" / "data" / "donors_5rows.json"


def _norm_L(rows: list[tuple[str, str, str, str]]) -> float:
    a = np.array([[float(x), float(y), float(t)] for _, x, y, t in rows], dtype=np.float64)
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    return float(max((a[:, 0] + e).max() - (a[:, 0] - e).min(), (a[:, 1] + e).max() - (a[:, 1] - e).min()))


def load_best_donor_rows(targets: list[int]) -> dict[int, list[tuple[str, str, str, str]]]:
    tset = set(targets)
    best_rows: dict[int, list[tuple[str, str, str, str]]] = {}
    best_L: dict[int, float] = {}

    merged = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    with merged.open() as f:
        tmp: dict[int, list[tuple[str, str, str, str]]] = {}
        for r in csv.DictReader(f):
            n = int(r["n"])
            if n in tset:
                tmp.setdefault(n, []).append((r["square_id"], r["x"], r["y"], r["angle"]))
        for n, rs in tmp.items():
            rs.sort(key=lambda z: int(z[0]))
            best_rows[n] = rs
            best_L[n] = _norm_L(rs)

    pi180 = math.pi / 180.0
    for fp in sorted((REPO / "poc").glob("*/final.csv")):
        if fp.parent.name == "swarm_interlock2" or fp.stat().st_size <= 20:
            continue
        tmp = {}
        try:
            with fp.open() as f:
                for r in csv.DictReader(f):
                    n_s, sid = r["id"].split("_")
                    n = int(n_s)
                    if n not in tset:
                        continue
                    xs = r["x"].lstrip("s")
                    ys = r["y"].lstrip("s")
                    ds = r["deg"].lstrip("s")
                    # Convert deg string to rad string via float? No, keep high precision via Decimal/mpmath if needed
                    tmp.setdefault(n, []).append((sid, xs, ys, repr(float(ds) * pi180), ds))
        except Exception:
            continue
        for n, rs in tmp.items():
            if len(rs) != n:
                continue
            rs.sort(key=lambda z: int(z[0]))
            import mpmath as mp
            with mp.workdps(60):
                rad_rs = [(sid, xs, ys, mp.nstr(mp.mpf(ds) * mp.pi / 180, 50)) for sid, xs, ys, _, ds in rs]
            L = _norm_L(rad_rs)
            if L < best_L[n] - 1e-12:
                best_L[n] = L
                best_rows[n] = rad_rs
                print(f"Donor n={n} updated from {fp.relative_to(REPO)} (L={L:.12f})", flush=True)
    return best_rows


def _scan_one(item):
    n, rows = item
    t0 = time.time()
    arr = np.array([[float(x), float(y), float(t)] for _, x, y, t in rows], dtype=np.float64)
    res, _, st = sp.scan_row(
        n, arr, step=0.0625, fmax=1.15, nmax=31, cap=2000, off=(0.015625, 0.015625), verbose=True
    )
    # Sanitize immediately against exact rows
    row = sp.Row(n, arr)
    tol = 1e-9
    clean = []
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
        if gg < 1 or w < 1 or h < 1:
            continue
        f = max(W - w, H - h)
        if f >= 0.999:
            continue
        clean.append(dict(src=n, w=w, h=h, g=gg, f=f, cnt=k, x0=X0, x1=X1, y0=Y0, y1=Y1))
    print(f"Done n={n}: raw={len(res)} clean={len(clean)} in {time.time()-t0:.1f}s", flush=True)
    return n, clean


def main():
    t0 = time.time()
    best_rows = load_best_donor_rows(TARGETS)
    DONORS_JSON.write_text(json.dumps({str(k): v for k, v in best_rows.items()}))
    with mp.get_context("fork").Pool(processes=2) as pool:
        results = pool.map(_scan_one, [(n, best_rows[n]) for n in TARGETS])
    all_pockets = []
    for _, clean in results:
        all_pockets.extend(clean)
    OUT_JSON.write_text(json.dumps(all_pockets, indent=1))
    print(f"Saved {len(all_pockets)} clean pockets to {OUT_JSON} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
