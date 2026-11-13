#!/usr/bin/env python3
"""Idea 15 Immediate Harvest (#90): Multi-margin joint_xytheta_slp on merged.csv[n]
+ 4 micro-jitter starts (1e-4 pos / 0.02 deg) + top /tmp/v15_top GPU basins.

Sweeps all 10 disjoint targets {910, 971, 972, 791, 969, 687, 851, 796, 367, 333}
capped at 4 workers (1 BLAS/OMP/HiGHS thread each), strictly verifying literal +
pandas certification and writing win_n<n>.csv + final.csv on any win.
"""
from __future__ import annotations

import csv
import json
import math
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from poc.swarm_inflation.poc import polish_and_certify as pc

REPO_ROOT = Path(__file__).resolve().parents[3]
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
DATA_DIR = REPO_ROOT / "poc" / "swarm_inflation" / "data"
FINAL_CSV = REPO_ROOT / "poc" / "swarm_inflation" / "final.csv"
DOC_OUT = REPO_ROOT / "poc" / "swarm_inflation" / "docs" / "idea15_harvest.json"
POOL_NS = [910, 971, 972, 791, 969, 687, 851, 796, 367, 333]
LP_MARGINS = (1e-7, 2e-8, 5e-9, 1e-9)


def _certify_deg_roundtrip(n: int, centers: np.ndarray, theta: np.ndarray) -> dict | None:
    """Certify using exact degrees roundtrip (matching final.csv id,x,y,deg)."""
    deg = np.degrees(theta)
    th_deg_rt = np.radians(np.array([float(repr(float(d))) for d in deg]))
    c_pos = pc.shift_positive(centers, th_deg_rt)
    cert = pc.dual_certify(n, c_pos, th_deg_rt)
    if not cert.get("ok"):
        return None
    cert["deg_rows"] = [
        {
            "id": f"{n:03d}_{i}" if n < 1000 else f"1000_{i}",
            "x": r["x"],
            "y": r["y"],
            "deg": repr(float(deg[i])),
        }
        for i, r in enumerate(cert["rows"])
    ]
    return cert


def _harvest_one_n(n: int) -> dict:
    """Run direct SLP + 4 micro-jitter starts (+ optional GPU v15_top) for one n."""
    t0 = time.time()
    inc_L = pc.load_incumbent_L(n, MERGED_CSV)
    df = pd.read_csv(MERGED_CSV)
    sub = df[df["n"] == n].sort_values("square_id")
    x0 = sub["x"].to_numpy(np.float64)
    y0 = sub["y"].to_numpy(np.float64)
    t_orig = sub["angle"].to_numpy(np.float64)
    c0 = pc.shift_positive(np.column_stack([x0, y0]), t_orig)

    starts: list[tuple[str, np.ndarray, np.ndarray]] = [("inc_orig", c0, t_orig.copy())]
    rng = np.random.default_rng(15100 + n)
    for j in range(4):
        dc = rng.normal(0.0, 1e-4, size=c0.shape)
        dt = np.radians(rng.normal(0.0, 0.02, size=len(t_orig)))
        starts.append((f"jit_{j}", pc.shift_positive(c0 + dc, t_orig + dt), t_orig + dt))

    # Also include top exported GPU v15 melt candidates if already generated
    for fm in (6, 5):
        bp = Path(f"/tmp/v15_top/top_n{n}_fm{fm}.bin")
        if bp.exists():
            with open(bp, "rb") as f:
                M, nn = struct.unpack("ii", f.read(8))
                for k in range(min(M, 8)):
                    gx = np.frombuffer(f.read(8 * nn), dtype=np.float64)
                    gy = np.frombuffer(f.read(8 * nn), dtype=np.float64)
                    gth = np.frombuffer(f.read(8 * nn), dtype=np.float64)
                    starts.append((f"v15_fm{fm}#{k}", np.column_stack([gx, gy]), gth))

    best_L: Decimal | None = None
    best_cert: dict | None = None
    best_label = ""
    best_pos = None
    best_th = None

    for label, pos_in, th_in in starts:
        c_slp, t_slp, L_slp = pc.fast_joint_xytheta_slp(
            pos_in, th_in, rounds=18, tr_th0=0.025, tr_xy0=0.12, lp_margins=LP_MARGINS
        )
        # Try each margin separately if near/under inc_L
        cands = [(c_slp, t_slp)]
        if Decimal(str(L_slp)) < inc_L + Decimal("0.002"):
            for m_lp in LP_MARGINS:
                cm, _ = pc.fixed_angle_lp_polish(c_slp, t_slp, rounds=8, margin=m_lp)
                cands.append((cm, t_slp))

        for cm, tm in cands:
            cert = _certify_deg_roundtrip(n, cm, tm)
            if cert is None:
                continue
            Lc = Decimal(cert["L_max"])
            if best_L is None or Lc < best_L:
                best_L = Lc
                best_cert = cert
                best_label = label
                best_pos = cm
                best_th = tm

    win = best_L is not None and best_L < inc_L
    out = {
        "n": n,
        "inc_L": str(inc_L),
        "best_L": str(best_L) if best_L is not None else None,
        "L_lit": best_cert["L_lit"] if best_cert else None,
        "L_pd": best_cert["L_pd"] if best_cert else None,
        "dL": float(inc_L - best_L) if best_L is not None else None,
        "win": bool(win),
        "best_label": best_label,
        "dt": round(time.time() - t0, 2),
    }
    if win and best_cert is not None and best_pos is not None and best_th is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pc.core.write_submission(DATA_DIR / f"win_n{n}.csv", {n: best_cert["rows"]}, validate=True)
        out["deg_rows"] = best_cert["deg_rows"]
    return out


def write_final_csv_from_wins(win_results: list[dict]) -> None:
    """Write poc/swarm_inflation/final.csv from all winning n results."""
    existing: dict[int, list[dict[str, str]]] = {}
    if FINAL_CSV.exists():
        with FINAL_CSV.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                nid = int(row["id"].split("_")[0])
                existing.setdefault(nid, []).append(row)
    for r in win_results:
        if r.get("win") and "deg_rows" in r:
            existing[r["n"]] = r["deg_rows"]
    with FINAL_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "x", "y", "deg"])
        w.writeheader()
        for kn in sorted(existing):
            for row in existing[kn]:
                w.writerow(row)


def main() -> None:
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_harvest_one_n, n): n for n in POOL_NS}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            print(
                f"[harvest] n={r['n']} win={r['win']} dL={r['dL']:+.3e} "
                f"src={r['best_label']} ({r['dt']}s)",
                flush=True,
            )
    results.sort(key=lambda r: r["n"])
    wins = [r for r in results if r["win"]]
    if wins:
        write_final_csv_from_wins(wins)
    clean_docs = [{k: v for k, v in r.items() if k != "deg_rows"} for r in results]
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUT.write_text(json.dumps(clean_docs, indent=2))
    print(f"Done in {time.time() - t0:.1f}s | wins={len(wins)}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
