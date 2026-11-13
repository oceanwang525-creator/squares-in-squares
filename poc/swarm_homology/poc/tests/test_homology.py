"""Unit tests for poc/swarm_homology H10 modular homology and final.csv validation."""
from __future__ import annotations

import csv
import io
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parents[1]
ROOT = POC_DIR.parents[2]
sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))

import h10_gate as hg  # noqa: E402
import h10_scale as hs  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402
import squares.core as sc  # noqa: E402


def _parse_ci_final(csv_text: str) -> dict[int, list[dict[str, str]]]:
    by_n: dict[int, list[dict[str, str]]] = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        n_str, sq_str = row["id"].split("_")
        by_n.setdefault(int(n_str), []).append(
            {
                "square_id": sq_str,
                "x": row["x"],
                "y": row["y"],
                "angle": repr(math.radians(float(row["deg"]))),
            }
        )
    return by_n


def test_hx_and_norm_invariants():
    th = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    hx = hg.hx_of(th)
    assert np.allclose(hx, [0.5, np.sqrt(0.5), 0.5])
    x = np.array([2.0, 4.0, 6.0])
    y = np.array([3.0, 5.0, 7.0])
    nx, ny, _, L = hg.norm(x, y, th)
    assert abs((nx - hx).min()) < 1e-12
    assert abs((ny - hx).min()) < 1e-12
    assert abs(L - max((nx + hx).max(), (ny + hx).max())) < 1e-12


def test_cpp_eval_matches_numba_sat():
    rng = np.random.default_rng(42)
    n = 16
    x = rng.uniform(0.5, 4.5, n)
    y = rng.uniform(0.5, 4.5, n)
    th = rng.uniform(0.0, np.pi / 2.0, n)
    pen_nb, L_nb = u.sat_max_overlap_and_bbox(x, y, th)
    pen_cpp, L_cpp = cpp_eval_sat_pen_bbox(x, y, th)
    assert abs(pen_nb - pen_cpp) < 1e-11
    assert abs(L_nb - L_cpp) < 1e-11


def test_grid_non_overlapping():
    gx, gy, gt = hg.grid(10.5, 5)
    assert len(gx) == 25
    pen, L = u.sat_max_overlap_and_bbox(gx, gy, gt)
    assert pen <= 1e-12
    assert abs(L - 5.0) < 1e-12


def test_wall_gap_slots_and_crop_seeds():
    # Construct a 5x5 upright packing on L=5.0 with an extra protruding square and one open top gap at x=2.5
    xs, ys = [], []
    for iy in range(5):
        for ix in range(5):
            if iy == 4 and ix == 2:
                continue
            xs.append(0.5 + ix)
            ys.append(0.5 + iy)
    xs.append(2.5)
    ys.append(4.8)  # protruding above Lt=5.0
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    t = np.zeros_like(x)
    idx_ex = len(x) - 1
    slots = hs.wall_gap_slots(x, y, t, Lt=5.0, idx_ex=idx_ex, n_slots=3)
    assert len(slots) >= 1
    best_clr, edge, px, py, cx, cy = slots[0]
    assert edge == "top" and abs(px - 2.5) < 0.2 and abs(py - 4.5) < 1e-6 and best_clr >= -0.05
    assert 1.5 <= cx <= 3.5 and abs(cy - 3.8) < 1e-6

    M = {
        len(x): (x, y, t, 5.2),
        len(x) + 10: (np.concatenate([x, np.full(10, 5.5)]), np.concatenate([y, np.linspace(0.5, 4.5, 10)]), np.zeros(len(x) + 10), 6.0),
    }
    p1 = hs.find_plus1_seeds(len(x) + 1, {len(x): (x[:-1], y[:-1], t[:-1], 5.0), len(x) + 1: (x, y, t, 5.5)})
    assert len(p1) >= 1
    q, m, ls, lab = hs.build_melt_jobs([("s0", x, y, t, 5.0, np.array([[2.5, 3.8]]))], L_inc=5.05, rng=np.random.default_rng(7), per_seed=2)
    assert len(q) == 2 and len(m) == 2 and ls[0][1] > 5.05 > ls[0][2]


def test_final_csv_if_present_valid_and_nonregressive():
    final_csv = POC_DIR.parent / "final.csv"
    if not final_csv.exists():
        return
    lit_txt = final_csv.read_text()
    lit = _parse_ci_final(lit_txt)
    if not lit:
        return
    pd_buf = io.StringIO()
    pd.read_csv(final_csv).to_csv(pd_buf, index=False)
    pds = _parse_ci_final(pd_buf.getvalue())
    for n, grp in lit.items():
        _, _, side_lit, reason_lit = sc.evaluate(n, grp)
        assert reason_lit == "ok" and side_lit is not None
        _, _, side_pd, reason_pd = sc.evaluate(n, pds[n])
        assert reason_pd == "ok" and side_pd is not None
        assert Decimal(side_lit) > 0 and Decimal(side_pd) > 0
