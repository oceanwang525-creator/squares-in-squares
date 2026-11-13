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
