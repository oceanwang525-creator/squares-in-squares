"""Unit tests for poc/swarm_sametilt H14 same-tilt dual-crystal construction and final.csv certification."""
from __future__ import annotations

import io
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

SAMETILT_DIR = Path(__file__).resolve().parents[1]
POC_DIR = SAMETILT_DIR / "poc"
ROOT = SAMETILT_DIR.parents[2]
sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(ROOT / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))

import sametilt_lib as st  # noqa: E402
import trackB_homology as tb  # noqa: E402
import union as auto_union  # noqa: E402


def test_hx_and_norm_pack():
    th = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    hx = st.hx_of(th)
    assert np.allclose(hx, [0.5, np.sqrt(0.5), 0.5])
    x = np.array([2.0, 4.0, 6.0])
    y = np.array([3.0, 5.0, 7.0])
    nx, ny, _, L = st.norm_pack(x, y, th)
    assert abs((nx - hx).min()) < 1e-12
    assert abs((ny - hx).min()) < 1e-12
    assert abs(L - max((nx + hx).max(), (ny + hx).max())) < 1e-12


def test_border_extend_seed_exact():
    # Extend a 2x2 grid (L=2.0) by 1 layer -> 3x3 (n=9, L=3.0, zero overlap)
    gx, gy = np.meshgrid([0.5, 1.5], [0.5, 1.5])
    x, y, th, L = tb.make_border_extend_seed(gx.ravel(), gy.ravel(), np.zeros(4), 2.0, layers=1)
    assert len(x) == 9
    pen, bb = st.cpp_eval_sat_pen_bbox(x, y, th)
    assert pen <= 1e-12
    assert abs(bb - 3.0) < 1e-4


def test_final_csv_structure_and_dual_cert():
    final_csv = SAMETILT_DIR / "final.csv"
    assert final_csv.exists()
    lit, pd_t = auto_union.load(final_csv)
    assert set(lit) == set(pd_t)
    for n, rows in lit.items():
        res = auto_union.check((n, rows, pd_t[n]))
        assert res is not None, f"Failed dual certification for n={n}"
        assert Decimal(res[0]) > 0 and Decimal(res[1]) > 0
