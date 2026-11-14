"""Unit tests for H101 2-D overhang pinwheel composer."""
from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import numpy as np

POC_DIR = Path(__file__).resolve().parents[1] / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import interlock2_lib as il  # noqa: E402
from squares import core  # noqa: E402


def _make_feasible_brick(w: int, h: int, f: float = 0.65) -> dict:
    """Create a valid 0-overlap brick of w*h squares with uniform spacing inside [0, w+f]x[0, h+f]."""
    xs, ys, ths = [], [], []
    for i in range(w):
        for j in range(h):
            xs.append(i + 0.5 + f * i / max(1, w - 1))
            ys.append(j + 0.5 + f * j / max(1, h - 1))
            ths.append(0.0)
    return {
        "w": w,
        "h": h,
        "g": 1,  # nominal g for candidate search tests
        "fx": f,
        "fy": f,
        "f": f,
        "x": np.array(xs, dtype=np.float64),
        "y": np.array(ys, dtype=np.float64),
        "th": np.array(ths, dtype=np.float64),
    }


def test_find_overhang_candidates_filters_correctly():
    by_whg = {
        (10, 10, 4): [{"uid": "a", "w": 10, "h": 10, "g": 4, "fx": 0.65, "fy": 0.65, "f": 0.65}],
        (11, 9, 5): [{"uid": "b", "w": 11, "h": 9, "g": 5, "fx": 0.66, "fy": 0.66, "f": 0.66}],
    }
    # At K=20: 10+11 = 21 => kx=1; 10+9 = 19 <= K-1 => ky=0, D = 4+5-1 = 8 => n = 408
    inc_L = {408: 20.67, 409: 20.68}
    zg = il.find_overhang_candidates(by_whg, inc_L, regime="zero_gap", dL_margin=0.004)
    assert len(zg) == 1
    assert zg[0]["n"] == 408 and zg[0]["kx"] == 1 and zg[0]["ky"] == 0 and zg[0]["D"] == 8

    # Check seam_touch excludes h1+h2 == K-1
    st = il.find_overhang_candidates(by_whg, inc_L, regime="seam_touch", dL_margin=0.004)
    assert len(st) == 0


def test_assemble_zero_gap_50d_certifies_cleanly():
    # Build two zero-overlap pockets with t = K - (h1+h2) = 2 (tests central strip too!)
    K = 16
    b1 = _make_feasible_brick(9, 7, 0.65)
    b2 = _make_feasible_brick(9, 7, 0.65)
    # w1+w2 = 18 = K+2 (kx=2), h1+h2 = 14 = K-2 (t=2)
    sq1, W1, H1 = il.brick_to_50d(b1)
    sq2, W2, H2 = il.brick_to_50d(b2)
    # Note: _make_feasible_brick has w*h squares each, so total squares = 9*7 + 9*7 + 7*9 + 7*9 + 2*1 = 254 = K^2 - 2
    n_test = K * K - 2
    rows = il.assemble_zero_gap_50d(K, n_test, sq1, W1, H1, 9, 7, sq2, W2, H2, 9, 7)
    assert rows is not None
    assert len(rows) == n_test
    L_lit = core.certify(n_test, rows)
    L_pan = core.certify(n_test, core.kaggle_roundtrip({n_test: rows})[n_test])
    assert float(L_lit) < K + 0.66
    assert float(L_pan) < K + 0.66


def test_assemble_seam_touch_float_counts():
    K = 16
    b1 = _make_feasible_brick(9, 8, 0.65)
    b2 = _make_feasible_brick(8, 8, 0.65)
    sq1, W1, H1 = il.brick_to_50d(b1)
    sq2, W2, H2 = il.brick_to_50d(b2)
    # w1+w2 = 17 = K+1, h1+h2 = 16 = K
    n_test = K * K
    asm = il.assemble_seam_touch_float(K, n_test, sq1, W1, H1, 9, 8, sq2, W2, H2, 8, 8)
    assert asm is not None
    ax, ay, at, aL, seam_box = asm
    assert len(ax) == n_test
    assert abs(aL - (K + 0.65)) < 1e-12
    assert seam_box[0] < seam_box[1] and seam_box[2] < seam_box[3]
