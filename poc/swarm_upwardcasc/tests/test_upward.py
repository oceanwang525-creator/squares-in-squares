#!/usr/bin/env python3
"""Unit tests for H98 upwardcasc L-border and xcross_expand seed generators."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

POC = Path(__file__).resolve().parents[1] / "poc"
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

import upward_engine as ue
import upward_lib as ul
from cpp_wrap import cpp_eval_sat_pen_bbox


def test_lborder_zero_overlap() -> None:
    # Synthetic (m-1, r) = (5, +2) -> n_donor = 27 -> n_target = (6, +2) = 38
    m_don = 5
    xs, ys = np.meshgrid(np.arange(0.5, m_don, 1.0), np.arange(0.5, m_don, 1.0))
    xd = np.concatenate([xs.ravel(), [5.5, 5.5]])
    yd = np.concatenate([ys.ravel(), [0.5, 1.5]])
    td = np.zeros(27, dtype=np.float64)
    xd, yd, td, Ld = ul.norm_pack(xd, yd, td)

    seeds = ue.generate_lborder_seeds(38, 27, xd, yd, td, Ld)
    assert len(seeds) == 8
    for name, x, y, th, nL, _ in seeds:
        assert len(x) == 38 and len(y) == 38 and len(th) == 38
        pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
        assert pen < 1e-12, f"{name} had pen={pen}"
        assert abs(bb - (Ld + 1.0)) < 1e-4, f"{name} had bb={bb} vs {Ld + 1.0}"


def test_xcross_expand_counts_and_clean_cut() -> None:
    # Synthetic 5x5 + 2 squares with an empty grid cut at x=2.0, y=2.0 -> 38 squares
    xs, ys = np.meshgrid(np.arange(0.5, 5.0, 1.0), np.arange(0.5, 5.0, 1.0))
    xd = np.concatenate([xs.ravel(), [5.5, 5.5]])
    yd = np.concatenate([ys.ravel(), [0.5, 1.5]])
    td = np.zeros(27, dtype=np.float64)
    xd, yd, td, Ld = ul.norm_pack(xd, yd, td)

    seeds = ue.generate_xcross_expand_seeds(38, 27, xd, yd, td, Ld)
    assert len(seeds) >= 2
    for name, x, y, th, nL, _ in seeds:
        assert len(x) == 38 and len(y) == 38 and len(th) == 38
        assert math.isfinite(nL)
