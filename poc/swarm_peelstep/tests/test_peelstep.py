#!/usr/bin/env python3
"""Unit tests for H109 (peelstep) pinwheel assembly, peeling, and pair selection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1] / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import peelstep_lib as pl  # noqa: E402


def _make_dummy_brick(w: int, h: int, g: int, f: float = 0.69) -> dict:
    # Synthetic non-overlapping-enough dummy geometry for shape/count unit tests
    xs, ys, ths = [], [], []
    for i in range(w):
        for j in range(h):
            xs.append(i + 0.5 + (f * i / max(1, w - 1)))
            ys.append(j + 0.5 + (f * j / max(1, h - 1)))
            ths.append(0.0)
    for k in range(g):
        xs.append(1.5 + 0.1 * k)
        ys.append(1.5 + 0.1 * k)
        ths.append(0.3)
    b = pl.make_brick(w, h, g, np.array(xs), np.array(ys), np.array(ths), tag=f"dummy_{w}x{h}_g{g}")
    b["f"] = f
    return b


def test_build_pinwheel_kxky_counts() -> None:
    m = 30
    for kx in (-1, 0, 1):
        for ky in (-1, 0, 1):
            w1, h1, g1 = 9, 21, 6
            w2 = m + kx - w1
            h2 = m + ky - h1
            g2 = 7
            b1 = _make_dummy_brick(w1, h1, g1)
            b2 = _make_dummy_brick(w2, h2, g2)
            if kx == 1 and ky == 1:
                r = g1 + g2 + 1
                res = pl.build_pinwheel_kxky(m, r, b1, b2, drop_mode="none")
                assert res is not None
                ax, ay, at, L, sm = res
                assert len(ax) == m * m + r
                assert len(sm) == m * m + r
                r_drop = g1 + g2
                res2 = pl.build_pinwheel_kxky(m, r_drop, b1, b2, drop_mode="b1_BR")
                assert res2 is not None
                assert len(res2[0]) == m * m + r_drop
            else:
                r = g1 + g2
                res = pl.build_pinwheel_kxky(m, r, b1, b2, drop_mode="none")
                assert res is not None
                ax, ay, at, L, sm = res
                assert len(ax) == m * m + r
                assert len(ay) == m * m + r
                assert len(at) == m * m + r
                assert L >= m


def test_pick_pairs_for_n_filter_and_kxky() -> None:
    # n=913 -> m=30, r=+13
    b1 = _make_dummy_brick(9, 21, 6)
    b2 = _make_dummy_brick(22, 9, 7)   # kx=+1, ky=0, g=13
    b3 = _make_dummy_brick(21, 10, 7)  # kx=0, ky=+1, g=13
    b4 = _make_dummy_brick(22, 10, 6)  # kx=+1, ky=+1, g=12 == r-1
    b_bad = _make_dummy_brick(20, 9, 7)  # f too high when overridden
    b_bad["f"] = 0.75
    pairs = pl.pick_pairs_for_n(913, [b1, b2, b3, b_bad], f_inc=0.707, max_excess=0.005, top_k=10)
    assert len(pairs) == 2
    assert all(p[3] == "none" for p in pairs)
    # Fallback to (1,1) when no non-(1,1) exists
    pairs_11 = pl.pick_pairs_for_n(913, [b1, b4], f_inc=0.707, max_excess=0.005, top_k=10)
    assert len(pairs_11) == 1
