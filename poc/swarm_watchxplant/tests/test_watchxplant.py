#!/usr/bin/env python3
"""Unit tests for H122 (watchxplant) brick extraction, pooling, and pinwheel assembly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1] / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import harvest_and_synth as hs  # noqa: E402
import watchxplant_lib as wx  # noqa: E402


def _make_dummy_brick(w: int, h: int, g: int, f: float = 0.69) -> dict:
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
    b = wx.make_brick(w, h, g, np.array(xs), np.array(ys), np.array(ths), tag=f"dummy_{w}x{h}_g{g}")
    b["f"] = f
    return b


def test_pinwheel_counts_and_kxky() -> None:
    m = 30
    b1 = _make_dummy_brick(9, 21, 6)
    b2 = _make_dummy_brick(21, 9, 7)
    res = wx.pl.build_pinwheel_kxky(m, 13, b1, b2, drop_mode="none")
    assert res is not None
    ax, ay, at, L, sm = res
    assert len(ax) == m * m + 13
    assert len(ay) == m * m + 13
    assert len(at) == m * m + 13
    assert len(sm) == m * m + 13
    assert L >= m


def test_canonical_shape_helper() -> None:
    b = _make_dummy_brick(12, 8, 5)
    assert hs._canonical_shape(b) == (8, 12, 5)


def test_fast_frontier_loads_known_n() -> None:
    fr = wx.fast_load_true_frontier([908, 973], certify=False)
    assert 908 in fr and 973 in fr
    assert len(fr[908]["x"]) == 908
    assert len(fr[973]["x"]) == 973
    assert fr[973]["L"] < 31.68
