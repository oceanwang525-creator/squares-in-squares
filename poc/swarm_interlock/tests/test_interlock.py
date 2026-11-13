"""Unit tests for H35 interlocking pinwheel assembly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1] / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import interlock_lib as il  # noqa: E402


def _make_dummy_brick(w: int, h: int, g: int, f: float = 0.65) -> dict:
    """Create a dummy brick with w*h unit squares + g slightly tilted interior squares."""
    xs, ys, ths = [], [], []
    for i in range(w):
        for j in range(h):
            xs.append(i + 0.5 + (0.0 if i < w // 2 else f))
            ys.append(j + 0.5 + (0.0 if j < h // 2 else f))
            ths.append(0.0)
    for k in range(g):
        xs.append(2.0 + 1.5 * k)
        ys.append(0.5 * h + 1.5 * (k % 2))
        ths.append(0.4)
    return {
        "w": w,
        "h": h,
        "g": g,
        "fx": f,
        "fy": f,
        "f": f,
        "x": np.array(xs, dtype=np.float64),
        "y": np.array(ys, dtype=np.float64),
        "th": np.array(ths, dtype=np.float64),
        "src_n": 0,
        "tag": "dummy",
    }


def test_assemble_interlock_count_and_disjoint_tilted():
    m = 29
    b1 = _make_dummy_brick(12, 12, 4, 0.65)
    b2 = _make_dummy_brick(20, 20, 6, 0.65)
    for drop_mode in ("b1", "b2", "regrid"):
        res = il.assemble_interlock(m, b1, b2, s1=0, s2=0, drop_from=drop_mode)
        assert res is not None, f"assemble_interlock returned None for {drop_mode}"
        x, y, th, L, I_box = res
        assert len(x) == m * m + 4 + 6
        assert abs(L - (m + 0.65)) < 1e-12
        assert I_box[0] == 9.0 and I_box[2] == 17.0
