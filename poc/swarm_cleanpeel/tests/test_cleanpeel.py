"""Unit tests for poc/swarm_cleanpeel H16 clean-peel pipeline and final.csv certification."""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

CLEANPEEL_DIR = Path(__file__).resolve().parents[1]
POC_DIR = CLEANPEEL_DIR / "poc"
ROOT = CLEANPEEL_DIR.parents[2]
for _p in (str(POC_DIR), str(ROOT / "poc" / "auto_union" / "poc"), str(ROOT / "poc" / "baseline" / "poc" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cleanpeel_lib as cp  # noqa: E402
import extend_upward as eu  # noqa: E402
import peel_engine as pe  # noqa: E402
import union as auto_union  # noqa: E402


def test_extend_one_layer_exact():
    gx, gy = np.meshgrid([0.5, 1.5], [0.5, 1.5])
    x, y, th, L = eu.extend_one_layer(gx.ravel(), gy.ravel(), np.zeros(4), 2.0)
    assert len(x) == 9
    pen, bb = cp.cpp_eval_sat_pen_bbox(x, y, th)
    assert pen <= 1e-12
    assert abs(bb - 3.0) < 1e-4


def test_hx_and_norm_pack():
    th = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    hx = cp.hx_of(th)
    assert np.allclose(hx, [0.5, np.sqrt(0.5), 0.5])
    x = np.array([2.0, 4.0, 6.0])
    y = np.array([3.0, 5.0, 7.0])
    nx, ny, _, L = cp.norm_pack(x, y, th)
    assert abs((nx - hx).min()) < 1e-12
    assert abs((ny - hx).min()) < 1e-12
    assert abs(L - max((nx + hx).max(), (ny + hx).max())) < 1e-12


def test_cleanpeel_seed_generation_synthetic():
    # 3x3 grid (L=3.0) with one slightly tilted interior square protruding past y=1.0
    gx, gy = np.meshgrid([0.5, 1.5, 2.5], [0.5, 1.5, 2.5])
    x = gx.ravel().copy()
    y = gy.ravel().copy()
    th = np.zeros(9)
    # Tilt square at (1.5, 1.5) slightly so after TL peel (y >= 1.02) it protrudes past y=0
    idx = int(np.argmin((x - 1.5) ** 2 + (y - 1.5) ** 2))
    y[idx] = 1.20
    th[idx] = np.radians(25.0)
    rng = np.random.default_rng(42)
    seeds = pe.generate_cleanpeel_seeds(4, 9, x, y, th, 3.0, 2.1, rng, max_prot=2)
    assert len(seeds) >= 2
    for sname, xs, ys, ts, Lt, ctrs in seeds:
        assert len(xs) == 4 and len(ys) == 4 and len(ts) == 4
        assert ctrs.shape[1] == 2


def test_final_csv_structure_and_dual_cert():
    final_csv = CLEANPEEL_DIR / "final.csv"
    assert final_csv.exists()
    lit, pd_t = auto_union.load(final_csv)
    assert set(lit) == set(pd_t)
    for n, rows in lit.items():
        res = auto_union.check((n, rows, pd_t[n]))
        assert res is not None, f"Failed dual certification for n={n}"
        assert Decimal(res[0]) > 0 and Decimal(res[1]) > 0
