"""Unit tests for poc/swarm_corrcasc H107 diagonal peel cascade."""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

CC_DIR = Path(__file__).resolve().parents[1]
POC_DIR = CC_DIR / "poc"
ROOT = CC_DIR.parents[1]
for _p in (str(POC_DIR), str(ROOT / "poc" / "auto_union" / "poc"), str(ROOT / "poc" / "baseline" / "poc" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audit_diagonal_pairs as adp  # noqa: E402
import corrcasc_lib as cc  # noqa: E402
import union as auto_union  # noqa: E402


def _grid(k: int):
    gx, gy = np.meshgrid(np.arange(k) + 0.5, np.arange(k) + 0.5)
    return gx.ravel().copy(), gy.ravel().copy(), np.zeros(k * k)


def test_diagonal_l_border_cut_on_grid_is_clean():
    # (m+1)^2 = 25 -> m^2 = 16: removing dn = 2m+1 = 9 squares from any corner L-border is exact.
    x, y, th = _grid(5)
    cuts = adp.audit_l_borders(x, y, th, 5.0, 1.0, 9)
    assert cuts and cuts[0]["score"] == 0.0 and cuts[0]["n_prot"] == 0


def test_two_step_cut_on_grid_is_clean():
    x, y, th = _grid(6)
    cuts = adp.audit_l_borders(x, y, th, 6.0, 2.0, 36 - 16)
    assert cuts[0]["score"] == 0.0


def test_hx_and_norm_pack():
    th = np.array([0.0, np.pi / 4.0])
    assert np.allclose(cc.hx_of(th), [0.5, np.sqrt(0.5)])
    nx, ny, _, L = cc.norm_pack(np.array([3.0, 5.0]), np.array([2.0, 2.0]), th)
    assert abs(L - (5.0 - 3.0 + 0.5 + np.sqrt(0.5))) < 1e-12


def test_final_csv_structure_and_dual_cert():
    final_csv = CC_DIR / "final.csv"
    assert final_csv.exists()
    assert final_csv.read_text().splitlines()[0] == "id,x,y,deg"
    lit, pd_t = auto_union.load(final_csv)
    assert set(lit) == set(pd_t)
    for n, rows in lit.items():
        res = auto_union.check((n, rows, pd_t[n]))
        assert res is not None
        assert Decimal(res[0]) > 0 and Decimal(res[1]) > 0
