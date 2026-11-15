"""Unit tests for poc/swarm_inflatedrift."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
POC_DIR = REPO_ROOT / "poc" / "swarm_inflatedrift" / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import arm_model as am  # noqa: E402
import inflatedrift_lib as il  # noqa: E402
import portfolio as pf  # noqa: E402
import pslp_4c as pslp  # noqa: E402


def test_cpp_overinflate_and_soft_melt():
    """Verify compiled C++ library handles both a_start > 1.0 over-inflation and a_start < 1.0 soft melt."""
    pf_data = pf.load_live_portfolio([25])
    inc = pf_data[25]
    x, y, th, L = inc["x"].copy(), inc["y"].copy(), inc["th"].copy(), float(inc["L"])
    pen0, L0 = il.sat_pen_bbox(x, y, th)
    assert pen0 <= 1e-12
    assert abs(L0 - L) < 1e-9

    # Over-inflation (a_start = 1.003 -> 1.0)
    res_over = il.cpp_batch_melt(
        [(x, y, th)], [[L + 0.01, L + 0.002]], a_start=1.003,
        n_melt_stages=3, seam_iters=0, full_iters=60, n_ruin_iters=0, n_threads=1, base_seed=7,
    )
    assert len(res_over) == 1
    assert res_over[0]["pen"] < 1e-6

    # Masked soft-a melt (a_start = 0.85 -> 1.0)
    mask = np.zeros(len(x), dtype=bool)
    mask[:8] = True
    res_mpk = il.cpp_batch_masked_melt(
        [(x, y, th)], [mask], [[L + 0.02, L + 0.005]], ils_iters_per_L=[1, 2],
        a_start=0.85, n_melt_stages=3, masked_iters=60, full_iters=60, n_threads=1, base_seed=9,
    )
    assert len(res_mpk) == 1
    assert res_mpk[0]["pen"] < 1e-6


def test_4corner_arm_model_and_coupled_polish():
    """Verify 4-corner static/rigid reduction + coupled pslp + lpslack + dual cert on n=55."""
    inc = pf.load_live_portfolio([55])[55]
    x, y, th, L = inc["x"], inc["y"], inc["th"], float(inc["L"])
    a0 = np.column_stack([x, y, th])
    a_n, L_n = am.normalize_arr(a0)
    m = am.build_model(a_n, L_n)
    assert "kept" in m["block_counts"]

    pol = il.coupled_pslp_lpslack_polish(
        55, x, y, th, old_L=inc["L_dec"], use_reduced=True, pslp_iters=80, run_lpslack=True, lpslack_time_limit=4.0
    )
    assert pol["viol"] <= 1e-10
    ver = pf.certify_state(55, pol["x"], pol["y"], pol["th"])
    assert ver["ok"]
    assert ver["L_lit"] == ver["L_pandas"]
