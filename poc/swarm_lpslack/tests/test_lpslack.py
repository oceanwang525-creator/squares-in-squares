"""Unit tests for H75 lpslack geometry, soft-angle SLP, and export checks."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "swarm_lpslack" / "poc"))
import lpslack  # noqa: E402


def test_active_branches_and_slp_kink() -> None:
    # Near 0 deg, both (+1,+1) and (+1,-1) must be active so rotation increases extent
    branches = lpslack._active_branches(0.0)
    assert (1, 1) in branches and (1, -1) in branches
    # Away from multiples of pi/2, only 1 branch is active
    branches_30 = lpslack._active_branches(math.radians(30.0))
    assert branches_30 == [(1, 1)]


def test_fixed_lp_and_soft_slp_on_two_squares() -> None:
    # Two squares with slight excess gap and non-zero angle
    centers = np.array([[0.55, 0.55], [1.75, 0.55]], dtype=np.float64)
    angles = np.array([0.1, 0.1], dtype=np.float64)
    out = lpslack.fixed_lp_loop(angles, centers, rounds=4, margin=1e-9)
    assert out is not None
    side, new_c, pairs, marginals = out
    assert side < lpslack.exact_side(centers, angles)
    assert lpslack.min_sat_gap(new_c, angles) > 0.0

    free_mask = lpslack.select_free_mask(angles, "non90", pairs, marginals)
    assert free_mask.all()
    slp_out = lpslack.solve_soft_angle_slp(angles, new_c, free_mask, tr_th=1e-2, tr_xy=0.2, margin=1e-8)
    assert slp_out is not None
    _, slp_c, slp_a, _ = slp_out
    fx2 = lpslack.fixed_lp_loop(slp_a, slp_c, rounds=4, margin=1e-9)
    assert fx2 is not None
    fx2_side, fx2_c, _, _ = fx2
    assert fx2_side < side
    assert lpslack.min_sat_gap(fx2_c, slp_a) > 0.0
