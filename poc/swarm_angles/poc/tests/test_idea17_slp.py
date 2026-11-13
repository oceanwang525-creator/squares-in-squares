"""Unit tests for Idea 17 joint (x, y, dtheta) Sequential Linear Programming (SLP)."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import idea17_run_slp as m17
import utils as u


def test_build_joint_slp_sparse_and_step() -> None:
    """Verify sparse LP matrix construction and joint (x, y, dtheta) tightening on a 5-square test configuration."""
    x0 = np.array([0.52, 1.54, 0.52, 1.54, 2.65], dtype=np.float64)
    y0 = np.array([0.52, 0.52, 1.54, 1.54, 1.05], dtype=np.float64)
    th0 = np.array([0.02, -0.02, 0.01, -0.01, math.radians(30.0)], dtype=np.float64)

    A, b = m17._build_joint_slp_sparse(x0, y0, th0, tr_xy=0.1, tr_th=0.02, margin=1e-8)
    assert A.shape[1] == 3 * len(x0) + 1
    assert A.shape[0] == len(b)
    assert A.shape[0] > 0

    bx, by, bth, bL, ok = m17.joint_xytheta_slp(x0, y0, th0, rounds=6, tr_th0=0.02, tr_xy0=0.1)
    pen, bbox = u.sat_max_overlap_and_bbox(bx, by, bth)
    assert ok
    assert pen <= 1e-7
    assert bL <= 3.35
    assert abs(bL - bbox) < 1e-5
