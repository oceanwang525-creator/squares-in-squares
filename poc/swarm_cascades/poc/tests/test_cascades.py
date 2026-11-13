"""Unit tests for swarm_cascades C++ SAT melt, Joint (x,y,dtheta) SLP, and certification."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import slp_utils as su  # noqa: E402


def test_cpp_energy_grad_matches_fd():
    rng = np.random.default_rng(7)
    n = 12
    L = 3.6
    q = np.zeros(3 * n, dtype=np.float64)
    q[:n] = rng.uniform(-1.5, 1.5, n)
    q[n : 2 * n] = rng.uniform(-1.5, 1.5, n)
    q[2 * n :] = rng.uniform(-0.6, 0.6, n)
    g = np.zeros_like(q)
    e0 = su.cpp().eval_sym_sat_energy_grad(
        n, 1,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        L, 2.0, 1.0, 1.0,
        g.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert e0 > 0.0
    eps = 1e-6
    for idx in range(0, 3 * n, 3):
        qp = q.copy(); qp[idx] += eps
        qm = q.copy(); qm[idx] -= eps
        gp = np.zeros_like(q)
        ep = su.cpp().eval_sym_sat_energy_grad(
            n, 1, qp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            L, 2.0, 1.0, 1.0, gp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        em = su.cpp().eval_sym_sat_energy_grad(
            n, 1, qm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            L, 2.0, 1.0, 1.0, gp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        fd = (ep - em) / (2 * eps)
        assert abs(fd - g[idx]) < 1e-4 * (1.0 + abs(fd))


def test_joint_slp_tightens_toy_5():
    # n=5 optimal is 2 + sqrt(2)/2 = 2.70710678...
    # Start slightly loose and slightly off-angle from 45 deg
    x = np.array([0.55, 2.25, 0.55, 2.25, 1.40], dtype=np.float64)
    y = np.array([0.55, 0.55, 2.25, 2.25, 1.40], dtype=np.float64)
    th = np.array([0.02, -0.02, 0.01, -0.01, np.radians(41.0)], dtype=np.float64)
    nx, ny, nth, L_best, ok = su.joint_xytheta_slp(x, y, th, rounds=14)
    assert ok
    assert L_best < 2.715
    res = su.verify_official(5, nx, ny, nth)
    assert res["ok"]


def test_seam_melt_repairs_overlap():
    # 9-square 3x3 grid with one square displaced into overlap, seam-masked melt repairs it
    x = np.array([0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5], dtype=np.float64)
    y = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5], dtype=np.float64)
    th = np.zeros(9, dtype=np.float64)
    x[4] += 0.22
    th[4] = 0.15
    seam = np.zeros(9, dtype=bool)
    seam[4] = True
    nx, ny, nth, pen, Lb = su.cpp_lbfgs_relax(
        x, y, th, 3.005, a_start=0.88, n_melt_stages=3,
        seam_mask_full=seam, seam_iters=140, full_iters=160,
    )
    assert pen < 1e-8
    assert Lb <= 3.005 + 1e-6
