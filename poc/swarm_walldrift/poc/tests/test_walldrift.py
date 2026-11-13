"""Unit tests for swarm_walldrift C++ SAT+drift melt, Joint SLP, and final.csv validation."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import slp_utils as su  # noqa: E402
import walldrift_run as wr  # noqa: E402


def test_cpp_drift_energy_grad_matches_fd():
    rng = np.random.default_rng(11)
    n = 10
    L = 3.5
    q = np.zeros(3 * n, dtype=np.float64)
    q[:n] = rng.uniform(-1.4, 1.4, n)
    q[n : 2 * n] = rng.uniform(-1.4, 1.4, n)
    q[2 * n :] = rng.uniform(-0.6, 0.6, n)
    drift = rng.uniform(-1.0, 1.0, 2 * n).astype(np.float64)
    drift_scale = 0.12
    g = np.zeros_like(q)

    e0 = su.cpp().eval_sym_sat_drift_energy_grad(
        n, 1,
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        L, 2.0, 1.0, 1.0,
        drift.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        drift_scale,
        g.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    assert np.isfinite(e0)
    eps = 1e-6
    gp = np.zeros_like(q)
    for idx in range(0, 3 * n, 3):
        qp = q.copy(); qp[idx] += eps
        qm = q.copy(); qm[idx] -= eps
        ep = su.cpp().eval_sym_sat_drift_energy_grad(
            n, 1, qp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            L, 2.0, 1.0, 1.0,
            drift.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            drift_scale,
            gp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        em = su.cpp().eval_sym_sat_drift_energy_grad(
            n, 1, qm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            L, 2.0, 1.0, 1.0,
            drift.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            drift_scale,
            gp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        fd = (ep - em) / (2.0 * eps)
        assert abs(fd - g[idx]) < 1e-4 * (1.0 + abs(fd))


def test_joint_slp_tightens_toy_5():
    x = np.array([0.55, 2.25, 0.55, 2.25, 1.40], dtype=np.float64)
    y = np.array([0.55, 0.55, 2.25, 2.25, 1.40], dtype=np.float64)
    th = np.array([0.02, -0.02, 0.01, -0.01, np.radians(41.0)], dtype=np.float64)
    nx, ny, nth, L_best, ok = su.joint_xytheta_slp(x, y, th, rounds=14)
    assert ok
    assert L_best < 2.715
    res = su.verify_official(5, nx, ny, nth)
    assert res["ok"]


def test_masked_pocket_drift_repairs_overlap():
    x = np.array([0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5], dtype=np.float64)
    y = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5], dtype=np.float64)
    th = np.zeros(9, dtype=np.float64)
    x[4] += 0.22
    th[4] = 0.15
    mask = np.zeros(9, dtype=bool)
    mask[4] = True
    drift = np.zeros(18, dtype=np.float64)
    drift[4] = -1.0
    res = su.cpp_batch_masked_melt(
        [(x, y, th)],
        [mask],
        [[3.02, 3.005]],
        ils_iters_per_L=[2, 4],
        a_start=0.86,
        n_melt_stages=3,
        masked_iters=120,
        full_iters=160,
        drift_xy_list=[drift],
        drift_strengths=[0.08],
        n_threads=1,
    )
    assert len(res) == 1
    assert res[0]["pen"] < 1e-8


def test_analyze_wall_imbalance_detects_weak_wall():
    x = np.array([0.5, 0.5, 0.5, 2.5, 1.5, 1.5], dtype=np.float64)
    y = np.array([0.5, 1.5, 2.5, 1.5, 0.5, 2.5], dtype=np.float64)
    th = np.zeros(6, dtype=np.float64)
    wb = wr.analyze_wall_imbalance(x, y, th, 3.0)
    assert wb["weak_wall"] == "R"
    assert wb["counts_1e8"]["R"] == 1
    assert wb["counts_1e8"]["L"] == 3


def test_final_csv_valid_with_auto_union():
    su.verify_final_csv_with_union(su.FINAL_CSV)
