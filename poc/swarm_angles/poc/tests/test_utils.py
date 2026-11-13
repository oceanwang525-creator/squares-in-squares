"""Unit tests for swarm_angles shared SAT/LP/MILP utilities."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import utils as u  # noqa: E402
from idea6_milp_tilted import fast_upright_count, milp_upright_fill  # noqa: E402


def test_sat_energy_and_grad_fd():
    rng = np.random.default_rng(7)
    n = 5
    x = rng.uniform(0.6, 2.4, n)
    y = rng.uniform(0.6, 2.4, n)
    th = rng.uniform(0.0, 0.8, n)
    state = np.concatenate([x, y, th])
    L = 2.6
    val0, grad0 = u.sat_penalty_grad(state, L, 1.0, 1.0)
    eps = 1e-6
    for i in range(len(state)):
        s_p = state.copy()
        s_m = state.copy()
        s_p[i] += eps
        s_m[i] -= eps
        v_p, _ = u.sat_penalty_grad(s_p, L, 1.0, 1.0)
        v_m, _ = u.sat_penalty_grad(s_m, L, 1.0, 1.0)
        fd = (v_p - v_m) / (2 * eps)
        assert abs(fd - grad0[i]) < 1e-5, f"grad mismatch idx={i}: fd={fd} ana={grad0[i]}"


def test_lp_tighten_and_official_cert_n5():
    # Known optimal n=5 has L = 2 + 1/sqrt(2) = 2.707106781...
    s2 = 1.0 / np.sqrt(2.0)
    L_opt = 2.0 + s2
    x = np.array([0.5, L_opt - 0.5, 0.5, L_opt - 0.5, 0.5 * L_opt])
    y = np.array([0.5, 0.5, L_opt - 0.5, L_opt - 0.5, 0.5 * L_opt])
    th = np.array([0.0, 0.0, 0.0, 0.0, 0.25 * np.pi])
    # Perturb slightly larger box, let LP recover tight L
    x_p = x.copy()
    x_p[1] += 0.02
    x_p[3] += 0.02
    y_p = y.copy()
    y_p[2] += 0.02
    y_p[3] += 0.02
    lx, ly, lp_val, ok = u.fixed_angle_lp_tighten(x_p, y_p, th, rounds=3)
    assert ok
    assert abs(lp_val - L_opt) < 1e-6
    cert = u.verify_official(5, lx, ly, th)
    assert cert["ok"]
    assert float(cert["L_lit"]) <= L_opt + 1e-6
    assert float(cert["L_pandas"]) <= L_opt + 1e-6
    assert cert["L_float"] <= L_opt + 1e-6


def test_milp_and_fast_upright_fill_n10_empty():
    L = 3.05
    tx = np.empty(0)
    ty = np.empty(0)
    tth = np.empty(0)
    n_up, ux, uy = milp_upright_fill(L, tx, ty, tth, min_gap=-1e-6)
    n_fast = fast_upright_count(L, tx, ty, tth, min_gap=-1e-6)
    assert n_up == 9
    assert n_fast == 9
    assert len(ux) == 9 and len(uy) == 9
