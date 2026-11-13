"""Tests for Idea 18 C++ spatial-hash SAT evaluator, L-BFGS inflation, and continuum seed generators."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import idea18_run as i18  # noqa: E402
import utils as u  # noqa: E402


def test_cpp_sat_and_grad_parity_against_numba() -> None:
    rng = np.random.default_rng(42)
    n = 120
    L = 10.4
    x = rng.uniform(0.1, L - 0.1, size=n)
    y = rng.uniform(0.1, L - 0.1, size=n)
    th = rng.uniform(-math.pi / 2, math.pi / 2, size=n)

    for side in (0.85, 1.0):
        pen_py, L_py = u.sat_max_overlap_and_bbox(x, y, th, side=side)
        pen_cpp, L_cpp = i18.cpp_eval_sat_pen_bbox(x, y, th, side=side)
        assert abs(pen_py - pen_cpp) < 1e-12
        assert abs(L_py - L_cpp) < 1e-12

        state = np.concatenate([x, y, th])
        e_py, g_py = u.sat_penalty_grad(state, L, w_wall=2.5, w_pair=1.3, side=side)
        e_cpp, g_cpp = i18.cpp_eval_energy_grad(x, y, th, L, w_wall=2.5, w_pair=1.3, side=side)
        assert abs(e_py - e_cpp) < 1e-10
        assert np.max(np.abs(g_py - g_cpp)) < 1e-9


def test_all_seed_families_exact_counts_and_bounds() -> None:
    n = 401
    L_tgt = 20.53
    rng = np.random.default_rng(18)

    s1 = i18.generate_pin4_fixed_seeds(n, L_tgt, rng)
    s2 = i18.generate_pinched_frame4_seeds(n, L_tgt)
    s3 = i18.generate_chamfer_core_seeds(n, L_tgt)
    s4 = i18.generate_taper_band_seeds(n, L_tgt)

    assert len(s1) > 0
    assert len(s2) > 0
    assert len(s3) > 0
    assert len(s4) > 0

    for fam in (s1, s2, s3, s4):
        lbl, x, y, th = fam[0]
        assert len(x) == n and len(y) == n and len(th) == n, lbl
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(th))


def test_cpp_inflation_solves_small_non45_target() -> None:
    # Calibration check: 17 squares in 4.73 (beats upright grid L=5.0, near incumbent 4.675530) via soft-core inflation
    n = 17
    L_tgt = 4.73
    rng = np.random.default_rng(77)
    states = []
    for _ in range(256):
        x = rng.uniform(0.5, L_tgt - 0.5, size=n)
        y = rng.uniform(0.5, L_tgt - 0.5, size=n)
        th = rng.uniform(-math.pi / 4, math.pi / 4, size=n)
        states.append((x, y, th))

    a_sched = [0.75, 0.82, 0.88, 0.92, 0.95, 0.97, 0.985, 0.993, 0.997, 1.00]
    iters = [140] * 6 + [250] * 4
    wang = [0.0] * len(a_sched)
    buf, Es, pens, Ls = i18.run_cpp_batch_inflate(
        states, [L_tgt] * len(states), a_sched, iters, wang,
        n_kicks=4, kick_ang=0.30, kick_xy=0.12, n_threads=7, base_seed=555 + n,
    )
    assert np.min(pens) < 1e-6
    best_i = int(np.argmin(pens + Es))
    ver = u.verify_official(n, buf[best_i, :n], buf[best_i, n : 2 * n], buf[best_i, 2 * n :])
    assert ver["ok"]
    assert ver["L_float"] <= L_tgt + 1e-6
