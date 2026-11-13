"""Unit tests for Idea 19 symmetry-reduced (C4/C2/Free) C++ spatial-hash SAT & ILS solver."""

import math
from pathlib import Path
import sys

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import utils as u  # noqa: E402
from idea19_run import (  # noqa: E402
    cpp_eval_sym_energy_grad,
    cpp_expand_sym,
    full_to_free_sym,
    run_cpp_batch_ils,
    sym_dim,
)


def _finite_diff_sym_grad(n: int, sym_mode: int, q: np.ndarray, L: float, side: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    g_fd = np.zeros_like(q)
    for i in range(len(q)):
        qp = q.copy()
        qm = q.copy()
        qp[i] += eps
        qm[i] -= eps
        ep, _ = cpp_eval_sym_energy_grad(n, sym_mode, qp, L, 2.0, 1.0, side)
        em, _ = cpp_eval_sym_energy_grad(n, sym_mode, qm, L, 2.0, 1.0, side)
        g_fd[i] = (ep - em) / (2.0 * eps)
    return g_fd


def test_c2_and_c4_gradient_chain_rule() -> None:
    """Verify exact C2 (even & odd n) and C4 (0 mod 4 & 1 mod 4) gradient folding vs finite differences."""
    rng = np.random.default_rng(1901)
    for n, mode, L in [(50, 2, 7.60), (51, 2, 7.65), (20, 4, 4.95), (21, 4, 5.05)]:
        _, _, dim = sym_dim(n, mode)
        q = rng.uniform(-0.32 * L, 0.32 * L, size=dim)
        # Keep angles non-degenerate so |cos|, |sin|, SAT winner axes are smooth locally
        m_orb, has_c, _ = sym_dim(n, mode)
        q[2 * m_orb : 2 * m_orb + m_orb] = rng.uniform(0.18, 0.68, size=m_orb)
        if has_c:
            q[-1] = 0.37

        e_cpp, g_cpp = cpp_eval_sym_energy_grad(n, mode, q, L, 2.0, 1.0, side=1.0)
        g_fd = _finite_diff_sym_grad(n, mode, q, L, side=1.0)
        assert e_cpp > 0.0
        max_diff = float(np.max(np.abs(g_cpp - g_fd)))
        assert max_diff < 2e-5, f"mode={mode}, n={n} max grad diff={max_diff}"

        # Also cross-check expanded energy matches Python utils.sat_penalty_grad at identical coords
        x, y, th = cpp_expand_sym(n, mode, L, q)
        e_py, _ = u.sat_penalty_grad(np.concatenate([x, y, th]), L, w_wall=2.0, w_pair=1.0, side=1.0)
        assert abs(e_cpp - e_py) < 1e-11


def test_c4_c2_and_free_ils_batch_convergence() -> None:
    """Verify C4 (n=17 at L=4.85), C2 (n=50 hybrid at L<=7.59), and Free (n=17 at L<=4.73) converge."""
    rng = np.random.default_rng(1902)
    n = 17

    # 1. C4 test (17 = 4*4 + 1 -> dim = 13) at C4-feasible L=4.85
    m_orb, has_c, dim4 = sym_dim(n, 4)
    assert (m_orb, has_c, dim4) == (4, 1, 13)
    q4_list = []
    for _ in range(16):
        rx = rng.uniform(0.55, 1.85, m_orb)
        ry = rng.uniform(0.55, 1.85, m_orb)
        rt = rng.uniform(0.0, math.pi / 2, m_orb)
        q4_list.append(np.concatenate([rx, ry, rt, [float(rng.uniform(0.0, math.pi / 4))]]))

    qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
        n, 4, q4_list, [[4.95, 4.86]] * len(q4_list), [6, 10],
        a_start=0.88, n_melt_stages=3, n_threads=4, base_seed=19020,
    )
    assert np.min(po) < 1e-10
    best_i = int(np.argmin(lf))
    assert lf[best_i] <= 4.86 + 1e-9
    x, y, th = cpp_expand_sym(n, 4, float(lf[best_i]), qf[best_i])
    pen_py, L_py = u.sat_max_overlap_and_bbox(x, y, th)
    assert pen_py < 1e-10
    assert abs(L_py - float(lf[best_i])) < 1e-10

    # 2. Free mode parity check with full_to_free_sym and n=17 non-45 target L <= 4.74
    q_free_0 = full_to_free_sym(x, y, th, L_py)
    assert len(q_free_0) == 3 * n
    xf, yf, tf = cpp_expand_sym(n, 1, L_py, q_free_0)
    assert np.allclose(xf, x) and np.allclose(yf, y) and np.allclose(tf, th)

    q1_list = [
        np.concatenate([rng.uniform(-1.85, 1.85, n), rng.uniform(-1.85, 1.85, n), rng.uniform(0.0, math.pi / 2, n)])
        for _ in range(24)
    ]
    _, qf1, lf1, _, po1, _ = run_cpp_batch_ils(
        n, 1, q1_list, [[4.82, 4.76, 4.735]] * len(q1_list), [8, 12, 16],
        a_start=0.86, n_melt_stages=3, n_threads=4, base_seed=19021,
    )
    assert np.min(po1) < 1e-10 and np.min(lf1) <= 4.74

