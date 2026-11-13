"""Unit tests for swarm_inflation SAT gradients, seeds, and polish/certification."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

POC_DIR = Path(__file__).resolve().parents[1] / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import idea7_chain_ruin as i7  # noqa: E402
import idea8_seam_surgery as i8  # noqa: E402
import polish_and_certify as pc  # noqa: E402
import run_inflation_v2 as riv2  # noqa: E402
import sat_cuda as sc_gpu  # noqa: E402
import seeds  # noqa: E402
import seeds_v2  # noqa: E402


def test_sat_analytical_gradient_matches_finite_difference():
    torch.manual_seed(7)
    B, N = 3, 12
    L = torch.full((B,), 3.4, dtype=torch.float64)
    pos = torch.rand((B, N, 2), dtype=torch.float64) * 3.0 + 0.2
    th = torch.rand((B, N), dtype=torch.float64) * 1.4 + 0.05

    for scale_a in (0.88, 0.95, 1.0):
        nbr_idx, nbr_mask, _ = sc_gpu.build_neighbor_list(pos, r_cut=2.1, k_max=11)
        assert nbr_mask.sum() > 0

        e0, _, gp, gt, _ = sc_gpu.sat_energy_and_grad(
            pos, th, scale_a, L, nbr_idx, nbr_mask, w_wall=5.0, w_align=0.02, sg_eps=1e-7
        )

        eps = 1e-6
        max_rel_err = 0.0
        for b in range(B):
            for i in range(N):
                for d in range(2):
                    p_plus = pos.clone()
                    p_minus = pos.clone()
                    p_plus[b, i, d] += eps
                    p_minus[b, i, d] -= eps
                    ep, *_ = sc_gpu.sat_energy_and_grad(
                        p_plus, th, scale_a, L, nbr_idx, nbr_mask, w_wall=5.0, w_align=0.02, sg_eps=1e-7
                    )
                    em, *_ = sc_gpu.sat_energy_and_grad(
                        p_minus, th, scale_a, L, nbr_idx, nbr_mask, w_wall=5.0, w_align=0.02, sg_eps=1e-7
                    )
                    fd = (ep[b] - em[b]) / (2.0 * eps)
                    an = gp[b, i, d]
                    denom = max(1.0, abs(float(fd)), abs(float(an)))
                    max_rel_err = max(max_rel_err, abs(float(fd - an)) / denom)

                t_plus = th.clone()
                t_minus = th.clone()
                t_plus[b, i] += eps
                t_minus[b, i] -= eps
                ep, *_ = sc_gpu.sat_energy_and_grad(
                    pos, t_plus, scale_a, L, nbr_idx, nbr_mask, w_wall=5.0, w_align=0.02, sg_eps=1e-7
                )
                em, *_ = sc_gpu.sat_energy_and_grad(
                    pos, t_minus, scale_a, L, nbr_idx, nbr_mask, w_wall=5.0, w_align=0.02, sg_eps=1e-7
                )
                fd = (ep[b] - em[b]) / (2.0 * eps)
                an = gt[b, i]
                denom = max(1.0, abs(float(fd)), abs(float(an)))
                max_rel_err = max(max_rel_err, abs(float(fd - an)) / denom)

        assert max_rel_err < 1e-5, f"scale_a={scale_a} max_rel_err={max_rel_err}"


def test_exact_allpairs_lambda_clears_overlap():
    # Two overlapping squares at distance 0.90 along x
    pos = np.array([[[0.5, 0.5], [1.4, 0.5]]], dtype=np.float64)
    th = np.array([[0.0, 0.0]], dtype=np.float64)
    L_cert, lam, pen = sc_gpu.exact_allpairs_lambda_and_L(pos, th, tau=1e-9)
    assert pen[0] > 0.09
    assert lam[0] > 1.10
    assert L_cert[0] > 2.0
    # After scaling centers by lam, overlap should be 0
    pos_scaled = pos * lam[0]
    _, lam2, pen2 = sc_gpu.exact_allpairs_lambda_and_L(pos_scaled, th, tau=0.0)
    assert pen2[0] == 0.0
    assert lam2[0] == 1.0


def test_seed_batch_shapes_and_bounds():
    n = 367
    L_targets = [19.672, 19.685]
    pos, th, box_L, meta = seeds.build_seed_batch(n=n, L_targets=L_targets, batch_size=15, rng_seed=42)
    assert pos.shape == (15, n, 2)
    assert th.shape == (15, n)
    assert box_L.shape == (15,)
    assert len(meta) == 15
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(th))
    for b in range(15):
        assert np.all(pos[b] >= 0.3)
        assert np.all(pos[b] <= box_L[b] - 0.3)


def test_polish_and_certify_synthetic_5():
    # Slightly expanded tilted-5 configuration; LP + dual certify should pack tightly
    s2 = math.sqrt(2.0)
    ang = np.array([0.0, 0.0, 0.0, 0.0, math.pi / 4.0], dtype=np.float64)
    L0 = 2.0 + 1.0 / s2 + 0.04
    pos = np.array(
        [
            [0.50, 0.50],
            [L0 - 0.50, 0.50],
            [0.50, L0 - 0.50],
            [L0 - 0.50, L0 - 0.50],
            [0.5 * L0, 0.5 * L0],
        ],
        dtype=np.float64,
    )
    p_feas, lam = pc.minimal_feasible_dilation(pos, ang)
    assert lam == 1.0
    p_lp, L_lp = pc.fixed_angle_lp_polish(p_feas, ang, rounds=4)
    p_slp, t_slp, L_slp = pc.lip_slp_polish(p_lp, ang, rounds=4)
    assert L_slp < L0
    cert = pc.dual_certify(5, p_slp, t_slp)
    assert cert["ok"]
    assert float(cert["L_lit"]) < 2.71


def test_seeds_v2_shapes_and_bounds():
    n = 367
    L_targets = [19.672, 19.685]
    pos, th, box_L, meta = seeds_v2.build_seed_batch_v2(n=n, L_targets=L_targets, batch_size=14, rng_seed=42)
    assert pos.shape == (14, n, 2)
    assert th.shape == (14, n)
    assert box_L.shape == (14,)
    assert len(meta) == 14
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(th))
    for b in range(14):
        assert np.all(pos[b] >= 0.3)
        assert np.all(pos[b] <= box_L[b] - 0.3)


def test_cuda_v2_synthetic_5_compresses_to_optimum():
    rng = np.random.default_rng(11)
    B, N = 24, 5
    pos0 = rng.uniform(0.6, 2.4, (B, N, 2))
    th0 = rng.uniform(-0.7, 0.7, (B, N))
    box_L = np.full(B, 3.0)
    out = riv2.run_cuda_batch(pos0, th0, box_L, use_elastic_L=1, fast_mode=1, seed=42)
    bi = int(np.argmin(out["L_cert"]))
    assert out["L_cert"][bi] < 2.71
    assert pc.sat_max_penetration(out["pos"][bi], out["th"][bi]) == 0.0
    cert = pc.dual_certify(5, out["pos"][bi], out["th"][bi])
    assert cert["ok"]
    assert float(cert["L_lit"]) < 2.708


def test_cuda_v2_multiwarp_n367_certification_matches_python():
    # Regression test for N > 256 (multi-warp shared reductions in sat_lmin.cu)
    n = 367
    pos0, th0, box_L, _ = seeds_v2.build_seed_batch_v2(
        n=n, L_targets=[19.671, 19.685], batch_size=4, rng_seed=7
    )
    out = riv2.run_cuda_batch(pos0, th0, box_L, use_elastic_L=1, fast_mode=1, seed=7)
    for b in range(4):
        assert pc.sat_max_penetration(out["pos"][b], out["th"][b]) <= 0.0
        _, py_lam = pc.minimal_feasible_dilation(out["pos"][b], out["th"][b])
        assert py_lam == 1.0
        py_L = pc.exact_bbox_side(out["pos"][b], out["th"][b])
        assert abs(py_L - float(out["L_cert"][b])) < 1e-6
        assert float(out["L_cert"][b]) < 21.5


def test_idea7_and_idea8_surgery_pools_and_fast_mode3():
    # Synthetic 16-square grid with 4 tilted interior squares
    pts = []
    for ix in range(4):
        for iy in range(4):
            ang = 0.45 if (1 <= ix <= 2 and 1 <= iy <= 2) else 0.0
            pts.append((0.6 + ix * 1.15, 0.6 + iy * 1.15, ang))
    pos = np.array([[x, y] for x, y, _ in pts], dtype=np.float64)
    th = np.array([a for _, _, a in pts], dtype=np.float64)
    info = i7.extract_lp_duals_and_holes(pos, th, polish_rounds=2)
    assert info["sq_load"].shape == (16,)
    assert len(info["holes"]) >= 1
    assert np.isfinite(info["L"])

    inc_L = info["L"] - 0.05
    sx7, sy7, st7 = i7.generate_surgical_pool(16, [info], inc_L=inc_L, count=8, rng_seed=123)
    assert sx7.shape == (8, 16)
    assert np.all(np.isfinite(sx7)) and np.all(np.isfinite(sy7)) and np.all(np.isfinite(st7))
    assert np.all(sx7 >= 0.4) and np.all(sy7 >= 0.4)

    sx8, sy8, st8 = i8.generate_idea8_pool(16, [info], inc_L=inc_L, count=8, rng_seed=456)
    assert sx8.shape == (8, 16)
    assert np.all(np.isfinite(sx8)) and np.all(np.isfinite(sy8)) and np.all(np.isfinite(st8))
    assert np.all(sx8 >= 0.4) and np.all(sy8 >= 0.4)

    # Verify fast_mode=3 (Idea 8 CUDA multi-temp anneal) converges cleanly on synthetic N=5
    rng = np.random.default_rng(19)
    pos5 = rng.uniform(0.6, 2.4, (12, 5, 2))
    th5 = rng.uniform(-0.7, 0.7, (12, 5))
    box5 = np.full(12, 3.0)
    out3 = riv2.run_cuda_batch(pos5, th5, box5, use_elastic_L=1, fast_mode=3, seed=42)
    assert float(np.min(out3["L_cert"])) < 2.72
    assert pc.sat_max_penetration(out3["pos"][int(np.argmin(out3["L_cert"]))], out3["th"][int(np.argmin(out3["L_cert"]))]) == 0.0
