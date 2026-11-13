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
import idea9_subpocket as i9  # noqa: E402
import idea10_gentle_transfer as i10  # noqa: E402
import idea11_force_chain as i11  # noqa: E402
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


def test_idea9_subpocket_freeze_and_cuda():
    # 5x5 grid with 4 tilted interior squares; verify frozen subset has 0 overlap and stays fixed in CUDA
    pts = []
    for ix in range(5):
        for iy in range(5):
            ang = 0.35 if (1 <= ix <= 2 and 1 <= iy <= 2) else 0.0
            pts.append((0.6 + ix * 1.15, 0.6 + iy * 1.15, ang))
    pos = np.array([[x, y] for x, y, _ in pts], dtype=np.float64)
    th = np.array([a for _, _, a in pts], dtype=np.float64)
    info = i7.extract_lp_duals_and_holes(pos, th, polish_rounds=1)
    L_base = float(info["L"])
    p_ord, t_ord, order, nmob = i9.select_mobile_and_shift_frozen(
        pos, th, L_base=L_base, L_target=L_base + 0.02, info=info, r_cut=2.0, max_mob=12
    )
    assert 4 <= nmob <= 12
    assert len(order) == 25
    assert pc.sat_max_penetration(p_ord[nmob:], t_ord[nmob:]) == 0.0

    bx, by, bth, nmob_b = i9.build_subpocket_batch(
        25, pos, th, L_target=L_base + 0.05, info=info, batch=8, r_cut=2.0, max_mob=12
    )
    ox, oy, oth, open_a, olam_a = i9.run_sat_sub_cuda(
        bx, by, bth, L_val=L_base + 0.05, n_mob=nmob_b, seed=77, mode=1
    )
    assert ox.shape == (8, 25)
    assert np.allclose(ox[:, nmob_b:], bx[:, nmob_b:], atol=1e-6)
    assert np.allclose(oy[:, nmob_b:], by[:, nmob_b:], atol=1e-6)
    assert float(open_a.min()) < 1e-3
    assert float(olam_a.min()) < 1.002


def test_idea10_gentle_transfer_pool():
    pts = []
    for ix in range(5):
        for iy in range(5):
            ang = -0.45 if abs(ix - iy) <= 1 else 0.0
            pts.append((0.6 + ix * 1.15, 0.6 + iy * 1.15, ang))
    pos = np.array([[x, y] for x, y, _ in pts], dtype=np.float64)
    th = np.array([a for _, _, a in pts], dtype=np.float64)
    info = i7.extract_lp_duals_and_holes(pos, th, polish_rounds=1)
    info["n_src"] = 25
    info["holes"] = i10.find_diverse_holes(np.stack([info["x"], info["y"]], axis=1), float(info["L"]), top_k=6)
    assert len(info["holes"]) >= 1

    # Test same-n (25 -> 25) and cross-n stoichiometry transfer (25 -> 27, 25 -> 24)
    for n_tgt in (25, 27, 24):
        sx, sy, st = i10.generate_idea10_pool(n_tgt, [info], inc_L=float(info["L"]) - 0.02, count=12, rng_seed=99)
        assert sx.shape == (12, n_tgt)
        assert sy.shape == (12, n_tgt)
        assert st.shape == (12, n_tgt)
        assert np.all(np.isfinite(sx)) and np.all(np.isfinite(sy)) and np.all(np.isfinite(st))
        assert np.all(sx >= 0.4) and np.all(sy >= 0.4)


def test_idea11_force_chain_and_fast_mode4():
    pts = []
    for ix in range(5):
        for iy in range(5):
            ang = -0.45 if abs(ix - iy) <= 1 else 0.0
            pts.append((0.6 + ix * 1.15, 0.6 + iy * 1.15, ang))
    pos = np.array([[x, y] for x, y, _ in pts], dtype=np.float64)
    th = np.array([a for _, _, a in pts], dtype=np.float64)
    info = i7.extract_lp_duals_and_holes(pos, th, polish_rounds=1)
    info["n_src"] = 25
    info["holes"] = i10.find_diverse_holes(np.stack([info["x"], info["y"]], axis=1), float(info["L"]), top_k=6)

    for n_tgt in (25, 27, 24):
        sx, sy, st = i11.generate_idea11_pool(n_tgt, [info], inc_L=float(info["L"]) - 0.02, count=15, rng_seed=111)
        assert sx.shape == (15, n_tgt)
        assert sy.shape == (15, n_tgt)
        assert st.shape == (15, n_tgt)
        assert np.all(np.isfinite(sx)) and np.all(np.isfinite(sy)) and np.all(np.isfinite(st))
        assert np.all(sx >= 0.4) and np.all(sy >= 0.4)

    # Verify fast_mode=4 (Idea 11 symmetric box-squeeze & split-half-crystal CUDA kernel)
    rng = np.random.default_rng(31)
    pos5 = rng.uniform(0.6, 2.4, (12, 5, 2))
    th5 = rng.uniform(-0.3, 0.3, (12, 5))
    box5 = np.full(12, 3.0)
    out4 = riv2.run_cuda_batch(pos5, th5, box5, use_elastic_L=1, fast_mode=4, seed=73)
    assert float(np.min(out4["L_cert"])) < 2.85
    assert pc.sat_max_penetration(out4["pos"][int(np.argmin(out4["L_cert"]))], out4["th"][int(np.argmin(out4["L_cert"]))]) == 0.0


def test_idea12_lpocket_and_bridge_identities():
    # Verify exact sheared strip width w*cos(th)+sin(th)=W and algebraic bridge root (a+f)^2+(b+f)^2=w^2
    # Example 1: P=7, w=8 sheared row across W = 7.664318 (n=971 scale)
    W = 7.664318
    w = 8
    R = math.hypot(w, 1.0)
    th = math.acos(W / R) + math.atan2(1.0, float(w))
    assert abs(w * math.cos(th) + math.sin(th) - W) < 1e-12
    # Place 1 sheared row of w=8 squares strictly inside [0, W] x [0, 5] and verify 0 overlap
    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (c + s)
    pos = np.array([[hx + q * c, hx + q * s] for q in range(w)], dtype=np.float64)
    ths = np.full(w, th, dtype=np.float64)
    assert np.min(pos[:, 0] - hx) >= -1e-12
    assert np.max(pos[:, 0] + hx) <= W + 1e-12
    assert pc.sat_max_penetration(pos, ths) < 1e-12

    # Example 2: Exact algebraic bridge roots (w=4, a=1, b=3 -> sqrt(7)-2; w=8, a=5, b=5 -> 4*sqrt(2)-5)
    for w_b, a, b in [(4, 1, 3), (8, 5, 5), (9, 4, 7), (5, 1, 4)]:
        f_r = 0.5 * (math.sqrt(2 * w_b * w_b - (a - b) ** 2) - (a + b))
        assert abs((a + f_r) ** 2 + (b + f_r) ** 2 - w_b * w_b) < 1e-12
        th_r = math.atan2(a + f_r, b + f_r)
        cr, sr = math.cos(th_r), math.sin(th_r)
        # w_b tilted squares touching end-to-end span exact vector (b+f_r, a+f_r)
        span_x = w_b * cr
        span_y = w_b * sr
        assert abs(span_x - (b + f_r)) < 1e-12
        assert abs(span_y - (a + f_r)) < 1e-12
        chain = np.array([[1.0 + (i + 0.5) * cr, 1.0 + (i + 0.5) * sr] for i in range(w_b)], dtype=np.float64)
        assert pc.sat_max_penetration(chain, np.full(w_b, th_r)) < 1e-12


def test_idea13_bridge_filter_and_sheared_corridor() -> None:
    """Verify Idea 13 sub-inc_f bridge gating and band-aligned sheared lattice invariants."""
    def strict_roots(f_inc: float, w_max: int = 12) -> list[tuple[int, int, int, float]]:
        out = []
        for w in range(1, w_max + 1):
            for a in range(w):
                for b in range(a, w):
                    disc = 2 * w * w - (a - b) * (a - b)
                    if disc < 0:
                        continue
                    f_r = -0.5 * (a + b) + 0.5 * math.sqrt(disc)
                    if 0.45 <= f_r < f_inc - 1e-4:
                        out.append((w, a, b, f_r))
        return out

    # n=971 (f_inc ~ 0.66432): w=1 @ 45 deg (0.7071) must be excluded; (8,5,5) & (4,1,3) kept
    r971 = strict_roots(0.664318)
    assert all(not (w == 1 and a == 0 and b == 0) for w, a, b, _ in r971)
    assert any((w, a, b) == (8, 5, 5) for w, a, b, _ in r971)
    assert any((w, a, b) == (4, 1, 3) for w, a, b, _ in r971)

    # n=796 (f_inc ~ 0.72154): w=1 @ 45 deg (0.7071) and (6,3,4) both strictly sub-inc_f
    r796 = strict_roots(0.721540)
    assert any((w, a, b) == (1, 0, 0) for w, a, b, _ in r796)
    assert any((w, a, b) == (6, 3, 4) for w, a, b, _ in r796)

    # Band-aligned sheared basis: sh = -cot(th - psi) makes e2_sheared parallel to band axis
    f4 = math.sqrt(7.0) - 2.0
    th = math.atan2(1.0 + f4, 3.0 + f4)
    psi = -0.25 * math.pi
    nx, ny = -math.sin(psi), math.cos(psi)
    sh = -math.cos(th - psi) / math.sin(th - psi)
    assert abs(sh) <= 1.0
    e1 = np.array([math.cos(th), math.sin(th)])
    e2 = np.array([-math.sin(th) + sh * math.cos(th), math.cos(th) + sh * math.sin(th)])
    assert abs(e2[0] * nx + e2[1] * ny) < 1e-12
    pts = np.array([10.0 + i * e1 + j * e2 for i in range(4) for j in range(6)], dtype=np.float64)
    assert pc.sat_max_penetration(pts, np.full(len(pts), th)) < 1e-12


def test_idea14_joint_xytheta_slp_and_twin_pocket() -> None:
    """Verify joint_xytheta_slp_polish beats fixed-angle LP minimum via cooperative rotation."""
    # Two valid non-overlapping 25-deg squares already tightened by fixed_angle_lp_polish:
    # fixed_angle_lp_polish cannot shrink further without changing theta, whereas
    # joint_xytheta_slp_polish rotates both squares toward 0 deg and shrinks L cleanly.
    pos = np.array([[0.75, 0.75], [2.05, 0.75]], dtype=np.float64)
    th = np.array([0.35, 0.35], dtype=np.float64)
    assert pc.sat_max_penetration(pos, th) < 1e-12
    pos_lp, L_lp = pc.fixed_angle_lp_polish(pos, th, rounds=6)
    pos_s, th_s, L_slp = pc.joint_xytheta_slp_polish(pos_lp, th, rounds=20)
    assert L_slp < L_lp - 0.03
    assert pc.sat_max_penetration(pos_s, th_s) < 1e-11






