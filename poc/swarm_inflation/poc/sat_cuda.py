"""Batched PyTorch CUDA/CPU SAT penalty optimizer with 0.88 -> 1.00 inflation.

Supports arbitrary batch sizes via chunked neighbor-list rebuilding, analytical
SAT penetration + wall gradients (no autograd overhead), soft nematic alignment
annealing, stress-weighted Langevin shakes, void-relocation moves, and float64
final polishing + exact all-pairs uniform expansion certification check.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def build_neighbor_list(
    pos: torch.Tensor,
    r_cut: float,
    k_max: int = 48,
    chunk_b: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Build per-square neighbor indices `(B, N, K)` using chunked `cdist` + `topk`."""
    B, N, _ = pos.shape
    k = min(k_max, max(1, N - 1))
    device = pos.device
    nbr_idx = torch.zeros((B, N, k), dtype=torch.long, device=device)
    nbr_mask = torch.zeros((B, N, k), dtype=torch.bool, device=device)
    overflow = False

    if N <= 1:
        return nbr_idx, nbr_mask, False

    for b0 in range(0, B, chunk_b):
        b1 = min(B, b0 + chunk_b)
        p_c = pos[b0:b1]
        dist = torch.cdist(p_c, p_c)
        dist.diagonal(dim1=1, dim2=2).fill_(1e9)
        vals, idx = torch.topk(dist, k=k, dim=-1, largest=False)
        nbr_idx[b0:b1] = idx
        nbr_mask[b0:b1] = vals < r_cut
        if k == k_max and (vals[:, :, -1] < r_cut * 0.85).any().item():
            overflow = True

    return nbr_idx, nbr_mask, overflow


def sat_energy_and_grad(
    pos: torch.Tensor,
    th: torch.Tensor,
    scale_a: float,
    box_L: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_mask: torch.Tensor,
    w_wall: float = 5.0,
    w_align: torch.Tensor | float = 0.0,
    sg_eps: float = 1e-6,
    micro_b: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact SAT penetration + wall energy and analytical gradients.

    Returns:
        total_e: `(B,)` total scalar energy per trajectory
        sq_e: `(B, N)` per-square local energy (for stress-weighted moves)
        g_pos: `(B, N, 2)` gradient w.r.t. `(x, y)`
        g_th: `(B, N)` gradient w.r.t. `theta`
        max_viol: `(B,)` max penetration or wall violation per trajectory
    """
    B, N, _ = pos.shape
    device = pos.device
    dtype = pos.dtype

    total_e = torch.zeros(B, dtype=dtype, device=device)
    sq_e = torch.zeros((B, N), dtype=dtype, device=device)
    g_pos = torch.zeros((B, N, 2), dtype=dtype, device=device)
    g_th = torch.zeros((B, N), dtype=dtype, device=device)
    max_viol = torch.zeros(B, dtype=dtype, device=device)

    ar_n = torch.arange(N, device=device)[None, :, None]
    r_cut2 = (2.05 * scale_a) ** 2
    align_cut2 = (1.30 * scale_a) ** 2
    half_a = 0.5 * scale_a
    eps2 = sg_eps * sg_eps

    if not torch.is_tensor(w_align):
        w_align_t = torch.full((B, 1, 1), float(w_align), dtype=dtype, device=device)
    else:
        w_align_t = w_align.to(dtype=dtype, device=device).view(B, 1, 1)
    use_align = bool((w_align_t > 0.0).any().item())

    for b0 in range(0, B, micro_b):
        b1 = min(B, b0 + micro_b)
        bm = b1 - b0
        p = pos[b0:b1]       # (bm, N, 2)
        t = th[b0:b1]        # (bm, N)
        L = box_L[b0:b1, None]  # (bm, 1)
        idx = nbr_idx[b0:b1]
        msk = nbr_mask[b0:b1]
        wa = w_align_t[b0:b1]

        c = torch.cos(t)
        s = torch.sin(t)
        sc = c / torch.sqrt(c * c + eps2)
        ss = s / torch.sqrt(s * s + eps2)
        ext = half_a * (torch.abs(c) + torch.abs(s))
        dext = half_a * (-sc * s + ss * c)

        v0 = (ext - p[..., 0]).clamp(min=0.0)
        v1 = (p[..., 0] + ext - L).clamp(min=0.0)
        v2 = (ext - p[..., 1]).clamp(min=0.0)
        v3 = (p[..., 1] + ext - L).clamp(min=0.0)

        ew = w_wall * (v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3)
        gx_w = 2.0 * w_wall * (-v0 + v1)
        gy_w = 2.0 * w_wall * (-v2 + v3)
        gth_w = 2.0 * w_wall * (v0 + v1 + v2 + v3) * dext
        w_max = torch.maximum(torch.maximum(v0, v1), torch.maximum(v2, v3)).amax(dim=-1)

        b_ar = torch.arange(bm, device=device)[:, None, None]
        pj = p[b_ar, idx]    # (bm, N, K, 2)
        cj = c[b_ar, idx]    # (bm, N, K)
        sj = s[b_ar, idx]    # (bm, N, K)

        dx = pj[..., 0] - p[:, :, None, 0]
        dy = pj[..., 1] - p[:, :, None, 1]
        d2 = dx * dx + dy * dy
        active = msk & (d2 < r_cut2)

        ci = c[:, :, None]
        si = s[:, :, None]
        cr = ci * cj + si * sj
        sr = si * cj - ci * sj
        rad = half_a * (1.0 + torch.abs(cr) + torch.abs(sr))

        scr = cr / torch.sqrt(cr * cr + eps2)
        ssr = sr / torch.sqrt(sr * sr + eps2)
        drad = half_a * (-scr * sr + ssr * cr)

        p0 = dx * ci + dy * si
        p1 = -dx * si + dy * ci
        p2 = dx * cj + dy * sj
        p3 = -dx * sj + dy * cj

        proj = torch.stack((p0, p1, p2, p3), dim=-1)
        ax_x = torch.stack((ci.expand_as(cj), -si.expand_as(cj), cj, -sj), dim=-1)
        ax_y = torch.stack((si.expand_as(cj), ci.expand_as(cj), sj, cj), dim=-1)

        z = torch.abs(proj) - rad[..., None]
        z_max, k_star = torch.max(z, dim=-1)
        pen = (-z_max).clamp(min=0.0)
        pen = torch.where(active, pen, 0.0)

        idx4 = k_star[..., None]
        p_star = torch.gather(proj, -1, idx4).squeeze(-1)
        ux = torch.gather(ax_x, -1, idx4).squeeze(-1)
        uy = torch.gather(ax_y, -1, idx4).squeeze(-1)

        tie_sgn = torch.where(idx > ar_n, 1.0, -1.0).to(dtype)
        sp = torch.where(p_star > 0.0, 1.0, torch.where(p_star < 0.0, -1.0, tie_sgn))

        is_i_ax = (k_star < 2).to(dtype)
        dpen_dxi = sp * ux
        dpen_dyi = sp * uy
        dpen_dthi = drad - sp * (dy * ux - dx * uy) * is_i_ax

        ov = pen > 0.0
        w_p = torch.where(ov, 2.0 * pen, 0.0)

        gx_p = (w_p * dpen_dxi).sum(dim=-1)
        gy_p = (w_p * dpen_dyi).sum(dim=-1)
        gth_p = (w_p * dpen_dthi).sum(dim=-1)
        ep_i = 0.5 * (pen * pen).sum(dim=-1)

        if use_align:
            tj = t[b_ar, idx]
            dth = t[:, :, None] - tj
            a_msk = msk & (d2 < align_cut2)
            ea_i = 0.5 * (wa * torch.where(a_msk, torch.sin(2.0 * dth) ** 2, 0.0)).sum(dim=-1)
            gth_a = (wa * torch.where(a_msk, 2.0 * torch.sin(4.0 * dth), 0.0)).sum(dim=-1)
        else:
            ea_i = 0.0
            gth_a = 0.0

        sq_loc = ew + ep_i
        sq_e[b0:b1] = sq_loc
        total_e[b0:b1] = (sq_loc + ea_i).sum(dim=-1)
        g_pos[b0:b1, :, 0] = gx_w + gx_p
        g_pos[b0:b1, :, 1] = gy_w + gy_p
        g_th[b0:b1] = gth_w + gth_p + gth_a
        p_max = pen.amax(dim=(-1, -2))
        max_viol[b0:b1] = torch.maximum(w_max, p_max)

    return total_e, sq_e, g_pos, g_th, max_viol


@torch.no_grad()
def _void_relocate_and_align(
    pos: torch.Tensor,
    th: torch.Tensor,
    sq_e: torch.Tensor,
    scale_a: float,
    box_L: torch.Tensor,
    n_moves: int = 3,
    n_probes: int = 192,
) -> None:
    """In-place greedy relocation of highest-stress squares into largest open voids."""
    B, N, _ = pos.shape
    device = pos.device
    dtype = pos.dtype
    if N < 4:
        return

    # Sample candidate probe coordinates per trajectory inside safe box interior
    margin = 0.55 * scale_a
    span = (box_L - 2.0 * margin).clamp(min=0.1)
    u = torch.rand((B, n_probes, 2), dtype=dtype, device=device)
    probes = margin + u * span[:, None, None]  # (B, P, 2)

    # Distance from each probe to nearest existing square center
    d_probe = torch.cdist(probes, pos)  # (B, P, N)
    min_d, near_idx = d_probe.min(dim=-1)  # (B, P)
    _, best_p_idx = torch.topk(min_d, k=n_moves, dim=-1, largest=True)  # (B, M)
    _, worst_i_idx = torch.topk(sq_e, k=n_moves, dim=-1, largest=True)  # (B, M)

    b_ar = torch.arange(B, device=device)[:, None]
    best_probes = probes[b_ar, best_p_idx]  # (B, M, 2)
    near_sq_at_probe = near_idx[b_ar, best_p_idx]  # (B, M)
    near_th = th[b_ar, near_sq_at_probe]  # (B, M)

    # Evaluate exact local energy before vs after at candidate slots with 4 angles
    cand_angles = torch.stack(
        [
            near_th,
            torch.zeros_like(near_th),
            near_th + math.pi * 0.5,
            torch.full_like(near_th, math.pi * 0.25),
        ],
        dim=-1,
    )  # (B, M, 4)

    for m in range(n_moves):
        sq_idx = worst_i_idx[:, m]  # (B,)
        cur_p = pos[torch.arange(B, device=device), sq_idx]  # (B, 2)
        cur_t = th[torch.arange(B, device=device), sq_idx]  # (B,)
        new_p = best_probes[:, m, :]  # (B, 2)

        old_e = sq_e[torch.arange(B, device=device), sq_idx]
        best_e = old_e.clone()
        best_p = cur_p.clone()
        best_t = cur_t.clone()

        # Check all 4 candidate orientations at new_p
        for a_k in range(4):
            try_t = cand_angles[:, m, a_k]
            e_cand = _single_square_local_energy(pos, th, sq_idx, new_p, try_t, scale_a, box_L)
            better = e_cand + 1e-6 < best_e
            best_e = torch.where(better, e_cand, best_e)
            best_p = torch.where(better[:, None], new_p, best_p)
            best_t = torch.where(better, try_t, best_t)

        # Also check swapping angle with nearest neighbor at old position or snapping angle
        pos[torch.arange(B, device=device), sq_idx] = best_p
        th[torch.arange(B, device=device), sq_idx] = best_t


def _single_square_local_energy(
    pos: torch.Tensor,
    th: torch.Tensor,
    sq_idx: torch.Tensor,
    cand_p: torch.Tensor,
    cand_t: torch.Tensor,
    scale_a: float,
    box_L: torch.Tensor,
    w_wall: float = 5.0,
) -> torch.Tensor:
    """Compute exact local SAT + wall energy for one candidate square per batch item."""
    B, N, _ = pos.shape
    device = pos.device
    half_a = 0.5 * scale_a
    ci = torch.cos(cand_t)
    si = torch.sin(cand_t)
    ext = half_a * (torch.abs(ci) + torch.abs(si))

    v0 = (ext - cand_p[:, 0]).clamp(min=0.0)
    v1 = (cand_p[:, 0] + ext - box_L).clamp(min=0.0)
    v2 = (ext - cand_p[:, 1]).clamp(min=0.0)
    v3 = (cand_p[:, 1] + ext - box_L).clamp(min=0.0)
    ew = w_wall * (v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3)

    dx = pos[:, :, 0] - cand_p[:, 0:1]
    dy = pos[:, :, 1] - cand_p[:, 1:2]
    d2 = dx * dx + dy * dy
    cj = torch.cos(th)
    sj = torch.sin(th)
    cr = ci[:, None] * cj + si[:, None] * sj
    sr = si[:, None] * cj - ci[:, None] * sj
    rad = half_a * (1.0 + torch.abs(cr) + torch.abs(sr))

    p0 = dx * ci[:, None] + dy * si[:, None]
    p1 = -dx * si[:, None] + dy * ci[:, None]
    p2 = dx * cj + dy * sj
    p3 = -dx * sj + dy * cj
    proj = torch.stack((p0, p1, p2, p3), dim=-1)
    z_max = (torch.abs(proj) - rad[..., None]).amax(dim=-1)
    pen = (-z_max).clamp(min=0.0)
    mask = d2 < (2.05 * scale_a) ** 2
    mask[torch.arange(B, device=device), sq_idx] = False
    pen = torch.where(mask, pen, 0.0)
    return ew + 0.5 * (pen * pen).sum(dim=-1)


@torch.no_grad()
def _orientation_swap_moves(
    pos: torch.Tensor,
    th: torch.Tensor,
    sq_e: torch.Tensor,
    scale_a: float,
    box_L: torch.Tensor,
    n_tries: int = 4,
) -> None:
    """Try snapping stressed squares to a neighbor's angle (or +pi/2) if local energy drops."""
    B, N, _ = pos.shape
    device = pos.device
    if N < 4:
        return
    _, worst_idx = torch.topk(sq_e, k=min(n_tries, N), dim=-1, largest=True)
    b_ar = torch.arange(B, device=device)

    for k in range(worst_idx.shape[1]):
        sq_i = worst_idx[:, k]
        p_i = pos[b_ar, sq_i]
        t_i = th[b_ar, sq_i]
        # Find nearest neighbor j != i
        d2 = ((pos - p_i[:, None, :]) ** 2).sum(dim=-1)
        d2[b_ar, sq_i] = 1e9
        _, nbr_j = d2.min(dim=-1)
        t_j = th[b_ar, nbr_j]
        e0 = _single_square_local_energy(pos, th, sq_i, p_i, t_i, scale_a, box_L)
        best_e = e0
        best_t = t_i.clone()
        for cand_t in (t_j, t_j + math.pi * 0.5, -t_j):
            e_try = _single_square_local_energy(pos, th, sq_i, p_i, cand_t, scale_a, box_L)
            better = e_try + 1e-7 < best_e
            best_e = torch.where(better, e_try, best_e)
            best_t = torch.where(better, cand_t, best_t)
        th[b_ar, sq_i] = best_t


@torch.no_grad()
def exact_allpairs_lambda_and_L(
    pos: np.ndarray | torch.Tensor,
    th: np.ndarray | torch.Tensor,
    tau: float = 1e-10,
    chunk_b: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact float64 all-pairs SAT clearance via minimal uniform dilation `lambda`.

    Computes `lam >= 1.0` so that scaling centers `(x, y) -> lam * (x, y)` gives
    at least `tau` separation on the best separating axis for every pair, and
    returns `(L_cert, lam, max_pen_at_1)` as float64 numpy arrays of shape `(B,)`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_t = torch.as_tensor(pos, dtype=torch.float64, device=device)
    t_t = torch.as_tensor(th, dtype=torch.float64, device=device)
    if p_t.ndim == 2:
        p_t = p_t.unsqueeze(0)
        t_t = t_t.unsqueeze(0)
    B, N, _ = p_t.shape

    lam_out = np.ones(B, dtype=np.float64)
    L_out = np.full(B, 1e9, dtype=np.float64)
    pen_out = np.full(B, 1e9, dtype=np.float64)

    for b0 in range(0, B, chunk_b):
        b1 = min(B, b0 + chunk_b)
        p = p_t[b0:b1]
        t = t_t[b0:b1]
        c = torch.cos(t)
        s = torch.sin(t)
        ext = 0.5 * (torch.abs(c) + torch.abs(s))

        # Shift to origin [0, ...] before scaling
        minx = (p[..., 0] - ext).amin(dim=-1, keepdim=True)
        miny = (p[..., 1] - ext).amin(dim=-1, keepdim=True)
        px = p[..., 0] - minx
        py = p[..., 1] - miny

        dx = px[:, :, None] - px[:, None, :]
        dy = py[:, :, None] - py[:, None, :]
        d2 = dx * dx + dy * dy

        ci = c[:, :, None]
        si = s[:, :, None]
        cj = c[:, None, :]
        sj = s[:, None, :]
        rad = 0.5 * (1.0 + torch.abs(ci * cj + si * sj) + torch.abs(si * cj - ci * sj))

        v1 = torch.abs(dx * ci + dy * si)
        v2 = torch.abs(-dx * si + dy * ci)
        v3 = torch.abs(dx * cj + dy * sj)
        v4 = torch.abs(-dx * sj + dy * cj)
        vmax = torch.maximum(torch.maximum(v1, v2), torch.maximum(v3, v4))

        iu = torch.triu_indices(N, N, offset=1, device=device)
        d2_u = d2[:, iu[0], iu[1]]
        vmax_u = vmax[:, iu[0], iu[1]]
        rad_u = rad[:, iu[0], iu[1]]

        near = d2_u < 2.05
        pen_u = torch.where(near, (rad_u - vmax_u).clamp(min=0.0), 0.0)
        pen_max = pen_u.amax(dim=-1)

        req = torch.where(
            near & (vmax_u > 1e-12),
            (rad_u + tau) / vmax_u.clamp(min=1e-12),
            torch.where(near, 1e9, 1.0),
        )
        lam = torch.maximum(torch.ones_like(pen_max), req.amax(dim=-1))

        px_s = px * lam[:, None]
        py_s = py * lam[:, None]
        Lx = (px_s + ext).amax(dim=-1) - (px_s - ext).amin(dim=-1)
        Ly = (py_s + ext).amax(dim=-1) - (py_s - ext).amin(dim=-1)
        Lc = torch.maximum(Lx, Ly)

        lam_out[b0:b1] = lam.cpu().numpy()
        L_out[b0:b1] = Lc.cpu().numpy()
        pen_out[b0:b1] = pen_max.cpu().numpy()

    return L_out, lam_out, pen_out


@torch.no_grad()
def inflate_batch(
    init_pos: np.ndarray,
    init_th: np.ndarray,
    box_L: np.ndarray | float,
    align_mask: np.ndarray | None = None,
    stages: list[tuple[float, int, float]] | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run batched 0.88 -> 1.00 inflation on GPU/CPU with shake/swap/alignment moves.

    `stages` is a list of `(scale_a, n_steps, lr_base)`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    if stages is None:
        stages = [
            (0.880, 120, 0.018),
            (0.900, 120, 0.016),
            (0.920, 140, 0.014),
            (0.935, 140, 0.012),
            (0.950, 160, 0.010),
            (0.962, 160, 0.008),
            (0.972, 180, 0.006),
            (0.980, 180, 0.0045),
            (0.986, 200, 0.0032),
            (0.991, 200, 0.0022),
            (0.995, 220, 0.0014),
            (0.998, 220, 0.0008),
            (0.9995, 220, 0.0004),
            (1.000, 300, 0.0002),
        ]

    B, N, _ = init_pos.shape
    pos = torch.tensor(init_pos, dtype=torch.float32, device=device)
    th = torch.tensor(init_th, dtype=torch.float32, device=device)
    if np.ndim(box_L) == 0:
        L_t = torch.full((B,), float(box_L), dtype=torch.float32, device=device)
    else:
        L_t = torch.tensor(box_L, dtype=torch.float32, device=device)

    if align_mask is None:
        align_vec = torch.zeros(B, dtype=torch.float32, device=device)
        align_vec[B // 2 :] = 1.0
    else:
        align_vec = torch.tensor(align_mask, dtype=torch.float32, device=device)

    # Adam moments
    m_p = torch.zeros_like(pos)
    v_p = torch.zeros_like(pos)
    m_t = torch.zeros_like(th)
    v_t = torch.zeros_like(th)
    beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
    step_global = 0

    nbr_rebuild_every = 10
    k_max = min(48, max(8, N - 1))

    for st_idx, (scale_a, n_steps, lr_base) in enumerate(stages):
        # Switch to float64 for final stages >= 0.995
        if scale_a >= 0.995 and pos.dtype != torch.float64:
            pos = pos.to(torch.float64)
            th = th.to(torch.float64)
            L_t = L_t.to(torch.float64)
            align_vec = align_vec.to(torch.float64)
            m_p = m_p.to(torch.float64).mul_(0.2)
            v_p = v_p.to(torch.float64).mul_(0.2)
            m_t = m_t.to(torch.float64).mul_(0.2)
            v_t = v_t.to(torch.float64).mul_(0.2)
        else:
            # Decay momentum slightly at each inflation jump
            m_p.mul_(0.5)
            m_t.mul_(0.5)

        # Nematic alignment weight active during [0.88, 0.965], 0 afterwards
        if scale_a < 0.965:
            w_align_val = 0.018 * max(0.0, (0.965 - scale_a) / 0.085)
            w_align_cur = align_vec * w_align_val
        else:
            w_align_cur = 0.0

        nbr_idx, nbr_mask, _ = build_neighbor_list(pos, r_cut=1.85 * scale_a, k_max=k_max)

        for it in range(n_steps):
            if it > 0 and it % nbr_rebuild_every == 0:
                nbr_idx, nbr_mask, _ = build_neighbor_list(pos, r_cut=1.85 * scale_a, k_max=k_max)

            sg_eps = 1e-5 if scale_a < 0.99 else 1e-7
            tot_e, sq_e, gp, gt, _ = sat_energy_and_grad(
                pos, th, scale_a, L_t, nbr_idx, nbr_mask,
                w_wall=5.0, w_align=w_align_cur, sg_eps=sg_eps,
            )

            # Active escape moves in early/mid stages before jam lock-in (a <= 0.962)
            if scale_a <= 0.962 and it in (n_steps // 4, n_steps // 2):
                # Stress-weighted Langevin shake on mobile half of stage
                mean_e = sq_e.mean(dim=-1, keepdim=True) + 1e-8
                stress = (sq_e / mean_e).clamp(min=0.05, max=3.5).sqrt()
                amp_xy = 0.025 * (1.0 - scale_a) * 4.0
                amp_th = 0.04 * (1.0 - scale_a) * 4.0
                pos.add_(torch.randn_like(pos) * (amp_xy * stress[:, :, None]))
                th.add_(torch.randn_like(th) * (amp_th * stress))
                # Downhill void relocation + orientation alignment
                _void_relocate_and_align(pos, th, sq_e, scale_a, L_t, n_moves=3, n_probes=192)
                _orientation_swap_moves(pos, th, sq_e, scale_a, L_t, n_tries=4)
                nbr_idx, nbr_mask, _ = build_neighbor_list(pos, r_cut=1.85 * scale_a, k_max=k_max)
                tot_e, sq_e, gp, gt, _ = sat_energy_and_grad(
                    pos, th, scale_a, L_t, nbr_idx, nbr_mask,
                    w_wall=5.0, w_align=w_align_cur, sg_eps=sg_eps,
                )

            # Cosine LR decay inside final half of stage
            frac = it / max(1, n_steps - 1)
            lr_mult = 1.0 if frac < 0.5 else 0.5 * (1.0 + math.cos(math.pi * (frac - 0.5) * 2.0))
            lr_xy = lr_base * (0.25 + 0.75 * lr_mult)
            lr_th = lr_xy * (1.3 if scale_a < 0.97 else 0.8)

            # Gradient clipping per trajectory to avoid explosion on overlapping seeds
            gnorm = torch.sqrt((gp * gp).sum(dim=(-1, -2)) + (gt * gt).sum(dim=-1) + 1e-16)
            gscale = torch.clamp(10.0 / gnorm, max=1.0)
            gp = gp * gscale[:, None, None]
            gt = gt * gscale[:, None]

            step_global += 1
            m_p.mul_(beta1).add_(gp, alpha=1.0 - beta1)
            v_p.mul_(beta2).addcmul_(gp, gp, value=1.0 - beta2)
            m_t.mul_(beta1).add_(gt, alpha=1.0 - beta1)
            v_t.mul_(beta2).addcmul_(gt, gt, value=1.0 - beta2)

            bc1 = 1.0 - beta1 ** min(step_global, 50)
            bc2 = 1.0 - beta2 ** min(step_global, 50)
            step_p = (m_p / bc1) / (torch.sqrt(v_p / bc2) + eps_adam)
            step_t = (m_t / bc1) / (torch.sqrt(v_t / bc2) + eps_adam)

            pos.add_(step_p, alpha=-lr_xy)
            th.add_(step_t, alpha=-lr_th)

            # Keep centers roughly inside box
            pos.clamp_(min=0.25 * scale_a)
            pos = torch.minimum(pos, (L_t - 0.25 * scale_a)[:, None, None])

        if verbose and (st_idx % 3 == 0 or st_idx == len(stages) - 1):
            _, _, _, _, mv = sat_energy_and_grad(
                pos, th, scale_a, L_t, nbr_idx, nbr_mask, w_wall=5.0, w_align=0.0, sg_eps=1e-7
            )
            print(
                f"  stage a={scale_a:.4f}: min_E={tot_e.min().item():.3e} "
                f"med_E={tot_e.median().item():.3e} min_max_viol={mv.min().item():.3e}",
                flush=True,
            )

    # Final float64 fine polish on top trajectories at a=1.0 using small-step Adam
    pos64 = pos.to(torch.float64)
    th64 = th.to(torch.float64)
    L64 = L_t.to(torch.float64)
    nbr_idx, nbr_mask, _ = build_neighbor_list(pos64, r_cut=1.75, k_max=k_max)
    tot_e, _, _, _, _ = sat_energy_and_grad(
        pos64, th64, 1.0, L64, nbr_idx, nbr_mask, w_wall=5.0, w_align=0.0, sg_eps=1e-8
    )

    k_top = min(B, 64)
    _, top_idx = torch.topk(tot_e, k=k_top, largest=False)
    pt = pos64[top_idx].clone()
    tt = th64[top_idx].clone()
    Lt = L64[top_idx]
    mpt = torch.zeros_like(pt)
    vpt = torch.zeros_like(pt)
    mtt = torch.zeros_like(tt)
    vtt = torch.zeros_like(tt)
    ni_t, nm_t, _ = build_neighbor_list(pt, r_cut=1.75, k_max=k_max)
    for it in range(250):
        if it > 0 and it % 25 == 0:
            ni_t, nm_t, _ = build_neighbor_list(pt, r_cut=1.75, k_max=k_max)
        _, _, gp, gt, _ = sat_energy_and_grad(
            pt, tt, 1.0, Lt, ni_t, nm_t, w_wall=5.0, w_align=0.0, sg_eps=1e-9
        )
        gnorm = torch.sqrt((gp * gp).sum(dim=(-1, -2)) + (gt * gt).sum(dim=-1) + 1e-24)
        gs = torch.clamp(1.0 / gnorm, max=1.0)
        gp = gp * gs[:, None, None]
        gt = gt * gs[:, None]
        mpt.mul_(0.85).add_(gp, alpha=0.15)
        vpt.mul_(0.99).addcmul_(gp, gp, value=0.01)
        mtt.mul_(0.85).add_(gt, alpha=0.15)
        vtt.mul_(0.99).addcmul_(gt, gt, value=0.01)
        bc1 = 1.0 - 0.85 ** (it + 1)
        bc2 = 1.0 - 0.99 ** (it + 1)
        lr_f = 1.5e-4 * (0.1 + 0.9 * (1.0 - it / 250.0))
        pt.add_((mpt / bc1) / (torch.sqrt(vpt / bc2) + 1e-12), alpha=-lr_f)
        tt.add_((mtt / bc1) / (torch.sqrt(vtt / bc2) + 1e-12), alpha=-lr_f)

    pos64[top_idx] = pt
    th64[top_idx] = tt
    tot_e, _, _, _, _ = sat_energy_and_grad(
        pos64, th64, 1.0, L64, nbr_idx, nbr_mask, w_wall=5.0, w_align=0.0, sg_eps=1e-9
    )

    pos_np = pos64.cpu().numpy()
    th_np = th64.cpu().numpy()
    L_cert, lam, pen_exact = exact_allpairs_lambda_and_L(pos_np, th_np, tau=1e-10)

    return {
        "pos": pos_np,
        "th": th_np,
        "L_cert": L_cert,
        "lam": lam,
        "pen_exact": pen_exact,
        "final_energy": tot_e.cpu().numpy(),
    }
