"""Idea 1: Cross-n interior seam surgery + C++ L-BFGS a-melt + Joint (x, y, dtheta) SLP
on initial targets {912, 738, 902}.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parent
DOCS_DIR = POC_DIR.parent / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import slp_utils as su  # noqa: E402


def get_norm_packing(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sub = df[df["n"] == n]
    x = sub["x"].to_numpy(copy=True)
    y = sub["y"].to_numpy(copy=True)
    th = sub["angle"].to_numpy(copy=True)
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x -= (x - hx).min()
    y -= (y - hx).min()
    L = float(max((x + hx).max(), (y + hx).max()))
    return x, y, th, L


def make_interior_cut_seed(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    target_n: int,
    dk: int,
    cx: float,
    cy: float,
    seam_radius: float = 2.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Cut dk upright rows/cols at (cx, cy) and matched tilted slice so count == target_n."""
    degs = np.degrees(th) % 90.0
    up = np.minimum(degs, 90.0 - degs) < 0.5

    rem_up = up & (((x >= cx) & (x < cx + dk)) | ((y >= cy) & (y < cy + dk)))
    # Try tilted cut widths wt near dk
    best_keep = None
    best_diff = 999999
    best_wt = float(dk)
    for wt in np.linspace(max(0.5, dk - 0.35), dk + 0.65, 41):
        rem_t = (~up) & (((x >= cx) & (x < cx + wt)) | ((y >= cy) & (y < cy + wt)))
        keep = ~(rem_up | rem_t)
        cnt = int(np.sum(keep))
        diff = abs(cnt - target_n)
        if diff < best_diff:
            best_diff = diff
            best_keep = keep
            best_wt = float(wt)
            if diff == 0:
                break

    if best_keep is None or best_diff > 6:
        return None

    keep = best_keep.copy()
    cnt = int(np.sum(keep))
    # Fine-adjust on seam boundary tilted squares if off by <= 6
    if cnt > target_n:
        # Drop tilted squares closest to cut lines (cx+wt/2, cy+wt/2)
        dist = np.minimum(np.abs(x - (cx + 0.5 * best_wt)), np.abs(y - (cy + 0.5 * best_wt)))
        cand = np.where(keep & (~up))[0]
        if len(cand) < cnt - target_n:
            cand = np.where(keep)[0]
        order = cand[np.argsort(dist[cand])]
        keep[order[: cnt - target_n]] = False
    elif cnt < target_n:
        # Restore tilted squares furthest from cut center among removed
        dist = np.minimum(np.abs(x - (cx + 0.5 * best_wt)), np.abs(y - (cy + 0.5 * best_wt)))
        cand = np.where(~keep)[0]
        order = cand[np.argsort(-dist[cand])]
        keep[order[: target_n - cnt]] = True

    if int(np.sum(keep)) != target_n:
        return None

    nx = x[keep].copy()
    ny = y[keep].copy()
    nth = th[keep].copy()
    # Shift upper/right parts by dk so upright core aligns seamlessly
    nx = np.where(nx >= cx + 0.5 * dk, nx - dk, nx)
    ny = np.where(ny >= cy + 0.5 * dk, ny - dk, ny)

    seam_mask = (np.abs(nx - cx) <= seam_radius) | (np.abs(ny - cy) <= seam_radius)
    return nx, ny, nth, seam_mask


def make_interior_insert_seed(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    target_n: int,
    cx: float,
    cy: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Expand donor (k_d -> k_d+1) by +1.0 at (cx, cy), fill upright seam + pocket seam to target_n."""
    nx = np.where(x >= cx, x + 1.0, x)
    ny = np.where(y >= cy, y + 1.0, y)
    nth = th.copy()

    # Fill upright column along x = cx + 0.5 and row along y = cy + 0.5 where empty
    pts_x, pts_y, pts_th = list(nx), list(ny), list(nth)
    L_est = float(max(nx.max(), ny.max())) + 0.5
    # Candidate upright grid points on the new cross
    cands = []
    # Find median upright phase
    degs = np.degrees(th) % 90.0
    up = np.minimum(degs, 90.0 - degs) < 0.5
    px = float(np.median(x[up] % 1.0)) if np.any(up) else 0.5
    py = float(np.median(y[up] % 1.0)) if np.any(up) else 0.5
    x_seam = math.floor(cx) + px + (1.0 if (math.floor(cx) + px) < cx else 0.0)
    y_seam = math.floor(cy) + py + (1.0 if (math.floor(cy) + py) < cy else 0.0)
    for k_i in range(int(round(L_est))):
        cands.append((x_seam, k_i + py, 0.0))
        cands.append((k_i + px, y_seam, 0.0))

    # Keep candidates with largest min distance to existing centers
    arr_xy = np.column_stack([nx, ny])
    scored = []
    for gx, gy, gt in cands:
        if gx < 0.5 or gx > L_est - 0.5 or gy < 0.5 or gy > L_est - 0.5:
            continue
        d2 = np.min((arr_xy[:, 0] - gx) ** 2 + (arr_xy[:, 1] - gy) ** 2)
        scored.append((d2, gx, gy, gt))
    scored.sort(reverse=True)
    need = target_n - len(nx)
    if need < 0 or len(scored) < need:
        return None
    for idx in range(need):
        _, gx, gy, gt = scored[idx]
        pts_x.append(gx)
        pts_y.append(gy)
        pts_th.append(gt + rng.normal(0.0, 0.05))

    rx = np.array(pts_x, dtype=np.float64)
    ry = np.array(pts_y, dtype=np.float64)
    rth = np.array(pts_th, dtype=np.float64)
    seam_mask = (np.abs(rx - cx - 0.5) <= 2.6) | (np.abs(ry - cy - 0.5) <= 2.6)
    return rx, ry, rth, seam_mask


def build_seeds_912(df: pd.DataFrame) -> list[dict]:
    seeds = []
    rng = np.random.default_rng(101)
    # 1. Direct + local corner-junction ruin on 912 itself
    x0, y0, th0, L0 = get_norm_packing(df, 912)
    seeds.append({
        "name": "912_direct",
        "x": x0, "y": y0, "th": th0, "seam": np.ones(912, dtype=bool),
        "L_sched": [L0, L0 - 0.005, L0 - 0.012],
        "a_start": 0.999, "n_melt": 1, "seam_iters": 0, "full_iters": 240, "ruin": 6,
    })
    for j in range(6):
        nx, ny, nth = x0.copy(), y0.copy(), th0.copy()
        # Perturb top-right pocket-seam junction [21..30.7]x[21..30.7] or whole pocket boundary
        if j < 4:
            mask = (nx > 20.5) & (ny > 20.5)
        else:
            degs = np.degrees(nth) % 90.0
            mask = np.minimum(degs, 90.0 - degs) >= 0.5
        nx[mask] += rng.normal(0.0, 0.08, size=int(np.sum(mask)))
        ny[mask] += rng.normal(0.0, 0.08, size=int(np.sum(mask)))
        nth[mask] += rng.normal(0.0, 0.06, size=int(np.sum(mask)))
        seeds.append({
            "name": f"912_junc_{j}",
            "x": nx, "y": ny, "th": nth, "seam": mask,
            "L_sched": [L0 + 0.01, L0 - 0.004, L0 - 0.012],
            "a_start": 0.93, "n_melt": 4, "seam_iters": 180, "full_iters": 240, "ruin": 10,
        })

    # 2. 973 -> 912 interior seam cuts (L=31.6966 -> 30.6966!)
    x973, y973, th973, L973 = get_norm_packing(df, 973)
    for idx, (cx, cy) in enumerate([(10.0, 10.0), (12.0, 12.0), (14.0, 14.0), (16.0, 16.0), (18.0, 18.0), (13.0, 17.0), (17.0, 13.0)]):
        res = make_interior_cut_seed(x973, y973, th973, 912, 1, cx, cy, seam_radius=2.8)
        if res is not None:
            nx, ny, nth, sm = res
            seeds.append({
                "name": f"973_to_912_cut{idx}",
                "x": nx, "y": ny, "th": nth, "seam": sm,
                "L_sched": [L973 - 0.96, L973 - 0.99, L973 - 1.00],
                "a_start": 0.89, "n_melt": 5, "seam_iters": 220, "full_iters": 260, "ruin": 12,
            })

    # 3. 972 -> 912 interior seam cuts (L=31.6840 -> 30.6840!)
    x972, y972, th972, L972 = get_norm_packing(df, 972)
    for idx, (cx, cy) in enumerate([(11.0, 11.0), (14.0, 14.0), (16.0, 16.0), (18.0, 18.0)]):
        res = make_interior_cut_seed(x972, y972, th972, 912, 1, cx, cy, seam_radius=2.8)
        if res is not None:
            nx, ny, nth, sm = res
            seeds.append({
                "name": f"972_to_912_cut{idx}",
                "x": nx, "y": ny, "th": nth, "seam": sm,
                "L_sched": [L972 - 0.95, L972 - 0.985, L972 - 1.00],
                "a_start": 0.89, "n_melt": 5, "seam_iters": 220, "full_iters": 260, "ruin": 12,
            })

    # 4. 852 -> 912 interior insert (L=29.6996 -> 30.6996!)
    x852, y852, th852, L852 = get_norm_packing(df, 852)
    for idx, (cx, cy) in enumerate([(12.0, 12.0), (15.0, 15.0)]):
        res = make_interior_insert_seed(x852, y852, th852, 912, cx, cy, rng)
        if res is not None:
            nx, ny, nth, sm = res
            seeds.append({
                "name": f"852_to_912_ins{idx}",
                "x": nx, "y": ny, "th": nth, "seam": sm,
                "L_sched": [L852 + 1.02, L852 + 1.002, L852 + 0.996],
                "a_start": 0.90, "n_melt": 5, "seam_iters": 220, "full_iters": 260, "ruin": 10,
            })
    return seeds


def build_seeds_738(df: pd.DataFrame) -> list[dict]:
    seeds = []
    rng = np.random.default_rng(202)
    x0, y0, th0, L0 = get_norm_packing(df, 738)
    seeds.append({
        "name": "738_direct",
        "x": x0, "y": y0, "th": th0, "seam": np.ones(738, dtype=bool),
        "L_sched": [L0, L0 - 0.005, L0 - 0.011],
        "a_start": 0.999, "n_melt": 1, "seam_iters": 0, "full_iters": 240, "ruin": 6,
    })
    for j in range(5):
        nx, ny, nth = x0.copy(), y0.copy(), th0.copy()
        if j < 3:
            # Defect regions: top-left (x<10, y>22) or bottom-right (x>20, y<10) or bend (x<10, y<10)
            mask = ((nx < 9.0) & (ny > 21.0)) | ((nx > 20.0) & (ny < 9.0)) | ((nx < 10.0) & (ny < 10.0))
        else:
            degs = np.degrees(nth) % 90.0
            mask = np.minimum(degs, 90.0 - degs) >= 0.5
        nx[mask] += rng.normal(0.0, 0.08, size=int(np.sum(mask)))
        ny[mask] += rng.normal(0.0, 0.08, size=int(np.sum(mask)))
        nth[mask] += rng.normal(0.0, 0.06, size=int(np.sum(mask)))
        seeds.append({
            "name": f"738_defect_{j}",
            "x": nx, "y": ny, "th": nth, "seam": mask,
            "L_sched": [L0 + 0.01, L0 - 0.003, L0 - 0.010],
            "a_start": 0.93, "n_melt": 4, "seam_iters": 180, "full_iters": 240, "ruin": 10,
        })

    # 971 -> 738 (4-unit interior cut, L=31.6643 -> 27.6643!)
    x971, y971, th971, L971 = get_norm_packing(df, 971)
    for idx, (cx, cy) in enumerate([(11.0, 11.0), (12.0, 13.0), (13.0, 15.0), (15.0, 13.0), (14.0, 14.0), (16.0, 16.0)]):
        res = make_interior_cut_seed(x971, y971, th971, 738, 4, cx, cy, seam_radius=3.0)
        if res is not None:
            nx, ny, nth, sm = res
            seeds.append({
                "name": f"971_to_738_cut{idx}",
                "x": nx, "y": ny, "th": nth, "seam": sm,
                "L_sched": [L971 - 3.96, L971 - 3.99, L971 - 4.00],
                "a_start": 0.88, "n_melt": 5, "seam_iters": 240, "full_iters": 260, "ruin": 12,
            })

    # 793 -> 738 (1-unit interior cut, L=28.6772 -> 27.6772!)
    x793, y793, th793, L793 = get_norm_packing(df, 793)
    for idx, (cx, cy) in enumerate([(11.0, 11.0), (14.0, 14.0), (16.0, 16.0), (18.0, 18.0)]):
        res = make_interior_cut_seed(x793, y793, th793, 738, 1, cx, cy, seam_radius=2.8)
        if res is not None:
            nx, ny, nth, sm = res
            seeds.append({
                "name": f"793_to_738_cut{idx}",
                "x": nx, "y": ny, "th": nth, "seam": sm,
                "L_sched": [L793 - 0.97, L793 - 0.995, L793 - 1.00],
                "a_start": 0.90, "n_melt": 5, "seam_iters": 220, "full_iters": 260, "ruin": 12,
            })

    # 737 -> 738 (+1 insertion into pocket gap)
    x737, y737, th737, L737 = get_norm_packing(df, 737)
    for idx, (px, py, ang) in enumerate([(16.2, 16.2, 24.0), (0.8, 15.8, 66.0), (26.5, 16.2, 24.0)]):
        nx = np.append(x737, px)
        ny = np.append(y737, py)
        nth = np.append(th737, math.radians(ang))
        sm = ((nx - px) ** 2 + (ny - py) ** 2) <= 12.0
        seeds.append({
            "name": f"737_plus1_{idx}",
            "x": nx, "y": ny, "th": nth, "seam": sm,
            "L_sched": [L737 + 0.055, L737 + 0.040, L737 + 0.032],
            "a_start": 0.89, "n_melt": 5, "seam_iters": 220, "full_iters": 260, "ruin": 12,
        })
    return seeds


def make_c2_twin_pocket_seed(k: int, w: int, m_len: int, deg: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct C2-symmetric twin corner pockets (each w x m_len upright replaced by w*m_len+1 tilted)."""
    L = float(k) + 0.54
    hc = 0.5 * L
    th_r = math.radians(deg)
    # Build k x k upright grid centered in [0, L]^2, remove BL (0..w-1, 0..m_len-1) and TR (k-w..k-1, k-m_len..k-1)
    # Wait: total squares = k^2 + 2 -> remove w*m_len from each corner, insert (w*m_len + 1) tilted in each corner!
    # Organize in C2 pairs so sym_mode=2 works directly!
    pairs_a = []
    pairs_b = []
    # Upright grid points (i, j) for i in 0..k-1, j in 0..k-1
    used = np.zeros((k, k), dtype=bool)
    used[:w, :m_len] = True
    used[k - w :, k - m_len :] = True
    for i in range(k):
        for j in range(k):
            if used[i, j]:
                continue
            ci, cj = k - 1 - i, k - 1 - j
            if (i, j) < (ci, cj):
                # Shift upright squares slightly outward toward corners
                ox = 0.5 + i + (0.54 if i >= k // 2 else 0.0)
                oy = 0.5 + j + (0.54 if j >= k // 2 else 0.0)
                pairs_a.append((ox, oy, 0.0))
                pairs_b.append((L - ox, L - oy, math.pi))
    # Tilted points inside [0, w+0.54] x [0, m_len+0.54] (and C2 image)
    cnt_p = w * m_len + 1
    c, s = math.cos(th_r), math.sin(th_r)
    pts = []
    for u in range(-m_len - 2, w + m_len + 3):
        for v in range(-m_len - 2, w + m_len + 3):
            px = 0.55 + u * c - v * s
            py = 0.55 + u * s + v * c
            if 0.35 <= px <= w + 0.35 and 0.35 <= py <= m_len + 0.35:
                pts.append((px, py))
    # Pad or trim to cnt_p
    pts.sort(key=lambda p: (p[0] + p[1]))
    while len(pts) < cnt_p:
        pts.append((rng.uniform(0.6, w + 0.1), rng.uniform(0.6, m_len + 0.1)))
    pts = pts[:cnt_p]
    for px, py in pts:
        pairs_a.append((px, py, th_r))
        pairs_b.append((L - px, L - py, th_r + math.pi))

    # Interleave pairs_a, pairs_b, plus center square if k is odd (here k=30 is even -> 0 center)
    n_tot = 2 * len(pairs_a)
    if (k * k) % 2 == 1:
        # center at (k//2, k//2)
        n_tot += 1
    rx = np.zeros(n_tot, dtype=np.float64)
    ry = np.zeros(n_tot, dtype=np.float64)
    rth = np.zeros(n_tot, dtype=np.float64)
    for idx, ((ax, ay, at), (bx, by, bt)) in enumerate(zip(pairs_a, pairs_b)):
        rx[2 * idx], ry[2 * idx], rth[2 * idx] = ax, ay, at
        rx[2 * idx + 1], ry[2 * idx + 1], rth[2 * idx + 1] = bx, by, bt
    if (k * k) % 2 == 1:
        rx[-1], ry[-1], rth[-1] = hc, hc, 0.0
    return rx, ry, rth


def build_seeds_902(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    free_seeds = []
    c2_seeds = []
    rng = np.random.default_rng(303)

    # 1. 901 -> 902 (+1 insertion at boundary/45-deg corners)
    x901, y901, th901, L901 = get_norm_packing(df, 901)
    # Candidate insertion spots near the tilted outlier (25.997, 12.564) and corners
    for idx, (px, py, deg) in enumerate([
        (25.95, 13.55, 15.0),
        (25.55, 25.55, 45.0),
        (12.50, 25.95, 25.0),
        (0.55, 15.0, 25.0),
    ]):
        nx = np.append(x901, px)
        ny = np.append(y901, py)
        nth = np.append(th901, math.radians(deg))
        sm = ((nx - px) ** 2 + (ny - py) ** 2) <= 20.0
        free_seeds.append({
            "name": f"901_plus1_{idx}",
            "x": nx, "y": ny, "th": nth, "seam": sm,
            "L_sched": [L901 + 0.032, L901 + 0.020, L901 + 0.013],
            "a_start": 0.90, "n_melt": 5, "seam_iters": 200, "full_iters": 240, "ruin": 12,
        })

    # 2. 904 -> 902 (-2 squares from w=3 33.56° band)
    x904, y904, th904, L904 = get_norm_packing(df, 904)
    degs4 = np.degrees(th904) % 90.0
    tilted4 = np.where(np.minimum(degs4, 90.0 - degs4) >= 0.5)[0]
    # Remove 2 squares from ends of the tilted band or outlier
    for idx, mode in enumerate(["ends", "mid", "outlier"]):
        keep = np.ones(904, dtype=bool)
        if mode == "ends":
            order = tilted4[np.argsort(x904[tilted4] + y904[tilted4])]
            keep[order[0]] = False
            keep[order[-1]] = False
        elif mode == "mid":
            order = tilted4[np.argsort(np.abs(x904[tilted4] - 15.0) + np.abs(y904[tilted4] - 15.0))]
            keep[order[0]] = False
            keep[order[1]] = False
        else:
            keep[tilted4[0]] = False
            keep[tilted4[-1]] = False
        nx, ny, nth = x904[keep], y904[keep], th904[keep]
        sm = np.minimum(np.degrees(nth) % 90.0, 90.0 - (np.degrees(nth) % 90.0)) >= 0.5
        free_seeds.append({
            "name": f"904_minus2_{mode}",
            "x": nx, "y": ny, "th": nth, "seam": sm,
            "L_sched": [L904 - 0.005, L904 - 0.020, L904 - 0.030],
            "a_start": 0.90, "n_melt": 5, "seam_iters": 200, "full_iters": 260, "ruin": 12,
        })

    # 3. C2 twin corner pockets in sym_mode=2
    for w, m_len, deg in [(5, 11, 24.5), (6, 13, 25.0), (7, 15, 24.5), (4, 15, 22.0), (8, 12, 26.5), (6, 10, 33.0)]:
        rx, ry, rth = make_c2_twin_pocket_seed(30, w, m_len, deg, rng)
        sm = np.minimum(np.degrees(rth) % 90.0, 90.0 - (np.degrees(rth) % 90.0)) >= 0.5
        c2_seeds.append({
            "name": f"902_c2_w{w}_m{m_len}_{deg:.1f}",
            "x": rx, "y": ry, "th": rth, "seam": sm,
            "L_sched": [30.580, 30.550, 30.533],
            "a_start": 0.88, "n_melt": 5, "seam_iters": 220, "full_iters": 260, "ruin": 14,
        })

    return free_seeds, c2_seeds


def polish_candidate(args: tuple) -> dict:
    """Worker: run joint_xytheta_slp + verify_official on one C++-melted candidate."""
    n, name, x, y, th, pen_cpp, L_cpp = args
    t0 = time.time()
    bx, by, bth, bL, bok = su.joint_xytheta_slp(x, y, th, rounds=16, tr_th0=0.025, tr_xy0=0.12)
    dt = time.time() - t0
    if not bok:
        return {"n": n, "name": name, "ok": False, "L_cpp": L_cpp, "pen_cpp": pen_cpp, "L_slp": None, "dt": dt}
    v = su.verify_official(n, bx, by, bth)
    return {
        "n": n,
        "name": name,
        "ok": v["ok"],
        "L_cpp": L_cpp,
        "pen_cpp": pen_cpp,
        "L_slp": v.get("L_float"),
        "L_lit": v.get("L_lit"),
        "L_pandas": v.get("L_pandas"),
        "dt": dt,
        "x": bx if v["ok"] else None,
        "y": by if v["ok"] else None,
        "th": bth if v["ok"] else None,
    }


def run_group_and_polish(
    n: int,
    seeds: list[dict],
    sym_mode: int,
    inc_L: float,
    front_L: float,
    out_records: list[dict],
) -> None:
    if not seeds:
        return
    print(f"[n={n} sym={sym_mode}] Running C++ batch_melt_solve on {len(seeds)} seeds (4 threads)...", flush=True)
    t0 = time.time()
    # Group by (a_start, n_melt, seam_iters, full_iters, ruin, len(L_sched)) or run uniformly per seed config
    # To keep batch calls fast and clean, run per-seed or grouped by identical schedule params
    cpp_res = []
    for s in seeds:
        res_b = su.cpp_batch_melt(
            [(s["x"], s["y"], s["th"])],
            [s["L_sched"]],
            sym_mode=sym_mode,
            a_start=s["a_start"],
            n_melt_stages=s["n_melt"],
            seam_masks=[s["seam"]],
            seam_iters=s["seam_iters"],
            full_iters=s["full_iters"],
            n_ruin_iters=s["ruin"],
            n_threads=1,
            base_seed=abs(hash(s["name"])) % 1000000,
        )[0]
        cpp_res.append((s["name"], res_b))
    print(f"[n={n} sym={sym_mode}] C++ melt done in {time.time()-t0:.1f}s.", flush=True)

    # Sort by (pen < 1e-7, pen, L_bbox) and polish top 10 (or all with pen < 0.05) in parallel (max_workers=4)
    cpp_res.sort(key=lambda item: (0 if item[1]["pen"] < 1e-6 else 1, item[1]["pen"], item[1]["L_bbox"]))
    to_polish = []
    for name, r in cpp_res[:12]:
        print(f"  cpp {name}: pen={r['pen']:.3e} L_bbox={r['L_bbox']:.6f} feas={r['best_L_feas']}", flush=True)
        if r["pen"] < 0.08:
            to_polish.append((n, name, r["x"], r["y"], r["th"], r["pen"], r["L_bbox"]))

    if not to_polish:
        out_records.append({"n": n, "sym": sym_mode, "note": "all pen >= 0.08", "best_pen": cpp_res[0][1]["pen"]})
        return

    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(polish_candidate, task) for task in to_polish]
        for fut in as_completed(futs):
            r = fut.result()
            rec = {k: v for k, v in r.items() if k not in ("x", "y", "th")}
            rec["inc_L"] = inc_L
            rec["front_L"] = front_L
            out_records.append(rec)
            print(
                f"  slp n={n} {r['name']}: ok={r['ok']} L_slp={r['L_slp']} (inc={inc_L:.6f}, front={front_L:.6f}) dt={r['dt']:.1f}s",
                flush=True,
            )
            if r["ok"] and r["L_slp"] is not None and r["L_slp"] < inc_L - 1e-12:
                win_info = su.save_win_to_final_csv(n, r["x"], r["y"], r["th"])
                print(f"  *** CERTIFIED WIN CHECK n={n} {r['name']}: {win_info} ***", flush=True)
                rec["win_info"] = win_info


def main() -> None:
    t_start = time.time()
    targets = [912, 738, 902]
    df, info = su.load_merged_and_gap(targets)
    print("Targets:", info, flush=True)

    records = []
    s912 = build_seeds_912(df)
    run_group_and_polish(912, s912, 1, info[912]["L_ours"], info[912]["L_front"], records)

    s738 = build_seeds_738(df)
    run_group_and_polish(738, s738, 1, info[738]["L_ours"], info[738]["L_front"], records)

    s902_free, s902_c2 = build_seeds_902(df)
    run_group_and_polish(902, s902_free, 1, info[902]["L_ours"], info[902]["L_front"], records)
    run_group_and_polish(902, s902_c2, 2, info[902]["L_ours"], info[902]["L_front"], records)

    out_path = DOCS_DIR / "idea1_results.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"Idea 1 finished in {time.time() - t_start:.1f}s. Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
