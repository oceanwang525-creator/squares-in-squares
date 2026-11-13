"""Idea 2: surplus_peel + phase-exact tilted-MILP + 5-mode inc_melt + soft C++ a-melt + joint (x,y,dtheta) SLP."""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import ctypes
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

THIS_DIR = Path(__file__).resolve().parent
SWARM_DIR = THIS_DIR.parent
REPO_ROOT = SWARM_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import utils as u  # noqa: E402

CPP_SRC = THIS_DIR / "cpp_melt.cpp"
SO_PATH = Path("/tmp/libreplace45_melt.so")
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
GAP_CSV = REPO_ROOT / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv"
DATA_DIR = SWARM_DIR / "data"
DOCS_DIR = SWARM_DIR / "docs"


def ensure_cpp_lib() -> ctypes.CDLL:
    """Compile cpp_melt.cpp to /tmp/libreplace45_melt.so if needed and return loaded CDLL."""
    if not SO_PATH.exists() or CPP_SRC.stat().st_mtime > SO_PATH.stat().st_mtime:
        subprocess.run(
            ["g++", "-O3", "-march=native", "-shared", "-fPIC", "-std=c++17", str(CPP_SRC), "-o", str(SO_PATH)],
            check=True,
        )
    lib = ctypes.CDLL(str(SO_PATH))
    lib.batch_melt_solve.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint64,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    lib.batch_melt_solve.restype = None
    return lib


def load_merged_mats() -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]]:
    """Load merged.csv into shifted (x, y, th, L) per n."""
    df = pd.read_csv(MERGED_CSV)
    mats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for n_val, g in df.groupby("n"):
        x = g["x"].to_numpy(dtype=np.float64, copy=True)
        y = g["y"].to_numpy(dtype=np.float64, copy=True)
        th = g["angle"].to_numpy(dtype=np.float64, copy=True)
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        x -= float((x - hx).min())
        y -= float((y - hx).min())
        L = float(max((x + hx).max(), (y + hx).max()))
        mats[int(n_val)] = (x, y, th, L)
    return df, mats


def load_front_map() -> dict[int, float]:
    """Read per_n_gap.csv target front L if available."""
    if not GAP_CSV.exists():
        return {}
    gdf = pd.read_csv(GAP_CSV)
    out: dict[int, float] = {}
    for r in gdf.itertuples():
        out[int(r.n)] = float(r.L_frontier)
    return out


def donor_tilt(th: np.ndarray) -> float:
    deg = np.degrees(th) % 90.0
    t = deg[(deg > 1.0) & (deg < 89.0)]
    return math.radians(float(np.median(t))) if len(t) else math.pi / 4.0


def clearance_points(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, n_pts: int = 8, step: float = 0.14
) -> list[tuple[float, float, float]]:
    """Find separated interior grid centers maximizing minimum center clearance."""
    xs = np.arange(0.58, max(0.65, L - 0.55), step)
    ys = np.arange(0.58, max(0.65, L - 0.55), step)
    if len(xs) == 0 or len(ys) == 0:
        return [(0.5 * L, 0.5 * L, 0.0)]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    px = gx.ravel()
    py = gy.ravel()
    # Chunked distance min to avoid large temp arrays
    min_d2 = np.full(len(px), 1e18, dtype=np.float64)
    for start in range(0, len(px), 2048):
        end = min(len(px), start + 2048)
        dx = px[start:end, None] - x[None, :]
        dy = py[start:end, None] - y[None, :]
        min_d2[start:end] = np.min(dx * dx + dy * dy, axis=1)
    order = np.argsort(-min_d2)
    out: list[tuple[float, float, float]] = []
    for i in order:
        if len(out) >= n_pts:
            break
        if all((px[i] - q[0]) ** 2 + (py[i] - q[1]) ** 2 > 0.85**2 for q in out):
            out.append((float(px[i]), float(py[i]), float(math.sqrt(min_d2[i]))))
    return out or [(0.5 * L, 0.5 * L, 0.0)]


def _peel_mask_split(x: np.ndarray, y: np.ndarray, L: float, cL: int, cR: int, cB: int, cT: int) -> np.ndarray:
    keep = np.ones(len(x), dtype=bool)
    if cL > 0:
        keep &= x >= cL + 0.05
    if cR > 0:
        keep &= x <= L - cR - 0.05
    if cB > 0:
        keep &= y >= cB + 0.05
    if cT > 0:
        keep &= y <= L - cT - 0.05
    return keep


def collect_surplus_peels(n: int, L_inc: float, mats: dict[int, tuple]) -> list[dict]:
    """Collect center-peel donors (rem in [n, n+6]) across wall splits and 5x5 interior cross-slices."""
    k = math.isqrt(n)
    cands: list[dict] = []
    seen = set()

    for n2 in range(n + 1, min(1000, n + 5 * k + 22) + 1):
        if n2 not in mats:
            continue
        x, y, th, L = mats[n2]
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        for c in (1, 2, 3):
            L_box = L - c
            if not (L_inc - 0.16 <= L_box <= L_inc - 0.0004):
                continue
            for cL in range(c + 1):
                cR = c - cL
                for cB in range(c + 1):
                    cT = c - cB
                    keep = _peel_mask_split(x, y, L, cL, cR, cB, cT)
                    nr = int(keep.sum())
                    if not (n <= nr <= n + 6):
                        continue
                    xs = x[keep] - cL
                    ys = y[keep] - cB
                    ts = th[keep]
                    h2 = hx[keep]
                    oh = np.maximum.reduce([
                        np.zeros(nr), h2 - xs, xs + h2 - L_box, h2 - ys, ys + h2 - L_box,
                    ])
                    surp = nr - n
                    max_oh = float(oh.max())
                    max_oh_rem = float(np.partition(oh, -(surp + 1))[-(surp + 1)]) if surp > 0 else max_oh
                    if max_oh_rem >= 0.36:
                        continue
                    key = (n2, nr, round(max_oh_rem, 4), round(L_box, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    cands.append({
                        "src": f"s{n2}_{cL}{cR}_{cB}{cT}",
                        "x": xs, "y": ys, "th": ts, "oh": oh,
                        "L": L_box, "surplus": surp, "max_oh": max_oh, "max_oh_rem": max_oh_rem,
                    })

        if n2 <= n + 3 * k:
            L_box = L - 1.0
            if L_inc - 0.08 <= L_box <= L_inc - 0.0004:
                for xc in np.linspace(2.5, L - 2.5, 5):
                    kx = (x < xc - 0.5) | (x > xc + 0.5)
                    for yc in np.linspace(2.5, L - 2.5, 5):
                        keep = kx & ((y < yc - 0.5) | (y > yc + 0.5))
                        nr = int(keep.sum())
                        if not (n <= nr <= n + 4):
                            continue
                        xs = np.where(x[keep] > xc + 0.5, x[keep] - 1.0, x[keep])
                        ys = np.where(y[keep] > yc + 0.5, y[keep] - 1.0, y[keep])
                        ts = th[keep]
                        h2 = hx[keep]
                        seam_d = np.minimum(np.abs(xs - (xc - 0.5)), np.abs(ys - (yc - 0.5)))
                        oh = np.maximum(0.0, 0.75 - seam_d) + np.maximum.reduce([
                            np.zeros(nr), h2 - xs, xs + h2 - L_box, h2 - ys, ys + h2 - L_box,
                        ])
                        surp = nr - n
                        max_oh = float(oh.max())
                        max_oh_rem = float(np.partition(oh, -(surp + 1))[-(surp + 1)]) if surp > 0 else max_oh
                        if max_oh_rem < 0.60:
                            key = (n2, nr, round(L_box, 6), round(max_oh_rem, 3))
                            if key not in seen:
                                seen.add(key)
                                cands.append({
                                    "src": f"x{n2}_{int(xc)}_{int(yc)}",
                                    "x": xs, "y": ys, "th": ts, "oh": oh,
                                    "L": L_box, "surplus": surp, "max_oh": max_oh, "max_oh_rem": max_oh_rem,
                                })

    cands.sort(key=lambda r: (r["max_oh_rem"], r["surplus"], r["L"]))
    return cands


def _surplus_drop_seeds(don: dict, n: int, L0: float, n_seeds: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Drop `surplus` worst overhang/seam squares, clamp into L0, and jitter seam neighborhoods."""
    x0, y0, th0, oh0 = don["x"], don["y"], don["th"], don["oh"]
    L_box = don["L"]
    surplus = don["surplus"]
    hx0 = 0.5 * (np.abs(np.cos(th0)) + np.abs(np.sin(th0)))

    rank = np.argsort(-oh0)
    pool_idx = rank[: max(surplus + 5, min(len(rank), 14))]
    seeds = []
    sc = L0 / L_box
    for s in range(n_seeds):
        if surplus == 0:
            keep = np.ones(len(x0), dtype=bool)
        elif s == 0:
            drop = rank[:surplus]
            keep = np.ones(len(x0), dtype=bool)
            keep[drop] = False
        else:
            drop = rng.choice(pool_idx, size=surplus, replace=False)
            keep = np.ones(len(x0), dtype=bool)
            keep[drop] = False

        x = x0[keep] * sc
        y = y0[keep] * sc
        th = th0[keep].copy()
        hx = hx0[keep]
        oh = oh0[keep]
        assert len(x) == n

        x = np.clip(x, hx + 0.005, L0 - hx - 0.005)
        y = np.clip(y, hx + 0.005, L0 - hx - 0.005)
        hot = np.where(oh > 0.01)[0]
        if len(hot) > 0 and s > 0:
            cx, cy = float(x[hot[s % len(hot)]]), float(y[hot[s % len(hot)]])
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= (4.0 if s % 2 == 1 else 6.0) ** 2
            nm = int(near.sum())
            x[near] = np.clip(x[near] + rng.uniform(-0.12, 0.12, nm), hx[near] + 0.005, L0 - hx[near] - 0.005)
            y[near] = np.clip(y[near] + rng.uniform(-0.12, 0.12, nm), hx[near] + 0.005, L0 - hx[near] - 0.005)
            th[near] += rng.uniform(-0.09, 0.09, nm)
        seeds.append((x, y, th))
    return seeds


def _extract_tilted_components(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> list[np.ndarray]:
    """Return connected clusters of non-0° squares (tilted gadgets)."""
    deg = np.degrees(th) % 90.0
    t_idx = np.where((deg > 1.0) & (deg < 89.0))[0]
    if len(t_idx) == 0:
        return []
    # Check 45° dominance
    n45 = int(np.sum(np.abs(deg[t_idx] - 45.0) < 0.25))
    if n45 > 0.5 * len(t_idx):
        return []
    xt, yt = x[t_idx], y[t_idx]
    m = len(t_idx)
    vis = np.zeros(m, dtype=bool)
    comps = []
    for i in range(m):
        if vis[i]:
            continue
        stack = [i]
        vis[i] = True
        comp = []
        while stack:
            u_i = stack.pop()
            comp.append(int(t_idx[u_i]))
            d2 = (xt - xt[u_i]) ** 2 + (yt - yt[u_i]) ** 2
            for v_i in np.where((~vis) & (d2 <= 1.9**2))[0]:
                vis[v_i] = True
                stack.append(int(v_i))
        comps.append(np.array(comp, dtype=np.int32))
    comps.sort(key=len, reverse=True)
    return comps


def tilted_milp_seeds(
    n: int, L_inc: float, L_tgt: float, L0: float, mats: dict[int, tuple], rng: np.random.Generator, n_seeds: int = 4
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Place exact non-45 tilted gadgets from clean donors + pack upright lattice via HiGHS MILP."""
    k = math.isqrt(n)
    donor_pool = []
    for n2, (x2, y2, th2, L2) in mats.items():
        if abs(n2 - n) > 4 * k + 15:
            continue
        f2 = L2 - math.isqrt(n2)
        if f2 >= (L_inc - k) - 0.003:
            continue
        comps = _extract_tilted_components(x2, y2, th2, L2)
        if comps:
            donor_pool.append((f2, abs(n2 - n), n2, x2, y2, th2, L2, comps))
    donor_pool.sort(key=lambda r: (r[0], r[1]))

    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _, _, n2, x2, y2, th2, L2, comps in donor_pool[:4]:
        if len(seeds) >= n_seeds:
            break
        # Keep top 1 or 2 largest components shifted into [0, L_tgt]
        take = np.concatenate(comps[: min(2, len(comps))])
        xt = x2[take].copy()
        yt = y2[take].copy()
        tt = th2[take].copy()
        hxt = 0.5 * (np.abs(np.cos(tt)) + np.abs(np.sin(tt)))
        # Shift each component toward nearest wall inside L_tgt
        for comp in comps[: min(2, len(comps))]:
            xc = x2[comp]
            yc = y2[comp]
            hc = 0.5 * (np.abs(np.cos(th2[comp])) + np.abs(np.sin(th2[comp])))
            minx, maxx = float((xc - hc).min()), float((xc + hc).max())
            miny, maxy = float((yc - hc).min()), float((yc + hc).max())
            dx = 0.0 if minx < L2 - maxx else (L_tgt - L2)
            dy = 0.0 if miny < L2 - maxy else (L_tgt - L2)
            mask = np.isin(take, comp)
            xt[mask] += dx
            yt[mask] += dy
        # Ensure inside [0, L_tgt]
        xt = np.clip(xt, hxt + 1e-4, L_tgt - hxt - 1e-4)
        yt = np.clip(yt, hxt + 1e-4, L_tgt - hxt - 1e-4)

        need = n - len(xt)
        if need <= 0:
            continue
        # Candidate upright 1x1 sites on integer phases inferred from donor + shifted phases
        deg2 = np.degrees(th2) % 90.0
        u_mask = (deg2 < 0.5) | (deg2 > 89.5)
        ux_off = float(np.median((x2[u_mask] - 0.5) % 1.0)) if np.any(u_mask) else 0.0
        uy_off = float(np.median((y2[u_mask] - 0.5) % 1.0)) if np.any(u_mask) else 0.0
        px_opts = sorted({0.5, L_tgt - math.floor(L_tgt) + 0.5, 0.5 + ux_off, (L_tgt % 1.0) + 0.5})
        py_opts = sorted({0.5, L_tgt - math.floor(L_tgt) + 0.5, 0.5 + uy_off, (L_tgt % 1.0) + 0.5})
        sites = []
        for ox in px_opts:
            for ix in range(int(L_tgt) + 1):
                cx = ox + ix
                if 0.5 - 1e-6 <= cx <= L_tgt - 0.5 + 1e-6:
                    for oy in py_opts:
                        for iy in range(int(L_tgt) + 1):
                            cy = oy + iy
                            if 0.5 - 1e-6 <= cy <= L_tgt - 0.5 + 1e-6:
                                sites.append((cx, cy))
        if not sites:
            continue
        s_arr = np.array(sites, dtype=np.float64)
        # Filter out sites overlapping tilted gadgets (`sat_max_overlap` exact for upright vs tilted)
        keep_s = np.ones(len(s_arr), dtype=bool)
        for tx, ty, tth in zip(xt, yt, tt):
            # Bounding box quick reject
            dx = np.abs(s_arr[:, 0] - tx)
            dy = np.abs(s_arr[:, 1] - ty)
            near = (dx < 1.25) & (dy < 1.25)
            for idx_s in np.where(near)[0]:
                pen, _ = u.sat_max_overlap_and_bbox(
                    np.array([tx, s_arr[idx_s, 0]]),
                    np.array([ty, s_arr[idx_s, 1]]),
                    np.array([tth, 0.0]),
                )
                if pen > 1e-5:
                    keep_s[idx_s] = False
        s_arr = s_arr[keep_s]
        if len(s_arr) == 0:
            continue

        # Conflict graph between candidate upright sites (Chebyshev < 1 - 1e-5)
        rows_c, cols_c = [], []
        r_idx = 0
        # Sort for grid-sweeping efficiency
        for i in range(len(s_arr)):
            dx = np.abs(s_arr[i + 1 :, 0] - s_arr[i, 0])
            dy = np.abs(s_arr[i + 1 :, 1] - s_arr[i, 1])
            conf = np.where((dx < 1.0 - 1e-5) & (dy < 1.0 - 1e-5))[0]
            for off in conf:
                j = i + 1 + int(off)
                rows_c.extend([r_idx, r_idx])
                cols_c.extend([i, j])
                r_idx += 1

        # Objective: maximize count + slight corner-preference weight
        m_s = len(s_arr)
        c_obj = -np.ones(m_s, dtype=np.float64)
        if r_idx > 0:
            A_c = csc_matrix((np.ones(len(rows_c)), (rows_c, cols_c)), shape=(r_idx, m_s))
            con = [
                LinearConstraint(A_c, -np.inf, np.ones(r_idx)),
                LinearConstraint(np.ones((1, m_s)), -np.inf, np.array([float(need)])),
            ]
        else:
            con = [LinearConstraint(np.ones((1, m_s)), -np.inf, np.array([float(need)]))]
        res = milp(
            c=c_obj,
            integrality=np.ones(m_s),
            bounds=Bounds(0, 1),
            constraints=con,
            options={"time_limit": 2.0},
        )
        if res.x is None:
            continue
        chosen = s_arr[res.x > 0.5]
        ux = chosen[:, 0]
        uy = chosen[:, 1]
        ut = np.zeros(len(ux), dtype=np.float64)
        defc = need - len(ux)
        if defc > 10:
            continue

        # Scale gently to L0 and fill any remaining defc squares in max-clearance voids
        sc = L0 / L_tgt
        xs = np.concatenate([xt, ux]) * sc
        ys = np.concatenate([yt, uy]) * sc
        ts = np.concatenate([tt, ut])
        if defc > 0:
            pts = clearance_points(xs, ys, ts, L0, n_pts=max(8, defc + 2), step=0.14)
            t_ang = donor_tilt(th2)
            for j in range(defc):
                px, py, _ = pts[j % len(pts)]
                xs = np.append(xs, px + float(rng.uniform(-0.04, 0.04)))
                ys = np.append(ys, py + float(rng.uniform(-0.04, 0.04)))
                ts = np.append(ts, 0.0 if j % 2 == 0 else t_ang)
        seeds.append((xs[:n], ys[:n], ts[:n]))
        if len(seeds) < n_seeds and defc <= 4:
            # Also add a jittered copy around the void/seam
            x2j, y2j, t2j = xs[:n].copy(), ys[:n].copy(), ts[:n].copy()
            if defc > 0:
                cx, cy = float(x2j[-1]), float(y2j[-1])
                near = ((x2j - cx) ** 2 + (y2j - cy) ** 2) <= 4.5**2
                nm = int(near.sum())
                x2j[near] += rng.uniform(-0.12, 0.12, nm)
                y2j[near] += rng.uniform(-0.12, 0.12, nm)
                t2j[near] += rng.uniform(-0.10, 0.10, nm)
            seeds.append((x2j, y2j, t2j))
    return seeds[:n_seeds]


def incumbent_bottleneck_seeds(
    x_inc: np.ndarray, y_inc: np.ndarray, th_inc: np.ndarray, L_inc: float, L0: float,
    n_seeds: int, rng: np.random.Generator, target_tilt: float
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """6-mode pocket melts: pin-teleport, wall-pin pocket, 45° band unlock, void collapse, wall-slack drift, global band shift."""
    n = len(x_inc)
    sc = L0 / L_inc
    xb, yb, tb = x_inc * sc, y_inc * sc, th_inc.copy()
    hx = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))
    margin = np.minimum.reduce([xb - hx, L0 - xb - hx, yb - hx, L0 - yb - hx])
    tight_idx = np.argsort(margin)[:20]
    voids = clearance_points(xb, yb, tb, L0, n_pts=8, step=0.14)

    deg = np.degrees(tb) % 90.0
    tilted_mask = (deg > 2.0) & (deg < 88.0)
    tilted_idx = np.where(tilted_mask)[0]

    mL = xb - hx - float((xb - hx).min())
    mR = float((xb + hx).max()) - (xb + hx)
    mB = yb - hx - float((yb - hx).min())
    mT = float((yb + hx).max()) - (yb + hx)
    cL, cR = int(np.sum(mL < 5e-4)), int(np.sum(mR < 5e-4))
    cB, cT = int(np.sum(mB < 5e-4)), int(np.sum(mT < 5e-4))
    dx_sign = 1.0 if cR <= cL else -1.0
    dy_sign = 1.0 if cT <= cB else -1.0

    seeds = []
    for s in range(n_seeds):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        mode = s % 6
        if mode == 0 and len(voids) > 0:
            k_move = 1 if (s // 6) % 2 == 0 else min(2, len(voids))
            picks = rng.choice(tight_idx[:12], size=k_move, replace=False)
            for j, pi in enumerate(picks):
                vx, vy, _ = voids[(s + j) % len(voids)]
                x[pi] = vx + float(rng.uniform(-0.08, 0.08))
                y[pi] = vy + float(rng.uniform(-0.08, 0.08))
                th[pi] = [0.0, target_tilt, math.pi / 4.0][(s + j) % 3] + float(rng.uniform(-0.08, 0.08))
                near = ((x - vx) ** 2 + (y - vy) ** 2) <= 4.2**2
                nm = int(near.sum())
                x[near] += rng.uniform(-0.12, 0.12, nm)
                y[near] += rng.uniform(-0.12, 0.12, nm)
                th[near] += rng.uniform(-0.10, 0.10, nm)
        elif mode == 1:
            pi = int(tight_idx[(s // 6) % len(tight_idx)])
            cx, cy = float(x[pi]), float(y[pi])
            rad = float(rng.uniform(4.2, 6.5))
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
            nm = int(near.sum())
            x[near] += rng.uniform(-0.14, 0.14, nm)
            y[near] += rng.uniform(-0.14, 0.14, nm)
            th[near] += rng.uniform(-0.18, 0.18, nm)
        elif mode == 2:
            if len(tilted_idx) > 0:
                pi = int(tilted_idx[(s * 7) % len(tilted_idx)])
                cx, cy = float(x[pi]), float(y[pi])
                near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(5.0, 7.5)) ** 2
            else:
                near = np.ones(n, dtype=bool)
            nm = int(near.sum())
            x[near] += rng.uniform(-0.14, 0.14, nm)
            y[near] += rng.uniform(-0.14, 0.14, nm)
            th[near] += rng.uniform(-0.16, 0.16, nm)
        elif mode == 3 and len(voids) > 0:
            vx, vy, _ = voids[(s // 6) % len(voids)]
            rad = float(rng.uniform(4.5, 7.5))
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
            nm = int(near.sum())
            pull = float(rng.uniform(0.03, 0.08))
            x[near] += pull * (vx - x[near]) + rng.uniform(-0.11, 0.11, nm)
            y[near] += pull * (vy - y[near]) + rng.uniform(-0.11, 0.11, nm)
            th[near] += rng.uniform(-0.16, 0.16, nm)
        elif mode == 4:
            drift = float(rng.uniform(0.03, 0.09))
            cx, cy = (x[tight_idx[0]], y[tight_idx[0]]) if len(voids) == 0 else (
                0.5 * (float(x[tight_idx[s % 6]]) + voids[0][0]),
                0.5 * (float(y[tight_idx[s % 6]]) + voids[0][1]),
            )
            half_x = (x < L0 * 0.65) if dx_sign > 0 else (x > L0 * 0.35)
            half_y = (y < L0 * 0.65) if dy_sign > 0 else (y > L0 * 0.35)
            x[half_x] += dx_sign * drift
            y[half_y] += dy_sign * 0.6 * drift
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(5.2, 8.0)) ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.13, 0.13, nm)
            y[near] += rng.uniform(-0.13, 0.13, nm)
            th[near] += rng.uniform(-0.14, 0.14, nm)
        else:
            # 45° band targeted unlock: nudge tilted band angles coherently toward target_tilt + local jitter
            if len(tilted_idx) > 0:
                sgn = 1.0 if (s // 6) % 2 == 0 else -1.0
                shift_ang = (target_tilt - math.pi / 4.0) * sgn
                th[tilted_idx] += shift_ang + rng.uniform(-0.08, 0.08, len(tilted_idx))
                x[tilted_idx] += rng.uniform(-0.10, 0.10, len(tilted_idx))
                y[tilted_idx] += rng.uniform(-0.10, 0.10, len(tilted_idx))
            else:
                x += rng.uniform(-0.08, 0.08, n)
                y += rng.uniform(-0.08, 0.08, n)
                th += rng.uniform(-0.10, 0.10, n)
        seeds.append((x, y, th))
    return seeds


def deficit_ins_seeds(
    mats: dict[int, tuple], n: int, L_inc: float, L0: float, n_seeds: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Insert 1 or 2 squares into clean n-1 / n-2 donors (important for n=974 from 973/972)."""
    seeds = []
    for d in (1, 2):
        n2 = n - d
        if n2 not in mats:
            continue
        x, y, th, L = mats[n2]
        if L >= L_inc - 0.0005:
            continue
        sc = (L0 - 1.0) / max(1e-9, L - 1.0)
        xb = 0.5 + (x - 0.5) * sc
        yb = 0.5 + (y - 0.5) * sc
        pts = clearance_points(xb, yb, th, L0, n_pts=8, step=0.10)
        tilt = donor_tilt(th)
        angs = [0.0, tilt, math.pi / 4.0]
        cnt = max(2, n_seeds // 2) if d == 1 else max(2, n_seeds - n_seeds // 2)
        for s in range(cnt):
            xs, ys, ts = xb.copy(), yb.copy(), th.copy()
            for j in range(d):
                px, py, _ = pts[(s + j) % len(pts)]
                xs = np.append(xs, px + float(rng.normal(0, 0.05)))
                ys = np.append(ys, py + float(rng.normal(0, 0.05)))
                ts = np.append(ts, angs[(s + j) % len(angs)] + float(rng.normal(0, 0.04)))
                near = ((xs - px) ** 2 + (ys - py) ** 2) <= 4.0**2
                nm = int(near.sum())
                xs[near] += rng.uniform(-0.12, 0.12, nm)
                ys[near] += rng.uniform(-0.12, 0.12, nm)
                ts[near] += rng.uniform(-0.08, 0.08, nm)
            seeds.append((xs, ys, ts))
    return seeds[:n_seeds]


def _polish_and_repair_worker(args: tuple) -> dict:
    """Finish a C++ basin with relax_lbfgs + joint_xytheta_slp + multi-margin LP tighten + exact verification."""
    idx, label, fam, n, x, y, th, cpp_E, cpp_pen, cpp_L = args
    t0 = time.time()

    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())

    if 1e-7 < cpp_pen < 0.05:
        L_try = cpp_L + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, _ = u.relax_lbfgs(x, y, th, L_try, maxiter=220, w_wall=4.0, w_pair=1.0, side=1.0)
        pen_r, L_r = u.sat_max_overlap_and_bbox(xr, yr, thr, side=1.0)
        if pen_r <= cpp_pen + 1e-6:
            x, y, th, cpp_pen, cpp_L = xr, yr, thr, pen_r, L_r

    pen_pre, L_pre = u.sat_max_overlap_and_bbox(x, y, th, side=1.0)
    ok_slp, L_slp = False, L_pre
    x_f, y_f, th_f = x.copy(), y.copy(), th.copy()

    if pen_pre < 0.035:
        slp_rounds = 10 if n >= 700 else 14
        xs, ys, ts, Ls, ok_s = u.joint_xytheta_slp(x, y, th, rounds=slp_rounds, tr_th0=0.025, tr_xy0=0.12, margin=1e-8)
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True

    ver: dict = {"ok": False}
    for m_lp in (1e-9, 5e-8, 2e-7, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = u.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=6, margin=m_lp)
        if ok_lp2 and ((m_lp == 1e-9 and L_lp2 < L_slp) or (m_lp > 1e-9 and L_lp2 < L_slp + 5e-6)):
            cand_ver = u.verify_official(n, xf2, yf2, th_f)
            if cand_ver.get("ok"):
                x_f, y_f, L_slp, ok_slp, ver = xf2, yf2, L_lp2, True, cand_ver
                break
            ver = cand_ver

    pen_fin, L_fin = u.sat_max_overlap_and_bbox(x_f, y_f, th_f, side=1.0)
    if not ver.get("ok") and pen_fin < 1e-6:
        ver = u.verify_official(n, x_f, y_f, th_f)
    if not ver.get("ok") and pen_fin < 5e-6:
        for eps in (2e-9, 1e-8, 5e-8, 2e-7, 1e-6, 5e-6):
            sc = 1.0 + eps
            cand_ver = u.verify_official(n, x_f * sc, y_f * sc, th_f)
            if cand_ver.get("ok"):
                x_f, y_f = x_f * sc, y_f * sc
                pen_fin, L_fin = u.sat_max_overlap_and_bbox(x_f, y_f, th_f, side=1.0)
                ver = cand_ver
                break

    ang_deg = np.degrees(th_f) % 90.0
    n45 = int(np.sum(np.abs(ang_deg - 45.0) < 0.2))
    n0 = int(np.sum((ang_deg < 0.2) | (ang_deg > 89.8)))
    return {
        "idx": idx,
        "label": label,
        "family": fam,
        "n": n,
        "cpp_E": float(cpp_E),
        "cpp_pen": float(cpp_pen),
        "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp),
        "pen_fin": float(pen_fin),
        "L_fin": float(L_fin),
        "n0": n0,
        "n45": n45,
        "non45_non0": int(n - n0 - n45),
        "verify": {k: v for k, v in ver.items() if k not in ("rows", "x", "y", "th")},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def _select_polish_indices(labels: list[str], lf: np.ndarray, po: np.ndarray, lb: np.ndarray, k_keep: int) -> list[int]:
    score = lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i]), float(lb[i]))
    order = sorted(range(len(labels)), key=score)
    chosen: list[int] = []
    used = set()
    for prefix, quota in (("s", 3), ("x", 2), ("milp", 2), ("inc_", max(3, k_keep // 3)), ("m", 2)):
        cnt = 0
        for i in order:
            if i not in used and labels[i].startswith(prefix):
                chosen.append(i)
                used.add(i)
                cnt += 1
                if cnt >= quota or len(chosen) >= k_keep:
                    break
    for i in order:
        if len(chosen) >= k_keep:
            break
        if i not in used:
            chosen.append(i)
            used.add(i)
    return chosen


def solve_target(
    n: int,
    df_merged: pd.DataFrame,
    mats: dict[int, tuple],
    front_map: dict[int, float],
    lib: ctypes.CDLL,
    n_threads: int,
    quick: bool,
) -> dict:
    t0 = time.time()
    inc_L_d, x_inc, y_inc, th_inc = u.load_incumbent_exact(n, df_merged)
    L_inc = float(inc_L_d)
    L_front = front_map.get(n, L_inc - 0.01)
    rng = np.random.default_rng(45200 + n)

    peels = collect_surplus_peels(n, L_inc, mats)[: (5 if quick else 8)]
    L0 = L_inc + 0.04
    L_sched_base = np.array(
        [L0, L_inc + 0.008, L_inc - 2e-4, max(L_front + 2e-4, L_inc - 0.005)],
        dtype=np.float64,
    )

    seed_triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    labels: list[str] = []
    ls_list: list[np.ndarray] = []

    n_peel_seeds = 3 if quick else 5
    for p in peels:
        L0_p = max(L0, min(L_inc + 0.14, p["L"] + min(0.12, p["max_oh_rem"]) + 0.02))
        ls_p = np.array([L0_p, L_inc + 0.012, L_inc - 2e-4, L_sched_base[-1]], dtype=np.float64)
        for i, tri in enumerate(_surplus_drop_seeds(p, n, L0_p, n_peel_seeds, rng)):
            seed_triples.append(tri)
            labels.append(f"{p['src']}_s{i}")
            ls_list.append(ls_p)

    # Phase-exact tilted-MILP seeds
    m_seeds = tilted_milp_seeds(n, L_inc, L_inc - 0.003, L0, mats, rng, n_seeds=3 if quick else 5)
    for i, tri in enumerate(m_seeds):
        seed_triples.append(tri)
        labels.append(f"milp_{i}")
        ls_list.append(L_sched_base)

    # Target tilt from top peel or family prior (~40.97° or ~24.8°)
    t_tilt = donor_tilt(peels[0]["th"]) if peels else (math.radians(40.97) if (L_inc - math.isqrt(n)) < 0.62 else math.radians(24.8))
    n_inc_seeds = 14 if quick else 24
    for i, tri in enumerate(incumbent_bottleneck_seeds(x_inc, y_inc, th_inc, L_inc, L0, n_inc_seeds, rng, t_tilt)):
        seed_triples.append(tri)
        labels.append(f"inc_melt_{i}")
        ls_list.append(L_sched_base)

    # Deficit n-1 / n-2 insertion seeds
    n_def_seeds = 6 if (n == 974 or not peels) else (3 if quick else 6)
    for i, tri in enumerate(deficit_ins_seeds(mats, n, L_inc, L0, n_def_seeds, rng)):
        seed_triples.append(tri)
        labels.append(f"m_{i}")
        ls_list.append(L_sched_base)

    batch = len(seed_triples)
    z_batch = np.zeros((batch, 3 * n), dtype=np.float64)
    for b, (xs, ys, ts) in enumerate(seed_triples):
        z_batch[b, :n] = xs
        z_batch[b, n : 2 * n] = ys
        z_batch[b, 2 * n :] = ts
    z_feas = np.zeros_like(z_batch)
    L_sched_batch = np.ascontiguousarray(np.vstack(ls_list), dtype=np.float64)

    a_sched = np.array([0.78, 0.86, 0.93, 0.98, 1.0], dtype=np.float64)
    iters_per_a = np.array([90, 100, 120, 140, 180], dtype=np.int32)
    ils_rounds = 4 if n >= 700 else 5

    out_best_L_feas = np.full(batch, 1e18, dtype=np.float64)
    out_E = np.zeros(batch, dtype=np.float64)
    out_pen = np.zeros(batch, dtype=np.float64)
    out_L_bbox = np.zeros(batch, dtype=np.float64)

    print(
        f"\n=== Idea 2 n={n} | inc={L_inc:.9f} front={L_front:.9f} | batch={batch} (peels={len(peels)}, milp={len(m_seeds)}) ===",
        flush=True,
    )
    for p in peels[:4]:
        print(f"   peel {p['src']}: surp=+{p['surplus']} L={p['L']:.6f} oh_rem={p['max_oh_rem']:.4f}", flush=True)

    lib.batch_melt_solve(
        batch,
        n,
        z_batch.ravel(),
        z_feas.ravel(),
        L_sched_batch.shape[1],
        L_sched_batch.ravel(),
        len(a_sched),
        a_sched,
        iters_per_a,
        ils_rounds,
        None,
        min(4, n_threads),
        452000 + n,
        out_best_L_feas,
        out_E,
        out_pen,
        out_L_bbox,
    )

    n_feas = int(np.sum(out_best_L_feas < 1e9))
    best_Lf = float(np.min(out_best_L_feas)) if n_feas else None
    min_pen_i = int(np.argmin(out_pen))
    print(
        f"  [C++ melt] dt={time.time()-t0:.1f}s feas={n_feas}/{batch} best_Lf={best_Lf} "
        f"min_pen={out_pen[min_pen_i]:.3e} ({labels[min_pen_i]} L={out_L_bbox[min_pen_i]:.6f})",
        flush=True,
    )

    k_keep = 8 if quick else 12
    keep = _select_polish_indices(labels, out_best_L_feas, out_pen, out_L_bbox, k_keep)
    tasks = []
    for i in keep:
        use_f = out_best_L_feas[i] < 1e9
        src_z = z_feas[i] if use_f else z_batch[i]
        fam = (
            "surplus_peel"
            if labels[i].startswith(("s", "x"))
            else ("tilted_milp" if labels[i].startswith("milp") else ("inc_melt" if labels[i].startswith("inc_") else "deficit_ins"))
        )
        tasks.append((
            i,
            labels[i],
            fam,
            n,
            src_z[:n].copy(),
            src_z[n : 2 * n].copy(),
            src_z[2 * n :].copy(),
            float(out_E[i]),
            0.0 if use_f else float(out_pen[i]),
            float(out_best_L_feas[i]) if use_f else float(out_L_bbox[i]),
        ))

    if min(4, n_threads) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(4, n_threads)) as ex:
            polished = list(ex.map(_polish_and_repair_worker, tasks))
    else:
        polished = [_polish_and_repair_worker(t) for t in tasks]

    best_win = None
    best_near = None
    for r in polished:
        ver = r["verify"]
        if ver.get("ok"):
            L_lit_d = Decimal(ver["L_lit"])
            L_pd_d = Decimal(ver["L_pandas"])
            print(
                f"   [POLISH OK] {r['label']} ({r['family']}): L_lit={ver['L_lit'][:16]} L_pd={ver['L_pandas'][:16]} "
                f"non45={r['non45_non0']} n45={r['n45']} dt={r['dt']}s",
                flush=True,
            )
            if L_lit_d < inc_L_d and L_pd_d < inc_L_d:
                cand_d = max(L_lit_d, L_pd_d)
                if best_win is None or cand_d < max(Decimal(best_win["verify"]["L_lit"]), Decimal(best_win["verify"]["L_pandas"])):
                    best_win = r
        else:
            if best_near is None or (r["pen_fin"], r["L_fin"]) < (best_near["pen_fin"], best_near["L_fin"]):
                best_near = r

    if best_win is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        rows = best_win["rows"]
        win_df = pd.DataFrame([{"n": n, "x": r["x"], "y": r["y"], "angle": r["angle"]} for r in rows])
        win_path = DATA_DIR / f"win_n{n}.csv"
        win_df.to_csv(win_path, index=False)
        u.write_final_csv_from_wins(DATA_DIR, SWARM_DIR / "final.csv")
        print(f"  >>> WIN CERTIFIED n={n}: {best_win['verify']['L_lit']} < inc {inc_L_d} -> {win_path}", flush=True)
    elif best_near is not None:
        print(
            f"  [NO WIN n={n}] best_near={best_near['label']} L={best_near['L_fin']:.6f} pen={best_near['pen_fin']:.3e} "
            f"non45={best_near['non45_non0']}",
            flush=True,
        )

    return {
        "n": n,
        "inc_L": str(inc_L_d),
        "front_L": L_front,
        "dt": round(time.time() - t0, 2),
        "batch": batch,
        "cpp_feas": n_feas,
        "win": None
        if best_win is None
        else {
            "label": best_win["label"],
            "family": best_win["family"],
            "L_lit": best_win["verify"]["L_lit"],
            "L_pandas": best_win["verify"]["L_pandas"],
            "non45_non0": best_win["non45_non0"],
            "n45": best_win["n45"],
        },
        "best_near": None
        if best_near is None
        else {
            "label": best_near["label"],
            "family": best_near["family"],
            "L_fin": best_near["L_fin"],
            "pen_fin": best_near["pen_fin"],
            "non45_non0": best_near["non45_non0"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=[581, 850, 448, 366])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tag", type=str, default="wave1")
    args = parser.parse_args()

    lib = ensure_cpp_lib()
    df_merged, mats = load_merged_mats()
    front_map = load_front_map()
    results = []
    for n in args.targets:
        results.append(solve_target(n, df_merged, mats, front_map, lib, min(4, args.threads), args.quick))

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = DOCS_DIR / f"idea2_results_{args.tag}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved summary -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
