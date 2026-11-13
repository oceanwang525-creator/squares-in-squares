"""Idea 4: De novo oriented-domain inflation (a: 0.90 -> 1.00) on n=401 (both free 401 and C4 1+4*100)."""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import utils as u

OUT_DIR = Path(__file__).resolve().parents[1] / "docs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def expand_c4_state(v: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand 1 center angle + m quadrant squares (x, y, th) into n = 1 + 4*m squares."""
    m = (len(v) - 1) // 3
    th0 = v[0]
    qx = v[1 : 1 + m]
    qy = v[1 + m : 1 + 2 * m]
    qth = v[1 + 2 * m :]

    half = 0.5 * L
    dx = qx - half
    dy = qy - half

    # 4 rotations by 0, 90, 180, 270 deg around (L/2, L/2)
    xs = np.concatenate([
        [half],
        half + dx,
        half - dy,
        half - dx,
        half + dy,
    ])
    ys = np.concatenate([
        [half],
        half + dy,
        half + dx,
        half - dy,
        half - dx,
    ])
    ths = np.concatenate([
        [th0],
        qth,
        qth + math.pi / 2,
        qth + math.pi,
        qth + 3 * math.pi / 2,
    ])
    return xs, ys, ths


def c4_penalty_grad(v: np.ndarray, L: float, w_wall: float, w_pair: float, side: float) -> tuple[float, np.ndarray]:
    xs, ys, ths = expand_c4_state(v, L)
    n = len(xs)
    m = (len(v) - 1) // 3
    state = np.concatenate([xs, ys, ths])
    e, g = u.sat_penalty_grad(state, L, w_wall, w_pair, side)
    gx, gy, gth = g[:n], g[n : 2 * n], g[2 * n :]

    gv = np.zeros_like(v)
    gv[0] = gth[0]

    # Chain rule from 4 copies back to (qx, qy, qth)
    s0 = 1
    s1 = 1 + m
    s2 = 1 + 2 * m
    s3 = 1 + 3 * m
    # Copy 0: (half+dx, half+dy, qth)
    # Copy 1: (half-dy, half+dx, qth+pi/2) -> d(x1)/dx = 0, d(x1)/dy = -1; d(y1)/dx = 1, d(y1)/dy = 0
    # Copy 2: (half-dx, half-dy, qth+pi)
    # Copy 3: (half+dy, half-dx, qth+3pi/2)
    gv[1 : 1 + m] = gx[s0 : s0 + m] + gy[s1 : s1 + m] - gx[s2 : s2 + m] - gy[s3 : s3 + m]
    gv[1 + m : 1 + 2 * m] = gy[s0 : s0 + m] - gx[s1 : s1 + m] - gy[s2 : s2 + m] + gx[s3 : s3 + m]
    gv[1 + 2 * m :] = gth[s0 : s0 + m] + gth[s1 : s1 + m] + gth[s2 : s2 + m] + gth[s3 : s3 + m]
    return e, gv


def make_seed(kind: str, seed: int, L: float, n: int = 401) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = 20
    # Base 20x20 grid + 1 extra
    xs = []
    ys = []
    ths = []
    scale = (L - 0.15) / k
    for i in range(k):
        for j in range(k):
            xs.append(0.075 + (i + 0.5) * scale)
            ys.append(0.075 + (j + 0.5) * scale)
            ths.append(0.0)
    xs = np.array(xs + [0.5 * L], dtype=np.float64)
    ys = np.array(ys + [0.5 * L], dtype=np.float64)
    ths = np.array(ths + [math.radians(30.0)], dtype=np.float64)

    if kind == "swirl_frame":
        # Tilted outer ring / corners, upright inner
        th_t = math.radians(rng.uniform(18.0, 42.0))
        dx = xs - 0.5 * L
        dy = ys - 0.5 * L
        r = np.maximum(np.abs(dx), np.abs(dy))
        mask = r > 0.30 * L
        # 4 quadrants get alternating chiral angle
        q1 = (dx >= 0) & (dy >= 0)
        q2 = (dx < 0) & (dy >= 0)
        q3 = (dx < 0) & (dy < 0)
        q4 = (dx >= 0) & (dy < 0)
        ths[mask & (q1 | q3)] = th_t
        ths[mask & (q2 | q4)] = -th_t
    elif kind == "tilted_core":
        th_t = math.radians(rng.uniform(20.0, 44.0))
        dx = xs - 0.5 * L
        dy = ys - 0.5 * L
        mask = (np.abs(dx) + np.abs(dy)) < 0.55 * L
        ths[mask] = th_t
    elif kind == "two_pockets":
        th_t = math.radians(rng.uniform(20.0, 42.0))
        dx = xs - 0.5 * L
        dy = ys - 0.5 * L
        m_nw = (dx < 0.05 * L) & (dy > -0.05 * L) & (np.abs(dx + dy) < 0.45 * L) & (dy - dx > 0.25 * L)
        m_se = (dx > -0.05 * L) & (dy < 0.05 * L) & (np.abs(dx + dy) < 0.45 * L) & (dx - dy > 0.25 * L)
        ths[m_nw] = th_t
        ths[m_se] = -th_t
    elif kind == "diag_shear":
        th_t = math.radians(rng.uniform(22.0, 42.0))
        band = np.abs(xs - ys) < rng.uniform(1.8, 4.5)
        ths[band] = th_t
    elif kind == "inc_shake":
        import pandas as pd
        df = pd.read_csv(u.REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv")
        sub = df[df["n"] == n]
        ix, iy, ia = sub["x"].to_numpy().copy(), sub["y"].to_numpy().copy(), sub["angle"].to_numpy().copy()
        hx = 0.5 * (np.abs(np.cos(ia)) + np.abs(np.sin(ia)))
        ix -= (ix - hx).min()
        iy -= (iy - hx).min()
        _, L0 = u.sat_max_overlap_and_bbox(ix, iy, ia)
        xs = ix * (L / L0)
        ys = iy * (L / L0)
        ths = ia + rng.normal(0.0, 0.22, size=n)

    xs = np.clip(xs + rng.normal(0.0, 0.06, size=n), 0.5, L - 0.5)
    ys = np.clip(ys + rng.normal(0.0, 0.06, size=n), 0.5, L - 0.5)
    ths = ths + rng.normal(0.0, 0.04, size=n)
    return xs, ys, ths


def run_free_inflation(args: tuple) -> dict:
    seed, kind, L_target, L_ours = args
    n = 401
    xs, ys, ths = make_seed(kind, seed, L_target, n)
    sides = [0.88, 0.92, 0.95, 0.975, 0.99, 0.996, 1.0]
    for idx, side in enumerate(sides):
        iters = 120 if idx < len(sides) - 1 else 250
        xs, ys, ths, pen = u.relax_lbfgs(xs, ys, ths, L_target, maxiter=iters, side=side)
    # Try LP tighten if pen is very small
    lp_L = None
    cert_ok = False
    if pen < 1e-6:
        lx, ly, lp_L_val, lok = u.fixed_angle_lp_tighten(xs, ys, ths, rounds=5)
        if lok:
            lp_L = lp_L_val
            if lp_L_val < L_ours - 1e-9:
                cert = u.verify_official(n, lx, ly, ths)
                if cert["ok"] and cert["L_float"] < L_ours:
                    cert_ok = True
                    return {"mode": "free", "kind": kind, "seed": seed, "pen": float(pen), "lp_L": float(lp_L), "cert": cert}
    return {"mode": "free", "kind": kind, "seed": seed, "pen": float(pen), "lp_L": float(lp_L) if lp_L else None, "cert_ok": cert_ok}


def run_c4_inflation(args: tuple) -> dict:
    seed, L_target, L_ours = args
    rng = np.random.default_rng(seed)
    m = 100
    # Initialize 100 squares in quadrant [L/2, L] x [L/2, L] (10x10 grid!)
    qx = np.empty(m, dtype=np.float64)
    qy = np.empty(m, dtype=np.float64)
    qth = np.zeros(m, dtype=np.float64)
    half = 0.5 * L_target
    step = (half - 0.1) / 10.0
    idx = 0
    for i in range(10):
        for j in range(10):
            qx[idx] = half + (i + 0.5) * step + rng.normal(0.0, 0.04)
            qy[idx] = half + (j + 0.5) * step + rng.normal(0.0, 0.04)
            idx += 1
    # Seed non-45 angle on outer/inner/all quadrant
    mode = seed % 3
    th_init = math.radians(rng.uniform(15.0, 42.0))
    if mode == 0:
        qth[:] = th_init + rng.normal(0.0, 0.05, size=m)
    elif mode == 1:
        # Outer rim tilted, inner upright
        r = np.maximum(qx - half, qy - half)
        qth[r > 0.55 * half] = th_init
        qth += rng.normal(0.0, 0.04, size=m)
    else:
        # Inner core tilted
        r = (qx - half) + (qy - half)
        qth[r < 0.7 * half] = th_init
        qth += rng.normal(0.0, 0.04, size=m)

    th0 = rng.uniform(0.0, math.pi / 4)
    v = np.concatenate([[th0], qx, qy, qth])
    sides = [0.88, 0.92, 0.95, 0.975, 0.99, 0.996, 1.0]
    for s_idx, side in enumerate(sides):
        iters = 140 if s_idx < len(sides) - 1 else 300
        res = minimize(
            lambda vv: c4_penalty_grad(vv, L_target, 1.0, 1.0, side),
            v,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": iters, "ftol": 1e-15, "gtol": 1e-12},
        )
        v = res.x
    xs, ys, ths = expand_c4_state(v, L_target)
    pen, cur_L = u.sat_max_overlap_and_bbox(xs, ys, ths, 1.0)
    hx = 0.5 * (np.abs(np.cos(ths)) + np.abs(np.sin(ths)))
    wpen = max(0.0, float((hx - xs).max()), float((xs + hx - L_target).max()), float((hx - ys).max()), float((ys + hx - L_target).max()))
    pen = max(pen, wpen)
    lp_L = None
    if pen < 1e-6:
        lx, ly, lp_L_val, lok = u.fixed_angle_lp_tighten(xs, ys, ths, rounds=5)
        if lok:
            lp_L = lp_L_val
            if lp_L_val < L_ours - 1e-9:
                cert = u.verify_official(401, lx, ly, ths)
                if cert["ok"] and cert["L_float"] < L_ours:
                    return {"mode": "c4", "seed": seed, "pen": float(pen), "lp_L": float(lp_L), "cert": cert}
    return {"mode": "c4", "seed": seed, "pen": float(pen), "lp_L": float(lp_L) if lp_L else None}


def main():
    info = u.load_incumbent_and_frontier([401])
    L_ours = info[401]["L_ours"]
    L_front = info[401]["L_front"]
    L_probe = 20.5315  # clear gap below 20.535534

    t0 = time.time()
    kinds = ["swirl_frame", "tilted_core", "two_pockets", "diag_shear", "inc_shake"]
    free_jobs = [(100 + i, kinds[i % len(kinds)], L_probe, L_ours) for i in range(21)]
    c4_jobs = [(500 + i, L_probe, L_ours) for i in range(21)]

    with ProcessPoolExecutor(max_workers=7) as ex:
        res_free = list(ex.map(run_free_inflation, free_jobs))
    with ProcessPoolExecutor(max_workers=7) as ex:
        res_c4 = list(ex.map(run_c4_inflation, c4_jobs))

    all_res = res_free + res_c4
    all_res.sort(key=lambda r: r["pen"])
    dt = time.time() - t0
    print(f"Idea 4 on n=401 (L_probe={L_probe}, ours={L_ours:.6f}, front={L_front:.6f}), time={dt:.1f}s", flush=True)
    for r in all_res[:10]:
        print(" ", {k: v for k, v in r.items() if k != "cert"}, flush=True)
        if "cert" in r:
            print("  CERTIFIED WIN:", r["cert"]["L_lit"], r["cert"]["L_pandas"], flush=True)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            # Save win CSV immediately
            import pandas as pd
            rows = [dict(id=idx, n=401, **row) for idx, row in enumerate(r["cert"]["rows"])]
            pd.DataFrame(rows, columns=["id", "n", "square_id", "x", "y", "angle"]).to_csv(DATA_DIR / "win_n401.csv", index=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = [{k: v for k, v in r.items() if k != "cert"} for r in all_res]
    (OUT_DIR / "idea4_inflation.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
