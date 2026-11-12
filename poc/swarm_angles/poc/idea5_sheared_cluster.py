"""Idea 5: 4-quadrant offset upright grids with carved cavities (n=50 / n=37 / n=908 style) + n=401 pocket expansion + C4 core-pocket inflation."""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import utils as u

OUT_DIR = Path(__file__).resolve().parents[1] / "docs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_offset_grid(k: int, L: float, sx: int, sy: int) -> tuple[np.ndarray, np.ndarray]:
    """Build k*k non-overlapping upright squares aligned to the 4 corners around split (sx, sy)."""
    f = L - k
    xs, ys = [], []
    for i in range(k):
        cx = (0.5 + i) if i < sx else (0.5 + i + f)
        for j in range(k):
            cy = (0.5 + j) if j < sy else (0.5 + j + f)
            xs.append(cx)
            ys.append(cy)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def carve_and_fill_worker(args: tuple) -> dict:
    (job_id, n_target, L_probe, sx, sy, carve_kind, param, angle_mode, seed) = args
    k = math.isqrt(n_target - 1)
    rng = np.random.default_rng(seed)
    gx, gy = build_offset_grid(k, L_probe, sx, sy)

    # Compute center of cavity
    cx0 = sx + 0.5 * (L_probe - k)
    cy0 = sy + 0.5 * (L_probe - k)
    dx = gx - cx0
    dy = gy - cy0

    if carve_kind == "diamond":
        # Central L1 diamond |dx| + |dy| < R
        dist = np.abs(dx) + np.abs(dy)
        order = np.argsort(dist)
        m_carve = int(param)
        carve_idx = order[:m_carve]
    elif carve_kind == "diag_band":
        # Band along x + y - L or x - y:
        band_dist = np.abs(dx - dy)
        # Prefer center first when band_dist ties
        score = band_dist + 0.05 * np.abs(dx + dy)
        order = np.argsort(score)
        m_carve = int(param)
        carve_idx = order[:m_carve]
    elif carve_kind == "nw_se_pockets":
        # Two symmetric cavities along NW (dx < 0, dy > 0) and SE (dx > 0, dy < 0)
        # Slope bias for 1:2 or 1:1 pockets
        slope = param[1]
        m_each = int(param[0])
        # Center of NW pocket around (-sx/2, (k-sy)/2), SE around ((k-sx)/2, -sy/2)
        c_nw = (-0.5 * sx, 0.5 * (k - sy))
        c_se = (0.5 * (k - sx), -0.5 * sy)
        d_nw = np.abs(dx - c_nw[0]) * slope + np.abs(dy - c_nw[1])
        d_se = np.abs(dx - c_se[0]) * slope + np.abs(dy - c_se[1])
        # Exclude near-origin overlap, take m_each from NW and m_each from SE disjointly
        ord_nw = np.argsort(d_nw)
        picked = set(int(i) for i in ord_nw[:m_each])
        ord_se = [int(i) for i in np.argsort(d_se) if int(i) not in picked]
        carve_idx = np.array(list(picked) + ord_se[:m_each], dtype=int)
    elif carve_kind == "frame_ring":
        # Carve outer/inner ring around centered (k-2w)^2 core
        m_carve = int(param)
        r = np.maximum(np.abs(gx - 0.5 * L_probe), np.abs(gy - 0.5 * L_probe))
        order = np.argsort(-r)  # outer frame
        carve_idx = order[:m_carve]
    else:
        raise ValueError(carve_kind)

    mask_keep = np.ones(len(gx), dtype=bool)
    mask_keep[carve_idx] = False
    ux, uy = gx[mask_keep], gy[mask_keep]
    hx, hy = gx[~mask_keep], gy[~mask_keep]
    n_up = len(ux)
    n_fill = n_target - n_up

    # Place n_fill tilted squares inside carved sites (+ extra jittered points near hole center)
    tx = np.empty(n_fill, dtype=np.float64)
    ty = np.empty(n_fill, dtype=np.float64)
    tth = np.empty(n_fill, dtype=np.float64)
    m_hole = len(hx)
    tx[:m_hole] = hx + rng.normal(0.0, 0.08, size=m_hole)
    ty[:m_hole] = hy + rng.normal(0.0, 0.08, size=m_hole)
    for idx in range(m_hole, n_fill):
        j = rng.integers(0, m_hole)
        tx[idx] = hx[j] + rng.normal(0.0, 0.25)
        ty[idx] = hy[j] + rng.normal(0.0, 0.25)

    th0 = math.radians(rng.uniform(18.0, 42.0))
    if angle_mode == "uniform":
        tth[:] = th0 + rng.normal(0.0, 0.03, size=n_fill)
    elif angle_mode == "chevron":
        # Two halves +-delta around 45 deg or +-th0
        base = math.radians(45.0) if rng.random() < 0.5 else th0
        delta = math.radians(rng.uniform(3.0, 14.0))
        tth[:] = np.where((tx - 0.5 * L_probe) + (ty - 0.5 * L_probe) >= 0.0, base - delta, base + delta)
    elif angle_mode == "chiral_nwse":
        tth[:] = np.where(tx < ty, th0, -th0)
    else:
        tth[:] = rng.uniform(0.25, 0.72, size=n_fill)

    x = np.concatenate([ux, tx])
    y = np.concatenate([uy, ty])
    th = np.concatenate([np.zeros(n_up, dtype=np.float64), tth])

    # Stage 1: Inflate only the carved tilted squares (keep corners frozen)
    free_tilted = np.zeros(n_target, dtype=bool)
    free_tilted[n_up:] = True
    for side, iters in [(0.86, 100), (0.91, 100), (0.95, 120), (0.98, 120), (1.0, 160)]:
        x, y, th, pen = u.relax_lbfgs(x, y, th, L_probe, free_mask=free_tilted, maxiter=iters, side=side)

    # Stage 2: If cavity pen is promising (< 0.035), unfreeze ALL squares!
    lp_L = None
    cert_ok = False
    if pen < 0.035:
        x, y, th, pen = u.relax_lbfgs(x, y, th, L_probe, free_mask=None, maxiter=250, side=1.0)
        if pen < 1e-6:
            lx, ly, lp_val, lok = u.fixed_angle_lp_tighten(x, y, th, rounds=5)
            if lok:
                lp_L = lp_val
                cert = u.verify_official(n_target, lx, ly, th)
                cert_ok = bool(cert["ok"])
                if cert_ok:
                    return {
                        "job_id": job_id,
                        "kind": carve_kind,
                        "param": str(param),
                        "angle_mode": angle_mode,
                        "sx": sx,
                        "sy": sy,
                        "n_up": n_up,
                        "n_fill": n_fill,
                        "pen": float(pen),
                        "lp_L": float(lp_L),
                        "cert_ok": cert_ok,
                        "cert": cert,
                    }

    return {
        "job_id": job_id,
        "kind": carve_kind,
        "param": str(param),
        "angle_mode": angle_mode,
        "sx": sx,
        "sy": sy,
        "n_up": n_up,
        "n_fill": n_fill,
        "pen": float(pen),
        "lp_L": float(lp_L) if lp_L else None,
        "cert_ok": cert_ok,
    }


def test_c4_core_ring_worker(args: tuple) -> dict:
    """C4 symmetric (2m+1)^2 upright (or tilted) core + 4 rotated quadrant pockets."""
    m_core_half, L_probe, L_ours, seed = args
    rng = np.random.default_rng(seed)
    n = 401
    core_side = 2 * m_core_half + 1
    core_n = core_side * core_side
    Q = (n - core_n) // 4  # 10, 28, 44, 58, ...

    # Quadrant Q squares live outside [-core_side/2, core_side/2]^2 in [L/2, L] x [0, L]?
    # Actually place Q squares in one fundamental wedge: x in [L/2, L], y in [0, L], outside core
    # Easier: build full C4 expansion! Core center (L/2, L/2) has m_core_half*(m_core_half+1) quadrant squares!
    half = 0.5 * L_probe
    # Core quadrant: i in 1..m_core_half, j in 0..m_core_half -> (2m+1)^2 = 1 + 4*m*(m+1)!
    qx, qy, qth = [], [], []
    # Core quadrant points in relative coords (dx, dy) rotated by 0,90,180,270 fill core cleanly!
    th_core = 0.0 if (seed % 2 == 0) else math.radians(rng.uniform(15.0, 40.0))
    c_c, s_c = math.cos(th_core), math.sin(th_core)
    for i in range(1, m_core_half + 1):
        for j in range(0, m_core_half + 1):
            rx = i * c_c - j * s_c
            ry = i * s_c + j * c_c
            qx.append(half + rx)
            qy.append(half + ry)
            qth.append(th_core)

    # Remaining Q squares in sector (x > L/2 - 0.5, y >= L/2 - 0.5 + ...) or L-strip around core!
    w_margin = half - 0.5 * core_side
    th_pock = math.radians(rng.uniform(16.0, 42.0))
    for q_idx in range(Q):
        # Distribute in top-right L-strip: either right edge [L - w_margin, L] x [L/2 - core_side/2, L]
        if q_idx % 2 == 0:
            px = L_probe - rng.uniform(0.5, max(0.6, w_margin))
            py = rng.uniform(half - 0.4 * core_side, L_probe - 0.5)
        else:
            px = rng.uniform(half - 0.4 * core_side, L_probe - 0.5)
            py = L_probe - rng.uniform(0.5, max(0.6, w_margin))
        qx.append(px)
        qy.append(py)
        qth.append(th_pock + rng.normal(0.0, 0.08))

    v = np.concatenate([[th_core], np.array(qx), np.array(qy), np.array(qth)])

    def expand(vv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m_tot = (len(vv) - 1) // 3
        dx = vv[1 : 1 + m_tot] - half
        dy = vv[1 + m_tot : 1 + 2 * m_tot] - half
        qt = vv[1 + 2 * m_tot :]
        xs = np.concatenate([[half], half + dx, half - dy, half - dx, half + dy])
        ys = np.concatenate([[half], half + dy, half + dx, half - dy, half - dx])
        ths = np.concatenate([[vv[0]], qt, qt + 0.5 * math.pi, qt + math.pi, qt + 1.5 * math.pi])
        return xs, ys, ths

    def c4_obj(vv: np.ndarray, sd: float) -> tuple[float, np.ndarray]:
        xs, ys, ths = expand(vv)
        m_tot = (len(vv) - 1) // 3
        e, g = u.sat_penalty_grad(np.concatenate([xs, ys, ths]), L_probe, 1.0, 1.0, sd)
        gx, gy, gth = g[:n], g[n : 2 * n], g[2 * n :]
        gv = np.zeros_like(vv)
        gv[0] = gth[0]
        s0, s1, s2, s3 = 1, 1 + m_tot, 1 + 2 * m_tot, 1 + 3 * m_tot
        gv[1 : 1 + m_tot] = gx[s0:s1] + gy[s1:s2] - gx[s2:s3] - gy[s3:]
        gv[1 + m_tot : 1 + 2 * m_tot] = gy[s0:s1] - gx[s1:s2] - gy[s2:s3] + gx[s3:]
        gv[1 + 2 * m_tot :] = gth[s0:s1] + gth[s1:s2] + gth[s2:s3] + gth[s3:]
        return e, gv

    for sd, iters in [(0.86, 120), (0.91, 120), (0.95, 140), (0.985, 160), (1.0, 240)]:
        res = minimize(lambda vv, s=sd: c4_obj(vv, s), v, method="L-BFGS-B", jac=True, options={"maxiter": iters, "ftol": 1e-15, "gtol": 1e-12})
        v = res.x

    xs, ys, ths = expand(v)
    # Break C4 symmetry in final polish
    xs, ys, ths, pen = u.relax_lbfgs(xs, ys, ths, L_probe, maxiter=220, side=1.0)
    lp_L = None
    if pen < 1e-6:
        lx, ly, lp_val, lok = u.fixed_angle_lp_tighten(xs, ys, ths, rounds=5)
        if lok:
            lp_L = lp_val
            cert = u.verify_official(n, lx, ly, ths)
            if cert["ok"] and cert["L_float"] < L_ours:
                return {"m_core_half": m_core_half, "Q": Q, "seed": seed, "pen": float(pen), "lp_L": float(lp_L), "cert": cert}
    return {"m_core_half": m_core_half, "Q": Q, "seed": seed, "pen": float(pen), "lp_L": float(lp_L) if lp_L else None}


def test_n401_incumbent_carve(L_probe: float, L_ours: float) -> list[dict]:
    """Take n=401 incumbent, expand pocket boundary by removing 1 layer of uprights, and re-pack at non-45deg."""
    df = pd.read_csv(u.REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv")
    sub = df[df["n"] == 401]
    ix, iy, ith = sub["x"].to_numpy().copy(), sub["y"].to_numpy().copy(), sub["angle"].to_numpy().copy()
    hx = 0.5 * (np.abs(np.cos(ith)) + np.abs(np.sin(ith)))
    ix -= (ix - hx).min()
    iy -= (iy - hx).min()
    _, L0 = u.sat_max_overlap_and_bbox(ix, iy, ith)
    scale = L_probe / L0
    # Upright squares are those with sin(2*th) ~ 0
    up_mask = np.abs(np.sin(2 * ith)) < 0.05
    ux0, uy0 = ix[up_mask] * scale, iy[up_mask] * scale
    tx0, ty0 = ix[~up_mask] * scale, iy[~up_mask] * scale

    results = []
    # Find nearest upright neighbors to tilted centers so we can drop 0, 6, 12, 18 boundary uprights
    d2 = np.min((ux0[:, None] - tx0[None, :]) ** 2 + (uy0[:, None] - ty0[None, :]) ** 2, axis=1)
    ord_near = np.argsort(d2)

    for drop_k in (0, 8, 16, 24):
        keep_mask = np.ones(len(ux0), dtype=bool)
        if drop_k > 0:
            keep_mask[ord_near[:drop_k]] = False
        ux = ux0[keep_mask]
        uy = uy0[keep_mask]
        n_fill = 401 - len(ux)
        for seed in range(6):
            rng = np.random.default_rng(7000 + 100 * drop_k + seed)
            tx = np.empty(n_fill)
            ty = np.empty(n_fill)
            tth = np.empty(n_fill)
            tx[: len(tx0)] = tx0
            ty[: len(tx0)] = ty0
            if drop_k > 0:
                dx_drop = ux0[~keep_mask]
                dy_drop = uy0[~keep_mask]
                tx[len(tx0) :] = dx_drop
                ty[len(tx0) :] = dy_drop
            th_new = math.radians(rng.choice([20.0, 26.565, 30.0, 33.69, 36.87, 41.0, 45.0]))
            mode = seed % 3
            if mode == 0:
                tth[:] = th_new
            elif mode == 1:
                tth[:] = np.where(tx < ty, th_new, -th_new)
            else:
                tth[:] = math.radians(45.0) + rng.normal(0.0, 0.12, size=n_fill)
            tx += rng.normal(0.0, 0.05, size=n_fill)
            ty += rng.normal(0.0, 0.05, size=n_fill)

            x = np.concatenate([ux, tx])
            y = np.concatenate([uy, ty])
            th = np.concatenate([np.zeros(len(ux)), tth])
            free_m = np.zeros(401, dtype=bool)
            free_m[len(ux) :] = True
            for sd, it in [(0.88, 100), (0.94, 120), (0.985, 140), (1.0, 180)]:
                x, y, th, pen = u.relax_lbfgs(x, y, th, L_probe, free_mask=free_m, maxiter=it, side=sd)
            if pen < 0.03:
                x, y, th, pen = u.relax_lbfgs(x, y, th, L_probe, free_mask=None, maxiter=250, side=1.0)
            lp_L = None
            cert_ok = False
            if pen < 1e-6:
                lx, ly, lp_val, lok = u.fixed_angle_lp_tighten(x, y, th, rounds=5)
                if lok:
                    lp_L = lp_val
                    cert = u.verify_official(401, lx, ly, th)
                    cert_ok = bool(cert["ok"] and cert["L_float"] < L_ours)
                    if cert_ok:
                        results.append({"drop_k": drop_k, "seed": seed, "pen": float(pen), "lp_L": float(lp_L), "cert": cert})
                        continue
            results.append({"drop_k": drop_k, "seed": seed, "pen": float(pen), "lp_L": float(lp_L) if lp_L else None, "cert_ok": cert_ok})
    results.sort(key=lambda r: r["pen"])
    return results


def main():
    t0 = time.time()
    info = u.load_incumbent_and_frontier([401])
    L_ours = info[401]["L_ours"]
    L_front = info[401]["L_front"]
    L_probe = 20.5315

    # Build carve jobs
    jobs = []
    jid = 0
    for sx, sy in [(10, 10), (9, 11), (8, 12), (7, 13)]:
        for m_carve in (24, 36, 49, 64, 81):
            for angle_mode in ("uniform", "chevron"):
                jobs.append((jid, 401, L_probe, sx, sy, "diamond", m_carve, angle_mode, 100 + jid))
                jid += 1
        for m_carve in (30, 45, 60, 75):
            for angle_mode in ("uniform", "chevron"):
                jobs.append((jid, 401, L_probe, sx, sy, "diag_band", m_carve, angle_mode, 100 + jid))
                jid += 1
    for sx, sy in [(10, 10), (7, 7), (6, 6), (13, 13)]:
        for m_each in (16, 25, 36, 49):
            for slope in (1.0, 2.0, 0.5):
                jobs.append((jid, 401, L_probe, sx, sy, "nw_se_pockets", (m_each, slope), "chiral_nwse", 100 + jid))
                jid += 1

    print(f"Running {len(jobs)} Idea 5A carve+inflate jobs on n=401 (L_probe={L_probe})...", flush=True)
    with ProcessPoolExecutor(max_workers=7) as ex:
        res_5a = list(ex.map(carve_and_fill_worker, jobs))
    res_5a.sort(key=lambda r: r["pen"])
    print("Top 8 Idea 5A:", flush=True)
    for r in res_5a[:8]:
        print(" ", {k: v for k, v in r.items() if k != "cert"}, flush=True)

    c4_jobs = []
    for m_h in (6, 7, 8, 9):
        for s in range(6):
            c4_jobs.append((m_h, L_probe, L_ours, 5000 + 20 * m_h + s))
    print(f"\nRunning {len(c4_jobs)} Idea 5D C4 core+4pocket jobs...", flush=True)
    with ProcessPoolExecutor(max_workers=7) as ex:
        res_5d = list(ex.map(test_c4_core_ring_worker, c4_jobs))
    res_5d.sort(key=lambda r: r["pen"])
    print("Top 6 Idea 5D:", flush=True)
    for r in res_5d[:6]:
        print(" ", {k: v for k, v in r.items() if k != "cert"}, flush=True)

    print("\nRunning Idea 5B n=401 pocket expansion...", flush=True)
    res_5b = test_n401_incumbent_carve(L_probe, L_ours)
    print("Top 6 Idea 5B:", flush=True)
    for r in res_5b[:6]:
        print(" ", {k: v for k, v in r.items() if k != "cert"}, flush=True)

    # Save any certified win!
    for r in res_5a + res_5d + res_5b:
        if "cert" in r and r["cert"]["ok"] and r["cert"]["L_float"] < L_ours:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            rows = [dict(id=idx, n=401, **row) for idx, row in enumerate(r["cert"]["rows"])]
            out_csv = DATA_DIR / "win_n401.csv"
            pd.DataFrame(rows, columns=["id", "n", "square_id", "x", "y", "angle"]).to_csv(out_csv, index=False)
            print(f"SAVED WIN {out_csv}: L_lit={r['cert']['L_lit']} L_pd={r['cert']['L_pandas']}", flush=True)
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "top_5a": [{k: v for k, v in r.items() if k != "cert"} for r in res_5a[:20]],
        "top_5d": [{k: v for k, v in r.items() if k != "cert"} for r in res_5d[:15]],
        "top_5b": [{k: v for k, v in r.items() if k != "cert"} for r in res_5b[:15]],
        "seconds": round(time.time() - t0, 2),
    }
    (OUT_DIR / "idea5_sheared_cluster.json").write_text(json.dumps(summary, indent=2))
    print(f"\nIdea 5 finished in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
