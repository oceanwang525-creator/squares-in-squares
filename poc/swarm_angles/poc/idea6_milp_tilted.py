"""Idea 6: Chamfered rotated/sheared lattices & asymmetric multi-blocks + exact MILP 4-corner upright fill."""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

import utils as u

OUT_DIR = Path(__file__).resolve().parents[1] / "docs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def milp_upright_fill(
    L: float,
    tx: np.ndarray,
    ty: np.ndarray,
    tth: np.ndarray,
    min_gap: float = 1e-7,
    time_limit: float = 1.5,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Find the exact maximum independent set of 4-corner-aligned upright 1x1 squares clearing tilted T."""
    k = int(math.floor(L))
    tc = np.cos(tth)
    ts = np.sin(tth)
    thx = 0.5 * (np.abs(tc) + np.abs(ts))

    cands_x, cands_y, cands_c = [], [], []
    ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    ii_f = ii.ravel()
    jj_f = jj.ravel()

    for corner, (ox, sx, oy, sy) in enumerate([
        (0.5, 1.0, 0.5, 1.0),
        (L - 0.5, -1.0, 0.5, 1.0),
        (0.5, 1.0, L - 0.5, -1.0),
        (L - 0.5, -1.0, L - 0.5, -1.0),
    ]):
        px = ox + sx * ii_f
        py = oy + sy * jj_f
        if len(tx) > 0:
            dx = px[:, None] - tx[None, :]
            dy = py[:, None] - ty[None, :]
            rad = 0.5 + thx[None, :] + min_gap
            p1 = np.abs(dx) < rad
            p2 = np.abs(dy) < rad
            p3 = np.abs(dx * tc[None, :] + dy * ts[None, :]) < rad
            p4 = np.abs(-dx * ts[None, :] + dy * tc[None, :]) < rad
            keep = ~np.any(p1 & p2 & p3 & p4, axis=1)
        else:
            keep = np.ones(len(px), dtype=bool)
        cands_x.append(px[keep])
        cands_y.append(py[keep])
        cands_c.append(np.full(int(keep.sum()), corner, dtype=np.int32))

    cx = np.concatenate(cands_x)
    cy = np.concatenate(cands_y)
    cc = np.concatenate(cands_c)
    m = len(cx)
    if m == 0:
        return 0, np.empty(0), np.empty(0)

    tree = cKDTree(np.column_stack([cx, cy]))
    pairs = tree.query_pairs(r=1.4143)
    real_pairs = [
        (i, j)
        for i, j in pairs
        if cc[i] != cc[j] and abs(cx[i] - cx[j]) < 1.0 + min_gap and abs(cy[i] - cy[j]) < 1.0 + min_gap
    ]

    if not real_pairs:
        return m, cx, cy

    rows = []
    cols = []
    for r_idx, (i, j) in enumerate(real_pairs):
        rows.extend([r_idx, r_idx])
        cols.extend([i, j])
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(real_pairs), m))
    res = milp(
        c=-np.ones(m),
        integrality=np.ones(m),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(A, -np.inf, 1.0),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0015},
    )
    if not res.success and res.x is None:
        return 0, np.empty(0), np.empty(0)
    sel = res.x > 0.5
    return int(sel.sum()), cx[sel], cy[sel]


def fast_upright_count(
    L: float,
    tx: np.ndarray,
    ty: np.ndarray,
    tth: np.ndarray,
    min_gap: float = -0.003,
) -> int:
    """Fast 4-corner layer-by-layer greedy independent-set lower bound (~0.3ms)."""
    k = int(math.floor(L))
    tc = np.cos(tth)
    ts = np.sin(tth)
    thx = 0.5 * (np.abs(tc) + np.abs(ts))
    ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    ii_f = ii.ravel()
    jj_f = jj.ravel()
    diag = ii_f + jj_f

    px_all, py_all, d_all = [], [], []
    for ox, sx, oy, sy in [
        (0.5, 1.0, 0.5, 1.0),
        (L - 0.5, -1.0, 0.5, 1.0),
        (0.5, 1.0, L - 0.5, -1.0),
        (L - 0.5, -1.0, L - 0.5, -1.0),
    ]:
        px = ox + sx * ii_f
        py = oy + sy * jj_f
        if len(tx) > 0:
            dx = px[:, None] - tx[None, :]
            dy = py[:, None] - ty[None, :]
            rad = 0.5 + thx[None, :] + min_gap
            keep = ~np.any(
                (np.abs(dx) < rad)
                & (np.abs(dy) < rad)
                & (np.abs(dx * tc[None, :] + dy * ts[None, :]) < rad)
                & (np.abs(-dx * ts[None, :] + dy * tc[None, :]) < rad),
                axis=1,
            )
        else:
            keep = np.ones(len(px), dtype=bool)
        px_all.append(px[keep])
        py_all.append(py[keep])
        d_all.append(diag[keep])

    cx = np.concatenate(px_all)
    cy = np.concatenate(py_all)
    cd = np.concatenate(d_all)
    if len(cx) == 0:
        return 0
    order = np.argsort(cd, kind="stable")
    cx, cy = cx[order], cy[order]
    kept_x = np.empty(len(cx), dtype=np.float64)
    kept_y = np.empty(len(cx), dtype=np.float64)
    nk = 0
    thr = 1.0 + min_gap
    for i in range(len(cx)):
        xi, yi = cx[i], cy[i]
        ok = True
        for m in range(nk - 1, -1, -1):
            if abs(xi - kept_x[m]) < thr and abs(yi - kept_y[m]) < thr:
                ok = False
                break
        if ok:
            kept_x[nk] = xi
            kept_y[nk] = yi
            nk += 1
    return nk


def build_chamfered_lattice(
    L: float,
    th: float,
    cx: float,
    cy: float,
    U: float,
    V: float,
    D_cut: float,
    shear: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate rotated (or sheared) 1x1 lattice squares inside [0,L]^2 & octagon |u|<=U, |v|<=V, |u|+|v|<=D_cut."""
    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (abs(c) + abs(s))
    lim = int(math.ceil(L)) + 2
    xs, ys = [], []
    # Note: v1 = (c, s), v2 = (-s + shear*c, c + shear*s) clears if |shear| <= 1 and ... wait:
    # Actually (-s, c) + lam*(c, s) has proj onto (c,s) = lam, onto (-s,c) = 1.0 -> ALWAYS non-overlapping!
    for i in range(-lim, lim + 1):
        for j in range(-lim, lim + 1):
            u_val = float(i)
            v_val = float(j)
            if abs(u_val) > U + 1e-9 or abs(v_val) > V + 1e-9:
                continue
            if abs(u_val) + abs(v_val) > D_cut + 1e-9:
                continue
            px = cx + u_val * c + v_val * (-s + shear * c)
            py = cy + u_val * s + v_val * (c + shear * s)
            if px - hx < -1e-6 or px + hx > L + 1e-6 or py - hx < -1e-6 or py + hx > L + 1e-6:
                continue
            xs.append(px)
            ys.append(py)
    if not xs:
        return np.empty(0), np.empty(0), np.empty(0)
    arr_x = np.array(xs, dtype=np.float64)
    arr_y = np.array(ys, dtype=np.float64)
    return arr_x, arr_y, np.full(len(arr_x), th, dtype=np.float64)


def build_rect(m1: int, m2: int, th: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c, s = math.cos(th), math.sin(th)
    xs = np.empty(m1 * m2, dtype=np.float64)
    ys = np.empty(m1 * m2, dtype=np.float64)
    idx = 0
    for i in range(m1):
        ui = i - 0.5 * (m1 - 1)
        for j in range(m2):
            vj = j - 0.5 * (m2 - 1)
            xs[idx] = cx + ui * c - vj * s
            ys[idx] = cy + ui * s + vj * c
            idx += 1
    return xs, ys, np.full(m1 * m2, th, dtype=np.float64)


def scan_chamfer_worker(args: tuple) -> dict:
    n_target, L_probe, thetas = args
    k = math.isqrt(n_target - 1)
    top = []
    half = 0.5 * L_probe

    shifts = [(0.0, 0.0), (0.5, 0.0), (0.25, 0.25), (0.5, 0.5)]
    for th in thetas:
        for shear in (0.0, 0.08, -0.08, 0.16):
            for sx, sy in shifts:
                for half_u in range(3, k // 2 + 4):
                    for U in (float(half_u), float(half_u) - 0.5):
                        for V in (U, max(1.5, U - 2.0)):
                            for d_slack in (U + V, 0.8 * (U + V), 0.65 * (U + V)):
                                tx, ty, tth = build_chamfered_lattice(L_probe, th, half + sx, half + sy, U, V, d_slack, shear)
                                if len(tx) == 0 or len(tx) >= n_target:
                                    continue
                                n_up = fast_upright_count(L_probe, tx, ty, tth, min_gap=-0.003)
                                cap = len(tx) + n_up
                                if len(top) < 8 or cap > top[-1][0]:
                                    cfg = {
                                        "kind": "chamfer",
                                        "th_deg": round(math.degrees(th), 3),
                                        "shear": shear,
                                        "shift": (sx, sy),
                                        "U": U,
                                        "V": V,
                                        "d_cut": round(d_slack, 2),
                                        "n_tilt": len(tx),
                                        "fast_up": n_up,
                                        "fast_cap": cap,
                                    }
                                    top.append((cap, cfg, tx, ty, tth))
                                    top.sort(key=lambda z: -z[0])
                                    top = top[:8]
    return {"top": top}


def scan_multiblock_worker(args: tuple) -> dict:
    n_target, L_probe, configs = args
    top = []
    half = 0.5 * L_probe

    for cfg in configs:
        kind = cfg[0]
        if kind == "two":
            _, m1, m2, th1, th2, dx, dy = cfg
            x1, y1, t1 = build_rect(m1, m2, th1, half - dx, half + dy)
            x2, y2, t2 = build_rect(m1, m2, th2, half + dx, half - dy)
            tx = np.concatenate([x1, x2])
            ty = np.concatenate([y1, y2])
            tth = np.concatenate([t1, t2])
        elif kind == "four_c4":
            _, m1, m2, th, rx, ry = cfg
            b_x, b_y, b_t = build_rect(m1, m2, th, half + rx, half + ry)
            dx0, dy0 = b_x - half, b_y - half
            tx = np.concatenate([half + dx0, half - dy0, half - dx0, half + dy0])
            ty = np.concatenate([half + dy0, half + dx0, half - dy0, half - dx0])
            tth = np.concatenate([b_t, b_t + 0.5 * math.pi, b_t + math.pi, b_t + 1.5 * math.pi])
        else:
            continue

        hx = 0.5 * (np.abs(np.cos(tth)) + np.abs(np.sin(tth)))
        if (tx - hx).min() < -1e-6 or (tx + hx).max() > L_probe + 1e-6:
            continue
        if (ty - hx).min() < -1e-6 or (ty + hx).max() > L_probe + 1e-6:
            continue
        pen, cL = u.sat_max_overlap_and_bbox(tx, ty, tth)
        if pen > 1e-5 or cL > L_probe + 1e-5:
            continue

        n_up = fast_upright_count(L_probe, tx, ty, tth, min_gap=-0.003)
        cap = len(tx) + n_up
        if len(top) < 8 or cap > top[-1][0]:
            mcfg = {"kind": kind, "params": [round(float(v), 3) if isinstance(v, float) else v for v in cfg[1:]], "n_tilt": len(tx), "fast_up": n_up, "fast_cap": cap}
            top.append((cap, mcfg, tx, ty, tth))
            top.sort(key=lambda z: -z[0])
            top = top[:8]
    return {"top": top}


def run_calibration():
    df = pd.read_csv(u.REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv")
    for n_cal in (50, 401, 598, 626):
        sub = df[df["n"] == n_cal]
        x, y, th = sub["x"].to_numpy().copy(), sub["y"].to_numpy().copy(), sub["angle"].to_numpy().copy()
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        x -= (x - hx).min()
        y -= (y - hx).min()
        L = max((x + hx).max(), (y + hx).max())
        mask_t = np.abs(np.sin(2 * th)) > 0.05
        n_up, _, _ = milp_upright_fill(L, x[mask_t], y[mask_t], th[mask_t], min_gap=-1e-6)
        n_fast = fast_upright_count(L, x[mask_t], y[mask_t], th[mask_t], min_gap=-1e-6)
        print(f"Calib n={n_cal} (L={L:.6f}): tilted={mask_t.sum()}, inc_up={(~mask_t).sum()}, fast_up={n_fast}, milp_up={n_up}, total={mask_t.sum()+n_up}", flush=True)


def eval_milp_top(n: int, L_probe: float, cands: list) -> dict:
    best_cap = 0
    best_cfg = None
    best_state = None
    for _, cfg, tx, ty, tth in cands:
        n_up, ux, uy = milp_upright_fill(L_probe, tx, ty, tth, min_gap=-0.002, time_limit=0.18)
        cap = len(tx) + n_up
        if cap > best_cap:
            best_cap = cap
            best_cfg = dict(cfg, n_up=n_up, cap=cap)
            best_state = (
                np.concatenate([tx, ux]),
                np.concatenate([ty, uy]),
                np.concatenate([tth, np.zeros(n_up)]),
            )
    return {"best_cap": best_cap, "best_cfg": best_cfg, "best_state": best_state}


def main():
    t0 = time.time()
    run_calibration()

    targets = [401, 442, 530]
    info = u.load_incumbent_and_frontier(targets)
    thetas = [math.radians(d) for d in [15.0, 18.0, 20.0, 23.2, 24.5, 26.565, 30.0, 33.69, 36.87, 40.0, 42.0, 45.0]]

    results = []
    for n in targets:
        k = math.isqrt(n - 1)
        L_ours = info[n]["L_ours"]
        L_front = info[n]["L_front"]
        L_probe = L_ours - 0.002
        t_n = time.time()

        # 6A: Chamfered cores (fast proxy -> top 18 MILP)
        chunks_th = [thetas[i::7] for i in range(7) if thetas[i::7]]
        with ProcessPoolExecutor(max_workers=7) as ex:
            res_6a = list(ex.map(scan_chamfer_worker, [(n, L_probe, c) for c in chunks_th]))
        top_6a = sorted([item for r in res_6a for item in r["top"]], key=lambda z: -z[0])[:18]
        best_6a = eval_milp_top(n, L_probe, top_6a)

        # 6B/6C: Multi-block (2-block & 4-block C4, fast proxy -> top 18 MILP)
        mb_cfgs = []
        for m1 in range(3, 8):
            for m2 in (m1, m1 - 1):
                for th in [math.radians(d) for d in (22.5, 26.565, 30.0, 33.69, 36.87, 41.0, 45.0)]:
                    for dx in (2.5, 4.0, 5.5, 6.0, 7.0):
                        for dy in (2.5, 3.0, 4.5, 6.0):
                            mb_cfgs.append(("two", m1, m2, th, th, dx, dy))
                            mb_cfgs.append(("two", m1, m2, th, 0.5 * math.pi - th, dx, dy))
        for m1 in range(2, 6):
            for th in [math.radians(d) for d in (18.435, 26.565, 30.0, 36.87, 45.0)]:
                for rx in (2.5, 4.0, 5.5, 7.0):
                    for ry in (1.5, 3.0, 5.0):
                        mb_cfgs.append(("four_c4", m1, m1, th, rx, ry))

        chunks_mb = [mb_cfgs[i::7] for i in range(7) if mb_cfgs[i::7]]
        with ProcessPoolExecutor(max_workers=7) as ex:
            res_6b = list(ex.map(scan_multiblock_worker, [(n, L_probe, c) for c in chunks_mb]))
        top_6b = sorted([item for r in res_6b for item in r["top"]], key=lambda z: -z[0])[:18]
        best_6b = eval_milp_top(n, L_probe, top_6b)

        best = best_6a if best_6a["best_cap"] >= best_6b["best_cap"] else best_6b
        rec = {
            "n": n,
            "L_ours": L_ours,
            "L_front": L_front,
            "L_probe": L_probe,
            "best_6a": best_6a["best_cfg"],
            "best_6b": best_6b["best_cfg"],
            "deficit": n - best["best_cap"],
            "seconds": round(time.time() - t_n, 2),
        }
        print(f"n={n} (L_probe={L_probe:.6f}): best_cap={best['best_cap']} (def={n-best['best_cap']}) in {rec['seconds']}s", flush=True)
        print(f"  6A={best_6a['best_cfg']}", flush=True)
        print(f"  6B={best_6b['best_cfg']}", flush=True)

        if best["best_cap"] >= n and best["best_state"] is not None:
            sx, sy, sth = best["best_state"]
            sx, sy, sth = sx[:n], sy[:n], sth[:n]
            rx, ry, rth, rpen = u.relax_lbfgs(sx, sy, sth, L_probe, maxiter=300)
            lx, ly, lp_val, lok = u.fixed_angle_lp_tighten(rx, ry, rth, rounds=5)
            rec["polish"] = {"rpen": float(rpen), "lp_ok": lok, "lp_L": float(lp_val) if lok else None}
            if lok and lp_val < L_ours:
                cert = u.verify_official(n, lx, ly, rth)
                rec["cert_ok"] = cert["ok"]
                if cert["ok"]:
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    rows = [dict(id=idx, n=n, **row) for idx, row in enumerate(cert["rows"])]
                    out_csv = DATA_DIR / f"win_n{n}.csv"
                    pd.DataFrame(rows, columns=["id", "n", "square_id", "x", "y", "angle"]).to_csv(out_csv, index=False)
                    print(f"  *** WIN SAVED {out_csv}: L_lit={cert['L_lit']} L_pd={cert['L_pandas']} vs L_ours={L_ours:.9f}", flush=True)
        results.append(rec)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "idea6_milp_tilted.json").write_text(json.dumps(results, indent=2))
    print(f"\nIdea 6 finished in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
