"""Idea 8: Bounded-width non-45° wall/corner stepped pockets (n=598/908), expanded C4/C2 pockets, & 4-edge pinched frame."""
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

import numba as nb
import numpy as np
import pandas as pd

import utils as u
from idea6_milp_tilted import build_rect, milp_upright_fill

OUT_DIR = Path(__file__).resolve().parents[1] / "docs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@nb.njit(cache=True)
def _greedy_fill_order(
    px: np.ndarray,
    py: np.ndarray,
    pc: np.ndarray,
    cd: np.ndarray,
    order_mode: int,
    thr: float,
) -> int:
    n = len(px)
    if n == 0:
        return 0
    # Score determines placement priority across corners:
    # 0: symmetric i+j; 1: SW/NE priority; 2: NE first then SW; 3: NW/SE priority; 4: SW first then NE
    key = np.empty(n, dtype=np.float64)
    for i in range(n):
        c = pc[i]
        d = cd[i]
        if order_mode == 0:
            key[i] = float(d)
        elif order_mode == 1:
            bias = 0.0 if (c == 0 or c == 3) else 100.0
            key[i] = bias + d
        elif order_mode == 2:
            bias = 0.0 if c == 3 else (50.0 if c == 0 else 100.0)
            key[i] = bias + d
        elif order_mode == 3:
            bias = 0.0 if (c == 1 or c == 2) else 100.0
            key[i] = bias + d
        else:
            bias = 0.0 if c == 0 else (50.0 if c == 3 else 100.0)
            key[i] = bias + d

    idx = np.argsort(key)
    kx = np.empty(n, dtype=np.float64)
    ky = np.empty(n, dtype=np.float64)
    nk = 0
    for r in range(n):
        j = idx[r]
        xi, yi = px[j], py[j]
        ok = True
        for m in range(nk - 1, -1, -1):
            if abs(xi - kx[m]) < thr and abs(yi - ky[m]) < thr:
                ok = False
                break
        if ok:
            kx[nk] = xi
            ky[nk] = yi
            nk += 1
    return nk


def fast_upright_best5(
    L: float,
    tx: np.ndarray,
    ty: np.ndarray,
    tth: np.ndarray,
    min_gap: float = 0.0,
) -> int:
    """Fast (~0.3ms) 5-order corner greedy lower bound matching MILP on asymmetric splits."""
    k = int(math.floor(L))
    ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    ii_f = ii.ravel()
    jj_f = jj.ravel()
    diag = ii_f + jj_f

    if len(tx) > 0:
        tc = np.cos(tth)
        ts = np.sin(tth)
        thx = 0.5 * (np.abs(tc) + np.abs(ts))

    px_l, py_l, pc_l, pd_l = [], [], [], []
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
            keep = ~np.any(
                (np.abs(dx) < rad)
                & (np.abs(dy) < rad)
                & (np.abs(dx * tc[None, :] + dy * ts[None, :]) < rad)
                & (np.abs(-dx * ts[None, :] + dy * tc[None, :]) < rad),
                axis=1,
            )
        else:
            keep = np.ones(len(px), dtype=bool)
        px_l.append(px[keep])
        py_l.append(py[keep])
        pc_l.append(np.full(int(keep.sum()), corner, dtype=np.int32))
        pd_l.append(diag[keep])

    px_all = np.concatenate(px_l)
    py_all = np.concatenate(py_l)
    pc_all = np.concatenate(pc_l)
    pd_all = np.concatenate(pd_l)
    thr = 1.0 + min_gap
    best = 0
    for mode in range(5):
        cnt = _greedy_fill_order(px_all, py_all, pc_all, pd_all, mode, thr)
        if cnt > best:
            best = cnt
    return best


def build_stepped_wall_pocket(
    L: float,
    th: float,
    x_off: float,
    y_bot: float,
    y_top: float,
    w: int,
    slope: float,
    mirror_se: bool,
    adj_wall: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build n=598/908-style sec(th) tilted pocket along NW/left wall (with optional SE mirror or adj 90-th wall)."""
    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (abs(c) + abs(s))
    xs, ys, ths = [], [], []
    sec = 1.0 / c
    for col in range(w):
        cx = x_off + hx + col * c
        if cx + hx > L + 1e-6:
            break
        # Stepped interior/diagonal clip on bottom/top
        yb = max(hx, y_bot + slope * col * c)
        yt = min(L - hx, y_top - 0.15 * slope * col * c)
        if yb > yt:
            continue
        j0 = math.ceil((yb - col * s) * c - 1e-9)
        j1 = math.floor((yt - col * s) * c + 1e-9)
        for j in range(j0, j1 + 1):
            cy = j * sec + col * s
            if yb - 1e-6 <= cy <= yt + 1e-6:
                xs.append(cx)
                ys.append(cy)
                ths.append(th)

    if adj_wall and len(xs) > 0:
        # 90-th adjacent pocket along bottom wall (reflected across x=y)
        n1 = len(xs)
        for i in range(n1):
            xs.append(ys[i])
            ys.append(xs[i])
            ths.append(0.5 * math.pi - th)
    elif mirror_se and len(xs) > 0:
        # Rotational C2 mirror in SE corner
        n1 = len(xs)
        for i in range(n1):
            xs.append(L - xs[i])
            ys.append(L - ys[i])
            ths.append(th)

    if not xs:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), np.array(ths, dtype=np.float64)


def scan_8a_worker(args: tuple) -> list:
    n_target, L_probe, thetas = args
    top = []
    for th in thetas:
        for w in (2, 3, 4, 5, 6, 8, 10):
            for x_off in (0.0, 1.0, 2.0, 3.0):
                for y_bot in (0.0, 4.0, 8.0, 11.0):
                    for slope in (0.0, 0.35, 0.5, 1.0):
                        for mode in (0, 1, 2):
                            mirror_se = mode == 1
                            adj_wall = mode == 2
                            if adj_wall and (x_off > 1.0 or y_bot < 4.0):
                                continue
                            tx, ty, tth = build_stepped_wall_pocket(
                                L_probe, th, x_off, y_bot, L_probe, w, slope, mirror_se, adj_wall
                            )
                            if len(tx) == 0 or len(tx) >= n_target:
                                continue
                            if mirror_se or adj_wall:
                                pen, cL = u.sat_max_overlap_and_bbox(tx, ty, tth)
                                if pen > 1e-6 or cL > L_probe + 1e-6:
                                    continue
                            nup = fast_upright_best5(L_probe, tx, ty, tth, min_gap=0.0)
                            cap = len(tx) + nup
                            if len(top) < 8 or cap > top[-1][0]:
                                cfg = {
                                    "kind": "8A_wall",
                                    "th_deg": round(math.degrees(th), 2),
                                    "w": w, "x_off": x_off, "y_bot": y_bot, "slope": slope, "mode": mode,
                                    "nt": len(tx), "fast_up": nup, "fast_cap": cap,
                                }
                                top.append((cap, cfg, tx, ty, tth))
                                top.sort(key=lambda z: -z[0])
                                top = top[:8]
    return top


def scan_8b_worker(args: tuple) -> list:
    n_target, L_probe, configs = args
    k = math.isqrt(n_target - 1)
    half = 0.5 * L_probe
    top = []

    for cfg in configs:
        kind = cfg[0]
        if kind == "c4_frame":
            _, m1, m2, th, rx, ry, c_up = cfg
            b_x, b_y, b_t = build_rect(m1, m2, th, half + rx, half + ry)
            dx0, dy0 = b_x - half, b_y - half
            px_t = np.concatenate([half + dx0, half - dy0, half - dx0, half + dy0])
            py_t = np.concatenate([half + dy0, half + dx0, half - dy0, half - dx0])
            pt_t = np.concatenate([b_t, b_t + 0.5 * math.pi, b_t + math.pi, b_t + 1.5 * math.pi])
            if c_up > 0:
                ux_c, uy_c, ut_c = build_rect(c_up, c_up, 0.0, half, half)
                tx = np.concatenate([px_t, ux_c])
                ty = np.concatenate([py_t, uy_c])
                tth = np.concatenate([pt_t, ut_c])
            else:
                tx, ty, tth = px_t, py_t, pt_t
        elif kind == "c2_bicrystal":
            _, m1, m2, th, dx, dy = cfg
            x1, y1, t1 = build_rect(m1, m2, th, half - dx, half + dy)
            x2, y2, t2 = build_rect(m1, m2, -th, half + dx, half - dy)
            tx = np.concatenate([x1, x2])
            ty = np.concatenate([y1, y2])
            tth = np.concatenate([t1, t2])
        elif kind == "908_oct":
            _, th, rx, ry, u_rad, v_rad, d_cut = cfg
            c, s = math.cos(th), math.sin(th)
            sec = 1.0 / c
            px_l, py_l = [], []
            for col in range(-u_rad, u_rad + 1):
                cx = rx + col * c
                for row in range(-v_rad - 4, v_rad + 5):
                    cy = ry + row * sec + col * s
                    if abs(col) + abs((cy - ry) * c) <= d_cut + 1e-6 and abs(cy - ry) <= v_rad + 0.5:
                        px_l.append(cx)
                        py_l.append(cy)
            if not px_l:
                continue
            p1x = np.array(px_l, dtype=np.float64)
            p1y = np.array(py_l, dtype=np.float64)
            tx = np.concatenate([p1x, L_probe - p1x])
            ty = np.concatenate([p1y, L_probe - p1y])
            tth = np.full(len(tx), th, dtype=np.float64)
        else:
            continue

        hx = 0.5 * (np.abs(np.cos(tth)) + np.abs(np.sin(tth)))
        if (tx - hx).min() < -1e-6 or (tx + hx).max() > L_probe + 1e-6:
            continue
        if (ty - hx).min() < -1e-6 or (ty + hx).max() > L_probe + 1e-6:
            continue
        pen, cL = u.sat_max_overlap_and_bbox(tx, ty, tth)
        if pen > 1e-6 or cL > L_probe + 1e-6:
            continue

        nup = fast_upright_best5(L_probe, tx, ty, tth, min_gap=0.0)
        cap = len(tx) + nup
        if len(top) < 8 or cap > top[-1][0]:
            mcfg = {"kind": kind, "params": [round(float(v), 3) if isinstance(v, float) else v for v in cfg[1:]], "nt": len(tx), "fast_up": nup, "fast_cap": cap}
            top.append((cap, mcfg, tx, ty, tth))
            top.sort(key=lambda z: -z[0])
            top = top[:8]
    return top


def main():
    t0 = time.time()
    targets = [401, 442, 530]
    info = u.load_incumbent_and_frontier(targets)
    thetas_8a = [math.radians(d) for d in (14.0, 16.5, 18.435, 19.5, 22.0, 24.5, 26.565, 30.0, 33.88, 36.87)]
    thetas_8b = [math.radians(d) for d in (15.0, 18.435, 20.0, 24.5, 26.565, 30.0, 33.69, 36.87, 41.0, 45.0)]

    results = []
    for n in targets:
        k = math.isqrt(n - 1)
        L_ours = info[n]["L_ours"]
        L_front = info[n]["L_front"]
        L_probe = L_ours - 0.001
        half = 0.5 * L_probe
        t_n = time.time()

        chunks_8a = [thetas_8a[i::7] for i in range(7) if thetas_8a[i::7]]
        with ProcessPoolExecutor(max_workers=7) as ex:
            res_8a = list(ex.map(scan_8a_worker, [(n, L_probe, c) for c in chunks_8a]))
        top_8a = sorted([it for sub in res_8a for it in sub], key=lambda z: -z[0])[:10]

        cfgs_8b = []
        # Expanded C4 without centered core (near corners/interior) + with pinched odd/even c_up core
        for m1 in range(1, 9):
            for m2 in range(1, min(6, m1 + 1)):
                for th in thetas_8b:
                    for rx in np.arange(2.0, half - 1.0, 1.5):
                        for ry in np.arange(1.0, half - 1.0, 1.5):
                            cfgs_8b.append(("c4_frame", m1, m2, th, float(rx), float(ry), 0))
        # Pinched 4-edge frame enclosing c_up x c_up upright core
        for c_up in range(max(4, k - 9), k - 1):
            margin = half - 0.5 * c_up
            for m1 in (1, 2, 3, 4):
                for m2 in range(3, c_up + 2, 2):
                    for th in [math.radians(d) for d in (16.5, 18.435, 22.5, 26.565, 30.0, 36.87)]:
                        for rx in (0.5 * c_up + 0.5 * m1 + 0.2, half - 0.5 * m1 - 0.1):
                            for ry in (0.0, 1.5, -1.5):
                                if rx > 0.5 * c_up:
                                    cfgs_8b.append(("c4_frame", m1, m2, th, float(rx), float(ry), c_up))
        # C2 +-theta bicrystal
        for m1 in range(2, 7):
            for m2 in (m1, m1 - 1):
                for th in [math.radians(d) for d in (18.435, 24.5, 26.565, 30.0, 36.87)]:
                    for dx in (2.0, 4.0, 6.0):
                        for dy in (2.0, 4.0, 6.0):
                            cfgs_8b.append(("c2_bicrystal", m1, m2, th, dx, dy))
        # 8C: n=908-style NW/SE C2 stepped sec(th) octagonal pockets
        for th in [math.radians(d) for d in (18.435, 22.0, 24.5, 26.565, 30.0, 33.88)]:
            for rx in (4.0, 5.5, 7.0, 8.5):
                for ry in (L_probe - 4.0, L_probe - 6.0, L_probe - 8.0):
                    for u_rad in (3, 4, 5):
                        for v_rad in (3, 4, 5):
                            cfgs_8b.append(("908_oct", th, rx, ry, u_rad, v_rad, float(u_rad + v_rad) * 0.75))

        chunks_8b = [cfgs_8b[i::7] for i in range(7) if cfgs_8b[i::7]]
        with ProcessPoolExecutor(max_workers=7) as ex:
            res_8b = list(ex.map(scan_8b_worker, [(n, L_probe, c) for c in chunks_8b]))
        top_8b = sorted([it for sub in res_8b for it in sub], key=lambda z: -z[0])[:12]

        best_cap = 0
        best_cfg = None
        for _, cfg, tx, ty, tth in top_8a + top_8b:
            nup_m, ux, uy = milp_upright_fill(L_probe, tx, ty, tth, min_gap=0.0, time_limit=0.25)
            cap = len(tx) + nup_m
            if cap > best_cap:
                best_cap = cap
                best_cfg = {**cfg, "milp_up": nup_m, "cap": cap}
            if cap >= n:
                sx = np.concatenate([tx, ux])[:n]
                sy = np.concatenate([ty, uy])[:n]
                sth = np.concatenate([tth, np.zeros(nup_m)])[:n]
                rx, ry, rth, rpen = u.relax_lbfgs(sx, sy, sth, L_probe, maxiter=300)
                lx, ly, lp_val, lok = u.fixed_angle_lp_tighten(rx, ry, rth, rounds=5)
                if lok and lp_val < L_ours:
                    cert = u.verify_official(n, lx, ly, rth)
                    if cert["ok"]:
                        DATA_DIR.mkdir(parents=True, exist_ok=True)
                        rows = [dict(id=idx, n=n, **row) for idx, row in enumerate(cert["rows"])]
                        out_csv = DATA_DIR / f"win_n{n}.csv"
                        pd.DataFrame(rows, columns=["id", "n", "square_id", "x", "y", "angle"]).to_csv(out_csv, index=False)
                        print(f"*** WIN SAVED {out_csv}: L_lit={cert['L_lit']} L_pd={cert['L_pandas']} vs L_ours={L_ours:.9f}", flush=True)

        rec = {
            "n": n,
            "L_ours": L_ours,
            "L_front": L_front,
            "L_probe": L_probe,
            "best": best_cfg,
            "deficit": n - best_cap,
            "seconds": round(time.time() - t_n, 2),
        }
        results.append(rec)
        print(f"n={n} (L_probe={L_probe:.6f}): milp_cap={best_cap} (def={n-best_cap}) best={best_cfg} ({rec['seconds']}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "idea8_pockets_frame.json").write_text(json.dumps(results, indent=2))
    print(f"Idea 8 finished in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
