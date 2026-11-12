"""Idea 7: Calibrated jogging sheared diagonal & off-diagonal bands (n=123/146/227/327 style) + MILP/L-BFGS polish."""
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
from idea6_milp_tilted import milp_upright_fill

OUT_DIR = Path(__file__).resolve().parents[1] / "docs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@nb.njit(cache=True)
def build_nw_se_staircases(L: float, a_se: int, a_nw: int) -> tuple[np.ndarray, np.ndarray]:
    n = a_se * (a_se + 1) // 2 + a_nw * (a_nw + 1) // 2
    ux = np.empty(n, dtype=np.float64)
    uy = np.empty(n, dtype=np.float64)
    idx = 0
    for i in range(a_se):
        for j in range(a_se - i):
            ux[idx] = L - 0.5 - i
            uy[idx] = 0.5 + j
            idx += 1
    for i in range(a_nw):
        for j in range(a_nw - i):
            ux[idx] = 0.5 + i
            uy[idx] = L - 0.5 - j
            idx += 1
    return ux, uy


@nb.njit(cache=True)
def check_col_ok(
    cx: float,
    cy: float,
    w: int,
    c: float,
    s: float,
    hx: float,
    shear: float,
    L: float,
    ux: np.ndarray,
    uy: np.ndarray,
    uc: np.ndarray,
    us: np.ndarray,
    uhx: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pc: np.ndarray,
    ps: np.ndarray,
    phx: np.ndarray,
    nt: int,
) -> bool:
    half_w = 0.5 * (w - 1)
    vx = -s + shear * c
    vy = c + shear * s
    for m in range(w):
        t = m - half_w
        qx = cx + t * vx
        qy = cy + t * vy
        if qx - hx < -1e-4 or qx + hx > L + 1e-4 or qy - hx < -1e-4 or qy + hx > L + 1e-4:
            return False
        if not u.sat_pair_clear_against_group(qx, qy, c, s, hx, ux, uy, uc, us, uhx, -1e-4):
            return False
        if nt > 0 and not u.sat_pair_clear_against_group(qx, qy, c, s, hx, px[:nt], py[:nt], pc[:nt], ps[:nt], phx[:nt], -1e-4):
            return False
    return True


@nb.njit(cache=True)
def pack_jogging_band(
    L: float,
    ux: np.ndarray,
    uy: np.ndarray,
    w_max: int,
    th: float,
    shear: float,
    u_lo: float,
    u_hi: float,
    v_lo: float,
    v_hi: float,
    v_mode: int,
    allow_taper: bool,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Pack sheared tilted columns along u45 with shear & v-jogging."""
    c = math.cos(th)
    s = math.sin(th)
    hx = 0.5 * (abs(c) + abs(s))
    vx = -s + shear * c
    vy = c + shear * s
    uc = np.ones(len(ux), dtype=np.float64)
    us = np.zeros(len(ux), dtype=np.float64)
    uhx = np.full(len(ux), 0.5, dtype=np.float64)

    px = np.empty(600, dtype=np.float64)
    py = np.empty(600, dtype=np.float64)
    pc = np.empty(600, dtype=np.float64)
    ps = np.empty(600, dtype=np.float64)
    phx = np.empty(600, dtype=np.float64)
    nt = 0
    inv_sq2 = 1.0 / math.sqrt(2.0)

    u_cur = u_lo
    w_min = 1 if allow_taper else max(1, w_max - 1)
    col_idx = 0
    v_mid = 0.5 * (v_lo + v_hi)
    v_prev = v_mid
    while u_cur <= u_hi and nt + w_max <= 600:
        placed = False
        for w in range(w_max, w_min - 1, -1):
            half_w = 0.5 * (w - 1)
            du = 0.0
            while du <= 0.55 and not placed:
                u_val = u_cur + du
                v_cnt = 0
                v_list = np.empty(160, dtype=np.float64)
                v_val = v_lo
                while v_val <= v_hi:
                    cx = (u_val + v_val) * inv_sq2
                    cy = (u_val - v_val) * inv_sq2
                    if check_col_ok(cx, cy, w, c, s, hx, shear, L, ux, uy, uc, us, uhx, px, py, pc, ps, phx, nt):
                        if v_cnt < 160:
                            v_list[v_cnt] = v_val
                            v_cnt += 1
                    v_val += 0.035
                if v_cnt > 0:
                    if v_mode == 0:
                        best_v = v_list[v_cnt // 2]
                    elif v_mode == 1:
                        best_v = v_list[0] if (col_idx % 2 == 0) else v_list[v_cnt - 1]
                    else:
                        # Jog upward in +0.075 steps unless near top, then drop
                        target_v = v_prev + 0.075 if v_prev + 0.075 <= v_list[v_cnt - 1] else v_list[0]
                        best_v = v_list[0]
                        best_d = 1e9
                        for vi in range(v_cnt):
                            d = abs(v_list[vi] - target_v)
                            if d < best_d:
                                best_d = d
                                best_v = v_list[vi]
                    v_prev = best_v
                    cx = (u_val + best_v) * inv_sq2
                    cy = (u_val - best_v) * inv_sq2
                    for m in range(w):
                        t = m - half_w
                        px[nt] = cx + t * vx
                        py[nt] = cy + t * vy
                        pc[nt] = c
                        ps[nt] = s
                        phx[nt] = hx
                        nt += 1
                    placed = True
                    col_idx += 1
                    u_cur = u_val + 0.85
                    break
                du += 0.03
            if placed:
                break
        if not placed:
            u_cur += 0.08
    return nt, px[:nt], py[:nt]


@nb.njit(cache=True)
def greedy_sw_ne_fill(
    L: float,
    ux: np.ndarray,
    uy: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    tc: np.ndarray,
    ts: np.ndarray,
    thx: np.ndarray,
) -> int:
    """Fast SW+NE corner upright count clearing NW/SE uprights and tilted band."""
    k = int(math.floor(L))
    uc = np.ones(len(ux), dtype=np.float64)
    us = np.zeros(len(ux), dtype=np.float64)
    uhx = np.full(len(ux), 0.5, dtype=np.float64)
    cnt = 0
    # SW (0.5+i, 0.5+j) & NE (L-0.5-i, L-0.5-j) up to small depth
    for corner in range(2):
        ox = 0.5 if corner == 0 else L - 0.5
        sx = 1.0 if corner == 0 else -1.0
        for d in range(min(k, 12)):
            added_layer = False
            for i in range(d + 1):
                j = d - i
                px = ox + sx * i
                py = ox + sx * j
                if not u.sat_pair_clear_against_group(px, py, 1.0, 0.0, 0.5, ux, uy, uc, us, uhx, -1e-4):
                    continue
                if not u.sat_pair_clear_against_group(px, py, 1.0, 0.0, 0.5, tx, ty, tc, ts, thx, -1e-4):
                    continue
                cnt += 1
                added_layer = True
            if not added_layer and d >= 2:
                break
    return cnt


def calibrate_227():
    L227 = 15.563625652
    ux, uy = build_nw_se_staircases(L227, 13, 12)
    best = 0
    for th_d in (33.88, 56.12):
        th = math.radians(th_d)
        for shear in (0.0, -0.04, 0.04):
            for v_mode in (0, 1, 2):
                for u_lo in (1.5, 2.2):
                    nt, tx, ty = pack_jogging_band(L227, ux, uy, 3, th, shear, u_lo, math.sqrt(2) * L227 - 1.2, -1.6, 1.0, v_mode, True)
                    if nt >= 48 or (shear == 0.0 and v_mode == 0 and u_lo == 1.5):
                        n_up_m, _, _ = milp_upright_fill(L227, tx, ty, np.full(nt, th), min_gap=-1e-4, time_limit=0.25)
                        tot = nt + n_up_m
                        if tot > best:
                            best = tot
                        print(f"Calib n=227 th={th_d} sh={shear} vm={v_mode} u0={u_lo}: nt={nt}, milp_up={n_up_m}, total={tot}", flush=True)
    return best


def scan_band_worker(args: tuple) -> dict:
    n_target, L_probe, a1_list, thetas = args
    k = math.isqrt(n_target - 1)
    top_candidates = []
    u_hi = math.sqrt(2.0) * L_probe - 0.7

    for a1 in a1_list:
        for a2 in range(max(2, a1 - 6), min(k, a1 + 2) + 1):
            if a1 + a2 > 2 * k:
                continue
            W45 = (2.0 * L_probe - a1 - a2 - 2.0) / math.sqrt(2.0)
            if W45 < 0.7 or W45 > 13.0:
                continue
            ux, uy = build_nw_se_staircases(L_probe, a1, a2)
            v_mid = (a2 - a1) / (2.0 * math.sqrt(2.0))
            v_lo = v_mid - 1.6
            v_hi = v_mid + 1.6
            w_approx = max(1, int(round(W45)))
            for w in range(max(1, w_approx - 1), w_approx + 2):
                for th in thetas:
                    c, s = math.cos(th), math.sin(th)
                    hx = 0.5 * (abs(c) + abs(s))
                    tc_buf = None
                    for shear in (0.0, -0.04, 0.04):
                        for v_mode in (0, 2):
                            nt, tx, ty = pack_jogging_band(L_probe, ux, uy, w, th, shear, 0.8, u_hi, v_lo, v_hi, v_mode, True)
                            if nt == 0:
                                continue
                            tc = np.full(nt, c)
                            ts = np.full(nt, s)
                            thx = np.full(nt, hx)
                            n_sw_ne = greedy_sw_ne_fill(L_probe, ux, uy, tx, ty, tc, ts, thx)
                            cap = len(ux) + nt + n_sw_ne
                            if len(top_candidates) < 8 or cap > top_candidates[-1][0]:
                                cfg = {
                                    "a1": a1, "a2": a2, "w": w,
                                    "th_deg": round(math.degrees(th), 2),
                                    "shear": shear, "v_mode": v_mode,
                                    "nt": int(nt), "n_nwse": len(ux), "n_swne": int(n_sw_ne), "fast_cap": int(cap),
                                }
                                top_candidates.append((cap, cfg, tx, ty, th))
                                top_candidates.sort(key=lambda z: -z[0])
                                top_candidates = top_candidates[:8]

    return {"top": [(c, cfg, tx, ty, th) for c, cfg, tx, ty, th in top_candidates]}


def main():
    t0 = time.time()
    calibrate_227()

    targets = [401, 442, 530]
    info = u.load_incumbent_and_frontier(targets)
    thetas = [math.radians(d) for d in np.linspace(18.0, 45.0, 19)]
    thetas += [math.radians(90.0 - d) for d in (24.5, 30.0, 33.88, 36.87, 39.65)]

    results = []
    for n in targets:
        k = math.isqrt(n - 1)
        L_ours = info[n]["L_ours"]
        L_front = info[n]["L_front"]
        L_probe = L_ours - 0.002
        t_n = time.time()

        a1_list = list(range(max(4, k - 14), k + 1))
        chunks = [a1_list[i::7] for i in range(7) if a1_list[i::7]]
        with ProcessPoolExecutor(max_workers=7) as ex:
            worker_res = list(ex.map(scan_band_worker, [(n, L_probe, c, thetas) for c in chunks]))

        merged_top = []
        for wr in worker_res:
            merged_top.extend(wr["top"])
        merged_top.sort(key=lambda z: -z[0])
        merged_top = merged_top[:16]

        best_milp_cap = 0
        best_detail = None
        for fast_cap, cfg, tx, ty, th in merged_top:
            tth = np.full(len(tx), th)
            n_up_m, mux, muy = milp_upright_fill(L_probe, tx, ty, tth, min_gap=0.0, time_limit=0.2)
            cap_m = len(tx) + n_up_m
            if cap_m > best_milp_cap:
                best_milp_cap = cap_m
                best_detail = {**cfg, "milp_up": n_up_m, "milp_cap": cap_m}
            if cap_m >= n:
                sx = np.concatenate([tx, mux])[:n]
                sy = np.concatenate([ty, muy])[:n]
                sth = np.concatenate([tth, np.zeros(n_up_m)])[:n]
                rx, ry, rth, rpen = u.relax_lbfgs(sx, sy, sth, L_probe, maxiter=250)
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
            "best": best_detail,
            "deficit": n - best_milp_cap,
            "seconds": round(time.time() - t_n, 2),
        }
        results.append(rec)
        print(f"n={n} (L_probe={L_probe:.6f}): milp_cap={best_milp_cap} (def={n-best_milp_cap}), best={best_detail} ({time.time()-t_n:.1f}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "idea7_sheared_band_free.json").write_text(json.dumps(results, indent=2))
    print(f"Done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
