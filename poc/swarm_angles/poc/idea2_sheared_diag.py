"""Idea 2: Sheared off-45deg diagonal band (generalized n=123/227/258) inside upright staircase channels."""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numba as nb
import numpy as np

import utils as u

OUT_DIR = Path(__file__).resolve().parents[1] / "docs"


def build_staircases(L: float, a1: int, a2: int, b1: int, b2: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ux, uy) for 4 upright staircases: SE(a1), NW(a2), SW(b1), NE(b2)."""
    pts = []
    # SW (b1)
    for i in range(b1):
        for j in range(b1 - i):
            pts.append((0.5 + i, 0.5 + j))
    # NE (b2)
    for i in range(b2):
        for j in range(b2 - i):
            pts.append((L - 0.5 - i, L - 0.5 - j))
    # SE (a1)
    for i in range(a1):
        for j in range(a1 - i):
            pts.append((L - 0.5 - i, 0.5 + j))
    # NW (a2)
    for i in range(a2):
        for j in range(a2 - i):
            pts.append((0.5 + i, L - 0.5 - j))
    arr = np.array(pts, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


@nb.njit(cache=True)
def check_col_valid(
    cx: float,
    cy: float,
    w: int,
    c: float,
    s: float,
    hx: float,
    L: float,
    ux: np.ndarray,
    uy: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    n_prev: int,
    pc: float,
    ps: float,
    phx: float,
) -> bool:
    """Check if a w-square column centered at (cx, cy) along (-s, c) clears walls, uprights, and prev squares."""
    # Column squares at (cx + t*(-s), cy + t*c) for t = m - (w-1)/2
    # Quick bounding box vs [0, L]^2
    half_w = 0.5 * (w - 1)
    minx = cx - half_w * abs(s) - hx
    maxx = cx + half_w * abs(s) + hx
    miny = cy - half_w * abs(c) - hx
    maxy = cy + half_w * abs(c) + hx
    if minx < -1e-4 or maxx > L + 1e-4 or miny < -1e-4 or maxy > L + 1e-4:
        return False

    for m in range(w):
        t = m - half_w
        qx = cx - t * s
        qy = cy + t * c
        # Check against upright squares
        for i in range(len(ux)):
            dx = qx - ux[i]
            if abs(dx) >= 0.5 + hx - 1e-4:
                continue
            dy = qy - uy[i]
            if abs(dy) >= 0.5 + hx - 1e-4:
                continue
            rad = 0.5 + hx - 1e-4
            if abs(dx * c + dy * s) < rad and abs(-dx * s + dy * c) < rad:
                return False
        # Check against previously placed tilted squares (all at same angle th -> pc=c, ps=s!)
        for i in range(n_prev):
            dx = qx - px[i]
            if abs(dx) >= 2.0 * hx - 1e-4:
                continue
            dy = qy - py[i]
            if abs(dy) >= 2.0 * hx - 1e-4:
                continue
            if abs(dx * c + dy * s) < 1.0 - 1e-4 and abs(-dx * s + dy * c) < 1.0 - 1e-4:
                return False
    return True


@nb.njit(cache=True)
def greedy_pack_channel(
    L: float,
    ux: np.ndarray,
    uy: np.ndarray,
    w: int,
    th: float,
    u_start: float,
    u_end: float,
    v_lo: float,
    v_hi: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Greedily pack columns of w (or reduced w near ends) tilted squares along u45 in [u_start, u_end]."""
    c = math.cos(th)
    s = math.sin(th)
    hx = 0.5 * (abs(c) + abs(s))
    max_sq = 400
    px = np.empty(max_sq, dtype=np.float64)
    py = np.empty(max_sq, dtype=np.float64)
    n_prev = 0

    inv_sq2 = 1.0 / math.sqrt(2.0)
    u = u_start
    while u <= u_end and n_prev + w <= max_sq:
        placed = False
        # Try placing full w-column at earliest u, scanning v
        v = v_lo
        while v <= v_hi:
            cx = (u + v) * inv_sq2
            cy = (u - v) * inv_sq2
            if check_col_valid(cx, cy, w, c, s, hx, L, ux, uy, px, py, n_prev, c, s, hx):
                half_w = 0.5 * (w - 1)
                for m in range(w):
                    t = m - half_w
                    px[n_prev] = cx - t * s
                    py[n_prev] = cy + t * c
                    n_prev += 1
                placed = True
                break
            v += 0.04
        if placed:
            u += 0.85
        else:
            u += 0.06
    return n_prev, px[:n_prev], py[:n_prev]


def scan_diag_worker(args: tuple) -> dict:
    n_target, L_probe, a1_list, thetas = args
    k = math.isqrt(n_target - 1)
    best_cap = 0
    best_cfg = None
    for a1 in a1_list:
        for a2 in (a1, a1 + 1):
            if a1 + a2 > 2 * k:
                continue
            # Estimate channel width W45
            W45 = (2.0 * L_probe - a1 - a2 - 2.0) / math.sqrt(2.0)
            if W45 < 0.8 or W45 > 16.0:
                continue
            w_approx = int(round(W45))
            for w in range(max(1, w_approx - 1), w_approx + 3):
                # End staircases b1, b2 <= L - max(a1, a2) - 1
                b_est = int(math.floor(w / math.sqrt(2.0)))
                for b1 in range(0, b_est + 2):
                    for b2 in (b1, b1 + 1):
                        if b1 + a1 > L_probe - 0.5 or b2 + a2 > L_probe - 0.5 or b1 + a2 > L_probe - 0.5 or b2 + a1 > L_probe - 0.5:
                            continue
                        ux, uy = build_staircases(L_probe, a1, a2, b1, b2)
                        # Quick check if staircases already overlap (can't if b+a <= floor(L))
                        u_start = (b1 + 1) / math.sqrt(2.0) + 0.4 if b1 > 0 else 0.5
                        u_end = math.sqrt(2.0) * L_probe - ((b2 + 1) / math.sqrt(2.0) + 0.4 if b2 > 0 else 0.5)
                        v_mid = (a2 - a1) / (2.0 * math.sqrt(2.0))
                        v_lo = v_mid - 1.0
                        v_hi = v_mid + 1.0
                        for th in thetas:
                            n_tilt, px, py = greedy_pack_channel(L_probe, ux, uy, w, th, u_start, u_end, v_lo, v_hi)
                            cap = len(ux) + n_tilt
                            if cap > best_cap:
                                best_cap = cap
                                best_cfg = {
                                    "a1": a1,
                                    "a2": a2,
                                    "b1": b1,
                                    "b2": b2,
                                    "w": w,
                                    "th_deg": round(math.degrees(th), 3),
                                    "W45": round(W45, 4),
                                    "n_up": len(ux),
                                    "n_tilt": int(n_tilt),
                                    "cap": int(cap),
                                }
    return {"best_cap": best_cap, "best_cfg": best_cfg}


def main():
    # Calibration on n=227 (L=15.563626, a1=12, a2=11 or 12, w=3, th=33.88 deg)
    L227 = 15.563625652
    ux, uy = build_staircases(L227, 12, 11, 2, 2)
    nt, _, _ = greedy_pack_channel(L227, ux, uy, 3, math.radians(33.88), 2.5, math.sqrt(2)*L227 - 2.5, -1.2, 1.2)
    print(f"Calibration n=227 test (a=12,11 b=2,2 w=3 @ 33.88deg): n_up={len(ux)}, n_tilt={nt}, total={len(ux)+nt}", flush=True)

    targets = [401, 442, 530]
    info = u.load_incumbent_and_frontier(targets)
    thetas = [math.radians(d) for d in np.linspace(1.0, 45.0, 45)]

    results = []
    for n in targets:
        k = math.isqrt(n - 1)
        L_ours = info[n]["L_ours"]
        L_front = info[n]["L_front"]
        L_probe = L_ours - 0.002
        t0 = time.time()
        a1_list = list(range(max(4, k - 12), k))
        chunks = [a1_list[i::7] for i in range(7) if a1_list[i::7]]
        with ProcessPoolExecutor(max_workers=7) as ex:
            res = list(ex.map(scan_diag_worker, [(n, L_probe, c, thetas) for c in chunks]))
        best = max(res, key=lambda r: r["best_cap"])
        dt = time.time() - t0
        rec = {
            "n": n,
            "L_ours": L_ours,
            "L_front": L_front,
            "L_probe": L_probe,
            "best_cfg": best["best_cfg"],
            "deficit": n - best["best_cap"],
            "seconds": round(dt, 2),
        }
        results.append(rec)
        print(f"n={n} (L_probe={L_probe:.6f}): cap={best['best_cap']} (def={n-best['best_cap']}), cfg={best['best_cfg']} ({dt:.1f}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "idea2_sheared_diag.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
