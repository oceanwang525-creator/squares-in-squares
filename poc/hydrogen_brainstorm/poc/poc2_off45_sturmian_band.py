"""POC 2: Off-45-Degree Sturmian-Staircase Diagonal Band.

Falsification test for Idea 3 on:
  - Calibration: n=269 (L=16.889730, ~41.97 deg), n=328 (L_ours=18.614369, L_front=18.599535)
  - 45_only high-gap targets: n=367, n=492, n=634, n=686
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

import geom_utils as gu

OUT_DIR = Path(__file__).resolve().parents[1] / "docs"


def build_off45_band_candidate(
    L: float,
    theta: float,
    w: int,
    v0: float,
    stagger: float = 0.0,
    se_anchor_mode: str = "corner_L0",
    allow_pre_pen: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an off-45 tilted band of width w along angle theta + Sturmian upright flanks.

    Returns (x, y, th, is_band_or_interface_mask).
    """
    ct, st = math.cos(theta), math.sin(theta)
    hx_t = 0.5 * (abs(ct) + abs(st))
    # Band frame: u = x*ct + y*st, v = -x*st + y*ct
    # Max possible u in [0, L]^2 is L*(ct + st)
    u_max = L * (ct + st)
    bx, by = [], []
    for col in range(w):
        vc = v0 - 0.5 * (w - 1) + col
        u_shift = (col % 2) * stagger
        for k in range(-2, int(math.ceil(u_max)) + 3):
            uc = u_shift + k
            xc = uc * ct - vc * st
            yc = uc * st + vc * ct
            if xc - hx_t >= 1e-6 and xc + hx_t <= L - 1e-6 and yc - hx_t >= 1e-6 and yc + hx_t <= L - 1e-6:
                bx.append(xc)
                by.append(yc)
    bx_arr = np.array(bx, dtype=np.float64)
    by_arr = np.array(by, dtype=np.float64)
    bth_arr = np.full(len(bx_arr), theta, dtype=np.float64)

    # Generate upright candidate lattices on NW and SE sides of band
    K_max = int(math.floor(L))
    ux, uy, u_iface = [], [], []
    # NW region: v > v0, anchored at (0, L)
    # SE region: v < v0, anchored at (L, 0) or (L, L) or (0, 0)
    for i in range(K_max):
        for j in range(K_max):
            # Test NW lattice anchored at (0, L): x = 0.5 + i, y = L - 0.5 - j
            cx_nw, cy_nw = 0.5 + i, L - 0.5 - j
            v_nw = -cx_nw * st + cy_nw * ct
            if v_nw > v0:
                if cx_nw + 0.5 <= L + 1e-9 and cy_nw - 0.5 >= -1e-9:
                    ux.append(cx_nw)
                    uy.append(cy_nw)
                    u_iface.append(0)
            # SE lattice:
            if se_anchor_mode == "corner_L0":
                cx_se, cy_se = L - 0.5 - i, 0.5 + j
            elif se_anchor_mode == "corner_00":
                cx_se, cy_se = 0.5 + i, 0.5 + j
            else:
                cx_se, cy_se = L - 0.5 - i, L - 0.5 - j
            v_se = -cx_se * st + cy_se * ct
            if v_se < v0:
                if cx_se - 0.5 >= -1e-9 and cx_se + 0.5 <= L + 1e-9 and cy_se - 0.5 >= -1e-9 and cy_se + 0.5 <= L + 1e-9:
                    ux.append(cx_se)
                    uy.append(cy_se)
                    u_iface.append(0)

    ux_arr = np.array(ux, dtype=np.float64)
    uy_arr = np.array(uy, dtype=np.float64)
    if len(bx_arr) == 0 or len(ux_arr) == 0:
        return bx_arr, by_arr, bth_arr, np.ones(len(bx_arr), dtype=bool)

    # Filter upright squares by SAT penetration against band squares
    keep = []
    iface_flag = []
    for idx in range(len(ux_arr)):
        xi, yi = ux_arr[idx], uy_arr[idx]
        # Quick bounding filter
        near = np.where((np.abs(bx_arr - xi) < 0.5 + hx_t + 0.1) & (np.abs(by_arr - yi) < 0.5 + hx_t + 0.1))[0]
        if len(near) == 0:
            keep.append(idx)
            iface_flag.append(False)
            continue
        max_pen = -1e9
        for j in near:
            dx = bx_arr[j] - xi
            dy = by_arr[j] - yi
            rad = 0.5 + hx_t
            p1 = rad - abs(dx)
            p2 = rad - abs(dy)
            p3 = rad - abs(dx * ct + dy * st)
            p4 = rad - abs(-dx * st + dy * ct)
            pen = min(p1, p2, p3, p4)
            if pen > max_pen:
                max_pen = pen
        if max_pen <= allow_pre_pen:
            keep.append(idx)
            iface_flag.append(max_pen > -0.25)

    fx = np.concatenate([bx_arr, ux_arr[keep]])
    fy = np.concatenate([by_arr, uy_arr[keep]])
    fth = np.concatenate([bth_arr, np.zeros(len(keep), dtype=np.float64)])
    free_mask = np.concatenate([np.ones(len(bx_arr), dtype=bool), np.array(iface_flag, dtype=bool)])
    return fx, fy, fth, free_mask


def test_target(n_target: int, L_ours: float, L_front: float) -> dict:
    t0 = time.time()
    L_probes = [L_ours - 0.001, L_front + 0.002] if L_ours - L_front > 0.004 else [L_ours - 0.0005]

    best_rigid_cap = 0
    best_rigid_cfg = None
    best_relax_cap = 0
    best_relax_cfg = None
    best_state = None

    angles_deg = [37.0, 38.5, 40.0, 41.0, 41.97, 42.5, 43.0, 43.5, 44.0, 44.2, 44.5]
    # Centerline v0 around diagonal v=0 (v of (L/2, L/2) is L*(cos-sin)/2)
    configs = 0

    for L_probe in L_probes:
        for deg in angles_deg:
            theta = math.radians(deg)
            v_mid = 0.5 * L_probe * (math.cos(theta) - math.sin(theta))
            for w in range(1, 8):
                for dv in np.linspace(-2.5, 2.5, 11):
                    v0 = v_mid + dv
                    for stagger in (0.0, 0.5):
                        for mode in ("corner_L0", "corner_00"):
                            configs += 1
                            # 1) Rigid check (allow_pre_pen = -1e-6)
                            x, y, th, _ = build_off45_band_candidate(
                                L_probe, theta, w, v0, stagger=stagger, se_anchor_mode=mode, allow_pre_pen=-1e-6
                            )
                            cap = len(x)
                            if cap > best_rigid_cap:
                                best_rigid_cap = cap
                                best_rigid_cfg = {
                                    "L_probe": L_probe, "deg": deg, "w": w, "dv": float(dv),
                                    "stagger": stagger, "mode": mode, "cap": cap,
                                }
                            # 2) Pre-screen soft interface (allow_pre_pen = 0.03) if close
                            if cap >= n_target - 8:
                                x2, y2, th2, fmask = build_off45_band_candidate(
                                    L_probe, theta, w, v0, stagger=stagger, se_anchor_mode=mode, allow_pre_pen=0.035
                                )
                                if len(x2) > best_relax_cap or len(x2) >= n_target:
                                    # Relax band + interface
                                    rx, ry, rth, rpen = gu.relax_lbfgs(x2, y2, th2, L_probe, free_mask=fmask, maxiter=150)
                                    if rpen < 1e-8 and len(x2) > best_relax_cap:
                                        best_relax_cap = len(x2)
                                        best_relax_cfg = {
                                            "L_probe": L_probe, "deg": deg, "w": w, "dv": float(dv),
                                            "stagger": stagger, "mode": mode, "cap": len(x2), "pen": float(rpen),
                                        }
                                        best_state = (rx, ry, rth, L_probe)

    certified_win = None
    relaxed_L = None
    if best_state is not None and len(best_state[0]) >= n_target:
        rx, ry, rth, bL = best_state
        # Drop least-central excess if > n_target
        x_t, y_t, th_t = rx[:n_target], ry[:n_target], rth[:n_target]
        lx, ly, lL, lok = gu.fixed_angle_lp_tighten(x_t, y_t, th_t, rounds=4)
        relaxed_L = lL
        if lok and lL < L_ours - 1e-9:
            cert = gu.verify_official(n_target, lx, ly, th_t)
            if cert["ok"] and cert["L_float"] < L_ours:
                certified_win = cert

    return {
        "n": n_target,
        "L_ours": L_ours,
        "L_front": L_front,
        "configs": configs,
        "best_rigid_cap": best_rigid_cap,
        "rigid_deficit": n_target - best_rigid_cap,
        "best_rigid_cfg": best_rigid_cfg,
        "best_relax_cap": best_relax_cap,
        "relax_deficit": n_target - best_relax_cap,
        "best_relax_cfg": best_relax_cfg,
        "relaxed_L": relaxed_L,
        "certified_win": {k: v for k, v in certified_win.items() if k != "rows"} if certified_win else None,
        "seconds": round(time.time() - t0, 2),
    }


def main():
    import pandas as pd

    inc = gu.load_incumbent_Ls()
    gap_df = pd.read_csv(gu.REPO_ROOT / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv").set_index("n")

    targets = [269, 328, 367, 492, 634, 686]
    results = []
    for n in targets:
        L_ours = inc[n]
        L_front = float(gap_df.loc[n, "L_frontier"])
        print(f"Running POC 2 on n={n} (ours={L_ours:.6f}, front={L_front:.6f})...", flush=True)
        res = test_target(n, L_ours, L_front)
        results.append(res)
        print(
            f"  -> n={n}: rigid_cap={res['best_rigid_cap']} (def={res['rigid_deficit']}), "
            f"relax_cap={res['best_relax_cap']} (def={res['relax_deficit']}), "
            f"win={res['certified_win'] is not None}, time={res['seconds']}s",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "poc2_off45_sturmian_band.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
