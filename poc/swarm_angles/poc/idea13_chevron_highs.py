"""Idea 13B: Exact HiGHS (time_limit=2.5s) on n=401, 442 for:
1) n=37-style 3-segment hinged chevron diagonal bands (45+d, 45, 45-d)
2) n=50-style edge-slipped (w / w+1) sheared bands (calibrated on n=50 & n=403)
3) Asymmetric unequal 2-block (U1 x V1 @ th1, U2 x V2 @ th2) + fine interface + L-BFGS/LP.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import math
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

from poc.swarm_angles.poc.idea12_highs_search import non_overlap_batch, make_block, sat_clear_np


def solve_grid9_tl(args):
    tag, fam, L, tilted, mg, augment_iface, tl = args
    k = int(math.floor(L))
    f = L - k
    offs = [0.0, 0.5 * f, f]

    if len(tilted) > 0:
        tarr = np.asarray(tilted, dtype=np.float64)
        tx, ty, tth = tarr[:, 0], tarr[:, 1], tarr[:, 2]
        tc, ts = np.cos(tth), np.sin(tth)
        thx = 0.5 * (np.abs(tc) + np.abs(ts))
        rad = 0.5 + thx + mg - 1e-7
    else:
        tx = ty = tc = ts = thx = rad = np.empty(0, dtype=np.float64)

    xs = [o + i + 0.5 for o in offs for i in range(k) if o + i + 0.5 + 0.5 <= L + 1e-7]
    ys = [o + j + 0.5 for o in offs for j in range(k) if o + j + 0.5 + 0.5 <= L + 1e-7]
    gx, gy = np.meshgrid(np.array(xs), np.array(ys), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    if augment_iface and len(tx) > 0:
        extra_offs = [0.25 * f, 0.75 * f, 0.5]
        ex = [o + i + 0.5 for o in extra_offs for i in range(k) if o + i + 0.5 <= L - 0.5 + 1e-7]
        ey = [o + j + 0.5 for o in offs + extra_offs for j in range(k) if o + j + 0.5 <= L - 0.5 + 1e-7]
        if ex and ey:
            egx, egy = np.meshgrid(np.array(ex), np.array(ey), indexing="ij")
            epts = np.stack([egx.ravel(), egy.ravel()], axis=1)
            d2 = (epts[:, 0:1] - tx[None, :]) ** 2 + (epts[:, 1:2] - ty[None, :]) ** 2
            near = np.min(d2, axis=1) <= 1.8 * 1.8
            if np.any(near):
                pts = np.vstack([pts, epts[near]])
                _, uidx = np.unique(np.round(pts, 7), axis=0, return_index=True)
                pts = pts[uidx]

    if len(tx) > 0:
        dx = pts[:, 0:1] - tx[None, :]
        dy = pts[:, 1:2] - ty[None, :]
        ov = (np.abs(dx) < rad[None, :]) & (np.abs(dy) < rad[None, :]) & \
             (np.abs(dx * tc[None, :] + dy * ts[None, :]) < rad[None, :]) & \
             (np.abs(-dx * ts[None, :] + dy * tc[None, :]) < rad[None, :])
        pts = pts[~np.any(ov, axis=1)]

    if len(pts) == 0:
        return (tag, fam, len(tilted), len(tilted), 0, np.empty((0, 2)))

    uxs = np.unique(np.round(pts[:, 0], 9))
    rows, cols = [], []
    rid = 0
    eps = 1e-9
    for xv in uxs:
        idxs = np.where((pts[:, 0] >= xv - eps) & (pts[:, 0] < xv + 1.0 - eps))[0]
        if len(idxs) <= 1:
            continue
        py_sub = pts[idxs, 1]
        for yv in np.unique(np.round(py_sub, 9)):
            cl = idxs[(py_sub >= yv - eps) & (py_sub < yv + 1.0 - eps)]
            if len(cl) >= 2:
                rows.extend([rid] * len(cl))
                cols.extend(cl.tolist())
                rid += 1

    if rid == 0:
        return (tag, fam, len(tilted) + len(pts), len(tilted), len(pts), pts)

    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(rid, len(pts))).tocsc()
    res = milp(c=-np.ones(len(pts)), integrality=np.ones(len(pts)), bounds=Bounds(0, 1),
               constraints=LinearConstraint(A, -np.inf, 1.0), options={"time_limit": tl})
    if res.x is None:
        return (tag, fam, len(tilted), len(tilted), 0, np.empty((0, 2)))
    sel = res.x > 0.5
    nup = int(np.sum(sel))
    return (tag, fam, len(tilted) + nup, len(tilted), nup, pts[sel])


def make_chevron_band(L: float, w: int, n_end: int, n_mid: int, d_deg: float,
                      u0_shift: float, v_jog_slope: float) -> list:
    """3-segment hinged diagonal band (45+d, 45, 45-d) generalizing n=37."""
    th_mid = 0.25 * math.pi
    d_rad = math.radians(d_deg)
    th_a = th_mid + d_rad
    th_b = th_mid - d_rad
    total_sl = 2 * n_end + n_mid
    c_m, s_m = math.cos(th_mid), math.sin(th_mid)
    diag_len = math.sqrt(2.0) * L
    u_start = 0.5 * diag_len - 0.5 * (total_sl - 1) + u0_shift

    pts = []
    # Build contiguous slices along u, hinged smoothly in angle between segments
    cx, cy = 0.5 * L, 0.5 * L
    for sl in range(total_sl):
        u_rel = sl - 0.5 * (total_sl - 1) + u0_shift
        if sl < n_end:
            th = th_a
        elif sl < n_end + n_mid:
            th = th_mid
        else:
            th = th_b
        ct, st = math.cos(th), math.sin(th)
        hx = 0.5 * (abs(ct) + abs(st))
        # Center along diagonal plus small transverse jog
        s_cx = cx + u_rel * c_m + v_jog_slope * (-s_m) * (abs(u_rel) / max(1.0, 0.5 * total_sl))
        s_cy = cy + u_rel * s_m + v_jog_slope * (c_m) * (abs(u_rel) / max(1.0, 0.5 * total_sl))
        for r in range(w):
            v_r = r - 0.5 * (w - 1)
            px = s_cx + v_r * (-st)
            py = s_cy + v_r * ct
            if px - hx < -1e-7 or px + hx > L + 1e-7 or py - hx < -1e-7 or py + hx > L + 1e-7:
                return []
            pts.append((px, py, th))
    if not non_overlap_batch(pts):
        return []
    return pts


def make_slipped_band(L: float, w: int, n_sl: int, deg: float, dv_step: float,
                      du_slip: float, u_off: float, alt_plus: bool) -> list:
    """n=50 / n=403 style edge-slipped (w / w+1) sheared diagonal band."""
    th = math.radians(deg)
    ct, st = math.cos(th), math.sin(th)
    hx = 0.5 * (abs(ct) + abs(st))
    # Rotate diagonal by 45deg center
    pts = []
    cx, cy = 0.5 * L, 0.5 * L
    for sl in range(n_sl):
        u = (sl - 0.5 * (n_sl - 1)) + u_off
        w_cur = w + (1 if (alt_plus and sl == n_sl // 2) else 0)
        # Center jog in v
        v_ctr = sl * dv_step - 0.5 * (n_sl - 1) * dv_step
        for r in range(w_cur):
            vr = v_ctr + (r - 0.5 * (w_cur - 1))
            slip = -du_slip if r == 0 else (du_slip if r == w_cur - 1 else 0.0)
            px = cx + (u + slip) * ct - vr * st
            py = cy + (u + slip) * st + vr * ct
            if px - hx < -1e-7 or px + hx > L + 1e-7 or py - hx < -1e-7 or py + hx > L + 1e-7:
                return []
            pts.append((px, py, th))
    if not non_overlap_batch(pts):
        return []
    return pts


def build_cands(n: int, k: int, L_probe: float) -> list:
    cands = []
    # 1) Chevron 3-segment bands (n=37 generalized)
    for w in [3, 4, 5, 6]:
        for d_deg in [2.5, 5.0, 8.0, 12.0]:
            for n_mid in [1, 2, 3, 5]:
                for n_end in range(2, k // 2):
                    if (2 * n_end + n_mid) * w > int(1.6 * k * w):
                        continue
                    for vj in [-0.35, 0.0, 0.35]:
                        pts = make_chevron_band(L_probe, w, n_end, n_mid, d_deg, 0.0, vj)
                        if pts and len(pts) < n:
                            cands.append((f"#{len(cands)}_chev_w{w}_e{n_end}_m{n_mid}_d{d_deg}_v{vj}",
                                          "chev", L_probe, pts, 0.0, False, 2.5))

    # 2) Slipped sheared diagonal bands (n=50 / n=403 generalized)
    for deg in [28.0, 33.8, 36.87, 40.5]:
        for w in [3, 4, 5, 6]:
            for n_sl in [5, 7, 9, 13, 17, 21, 25]:
                if n_sl * w > n - 100:
                    continue
                for dv in [-0.15, 0.0, 0.15, 0.25]:
                    for slip in [0.0, 0.18]:
                        pts = make_slipped_band(L_probe, w, n_sl, deg, dv, slip, 0.0, (n_sl <= 9))
                        if pts and len(pts) < n:
                            cands.append((f"#{len(cands)}_slip_w{w}_n{n_sl}_d{deg}_dv{dv}_s{slip}",
                                          "slip", L_probe, pts, 0.0, False, 2.5))

    # 3) Asymmetric 2-block tilted at (45+d, 45-d) or unequal U x V
    for d_deg in [0.0, 4.0, 8.0, 12.0, 20.0]:
        for u1, v1 in [(4, 5), (5, 5), (5, 6), (6, 6), (7, 7)]:
            for cx1 in [u1 * 0.72 + 0.5, u1 * 0.72 + 2.0]:
                for cy1 in [L_probe - v1 * 0.72 - 0.5, L_probe - v1 * 0.72 - 2.2]:
                    b1 = make_block(L_probe, u1, v1, math.radians(45.0 + d_deg), cx1, cy1, 0)
                    b2 = make_block(L_probe, u1, v1, math.radians(45.0 - d_deg), L_probe - cy1, L_probe - cx1, 0)
                    if b1 and b2 and non_overlap_batch(b1 + b2):
                        cands.append((f"#{len(cands)}_2blk_{u1}x{v1}_d{d_deg}",
                                      "2blk", L_probe, b1 + b2, 0.0, False, 2.5))
    return cands


def main():
    t0 = time.time()
    # Sanity calibrate slipped band on n=50
    L50 = 7.571428571
    b50 = make_slipped_band(L50, 3, 5, 36.87, 0.22, 0.19, 0.0, True)
    c50 = solve_grid9_tl(("c50", "slip", L50, b50, 0.0, False, 2.5)) if b50 else ("", "", 0, 0, 0, None)
    print(f"Calib n=50 slipped band exact HiGHS: {c50[2]}/50 (nt={c50[3]})", flush=True)

    targets = [
        (401, 20, 20.535533905933),
        (442, 21, 21.535533905933),
    ]
    summary = []
    u = None
    with ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn")) as ex:
        for n, k, L_base in targets:
            tn = time.time()
            L_probe = L_base - 0.001
            cands = build_cands(n, k, L_probe)
            print(f"\nn={n} (L_probe={L_probe:.6f}): evaluating {len(cands)} candidates (tl=2.5s)...", flush=True)
            res = list(ex.map(solve_grid9_tl, cands, chunksize=4))
            by_fam = {}
            by_fam_list = {}
            for tag, fam, tot, nt, nup, up_pts in res:
                by_fam_list.setdefault(fam, []).append((tot, tag, nt, nup))
                if fam not in by_fam or tot > by_fam[fam][0]:
                    by_fam[fam] = (tot, tag, nt, nup)
            for fam, (tot, tag, nt, nup) in sorted(by_fam.items()):
                print(f"  fam={fam:4s}: exact_Grid9={tot:3d}/{n} [{tag} nt={nt} up={nup}]", flush=True)

            cand_map = {c[0]: c for c in cands}
            top_tags = set()
            for fam, lst in by_fam_list.items():
                for _, tag, _, _ in sorted(lst, key=lambda x: -x[0])[:6]:
                    top_tags.add(tag)
            fine_tasks = []
            for tag in sorted(top_tags):
                _, fam, L_, tilted, _, _, _ = cand_map[tag]
                fine_tasks.append((tag + "|aug0", fam, L_, tilted, 0.0, True, 3.0))
                fine_tasks.append((tag + "|soft8", fam, L_, tilted, -0.08, True, 3.0))
            fine_res = list(ex.map(solve_grid9_tl, fine_tasks, chunksize=2))
            best_fine = max(fine_res, key=lambda x: x[2])
            print(f"  Fine+soft best: {best_fine[2]}/{n} ({best_fine[0]})", flush=True)

            win_L = L_base
            for tag, fam, tot, nt, nup, up_pts in fine_res:
                if tot >= n:
                    if u is None:
                        from poc.swarm_angles.poc import utils as u
                    base_tag = tag.split("|")[0]
                    tilted = np.asarray(cand_map[base_tag][3], dtype=np.float64)
                    if tot > n and len(tilted) > 0:
                        need = n - len(tilted)
                        d2 = np.min((up_pts[:, 0:1] - tilted[:, 0][None, :]) ** 2 +
                                    (up_pts[:, 1:2] - tilted[:, 1][None, :]) ** 2, axis=1)
                        up_pts = up_pts[np.argsort(-d2)[:need]]
                    else:
                        up_pts = up_pts[: n - len(tilted)]
                    all_x = np.concatenate([tilted[:, 0], up_pts[:, 0]])
                    all_y = np.concatenate([tilted[:, 1], up_pts[:, 1]])
                    all_th = np.concatenate([tilted[:, 2], np.zeros(len(up_pts))])
                    rx, ry, rth, rL, ok = u.relax_lbfgs(all_x, all_y, all_th, L_probe)
                    if ok:
                        lx, ly, lL, ok2 = u.fixed_angle_lp_tighten(rx, ry, rth, rounds=5, margin=1e-9)
                        if ok2 and lL < win_L - 1e-11:
                            ver = u.verify_official(n, lx, ly, rth)
                            if ver["ok"] and ver["L_lit"] < win_L - 1e-11:
                                win_L = ver["L_lit"]
                                pd.DataFrame(ver["rows"], columns=["id", "n", "square_id", "x", "y", "angle"]).to_csv(
                                    f"poc/swarm_angles/data/cand13b_n{n}.csv", index=False
                                )
                                print(f"  *** CERTIFIED WIN n={n}: L_lit={win_L:.12f} (< {L_base:.12f}) via {tag}! ***", flush=True)

            summary.append({
                "n": n, "L_base": L_base,
                "by_fam": {k_: v_[0] for k_, v_ in by_fam.items()},
                "fine_best": best_fine[2], "fine_tag": best_fine[0],
                "win_L": win_L, "seconds": round(time.time() - tn, 2),
            })

    with open("poc/swarm_angles/docs/idea13b_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nIdea 13B finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
