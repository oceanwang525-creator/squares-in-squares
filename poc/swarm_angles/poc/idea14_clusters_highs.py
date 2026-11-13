"""Idea 14A: Calibrated n=37 chevron, n=50 bell-slipped & Pythagorean multi-diamond clusters,
asymmetric 2blk, and chiral 4pin exact HiGHS (time_limit=2.5s) + L-BFGS/LP on n=401, 442.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import math
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = str(Path(__file__).resolve().parent)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from poc.swarm_angles.poc.idea12_highs_search import non_overlap_batch, make_block
from poc.swarm_angles.poc.idea13_chevron_highs import solve_grid9_tl


def make_calibrated_chevron(
    L: float, w: int, n_end: int, n_mid: int, d_deg: float,
    v_off: float = 0.0, cx: float | None = None, cy: float | None = None,
) -> list:
    """3-segment hinged chevron band pivoting at bottom edge v_min (calibrated to n=37)."""
    if cx is None:
        cx = 0.5 * L
    if cy is None:
        cy = 0.5 * L
    th_m = 0.25 * math.pi
    d_rad = math.radians(d_deg)
    th_l = th_m + d_rad
    th_r = th_m - d_rad
    cm, sm = math.cos(th_m), math.sin(th_m)
    cl, sl_ = math.cos(th_l), math.sin(th_l)
    cr, sr = math.cos(th_r), math.sin(th_r)

    pts = []
    u_mid_min = -0.5 * (n_mid - 1)
    u_mid_max = 0.5 * (n_mid - 1)
    for k in range(n_mid):
        u = u_mid_min + k
        for r in range(w):
            v = v_off + (r - 0.5 * (w - 1))
            pts.append((cx + u * cm - v * sm, cy + u * sm + v * cm, th_m))

    # Hinge around the bottom edge (r=0, v_min = v_off - 0.5*(w-1)) + small transverse match
    v_min = v_off - 0.5 * (w - 1)
    # In local rotated frame, center of slice r=0..w-1 is v_ctr = v_min + 0.5*(w-1)
    lo, hi = 0.96, 1.15 + 0.9 * abs(math.sin(d_rad)) * w
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        # Pivot at bottom edge of center slice
        lbx_bot = cx + u_mid_min * cm - v_min * sm - mid * cm
        lby_bot = cy + u_mid_min * sm + v_min * cm - mid * sm
        test = list(pts)
        for j in range(n_end):
            for r in range(w):
                # step along (-cl, -sl_) for slice j and (-sl_, cl) for r
                px = lbx_bot - j * cl - r * sl_
                py = lby_bot - j * sl_ + r * cl
                test.append((px, py, th_l))
        if non_overlap_batch(test):
            hi = mid
        else:
            lo = mid
    du_hinge = hi + 1e-4

    lbx_bot = cx + u_mid_min * cm - v_min * sm - du_hinge * cm
    lby_bot = cy + u_mid_min * sm + v_min * cm - du_hinge * sm
    rbx_bot = cx + u_mid_max * cm - v_min * sm + du_hinge * cm
    rby_bot = cy + u_mid_max * sm + v_min * cm + du_hinge * sm

    for j in range(n_end):
        for r in range(w):
            pts.append((lbx_bot - j * cl - r * sl_, lby_bot - j * sl_ + r * cl, th_l))
            pts.append((rbx_bot + j * cr - r * sr, rby_bot + j * sr + r * cr, th_r))

    for px, py, th in pts:
        hx = 0.5 * (abs(math.cos(th)) + abs(math.sin(th)))
        if px - hx < -1e-7 or px + hx > L + 1e-7 or py - hx < -1e-7 or py + hx > L + 1e-7:
            return []
    if not non_overlap_batch(pts):
        return []
    return pts


def make_bell_slipped_cluster(
    L: float, w: int, n_sl: int, deg: float, dv: float, du_slip: float,
    center_plus1: bool = True, cx: float | None = None, cy: float | None = None,
) -> list:
    """Calibrated n=50 symmetric S/bell slipped diamond cluster."""
    if cx is None:
        cx = 0.5 * L
    if cy is None:
        cy = 0.5 * L
    th = math.radians(deg)
    ct, st = math.cos(th), math.sin(th)
    hx = 0.5 * (abs(ct) + abs(st))
    mid = 0.5 * (n_sl - 1)
    pts = []
    for sl in range(n_sl):
        s = sl - mid
        w_cur = w + (1 if (center_plus1 and abs(s) < 0.25) else 0)
        if mid >= 1.5 and abs(s) > 0.25 and abs(s) < mid - 0.25:
            sgn = -1.0 if s > 0 else 1.0
            v_ctr = sgn * dv
        else:
            v_ctr = 0.0
        for r in range(w_cur):
            vr = v_ctr + (r - 0.5 * (w_cur - 1))
            slip = 0.0
            if s <= 0.25 and r == w_cur - 1:
                slip = -du_slip
            elif s >= -0.25 and r == 0:
                slip = du_slip
            u = s + slip
            px = cx + u * ct - vr * st
            py = cy + u * st + vr * ct
            if px - hx < -1e-7 or px + hx > L + 1e-7 or py - hx < -1e-7 or py + hx > L + 1e-7:
                return []
            pts.append((px, py, th))
    if not non_overlap_batch(pts):
        return []
    return pts


def build_cands(n: int, k: int, L_probe: float) -> list:
    cands = []
    mg_coarse = -0.0015
    tl = 2.0

    # 1) Calibrated n=37 chevrons (~144 cands)
    for w in [3, 4, 5, 6]:
        for d_deg in [2.913, 6.0, 10.0]:
            for n_mid in [1, 3]:
                for n_end in [2, 3, 4]:
                    for v_off in [-0.32, 0.32]:
                        pts = make_calibrated_chevron(
                            L_probe, w, n_end, n_mid, d_deg, v_off, 0.5 * L_probe, 0.5 * L_probe
                        )
                        if pts and len(pts) < n:
                            cands.append((f"#{len(cands)}_chev_w{w}_e{n_end}_m{n_mid}_d{d_deg}_v{v_off}",
                                          "chev", L_probe, pts, mg_coarse, False, tl))

    # 2) Calibrated bell-slipped & Pythagorean single clusters (~450 cands)
    pyth_degs = [
        round(math.degrees(math.atan2(7, 24)), 4),
        round(math.degrees(math.atan2(5, 12)), 4),
        round(math.degrees(math.atan2(8, 15)), 4),
        round(math.degrees(math.atan2(3, 4)), 4),
        40.5,
    ]
    for deg in pyth_degs:
        for w in [3, 4, 5, 6]:
            for n_sl in [3, 5, 7, 9]:
                for dv, slip in [(0.0, 0.0), (0.20, 4.0 / 21.0), (-0.15, 0.0)]:
                    for dx, dy in [(0.0, 0.0), (-0.5, -0.5)]:
                        pts = make_bell_slipped_cluster(
                            L_probe, w, n_sl, deg, dv, slip, True, 0.5 * L_probe + dx, 0.5 * L_probe + dy
                        )
                        if pts and len(pts) < n:
                            cands.append((f"#{len(cands)}_bell_w{w}_n{n_sl}_d{deg}_dv{dv}_{dx}_{dy}",
                                          "bell", L_probe, pts, mg_coarse, False, tl))

    # 3) Multi-cluster anti-diagonal (2, 3, 4 clusters in stepped grid + SW-shifted L-border)
    for m_cl in [2, 3, 4]:
        for deg in [22.6199, 28.0725, 36.8699, 40.5]:
            for w, n_sl, cplus in [(3, 5, True), (4, 5, False), (4, 5, True)]:
                for l_off in [0.0, -1.0, -2.0, -3.0]:
                    span = L_probe + l_off
                    if span < 8.0:
                        continue
                    pts = []
                    ok = True
                    for idx in range(m_cl):
                        t = (idx + 0.5) / m_cl
                        cx_i = t * (span - 4.0) + 2.0
                        cy_i = (1.0 - t) * (span - 4.0) + 2.0
                        cl = make_bell_slipped_cluster(L_probe, w, n_sl, deg, 0.20, 4.0 / 21.0, cplus, cx_i, cy_i)
                        if not cl:
                            ok = False
                            break
                        pts.extend(cl)
                    if ok and len(pts) < n and non_overlap_batch(pts):
                        cands.append((f"#{len(cands)}_mcl{m_cl}_w{w}_n{n_sl}_d{deg}_lo{l_off}",
                                      "mcl", L_probe, pts, mg_coarse, False, tl))

    # 4) Fixed asymmetric 2blk + C4 4pin corner/edge blocks
    for deg in [22.6199, 28.0725, 36.8699, 40.5, 42.0, 45.0]:
        for u1, v1 in [(4, 5), (5, 5), (5, 6), (6, 6)]:
            for a1 in [3.5, 4.27, 5.27]:
                for a2 in [12.27, 13.27]:
                    b1 = make_block(L_probe, u1, v1, math.radians(deg), a1, a2, 0)
                    b2 = make_block(L_probe, u1, v1, math.radians(90.0 - deg), a2, a1, 0)
                    if b1 and b2 and non_overlap_batch(b1 + b2):
                        cands.append((f"#{len(cands)}_2blk_{u1}x{v1}_d{deg}_{a1}_{a2}",
                                      "2blk", L_probe, b1 + b2, mg_coarse, False, tl))

    for deg in [5.0, 12.0, 22.6199, 30.0, 36.8699]:
        th = math.radians(deg)
        for u1, v1 in [(3, 5), (4, 6), (5, 7), (4, 8)]:
            for pad in [0.0, 1.0, 2.0]:
                hx = 0.5 * (u1 * abs(math.cos(th)) + v1 * abs(math.sin(th)))
                hy = 0.5 * (u1 * abs(math.sin(th)) + v1 * abs(math.cos(th)))
                cxs = [hx + pad, L_probe - hy - pad, L_probe - hx - pad, hy + pad]
                cys = [hy + pad, hx + pad, L_probe - hy - pad, L_probe - hx - pad]
                angs = [th, th + 0.5 * math.pi, th + math.pi, th + 1.5 * math.pi]
                pts = []
                ok = True
                for aa, cx_, cy_ in zip(angs, cxs, cys):
                    blk = make_block(L_probe, u1, v1, aa, cx_, cy_, 0)
                    if not blk:
                        ok = False
                        break
                    pts.extend(blk)
                if ok and len(pts) < n and non_overlap_batch(pts):
                    cands.append((f"#{len(cands)}_4pin_{u1}x{v1}_d{deg}_p{pad}",
                                  "4pin", L_probe, pts, mg_coarse, False, tl))
    return cands


def main():
    t0 = time.time()
    L37 = 6.598619609
    b37 = make_calibrated_chevron(L37, 3, 2, 1, 2.913, 0.32424, 0.5 * L37, 0.5 * L37)
    r37 = solve_grid9_tl(("c37", "chev", L37, b37, -0.0015, True, 3.0)) if b37 else ("", "", 0, 0, 0, None)
    print(f"Calib n=37 calibrated chevron Grid9: {r37[2]}/37 (nt={r37[3]} up={r37[4]})", flush=True)

    L50 = 7.571428571
    b50 = make_bell_slipped_cluster(L50, 3, 5, math.degrees(math.atan2(3, 4)), 0.20, 4.0 / 21.0, True, 0.5 * L50, 0.5 * L50)
    r50 = solve_grid9_tl(("c50", "bell", L50, b50, 0.0, False, 3.0)) if b50 else ("", "", 0, 0, 0, None)
    print(f"Calib n=50 bell-slipped Grid9: {r50[2]}/50 (nt={r50[3]} up={r50[4]})", flush=True)

    targets = [
        (401, 20, 20.535533905933),
        (442, 21, 21.535533905933),
    ]
    summary = []
    near_misses = {}
    u = None
    with ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn")) as ex:
        for n, k, L_base in targets:
            tn = time.time()
            L_probe = L_base - 0.001
            cands = build_cands(n, k, L_probe)
            fam_cnt = {}
            for c in cands:
                fam_cnt[c[1]] = fam_cnt.get(c[1], 0) + 1
            print(f"\nn={n} (L_probe={L_probe:.6f}): evaluating {len(cands)} cands {fam_cnt}...", flush=True)
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
                for _, tag, _, _ in sorted(lst, key=lambda x: -x[0])[:5]:
                    top_tags.add(tag)
            fine_tasks = []
            for tag in sorted(top_tags):
                _, fam, L_, tilted, _, _, _ = cand_map[tag]
                fine_tasks.append((tag + "|aug0", fam, L_, tilted, 0.0, True, 3.5))
                fine_tasks.append((tag + "|soft8", fam, L_, tilted, -0.08, True, 3.5))
            fine_res = list(ex.map(solve_grid9_tl, fine_tasks, chunksize=2))
            best_fine = max(fine_res, key=lambda x: x[2])
            print(f"  Fine+soft best: {best_fine[2]}/{n} ({best_fine[0]})", flush=True)

            near_list = []
            for tag, fam, tot, nt, nup, up_pts in sorted(fine_res, key=lambda x: -x[2])[:8]:
                base_tag = tag.split("|")[0]
                tilted = np.asarray(cand_map[base_tag][3], dtype=np.float64)
                near_list.append({
                    "tag": tag, "tot": tot, "nt": nt, "nup": nup,
                    "tilted": tilted.tolist(), "up": np.asarray(up_pts, dtype=np.float64).tolist(),
                })
            near_misses[str(n)] = near_list

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
                    rx, ry, rth, pen = u.relax_lbfgs(all_x, all_y, all_th, L_probe, maxiter=500)
                    if pen < 1e-7:
                        lx, ly, lL, ok2 = u.fixed_angle_lp_tighten(rx, ry, rth, rounds=5, margin=1e-9)
                        if ok2 and lL < win_L - 1e-11:
                            ver = u.verify_official(n, lx, ly, rth)
                            if ver["ok"] and ver["L_float"] < win_L - 1e-11:
                                win_L = ver["L_float"]
                                pd.DataFrame(ver["rows"], columns=["id", "n", "square_id", "x", "y", "angle"]).to_csv(
                                    f"poc/swarm_angles/data/cand14a_n{n}.csv", index=False
                                )
                                print(f"  *** CERTIFIED WIN n={n}: L={win_L:.12f} (< {L_base:.12f}) via {tag}! ***", flush=True)

            summary.append({
                "n": n, "L_base": L_base, "cands": fam_cnt,
                "by_fam": {k_: v_[0] for k_, v_ in by_fam.items()},
                "fine_best": best_fine[2], "fine_tag": best_fine[0],
                "win_L": win_L, "seconds": round(time.time() - tn, 2),
            })

    with open("poc/swarm_angles/docs/idea14a_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("/home/lucas/.gemini/antigravity-cli/brain/95f98074-e284-43bd-8857-8c64d023144c/scratch/idea14a_nearmisses.json", "w") as f:
        json.dump(near_misses, f)
    print(f"\nIdea 14A finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
