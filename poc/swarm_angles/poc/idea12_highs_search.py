"""Idea 12B: Exact HiGHS Clique-MIS + Interface Fine/Sliding Grid + L-BFGS/LP search
across multi-diamond clusters, n=847 hybrids, true oblique-cut n=971/908 pockets,
4-edge chiral frames, and calibrated DP jogging diagonal bands on D=1 targets.
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


def sat_clear_np(px: float, py: float, tx: np.ndarray, ty: np.ndarray,
                 tc: np.ndarray, ts: np.ndarray, thx: np.ndarray, mg: float) -> bool:
    if len(tx) == 0:
        return True
    rad = 0.5 + thx + mg - 1e-7
    dx = px - tx
    m = np.abs(dx) < rad
    if not np.any(m):
        return True
    dy = py - ty[m]
    rad = rad[m]
    m2 = np.abs(dy) < rad
    if not np.any(m2):
        return True
    dx = dx[m][m2]
    dy = dy[m2]
    rad = rad[m2]
    c = tc[m][m2]
    s = ts[m][m2]
    return not np.any((np.abs(dx * c + dy * s) < rad) & (np.abs(-dx * s + dy * c) < rad))


def build_grid9_and_solve(args):
    """Worker: exact HiGHS clique-MIS on Grid9 (or interface-augmented Grid)."""
    tag, fam, L, tilted, mg, augment_iface = args
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

    # Vectorized Grid9 candidate filter
    xs = []
    ys = []
    for ox in offs:
        for i in range(k):
            px = ox + i + 0.5
            if px + 0.5 <= L + 1e-7:
                xs.append(px)
    for oy in offs:
        for j in range(k):
            py = oy + j + 0.5
            if py + 0.5 <= L + 1e-7:
                ys.append(py)
    gx, gy = np.meshgrid(np.array(xs), np.array(ys), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    if augment_iface and len(tx) > 0:
        extra_offs = [0.25 * f, 0.75 * f, 0.5]
        ex = [o + i + 0.5 for o in extra_offs for i in range(k) if o + i + 0.5 <= L - 0.5 + 1e-7]
        ey = [o + j + 0.5 for o in offs + extra_offs for j in range(k) if o + j + 0.5 <= L - 0.5 + 1e-7]
        if ex and ey:
            egx, egy = np.meshgrid(np.array(ex), np.array(ey), indexing="ij")
            epts = np.stack([egx.ravel(), egy.ravel()], axis=1)
            # Keep only within 1.8 units of any tilted square center
            d2 = (epts[:, 0:1] - tx[None, :]) ** 2 + (epts[:, 1:2] - ty[None, :]) ** 2
            near = np.min(d2, axis=1) <= 1.8 * 1.8
            if np.any(near):
                pts = np.vstack([pts, epts[near]])
                # Deduplicate rounded
                _, uidx = np.unique(np.round(pts, 7), axis=0, return_index=True)
                pts = pts[uidx]

    if len(tx) > 0:
        dx = pts[:, 0:1] - tx[None, :]
        dy = pts[:, 1:2] - ty[None, :]
        ov = (np.abs(dx) < rad[None, :]) & (np.abs(dy) < rad[None, :]) & \
             (np.abs(dx * tc[None, :] + dy * ts[None, :]) < rad[None, :]) & \
             (np.abs(-dx * ts[None, :] + dy * tc[None, :]) < rad[None, :])
        keep = ~np.any(ov, axis=1)
        pts = pts[keep]

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
    tl = 1.5 if augment_iface else 0.55
    res = milp(c=-np.ones(len(pts)), integrality=np.ones(len(pts)), bounds=Bounds(0, 1),
               constraints=LinearConstraint(A, -np.inf, 1.0),
               options={"time_limit": tl})
    if res.x is None:
        return (tag, fam, len(tilted), len(tilted), 0, np.empty((0, 2)))
    sel = res.x > 0.5
    nup = int(np.sum(sel))
    return (tag, fam, len(tilted) + nup, len(tilted), nup, pts[sel])


def non_overlap_batch(sq: list) -> bool:
    if not sq:
        return False
    arr = np.asarray(sq, dtype=np.float64)
    x, y, th = arr[:, 0], arr[:, 1], arr[:, 2]
    c, s = np.cos(th), np.sin(th)
    hx = 0.5 * (np.abs(c) + np.abs(s))
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    bb = hx[None, :] + hx[:, None] - 1e-7
    cand = np.triu((np.abs(dx) < bb) & (np.abs(dy) < bb), k=1)
    ii, jj = np.where(cand)
    if len(ii) == 0:
        return True
    rad = 0.5 * (1.0 + np.abs(c[ii] * c[jj] + s[ii] * s[jj]) + np.abs(c[ii] * s[jj] - s[ii] * c[jj])) - 1e-7
    ov = (np.abs(dx[ii, jj] * c[ii] + dy[ii, jj] * s[ii]) < rad) & \
         (np.abs(-dx[ii, jj] * s[ii] + dy[ii, jj] * c[ii]) < rad) & \
         (np.abs(dx[ii, jj] * c[jj] + dy[ii, jj] * s[jj]) < rad) & \
         (np.abs(-dx[ii, jj] * s[jj] + dy[ii, jj] * c[jj]) < rad)
    return not np.any(ov)


def make_block(L: float, U: int, V: int, th: float, cx: float, cy: float, cut: int = 0) -> list:
    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (abs(c) + abs(s))
    u0, v0 = -0.5 * (U - 1), -0.5 * (V - 1)
    out = []
    for iu in range(U):
        for iv in range(V):
            if min(iu, U - 1 - iu) + min(iv, V - 1 - iv) < cut:
                continue
            u, v = u0 + iu, v0 + iv
            x = cx + u * c - v * s
            y = cy + u * s + v * c
            if x - hx < -1e-7 or x + hx > L + 1e-7 or y - hx < -1e-7 or y + hx > L + 1e-7:
                return []
            out.append((x, y, th))
    return out


def make_971_oblique(L: float, th: float, wp: float, x_lo: float, x_hi: float,
                     y_lo: float, u_cut: float, mode: int) -> list:
    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (abs(c) + abs(s))
    p1 = []
    max_i = int(math.ceil(L * 1.6))
    for iu in range(-max_i, max_i + 1):
        u = u_cut + iu
        if u < u_cut - 1e-7:
            continue
        for iv in range(-max_i, max_i + 1):
            v = float(iv)
            x = u * c - v * s
            y = u * s + v * c
            if x - hx >= x_lo - 1e-7 and x + hx <= x_hi + 1e-7 and \
               y - hx >= y_lo - 1e-7 and y + hx <= y_lo + wp + 1e-7:
                p1.append((x, y, th))
    if not p1:
        return []
    out = list(p1)
    if mode == 0:
        out.extend([(q[1], q[0], 0.5 * math.pi - th) for q in p1])
    elif mode == 1:
        out.extend([(L - q[1], L - q[0], 0.5 * math.pi - th) for q in p1])
    elif mode == 2:
        out.extend([(L - q[0], L - q[1], th) for q in p1])
    elif mode == 3:
        out.extend([(L - q[1], q[0], th + 0.5 * math.pi) for q in p1])
        out.extend([(L - q[0], L - q[1], th) for q in p1])
        out.extend([(q[1], L - q[0], th + 0.5 * math.pi) for q in p1])
    if not non_overlap_batch(out):
        return []
    return out


def make_dp_jog_band(L: float, k: int, a_nw: int, a_se: int, w: int, th: float, u_off: float = 0.0) -> list:
    f = L - k
    stair = []
    for i in range(k):
        for j in range(k):
            dnw = i + (k - 1 - j)
            if a_nw - 3 <= dnw < a_nw:
                stair.append((i + 0.5, j + f + 0.5))
            dse = (k - 1 - i) + j
            if a_se - 3 <= dse < a_se:
                stair.append((i + f + 0.5, j + 0.5))
    st_arr = np.asarray(stair, dtype=np.float64) if stair else np.empty((0, 2))
    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (abs(c) + abs(s))
    rad = 0.5 + hx - 1e-7

    max_sl = int(math.ceil(L * 1.45)) + 3
    u_start = 0.7 * (c + s) + u_off
    feas = []
    nv = 65
    v_offsets = (np.arange(nv) - nv // 2) * 0.035

    for sl in range(max_sl):
        u = u_start + sl
        x_nom = u / (c + s)
        v_ctr = x_nom * (c - s) - 0.5 * (w - 1)
        f_sl = []
        for dv in v_offsets:
            v0 = v_ctr + dv
            ok = True
            for r in range(w):
                vr = v0 + r
                px = u * c - vr * s
                py = u * s + vr * c
                if px - hx < -1e-7 or px + hx > L + 1e-7 or py - hx < -1e-7 or py + hx > L + 1e-7:
                    ok = False
                    break
                if len(st_arr):
                    dx = px - st_arr[:, 0]
                    dy = py - st_arr[:, 1]
                    if np.any((np.abs(dx) < rad) & (np.abs(dy) < rad) &
                              (np.abs(dx * c + dy * s) < rad) & (np.abs(-dx * s + dy * c) < rad)):
                        ok = False
                        break
            if ok:
                f_sl.append(float(v0))
        feas.append(f_sl)

    dp_len = [[1] * len(fsl) for fsl in feas]
    dp_prev = [[-1] * len(fsl) for fsl in feas]
    best_len, end_sl, end_vi = 0, -1, -1
    for sl in range(max_sl):
        for vi, vc in enumerate(feas[sl]):
            if sl > 0:
                for pj, vp in enumerate(feas[sl - 1]):
                    if abs(vc - vp) <= 0.48 and dp_len[sl - 1][pj] + 1 > dp_len[sl][vi]:
                        dp_len[sl][vi] = dp_len[sl - 1][pj] + 1
                        dp_prev[sl][vi] = pj
            if dp_len[sl][vi] > best_len:
                best_len = dp_len[sl][vi]
                end_sl, end_vi = sl, vi

    if best_len < 3:
        return []
    band = []
    csl, cvi = end_sl, end_vi
    while csl >= 0 and cvi >= 0:
        u = u_start + csl
        v0 = feas[csl][cvi]
        for r in range(w):
            vr = v0 + r
            band.append((u * c - vr * s, u * s + vr * c, th))
        cvi = dp_prev[csl][cvi]
        csl -= 1
    band.reverse()
    return band


def build_candidates(n: int, k: int, L_probe: float) -> list:
    cands = []

    # Fam A: Single & Dual Asymmetric Diamonds / Rectangles (with off-center & multi-corner)
    for deg in [35.0, 40.0, 43.0, 45.0]:
        th = math.radians(deg)
        for m in [4, 5, 6, 7]:
            for cx1 in [m * 0.72 + 0.5, m * 0.72 + 2.0]:
                for cy1 in [L_probe - m * 0.72 - 0.5, L_probe - m * 0.72 - 2.5]:
                    b1 = make_block(L_probe, m, m, th, cx1, cy1, 0)
                    b2 = make_block(L_probe, m, m, th, L_probe - cy1, L_probe - cx1, 0)
                    if b1 and b2 and non_overlap_batch(b1 + b2):
                        cands.append((f"#{len(cands)}_2blk_{m}x{m}_d{deg}", "2blk", L_probe, b1 + b2, 0.0, False))

    # Fam B: 4-Block C4 Chiral Pinwheel (`a x b` blocks at 14..42 deg around center)
    for deg in [14.0, 22.0, 30.0, 37.0, 42.0]:
        th = math.radians(deg)
        for a, b in [(3, 2), (4, 2), (4, 3), (5, 3), (6, 3), (6, 4)]:
            for d_out in [0.24 * L_probe, 0.34 * L_probe]:
                for d_side in [-0.18 * L_probe, 0.0, 0.18 * L_probe]:
                    bx0, by0 = 0.5 * L_probe + d_side, 0.5 * L_probe - d_out
                    blk = make_block(L_probe, a, b, th, bx0, by0, 0)
                    if not blk:
                        continue
                    half_L = 0.5 * L_probe
                    all4 = []
                    for rot in range(4):
                        cr, sr = math.cos(rot * 0.5 * math.pi), math.sin(rot * 0.5 * math.pi)
                        for qx, qy, qth in blk:
                            dx, dy = qx - half_L, qy - half_L
                            all4.append((half_L + dx * cr - dy * sr, half_L + dx * sr + dy * cr, qth + rot * 0.5 * math.pi))
                    if non_overlap_batch(all4):
                        cands.append((f"#{len(cands)}_4pin_{a}x{b}_d{deg}", "4pin", L_probe, all4, 0.0, False))

    # Fam C: Oblique-Cut n=971 / n=908 / 4-Edge Chiral Frames (`wp=1.45` narrow edge strips included!)
    for deg in [12.0, 18.0, 25.2, 32.0, 36.87]:
        th = math.radians(deg)
        for wp in [1.45, 2.3, 3.4, 4.8]:
            for x_lo in [0.0, 2.0, 6.0]:
                if x_lo >= L_probe - 6.0:
                    continue
                for dx_hi in [0.0, 2.5]:
                    x_hi = L_probe - dx_hi
                    if x_hi - x_lo < 5.0:
                        continue
                    for u_off in [0.0, 2.5]:
                        u_cut = x_lo * math.cos(th) + u_off
                        for md in [-1, 0, 1, 3]:
                            t971 = make_971_oblique(L_probe, th, wp, x_lo, x_hi, 0.0, u_cut, md)
                            if t971 and len(t971) < n:
                                cands.append((f"#{len(cands)}_971_d{deg}_w{wp}_m{md}", "971", L_probe, t971, 0.0, False))

    # Fam D: n=847 Hybrids (45-deg m x m off-center + 1 or 2 non-45 corner pockets)
    for m_dim in range(max(5, k - 12), min(k - 3, 20), 2):
        hspan = 0.5 * m_dim * math.sqrt(2.0)
        if 2.0 * hspan > L_probe - 0.3:
            continue
        for off in [0.5, 2.5]:
            dcx = L_probe - hspan - 0.5 - off
            dcy = hspan + 0.5 + off * 0.25
            b45 = make_block(L_probe, m_dim, m_dim, 0.25 * math.pi, dcx, dcy, 0)
            if not b45:
                continue
            for pdeg in [24.0, 31.0, 37.4]:
                for pw, ph in [(2, 4), (3, 4), (3, 5)]:
                    pkt = make_block(L_probe, pw, ph, math.radians(pdeg), 2.8, L_probe - 3.2, 0)
                    if pkt and non_overlap_batch(b45 + pkt):
                        cands.append((f"#{len(cands)}_847_m{m_dim}_pd{pdeg}_{pw}x{ph}", "847", L_probe, b45 + pkt, 0.0, False))

    # Fam E: Calibrated DP Jogging Diagonal Bands
    for deg in [26.0, 33.0, 34.58, 39.0, 44.0]:
        th = math.radians(deg)
        for w in [2, 3, 4, 5]:
            for a_nw in range(max(4, k - w - 3), k - w + 2):
                for u_off in [0.0, 0.5, 0.96]:
                    dp = make_dp_jog_band(L_probe, k, a_nw, a_nw, w, th, u_off)
                    if dp and len(dp) < n:
                        cands.append((f"#{len(cands)}_dp_w{w}_d{deg}_a{a_nw}_u{u_off}", "dp", L_probe, dp, 0.0, False))

    # Area prefilter so hopeless giant voids skip HiGHS
    kept = []
    for item in cands:
        tilted = item[3]
        if len(tilted) + int(L_probe * L_probe - len(tilted)) >= n - 2:
            kept.append(item)
    return kept


def main():
    t0 = time.time()
    # Quick calibration check on n=403 DP band + exact HiGHS
    L403 = 20.564935730
    dp403_best = 0
    for a in range(15, 19):
        for u_off in [0.0, 0.35, 0.70, 0.96]:
            b = make_dp_jog_band(L403, 20, a, a, 3, math.radians(34.58), u_off)
            if b:
                _, fam, tot, nt, nup, _ = build_grid9_and_solve(("c403", "dp", L403, b, 0.0, False))
                dp403_best = max(dp403_best, tot)
    print(f"Calib DP band on n=403 with exact HiGHS: {dp403_best}/403", flush=True)

    targets = [
        (401, 20, 20.535533905933, 20.526135333409),
        (626, 25, 25.520815280171, 25.511048764219),
        (842, 29, 29.520815280171, 29.502152707903),
    ]

    summary = []
    u = None

    with ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn")) as ex:
        for n, k, L_base, L_front in targets:
            tn = time.time()
            L_probe = L_base - 0.001
            cands = build_candidates(n, k, L_probe)
            print(f"\nn={n} (L_probe={L_probe:.6f}): evaluating {len(cands)} candidates via exact HiGHS...", flush=True)
            res = []
            for start in range(0, len(cands), 140):
                res.extend(ex.map(build_grid9_and_solve, cands[start:start + 140], chunksize=4))
                best_so_far = max((r[2] for r in res), default=0)
                print(f"  progress {len(res)}/{len(cands)} (best={best_so_far}/{n}, {time.time()-tn:.1f}s)", flush=True)

            by_fam = {}
            by_fam_list = {}
            for tag, fam, tot, nt, nup, up_pts in res:
                by_fam_list.setdefault(fam, []).append((tot, tag, nt, nup))
                if fam not in by_fam or tot > by_fam[fam][0]:
                    by_fam[fam] = (tot, tag, nt, nup)
            for fam, (tot, tag, nt, nup) in sorted(by_fam.items()):
                print(f"  fam={fam:4s}: exact_Grid9={tot:3d}/{n} [{tag} nt={nt} up={nup}]", flush=True)

            # Top 15 overall + top 3 per family -> interface-augmented + soft (-0.08) -> L-BFGS/LP
            top_tags = {r[0] for r in sorted(res, key=lambda x: -x[2])[:15]}
            for fam, lst in by_fam_list.items():
                for _, tag, _, _ in sorted(lst, key=lambda x: -x[0])[:3]:
                    top_tags.add(tag)
            fine_tasks = []
            cand_map = {c[0]: c for c in cands}
            for tag in sorted(top_tags):
                _, fam, L_, tilted, _, _ = cand_map[tag]
                fine_tasks.append((tag + "|aug0", fam, L_, tilted, 0.0, True))
                fine_tasks.append((tag + "|soft8", fam, L_, tilted, -0.08, True))
            fine_res = list(ex.map(build_grid9_and_solve, fine_tasks, chunksize=2))
            best_fine = max(fine_res, key=lambda x: x[2])
            print(f"  Top-25 fine+soft best: {best_fine[2]}/{n} ({best_fine[0]})", flush=True)

            win_L = L_base
            for tag, fam, tot, nt, nup, up_pts in fine_res:
                if tot >= n:
                    if u is None:
                        from poc.swarm_angles.poc import utils as u
                    base_tag = tag.split("|")[0]
                    tilted = np.asarray(cand_map[base_tag][3], dtype=np.float64)
                    # Smart trim worst overlapping upright squares if tot > n
                    if tot > n and len(tilted) > 0:
                        need = n - len(tilted)
                        # Score by min dist to tilted
                        d2 = np.min((up_pts[:, 0:1] - tilted[:, 0][None, :]) ** 2 +
                                    (up_pts[:, 1:2] - tilted[:, 1][None, :]) ** 2, axis=1)
                        keep_u = np.argsort(-d2)[:need]
                        up_pts = up_pts[keep_u]
                    else:
                        up_pts = up_pts[: n - len(tilted)]
                    all_x = np.concatenate([tilted[:, 0], up_pts[:, 0]]) if len(tilted) else up_pts[:, 0]
                    all_y = np.concatenate([tilted[:, 1], up_pts[:, 1]]) if len(tilted) else up_pts[:, 1]
                    all_th = np.concatenate([tilted[:, 2], np.zeros(len(up_pts))]) if len(tilted) else np.zeros(len(up_pts))
                    rx, ry, rth, rL, ok = u.relax_lbfgs(all_x, all_y, all_th, L_probe)
                    if ok:
                        lx, ly, lL, ok2 = u.fixed_angle_lp_tighten(rx, ry, rth, rounds=5, margin=1e-9)
                        if ok2 and lL < win_L - 1e-11:
                            ver = u.verify_official(n, lx, ly, rth)
                            if ver["ok"] and ver["L_lit"] < win_L - 1e-11:
                                win_L = ver["L_lit"]
                                df_out = pd.DataFrame(ver["rows"], columns=["id", "n", "square_id", "x", "y", "angle"])
                                df_out.to_csv(f"poc/swarm_angles/data/cand12b_n{n}.csv", index=False)
                                print(f"  *** CERTIFIED WIN n={n}: L_lit={win_L:.12f} (< {L_base:.12f}) via {tag}! ***", flush=True)

            summary.append({
                "n": n, "L_base": L_base, "L_front": L_front,
                "by_fam": {k_: v_[0] for k_, v_ in by_fam.items()},
                "fine_best": best_fine[2], "fine_tag": best_fine[0],
                "win_L": win_L, "seconds": round(time.time() - tn, 2),
            })

    with open("poc/swarm_angles/docs/idea12b_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nIdea 12B finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
