"""Idea 15B: Exact Grid9 HiGHS on large-k D=1 roots n=530 (k=23) and n=842 (k=29)
testing large rotated M x N cores, 2..5 anti-diagonal Pythagorean clusters,
calibrated bell/chev bands, and chiral 4pin frames.
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
from poc.swarm_angles.poc.idea14_clusters_highs import make_calibrated_chevron, make_bell_slipped_cluster


def build_large_k_cands(n: int, k: int, L_probe: float) -> list:
    cands = []
    mg = -0.0015
    tl = 2.0

    # 1) Rotated M x M and A x B blocks (M=10..24) at non-45 and 45deg, centered & shifted
    m_vals = [10, 12, 14, 15, 16, 17, 18, 20, 22] if k >= 29 else [10, 12, 14, 15, 16, 17, 18]
    for M in m_vals:
        for deg in [10.0, 22.6199, 30.0, 36.8699, 41.0, 45.0]:
            for dx, dy in [(0.0, 0.0), (-0.5, 0.0)]:
                blk = make_block(L_probe, M, M, math.radians(deg), 0.5 * L_probe + dx, 0.5 * L_probe + dy, 0)
                if blk and len(blk) < n:
                    cands.append((f"#{len(cands)}_rot_{M}x{M}_d{deg}_{dx}",
                                  "rot", L_probe, blk, mg, False, tl))

    # 2) Multi-cluster anti-diagonal (2..5 clusters)
    for m_cl in ([2, 3, 4, 5] if k >= 29 else [2, 3, 4]):
        for deg in [22.6199, 28.0725, 36.8699, 40.5]:
            for w, n_sl, cplus in [(3, 5, True), (4, 5, True), (4, 7, False)]:
                for l_off in [0.0, -2.0, -4.0]:
                    span = L_probe + l_off
                    if span < 10.0:
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
                                      "mcl", L_probe, pts, mg, False, tl))

    # 3) Single bell & chevron clusters (small to medium)
    for deg in [16.2602, 22.6199, 28.0725, 36.8699, 40.5]:
        for w in [3, 4, 5, 6]:
            for n_sl in [3, 5, 7, 9]:
                for dx, dy in [(0.0, 0.0), (-0.5, -0.5)]:
                    pts = make_bell_slipped_cluster(
                        L_probe, w, n_sl, deg, 0.20, 4.0 / 21.0, True, 0.5 * L_probe + dx, 0.5 * L_probe + dy
                    )
                    if pts and len(pts) < n:
                        cands.append((f"#{len(cands)}_bell_w{w}_n{n_sl}_d{deg}_{dx}",
                                      "bell", L_probe, pts, mg, False, tl))

    for w in [3, 4, 5]:
        for d_deg in [2.913, 6.0, 10.0]:
            for n_end in [2, 4, 6]:
                for v_off in [-0.32, 0.32]:
                    pts = make_calibrated_chevron(L_probe, w, n_end, 1, d_deg, v_off, 0.5 * L_probe, 0.5 * L_probe)
                    if pts and len(pts) < n:
                        cands.append((f"#{len(cands)}_chev_w{w}_e{n_end}_d{d_deg}_v{v_off}",
                                      "chev", L_probe, pts, mg, False, tl))

    # 4) Chiral 4pin corner/edge blocks
    for deg in [5.0, 12.0, 22.6199, 30.0, 36.8699]:
        th = math.radians(deg)
        for u1, v1 in [(4, 6), (5, 8), (6, 9), (4, 10)]:
            for pad in [0.0, 1.5, 3.0]:
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
                                  "4pin", L_probe, pts, mg, False, tl))
    return cands


def main():
    t0 = time.time()
    targets = [
        (530, 23, 23.535533905933),
        (842, 29, 29.520815280171),
    ]
    summary = []
    u = None
    with ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn")) as ex:
        for n, k, L_base in targets:
            tn = time.time()
            L_probe = L_base - 0.001
            cands = build_large_k_cands(n, k, L_probe)
            fam_cnt = {}
            for c in cands:
                fam_cnt[c[1]] = fam_cnt.get(c[1], 0) + 1
            print(f"\nn={n} (L_probe={L_probe:.6f}): evaluating {len(cands)} cands {fam_cnt}...", flush=True)
            res = list(ex.map(solve_grid9_tl, cands, chunksize=3))
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
                for _, tag, _, _ in sorted(lst, key=lambda x: -x[0])[:4]:
                    top_tags.add(tag)
            fine_tasks = []
            for tag in sorted(top_tags):
                _, fam, L_, tilted, _, _, _ = cand_map[tag]
                fine_tasks.append((tag + "|aug0", fam, L_, tilted, 0.0, True, 3.5))
                fine_tasks.append((tag + "|soft8", fam, L_, tilted, -0.08, True, 3.5))
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
                    rx, ry, rth, pen = u.relax_lbfgs(all_x, all_y, all_th, L_probe, maxiter=500)
                    if pen < 5e-4:
                        lx, ly, lL, ok2 = u.fixed_angle_lp_tighten(rx, ry, rth, rounds=6, margin=1e-9)
                        if ok2 and lL < win_L - 1e-11:
                            ver = u.verify_official(n, lx, ly, rth)
                            if ver["ok"] and ver["L_float"] < win_L - 1e-11:
                                win_L = ver["L_float"]
                                df_out = pd.DataFrame(ver["rows"])
                                df_out["n"] = n
                                df_out["id"] = [f"{n}_{sid}" for sid in df_out["square_id"]]
                                Path("poc/swarm_angles/data").mkdir(parents=True, exist_ok=True)
                                df_out[["id", "n", "square_id", "x", "y", "angle"]].to_csv(
                                    f"poc/swarm_angles/data/cand15b_n{n}.csv", index=False
                                )
                                print(f"  *** CERTIFIED WIN n={n}: L={win_L:.12f} (< {L_base:.12f}) via {tag}! ***", flush=True)

            summary.append({
                "n": n, "L_base": L_base, "cands": fam_cnt,
                "by_fam": {k_: v_[0] for k_, v_ in by_fam.items()},
                "fine_best": best_fine[2], "fine_tag": best_fine[0],
                "win_L": win_L, "seconds": round(time.time() - tn, 2),
            })

    with open("poc/swarm_angles/docs/idea15b_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nIdea 15B finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
