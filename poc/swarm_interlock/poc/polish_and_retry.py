#!/usr/bin/env python3
"""Focused post-sweep polish of n=851 + high-effort retry on razor-close misses (n=684, n=794)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MECHWALK_POC = Path(__file__).resolve().parents[3] / "poc" / "swarm_mechwalk" / "poc"
if str(MECHWALK_POC) not in sys.path:
    sys.path.append(str(MECHWALK_POC))

import interlock_lib as il  # noqa: E402
import interlock_eval as ie  # noqa: E402
import brick_slp  # noqa: E402
import frontier_io  # noqa: E402
import mechwalk_lib as ml  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402


def polish_n851() -> None:
    wp = il.DATA_DIR / "win_n851.json"
    d = json.loads(wp.read_text())
    x = np.array([float(r["x"]) for r in d["rows"]], dtype=np.float64)
    y = np.array([float(r["y"]) for r in d["rows"]], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in d["rows"]], dtype=np.float64)
    L_cur = Decimal(d["L_lit"])
    L_f = float(L_cur)
    print(f"[Polish 851] Starting L = {L_cur}", flush=True)

    # 1) Soft-mode nullspace walk
    jac, info = ml.contact_jacobian(x, y, th, contact_tol=1e-4)
    _, vt = ml.soft_modes(jac, k=12)
    rng = np.random.default_rng(91851)
    best_x, best_y, best_th = x.copy(), y.copy(), th.copy()
    for step_idx in range(min(8, len(vt))):
        v = ml.remove_rigid_motion(vt[step_idx], best_x, best_y)
        v = v / (np.linalg.norm(v) + 1e-12)
        for sign in (-1.0, 1.0):
            for amp in (0.04, 0.10):
                xp = best_x + sign * amp * v[:851]
                yp = best_y + sign * amp * v[851:1702]
                tp = best_th + sign * amp * 0.5 * v[1702:]
                pen, bb = cpp_eval_sat_pen_bbox(xp, yp, tp)
                if pen > 0.05:
                    continue
                if pen > 1e-7:
                    xp, yp, tp, _ = u.relax_lbfgs(xp, yp, tp, bb + 0.002, maxiter=250)
                xs, ys, ts, Ls, ok = brick_slp.joint_xytheta_slp_tight(xp, yp, tp)
                if not ok:
                    continue
                cert = frontier_io.certify_against_frontier(851, xs, ys, ts, L_cur)
                if cert and cert["is_real_win"]:
                    print(f"  [Mechwalk win] mode={step_idx} sign={sign} amp={amp} -> L={cert['L_lit'][:16]} dL={cert['dL']:+.3e}", flush=True)
                    if ie.save_win_interlock(851, cert, d["label"] + f"_mw{step_idx}"):
                        L_cur = Decimal(cert["L_lit"])
                        best_x, best_y, best_th = xs, ys, ts
                        L_f = float(L_cur)
                        break

    # 2) Unit-seam masked melt from L_f down toward 29.657
    t_deg = np.degrees(np.mod(best_th, np.pi / 2))
    is_unit = np.minimum(t_deg, 90.0 - t_deg) < 2.0
    q_list, m_list, ls_list, labels = [], [], [], []
    for i in range(18):
        L0 = L_f + 0.020
        sc = L0 / L_f
        xs, ys, ts = best_x * sc, best_y * sc, best_th.copy()
        cx, cy = float(rng.uniform(7.0, 22.0)), float(rng.uniform(7.0, 22.0))
        rad = float(rng.uniform(6.0, 13.0))
        near = (((xs - cx) ** 2 + (ys - cy) ** 2) <= rad * rad) & is_unit
        if int(near.sum()) < 10:
            near = ((xs - cx) ** 2 + (ys - cy) ** 2) <= 7.5 ** 2
        nm = int(near.sum())
        xs[near] += rng.uniform(-0.12, 0.12, nm)
        ys[near] += rng.uniform(-0.12, 0.12, nm)
        ts[near] += rng.uniform(-0.15, 0.15, nm)
        q_list.append(full_to_free_sym(xs, ys, ts, L0))
        m_list.append(np.concatenate([near.astype(np.uint8)] * 3))
        tgt = L_f - float(rng.uniform(0.002, 0.014))
        ls_list.append([L0, 0.5 * (L0 + L_f), L_f - 0.0005, tgt])
        labels.append(f"p851_{i}")

    qb, qf, lf, _, po, _ = run_cpp_masked_melt(
        851, 1, q_list, m_list, ls_list, [6, 10, 14, 18],
        a_start=0.80, n_melt_stages=4, masked_iters=320, full_iters=320,
        n_threads=3, base_seed=95100,
    )
    order = sorted(range(len(labels)), key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])))[:6]
    tasks = []
    for i in order:
        use_f = lf[i] < 1e9
        L_ref = float(lf[i]) if use_f else ls_list[i][-1]
        xp, yp, tp = cpp_expand_sym(851, 1, L_ref, qf[i] if use_f else qb[i])
        tasks.append((851, xp, yp, tp, str(L_cur), d["label"] + "_" + labels[i]))
    with mp.Pool(processes=3) as pool:
        pol = pool.map(ie._polish_one, tasks)
    for r in pol:
        if r.get("ok") and r.get("is_real_win"):
            print(f"  [Melt851 win] L={r['L_lit'][:16]} dL={r['dL']:+.3e}", flush=True)
            ie.save_win_interlock(851, r, r["label"])


def retry_close_misses() -> None:
    targets = [684, 794]
    packs, best_L = il.fast_load_frontier(target_ns=targets)
    bricks = il.extract_all_bricks(packs, compress_key_bricks=True)
    for n in targets:
        print(f"\n[Retry] n={n}, L_inc={best_L[n]}", flush=True)
        cands = ie.find_candidates_for_n(n, bricks, float(best_L[n]), max_pairs=2)
        L_inc_dec = best_L[n]
        for c_idx, cand in enumerate(cands):
            ax, ay, at, aL, I_box, tag = cand["x"], cand["y"], cand["th"], cand["L"], cand["I_box"], cand["tag"]
            print(f"  [cand {c_idx+1}/{len(cands)}] {tag} (kx={cand['kx']}, ky={cand['ky']}, L0={aL:.6f})", flush=True)
            rng = np.random.default_rng(99000 + n * 17 + c_idx)
            q_list, m_list, ls_list, labels = il.build_interlock_melt_seeds(
                n, ax, ay, at, aL, I_box, rng, L_expand=0.060, L_inc=float(L_inc_dec)
            )
            qb, qf, lf, _, po, _ = run_cpp_masked_melt(
                n, 1, q_list, m_list, ls_list, [7, 12, 16, 20],
                a_start=0.86, n_melt_stages=4, masked_iters=420, full_iters=420,
                n_threads=3, base_seed=99500 + n * 13 + c_idx * 50,
            )
            n_feas = int(np.sum(lf < 1e9))
            min_po = float(np.min(po))
            order = sorted(range(len(labels)), key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])))[:6]
            tasks = []
            for i in order:
                use_f = lf[i] < 1e9
                L_ref = float(lf[i]) if use_f else ls_list[i][-1]
                xp, yp, tp = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
                tasks.append((n, xp, yp, tp, str(L_inc_dec), f"{tag}_{labels[i]}_retry"))
            with mp.Pool(processes=3) as pool:
                pol = pool.map(ie._polish_one, tasks)
            ok_list = [r for r in pol if r.get("ok")]
            if ok_list:
                b = min(ok_list, key=lambda r: r["dL"])
                print(f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} best_dL={b['dL']:+.6e} L={b['L_lit'][:14]}", flush=True)
                if b["is_real_win"]:
                    ie.save_win_interlock(n, b, b["label"])
                    L_inc_dec = min(L_inc_dec, Decimal(b["L_lit"]))
            else:
                print(f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} ALL FAILED SLP", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    polish_n851()
    retry_close_misses()
    print(f"\n[Done] polish_and_retry finished in {time.time()-t0:.1f}s", flush=True)
