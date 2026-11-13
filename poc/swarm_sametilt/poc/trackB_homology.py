#!/usr/bin/env python3
"""Track B: Homologous m -> m +/- k border extension and peel from GuestData n=910 & n=909 (and cascading wins)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sametilt_lib as st  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402

DOCS = st.DOCS_DIR


def make_border_extend_seed(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, layers: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extend a normalized packing [0, L]^2 by `layers` upright 1-unit L-strips on top & right."""
    cx, cy, ct = x.copy(), y.copy(), th.copy()
    cur_L = L
    for _ in range(layers):
        m = int(np.floor(cur_L))
        b = np.arange(0.5, m, 1.0)
        off = cur_L + 0.5 + 1e-8
        cx = np.concatenate([cx, np.full(m, off), b, [off]])
        cy = np.concatenate([cy, b, np.full(m, off), [off]])
        ct = np.concatenate([ct, np.zeros(m + m + 1)])
        cur_L += 1.0 + 1e-8
    return st.norm_pack(cx, cy, ct)


def build_melt_jobs_for_seed(
    label: str,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    Lt: float,
    centers: np.ndarray,
    L_inc: float,
    rng: np.random.Generator,
    n_seeds: int = 8,
    R_range: tuple[float, float] = (5.5, 11.5),
    allow_full: bool = True,
):
    n = len(X)
    pen, bb = cpp_eval_sat_pen_bbox(X, Y, T)
    ov = st.overlap_pairs(X, Y, T)
    hot = np.zeros(n, bool)
    for _, i, j in ov:
        hot[i] = hot[j] = True
    pts = [np.c_[X[hot], Y[hot]]] if hot.any() else []
    if len(centers):
        pts.append(centers)
    ctrs = np.vstack(pts) if pts else np.array([[Lt / 2, Lt / 2]])

    L_ref = max(Lt, bb if bb < Lt + 0.005 else Lt)
    L0 = L_ref + 0.035
    off = 0.5 * (L0 - L_ref)
    xb = X + off
    yb = Y + off
    h = st.hx_of(T)
    xb = np.clip(xb, h + 1e-4, L0 - h - 1e-4)
    yb = np.clip(yb, h + 1e-4, L0 - h - 1e-4)

    # Target schedule aiming between Lt and L_inc - 2e-4
    L_goal1 = min(Lt, L_inc - 2e-4)
    L_goal2 = min(L_inc - 2e-4, max(Lt + 0.003, 0.5 * (Lt + L_inc))) if Lt < L_inc - 4e-4 else L_inc - 2e-4
    L_mid = 0.5 * (L0 + L_goal1)
    sched1 = [L0, max(L_inc, Lt) + 0.008, L_mid, L_goal1]
    sched2 = [L0, max(L_inc, Lt) + 0.006, max(L_goal2, Lt) + 0.002, L_goal2]

    q, m, ls, lab = [], [], [], []
    for s in range(n_seeds):
        is_full = allow_full and n_seeds >= 2 and s == n_seeds - 1
        if is_full:
            near = np.ones(n, dtype=bool)
            x2, y2, t2 = xb.copy(), yb.copy(), T.copy()
            x2 += rng.uniform(-0.015, 0.015, n)
            y2 += rng.uniform(-0.015, 0.015, n)
            t2 += rng.uniform(-0.04, 0.04, n)
            q.append(full_to_free_sym(x2, y2, t2, L0))
            mm = near.astype(np.uint8)
            m.append(np.concatenate([mm, mm, mm]))
            ls.append(sched2)
            lab.append(f"{label}_FULL_s{s}")
            continue
        R = float(rng.uniform(R_range[0], R_range[1]))
        d2 = ((xb[:, None] - (ctrs[None, :, 0] + off)) ** 2 + (yb[:, None] - (ctrs[None, :, 1] + off)) ** 2).min(axis=1)
        near = d2 <= R * R
        near |= (xb - h < off) | (xb + h > L0 - off) | (yb - h < off) | (yb + h > L0 - off)
        if near.sum() < 12:
            # Always include nearest 20 squares to ctrs
            near[np.argsort(d2)[: min(20, n)]] = True
        x2, y2, t2 = xb.copy(), yb.copy(), T.copy()
        k = int(near.sum())
        jit = 0.0 if s == 0 else (0.05 if s % 2 == 1 else 0.10)
        if jit > 0:
            x2[near] += rng.uniform(-jit, jit, k)
            y2[near] += rng.uniform(-jit, jit, k)
            t2[near] += rng.uniform(-0.22, 0.22, k)
        q.append(full_to_free_sym(x2, y2, t2, L0))
        mm = near.astype(np.uint8)
        m.append(np.concatenate([mm, mm, mm]))
        ls.append(sched1 if s % 2 == 0 else sched2)
        lab.append(f"{label}_R{R:.1f}_s{s}")
    return q, m, ls, lab


def run_seed_batch(
    n_target: int,
    seed_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    L_inc_dec: Decimal,
    threads: int,
    rng: np.random.Generator,
    n_seeds_per_spec: int = 8,
    masked_iters: int = 200,
    full_iters: int = 220,
    top_polish: int = 6,
    R_range: tuple[float, float] = (5.5, 11.5),
    allow_full: bool = True,
) -> dict | None:
    if not seed_specs:
        return None
    L_inc = float(L_inc_dec)
    t0 = time.time()

    # Check if any raw seed already certifies as a win!
    best_cert = None
    best_label = None
    for sname, X, Y, T, Lt, _ in seed_specs:
        pen, _ = cpp_eval_sat_pen_bbox(X, Y, T)
        if pen < 1e-7:
            cert = st.polish_candidate(n_target, X, Y, T, L_inc_dec)
            if cert is not None:
                print(f"[{n_target}] raw-polish {sname}: L={cert['L_float']:.12f} dL={cert['dL']:+.6e}", flush=True)
                if cert["is_real_win"] and (best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"])):
                    best_cert = cert
                    best_label = f"{sname}_raw"
                    st.save_win_and_rebuild_final(n_target, best_cert, best_label)
                    print(f"[{n_target}] *** REAL WIN (raw-polish) *** {best_label} L={best_cert['L_lit'][:20]} dL={best_cert['dL']:+.6e}", flush=True)

    allq, allm, alls, alll = [], [], [], []
    for sname, X, Y, T, Lt, ctrs in seed_specs:
        q, m, ls, lab = build_melt_jobs_for_seed(
            sname,
            X,
            Y,
            T,
            Lt,
            ctrs,
            L_inc,
            rng,
            n_seeds=n_seeds_per_spec,
            R_range=R_range,
            allow_full=allow_full,
        )
        allq += q
        allm += m
        alls += ls
        alll += lab

    qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
        n_target,
        1,
        allq,
        allm,
        alls,
        [6, 10, 14, 18],
        a_start=0.80,
        n_melt_stages=4,
        masked_iters=masked_iters,
        full_iters=full_iters,
        n_threads=min(4, threads),
        base_seed=int(rng.integers(10000, 999999)),
    )
    n_feas = int((lf < 1e9).sum())
    print(f"[{n_target}] melt ({len(alll)} jobs) {time.time()-t0:.1f}s feas={n_feas}/{len(lf)} best_Lf={float(lf.min()):.9f} min_pen={float(po.min()):.2e}", flush=True)

    order = sorted(range(len(alll)), key=lambda i: (0 if lf[i] < 1e9 else 1, lf[i] if lf[i] < 1e9 else po[i]))[:top_polish]
    for i in order:
        use_f = lf[i] < 1e9
        Lr = float(lf[i]) if use_f else alls[i][-1]
        x, y, th = cpp_expand_sym(n_target, 1, Lr, qf[i] if use_f else qb[i])
        cert = st.polish_candidate(n_target, x, y, th, L_inc_dec)
        if cert is None:
            print(f"[{n_target}]   {alll[i]}: cppLf={lf[i]:.9f} pen={po[i]:.2e} -> polish=FAIL", flush=True)
            continue
        print(f"[{n_target}]   {alll[i]}: cppLf={lf[i]:.9f} -> L={cert['L_float']:.12f} dL={cert['dL']:+.6e}", flush=True)
        if cert["is_real_win"] and (best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"])):
            best_cert = cert
            best_label = alll[i]
            st.save_win_and_rebuild_final(n_target, best_cert, best_label)
            print(f"[{n_target}] *** REAL WIN *** {best_label} L={best_cert['L_lit'][:20]} dL={best_cert['dL']:+.6e}", flush=True)

    return best_cert


def generate_peel_seeds_for_target(
    n_target: int,
    donors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    L_inc: float,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """Generate BL/TL/BR peel seeds from same-tilt donors (n=910, 909, or cascaded wins)."""
    seeds = []
    for src_n, (x, y, t, L) in sorted(donors.items(), reverse=True):
        if src_n <= n_target:
            continue
        h = st.hx_of(t)
        for k in range(1, 7):
            Lt = L - k
            if Lt >= L_inc - 2e-4:
                continue
            for cname, keep, dx, dy in [
                ("BL", (x <= L - k - 0.02) & (y <= L - k - 0.02), 0.0, 0.0),
                ("TL", (x <= L - k - 0.02) & (y >= k + 0.02), 0.0, -float(k)),
                ("BR", (x >= k + 0.02) & (y <= L - k - 0.02), -float(k), 0.0),
            ]:
                cnt = int(keep.sum())
                diff_n = cnt - n_target
                if diff_n < -2 or diff_n > 4:
                    continue
                xk, yk, tk, hk = x[keep] + dx, y[keep] + dy, t[keep].copy(), h[keep]
                prot_mask = (xk - hk < -1e-5) | (xk + hk > Lt + 1e-5) | (yk - hk < -1e-5) | (yk + hk > Lt + 1e-5)
                p_idx = np.where(prot_mask)[0]
                if len(p_idx) > 16:
                    continue

                Lt_use = min(L_inc - 5e-4, max(Lt, L_inc - 0.018))
                if diff_n == 0:
                    ctrs = np.c_[xk[p_idx], yk[p_idx]] if len(p_idx) else np.array([[Lt / 2, Lt / 2]])
                    seeds.append((f"peel{k}_{src_n}_{cname}_raw", xk, yk, tk, Lt_use, ctrs))
                    degk = np.degrees(tk) % 90.0
                    if cname == "BL":
                        left65 = (degk > 50.0) & (degk < 80.0) & (xk < 11.2) & (yk > 9.5)
                        if left65.any():
                            xs, ys, ts = xk.copy(), yk.copy(), tk.copy()
                            ys[left65] -= 0.32
                            seeds.append((
                                f"peel{k}_{src_n}_BL_slide65",
                                xs, ys, ts, Lt_use,
                                np.array([[5.3, 9.4], [5.3, max(10.0, Lt - 1.5)]]),
                            ))
                    if cname == "TL":
                        bot65 = (degk > 50.0) & (degk < 80.0) & (xk > 10.5) & (yk < 10.5)
                        if bot65.any():
                            xs, ys, ts = xk.copy(), yk.copy(), tk.copy()
                            dy_b = min(0.28, max(0.06, float(-(ys[bot65] - hk[bot65]).min()) + 0.02))
                            ys[bot65] += dy_b
                            seeds.append((
                                f"peel{k}_{src_n}_TL_slidebot",
                                xs, ys, ts, Lt_use,
                                np.array([[19.0, 1.0], [19.0, 9.5], [26.0, 3.3]]),
                            ))
                    if 1 <= len(p_idx) <= 6:
                        keep_in = ~prot_mask
                        vp = st.clearance_points(xk[keep_in], yk[keep_in], tk[keep_in], Lt, len(p_idx) + 5)
                        if len(vp) >= len(p_idx):
                            for shift in range(min(3, len(vp) - len(p_idx) + 1)):
                                xr, yr, tr = xk.copy(), yk.copy(), tk.copy()
                                vctrs = []
                                for idx_i, sq_i in enumerate(p_idx):
                                    vx, vy, _ = vp[shift + idx_i]
                                    xr[sq_i], yr[sq_i], tr[sq_i] = vx, vy, 0.0
                                    vctrs.append([vx, vy])
                                seeds.append((
                                    f"peel{k}_{src_n}_{cname}_v{shift}",
                                    xr, yr, tr, Lt,
                                    np.vstack([ctrs, np.array(vctrs)]),
                                ))
                elif diff_n > 0:
                    # Drop `diff_n` squares (prioritize protruding squares first, then highest-clearance)
                    drop_cands = []
                    if len(p_idx) >= diff_n:
                        # Sort protruding squares by how far they protrude
                        prot_amt = np.maximum.reduce([
                            -(xk[p_idx] - hk[p_idx]),
                            (xk[p_idx] + hk[p_idx]) - Lt,
                            -(yk[p_idx] - hk[p_idx]),
                            (yk[p_idx] + hk[p_idx]) - Lt,
                        ])
                        drop_idx = p_idx[np.argsort(-prot_amt)[:diff_n]]
                        drop_cands.append(("dropProt", drop_idx))
                        if len(p_idx) > diff_n:
                            # Also try dropping smallest protruding or different subset
                            drop_idx2 = p_idx[np.argsort(prot_amt)[:diff_n]]
                            drop_cands.append(("dropMinProt", drop_idx2))
                    else:
                        # Drop all protruding + highest-x/y axis squares near boundary
                        rem_need = diff_n - len(p_idx)
                        non_prot = np.where(~prot_mask)[0]
                        score_rm = np.maximum(xk[non_prot], yk[non_prot])
                        extra = non_prot[np.argsort(-score_rm)[:rem_need]]
                        drop_cands.append(("dropMix", np.concatenate([p_idx, extra])))

                    for dtag, d_idx in drop_cands:
                        km = np.ones(len(xk), bool)
                        km[d_idx] = False
                        xr, yr, tr = xk[km], yk[km], tk[km]
                        hr = hk[km]
                        pm2 = (xr - hr < -1e-5) | (xr + hr > Lt + 1e-5) | (yr - hr < -1e-5) | (yr + hr > Lt + 1e-5)
                        p2 = np.where(pm2)[0]
                        ctrs = np.c_[xr[p2], yr[p2]] if len(p2) else np.array([[xk[d_idx[0]], yk[d_idx[0]]]])
                        seeds.append((f"peel{k}_{src_n}_{cname}_{dtag}_raw", xr, yr, tr, Lt, ctrs))
                        if 1 <= len(p2) <= 6:
                            keep_in = ~pm2
                            vp = st.clearance_points(xr[keep_in], yr[keep_in], tr[keep_in], Lt, len(p2) + 4)
                            if len(vp) >= len(p2):
                                xv, yv, tv = xr.copy(), yr.copy(), tr.copy()
                                vctrs = []
                                for idx_i, sq_i in enumerate(p2):
                                    vx, vy, _ = vp[idx_i]
                                    xv[sq_i], yv[sq_i], tv[sq_i] = vx, vy, 0.0
                                    vctrs.append([vx, vy])
                                seeds.append((
                                    f"peel{k}_{src_n}_{cname}_{dtag}_reloc",
                                    xv, yv, tv, Lt,
                                    np.vstack([ctrs, np.array(vctrs)]),
                                ))
                else:
                    # diff_n < 0 (-1 or -2): insert |diff_n| into voids after relocating protruding
                    xr, yr, tr = xk.copy(), yk.copy(), tk.copy()
                    need_ins = -diff_n
                    vp = st.clearance_points(xr, yr, tr, Lt, len(p_idx) + need_ins + 4)
                    if len(vp) >= len(p_idx) + need_ins:
                        vctrs = []
                        for idx_i, sq_i in enumerate(p_idx):
                            vx, vy, _ = vp[idx_i]
                            xr[sq_i], yr[sq_i], tr[sq_i] = vx, vy, 0.0
                            vctrs.append([vx, vy])
                        for j in range(need_ins):
                            vx, vy, _ = vp[len(p_idx) + j]
                            xr = np.append(xr, vx)
                            yr = np.append(yr, vy)
                            tr = np.append(tr, 0.0)
                            vctrs.append([vx, vy])
                        seeds.append((f"peel{k}_{src_n}_{cname}_ins{need_ins}", xr, yr, tr, Lt, np.array(vctrs)))

    # Prioritize lower prot and larger L_inc - Lt gap
    def seed_priority(item):
        sname, X, Y, T, Lt, _ = item
        h = st.hx_of(T)
        prot = int(((X - h < -1e-5) | (X + h > Lt + 1e-5) | (Y - h < -1e-5) | (Y + h > Lt + 1e-5)).sum())
        pen, _ = cpp_eval_sat_pen_bbox(X, Y, T)
        return (prot, pen, Lt - L_inc)

    seeds.sort(key=seed_priority)
    return seeds[:6]


def generate_plusk_seeds(
    n_target: int,
    donors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    L_inc: float,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """Insert k = 1..5 squares into voids of smaller same-tilt donors (e.g. 909->911, 971->972..975, 850->851..855)."""
    candidates = []
    for src_n, (x, y, t, L) in donors.items():
        k = n_target - src_n
        if k < 1 or k > 5:
            continue
        if L >= L_inc - 2e-4:
            continue
        candidates.append((k, L, src_n, x, y, t))
    candidates.sort(key=lambda item: (item[0], item[1], -item[2]))

    seeds = []
    for k, L, src_n, x, y, t in candidates:
        vp = st.clearance_points(x, y, t, L, max(10, k + 6), step=0.09, min_sep=0.80)
        if len(vp) < k:
            continue
        # Generate up to 3 placement combinations + angle choices
        combos = [list(range(k))]
        if len(vp) > k:
            combos.append([0] + list(range(2, k + 1)) if k > 1 else [1])
        if len(vp) > k + 1 and k <= 3:
            combos.append(list(range(1, k + 1)))
        Lt = min(L_inc - 5e-4, max(L + 0.005 * k, L_inc - 0.012))
        for ci, idxs in enumerate(combos):
            for ang_mode, ang_val in (("u0", 0.0), ("t65", np.radians(65.5))):
                if ci > 0 and ang_mode != "u0":
                    continue
                xr, yr, tr = x.copy(), y.copy(), t.copy()
                vctrs = []
                for idx in idxs:
                    vx, vy, _ = vp[idx]
                    xr = np.append(xr, vx)
                    yr = np.append(yr, vy)
                    tr = np.append(tr, ang_val)
                    vctrs.append([vx, vy])
                seeds.append((f"ins{k}_{src_n}_c{ci}_{ang_mode}", xr, yr, tr, Lt, np.array(vctrs)))
    return seeds[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    st.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    st.DATA_DIR.mkdir(parents=True, exist_ok=True)
    packs, best_L = st.load_frontier()
    rng = np.random.default_rng(20261113)

    # Proven donors from merged frontier + same-tilt atlas
    donors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for dn in (970, 972, 973, 911, 910, 909, 850, 793, 530, 442):
        if dn in packs:
            donors[dn] = packs[dn]

    # Load any previously saved sametilt wins into donors
    for p in sorted(st.DATA_DIR.glob("win_n*.json")):
        d = json.loads(p.read_text())
        wn = int(d["n"])
        rows = d["rows"]
        xw = np.array([float(r["x"]) for r in rows])
        yw = np.array([float(r["y"]) for r in rows])
        tw = np.array([float(r["angle"]) for r in rows])
        wp = st.norm_pack(xw, yw, tw)
        if wn not in donors or wp[3] < donors[wn][3]:
            donors[wn] = wp

    default_targets = [
        # Immediate upward L-border extensions
        909, 577, 485, 971,
        # r = +9 downward peels (from 970, 909, 850, 793)
        850, 738, 685, 634,
        # r = +10 downward peels (from 910, 971)
        851, 794, 739, 686,
        # r = +11 downward peels (from 972, 911)
        911, 852, 795, 740,
        # r = +12 downward peels (from 973)
        912, 853, 796, 741,
        # Extra expansions around 972..975 and 854..856
        972, 973, 974, 975, 854, 855,
    ]
    targets = args.targets if args.targets else default_targets

    summary = {}
    for n_t in targets:
        for p in sorted(st.DATA_DIR.glob("win_n*.json")):
            d = json.loads(p.read_text())
            wn = int(d["n"])
            rows = d["rows"]
            w_pack = st.norm_pack(
                np.array([float(r["x"]) for r in rows]),
                np.array([float(r["y"]) for r in rows]),
                np.array([float(r["angle"]) for r in rows]),
            )
            if wn not in donors or w_pack[3] < donors[wn][3]:
                donors[wn] = w_pack
            if Decimal(d["L_lit"]) < best_L[wn]:
                best_L[wn] = Decimal(d["L_lit"])

        L_inc_dec = best_L[n_t]
        L_inc = float(L_inc_dec)
        print(f"\n=== Track B Target n={n_t} (frontier L={L_inc:.12f}) ===", flush=True)
        seeds = []
        # Check any exact 1-layer upward L-border extension from packs or donors
        ext_pool = dict(packs)
        ext_pool.update(donors)
        for src_n, (xs, ys, ts, Ls) in sorted(ext_pool.items()):
            ms = int(np.floor(Ls))
            if src_n + 2 * ms + 1 == n_t and Ls + 1.0 < L_inc - 1e-8:
                xe, ye, te, Le = make_border_extend_seed(xs, ys, ts, Ls, layers=1)
                seeds.append((f"ext{2*ms+1}_{src_n}", xe, ye, te, Le, np.array([[5.3, 9.4], [max(2.0, Ls - 2.0), 5.0]])))
        seeds += generate_peel_seeds_for_target(n_t, donors, L_inc)
        seeds += generate_plusk_seeds(n_t, donors, L_inc)
        if not seeds:
            print(f"[{n_t}] No eligible seeds below frontier L.", flush=True)
            continue

        cert = run_seed_batch(
            n_t,
            seeds[:8],
            L_inc_dec,
            args.threads,
            rng,
            n_seeds_per_spec=6,
            masked_iters=args.iters,
            full_iters=args.iters + 20,
            top_polish=3,
        )
        if cert is not None and cert["is_real_win"]:
            # Update best_L and cascading donor pool!
            best_L[n_t] = Decimal(cert["L_lit"])
            rows = cert["rows"]
            donors[n_t] = st.norm_pack(
                np.array([float(r["x"]) for r in rows]),
                np.array([float(r["y"]) for r in rows]),
                np.array([float(r["angle"]) for r in rows]),
            )
            summary[n_t] = {"L": cert["L_lit"], "dL": cert["dL"]}
        else:
            summary[n_t] = None
        (DOCS / "trackB_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
