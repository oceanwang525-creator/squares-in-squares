#!/usr/bin/env python3
"""H10 scaled solver against origin/main (e7f910b):
1. 4-corner 1-unit crops (prot=1..2) with protruding-square relocation (void/gap tuck) + masked melt.
2. +1 void-insert from lower-frac n-1 donors (e.g. 804->805, 862->863).
3. Checkpointed wins written to poc/swarm_homology/data/win_n{n}.csv and merged into final.csv.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h10_gate as hg  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402

REPO = HERE.parents[2]
DATA = HERE.parent / "data"
DOCS = HERE.parent / "docs"
FINAL_CSV = HERE.parent / "final.csv"


def load_origin_main_merged() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    txt = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "origin/main:poc/pell_corrugation/data/border-scan/merged.csv"],
        text=True,
    )
    df = pd.read_csv(io.StringIO(txt))
    out = {}
    for n, s in df.groupby("n"):
        out[int(n)] = hg.norm(s["x"].to_numpy(), s["y"].to_numpy(), s["angle"].to_numpy())
    return out


def find_crop_seeds(
    n_target: int,
    M: dict[int, tuple],
    max_prot: int = 2,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """Find 4-corner 1-unit crops n_src -> n_target with prot <= max_prot and Lt < L_inc - 1e-4."""
    L_inc = M[n_target][3]
    seeds = []
    cands = []
    for n_src, (x, y, t, L) in M.items():
        if n_src <= n_target or n_src > n_target + 80:
            continue
        Lt = L - 1.0
        if Lt >= L_inc - 1e-4:
            continue
        hx = hg.hx_of(t)
        corners = [
            ("BL", (x <= L - 1.05) & (y <= L - 1.05), 0.0, 0.0),
            ("TL", (x <= L - 1.05) & (y >= 1.05), 0.0, -1.0),
            ("BR", (x >= 1.05) & (y <= L - 1.05), -1.0, 0.0),
            ("TR", (x >= 1.05) & (y >= 1.05), -1.0, -1.0),
        ]
        for cname, keep, dx, dy in corners:
            if int(keep.sum()) != n_target:
                continue
            xk = x[keep] + dx
            yk = y[keep] + dy
            tk = t[keep].copy()
            hk = hx[keep]
            prot_mask = (xk - hk < -1e-6) | (xk + hk > Lt + 1e-6) | (yk - hk < -1e-6) | (yk + hk > Lt + 1e-6)
            prot = int(prot_mask.sum())
            if prot <= max_prot:
                cands.append((prot, Lt - L_inc, n_src, cname, xk, yk, tk, Lt, prot_mask))

    cands.sort(key=lambda r: (r[0], r[1]))
    seen_src = set()
    for prot, dL, n_src, cname, xk, yk, tk, Lt, prot_mask in cands:
        if (n_src, cname) in seen_src or len(seen_src) >= 2:
            continue
        seen_src.add((n_src, cname))
        p_idx = np.where(prot_mask)[0]
        ctrs = np.c_[xk[p_idx], yk[p_idx]] if len(p_idx) else np.array([[Lt / 2, Lt / 2]])
        seeds.append((f"crop_{n_src}_{cname}_raw", xk, yk, tk, Lt, ctrs))
        if len(p_idx) > 0:
            # Relocate protruding square(s) into top clearance voids at Lt
            x_base, y_base, t_base = xk.copy(), yk.copy(), tk.copy()
            keep_in = ~prot_mask
            vp = hg.clearance_points(x_base[keep_in], y_base[keep_in], t_base[keep_in], Lt, max(6, len(p_idx) + 3))
            if len(vp) >= len(p_idx):
                for v_shift in range(min(3, len(vp) - len(p_idx) + 1)):
                    xr, yr, tr = x_base.copy(), y_base.copy(), t_base.copy()
                    vctrs = []
                    for k_i, idx in enumerate(p_idx):
                        vx, vy, _ = vp[v_shift + k_i]
                        xr[idx], yr[idx], tr[idx] = vx, vy, 0.0
                        vctrs.append([vx, vy])
                    seeds.append((
                        f"crop_{n_src}_{cname}_v{v_shift}",
                        xr, yr, tr, Lt, np.vstack([ctrs, np.array(vctrs)]),
                    ))
            # For prot == 1, also tuck upright near the protruding wall along the edge gap
            if len(p_idx) == 1:
                idx = int(p_idx[0])
                px, py = float(xk[idx]), float(yk[idx])
                for k_t, delta in enumerate((-1.2, 0.0, 1.2)):
                    xt, yt, tt = x_base.copy(), y_base.copy(), t_base.copy()
                    if py > Lt - 1.0:
                        xt[idx] = np.clip(px + delta, 0.5, Lt - 0.5)
                        yt[idx] = Lt - 0.5
                    elif py < 1.0:
                        xt[idx] = np.clip(px + delta, 0.5, Lt - 0.5)
                        yt[idx] = 0.5
                    elif px > Lt - 1.0:
                        xt[idx] = Lt - 0.5
                        yt[idx] = np.clip(py + delta, 0.5, Lt - 0.5)
                    else:
                        xt[idx] = 0.5
                        yt[idx] = np.clip(py + delta, 0.5, Lt - 0.5)
                    tt[idx] = 0.0
                    seeds.append((
                        f"crop_{n_src}_{cname}_t{k_t}",
                        xt, yt, tt, Lt, np.array([[xt[idx], yt[idx]]]),
                    ))
    return seeds


def find_plus1_seeds(
    n_target: int,
    M: dict[int, tuple],
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """Try inserting 1 square into n_target - 1 when L(n_target - 1) < L(n_target) - 1e-3."""
    if (n_target - 1) not in M:
        return []
    x, y, t, L_prev = M[n_target - 1]
    L_inc = M[n_target][3]
    if L_prev >= L_inc - 1e-3:
        return []
    vp = hg.clearance_points(x, y, t, L_prev, 4)
    out = []
    for s, (vx, vy, _) in enumerate(vp[:3]):
        out.append((
            f"plus1_{n_target-1}_v{s}",
            np.append(x, vx),
            np.append(y, vy),
            np.append(t, 0.0 if s < 2 else 0.4),
            L_prev,
            np.array([[vx, vy]]),
        ))
    return out


def build_melt_jobs(seeds, L_inc: float, rng: np.random.Generator, per_seed: int = 5):
    q, m, ls, lab = [], [], [], []
    for name, X, Y, T, Lt, ctrs in seeds:
        L0 = Lt + 0.032
        off = 0.5 * (L0 - Lt)
        xb = X + off
        yb = Y + off
        h = hg.hx_of(T)
        xb = np.clip(xb, h + 1e-4, L0 - h - 1e-4)
        yb = np.clip(yb, h + 1e-4, L0 - h - 1e-4)

        gap = L_inc - Lt
        sched_a = [L0, L_inc - 0.05 * gap, L_inc - 0.35 * gap, L_inc - 0.75 * gap]
        sched_b = [L0, L_inc + 0.002, L_inc - 0.15 * gap, Lt + 0.0005]
        for s in range(per_seed):
            R = float(rng.uniform(4.8, 7.6))
            d2 = ((xb[:, None] - (ctrs[None, :, 0] + off)) ** 2 + (yb[:, None] - (ctrs[None, :, 1] + off)) ** 2).min(axis=1)
            near = d2 <= R * R
            near |= (xb - h < off) | (xb + h > L0 - off) | (yb - h < off) | (yb + h > L0 - off)
            x2, y2, t2 = xb.copy(), yb.copy(), T.copy()
            k = int(near.sum())
            if s > 0:
                x2[near] += rng.uniform(-0.09, 0.09, k)
                y2[near] += rng.uniform(-0.09, 0.09, k)
                t2[near] += rng.uniform(-0.22, 0.22, k)
            q.append(full_to_free_sym(x2, y2, t2, L0))
            mm = near.astype(np.uint8)
            m.append(np.concatenate([mm, mm, mm]))
            ls.append(sched_a if s % 2 == 0 else sched_b)
            lab.append(f"{name}_R{R:.1f}_s{s}")
    return q, m, ls, lab


def run_one(n: int, M: dict[int, tuple], threads: int, existing_best: float | None = None):
    L_inc = M[n][3]
    L_ref = min(L_inc, existing_best) if existing_best is not None else L_inc
    seeds = find_crop_seeds(n, M, max_prot=2) + find_plus1_seeds(n, M)
    if not seeds:
        print(f"[{n}] no qualifying homology seeds below L_inc={L_inc:.9f}", flush=True)
        return None
    # Keep up to 10 diverse seeds
    seeds = seeds[:10]
    rng = np.random.default_rng(7700 + n)
    t0 = time.time()
    q, m, ls, lab = build_melt_jobs(seeds, L_inc, rng, per_seed=4)
    qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
        n, 1, q, m, ls, [5, 8, 12, 15],
        a_start=0.78, n_melt_stages=4, masked_iters=180, full_iters=200,
        n_threads=threads, base_seed=51000 + n,
    )
    n_feas = int((lf < 1e9).sum())
    print(f"[{n}] seeds={len(seeds)} jobs={len(q)} melt {time.time()-t0:.1f}s feas={n_feas} "
          f"best_Lf={float(lf.min()):.9f} min_pen={float(po.min()):.2e} (L_inc={L_inc:.9f})", flush=True)

    order = sorted(range(len(lab)), key=lambda i: (0 if lf[i] < 1e9 else 1, lf[i] if lf[i] < 1e9 else po[i]))[:6]
    best = None
    for i in order:
        use_f = lf[i] < 1e9
        Lr = float(lf[i]) if use_f else ls[i][-1]
        x, y, th = cpp_expand_sym(n, 1, Lr, qf[i] if use_f else qb[i])
        v = hg.polish(n, x, y, th, L_inc)
        if v is None:
            continue
        Lv = v["L_float"]
        if best is None or Lv < best[0]:
            best = (Lv, lab[i], v)
    if best is None:
        print(f"[{n}] no valid polished candidate ({time.time()-t0:.1f}s)", flush=True)
        return {"n": n, "L_inc": L_inc, "win": False, "dt": round(time.time() - t0, 1)}

    Lv, label, v = best
    dL = Lv - L_inc
    is_win = dL <= -1e-9
    improved = Lv < L_ref - 1e-9
    print(f"[{n}] BEST {label}: L={Lv:.15f} dL={dL:+.6e} ({'REAL WIN' if is_win else 'miss'}) [{time.time()-t0:.1f}s]", flush=True)
    if is_win and improved:
        DATA.mkdir(parents=True, exist_ok=True)
        u.write_final_csv_from_wins({n: v}, DATA / f"win_n{n}.csv")
    return {"n": n, "L_inc": L_inc, "L_best": Lv, "dL": dL, "label": label, "win": is_win, "dt": round(time.time() - t0, 1)}


def rebuild_final_csv(M: dict[int, tuple]) -> list[tuple[int, float, float]]:
    """Rebuild poc/swarm_homology/final.csv from all verified wins strictly beating origin/main."""
    wins_out = {}
    summary = []
    for p in sorted(DATA.glob("win_n*.csv")):
        n = int(p.stem.split("_n")[1])
        df = pd.read_csv(p)
        x = df["x"].to_numpy(float)
        y = df["y"].to_numpy(float)
        th = np.radians(df["deg"].to_numpy(float))
        v = u.verify_official(n, x, y, th)
        if not v.get("ok"):
            continue
        L_inc = M[n][3]
        dL = v["L_float"] - L_inc
        if dL <= -1e-9:
            wins_out[n] = v
            summary.append((n, v["L_float"], dL))
    if wins_out:
        u.write_final_csv_from_wins(wins_out, FINAL_CSV)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[
        910, 863, 805, 924, 793, 332, 683, 445, 545, 450, 681, 582, 539, 534, 336, 686, 365, 584, 698,
        804, 984, 691, 745, 921, 693, 862, 923, 748,
    ])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-misses", type=int, default=12)
    a = ap.parse_args()
    M = load_origin_main_merged()
    threads = min(4, max(1, a.threads))

    existing = {}
    for p in DATA.glob("win_n*.csv"):
        n = int(p.stem.split("_n")[1])
        df = pd.read_csv(p)
        _, _, _, L_ex = hg.norm(df["x"].to_numpy(), df["y"].to_numpy(), np.radians(df["deg"].to_numpy()))
        existing[n] = L_ex

    results = []
    misses = 0
    for n in a.targets:
        res = run_one(n, M, threads, existing_best=existing.get(n))
        if res is not None:
            results.append(res)
            if res.get("win"):
                misses = 0
            else:
                misses += 1
        summary = rebuild_final_csv(M)
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "scale_results.json").write_text(json.dumps({"results": results, "wins": summary}, indent=2))
        if misses >= a.max_misses:
            print(f"Stopping early after {misses} consecutive non-wins.", flush=True)
            break

    print("FINAL WINS vs origin/main:", rebuild_final_csv(M), flush=True)


if __name__ == "__main__":
    main()
