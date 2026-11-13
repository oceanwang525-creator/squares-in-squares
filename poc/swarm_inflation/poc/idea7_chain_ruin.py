"""Idea 7: LP Dual Force-Chain Surgery + Warm-Start Surgical Pool Generator.

Never reads incumbent coordinates from merged.csv (only scalar L_ours via
polish_and_certify.load_incumbent_L). Extracts exact HiGHS LP dual shadow
prices on top distinct scratch basins (v5/v6 outputs) and generates targeted
surgical perturbations (Op A: chain-evict teleport into top holes, Op B:
bottleneck cluster angle flip, Op C: cut-plane strip compression, Op D: deep
band-tilt + dual-active melt) for batched RTX 4070 SUPER CUDA re-inflation.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import polish_and_certify as pc  # noqa: E402


def _read_top_bin(path: Path, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if len(raw) < 8:
        return []
    m, nn = struct.unpack("<ii", raw[:8])
    if nn != n or m <= 0:
        return []
    arr = np.frombuffer(raw[8:], dtype=np.float64).reshape(m, 3, n).copy()
    out = []
    for k in range(m):
        pos = np.stack([arr[k, 0], arr[k, 1]], axis=-1)
        th = arr[k, 2]
        out.append((pos, th))
    return out


def load_unique_scratch_basins(
    n: int,
    bin_dirs: list[Path],
    data_dir: Path,
    max_keep: int = 12,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Load unique scratch basins from .bin exports and cand_v*.csv, sorted by L."""
    raw_list: list[tuple[np.ndarray, np.ndarray, float]] = []
    for d in bin_dirs:
        for suf in ("_fm0.bin", "_fm1.bin"):
            for pos, th in _read_top_bin(d / f"top_n{n}{suf}", n):
                p_f, lam = pc.minimal_feasible_dilation(pos, th)
                L_c = pc.exact_bbox_side(p_f, th)
                raw_list.append((p_f, th, float(L_c)))
    for csv_p in sorted(data_dir.glob(f"cand_v*_n{n}.csv")):
        try:
            df = pd.read_csv(csv_p)
            if len(df) != n:
                continue
            pos = df[["x", "y"]].to_numpy().astype(np.float64).copy()
            th = df["angle"].to_numpy().astype(np.float64).copy()
            p_f, _ = pc.minimal_feasible_dilation(pos, th)
            L_c = pc.exact_bbox_side(p_f, th)
            raw_list.append((p_f, th, float(L_c)))
        except Exception:
            continue

    raw_list.sort(key=lambda t: t[2])
    kept: list[tuple[np.ndarray, np.ndarray, float]] = []
    seen_sigs: list[tuple[float, int, float]] = []
    for pos, th, L_c in raw_list:
        deg = np.degrees(th) % 90.0
        up = np.minimum(deg, 90.0 - deg) < 10.0
        non_up = int((~up).sum())
        med_d = float(np.median(deg[~up])) if non_up > 0 else 0.0
        # Keep if topological signature or L differs cleanly
        dup = False
        for sL, snu, smd in seen_sigs:
            if abs(L_c - sL) < 8e-4 and abs(non_up - snu) <= 1 and abs(med_d - smd) < 0.8:
                dup = True
                break
        if not dup:
            kept.append((pos, th, L_c))
            seen_sigs.append((L_c, non_up, med_d))
            if len(kept) >= max_keep:
                break
    return kept


def extract_lp_duals_and_holes(
    pos: np.ndarray,
    th: np.ndarray,
    polish_rounds: int = 4,
) -> dict:
    """Polish a basin slightly and extract LP active dual loads, chains, and holes."""
    n = len(pos)
    p_lp, _ = pc.fixed_angle_lp_polish(pos, th, rounds=polish_rounds)
    p_slp, t_slp, _ = pc.lip_slp_polish(p_lp, th, rounds=polish_rounds)
    p_fin, L_fin = pc.fixed_angle_lp_polish(p_slp, t_slp, rounds=max(2, polish_rounds // 2))

    c = np.cos(t_slp)
    s = np.sin(t_slp)
    ext = 0.5 * (np.abs(c) + np.abs(s))
    x0 = float((p_fin[:, 0] - ext).min())
    y0 = float((p_fin[:, 1] - ext).min())
    x = p_fin[:, 0] - x0
    y = p_fin[:, 1] - y0
    L0 = float(max((x + ext).max(), (y + ext).max()))

    # Build sparse linearized LP around (x, y, t_slp)
    rows, cols, vals, ub = [], [], [], []
    row_meta: list[tuple[str, int, int]] = []
    rc = 0
    for i in range(n):
        rows.append(rc); cols.append(i); vals.append(-1.0)
        ub.append(float(x[i] - ext[i])); row_meta.append(("wL", i, -1)); rc += 1
        rows.extend([rc, rc]); cols.extend([i, 2 * n]); vals.extend([1.0, -1.0])
        ub.append(float(L0 - x[i] - ext[i])); row_meta.append(("wR", i, -1)); rc += 1
        rows.append(rc); cols.append(n + i); vals.append(-1.0)
        ub.append(float(y[i] - ext[i])); row_meta.append(("wB", i, -1)); rc += 1
        rows.extend([rc, rc]); cols.extend([n + i, 2 * n]); vals.extend([1.0, -1.0])
        ub.append(float(L0 - y[i] - ext[i])); row_meta.append(("wT", i, -1)); rc += 1

    e1 = np.stack([c, s], axis=1)
    e2 = np.stack([-s, c], axis=1)
    for i in range(n):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        m = (dx * dx + dy * dy) < 2.25
        if not np.any(m):
            continue
        idx = np.where(m)[0] + i + 1
        d = np.stack([dx[m], dy[m]], axis=1)
        for k, j in enumerate(idx):
            best_a = None
            best_sl = -1e9
            for ax_vec, rad in (
                (e1[i], 0.5 + 0.5 * (abs(e1[j] @ e1[i]) + abs(e2[j] @ e1[i]))),
                (e2[i], 0.5 + 0.5 * (abs(e1[j] @ e2[i]) + abs(e2[j] @ e2[i]))),
                (e1[j], 0.5 + 0.5 * (abs(e1[i] @ e1[j]) + abs(e2[i] @ e1[j]))),
                (e2[j], 0.5 + 0.5 * (abs(e1[i] @ e2[j]) + abs(e2[i] @ e2[j]))),
            ):
                proj = float(d[k] @ ax_vec)
                sl = abs(proj) - rad
                if sl > best_sl:
                    best_sl = sl
                    sgn = 1.0 if proj >= 0.0 else -1.0
                    best_a = (float(sgn * ax_vec[0]), float(sgn * ax_vec[1]), float(rad))
            ax, ay, rad = best_a
            cur_p = ax * float(d[k, 0]) + ay * float(d[k, 1])
            rows.extend([rc, rc, rc, rc])
            cols.extend([i, int(j), n + i, n + int(j)])
            vals.extend([ax, -ax, ay, -ay])
            ub.append(cur_p - rad)
            row_meta.append(("pair", i, int(j)))
            rc += 1

    A = coo_matrix((vals, (rows, cols)), shape=(rc, 2 * n + 1)).tocsc()
    c_obj = np.zeros(2 * n + 1)
    c_obj[-1] = 1.0
    bounds = [(-0.02, 0.02)] * (2 * n) + [(-0.1, 0.1)]
    res = linprog(c_obj, A_ub=A, b_ub=np.array(ub), bounds=bounds, method="highs")

    sq_load = np.zeros(n, dtype=np.float64)
    active_adj: list[list[int]] = [[] for _ in range(n)]
    wall_pins: list[tuple[str, int, float]] = []
    if res.success and res.ineqlin is not None:
        marg = -res.ineqlin.marginals
        for r in range(rc):
            m_val = float(marg[r])
            if m_val <= 1e-4:
                continue
            typ, i, j = row_meta[r]
            if typ == "pair":
                sq_load[i] += m_val
                sq_load[j] += m_val
                active_adj[i].append(j)
                active_adj[j].append(i)
            else:
                sq_load[i] += m_val
                wall_pins.append((typ, i, m_val))

    # Top-5 largest holes on [0.5, L0-0.5]^2 (120x120 coarse search)
    G = 120
    gx = np.linspace(0.50, L0 - 0.50, G)
    GX, GY = np.meshgrid(gx, gx)
    d2 = (GX[..., None] - x[None, None, :]) ** 2 + (GY[..., None] - y[None, None, :]) ** 2
    md = np.sqrt(d2.min(axis=-1))
    holes: list[tuple[float, float, float]] = []
    for flat_idx in np.argsort(md.ravel())[::-1]:
        iy, ix = divmod(int(flat_idx), G)
        hx, hy, hr = float(gx[ix]), float(gx[iy]), float(md[iy, ix])
        if all((hx - ax) ** 2 + (hy - ay) ** 2 > 3.2 for ax, ay, _ in holes):
            holes.append((hx, hy, hr))
            if len(holes) >= 5:
                break

    return {
        "x": x,
        "y": y,
        "th": t_slp,
        "L": L0,
        "sq_load": sq_load,
        "active_adj": active_adj,
        "wall_pins": wall_pins,
        "holes": holes,
    }


def generate_surgical_pool(
    n: int,
    basins_info: list[dict],
    inc_L: float,
    count: int = 512,
    rng_seed: int = 707,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build `count` surgical ruin-and-recreate seeds from LP dual chains."""
    rng = np.random.default_rng(rng_seed + n * 13)
    out_x = np.empty((count, n), dtype=np.float64)
    out_y = np.empty((count, n), dtype=np.float64)
    out_th = np.empty((count, n), dtype=np.float64)
    nb = len(basins_info)
    cand_angs = np.array([0.0, math.pi / 4.0, -math.pi / 4.0, 0.68, -0.68, 0.4636, -0.4636])

    for idx in range(count):
        b = basins_info[idx % nb]
        x = b["x"].copy()
        y = b["y"].copy()
        th = b["th"].copy()
        L0 = b["L"]
        sq_load = b["sq_load"]
        adj = b["active_adj"]
        holes = b["holes"]

        # Scale into [inc_L - 0.008, inc_L - 0.0005]
        L_tgt = inc_L - rng.uniform(0.0008, 0.009)
        sc = L_tgt / L0
        x *= sc
        y *= sc

        deg = np.degrees(th) % 90.0
        non_up = np.minimum(deg, 90.0 - deg) > 10.0
        med_th_band = float(np.median(th[non_up])) if np.any(non_up) else (math.pi / 4.0)

        top_dual = np.argsort(sq_load)[::-1]
        n_active = max(6, int((sq_load > 0.05).sum()))
        top_pool = top_dual[: min(18, n_active)]
        op = idx % 4

        if op == 0 and holes:
            # Op A: Chain-evict teleport into top hole + melt around both ends
            p = int(rng.choice(top_pool))
            hx, hy, _ = holes[int(rng.integers(0, min(3, len(holes))))]
            hx_s, hy_s = hx * sc, hy * sc
            px_old, py_old = float(x[p]), float(y[p])

            # Optionally move 1 active neighbor too
            movers = [p]
            if adj[p] and rng.random() < 0.35:
                movers.append(int(rng.choice(adj[p])))
            for m_k, mi in enumerate(movers):
                x[mi] = np.clip(hx_s + (0.0 if m_k == 0 else rng.uniform(-0.7, 0.7)), 0.48, L_tgt - 0.48)
                y[mi] = np.clip(hy_s + (0.0 if m_k == 0 else rng.uniform(-0.7, 0.7)), 0.48, L_tgt - 0.48)
                th[mi] = float(med_th_band if rng.random() < 0.5 else rng.choice(cand_angs))

            d2_old = (x - px_old) ** 2 + (y - py_old) ** 2
            d2_new = (x - hx_s) ** 2 + (y - hy_s) ** 2
            melt = (d2_old < 7.0) | (d2_new < 5.5)
            x[melt] += rng.uniform(-0.18, 0.18, size=int(melt.sum()))
            y[melt] += rng.uniform(-0.18, 0.18, size=int(melt.sum()))
            th[melt] += rng.uniform(-0.35, 0.35, size=int(melt.sum()))

        elif op == 1:
            # Op B: Bottleneck cluster coherent angle flip + seam melt
            seed_p = int(rng.choice(top_pool[: min(8, len(top_pool))]))
            cluster = {seed_p}
            frontier = [seed_p]
            target_sz = int(rng.integers(3, 10))
            while frontier and len(cluster) < target_sz:
                cur = frontier.pop(0)
                for nb_i in adj[cur]:
                    if nb_i not in cluster and sq_load[nb_i] > 0.15:
                        cluster.add(nb_i)
                        frontier.append(nb_i)
                        if len(cluster) >= target_sz:
                            break
            cl_idx = np.array(sorted(cluster), dtype=int)
            dth = float(rng.choice([-0.45, -0.30, -0.18, 0.18, 0.30, 0.45]))
            th[cl_idx] += dth
            x[cl_idx] += rng.uniform(-0.12, 0.12, size=len(cl_idx))
            y[cl_idx] += rng.uniform(-0.12, 0.12, size=len(cl_idx))
            # Soften immediate neighborhood
            cx_m, cy_m = float(x[cl_idx].mean()), float(y[cl_idx].mean())
            near = ((x - cx_m) ** 2 + (y - cy_m) ** 2) < 9.0
            x[near] += rng.uniform(-0.08, 0.08, size=int(near.sum()))
            y[near] += rng.uniform(-0.08, 0.08, size=int(near.sum()))
            th[near] += rng.uniform(-0.15, 0.15, size=int(near.sum()))

        elif op == 2:
            # Op C: Cut-plane strip compression bisecting highest-load bridge
            p = int(rng.choice(top_pool[: min(6, len(top_pool))]))
            cut_mode = int(rng.integers(0, 3))
            delta = float(rng.uniform(0.025, 0.065))
            if cut_mode == 0:
                mask_hi = x > x[p]
                x[mask_hi] -= delta
                seam = np.abs(x - x[p]) < 2.4
            elif cut_mode == 1:
                mask_hi = y > y[p]
                y[mask_hi] -= delta
                seam = np.abs(y - y[p]) < 2.4
            else:
                u = x - y
                u0 = u[p]
                mask_hi = u > u0
                x[mask_hi] -= 0.707 * delta
                y[mask_hi] += 0.707 * delta
                seam = np.abs(u - u0) < 2.6
            x[seam] += rng.uniform(-0.14, 0.14, size=int(seam.sum()))
            y[seam] += rng.uniform(-0.14, 0.14, size=int(seam.sum()))
            th[seam] += rng.uniform(-0.30, 0.30, size=int(seam.sum()))

        else:
            # Op D: Deep band-tilt + active-chain melt + micro-compression
            dth_b = float(rng.uniform(-0.12, 0.12))
            th[non_up] += dth_b
            act = sq_load > 0.05
            x[act] += rng.uniform(-0.11, 0.11, size=int(act.sum()))
            y[act] += rng.uniform(-0.11, 0.11, size=int(act.sum()))
            th[act] += rng.uniform(-0.22, 0.22, size=int(act.sum()))
            x += rng.uniform(-0.015, 0.015, size=n)
            y += rng.uniform(-0.015, 0.015, size=n)

        # Random symmetry transform for diversity
        if rng.random() < 0.5:
            x = L_tgt - x
            th = -th
        if rng.random() < 0.5:
            x, y = y.copy(), x.copy()
            th = -th

        out_x[idx] = np.clip(x, 0.45, L_tgt - 0.45)
        out_y[idx] = np.clip(y, 0.45, L_tgt - 0.45)
        out_th[idx] = th

    return out_x, out_y, out_th


def write_surg_bin(path: Path, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m, n = x.shape
    with path.open("wb") as f:
        f.write(struct.pack("<ii", m, n))
        for k in range(m):
            f.write(np.ascontiguousarray(x[k], dtype=np.float64).tobytes())
            f.write(np.ascontiguousarray(y[k], dtype=np.float64).tobytes())
            f.write(np.ascontiguousarray(th[k], dtype=np.float64).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", default="367,910,971")
    parser.add_argument("--bin-dirs", default="/tmp/v6_top,/tmp/v5_top")
    parser.add_argument("--out-dir", default="/tmp/v7_warm")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--polish-rounds", type=int, default=4)
    args = parser.parse_args()

    n_list = [int(t) for t in args.n_list.split(",") if t.strip()]
    bin_dirs = [Path(p) for p in args.bin_dirs.split(",") if p.strip()]
    data_dir = Path("poc/swarm_inflation/data")
    out_dir = Path(args.out_dir)

    for n in n_list:
        inc_L = float(pc.load_incumbent_L(n))
        basins = load_unique_scratch_basins(n, bin_dirs, data_dir, max_keep=10)
        if not basins:
            print(f"[surg] n={n}: no warm basins found, skipping")
            continue
        infos = [extract_lp_duals_and_holes(p, t, polish_rounds=args.polish_rounds) for p, t, _ in basins[:8]]
        best_pol_L = min(inf["L"] for inf in infos)
        sx, sy, sth = generate_surgical_pool(n, infos, inc_L, count=args.count, rng_seed=args.seed)
        out_p = out_dir / f"surg_n{n}.bin"
        write_surg_bin(out_p, sx, sy, sth)
        print(f"[surg] n={n}: basins={len(infos)} best_L={best_pol_L:.6f} (inc={inc_L:.6f}) -> wrote {args.count} surgical seeds to {out_p}", flush=True)


if __name__ == "__main__":
    main()
