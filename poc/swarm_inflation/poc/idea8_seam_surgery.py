"""Idea 8: Seam Phase-Slide, Corner-Tip Refold & Stoichiometry Surgery.

Never reads incumbent coordinates from merged.csv (only scalar inc_L via
polish_and_certify.load_incumbent_L). Reuses LP-dual extraction from
idea7_chain_ruin and applies targeted topological moves on scratch basins:
  - Op E (seam_slide): PCA-detected band tangent/normal shift of one upright
    grain to break pinch bridges and sweep inter-tooth phase + close seam gaps.
  - Op F (corner_refold): relocate blunt corner-tip tilted squares into flush
    0-deg wall pockets along x=0.5 / y=0.5 / x=L-0.5 / y=L-0.5.
  - Op G (seam_stoich): convert high-dual seam squares between band angle and
    0-deg (local band widening / narrowing) with local compression to inc_L.
  - Op H (dual_tilt_evict): bottleneck chain teleport + coherent band tilt.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import idea7_chain_ruin as i7  # noqa: E402
import polish_and_certify as pc  # noqa: E402


def _analyze_band_geometry(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L0: float
) -> dict:
    """Detect upright vs band squares and compute PCA band tangent/normal axes."""
    deg = ((np.degrees(th) % 90.0) + 45.0) % 90.0 - 45.0
    up_mask = np.abs(deg) < 7.0
    band_mask = ~up_mask
    pts = np.stack([x, y], axis=1)

    if int(band_mask.sum()) >= 8 and int(up_mask.sum()) >= 12:
        bpts = pts[band_mask]
        cb = bpts.mean(axis=0)
        cov = np.cov((bpts - cb).T)
        evals, evecs = np.linalg.eigh(cov)
        t_vec = evecs[:, 1] / max(1e-12, float(np.linalg.norm(evecs[:, 1])))
        n_vec = np.array([-t_vec[1], t_vec[0]], dtype=np.float64)
        med_th_band = float(np.median(th[band_mask]))
    else:
        cb = np.array([0.5 * L0, 0.5 * L0], dtype=np.float64)
        t_vec = np.array([math.sqrt(0.5), -math.sqrt(0.5)], dtype=np.float64)
        n_vec = np.array([math.sqrt(0.5), math.sqrt(0.5)], dtype=np.float64)
        med_th_band = math.pi / 4.0

    u_perp = (pts - cb) @ n_vec
    g_lo = up_mask & (u_perp < 0.0)
    g_hi = up_mask & (u_perp >= 0.0)
    if int(band_mask.sum()) > 0:
        b_min = float(u_perp[band_mask].min())
        b_max = float(u_perp[band_mask].max())
        seam_dist = np.minimum(np.abs(u_perp - b_min), np.abs(u_perp - b_max))
    else:
        seam_dist = np.abs(u_perp)

    return {
        "up_mask": up_mask,
        "band_mask": band_mask,
        "t_vec": t_vec,
        "n_vec": n_vec,
        "u_perp": u_perp,
        "g_lo": g_lo,
        "g_hi": g_hi,
        "seam_dist": seam_dist,
        "med_th_band": med_th_band,
    }


def _find_wall_pockets(x: np.ndarray, y: np.ndarray, L: float, top_k: int = 16) -> list[tuple[float, float, float]]:
    """Find best empty wall-flush positions along all 4 sides for 0-deg plugs."""
    pts = np.stack([x, y], axis=1)
    grid_1d = np.linspace(0.50, L - 0.50, max(24, int(round(L * 4.0))))
    cands: list[tuple[float, float]] = []
    for v in grid_1d:
        cands.extend([(0.50, float(v)), (L - 0.50, float(v)), (float(v), 0.50), (float(v), L - 0.50)])
    carr = np.array(cands, dtype=np.float64)
    d2 = ((carr[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    md = np.sqrt(d2.min(axis=-1))
    order = np.argsort(md)[::-1]
    out: list[tuple[float, float, float]] = []
    for idx in order:
        px, py, hr = float(carr[idx, 0]), float(carr[idx, 1]), float(md[idx])
        if all((px - ox) ** 2 + (py - oy) ** 2 > 1.25 for ox, oy, _ in out):
            out.append((px, py, hr))
            if len(out) >= top_k:
                break
    return out


def generate_idea8_pool(
    n: int,
    basins_info: list[dict],
    inc_L: float,
    count: int = 512,
    rng_seed: int = 808,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate targeted seam/corner/stoichiometry surgical starts from basins."""
    rng = np.random.default_rng(rng_seed + n)
    out_x = np.zeros((count, n), dtype=np.float64)
    out_y = np.zeros((count, n), dtype=np.float64)
    out_th = np.zeros((count, n), dtype=np.float64)
    cand_angs = np.array(
        [0.0, 0.32175, -0.32175, 0.46365, -0.46365, 0.678, -0.678, math.pi / 4.0],
        dtype=np.float64,
    )

    nb = len(basins_info)
    # Bias heavily toward top 3 distinct basins
    weights = np.array([1.0 / (k + 1.2) for k in range(nb)], dtype=np.float64)
    weights /= weights.sum()

    for idx in range(count):
        b_idx = int(rng.choice(nb, p=weights)) if idx >= nb else idx
        b = basins_info[b_idx]
        x = b["x"].copy()
        y = b["y"].copy()
        th = b["th"].copy()
        L0 = float(b["L"])
        sq_load = b["sq_load"]
        adj = b["active_adj"]
        holes = b["holes"]

        # Clamp strictly below inc_L across [inc_L - 0.009, inc_L - 0.0006]
        L_tgt = inc_L - rng.uniform(0.0006, 0.0090)
        sc = L_tgt / L0
        x *= sc
        y *= sc

        geom = _analyze_band_geometry(x, y, th, L_tgt)
        up_mask = geom["up_mask"]
        band_mask = geom["band_mask"]
        t_vec = geom["t_vec"]
        n_vec = geom["n_vec"]
        g_lo = geom["g_lo"]
        g_hi = geom["g_hi"]
        seam_dist = geom["seam_dist"]
        med_th_b = geom["med_th_band"]

        op = idx % 4
        if op == 0:
            # Op E: Seam Phase-Slide & Diagonal Gap Closure
            grp = g_hi if (g_hi.sum() >= 8 and (g_lo.sum() < 8 or rng.random() < 0.65)) else g_lo
            if not np.any(grp):
                grp = up_mask
            sgn = -1.0 if np.array_equal(grp, g_hi) else 1.0  # move inward toward band center
            ds = float(rng.uniform(-0.78, 0.78))
            dn = float(rng.uniform(0.04, 0.36)) * sgn
            dx_sh = ds * float(t_vec[0]) + dn * float(n_vec[0])
            dy_sh = ds * float(t_vec[1]) + dn * float(n_vec[1])
            x[grp] += dx_sh
            y[grp] += dy_sh
            # Optional slight group angle perturbation (+/- 2 deg)
            if rng.random() < 0.35:
                th[grp] += float(rng.uniform(-0.035, 0.035))
            # Melt the 2.4-wide seam strip around band boundary so teeth re-interlock
            seam = seam_dist < float(rng.uniform(1.1, 1.8))
            x[seam] += rng.uniform(-0.14, 0.14, size=int(seam.sum()))
            y[seam] += rng.uniform(-0.14, 0.14, size=int(seam.sum()))
            th[seam] += rng.uniform(-0.26, 0.26, size=int(seam.sum()))

        elif op == 1:
            # Op F: Blunt Corner-Tip Strip Refold into Flush Wall Pockets
            margin = float(rng.uniform(1.6, 3.0))
            in_wall_tip = (x < margin) | (x > L_tgt - margin) | (y < margin) | (y > L_tgt - margin)
            tip_idx = np.where(band_mask & in_wall_tip)[0]
            if len(tip_idx) == 0:
                tip_idx = np.where(in_wall_tip & (sq_load > 0.01))[0]
            if len(tip_idx) == 0:
                tip_idx = np.argsort(sq_load)[-10:]

            wall_pockets = _find_wall_pockets(x, y, L_tgt, top_k=14)
            k_move = min(len(tip_idx), len(wall_pockets), int(rng.integers(3, 11)))
            if k_move > 0:
                chosen = rng.choice(tip_idx, size=k_move, replace=False)
                for mk, sq_i in enumerate(chosen):
                    wx, wy, _ = wall_pockets[mk]
                    old_x, old_y = float(x[sq_i]), float(y[sq_i])
                    x[sq_i] = wx + float(rng.uniform(-0.08, 0.08))
                    y[sq_i] = wy + float(rng.uniform(-0.08, 0.08))
                    th[sq_i] = 0.0 if rng.random() < 0.72 else float(rng.choice(cand_angs))
                    near_tip = ((x - old_x) ** 2 + (y - old_y) ** 2) < 4.5
                    x[near_tip] += rng.uniform(-0.13, 0.13, size=int(near_tip.sum()))
                    y[near_tip] += rng.uniform(-0.13, 0.13, size=int(near_tip.sum()))
                    th[near_tip] += rng.uniform(-0.22, 0.22, size=int(near_tip.sum()))

        elif op == 2:
            # Op G: Seam Stoichiometry Shift (band <-> upright conversion)
            seam_cand = np.where(seam_dist < 1.9)[0]
            if len(seam_cand) < 4:
                seam_cand = np.argsort(sq_load)[-16:]
            k_conv = min(len(seam_cand), int(rng.integers(2, 9)))
            convert_to_up = bool(rng.random() < 0.55)
            sub = seam_cand[band_mask[seam_cand]] if convert_to_up else seam_cand[up_mask[seam_cand]]
            if len(sub) < 2:
                sub = seam_cand
            # Prioritize high LP dual load along the seam
            sub_order = sub[np.argsort(-sq_load[sub])]
            pick = sub_order[: min(len(sub_order), k_conv)]
            if len(pick) > 0:
                if convert_to_up:
                    th[pick] = float(rng.uniform(-0.03, 0.03))
                else:
                    th[pick] = med_th_b + float(rng.uniform(-0.05, 0.05))
                cx_m, cy_m = float(x[pick].mean()), float(y[pick].mean())
                x[pick] += rng.uniform(-0.18, 0.18, size=len(pick))
                y[pick] += rng.uniform(-0.18, 0.18, size=len(pick))
                near = ((x - cx_m) ** 2 + (y - cy_m) ** 2) < 9.0
                x[near] += rng.uniform(-0.12, 0.12, size=int(near.sum()))
                y[near] += rng.uniform(-0.12, 0.12, size=int(near.sum()))
                th[near] += rng.uniform(-0.22, 0.22, size=int(near.sum()))

        else:
            # Op H: LP-Dual Bottleneck Teleport + Coherent Band Angle Sweep
            dth_b = float(rng.uniform(-0.14, 0.14))
            th[band_mask] += dth_b
            top_dual = np.argsort(sq_load)[::-1][: min(14, n)]
            if holes and len(top_dual) > 0:
                p = int(rng.choice(top_dual))
                hx, hy, _ = holes[int(rng.integers(0, min(4, len(holes))))]
                px_o, py_o = float(x[p]), float(y[p])
                x[p] = np.clip(hx * sc, 0.48, L_tgt - 0.48)
                y[p] = np.clip(hy * sc, 0.48, L_tgt - 0.48)
                th[p] = float(rng.choice(cand_angs))
                melt = (((x - px_o) ** 2 + (y - py_o) ** 2) < 6.0) | (sq_load > 0.04)
            else:
                melt = sq_load > 0.04
            x[melt] += rng.uniform(-0.13, 0.13, size=int(melt.sum()))
            y[melt] += rng.uniform(-0.13, 0.13, size=int(melt.sum()))
            th[melt] += rng.uniform(-0.24, 0.24, size=int(melt.sum()))

        # Symmetry flip diversity
        if rng.random() < 0.5:
            x = L_tgt - x
            th = -th
        if rng.random() < 0.5:
            x, y = y.copy(), x.copy()
            th = -th

        out_x[idx] = np.clip(x, 0.46, L_tgt - 0.46)
        out_y[idx] = np.clip(y, 0.46, L_tgt - 0.46)
        out_th[idx] = th

    return out_x, out_y, out_th


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", default="367,910,972")
    parser.add_argument("--bin-dirs", default="/tmp/v7_top,/tmp/v6_top,/tmp/v5_top")
    parser.add_argument("--out-dir", default="/tmp/v8_warm")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=808)
    parser.add_argument("--polish-rounds", type=int, default=3)
    args = parser.parse_args()

    n_list = [int(t) for t in args.n_list.split(",") if t.strip()]
    bin_dirs = [Path(p) for p in args.bin_dirs.split(",") if p.strip()]
    data_dir = Path("poc/swarm_inflation/data")
    out_dir = Path(args.out_dir)

    for n in n_list:
        inc_L = float(pc.load_incumbent_L(n))
        basins = i7.load_unique_scratch_basins(n, bin_dirs, data_dir, max_keep=12)
        if not basins:
            print(f"[surg8] n={n}: no warm basins found, skipping", flush=True)
            continue
        infos = [i7.extract_lp_duals_and_holes(p, t, polish_rounds=args.polish_rounds) for p, t, _ in basins[:8]]
        best_pol_L = min(inf["L"] for inf in infos)
        sx, sy, sth = generate_idea8_pool(n, infos, inc_L, count=args.count, rng_seed=args.seed)
        out_p = out_dir / f"surg_n{n}.bin"
        i7.write_surg_bin(out_p, sx, sy, sth)
        print(
            f"[surg8] n={n}: basins={len(infos)} best_L={best_pol_L:.6f} (inc={inc_L:.6f}) -> {args.count} seeds -> {out_p}",
            flush=True,
        )


if __name__ == "__main__":
    main()
