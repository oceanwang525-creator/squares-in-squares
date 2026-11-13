"""Idea 10A: Gentle Elastic-L Cross-n Stoichiometry Transfer & Seam-Hole Surgery.

Generates precision surgical and cross-n stoichiometry transfer pools
(e.g. cand_v8_n969 @ 31.649192 -> n=971/972 via +2/+3 seam-hole insertion,
972 @ 31.708429 -> n=971/969 via high-dual bottleneck eviction, and same-n
gentle seam-hole interlock) designed specifically for `sat_lmin_cuda`
`fast_mode=2` (`a: 0.945 -> 1.000` gentle full-crystal breathing without
random void teleports). Never reads incumbent coordinates from merged.csv.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import idea7_chain_ruin as i7  # noqa: E402
import idea8_seam_surgery as i8  # noqa: E402
import polish_and_certify as pc  # noqa: E402


def find_diverse_holes(pos: np.ndarray, L: float, top_k: int = 10) -> list[tuple[float, float, float]]:
    """Find top_k well-separated empty void centers in [0.55, L - 0.55]^2."""
    g = np.linspace(0.56, L - 0.56, max(30, int(round(L * 3.5))))
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    d2 = ((pts[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    order = np.argsort(-d2)
    out: list[tuple[float, float, float]] = []
    for idx in order:
        px, py = float(pts[idx, 0]), float(pts[idx, 1])
        r = float(math.sqrt(max(0.0, float(d2[idx]))))
        if all((px - ox) ** 2 + (py - oy) ** 2 > 1.35 for ox, oy, _ in out):
            out.append((px, py, r))
            if len(out) >= top_k:
                break
    if not out:
        out.append((0.5 * L, 0.5 * L, 0.5))
    return out


def load_source_basins_for_target(
    n_target: int,
    bin_dirs: list[Path],
    data_dir: Path,
    polish_rounds: int = 2,
) -> list[dict]:
    """Load same-n and neighbor-n scratch basins with LP duals and diverse holes."""
    neighbors: dict[int, list[int]] = {
        969: [969, 972, 971],
        971: [969, 971, 972],
        972: [969, 972, 971],
        791: [791, 796],
        796: [796, 791],
    }
    src_ns = neighbors.get(n_target, [n_target])
    infos: list[dict] = []

    for n_src in src_ns:
        # First check explicit cand_v10 / cand_v9 / cand_v8 / cand_v7 CSVs in data_dir
        csv_cands = [
            data_dir / f"cand_v10_n{n_src}.csv",
            data_dir / f"cand_v9_n{n_src}.csv",
            data_dir / f"cand_v8_n{n_src}.csv",
            data_dir / f"cand_v7_n{n_src}.csv",
        ]
        seen_L: list[float] = []
        for cp in csv_cands:
            if not cp.exists():
                continue
            df = pd.read_csv(cp)
            if len(df) != n_src:
                continue
            p = df[["x", "y"]].to_numpy(dtype=np.float64)
            t = df["angle"].to_numpy(dtype=np.float64)
            L_val = pc.exact_bbox_side(p, t)
            if any(abs(L_val - sl) < 1e-5 for sl in seen_L):
                continue
            seen_L.append(L_val)
            inf = i7.extract_lp_duals_and_holes(p, t, polish_rounds=polish_rounds)
            inf["n_src"] = n_src
            inf["holes"] = find_diverse_holes( np.stack([inf["x"], inf["y"]], axis=1), float(inf["L"]), top_k=10 )
            infos.append(inf)

        # Also add top unique basins from bin_dirs for same-n
        if n_src == n_target:
            raw_b = i7.load_unique_scratch_basins(n_src, bin_dirs, data_dir, max_keep=6)
            for p, t, L_val in raw_b[:4]:
                if any(abs(L_val - sl) < 1e-4 for sl in seen_L):
                    continue
                seen_L.append(L_val)
                inf = i7.extract_lp_duals_and_holes(p, t, polish_rounds=max(1, polish_rounds - 1))
                inf["n_src"] = n_src
                inf["holes"] = find_diverse_holes(np.stack([inf["x"], inf["y"]], axis=1), float(inf["L"]), top_k=10)
                infos.append(inf)

    # Sort so tightest effective L (especially cand_v8_n969 @ 31.649) comes first
    infos.sort(key=lambda d: float(d["L"]) + (0.0 if d["n_src"] == 969 and n_target in (971, 972) else 0.005 * abs(n_target - d["n_src"])))
    return infos[:10]


def generate_idea10_pool(
    n_target: int,
    basins_info: list[dict],
    inc_L: float,
    count: int = 512,
    rng_seed: int = 1010,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate gentle surgical and cross-n stoichiometry trajectories for fast_mode=2."""
    rng = np.random.default_rng(rng_seed + n_target * 17)
    out_x = np.zeros((count, n_target), dtype=np.float64)
    out_y = np.zeros((count, n_target), dtype=np.float64)
    out_th = np.zeros((count, n_target), dtype=np.float64)

    pyth_angs = np.array([
        0.0,
        math.atan2(1.0, 2.0), -math.atan2(1.0, 2.0),
        math.atan2(5.0, 12.0), -math.atan2(5.0, 12.0),
        math.atan2(8.0, 15.0), -math.atan2(8.0, 15.0),
        math.atan2(3.0, 4.0), -math.atan2(3.0, 4.0),
        math.radians(40.53), math.radians(-38.48),
    ], dtype=np.float64)

    nb = len(basins_info)
    weights = np.array([1.0 / (k + 1.0) for k in range(nb)], dtype=np.float64)
    weights /= weights.sum()

    for idx in range(count):
        b_idx = idx % min(3, nb) if idx < count // 2 else int(rng.choice(nb, p=weights))
        b = basins_info[b_idx]
        x0 = b["x"].copy()
        y0 = b["y"].copy()
        th0 = b["th"].copy()
        L0 = float(b["L"])
        n_src = int(b.get("n_src", len(x0)))
        sq_load = b["sq_load"].copy()
        holes = b["holes"]
        delta_n = n_target - n_src

        # Choose target box L strictly below inc_L
        if L0 < inc_L - 0.002:
            # Source basin is ALREADY smaller than inc_L (e.g. 969 @ 31.6492 < inc(971)=31.6643)!
            L_tgt = float(rng.uniform(L0 + 0.0005, inc_L - 0.0006))
        else:
            L_tgt = inc_L - float(rng.uniform(0.0005, 0.0065))

        sc = L_tgt / L0
        x = x0 * sc
        y = y0 * sc
        th = th0.copy()

        geom = i8._analyze_band_geometry(x, y, th, L_tgt)
        up_mask = geom["up_mask"]
        band_mask = geom["band_mask"]
        t_vec = geom["t_vec"]
        n_vec = geom["n_vec"]
        g_lo = geom["g_lo"]
        g_hi = geom["g_hi"]
        seam_dist = geom["seam_dist"]
        med_th_b = geom["med_th_band"]

        if delta_n > 0:
            # Cross-n Stoichiometry Insertion (+1, +2, +3, +5) into top seam voids
            nh = len(holes)
            combos = list(itertools.combinations(range(min(8, nh)), min(delta_n, nh)))
            chosen_h = combos[idx % len(combos)] if combos else tuple(range(delta_n))
            ins_x = np.empty(delta_n, dtype=np.float64)
            ins_y = np.empty(delta_n, dtype=np.float64)
            ins_th = np.empty(delta_n, dtype=np.float64)

            for k in range(delta_n):
                h_i = chosen_h[k % len(chosen_h)]
                hx, hy, _ = holes[h_i]
                hx_s = np.clip(hx * sc + float(rng.normal(0.0, 0.04)), 0.54, L_tgt - 0.54)
                hy_s = np.clip(hy * sc + float(rng.normal(0.0, 0.04)), 0.54, L_tgt - 0.54)
                ins_x[k] = hx_s
                ins_y[k] = hy_s
                ang_mode = (idx + k) % 4
                if ang_mode == 0:
                    ins_th[k] = med_th_b + float(rng.normal(0.0, 0.03))
                elif ang_mode == 1:
                    ins_th[k] = 0.0
                elif ang_mode == 2:
                    ins_th[k] = -med_th_b + float(rng.normal(0.0, 0.03))
                else:
                    ins_th[k] = float(rng.choice(pyth_angs))

                # Gentle radial backoff of local neighbors around inserted square
                dx_n = x - hx_s
                dy_n = y - hy_s
                dist_n = np.sqrt(dx_n * dx_n + dy_n * dy_n) + 1e-9
                near = dist_n < 1.85
                push = np.where(near, (1.85 - dist_n) * float(rng.uniform(0.035, 0.075)), 0.0)
                x += (dx_n / dist_n) * push
                y += (dy_n / dist_n) * push
                th[near] += rng.normal(0.0, 0.025, size=int(near.sum()))

            x = np.concatenate([x, ins_x])
            y = np.concatenate([y, ins_y])
            th = np.concatenate([th, ins_th])

        elif delta_n < 0:
            # Cross-n Stoichiometry Eviction (-1, -2, -3, -5) of high LP-dual seam squares
            n_rem = -delta_n
            seam_pool = np.where((seam_dist < 2.2) | (sq_load > 0.02))[0]
            if len(seam_pool) < n_rem + 2:
                seam_pool = np.arange(n_src)
            # Rank by LP dual load + random tie-break
            scores = sq_load[seam_pool] + rng.uniform(0.0, 0.04, size=len(seam_pool))
            rem_local = np.argsort(-scores)[:n_rem]
            rem_idx = seam_pool[rem_local]
            keep_mask = np.ones(n_src, dtype=bool)
            keep_mask[rem_idx] = False

            rem_pts = np.stack([x[rem_idx], y[rem_idx]], axis=1)
            x = x[keep_mask]
            y = y[keep_mask]
            th = th[keep_mask]

            # Gently pull neighbors inward toward vacated slots to collapse box L
            for rx, ry in rem_pts:
                dx_n = rx - x
                dy_n = ry - y
                dist_n = np.sqrt(dx_n * dx_n + dy_n * dy_n) + 1e-9
                near = dist_n < 2.4
                pull = np.where(near, (2.4 - dist_n) * float(rng.uniform(0.03, 0.07)), 0.0)
                x += (dx_n / dist_n) * pull
                y += (dy_n / dist_n) * pull

        else:
            # Same-n precision gentle surgical operators for fast_mode=2
            op = idx % 5
            if op == 0 or op == 4:
                # Mode A / A2: Single or Dual-Grain Seam Phase-Slide & Active-Bridge Unbuckling
                if op == 0 or g_lo.sum() < 8 or g_hi.sum() < 8:
                    grp = g_hi if (g_hi.sum() >= 8 and (g_lo.sum() < 8 or idx % 2 == 0)) else g_lo
                    if not np.any(grp):
                        grp = up_mask
                    sgn = -1.0 if np.array_equal(grp, g_hi) else 1.0
                    ds = float(rng.uniform(-0.32, 0.32))
                    dn = float(rng.uniform(0.004, 0.045)) * sgn
                    x[grp] += ds * float(t_vec[0]) + dn * float(n_vec[0])
                    y[grp] += ds * float(t_vec[1]) + dn * float(n_vec[1])
                else:
                    # Simultaneous dual-grain inward squeeze + differential tangent shear
                    ds_hi = float(rng.uniform(-0.28, 0.28))
                    ds_lo = float(rng.uniform(-0.28, 0.28))
                    dn_hi = -float(rng.uniform(0.003, 0.026))
                    dn_lo = float(rng.uniform(0.003, 0.026))
                    x[g_hi] += ds_hi * float(t_vec[0]) + dn_hi * float(n_vec[0])
                    y[g_hi] += ds_hi * float(t_vec[1]) + dn_hi * float(n_vec[1])
                    x[g_lo] += ds_lo * float(t_vec[0]) + dn_lo * float(n_vec[0])
                    y[g_lo] += ds_lo * float(t_vec[1]) + dn_lo * float(n_vec[1])

                if rng.random() < 0.45:
                    th[band_mask] += float(rng.uniform(-0.022, 0.022))
                seam = (seam_dist < float(rng.uniform(1.1, 1.7))) | (sq_load > 0.025)
                j_pos = float(rng.uniform(0.010, 0.028))
                j_ang = float(rng.uniform(0.015, 0.040))
                x[seam] += rng.normal(0.0, j_pos, size=int(seam.sum()))
                y[seam] += rng.normal(0.0, j_pos, size=int(seam.sum()))
                th[seam] += rng.normal(0.0, j_ang, size=int(seam.sum()))

            elif op == 1:
                # Mode B: Seam Hole Interlock & 0-deg <-> Tilted Conversion
                k_holes = min(len(holes), int(rng.integers(1, 4)))
                for hi in range(k_holes):
                    hx, hy, _ = holes[hi]
                    hx_s, hy_s = hx * sc, hy * sc
                    d2_h = (x - hx_s) ** 2 + (y - hy_s) ** 2
                    near_h = np.argsort(d2_h)[: int(rng.integers(2, 5))]
                    for sq_i in near_h:
                        # Pull square 25%..55% toward void center and flip orientation
                        alpha = float(rng.uniform(0.20, 0.50))
                        x[sq_i] = (1.0 - alpha) * x[sq_i] + alpha * hx_s
                        y[sq_i] = (1.0 - alpha) * y[sq_i] + alpha * hy_s
                        if up_mask[sq_i]:
                            th[sq_i] = med_th_b + float(rng.normal(0.0, 0.03))
                        else:
                            th[sq_i] = 0.0 if rng.random() < 0.7 else float(rng.choice(pyth_angs))

            elif op == 2:
                # Mode C: LP-Dual Bottleneck Square Relocation to Best Seam Void
                top_dual = np.argsort(-sq_load)[: min(12, n_target)]
                k_mv = int(rng.integers(1, 3))
                chosen = rng.choice(top_dual, size=min(k_mv, len(top_dual)), replace=False)
                for mk, sq_i in enumerate(chosen):
                    hx, hy, _ = holes[(idx + mk) % min(5, len(holes))]
                    ox_i, oy_i = float(x[sq_i]), float(y[sq_i])
                    x[sq_i] = np.clip(hx * sc + float(rng.normal(0.0, 0.03)), 0.52, L_tgt - 0.52)
                    y[sq_i] = np.clip(hy * sc + float(rng.normal(0.0, 0.03)), 0.52, L_tgt - 0.52)
                    th[sq_i] = med_th_b if rng.random() < 0.5 else 0.0
                    # Pull neighbors of old position slightly inward
                    dx_o = ox_i - x
                    dy_o = oy_i - y
                    do = np.sqrt(dx_o * dx_o + dy_o * dy_o) + 1e-9
                    near_o = (do < 2.0) & (np.arange(n_target) != sq_i)
                    x[near_o] += (dx_o[near_o] / do[near_o]) * 0.04
                    y[near_o] += (dy_o[near_o] / do[near_o]) * 0.04

            else:
                # Mode D: Coherent Band Micro-Rotation + Flush Shear Glide
                target_th = med_th_b + float(rng.uniform(-0.04, 0.04))
                if rng.random() < 0.35:
                    # Snap band toward nearest Pythagorean angle
                    diffs = np.abs(pyth_angs[1:] - med_th_b)
                    target_th = float(pyth_angs[1 + int(np.argmin(diffs))])
                dth = target_th - med_th_b
                th[band_mask] += dth + rng.normal(0.0, 0.015, size=int(band_mask.sum()))
                # Shear glide along band tangent proportional to normal coordinate
                u_n = geom["u_perp"]
                glide = float(rng.uniform(-0.025, 0.025)) * u_n
                x[band_mask] += glide[band_mask] * float(t_vec[0])
                y[band_mask] += glide[band_mask] * float(t_vec[1])

        # Occasional symmetry reflection for diversity
        if idx % 7 == 5:
            x = L_tgt - x
            th = -th
        elif idx % 7 == 6:
            x, y = y.copy(), x.copy()
            th = -th

        out_x[idx] = np.clip(x, 0.48, L_tgt - 0.48)
        out_y[idx] = np.clip(y, 0.48, L_tgt - 0.48)
        out_th[idx] = th

    return out_x, out_y, out_th


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", default="969,971,972,791,367,851,910,796,333,687")
    parser.add_argument("--bin-dirs", default="/tmp/v10a_top,/tmp/v10b_top,/tmp/v9b_top,/tmp/v9_top,/tmp/v8_top")
    parser.add_argument("--out-dir", default="/tmp/v10_surg")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--polish-rounds", type=int, default=2)
    args = parser.parse_args()

    n_list = [int(t) for t in args.n_list.split(",") if t.strip()]
    bin_dirs = [Path(p) for p in args.bin_dirs.split(",") if p.strip()]
    data_dir = Path("poc/swarm_inflation/data")
    out_dir = Path(args.out_dir)

    for n in n_list:
        inc_L = float(pc.load_incumbent_L(n))
        infos = load_source_basins_for_target(n, bin_dirs, data_dir, polish_rounds=args.polish_rounds)
        if not infos:
            print(f"[surg10] n={n}: no source basins found, skipping", flush=True)
            continue
        src_summary = ", ".join(f"n={d['n_src']}@{d['L']:.5f}" for d in infos[:4])
        sx, sy, sth = generate_idea10_pool(n, infos, inc_L, count=args.count, rng_seed=args.seed)
        out_p = out_dir / f"surg_n{n}.bin"
        i7.write_surg_bin(out_p, sx, sy, sth)
        print(
            f"[surg10] n={n} (inc={inc_L:.6f}): {len(infos)} basins [{src_summary}] -> {args.count} gentle seeds -> {out_p}",
            flush=True,
        )


if __name__ == "__main__":
    main()
