"""Idea 11: Active LP-Dual Force-Chain Buckling, Slip-Line Shear & Corner-Matched Half-Crystal Glide.

Key physical fix over Idea 10:
In SW-to-NE diagonal-band layouts (`sband6`, `fband8`, `asym8`, `asym9`), the two `0°`
half-crystals sit in the NW corner (`x=0, y=L0`) and SE corner (`x=L0, y=0`).
Shrinking the box from `[0, L0]^2` to `[0, L_tgt]^2` (`delta_L = L0 - L_tgt`) without
crushing outer walls requires shifting EACH half-crystal according to the walls it
actually touches:
  - NW half-crystal (`mean_x < 0.5*L0, mean_y > 0.5*L0`): shift `(0, -delta_L)`
  - SE half-crystal (`mean_x > 0.5*L0, mean_y < 0.5*L0`): shift `(-delta_L, 0)`
  - SW half-crystal (`mean_x < 0.5*L0, mean_y < 0.5*L0`): shift `(0, 0)`
  - NE half-crystal (`mean_x > 0.5*L0, mean_y > 0.5*L0`): shift `(-delta_L, -delta_L)`
This preserves 0.0 internal overlap and 0.0 wall violation across both `0°` crystals,
focusing 100% of `delta_L` into normal closure across the diagonal tilted band, where
active LP-dual force-chain buckling (`op=0`), slip-line shear (`op=1`), bottleneck-to-hole
vacancy hopping (`op=2`), dual-weighted rattler drift (`op=3`), and cross-`n` transfer (`op=4`)
unlock lower-L packings under `sat_lmin_cuda --fast-mode 4`.
Never reads incumbent coordinates from merged.csv.
"""
from __future__ import annotations

import argparse
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
import idea10_gentle_transfer as i10  # noqa: E402
import polish_and_certify as pc  # noqa: E402


def _corner_matched_crystal_shift(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L0: float,
    L_tgt: float,
    geom: dict,
    ds_hi: float = 0.0,
    ds_lo: float = 0.0,
    ds_band: float = 0.0,
    dn_band: float = 0.0,
    dth_band: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift g_hi, g_lo, and band into [0, L_tgt]^2 matching each half-crystal's anchored walls."""
    delta_L = L0 - L_tgt
    x_out = x.copy()
    y_out = y.copy()
    th_out = th.copy()

    g_hi = geom["g_hi"]
    g_lo = geom["g_lo"]
    band_mask = geom["band_mask"]
    t_vec = geom["t_vec"]
    n_vec = geom["n_vec"]

    for mask, ds_g in ((g_hi, ds_hi), (g_lo, ds_lo)):
        if not np.any(mask):
            continue
        mx = float(np.mean(x[mask]))
        my = float(np.mean(y[mask]))
        dx_wall = -delta_L if mx > 0.5 * L0 else 0.0
        dy_wall = -delta_L if my > 0.5 * L0 else 0.0
        x_out[mask] += dx_wall + ds_g * t_vec[0]
        y_out[mask] += dy_wall + ds_g * t_vec[1]

    if np.any(band_mask):
        # Band squares scale smoothly along box coordinates + tangent/normal glide
        bx = x[band_mask]
        by = y[band_mask]
        x_out[band_mask] = bx - delta_L * np.clip(bx / max(L0, 1e-6), 0.0, 1.0) + ds_band * t_vec[0] + dn_band * n_vec[0]
        y_out[band_mask] = by - delta_L * np.clip(by / max(L0, 1e-6), 0.0, 1.0) + ds_band * t_vec[1] + dn_band * n_vec[1]
        th_out[band_mask] += dth_band

    return x_out, y_out, th_out


def load_idea11_basins(
    n_target: int,
    bin_dirs: list[Path],
    data_dir: Path,
    polish_rounds: int = 1,
) -> list[dict]:
    """Load tightest cand_v11/v10/v9/v8 and top binary basins (same-n + cross-n neighbors)."""
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
        csv_cands = [
            data_dir / f"cand_v11_n{n_src}.csv",
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
            inf["holes"] = i10.find_diverse_holes(np.stack([inf["x"], inf["y"]], axis=1), float(inf["L"]), top_k=12)
            infos.append(inf)

        if n_src == n_target:
            raw_b = i7.load_unique_scratch_basins(n_src, bin_dirs, data_dir, max_keep=6)
            for p, t, L_val in raw_b[:4]:
                if any(abs(L_val - sl) < 1e-4 for sl in seen_L):
                    continue
                seen_L.append(L_val)
                inf = i7.extract_lp_duals_and_holes(p, t, polish_rounds=max(1, polish_rounds - 1))
                inf["n_src"] = n_src
                inf["holes"] = i10.find_diverse_holes(np.stack([inf["x"], inf["y"]], axis=1), float(inf["L"]), top_k=12)
                infos.append(inf)

    infos.sort(key=lambda d: float(d["L"]) + (0.0 if d["n_src"] == 969 and n_target in (971, 972) else 0.004 * abs(n_target - d["n_src"])))
    return infos[:10]


def generate_idea11_pool(
    n_target: int,
    basins_info: list[dict],
    inc_L: float,
    count: int = 1024,
    rng_seed: int = 1100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 5-operator force-chain & corner-matched glide surgical pool."""
    rng = np.random.default_rng(rng_seed + n_target * 23)
    out_x = np.zeros((count, n_target), dtype=np.float64)
    out_y = np.zeros((count, n_target), dtype=np.float64)
    out_th = np.zeros((count, n_target), dtype=np.float64)

    pyth_angs = np.array([
        0.0,
        math.atan2(1.0, 2.0), -math.atan2(1.0, 2.0),
        math.atan2(5.0, 12.0), -math.atan2(5.0, 12.0),
        math.atan2(8.0, 15.0), -math.atan2(8.0, 15.0),
        math.atan2(3.0, 4.0), -math.atan2(3.0, 4.0),
    ], dtype=np.float64)

    nb = len(basins_info)
    weights = np.array([1.0 / (k + 0.8) for k in range(nb)], dtype=np.float64)
    weights /= weights.sum()

    for idx in range(count):
        b_idx = idx % min(3, nb) if idx < (2 * count) // 3 else int(rng.choice(nb, p=weights))
        b = basins_info[b_idx]
        x = b["x"].copy()
        y = b["y"].copy()
        th = b["th"].copy()
        L0 = float(b["L"])
        n_src = int(b.get("n_src", len(x)))
        sq_load = b["sq_load"].copy()
        holes = b["holes"]
        delta_n = n_target - n_src

        # Step 1: Resolve cross-n stoichiometry delta_n at L0 before corner-matched shift
        geom0 = i8._analyze_band_geometry(x, y, th, L0)
        t_vec = geom0["t_vec"]
        n_vec = geom0["n_vec"]
        bmask0 = geom0["band_mask"]
        band_th = float(np.median(th[bmask0])) if np.any(bmask0) else -0.4636476
        seam_dist = geom0["seam_dist"]

        if delta_n > 0:
            add_x, add_y, add_th = [], [], []
            n_h = len(holes)
            for k in range(delta_n):
                h_idx = (idx // 5 + k * 2 + int(rng.integers(0, max(1, min(4, n_h))))) % n_h
                hx, hy, _ = holes[h_idx]
                px = float(np.clip(hx + rng.normal(0.0, 0.06), 0.55, L0 - 0.55))
                py = float(np.clip(hy + rng.normal(0.0, 0.06), 0.55, L0 - 0.55))
                d2 = (x - px) ** 2 + (y - py) ** 2
                near = np.argsort(d2)[:6]
                ang_mode = (idx + k) % 3
                if ang_mode == 0:
                    pth = float(th[near[0]]) + float(rng.normal(0.0, 0.015))
                elif ang_mode == 1:
                    pth = band_th + float(rng.normal(0.0, 0.02))
                else:
                    pth = 0.0
                for j in near:
                    dx = x[j] - px
                    dy = y[j] - py
                    dist = math.hypot(dx, dy)
                    if 1e-6 < dist < 1.05:
                        push = 0.20 * (1.05 - dist)
                        x[j] += (dx / dist) * push
                        y[j] += (dy / dist) * push
                add_x.append(px)
                add_y.append(py)
                add_th.append(pth)
            x = np.concatenate([x, np.array(add_x, dtype=np.float64)])
            y = np.concatenate([y, np.array(add_y, dtype=np.float64)])
            th = np.concatenate([th, np.array(add_th, dtype=np.float64)])
            sq_load = np.concatenate([sq_load, np.full(delta_n, float(np.median(sq_load)))])
        elif delta_n < 0:
            k_rem = -delta_n
            seam_bonus = np.exp(-0.5 * (seam_dist / 1.5) ** 2)
            score = (sq_load + 1e-4) * (1.0 + 2.0 * seam_bonus) * rng.uniform(0.6, 1.4, size=len(x))
            evict = np.argsort(score)[-k_rem:]
            ev_pts = np.stack([x[evict], y[evict]], axis=1)
            keep = np.ones(len(x), dtype=bool)
            keep[evict] = False
            x = x[keep]
            y = y[keep]
            th = th[keep]
            sq_load = sq_load[keep]
            for ep in ev_pts:
                d2 = (x - ep[0]) ** 2 + (y - ep[1]) ** 2
                near = np.argsort(d2)[:8]
                for j in near:
                    x[j] += 0.15 * (ep[0] - x[j])
                    y[j] += 0.15 * (ep[1] - y[j])

        # Choose target box L_tgt:
        # 50% sub-incumbent strike (inc_L - 0.0003..0.0040), 50% monotonic step-down below L0
        gap = max(0.0, L0 - inc_L)
        if L0 < inc_L - 0.001:
            L_tgt = float(rng.uniform(L0 + 0.0002, inc_L - 0.0004))
        elif idx % 2 == 0 or gap <= 0.004:
            L_tgt = inc_L - float(rng.uniform(0.0003, 0.0042))
        else:
            step_down = float(rng.uniform(0.0012, min(0.0120, max(0.0025, 0.75 * gap))))
            L_tgt = max(inc_L - 0.0015, L0 - step_down)

        geom = i8._analyze_band_geometry(x, y, th, L0)
        t_vec = geom["t_vec"]
        n_vec = geom["n_vec"]
        band_mask = geom["band_mask"]
        seam_dist = geom["seam_dist"]

        op = idx % 5
        ds_hi = float(rng.uniform(-0.025, 0.025))
        ds_lo = float(rng.uniform(-0.025, 0.025))
        ds_band = float(rng.uniform(-0.045, 0.045))
        dn_band = float(rng.uniform(-0.020, 0.020))
        dth_band = float(rng.uniform(-0.012, 0.012))

        # Apply corner-matched half-crystal shift into [0, L_tgt]^2
        x, y, th = _corner_matched_crystal_shift(
            x, y, th, L0, L_tgt, geom,
            ds_hi=ds_hi, ds_lo=ds_lo, ds_band=ds_band, dn_band=dn_band, dth_band=dth_band,
        )

        # Active LP-dual force chain indices
        active_idx = np.where(sq_load > 0.015)[0]
        if len(active_idx) < 6:
            active_idx = np.argsort(sq_load)[-16:]

        if op == 0:
            # Op 0: Active LP-Dual Force-Chain Transverse Buckling
            # Select 1..4 high-dual squares in/near the seam and kick transversely + snap angle
            seam_active = [i for i in active_idx if seam_dist[i] < 2.2]
            if not seam_active:
                seam_active = list(active_idx)
            k_buck = int(rng.integers(1, min(5, len(seam_active) + 1)))
            probs = sq_load[seam_active] + 1e-3
            probs /= probs.sum()
            chosen = rng.choice(seam_active, size=k_buck, replace=False, p=probs)
            hx, hy, _ = holes[idx % len(holes)]
            hx_s = hx * (L_tgt / L0)
            hy_s = hy * (L_tgt / L0)
            for ci in chosen:
                vx = hx_s - x[ci]
                vy = hy_s - y[ci]
                vn = math.hypot(vx, vy)
                if vn > 1e-6:
                    step_b = float(rng.uniform(0.025, 0.110))
                    # Transverse + void-directed kick
                    x[ci] += step_b * (0.65 * (vx / vn) + 0.35 * t_vec[0] * rng.choice([-1.0, 1.0]))
                    y[ci] += step_b * (0.65 * (vy / vn) + 0.35 * t_vec[1] * rng.choice([-1.0, 1.0]))
                if rng.random() < 0.65:
                    th[ci] = 0.0 if abs(th[ci]) > 0.15 else (band_th + float(rng.normal(0.0, 0.02)))
                else:
                    th[ci] += float(rng.normal(0.0, 0.05))

        elif op == 1:
            # Op 1: Slip-Line Shear through Peak-Dual Bottleneck Square
            top_chain = np.argsort(sq_load)[-8:]
            piv = int(rng.choice(top_chain))
            px, py = float(x[piv]), float(y[piv])
            # Slip direction: either band tangent t_vec or ±45 deg
            if (idx // 5) % 2 == 0:
                sx_dir, sy_dir = float(t_vec[0]), float(t_vec[1])
                nx_dir, ny_dir = float(n_vec[0]), float(n_vec[1])
            else:
                ang_s = math.pi * 0.25 if (idx // 10) % 2 == 0 else -math.pi * 0.25
                sx_dir, sy_dir = math.cos(ang_s), math.sin(ang_s)
                nx_dir, ny_dir = -sy_dir, sx_dir
            signed_n = (x - px) * nx_dir + (y - py) * ny_dir
            band_strip = np.abs(signed_n) < float(rng.uniform(1.2, 2.8))
            slip_mag = float(rng.uniform(0.012, 0.055)) * (1.0 if (idx % 2 == 0) else -1.0)
            x[band_strip] += np.sign(signed_n[band_strip]) * slip_mag * sx_dir
            y[band_strip] += np.sign(signed_n[band_strip]) * slip_mag * sy_dir
            th[band_strip] += rng.normal(0.0, 0.015, size=int(band_strip.sum()))

        elif op == 2:
            # Op 2: Bottleneck-to-Seam-Void Vacancy Hop + Local Chain Relaxation
            seam_Order = np.argsort(sq_load * np.exp(-0.3 * seam_dist))
            piv = int(seam_Order[-1 - (idx // 5) % min(6, len(seam_Order))])
            h_idx = (idx // 5) % min(6, len(holes))
            hx, hy, hr = holes[h_idx]
            hx_s = hx * (L_tgt / L0)
            hy_s = hy * (L_tgt / L0)
            old_x, old_y = float(x[piv]), float(y[piv])
            if hr > 0.76 and (idx // 5) % 3 == 0:
                # Hop bottleneck square into the largest seam void and pull neighbors into old spot
                x[piv] = float(np.clip(hx_s + rng.normal(0.0, 0.04), 0.55, L_tgt - 0.55))
                y[piv] = float(np.clip(hy_s + rng.normal(0.0, 0.04), 0.55, L_tgt - 0.55))
                th[piv] = band_th if rng.random() < 0.5 else 0.0
                d2_old = (x - old_x) ** 2 + (y - old_y) ** 2
                d2_old[piv] = 1e9
                near_old = np.argsort(d2_old)[:6]
                for j in near_old:
                    x[j] += 0.18 * (old_x - x[j])
                    y[j] += 0.18 * (old_y - y[j])
            else:
                # Continuous chain drift from piv toward hole
                d2_line = (x - 0.5 * (old_x + hx_s)) ** 2 + (y - 0.5 * (old_y + hy_s)) ** 2
                chain_nb = np.argsort(d2_line)[:int(rng.integers(5, 14))]
                dx_h = hx_s - old_x
                dy_h = hy_s - old_y
                dn_h = math.hypot(dx_h, dy_h)
                if dn_h > 1e-6:
                    step_c = float(rng.uniform(0.02, 0.08))
                    for j in chain_nb:
                        x[j] += step_c * (dx_h / dn_h) + float(rng.normal(0.0, 0.008))
                        y[j] += step_c * (dy_h / dn_h) + float(rng.normal(0.0, 0.008))

        elif op == 3:
            # Op 3: Dual-Weighted Rattler Drift + Seam-Boundary 0-to-Tilted Conversion
            rattlers = np.where((sq_load <= 1e-4) & (seam_dist < 2.5))[0]
            if len(rattlers) > 0:
                hx, hy, _ = holes[(idx // 5) % len(holes)]
                hx_s, hy_s = hx * (L_tgt / L0), hy * (L_tgt / L0)
                for r_i in rattlers:
                    vx, vy = hx_s - x[r_i], hy_s - y[r_i]
                    vn = math.hypot(vx, vy)
                    if vn > 1e-6:
                        x[r_i] += 0.025 * (vx / vn)
                        y[r_i] += 0.025 * (vy / vn)
            # Convert 1..3 seam-adjacent squares between 0 deg and band_th
            seam_cands = np.where((seam_dist > 0.45) & (seam_dist < 1.45))[0]
            if len(seam_cands) > 0:
                k_conv = int(rng.integers(1, min(4, len(seam_cands) + 1)))
                pick = rng.choice(seam_cands, size=k_conv, replace=False)
                for ci in pick:
                    if abs(th[ci]) < 0.12:
                        th[ci] = band_th + float(rng.normal(0.0, 0.02))
                    else:
                        th[ci] = 0.0

        else:
            # Op 4: Pure Ultra-Clean Corner-Matched Half-Crystal Glide + Micro Seam Polish
            near_seam = seam_dist < 1.35
            if np.any(near_seam):
                n_s = int(near_seam.sum())
                s_jit = float(rng.uniform(0.004, 0.028))
                x[near_seam] += rng.normal(0.0, s_jit, size=n_s) * t_vec[0]
                y[near_seam] += rng.normal(0.0, s_jit, size=n_s) * t_vec[1]
                th[near_seam] += rng.normal(0.0, 0.008, size=n_s)

        # Wrap angles into [-pi/4, pi/4) and clamp strictly inside [ext, L_tgt - ext]
        th = ((th + math.pi * 0.25) % (math.pi * 0.5)) - math.pi * 0.25
        ext = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        x = np.clip(x, ext + 1e-6, L_tgt - ext - 1e-6)
        y = np.clip(y, ext + 1e-6, L_tgt - ext - 1e-6)

        out_x[idx] = x
        out_y[idx] = y
        out_th[idx] = th

    return out_x, out_y, out_th


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=str, default="791,969,367,972,971,851,910,796,333,687")
    ap.add_argument("--bin-dirs", type=str, default="/tmp/v11_top,/tmp/v10a_top,/tmp/v10b_top,/tmp/v9_top,/tmp/v8_top")
    ap.add_argument("--data-dir", type=str, default="poc/swarm_inflation/data")
    ap.add_argument("--out-dir", type=str, default="/tmp/v11_surg")
    ap.add_argument("--count", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1100)
    ap.add_argument("--polish-rounds", type=int, default=1)
    args = ap.parse_args()

    n_list = [int(s) for s in args.n_list.split(",") if s.strip()]
    bin_dirs = [Path(s) for s in args.bin_dirs.split(",") if s.strip()]
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in n_list:
        inc_L = float(pc.load_incumbent_L(n))
        infos = load_idea11_basins(n, bin_dirs, data_dir, polish_rounds=args.polish_rounds)
        if not infos:
            print(f"[surg11] n={n}: no basins found, skipping", flush=True)
            continue
        sx, sy, st = generate_idea11_pool(n, infos, inc_L=inc_L, count=args.count, rng_seed=args.seed)
        out_path = out_dir / f"surg_n{n}.bin"
        i7.write_surg_bin(out_path, sx, sy, st)
        src_summary = ", ".join(f"n={d['n_src']}@{d['L']:.5f}" for d in infos[:4])
        print(
            f"[surg11] n={n} (inc={inc_L:.6f}): {len(infos)} basins [{src_summary}] "
            f"-> {args.count} force-chain seeds -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
