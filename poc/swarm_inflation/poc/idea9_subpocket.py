"""Idea 9A: Fixed-Crystal GPU Sub-Pocket Inflation (`sat_sub_cuda`) + LP/SLP Polish.

Locks flawless 0-deg deep-interior crystal squares (rigidity-verified per
connected component and wall affinity at L_target) and inflates only the active
LP-dual bridge / tilted band / tip-wedge / void neighborhood (`N_mob <= 250`)
directly at `L_target < inc_L` (`a: 0.72 -> 1.00`, multi-reheat + void-jump).
Supports both same-n compression and from-scratch stoichiometry lifts (e.g. 969->971/972).
"""
from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import idea7_chain_ruin as i7  # noqa: E402
import polish_and_certify as pc  # noqa: E402

SAT_SUB_BIN = Path("/tmp/sat_sub_cuda")
DATA_DIR = Path("poc/swarm_inflation/data")


def _connected_upright_components(pos: np.ndarray, up_idx: np.ndarray) -> list[np.ndarray]:
    m = len(up_idx)
    if m == 0:
        return []
    pts = pos[up_idx]
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    adj = d2 < 1.12
    vis = np.zeros(m, dtype=bool)
    comps: list[np.ndarray] = []
    for s in range(m):
        if vis[s]:
            continue
        stack = [s]
        vis[s] = True
        cur = []
        while stack:
            u = stack.pop()
            cur.append(int(up_idx[u]))
            for v in np.flatnonzero(adj[u]):
                if not vis[v]:
                    vis[v] = True
                    stack.append(int(v))
        comps.append(np.array(cur, dtype=np.int64))
    return comps


def select_mobile_and_shift_frozen(
    pos: np.ndarray,
    th: np.ndarray,
    L_base: float,
    L_target: float,
    info: dict,
    r_cut: float = 3.2,
    max_mob: int = 240,
    focus_hole: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Partition squares into [0..N_mob-1] mobile and [N_mob..N-1] strictly non-overlapping frozen."""
    n = len(pos)
    deg = (np.degrees(th) % 90.0)
    deg = np.where(deg > 45.0, deg - 90.0, deg)
    sq_load = info["sq_load"]
    holes = info["holes"]

    # Seed mobile set: tilted + high-dual + hole vicinities, or a single dense connected ball
    tilted = np.abs(deg) >= 2.2
    high_dual = sq_load > 0.05
    if focus_hole >= 0 and holes:
        hx, hy, _ = holes[focus_hole % len(holes)]
        d2_anc = (pos[:, 0] - hx) ** 2 + (pos[:, 1] - hy) ** 2
        order = np.argsort(d2_anc)
        mob_mask = np.zeros(n, dtype=bool)
        mob_mask[order[: min(max_mob, max(96, int((d2_anc <= (r_cut + 2.5) ** 2).sum())))]] = True
    else:
        seed_mask = tilted | high_dual
        seed_pts = [pos[seed_mask]]
        if holes:
            hpts = np.array([[hx, hy] for hx, hy, _ in holes[:4]], dtype=np.float64)
            seed_pts.append(hpts)
        anchors = np.vstack([p for p in seed_pts if len(p) > 0])
        d2_anc = ((pos[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        mob_mask = seed_mask | (d2_anc <= r_cut * r_cut)
        if mob_mask.sum() > max_mob:
            order = np.argsort(np.where(tilted, -10.0, np.where(high_dual, -5.0, d2_anc)))
            mob_mask[:] = False
            mob_mask[order[:max_mob]] = True
        elif mob_mask.sum() < min(96, n // 2):
            order = np.argsort(d2_anc)
            mob_mask[order[: min(max_mob, max(96, n // 3))]] = True

    # Shift frozen upright components safely to fit [0, L_target]^2 without intra-overlap
    fix_idx = np.flatnonzero(~mob_mask)
    pos_out = pos.copy()
    th_out = th.copy()
    # Snap tiny residual tilt in frozen upright squares? Keep original exact th_out to preserve zero overlap
    dL = float(L_target - L_base)
    for comp in _connected_upright_components(pos_out, fix_idx):
        cx = pos_out[comp, 0]
        cy = pos_out[comp, 1]
        c_th = th_out[comp]
        ext = 0.5 * (np.abs(np.cos(c_th)) + np.abs(np.sin(c_th)))
        d_x0 = float((cx - ext).min())
        d_x1 = float(L_base - (cx + ext).max())
        d_y0 = float((cy - ext).min())
        d_y1 = float(L_base - (cy + ext).max())
        sx = dL if (d_x1 < d_x0 and abs(dL) < 0.25) else 0.0
        sy = dL if (d_y1 < d_y0 and abs(dL) < 0.25) else 0.0
        pos_out[comp, 0] += sx
        pos_out[comp, 1] += sy

    # Verify zero wall or frozen-frozen overlap; promote offenders to mobile (up to max_mob)
    fix_set = set(fix_idx.tolist())
    changed = True
    while changed:
        changed = False
        f_list = sorted(fix_set)
        if not f_list:
            break
        f_arr = np.array(f_list, dtype=np.int64)
        fp = pos_out[f_arr]
        ft = th_out[f_arr]
        cf, sf = np.cos(ft), np.sin(ft)
        ext = 0.5 * (np.abs(cf) + np.abs(sf))
        bad_wall = (
            (fp[:, 0] - ext < 1e-7)
            | (fp[:, 0] + ext > L_target - 1e-7)
            | (fp[:, 1] - ext < 1e-7)
            | (fp[:, 1] + ext > L_target - 1e-7)
        )
        if bad_wall.any():
            for idx_b in f_arr[bad_wall]:
                if len(fix_set) <= n - max_mob:
                    break
                fix_set.remove(int(idx_b))
                mob_mask[int(idx_b)] = True
                changed = True
            continue

        # Pairwise frozen-frozen SAT clearance check
        dx = fp[None, :, 0] - fp[:, None, 0]
        dy = fp[None, :, 1] - fp[:, None, 1]
        near = (dx * dx + dy * dy) < 2.2
        np.fill_diagonal(near, False)
        ii, jj = np.where(np.triu(near, k=1))
        if len(ii) > 0:
            c1, s1 = cf[ii], sf[ii]
            c2, s2 = cf[jj], sf[jj]
            ddx, ddy = dx[ii, jj], dy[ii, jj]
            rad = 0.5 * (1.0 + np.abs(c1 * c2 + s1 * s2) + np.abs(s1 * c2 - c1 * s2))
            pmax = np.maximum.reduce([
                np.abs(ddx * c1 + ddy * s1),
                np.abs(-ddx * s1 + ddy * c1),
                np.abs(ddx * c2 + ddy * s2),
                np.abs(-ddx * s2 + ddy * c2),
            ])
            ov = rad - pmax
            bad_p = np.flatnonzero(ov > -1e-7)
            if len(bad_p) > 0:
                for k in bad_p[:8]:
                    idx_rem = int(f_arr[ii[k]])
                    if len(fix_set) > n - max_mob:
                        fix_set.discard(idx_rem)
                        mob_mask[idx_rem] = True
                        changed = True

    mob_idx = np.flatnonzero(mob_mask)
    fix_idx = np.flatnonzero(~mob_mask)
    order = np.concatenate([mob_idx, fix_idx])
    return pos_out[order], th_out[order], order, int(len(mob_idx))


def build_subpocket_batch(
    n_target: int,
    base_pos: np.ndarray,
    base_th: np.ndarray,
    L_target: float,
    info: dict,
    batch: int = 512,
    r_cut: float = 3.2,
    rng_seed: int = 909,
    max_mob: int = 240,
    focus_hole: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Create `batch` diverse sub-pocket re-inflation trajectories at fixed `L_target`."""
    n_base = len(base_pos)
    delta_n = n_target - n_base
    if delta_n < 0:
        raise ValueError("n_target must be >= n_base for sub-pocket builder")

    L_base = float(info["L"])
    pos_ord, th_ord, _perm, n_mob_base = select_mobile_and_shift_frozen(
        base_pos, base_th, L_base=L_base, L_target=L_target, info=info,
        r_cut=r_cut, max_mob=max(64, max_mob - delta_n), focus_hole=focus_hole,
    )

    holes = info["holes"] or [(L_target * 0.5, L_target * 0.5, 0.5)]
    n_mob = n_mob_base + delta_n
    # Place mobile first [0..n_mob_base-1], then inserted [delta_n], then frozen [n_mob..n_target-1]
    mob_p = pos_ord[:n_mob_base]
    mob_t = th_ord[:n_mob_base]
    fix_p = pos_ord[n_mob_base:]
    fix_t = th_ord[n_mob_base:]

    rng = np.random.default_rng(rng_seed + n_target * 31)
    bx = np.empty((batch, n_target), dtype=np.float32)
    by = np.empty((batch, n_target), dtype=np.float32)
    bth = np.empty((batch, n_target), dtype=np.float32)
    bL = np.full(batch, L_target, dtype=np.float32)

    deg_m = (np.degrees(mob_t) % 90.0)
    deg_m = np.where(deg_m > 45.0, deg_m - 90.0, deg_m)
    tilt_ang = float(np.median(mob_t[np.abs(deg_m) >= 3.0])) if np.any(np.abs(deg_m) >= 3.0) else -0.4636476
    cand_angs = np.array([
        0.0, tilt_ang, -tilt_ang, -0.46364761, 0.46364761,
        -0.64350111, 0.64350111, -0.39479112, 0.78539816, -0.490037
    ], dtype=np.float64)

    for b in range(batch):
        mp = mob_p.copy()
        mt = mob_t.copy()
        mode = b % 5

        if delta_n > 0:
            ins_p = np.empty((delta_n, 2), dtype=np.float64)
            ins_t = np.empty(delta_n, dtype=np.float64)
            for k in range(delta_n):
                hx, hy, _ = holes[(b + k) % len(holes)]
                ins_p[k, 0] = np.clip(hx + rng.normal(0.0, 0.25), 0.7, L_target - 0.7)
                ins_p[k, 1] = np.clip(hy + rng.normal(0.0, 0.25), 0.7, L_target - 0.7)
                ins_t[k] = cand_angs[int(rng.integers(0, len(cand_angs)))]
            mp_all = np.vstack([mp, ins_p])
            mt_all = np.concatenate([mt, ins_t])
        else:
            mp_all = mp
            mt_all = mt

        # Gentle perturbations that preserve corridor stoichiometry while exploring pockets
        cm = mp_all.mean(axis=0)
        if b == 0 and delta_n == 0:
            pass
        elif mode == 0:
            mp_all = cm + (mp_all - cm) * rng.uniform(0.996, 1.000) + rng.normal(0.0, 0.012, size=mp_all.shape)
            mt_all += rng.normal(0.0, 0.025, size=n_mob)
        elif mode == 1:
            mp_all += rng.normal(0.0, 0.032, size=mp_all.shape)
            re_ang = rng.random(n_mob) < 0.18
            mt_all[re_ang] = rng.choice(cand_angs, size=int(re_ang.sum()))
        elif mode == 2:
            dt = float(rng.choice([-0.08, -0.04, 0.04, 0.08, 0.12, -0.12]))
            re_t = np.abs((np.degrees(mt_all) % 90.0)) > 2.0
            mt_all[re_t] += dt
            mp_all += rng.normal(0.0, 0.025, size=mp_all.shape)
        elif mode == 3:
            mp_all += rng.normal(0.0, 0.060, size=mp_all.shape)
            wall_n = (mp_all[:, 0] < 1.8) | (mp_all[:, 0] > L_target - 1.8) | (mp_all[:, 1] < 1.8) | (mp_all[:, 1] > L_target - 1.8)
            mt_all[wall_n & (rng.random(n_mob) < 0.5)] = 0.0
            core_n = ~wall_n
            pick_t = float(rng.choice(cand_angs[1:]))
            mt_all[core_n & (rng.random(n_mob) < 0.30)] = pick_t
        else:
            mp_all += rng.normal(0.0, 0.025, size=mp_all.shape)
            k_sw = int(rng.integers(1, min(4, n_mob)))
            idxs = rng.choice(n_mob, size=k_sw, replace=False)
            for idx_s in idxs:
                hx, hy, _ = holes[int(rng.integers(0, len(holes)))]
                mp_all[idx_s, 0] = hx + rng.normal(0.0, 0.15)
                mp_all[idx_s, 1] = hy + rng.normal(0.0, 0.15)
                mt_all[idx_s] = cand_angs[int(rng.integers(0, len(cand_angs)))]

        mp_all = np.clip(mp_all, 0.55, L_target - 0.55)
        full_p = np.vstack([mp_all, fix_p])
        full_t = np.concatenate([mt_all, fix_t])
        bx[b] = full_p[:, 0].astype(np.float32)
        by[b] = full_p[:, 1].astype(np.float32)
        bth[b] = full_t.astype(np.float32)

    return bx, by, bth, n_mob


def run_sat_sub_cuda(
    bx: np.ndarray,
    by: np.ndarray,
    bth: np.ndarray,
    L_val: float,
    n_mob: int,
    seed: int = 909,
    mode: int = 0,
    sat_bin: Path = SAT_SUB_BIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not sat_bin.exists():
        cu_src = Path(__file__).resolve().parent / "sat_sub.cu"
        subprocess.run(["nvcc", "-O3", "-arch=sm_89", str(cu_src), "-o", str(sat_bin)], check=True)
    B, N = bx.shape
    bL = np.full(B, L_val, dtype=np.float32)
    in_bin = f"/tmp/sub_in_n{N}_{n_mob}.bin"
    out_bin = f"/tmp/sub_out_n{N}_{n_mob}.bin"
    with open(in_bin, "wb") as f:
        f.write(struct.pack("<iiiiI", B, N, n_mob, mode, seed & 0xFFFFFFFF))
        f.write(np.ascontiguousarray(bx, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(by, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(bth, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(bL, dtype=np.float32).tobytes())
    subprocess.run([str(sat_bin), "--in", in_bin, "--out", out_bin], check=True)
    raw = Path(out_bin).read_bytes()
    off = 0
    sz_bn = B * N * 8
    ox = np.frombuffer(raw, dtype=np.float64, count=B * N, offset=off).reshape(B, N).copy(); off += sz_bn
    oy = np.frombuffer(raw, dtype=np.float64, count=B * N, offset=off).reshape(B, N).copy(); off += sz_bn
    oth = np.frombuffer(raw, dtype=np.float64, count=B * N, offset=off).reshape(B, N).copy(); off += sz_bn
    open_arr = np.frombuffer(raw, dtype=np.float64, count=B, offset=off).copy(); off += B * 8
    olam_arr = np.frombuffer(raw, dtype=np.float64, count=B, offset=off).copy()
    return ox, oy, oth, open_arr, olam_arr


def solve_subpocket_target(
    n_target: int,
    src_csv: Path,
    inc_L: float,
    batch: int = 512,
    r_cuts: tuple[float, ...] = (2.8, 3.8),
    deltas: tuple[float, ...] = (0.001, 0.004, 0.010),
    lp_rounds: int = 5,
    slp_rounds: int = 6,
    slp_top_k: int = 3,
) -> dict:
    df = pd.read_csv(src_csv)
    base_pos = df[["x", "y"]].to_numpy(dtype=np.float64)
    base_th = df["angle"].to_numpy(dtype=np.float64)
    info = i7.extract_lp_duals_and_holes(base_pos, base_th)

    cand_pos = []
    cand_th = []
    cand_meta = []
    cand_score = []
    delta_n = n_target - len(base_pos)
    configs = [(-1, 2.8, 0), (0, 3.6, 0), (1, 3.6, 0), (0, 4.0, 2)]
    if delta_n > 0:
        configs = [(0, 4.0, 2), (1, 4.0, 2), (-1, 3.2, 2)]

    for ci, (fh, rc, kmode) in enumerate(configs):
        for di, dlt in enumerate(deltas):
            L_tgt = inc_L - dlt
            bx, by, bth, n_mob = build_subpocket_batch(
                n_target, base_pos, base_th, L_target=L_tgt, info=info,
                batch=batch, r_cut=rc, rng_seed=910 + ci * 97 + di * 17,
                focus_hole=fh,
            )
            ox, oy, oth, open_a, olam_a = run_sat_sub_cuda(
                bx, by, bth, L_val=L_tgt, n_mob=n_mob, seed=910 + ci * 97 + di * 17, mode=kmode,
            )
            eff_L = L_tgt * olam_a
            order = np.argsort(eff_L)
            print(
                f"[sub9] n={n_target} (src={len(base_pos)}) fh={fh} rc={rc:.1f} km={kmode} mob={n_mob} "
                f"L_tgt={L_tgt:.5f} -> best_lam={olam_a[order[0]]:.6f} eff_L={eff_L[order[0]]:.6f} pen={open_a[order[0]]:.2e}",
                flush=True,
            )
            for k in order[:4]:
                if olam_a[k] < 1.045 and eff_L[k] < inc_L + 0.08:
                    cand_pos.append(np.stack([ox[k], oy[k]], axis=1))
                    cand_th.append(oth[k])
                    cand_meta.append(f"sub9(src={len(base_pos)},fh={fh},km={kmode},Lt={L_tgt:.4f},lam={olam_a[k]:.5f})")
                    cand_score.append(float(eff_L[k]))

    if not cand_pos:
        return {"n": n_target, "win": False, "best_L_pd": None, "inc_L": str(inc_L)}

    sel = np.argsort(cand_score)[:8]
    pos_b = np.stack([cand_pos[i] for i in sel], axis=0)
    th_b = np.stack([cand_th[i] for i in sel], axis=0)
    meta_b = [cand_meta[i] for i in sel]
    win_csv = DATA_DIR / f"win_n{n_target}.csv"
    best_csv = DATA_DIR / f"cand_sub9_n{n_target}.csv"
    res = pc.polish_candidates(
        n_target, pos_b, th_b, meta=meta_b, out_win_csv=win_csv,
        lp_rounds=lp_rounds, slp_rounds=slp_rounds, slp_top_k=slp_top_k,
        out_best_csv=best_csv,
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="969:cand_v8_n969.csv,971:cand_v8_n969.csv,972:cand_v8_n969.csv,367:cand_v8_n367.csv,851:cand_v8_n851.csv,972:cand_v8_n972.csv")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    for spec in args.tasks.split(","):
        if not spec.strip():
            continue
        n_str, csv_name = spec.split(":", 1)
        n_tgt = int(n_str)
        src_p = DATA_DIR / csv_name
        if not src_p.exists():
            continue
        inc_L = float(pc.load_incumbent_L(n_tgt))
        res = solve_subpocket_target(n_tgt, src_p, inc_L=inc_L, batch=args.batch)
        print(
            f"=== [sub9 DONE] n={n_tgt} src={csv_name}: inc={res['inc_L']} best_pd={res.get('best_L_pd')} win={res.get('win')} ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
