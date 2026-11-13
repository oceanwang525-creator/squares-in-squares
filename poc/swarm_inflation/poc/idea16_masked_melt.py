#!/usr/bin/env python3
"""Idea 16: Frozen-Exterior Masked-Pocket GPU Soft-a Melt (0.80 -> 1.00).

Six structural-reorganization masked families per target n (batch=1024):
  0 (mvoid16) : R=4.5..7.5 void-collapse melt around max-clearance voids
  1 (mpin16)  : 1-2 boundary-pin teleports into voids, melting BOTH src & dst cavities
  2 (mwall16) : weak-wall contact-balancing strip melt (3.5..6.0 deep)
  3 (mseam16) : tilted/upright grain-boundary seam melt (R=4.8..7.5)
  4 (mpeel16) : masked surplus-peel (n+1..n+5k -> n & n+1->n tip deletions, cut R<=4.5 mobile)
  5 (mband16) : diagonal corridor or R=7.5..9.5 quadrant reorganization
Freezes exterior squares (mob=0) through a=0.80..0.9985, unfreezes all at a=1.000,
and finishes top GPU+AL basins with fast_joint_xytheta_slp + multi-margin LP + dual_certify.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.spatial import cKDTree

from poc.swarm_inflation.poc import idea15_direct_slp_harvest as h15
from poc.swarm_inflation.poc import idea15_inc_melt as m15
from poc.swarm_inflation.poc import polish_and_certify as pc

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "poc" / "swarm_inflation" / "data"
ALL_TEN = [687, 796, 367, 971, 910, 972, 791, 969, 851, 333]
PILOT_THREE = [687, 796, 367]


def _seam_anchors(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Return indices of tilted squares bordering upright squares (or all indices fallback)."""
    deg = np.degrees(th) % 90.0
    tilted = (deg > 2.5) & (deg < 87.5)
    upright = ~tilted
    if np.any(tilted) and np.any(upright):
        tree_u = cKDTree(np.column_stack([x[upright], y[upright]]))
        t_idx = np.where(tilted)[0]
        dists, _ = tree_u.query(np.column_stack([x[t_idx], y[t_idx]]), k=1)
        near_t = t_idx[dists < 2.2]
        if len(near_t) > 0:
            return near_t
        return t_idx
    return np.arange(len(x))


def build_masked_batch_for_n(
    n: int,
    mats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    batch: int = 1024,
    rng_seed: int = 1600,
    deltas: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct `batch` perturbed layouts + binary mobile masks (1=mobile, 0=frozen)."""
    rng = np.random.default_rng(rng_seed + n * 53)
    x_inc, y_inc, th_inc, L_inc = mats[n]
    hx = 0.5 * (np.abs(np.cos(th_inc)) + np.abs(np.sin(th_inc)))
    margin = np.minimum.reduce([x_inc - hx, L_inc - x_inc - hx, y_inc - hx, L_inc - y_inc - hx])
    tight_idx = np.argsort(margin)[:22]
    voids = m15.clearance_points(x_inc, y_inc, th_inc, L_inc, n_pts=10, step=0.14)
    tilt = m15.donor_tilt(th_inc)
    seams = _seam_anchors(x_inc, y_inc, th_inc)

    mL = x_inc - hx - float((x_inc - hx).min())
    mR = float((x_inc + hx).max()) - (x_inc + hx)
    mB = y_inc - hx - float((y_inc - hx).min())
    mT = float((y_inc + hx).max()) - (y_inc + hx)
    counts = [
        ("L", int(np.sum(mL < 5e-4))),
        ("R", int(np.sum(mR < 5e-4))),
        ("B", int(np.sum(mB < 5e-4))),
        ("T", int(np.sum(mT < 5e-4))),
    ]
    counts.sort(key=lambda kv: kv[1])
    weak_walls = [counts[0][0], counts[1][0]]

    border_peels, cross_peels = m15.collect_surplus_peels(n, L_inc, mats)
    all_peels = border_peels + cross_peels

    if deltas is None:
        deltas = (0.00015, 0.0004, 0.0010, 0.0022, 0.0045, 0.0080)
    out_x = np.zeros((batch, n), dtype=np.float64)
    out_y = np.zeros((batch, n), dtype=np.float64)
    out_th = np.zeros((batch, n), dtype=np.float64)
    out_mob = np.zeros((batch, n), dtype=np.uint8)

    for b in range(batch):
        fam = b % 6
        grp = b // 6
        delta = deltas[grp % len(deltas)]
        L0 = L_inc - delta
        sc = L0 / L_inc
        x = x_inc * sc
        y = y_inc * sc
        th = th_inc.copy()
        mob = np.zeros(n, dtype=bool)

        if fam == 0 and voids:
            vx, vy, _ = voids[grp % len(voids)]
            rad = float(rng.uniform(4.5, 7.5))
            mob = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
            nm = int(mob.sum())
            pull = float(rng.uniform(0.03, 0.09))
            x[mob] += pull * (vx - x[mob]) + rng.uniform(-0.14, 0.14, nm)
            y[mob] += pull * (vy - y[mob]) + rng.uniform(-0.14, 0.14, nm)
            th[mob] += rng.uniform(-0.28, 0.28, nm)
            if grp % 3 == 0 and nm > 4:
                flip = rng.random(nm) < 0.22
                idx_m = np.where(mob)[0][flip]
                th[idx_m] = rng.choice([0.0, 0.25 * math.pi, tilt], size=len(idx_m))
        elif fam == 1 and voids:
            k_mv = 1 if (grp % 2 == 0) else min(2, len(voids))
            picks = rng.choice(tight_idx[:16], size=k_mv, replace=False)
            for j, pi in enumerate(picks):
                sx0, sy0 = float(x[pi]), float(y[pi])
                vx, vy, _ = voids[(grp + j) % len(voids)]
                mob |= (((x - sx0) ** 2 + (y - sy0) ** 2) <= 4.5 ** 2)
                mob |= (((x - vx) ** 2 + (y - vy) ** 2) <= 4.8 ** 2)
                x[pi] = vx + float(rng.uniform(-0.08, 0.08))
                y[pi] = vy + float(rng.uniform(-0.08, 0.08))
                th[pi] = [0.0, 0.25 * math.pi, tilt][(grp + j) % 3] + float(rng.uniform(-0.10, 0.10))
            nm = int(mob.sum())
            x[mob] += rng.uniform(-0.14, 0.14, nm)
            y[mob] += rng.uniform(-0.14, 0.14, nm)
            th[mob] += rng.uniform(-0.22, 0.22, nm)
        elif fam == 2:
            w = weak_walls[grp % len(weak_walls)]
            depth = float(rng.uniform(3.6, 6.2))
            if w == "L":
                mob = x <= depth
                drift = (float(rng.uniform(-0.08, -0.02)), 0.0)
            elif w == "R":
                mob = x >= L0 - depth
                drift = (float(rng.uniform(0.02, 0.08)), 0.0)
            elif w == "B":
                mob = y <= depth
                drift = (0.0, float(rng.uniform(-0.08, -0.02)))
            else:
                mob = y >= L0 - depth
                drift = (0.0, float(rng.uniform(0.02, 0.08)))
            nm = int(mob.sum())
            x[mob] += drift[0] + rng.uniform(-0.12, 0.12, nm)
            y[mob] += drift[1] + rng.uniform(-0.12, 0.12, nm)
            th[mob] += rng.uniform(-0.20, 0.20, nm)
        elif fam == 3:
            ai = int(seams[grp % len(seams)])
            ax0, ay0 = float(x[ai]), float(y[ai])
            rad = float(rng.uniform(4.8, 7.5))
            mob = ((x - ax0) ** 2 + (y - ay0) ** 2) <= rad * rad
            nm = int(mob.sum())
            x[mob] += rng.uniform(-0.15, 0.15, nm)
            y[mob] += rng.uniform(-0.15, 0.15, nm)
            th[mob] += rng.uniform(-0.30, 0.30, nm)
        elif fam == 4 and all_peels:
            don = all_peels[grp % min(len(all_peels), 24)]
            x0, y0, th0, oh0 = don["x"], don["y"], don["th"], don["oh"]
            surp = don["surplus"]
            L_box = don["L"]
            L0 = min(L_box, L_inc - deltas[grp % 4])
            scp = L0 / L_box
            rank = np.argsort(-oh0)
            pool_idx = rank[: max(surp + 5, min(len(rank), 14))]
            keep = np.ones(len(x0), dtype=bool)
            if surp > 0:
                drop = rank[:surp] if (grp % 4 == 0) else rng.choice(pool_idx, size=surp, replace=False)
                keep[drop] = False
                dx_drop = x0[drop] * scp
                dy_drop = y0[drop] * scp
            else:
                dx_drop = np.empty(0)
                dy_drop = np.empty(0)
            x = x0[keep] * scp
            y = y0[keep] * scp
            th = th0[keep].copy()
            oh = oh0[keep]
            mob = oh > 0.005
            for xd, yd in zip(dx_drop, dy_drop):
                mob |= (((x - xd) ** 2 + (y - yd) ** 2) <= 4.8 ** 2)
            if int(mob.sum()) < 24:
                hot_i = int(np.argmax(oh))
                mob |= (((x - x[hot_i]) ** 2 + (y - y[hot_i]) ** 2) <= 5.2 ** 2)
            nm = int(mob.sum())
            x[mob] += rng.uniform(-0.13, 0.13, nm)
            y[mob] += rng.uniform(-0.13, 0.13, nm)
            th[mob] += rng.uniform(-0.18, 0.18, nm)
        else:
            if grp % 2 == 0:
                sgn = 1.0 if ((grp // 2) % 2 == 0) else -1.0
                off = float(rng.uniform(-0.22 * L0, 0.22 * L0))
                hw = float(rng.uniform(3.2, 5.2))
                d = np.abs((x - 0.5 * L0) + sgn * (y - 0.5 * L0) - off) / math.sqrt(2.0)
                mob = d <= hw
            else:
                margin_q = min(5.0, 0.25 * L0)
                qx = float(rng.uniform(margin_q, max(margin_q + 1e-6, L0 - margin_q)))
                qy = float(rng.uniform(margin_q, max(margin_q + 1e-6, L0 - margin_q)))
                rad_q = float(rng.uniform(min(7.0, 0.45 * L0), min(9.2, 0.65 * L0)))
                mob = ((x - qx) ** 2 + (y - qy) ** 2) <= rad_q ** 2
            nm = int(mob.sum())
            x[mob] += rng.uniform(-0.15, 0.15, nm)
            y[mob] += rng.uniform(-0.15, 0.15, nm)
            th[mob] += rng.uniform(-0.32, 0.32, nm)

        x, y, th = m15._clamp_box(x, y, th, L0, pad=0.002)
        out_x[b] = x
        out_y[b] = y
        out_th[b] = th
        out_mob[b] = mob.astype(np.uint8)

    return out_x, out_y, out_th, out_mob


def write_mask_bin(path: Path, mob: np.ndarray) -> None:
    M, n = mob.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", int(M), int(n)))
        f.write(np.ascontiguousarray(mob, dtype=np.uint8).tobytes())


def _polish_gpu_exports(n: int, export_dir: Path, fast_mode: int) -> dict:
    """Finish top exported masked-melt GPU basins with fast_joint_xytheta_slp + dual_certify."""
    t0 = time.time()
    inc_L = pc.load_incumbent_L(n)
    fp = export_dir / f"top_n{n}_fm{fast_mode}.bin"
    if not fp.exists():
        return {"n": n, "win": False, "real_win": False, "dL": None, "dt": 0.0}
    pos, th = m15.read_top_bin(fp)

    best_L: Decimal | None = None
    best_cert: dict | None = None
    best_k = -1
    for k in range(min(len(pos), 16)):
        c_slp, t_slp, L_slp = pc.fast_joint_xytheta_slp(
            pos[k], th[k], rounds=16, tr_th0=0.020, tr_xy0=0.08, lp_margins=h15.LP_MARGINS
        )
        cands = [(c_slp, t_slp)]
        if Decimal(str(L_slp)) < inc_L + Decimal("0.005"):
            for m_lp in h15.LP_MARGINS:
                cm, _ = pc.fixed_angle_lp_polish(c_slp, t_slp, rounds=8, margin=m_lp)
                cands.append((cm, t_slp))
        for cm, tm in cands:
            cert = h15._certify_deg_roundtrip(n, cm, tm)
            if cert is None:
                continue
            Lc = Decimal(cert["L_max"])
            if best_L is None or Lc < best_L:
                best_L = Lc
                best_cert = cert
                best_k = k

    dL = float(best_L - inc_L) if best_L is not None else None
    strict_win = best_L is not None and best_L < inc_L
    real_win = strict_win and (dL is not None and dL <= -1e-9)
    out = {
        "n": n,
        "inc_L": str(inc_L),
        "best_L": str(best_L) if best_L is not None else None,
        "dL": dL,
        "win": bool(strict_win),
        "real_win": bool(real_win),
        "best_k": best_k,
        "dt": round(time.time() - t0, 2),
    }
    if strict_win and best_cert is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pc.core.write_submission(DATA_DIR / f"win_n{n}.csv", {n: best_cert["rows"]}, validate=True)
        out["deg_rows"] = best_cert["deg_rows"]
    return out


def run_idea16(n_list: list[int], batch: int, seed: int, fast_mode: int = 7) -> None:
    mats = m15.load_all_incumbents()
    surg_dir = Path("/tmp/v16_surg")
    surg_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path("/tmp/v16_top")
    export_dir.mkdir(parents=True, exist_ok=True)

    for n in n_list:
        sx, sy, st, smob = build_masked_batch_for_n(n, mats, batch=batch, rng_seed=seed)
        m15.write_surg_bin(surg_dir / f"surg_n{n}.bin", sx, sy, st)
        write_mask_bin(surg_dir / f"mask_n{n}.bin", smob)
        avg_mob = float(smob.mean()) * 100.0
        print(f"[v16_seed] n={n} batch={batch} avg_mobile={avg_mob:.1f}%", flush=True)

    n_arg = ",".join(str(x) for x in n_list)
    env = dict(os.environ, OMP_NUM_THREADS="4")
    subprocess.run(
        [
            "/tmp/cpp_pipeline",
            "--idea", "16",
            "--n-list", n_arg,
            "--batch", str(batch),
            "--fast-mode", str(fast_mode),
            "--elastic-L", "1",
            "--seed", str(seed),
            "--surg-dir", str(surg_dir),
            "--export-top", str(export_dir),
        ],
        check=True,
        env=env,
    )

    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_polish_gpu_exports, n, export_dir, fast_mode): n for n in n_list}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "REAL_WIN" if r["real_win"] else ("TIE" if r["win"] else "no_win")
            print(f"[v16_slp] n={r['n']} {tag} dL={r['dL']} best_k={r['best_k']} ({r['dt']}s)", flush=True)

    wins = [r for r in results if r["win"]]
    if wins:
        h15.write_final_csv_from_wins(wins)
    clean = [{k: v for k, v in r.items() if k != "deg_rows"} for r in sorted(results, key=lambda x: x["n"])]
    out_doc = REPO_ROOT / "poc" / "swarm_inflation" / "docs" / "idea16_results.json"
    out_doc.write_text(json.dumps(clean, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=str, default="687,796,367")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1600)
    ap.add_argument("--fast-mode", type=int, default=7)
    args = ap.parse_args()
    n_list = [int(s) for s in args.n_list.split(",") if s.strip()]
    run_idea16(n_list, args.batch, args.seed, args.fast_mode)


if __name__ == "__main__":
    main()
