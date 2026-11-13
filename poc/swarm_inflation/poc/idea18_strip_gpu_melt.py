#!/usr/bin/env python3
"""Idea 18: GPU End-Cap Strip-Gap Release + Masked-Junction Soft-a Melt (fm=7 & fm=5).

Detects trapped 0.15..1.20 corner and mid-strip gaps along upright wall strips (L, R, B, T),
coherently slides contiguous blocks to release slack at the adjacent tilted junction or seam,
and runs batch=1024 GPU soft-a relaxation (both masked fast_mode=7 and unmasked fast_mode=5)
followed by 4-worker parallel fast_joint_xytheta_slp + multi-margin LP + dual_certify.
Writes to final.csv strictly on REAL WIN (dL <= -1e-9 vs current merged.csv, omitting ties).
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

from poc.swarm_inflation.poc import idea15_direct_slp_harvest as h15
from poc.swarm_inflation.poc import idea15_inc_melt as m15
from poc.swarm_inflation.poc import idea16_masked_melt as i16
from poc.swarm_inflation.poc import idea17_scale_gpu_melt as i17
from poc.swarm_inflation.poc import polish_and_certify as pc

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "poc" / "swarm_inflation" / "data"
CKPT_PATH = REPO_ROOT / "poc" / "swarm_inflation" / "docs" / "idea18_scale_ckpt.json"
GATE_THREE = [971, 791, 911]
SCALE_TARGETS = [
    686, 751, 985, 859, 639, 690, 696, 498, 592, 739,
    802, 909, 975, 972, 687, 854, 918, 980, 746, 587,
]
ARMSLACK_SKIP_NS = {911, 800, 796}
STRIP_DELTAS = (-0.016, -0.004, 0.0002, 0.0010, 0.0025, 0.0050)


def detect_wall_strip_gaps(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float
) -> list[dict]:
    """Identify coherent upright wall strips with corner or mid-strip gaps in [0.15, 1.25]."""
    deg = np.degrees(th) % 90.0
    up = (deg < 1.5) | (deg > 88.5)
    specs = [
        ("R", up & (x > L - 0.72), y, True),
        ("L", up & (x < 0.72), y, True),
        ("B", up & (y < 0.72), x, False),
        ("T", up & (y > L - 0.72), x, False),
    ]
    traps: list[dict] = []
    for wall, mask, coord, along_y in specs:
        idx = np.where(mask)[0]
        if len(idx) < 3:
            continue
        order = np.argsort(coord[idx])
        idx = idx[order]
        vals = coord[idx]

        # Contiguous chains separated by <= 1.15
        chains: list[np.ndarray] = []
        cur = [0]
        for k in range(1, len(vals)):
            if vals[k] - vals[k - 1] <= 1.15:
                cur.append(k)
            else:
                chains.append(idx[cur])
                cur = [k]
        chains.append(idx[cur])

        # Corner trap at low end (coord[0] - 0.5 in [0.15, 1.25])
        c0 = chains[0]
        g_lo = float(coord[c0[0]] - 0.5)
        if 0.15 <= g_lo <= 1.25 and len(c0) >= 3:
            traps.append(
                {
                    "wall": wall,
                    "kind": "lo_corner",
                    "along_y": along_y,
                    "chain": c0,
                    "gap": g_lo,
                    "sign": -1.0,  # slide toward 0.5
                    "junc": (
                        float(x[c0[-1]]),
                        float(y[c0[-1]]),
                    ),
                }
            )

        # Corner trap at high end ((L - 0.5) - coord[-1] in [0.15, 1.25])
        c1 = chains[-1]
        g_hi = float((L - 0.5) - coord[c1[-1]])
        if 0.15 <= g_hi <= 1.25 and len(c1) >= 3:
            traps.append(
                {
                    "wall": wall,
                    "kind": "hi_corner",
                    "along_y": along_y,
                    "chain": c1,
                    "gap": g_hi,
                    "sign": +1.0,  # slide toward L - 0.5
                    "junc": (
                        float(x[c1[0]]),
                        float(y[c1[0]]),
                    ),
                }
            )

        # Mid-strip gaps between consecutive chains
        for ci in range(len(chains) - 1):
            gap = float(coord[chains[ci + 1][0]] - coord[chains[ci][-1]] - 1.0)
            if 0.15 <= gap <= 1.25:
                # Slide the shorter adjacent chain across the gap toward the other
                if len(chains[ci]) <= len(chains[ci + 1]):
                    ch, sgn = chains[ci], +1.0
                    jx, jy = float(x[ch[-1]]), float(y[ch[-1]])
                else:
                    ch, sgn = chains[ci + 1], -1.0
                    jx, jy = float(x[ch[0]]), float(y[ch[0]])
                if len(ch) >= 2:
                    traps.append(
                        {
                            "wall": wall,
                            "kind": "mid_gap",
                            "along_y": along_y,
                            "chain": ch,
                            "gap": gap,
                            "sign": sgn,
                            "junc": (jx, jy),
                        }
                    )

    traps.sort(key=lambda t: (0 if "corner" in t["kind"] else 1, -len(t["chain"]), -t["gap"]))
    return traps


def _apply_strip_slide(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L0: float,
    trap: dict,
    rng: np.random.Generator,
    mode: int,
) -> np.ndarray:
    """Slide a trapped chain coherently and return boolean mask of active pocket."""
    sc = L0 / (L0 + 0.0)  # coords already scaled to L0
    del sc
    chain = trap["chain"]
    m = len(chain)
    gap = trap["gap"] * (L0 / max(1.0, L0))
    sgn = trap["sign"]
    along_y = trap["along_y"]
    wall = trap["wall"]

    # Choose how many squares slide cleanly vs how many enter junction gap
    k_tip = 1 if (mode % 3 == 0) else (2 if (m >= 4 and mode % 3 == 1) else 0)
    if sgn < 0:
        slide_idx = chain[: m - k_tip]
        tip_idx = chain[m - k_tip :] if k_tip > 0 else np.empty(0, dtype=int)
    else:
        slide_idx = chain[k_tip:]
        tip_idx = chain[:k_tip] if k_tip > 0 else np.empty(0, dtype=int)

    if along_y:
        y[slide_idx] += sgn * gap
    else:
        x[slide_idx] += sgn * gap

    # Push tip square(s) slightly inward and along released gap
    inward = float(rng.uniform(0.10, 0.36))
    along_frac = float(rng.uniform(0.65, 1.15))
    for ti in tip_idx:
        if wall == "R":
            x[ti] -= inward
        elif wall == "L":
            x[ti] += inward
        elif wall == "T":
            y[ti] -= inward
        else:
            y[ti] += inward
        if along_y:
            y[ti] += sgn * along_frac * gap
        else:
            x[ti] += sgn * along_frac * gap
        if mode % 4 == 0:
            th[ti] += float(rng.uniform(-0.22, 0.22))

    # Build mobile pocket covering the full slid chain + generous radius around junction
    jx, jy = trap["junc"]
    rad = float(rng.uniform(5.2, 9.2))
    mob = ((x - jx) ** 2 + (y - jy) ** 2) <= rad * rad
    mob[chain] = True
    # Expand neighborhood along the slid chain by 2.2 units so adjacent tilted squares can shift
    for si in chain[::2]:
        mob |= ((x - x[si]) ** 2 + (y - y[si]) ** 2) <= 2.4 ** 2

    nm = int(mob.sum())
    # Perturb non-slide mobile squares to unlock the junction, smaller noise on slid strip
    non_slide = mob.copy()
    non_slide[slide_idx] = False
    nns = int(non_slide.sum())
    x[non_slide] += rng.uniform(-0.14, 0.14, nns)
    y[non_slide] += rng.uniform(-0.14, 0.14, nns)
    th[non_slide] += rng.uniform(-0.22, 0.22, nns)
    x[slide_idx] += rng.uniform(-0.04, 0.04, len(slide_idx))
    y[slide_idx] += rng.uniform(-0.04, 0.04, len(slide_idx))
    th[slide_idx] += rng.uniform(-0.04, 0.04, len(slide_idx))
    del nm
    return mob


def build_strip_batch_for_n(
    n: int,
    mats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    batch: int = 1024,
    rng_seed: int = 1800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Generate `batch` strip-release + masked-junction seeds and mobile masks."""
    rng = np.random.default_rng(rng_seed + n * 71)
    x_inc, y_inc, th_inc, L_inc = mats[n]
    traps = detect_wall_strip_gaps(x_inc, y_inc, th_inc, L_inc)

    # Also build general Idea 16 fallback batch for 25% diversity
    fx, fy, fth, fmob = i16.build_masked_batch_for_n(
        n, mats, batch=max(256, batch // 4), rng_seed=rng_seed + 50, deltas=STRIP_DELTAS
    )

    out_x = np.zeros((batch, n), dtype=np.float64)
    out_y = np.zeros((batch, n), dtype=np.float64)
    out_th = np.zeros((batch, n), dtype=np.float64)
    out_mob = np.zeros((batch, n), dtype=np.uint8)

    n_strip = batch - len(fx)
    for b in range(batch):
        if b >= n_strip or not traps:
            fb = b % len(fx)
            out_x[b], out_y[b], out_th[b], out_mob[b] = fx[fb], fy[fb], fth[fb], fmob[fb]
            continue

        delta = STRIP_DELTAS[b % len(STRIP_DELTAS)]
        L0 = L_inc - delta
        sc = L0 / L_inc
        x = x_inc * sc
        y = y_inc * sc
        th = th_inc.copy()

        # Primary trap gets 70% weight; secondary/combo traps get 30%
        t_idx = 0 if (b % 10 < 7 or len(traps) == 1) else (b % len(traps))
        trap = dict(traps[t_idx])
        trap["junc"] = (trap["junc"][0] * sc, trap["junc"][1] * sc)
        trap["gap"] = trap["gap"] * sc
        mob = _apply_strip_slide(x, y, th, L0, trap, rng, mode=b)

        # Occasional combo: apply primary + second distinct wall trap simultaneously
        if len(traps) >= 2 and b % 5 == 0:
            for alt in traps[1:]:
                if alt["wall"] != trap["wall"]:
                    alt_s = dict(alt)
                    alt_s["junc"] = (alt["junc"][0] * sc, alt["junc"][1] * sc)
                    alt_s["gap"] = alt["gap"] * sc
                    mob |= _apply_strip_slide(x, y, th, L0, alt_s, rng, mode=b + 1)
                    break

        x, y, th = m15._clamp_box(x, y, th, L0, pad=0.002)
        out_x[b], out_y[b], out_th[b], out_mob[b] = x, y, th, mob.astype(np.uint8)

    return out_x, out_y, out_th, out_mob, len(traps)


def _run_gpu(
    n: int,
    sx: np.ndarray,
    sy: np.ndarray,
    st: np.ndarray,
    smob: np.ndarray,
    fast_mode: int,
    seed: int,
) -> Path:
    tag = f"fm{fast_mode}"
    surg_dir = Path(f"/tmp/v18_surg_{tag}")
    export_dir = Path(f"/tmp/v18_top_{tag}")
    surg_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    m15.write_surg_bin(surg_dir / f"surg_n{n}.bin", sx, sy, st)
    idea_flag = "16" if fast_mode == 7 else "15"
    if fast_mode == 7:
        i16.write_mask_bin(surg_dir / f"mask_n{n}.bin", smob)
    env = dict(os.environ, OMP_NUM_THREADS="4")
    subprocess.run(
        [
            "/tmp/cpp_pipeline",
            "--idea", idea_flag,
            "--n-list", str(n),
            "--batch", str(len(sx)),
            "--fast-mode", str(fast_mode),
            "--elastic-L", "1",
            "--seed", str(seed),
            "--surg-dir", str(surg_dir),
            "--export-top", str(export_dir),
        ],
        check=True,
        env=env,
    )
    return export_dir / f"top_n{n}_fm{fast_mode}.bin"


def run_one_n(
    n: int,
    mats: dict,
    batch: int,
    top_k_fm5: int,
    top_k_fm7: int,
    seed: int,
) -> dict:
    t0 = time.time()
    inc_L = pc.load_incumbent_L(n)
    sx, sy, st, smob, n_traps = build_strip_batch_for_n(n, mats, batch=batch, rng_seed=seed)
    print(f"[v18_seed] n={n} traps={n_traps} avg_mob={float(smob.mean())*100:.1f}%", flush=True)

    tasks: list[tuple[int, str, np.ndarray, np.ndarray, str]] = []
    for fm, tk in ((5, top_k_fm5), (7, top_k_fm7)):
        if tk <= 0:
            continue
        fp = _run_gpu(n, sx, sy, st, smob, fast_mode=fm, seed=seed + fm * 10)
        if fp.exists():
            pos, th = m15.read_top_bin(fp)
            for k in range(min(len(pos), tk)):
                tasks.append((n, f"v18_fm{fm}#{k}", pos[k], th[k], str(inc_L)))

    best_L: Decimal | None = None
    best_label = ""
    best_cert: dict | None = None
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(i17._polish_one_basin, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            if res["L_max"] is None:
                continue
            Lc = Decimal(res["L_max"])
            if best_L is None or Lc < best_L:
                best_L = Lc
                best_label = res["label"]
                best_cert = res["cert"]

    dL = float(best_L - inc_L) if best_L is not None else None
    real_win = best_L is not None and dL is not None and dL <= -1e-9
    out = {
        "n": n,
        "inc_L": str(inc_L),
        "best_L": str(best_L) if best_L is not None else None,
        "dL": dL,
        "real_win": bool(real_win),
        "best_label": best_label,
        "n_traps": n_traps,
        "dt": round(time.time() - t0, 2),
    }
    if real_win and best_cert is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pc.core.write_submission(DATA_DIR / f"win_n{n}.csv", {n: best_cert["rows"]}, validate=True)
        if n not in ARMSLACK_SKIP_NS:
            h15.write_final_csv_from_wins([{"n": n, "win": True, "deg_rows": best_cert["deg_rows"]}])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=str, default=",".join(str(x) for x in GATE_THREE))
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--top-k-fm5", type=int, default=8)
    ap.add_argument("--top-k-fm7", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1800)
    ap.add_argument("--stop-consec-nonreal", type=int, default=12)
    ap.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    args = ap.parse_args()

    n_list = [int(s) for s in args.n_list.split(",") if s.strip()]
    mats = m15.load_all_incumbents()
    ckpt: dict[str, dict] = {}
    if args.ckpt.exists():
        ckpt = json.loads(args.ckpt.read_text())

    consec_nonreal = 0
    for n in n_list:
        if str(n) in ckpt:
            r = ckpt[str(n)]
            consec_nonreal = 0 if r.get("real_win") else consec_nonreal + 1
            print(f"[v18_skip] n={n} already checkpointed dL={r.get('dL')}", flush=True)
            continue
        r = run_one_n(
            n,
            mats,
            batch=args.batch,
            top_k_fm5=args.top_k_fm5,
            top_k_fm7=args.top_k_fm7,
            seed=args.seed,
        )
        ckpt[str(n)] = r
        args.ckpt.parent.mkdir(parents=True, exist_ok=True)
        args.ckpt.write_text(json.dumps(ckpt, indent=2))
        tag = "REAL_WIN" if r["real_win"] else "no_win"
        print(f"[v18_done] n={n} {tag} dL={r['dL']} src={r['best_label']} ({r['dt']}s)", flush=True)
        if r["real_win"]:
            consec_nonreal = 0
        else:
            consec_nonreal += 1
            if args.stop_consec_nonreal > 0 and consec_nonreal >= args.stop_consec_nonreal:
                print(f"[v18_stop] {consec_nonreal} consecutive non-real targets reached.", flush=True)
                break


if __name__ == "__main__":
    main()
