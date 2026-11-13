#!/usr/bin/env python3
"""Idea 17: Scaled Dual-Track GPU Melt (Idea 15 fm=5 + Idea 16 fm=7 + Expansion Deltas).

Per target n (disjoint pool + #147 targets), runs:
  - Track A: Idea 15 GPU soft-a melt (fast_mode=5, batch=1024) -> top_k basins
  - Track B: Idea 16 frozen-exterior masked-pocket melt (fast_mode=7, batch=1024)
             with mixed expansion (+0.024, +0.008) & compression deltas -> top_k basins
  - Track C: 0-jitter + 3 micro-jitter incumbent starts
Polishes all basins in parallel across 4 CPU workers (1 BLAS/OMP/HiGHS thread each),
checkpoints per-n JSON after every target, and updates win_n<n>.csv + final.csv
strictly when both literal and pandas roundtrip certifications improve on incumbent.
"""
from __future__ import annotations

import argparse
import json
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
from poc.swarm_inflation.poc import polish_and_certify as pc

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "poc" / "swarm_inflation" / "data"
CKPT_PATH = REPO_ROOT / "poc" / "swarm_inflation" / "docs" / "idea17_ckpt.json"
DEFAULT_TARGETS = [791, 969, 851, 796, 333, 970, 911, 972, 910, 973]
MIXED_DELTAS = (-0.024, -0.008, 0.00015, 0.0006, 0.0020, 0.0060)


def _polish_one_basin(args: tuple[int, str, np.ndarray, np.ndarray, str]) -> dict:
    """Polish one candidate basin with fast_joint_xytheta_slp + multi-margin LP + dual cert."""
    n, label, pos_in, th_in, inc_str = args
    inc_L = Decimal(inc_str)
    t0 = time.time()
    c_slp, t_slp, L_slp = pc.fast_joint_xytheta_slp(
        pos_in, th_in, rounds=16, tr_th0=0.020, tr_xy0=0.08, lp_margins=h15.LP_MARGINS
    )
    cands = [(c_slp, t_slp)]
    if Decimal(str(L_slp)) < inc_L + Decimal("0.003"):
        for m_lp in h15.LP_MARGINS:
            cm, _ = pc.fixed_angle_lp_polish(c_slp, t_slp, rounds=8, margin=m_lp)
            cands.append((cm, t_slp))

    best_L: Decimal | None = None
    best_cert: dict | None = None
    for cm, tm in cands:
        cert = h15._certify_deg_roundtrip(n, cm, tm)
        if cert is None:
            continue
        Lc = Decimal(cert["L_max"])
        if best_L is None or Lc < best_L:
            best_L = Lc
            best_cert = cert

    return {
        "label": label,
        "L_max": str(best_L) if best_L is not None else None,
        "cert": best_cert,
        "dt": round(time.time() - t0, 2),
    }


def _run_gpu_track_15(n: int, mats: dict, batch: int, seed: int) -> Path:
    surg_dir = Path("/tmp/v17_surg_15")
    export_dir = Path("/tmp/v17_top_15")
    surg_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, st, _ = m15.build_seeds_for_n(n, mats, count=batch, rng_seed=seed)
    m15.write_surg_bin(surg_dir / f"surg_n{n}.bin", sx, sy, st)
    env = dict(os.environ, OMP_NUM_THREADS="4")
    subprocess.run(
        [
            "/tmp/cpp_pipeline",
            "--idea", "15",
            "--n-list", str(n),
            "--batch", str(batch),
            "--fast-mode", "5",
            "--elastic-L", "1",
            "--seed", str(seed),
            "--surg-dir", str(surg_dir),
            "--export-top", str(export_dir),
        ],
        check=True,
        env=env,
    )
    return export_dir / f"top_n{n}_fm5.bin"


def _run_gpu_track_16(n: int, mats: dict, batch: int, seed: int) -> Path:
    surg_dir = Path("/tmp/v17_surg_16")
    export_dir = Path("/tmp/v17_top_16")
    surg_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, st, smob = i16.build_masked_batch_for_n(
        n, mats, batch=batch, rng_seed=seed, deltas=MIXED_DELTAS
    )
    m15.write_surg_bin(surg_dir / f"surg_n{n}.bin", sx, sy, st)
    i16.write_mask_bin(surg_dir / f"mask_n{n}.bin", smob)
    env = dict(os.environ, OMP_NUM_THREADS="4")
    subprocess.run(
        [
            "/tmp/cpp_pipeline",
            "--idea", "16",
            "--n-list", str(n),
            "--batch", str(batch),
            "--fast-mode", "7",
            "--elastic-L", "1",
            "--seed", str(seed),
            "--surg-dir", str(surg_dir),
            "--export-top", str(export_dir),
        ],
        check=True,
        env=env,
    )
    return export_dir / f"top_n{n}_fm7.bin"


def run_one_target(
    n: int, mats: dict, batch: int, top_k: int, seed: int
) -> dict:
    """Run GPU fm=5 + masked fm=7 + parallel 4-worker SLP polish for one n."""
    t0 = time.time()
    inc_L = pc.load_incumbent_L(n)
    x0, y0, th0, _ = mats[n]
    c0 = pc.shift_positive(np.column_stack([x0, y0]), th0)

    tasks: list[tuple[int, str, np.ndarray, np.ndarray, str]] = [
        (n, "inc_orig", c0, th0.copy(), str(inc_L))
    ]
    rng = np.random.default_rng(seed + n * 19)
    for j in range(3):
        dc = rng.normal(0.0, 1e-4, size=c0.shape)
        dt = np.radians(rng.normal(0.0, 0.02, size=len(th0)))
        tasks.append((n, f"jit_{j}", pc.shift_positive(c0 + dc, th0 + dt), th0 + dt, str(inc_L)))

    fp15 = _run_gpu_track_15(n, mats, batch=batch, seed=seed)
    if fp15.exists():
        pos15, th15 = m15.read_top_bin(fp15)
        for k in range(min(len(pos15), top_k)):
            tasks.append((n, f"v15_fm5#{k}", pos15[k], th15[k], str(inc_L)))

    fp16 = _run_gpu_track_16(n, mats, batch=batch, seed=seed + 100)
    if fp16.exists():
        pos16, th16 = m15.read_top_bin(fp16)
        for k in range(min(len(pos16), top_k)):
            tasks.append((n, f"v16_fm7#{k}", pos16[k], th16[k], str(inc_L)))

    best_L: Decimal | None = None
    best_label = ""
    best_cert: dict | None = None
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_polish_one_basin, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            if res["L_max"] is None:
                continue
            Lc = Decimal(res["L_max"])
            if best_L is None or Lc < best_L:
                best_L = Lc
                best_label = res["label"]
                best_cert = res["cert"]

    win_csv = DATA_DIR / f"win_n{n}.csv"
    floor_L = inc_L
    if win_csv.exists():
        try:
            subs = pc.core.read_submission(win_csv)
            if n in subs:
                floor_L = min(floor_L, Decimal(str(pc.core.certify(n, subs[n]))))
        except Exception:
            pass

    dL = float(best_L - inc_L) if best_L is not None else None
    strict_win = best_L is not None and best_L < floor_L
    real_win = strict_win and (dL is not None and dL <= -1e-9)
    # Guard against ULP-tiny ties that may flip under pandas 3.1 float formatting
    safe_tie = strict_win and not real_win and (dL is not None and dL <= -1e-13)

    out = {
        "n": n,
        "inc_L": str(inc_L),
        "best_L": str(best_L) if best_L is not None else None,
        "dL": dL,
        "real_win": bool(real_win),
        "safe_tie": bool(safe_tie),
        "best_label": best_label,
        "n_basins": len(tasks),
        "dt": round(time.time() - t0, 2),
    }
    if (real_win or safe_tie) and best_cert is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pc.core.write_submission(DATA_DIR / f"win_n{n}.csv", {n: best_cert["rows"]}, validate=True)
        h15.write_final_csv_from_wins([{"n": n, "win": True, "deg_rows": best_cert["deg_rows"]}])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=str, default=",".join(str(x) for x in DEFAULT_TARGETS))
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1700)
    ap.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    args = ap.parse_args()

    n_list = [int(s) for s in args.n_list.split(",") if s.strip()]
    mats = m15.load_all_incumbents()

    ckpt: dict[str, dict] = {}
    if args.ckpt.exists():
        ckpt = json.loads(args.ckpt.read_text())

    for n in n_list:
        if str(n) in ckpt:
            print(f"[v17_skip] n={n} already checkpointed: dL={ckpt[str(n)].get('dL')}", flush=True)
            continue
        r = run_one_target(n, mats, batch=args.batch, top_k=args.top_k, seed=args.seed)
        ckpt[str(n)] = r
        args.ckpt.parent.mkdir(parents=True, exist_ok=True)
        args.ckpt.write_text(json.dumps(ckpt, indent=2))
        tag = "REAL_WIN" if r["real_win"] else ("SAFE_TIE" if r["safe_tie"] else "no_win")
        print(
            f"[v17_done] n={n} {tag} dL={r['dL']} src={r['best_label']} "
            f"basins={r['n_basins']} ({r['dt']}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
