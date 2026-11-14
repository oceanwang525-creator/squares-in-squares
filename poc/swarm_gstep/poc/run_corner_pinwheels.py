#!/usr/bin/env python3
"""Seam-melt runner for 1-corner-overlap (w1+w2==m, h1+h2==m) and zero-overlap pinwheels."""
from __future__ import annotations

import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gstep_lib as gl  # noqa: E402
import run_gstep_gate as gate  # noqa: E402

REPO_ROOT = gl.REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "poc" / "swarm_lpslack" / "poc"))
import lpslack  # noqa: E402


def build_pinwheel_allow_corner(
    m: int, b1: dict, b2: dict, sf: float = 0.0, f_override: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    if w1 + w2 > m or h1 + h2 > m:
        return None
    if w1 + w2 + h1 + h2 > 2 * m:
        return None
    f = max(b1["f"], b2["f"]) if f_override is None else f_override
    L = m + f
    dx1 = sf * (f - b1["fx"])
    dy1 = (m - h1) + (1.0 - sf) * (f - b1["fy"])
    dx2 = (m - w2) + (1.0 - sf) * (f - b2["fx"])
    dy2 = sf * (f - b2["fy"])

    gx, gy = [], []
    for i in range(m):
        for j in range(m):
            if i < w1 and j >= m - h1:
                continue
            if i >= m - w2 and j < h2:
                continue
            if i < m - w2 and j < m - h1:
                gx.append(i + 0.5)
                gy.append(j + 0.5)
            else:
                gx.append(i + f + 0.5)
                gy.append(j + f + 0.5)

    ax = np.concatenate([b1["x"] + dx1, b2["x"] + dx2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([b1["y"] + dy1, b2["y"] + dy2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([b1["th"], b2["th"], np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != m * m + b1["g"] + b2["g"]:
        return None
    seam = (
        ((ax >= w1 - 2.2) & (ax <= w1 + f + 2.2))
        | ((ay >= m - h1 - 2.2) & (ay <= m - h1 + 2.2))
        | ((ax >= m - w2 - 2.2) & (ax <= m - w2 + 2.2))
        | ((ay >= h2 - 2.2) & (ay <= h2 + f + 2.2))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def solve_corner_pair(
    n: int, m: int, b1_base: dict, b2_base: dict, inc_dec: Decimal, tag: str, base_seed: int
) -> dict | None:
    seeds, masks, scheds, labels = [], [], [], []
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = gl.bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            if (m1, m2) not in (("id", "id"), ("flipx", "flipy"), ("id", "flipx"), ("rot180", "rot180")):
                continue
            b2 = gl.bl.transform_brick(b2_base, m2)
            f_use = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            built = build_pinwheel_allow_corner(m, b1, b2, sf=0.0, f_override=f_use)
            if built is None:
                continue
            ax, ay, at, _, sm = built
            seeds.append((ax, ay, at))
            masks.append(sm)
            L_tgt = m + f_use
            scheds.append([L_tgt + 0.015, L_tgt + 0.004, L_tgt + 5e-4])
            labels.append(f"{tag}_{m1}_{m2}")

    if not seeds:
        return None

    best_cert: dict | None = None
    relaxed = gl.bl.su.cpp_batch_melt(
        seeds, scheds, sym_mode=1, a_start=0.96, n_melt_stages=3,
        seam_masks=masks, seam_iters=140,
        full_iters=220, n_ruin_iters=0, n_threads=2, base_seed=base_seed,
    )
    pool = []
    for bi, row in enumerate(relaxed):
        nx, ny, nth, Lm = gl.bl.normalize(row["x"], row["y"], row["th"])
        lbl = f"{labels[bi]}_cmelt"
        if row["best_L_feas"] is not None:
            pool.append((0.0, Lm, nx, ny, nth, lbl))
        elif row["pen"] < 0.02:
            pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
    pool.sort(key=lambda t: (t[0], t[1]))
    for _, _, cx, cy, cth, lbl in pool[:2]:
        cert = gl.bl.polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=10)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert

    if best_cert is not None and best_cert["L_float"] < float(inc_dec) + 0.002:
        rs = sorted(best_cert["rows"], key=lambda r: int(r["square_id"]))
        res_lp = lpslack.optimize_group(
            n, rs, str(inc_dec), outer_iters=5, fixed_rounds=8,
            slp_modes=("dual_non90", "non90"), tr_th_schedule=(4e-4, 1e-4), time_limit=10.0,
        )
        if res_lp.get("win") and res_lp.get("rows"):
            lit, pan = Decimal(res_lp["new_L_lit"]), Decimal(res_lp["new_L_pan"])
            worst = max(lit, pan)
            if worst < max(Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pandas"])):
                best_cert["L_lit"] = str(lit)
                best_cert["L_pandas"] = str(pan)
                best_cert["L_dec_str"] = str(worst)
                best_cert["L_float"] = float(worst)
                best_cert["dL"] = float(worst - inc_dec)
                best_cert["is_real_win"] = bool(worst - inc_dec <= Decimal("-1e-9"))
                best_cert["rows"] = res_lp["rows"]
                best_cert["label"] = best_cert["label"] + "+lpslack"
    return best_cert


def main() -> None:
    import math
    args = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [974, 912, 973, 913, 853, 975]
    frontier = gl.fast_load_true_frontier(args, certify=True)
    bricks = gate.load_all_bricks()
    for n in args:
        inc_dec = frontier[n]["L_dec"]
        wn = gl.DATA_DIR / f"win_n{n}.json"
        if wn.exists():
            import json
            wd = json.loads(wn.read_text(encoding="utf-8"))
            inc_dec = min(inc_dec, max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])))
        m = int(math.floor(float(inc_dec)))
        r = n - m * m
        f_inc = float(inc_dec) - m

        pairs = []
        seen = set()
        for i, b1 in enumerate(bricks):
            for j, b2 in enumerate(bricks):
                if i > j or b1["g"] + b2["g"] != r:
                    continue
                if b1["w"] + b2["w"] > m or b1["h"] + b2["h"] > m:
                    continue
                if b1["w"] + b2["w"] + b1["h"] + b2["h"] != 2 * m:
                    continue
                fmax = max(b1["f"], b2["f"])
                if fmax >= f_inc - 5e-4:
                    continue
                ptag = tuple(sorted([(b1["tag"], b1["w"], b1["h"], b1["g"]), (b2["tag"], b2["w"], b2["h"], b2["g"])]))
                if ptag in seen:
                    continue
                seen.add(ptag)
                pairs.append((fmax, b1, b2))
        pairs.sort(key=lambda t: t[0])
        print(f"\n[CORNER n={n} m={m} r=+{r} inc={float(inc_dec):.9f}]: {len(pairs)} corner pairs", flush=True)
        for pidx, (fmax, b1, b2) in enumerate(pairs[:2]):
            print(f"  try {b1['tag']}({b1['w']}x{b1['h']},g={b1['g']},f={b1['f']:.5f}) + {b2['tag']}({b2['w']}x{b2['h']},g={b2['g']},f={b2['f']:.5f}) fmax={fmax:.6f}", flush=True)
            cert = solve_corner_pair(n, m, b1, b2, inc_dec, f"cn{n}_{b1['tag']}_{b2['tag']}", n * 1000 + pidx)
            if cert is not None:
                print(f"  -> {cert['label']}: L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} win={cert['is_real_win']}", flush=True)
                if cert["is_real_win"]:
                    gl.save_win_locked(cert)
                    inc_dec = max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))


if __name__ == "__main__":
    main()
