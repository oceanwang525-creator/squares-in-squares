#!/usr/bin/env python3
"""Squeeze existing H94 wins and ultra-near-misses via pslp (kink-safe SLP) + lpslack."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "poc" / "row_ratchet_980" / "poc"))

import band3_lib as b3  # noqa: E402
sys.path.insert(0, str(HERE))
import pslp  # noqa: E402
from run_scale import _lpslack_polish, solve_pair_scale  # noqa: E402

for extra_dir in (
    REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data",
    REPO_ROOT / "poc" / "swarm_gstep" / "data",
    REPO_ROOT / "poc" / "swarm_stepdown" / "data",
    REPO_ROOT / "poc" / "swarm_peelstep" / "data",
):
    if extra_dir.exists() and extra_dir not in b3.EXTERNAL_BRICK_DIRS:
        b3.EXTERNAL_BRICK_DIRS.append(extra_dir)


def pslp_and_certify(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    inc_dec: Decimal,
    label: str,
    shake_rounds: int = 6,
    seed: int = 123,
) -> dict | None:
    m = int(math.floor(float(inc_dec)))
    nx, ny, nth, L0 = b3.bl.normalize(x, y, th)
    f0 = L0 - m
    f1, X1, Y1, T1, v1, _ = pslp.squeeze(nx, ny, nth, m, m, f0, max_it=250)
    best_x, best_y, best_t, best_f = X1, Y1, T1, f1
    if v1 > 1e-12:
        best_x, best_y, best_t, best_f = nx, ny, nth, f0

    if shake_rounds > 0:
        rng = np.random.default_rng(seed)
        f2, X2, Y2, T2 = pslp.shake_squeeze(best_x, best_y, best_t, m, m, best_f, rounds=shake_rounds, rng=rng)
        if f2 < best_f - 1e-10 and pslp.viol(X2, Y2, T2, m, m, f2) <= 1e-12:
            best_x, best_y, best_t, best_f = X2, Y2, T2, f2

    cert = b3.bl.polish_and_certify(n, best_x, best_y, best_t, inc_dec, slp_rounds=16)
    if cert is not None:
        cert["label"] = label + "+pslp"
        if cert["dL"] <= 0.002:
            cert = _lpslack_polish(n, cert, inc_dec, time_limit=20.0)
    return cert


def _rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rs = sorted(rows, key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in rs], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rs], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
    return x, y, th


def main() -> None:
    t0 = time.time()
    all_ns = [974, 743, 914, 916, 690, 745, 691]
    frontier = b3.gl.fast_load_true_frontier(all_ns, certify=True)

    print("=== 1) Checking & pslp-squeezing saved win_n914.json vs origin/main ===", flush=True)
    for n in [914]:
        wp = b3.DATA_DIR / f"win_n{n}.json"
        if not wp.exists():
            continue
        wd = json.loads(wp.read_text(encoding="utf-8"))
        inc_dec = frontier[n]["L_dec"]
        cur_dec = max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"]))
        x0, y0, th0 = _rows_to_arrays(wd["rows"])
        cert = pslp_and_certify(
            n, x0, y0, th0, inc_dec, wd.get("label", f"n{n}"), shake_rounds=2, seed=9000 + n
        )
        if cert is not None:
            new_dec = max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
            if new_dec < cur_dec - Decimal("1e-9") and new_dec < inc_dec - Decimal("1e-9"):
                b3.save_win_locked(cert)
                print(
                    f"  [IMPROVED n={n}] {cur_dec} -> {new_dec} (dL vs main = {cert['dL']:+.6e}, {cert['label']})",
                    flush=True,
                )
            else:
                print(
                    f"  [KEEP n={n}] cur={str(cur_dec)[:18]} (dL vs main={float(cur_dec - inc_dec):+.6e}) | pslp={str(new_dec)[:18]}",
                    flush=True,
                )

    print("\n=== 2) Testing near-misses & zero-ov n=916 with pslp + lpslack ===", flush=True)
    bricks = b3.load_all_bricks(include_external=True)
    rescue_specs = [
        (690, "8x18+18x8_ct1", lambda b: (b["w"], b["h"], b["g"]) in ((8, 18, 7), (18, 8, 7))),
        (916, "9x18+21x11_z", lambda b: True),
        (745, "9x18+18x9_ct1", lambda b: (b["w"], b["h"], b["g"]) in ((9, 18, 8), (18, 9, 8))),
        (691, "8x17+18x9_ct1", lambda b: (b["w"], b["h"], b["g"]) in ((8, 17, 7), (18, 9, 8))),
    ]

    for n, tag_lbl, pred in rescue_specs:
        inc_dec = frontier[n]["L_dec"]
        inc_f = float(inc_dec)
        m = int(math.floor(inc_f))
        r = n - m * m
        f_inc = inc_f - m
        sub = [b for b in bricks if pred(b)]
        cands = []
        seen_k = set()
        for i, b1 in enumerate(sub):
            for j, b2 in enumerate(sub):
                if i > j or b1["g"] + b2["g"] != r:
                    continue
                w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
                if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
                    continue
                if w1 + w2 <= m and h1 + h2 <= m:
                    fm = max(b1["f"], b2["f"])
                    if fm < f_inc - 3e-4:
                        k = (w1, h1, b1["g"], w2, h2, b2["g"])
                        if k not in seen_k:
                            seen_k.add(k)
                            cands.append((fm, b1, b2))
        cands.sort(key=lambda t: t[0])
        print(f"\n--- Rescue n={n} (m={m}, r=+{r}) inc={inc_dec} cands={len(cands)} ---", flush=True)
        for idx, (fm, b1, b2) in enumerate(cands[:2]):
            cert, diag = solve_pair_scale(n, m, b1, b2, inc_dec, f"n{n}_{tag_lbl}_{idx}", base_seed=70000 + n + idx)
            if cert is not None:
                cx, cy, ct = _rows_to_arrays(cert["rows"])
                cert2 = pslp_and_certify(
                    n, cx, cy, ct, inc_dec, cert["label"], shake_rounds=2, seed=80000 + n + idx
                )
                if cert2 is not None and cert2["L_float"] < cert["L_float"]:
                    cert = cert2
                print(
                    f"  -> L={cert['L_float']:.9f} dL={cert['dL']:+.6e} win={cert['is_real_win']} ({cert['label']})",
                    flush=True,
                )
                if cert["is_real_win"]:
                    b3.save_win_locked(cert)
                    print(f"*** NEW WIN n={n}: dL={cert['dL']:+.6e} ***", flush=True)
            else:
                print(f"  -> None | diag={diag[:2]}", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
