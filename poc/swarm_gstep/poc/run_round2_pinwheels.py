#!/usr/bin/env python3
"""Round-2 H79 (gstep): Stronger 16-orientation 1-corner & zero-overlap pinwheel melts + lpslack."""
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
sys.path.insert(0, str(HERE))
import gstep_lib as gl  # noqa: E402
import run_corner_pinwheels as cpw  # noqa: E402

REPO_ROOT = gl.REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "poc" / "swarm_lpslack" / "poc"))
import lpslack  # noqa: E402


def load_extended_bricks() -> list[dict]:
    """Load source + gstep/data + b7xplant/data + b8xplant/data bricks, top-1 per (w, h, g)."""
    all_b: list[dict] = []

    def _add(b: dict, tag: str) -> None:
        d = dict(b)
        d["tag"] = tag
        all_b.append(d)
        dt = gl.bl.transform_brick(d, "T")
        dt["tag"] = tag + "_T"
        all_b.append(dt)

    for nm, b in gl.load_source_bricks().items():
        _add(b, nm)

    for folder in (
        gl.DATA_DIR,
        REPO_ROOT / "poc" / "swarm_b7xplant" / "data",
        REPO_ROOT / "poc" / "swarm_b8xplant" / "data",
    ):
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.json")):
            if "win_" in p.name:
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if all(k in d for k in ("w", "h", "g", "x", "y", "th")):
                b = gl.make_brick(
                    int(d["w"]), int(d["h"]), int(d["g"]),
                    np.array(d["x"], dtype=np.float64),
                    np.array(d["y"], dtype=np.float64),
                    np.array(d["th"], dtype=np.float64),
                    tag=p.stem,
                )
                _add(b, p.stem)

    best_shape: dict[tuple[int, int, int], dict] = {}
    for b in all_b:
        k = (b["w"], b["h"], b["g"])
        if k not in best_shape or b["f"] < best_shape[k]["f"]:
            best_shape[k] = b
    return [best_shape[k] for k in sorted(best_shape)]


def solve_pair_16(
    n: int, m: int, b1_base: dict, b2_base: dict, inc_dec: Decimal, tag: str, base_seed: int
) -> dict | None:
    seeds, masks, scheds, labels = [], [], [], []
    is_sym = (b1_base["tag"] == b2_base["tag"])
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = gl.bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = gl.bl.transform_brick(b2_base, m2)
            f_use = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            built = cpw.build_pinwheel_allow_corner(m, b1, b2, sf=0.0, f_override=f_use)
            if built is None:
                continue
            ax, ay, at, _, sm = built
            L_tgt = m + f_use
            for stype, off in (("sA", (0.026, 0.009, 0.002)), ("sB", (0.016, 0.004, 0.0))):
                if is_sym and stype == "sB" and (m1, m2) not in (("id", "id"), ("id", "flipx"), ("flipx", "flipy")):
                    continue
                seeds.append((ax, ay, at))
                masks.append(sm)
                scheds.append([L_tgt + off[0], L_tgt + off[1], L_tgt + off[2]])
                labels.append(f"{tag}_{m1}_{m2}_{stype}")

    if not seeds:
        return None

    best_cert: dict | None = None
    pool = []
    for start in range(0, len(seeds), 24):
        end = min(len(seeds), start + 24)
        relaxed = gl.bl.su.cpp_batch_melt(
            seeds[start:end], scheds[start:end], sym_mode=1, a_start=0.955, n_melt_stages=3,
            seam_masks=masks[start:end], seam_iters=180,
            full_iters=240, n_ruin_iters=0, n_threads=2, base_seed=base_seed + start,
        )
        for bi, row in enumerate(relaxed):
            nx, ny, nth, Lm = gl.bl.normalize(row["x"], row["y"], row["th"])
            lbl = f"{labels[start + bi]}_r2"
            if row["best_L_feas"] is not None:
                pool.append((0.0, Lm, nx, ny, nth, lbl))
            elif row["pen"] < 0.02:
                pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))

    pool.sort(key=lambda t: (t[0], t[1]))
    for _, _, cx, cy, cth, lbl in pool[:3]:
        cert = gl.bl.polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=12)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert

    if best_cert is not None and best_cert["L_float"] < float(inc_dec) + 0.0015:
        rs = sorted(best_cert["rows"], key=lambda r: int(r["square_id"]))
        res_lp = lpslack.optimize_group(
            n, rs, str(inc_dec), outer_iters=6, fixed_rounds=10,
            slp_modes=("dual_non90", "non90", "all"), tr_th_schedule=(5e-4, 1e-4, 3e-5), time_limit=12.0,
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
    t0 = time.time()
    args = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [912, 914, 976, 913, 915, 853, 854, 855, 797, 798, 742, 743]
    frontier = gl.fast_load_true_frontier(args, certify=True)
    bricks = load_extended_bricks()
    print(f"Loaded {len(bricks)} unique (w,h,g) bricks; targets={args}", flush=True)

    for n in args:
        inc_dec = frontier[n]["L_dec"]
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
                s = b1["w"] + b2["w"] + b1["h"] + b2["h"]
                if s > 2 * m:
                    continue
                fmax = max(b1["f"], b2["f"])
                if fmax >= f_inc - 1e-4:
                    continue
                ptag = tuple(sorted([(b1["tag"], b1["w"], b1["h"], b1["g"]), (b2["tag"], b2["w"], b2["h"], b2["g"])]))
                if ptag in seen:
                    continue
                seen.add(ptag)
                pairs.append((fmax, s == 2 * m, b1, b2))
        pairs.sort(key=lambda t: (t[0], t[1]))
        print(f"\n[R2 n={n} m={m} r=+{r} inc={float(inc_dec):.9f}]: {len(pairs)} winning-excess pairs", flush=True)
        for pidx, (fmax, is_corn, b1, b2) in enumerate(pairs[:2]):
            ov = "1CORN" if is_corn else "ZERO"
            print(f"  [{ov}] {b1['tag']}({b1['w']}x{b1['h']},g={b1['g']},f={b1['f']:.5f}) + {b2['tag']}({b2['w']}x{b2['h']},g={b2['g']},f={b2['f']:.5f}) fmax={fmax:.6f}", flush=True)
            cert = solve_pair_16(n, m, b1, b2, inc_dec, f"r2n{n}_{b1['tag']}_{b2['tag']}", n * 100 + pidx)
            if cert is not None:
                print(f"  -> {cert['label']}: L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} win={cert['is_real_win']}", flush=True)
                if cert["is_real_win"]:
                    gl.save_win_locked(cert)
                    inc_dec = max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
                    f_inc = float(inc_dec) - m
    print(f"\nROUND-2 PINWHEELS COMPLETE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
