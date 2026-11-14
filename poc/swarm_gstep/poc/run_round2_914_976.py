#!/usr/bin/env python3
"""Targeted Round-2 surgical sweep on n=853, n=914 (B7_9x21_vs_bottom), n=976 (B8_11x20_vs_k4 + B7_11x19)."""
from __future__ import annotations

import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gstep_lib as gl  # noqa: E402
import run_gstep_gate as gate  # noqa: E402
import run_round2_pinwheels as r2p  # noqa: E402


def main() -> None:
    t0 = time.time()
    bricks = {b["tag"]: b for b in r2p.load_extended_bricks()}
    # Ensure exact bricks are loaded by filename
    for nm in ("brick_B7_9x21_vs_bottom", "brick_B7_9x21_bottom", "brick_B8_11x20_vs_k4", "brick_B7_11x19_n975_cut", "brick_B6_9x18_n796_cut", "B6_10x20_mw972_top"):
        p = gl.DATA_DIR / f"{nm}.json"
        if not p.exists():
            p = gl.REPO_ROOT / "poc" / "swarm_b7xplant" / "data" / f"{nm}.json"
        if p.exists():
            import json, numpy as np
            d = json.loads(p.read_text(encoding="utf-8"))
            b = gl.make_brick(int(d["w"]), int(d["h"]), int(d["g"]), np.array(d["x"]), np.array(d["y"]), np.array(d["th"]), tag=nm)
            bricks[nm] = b
            bt = gl.bl.transform_brick(b, "T")
            bt["tag"] = nm + "_T"
            bricks[nm + "_T"] = bt

    targets = [853, 914, 976]
    fr = gl.fast_load_true_frontier(targets, certify=True)

    # 1. n=853: zero-overlap direct + lpslack (B6_9x18_n796_cut + B6_10x20_mw972_top_T)
    if "brick_B6_9x18_n796_cut" in bricks and "B6_10x20_mw972_top_T" in bricks:
        inc = fr[853]["L_dec"]
        print(f"\n[n=853 inc={float(inc):.9f}] testing B6_9x18_n796_cut + B6_10x20_mw972_top_T...", flush=True)
        cert = gate.solve_pair_fast(853, 29, bricks["brick_B6_9x18_n796_cut"], bricks["B6_10x20_mw972_top_T"], inc, "r2_853_zero", 85301, do_melt=False)
        if cert is not None:
            print(f"  -> {cert['label']}: L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} win={cert['is_real_win']}", flush=True)
            if cert["is_real_win"]:
                gl.save_win_locked(cert)

    # 2. n=976: zero-overlap direct + lpslack + melt (B8_11x20_vs_k4 + B7_11x19_n975_cut_T)
    if "brick_B8_11x20_vs_k4" in bricks and "brick_B7_11x19_n975_cut_T" in bricks:
        inc = fr[976]["L_dec"]
        print(f"\n[n=976 inc={float(inc):.9f}] testing B8_11x20_vs_k4 + B7_11x19_n975_cut_T (ZERO OV)...", flush=True)
        cert = gate.solve_pair_fast(976, 31, bricks["brick_B8_11x20_vs_k4"], bricks["brick_B7_11x19_n975_cut_T"], inc, "r2_976_zero", 97601, do_melt=True)
        if cert is not None:
            print(f"  -> {cert['label']}: L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} win={cert['is_real_win']}", flush=True)
            if cert["is_real_win"]:
                gl.save_win_locked(cert)

    # 3. n=914: 16-orientation 1-corner melt on B7_9x21_vs_bottom + T
    if "brick_B7_9x21_vs_bottom" in bricks:
        inc = fr[914]["L_dec"]
        print(f"\n[n=914 inc={float(inc):.9f}] testing B7_9x21_vs_bottom(f={bricks['brick_B7_9x21_vs_bottom']['f']:.5f}) + T...", flush=True)
        cert = r2p.solve_pair_16(914, 30, bricks["brick_B7_9x21_vs_bottom"], bricks["brick_B7_9x21_vs_bottom_T"], inc, "r2_914_9x21vs", 91401)
        if cert is not None:
            print(f"  -> {cert['label']}: L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} win={cert['is_real_win']}", flush=True)
            if cert["is_real_win"]:
                gl.save_win_locked(cert)

    print(f"\nDone targeted 853/976/914 in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
