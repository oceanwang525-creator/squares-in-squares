#!/usr/bin/env python3
"""Hypothesis H77 Round 3b (vs origin/main ccdac8f):
1. Walk new vstrip492 g=6,7 bricks (B7_9x22_n975_TL_mw, B7_10x22, etc.) + ccdac8f merged cuts.
2. Targeted zero-overlap + seam-melt pinwheels + deep pslp.shake_squeeze + lpslack
   for n in [975, 974, 976, 855, 916, 856, 977, 915, 798, 743].
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

POC_DIR = Path(__file__).resolve().parent
REPO_ROOT = POC_DIR.parents[2]
RATCHET_POC = REPO_ROOT / "poc" / "row_ratchet_980" / "poc"
for _p in (POC_DIR, RATCHET_POC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b7_lib as b7  # noqa: E402
import b8_lib as bl  # noqa: E402
import run_gate_fast as gf  # noqa: E402
import run_scale2 as rs2  # noqa: E402
import run_scale3 as rs3  # noqa: E402


EXTRA_BRICKS = [
    ("vs492_9x22_n975_g7", REPO_ROOT / "poc/swarm_vstrip492/data/brick_B7_9x22_n975_TL_mw.json"),
    ("vs492_10x22_n911_vs0_g7", REPO_ROOT / "poc/swarm_vstrip492/data/brick_B6_10x21_n911_BR_mw_vs0_B7_10x22.json"),
    ("vs492_10x22_n911_vs1_g7", REPO_ROOT / "poc/swarm_vstrip492/data/brick_B6_10x21_n911_BR_mw_vs1_B7_10x22.json"),
    ("vs492_10x21_n911_g6", REPO_ROOT / "poc/swarm_vstrip492/data/brick_B6_10x21_n911_BR_mw.json"),
    ("vs492_9x21_n975_g6", REPO_ROOT / "poc/swarm_vstrip492/data/brick_B6_9x21_n975_TL_mw.json"),
    ("vs492_10x20_n852_g6", REPO_ROOT / "poc/swarm_vstrip492/data/brick_B6_10x20_n852_TL_mw.json"),
]


def _walk_job(args: tuple[dict, str]) -> tuple[str, float, float]:
    b, name = args
    res = gf.quick_walk_b8(b, name)
    return name, float(b["f"]), float(res["f"])


def main() -> None:
    t0 = time.time()
    target_ns = [975, 974, 976, 855, 916, 856, 977, 915, 798, 743]
    port = b7.load_portfolio_targets(target_ns + [914, 854, 913])

    # Clear stale untracked win_n975.json first so save_win_b7 never re-exports an uncertified file
    old_975_path = b7.DATA_DIR / "win_n975.json"
    old_975_data = None
    if old_975_path.exists():
        old_975_data = json.loads(old_975_path.read_text(encoding="utf-8"))
        old_975_path.unlink()

    to_walk: list[tuple[dict, str]] = []
    for name, path in EXTRA_BRICKS:
        out_p = b7.DATA_DIR / f"brick_{name}.json"
        if not out_p.exists() and path.exists():
            to_walk.append((rs3.load_json_brick(path, name), name))

    # Extract clean g=6..9 cuts from updated ccdac8f merged.csv (including gstep's n=975, n=973, vstrip852's n=854/913)
    for src_n, specs in [
        (975, [(9, 22, 7), (10, 22, 8), (11, 21, 8)]),
        (854, [(9, 18, 6), (11, 20, 7), (12, 20, 8)]),
        (913, [(9, 18, 6), (11, 21, 7)]),
        (743, [(9, 18, 7), (10, 18, 7), (11, 18, 8)]),
    ]:
        px, py, pth, pL = port[src_n]["x"], port[src_n]["y"], port[src_n]["th"], port[src_n]["L"]
        for w, h, g in specs:
            for corner in ("TL", "TR", "BL", "BR"):
                nm = f"cut2_n{src_n}_{w}x{h}_g{g}_{corner}"
                if (b7.DATA_DIR / f"brick_{nm}.json").exists():
                    continue
                b = b7.extract_corner_cut(px, py, pth, pL, corner, w, h, g, src_n, nm)
                if b is not None and b["f"] < 0.76:
                    to_walk.append((b, nm))
                    break

    print(f"[STAGE 1] Walking {len(to_walk)} bricks with 2 workers...", flush=True)
    if to_walk:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
            for name, f0, f1 in ex.map(_walk_job, to_walk):
                print(f"  [WALKED] {name}: {f0:.9f} -> {f1:.9f} (df={f1-f0:+.6e})", flush=True)

    # First polish old win_n975 seed with deep pslp + lpslack against new ccdac8f n=975
    if old_975_data is not None:
        rs = sorted(old_975_data["rows"], key=lambda r: int(r["square_id"]))
        ox = np.array([float(r["x"]) for r in rs], dtype=np.float64)
        oy = np.array([float(r["y"]) for r in rs], dtype=np.float64)
        ot = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
        print(f"[POLISH OLD n=975] vs ccdac8f inc={str(port[975]['L_dec'])[:24]}", flush=True)
        c = rs2.deep_polish_and_certify(975, ox, oy, ot, port[975]["L_dec"], "repolish_975_ccdac8f")
        if c is not None:
            print(f"  -> {c['label']}: L={c['L_dec_str'][:24]} dL={c['dL']:+.6e} win={c['is_real_win']}", flush=True)
            if c["is_real_win"]:
                b7.save_win_b7(c)

    # Load full brick catalog
    catalog: list[dict] = []
    for p in sorted(b7.DATA_DIR.glob("brick_*.json")):
        b = rs3.load_json_brick(p, p.stem[len("brick_"):])
        catalog.append(b)
        if b["w"] != b["h"]:
            bt = bl.transform_brick(b, "T")
            bt["name"] = b["name"] + "_T"
            catalog.append(bt)

    wins = []
    report = {}
    for n in target_ns:
        m = int(math.isqrt(n))
        r = n - m * m
        inc = port[n]["L_dec"]
        wp = b7.DATA_DIR / f"win_n{n}.json"
        if wp.exists():
            inc = min(inc, Decimal(json.loads(wp.read_text(encoding="utf-8"))["L_dec_str"]))
        f_inc = float(inc) - m

        cands = []
        seen_sigs = set()
        for b1 in catalog:
            if b1["f"] > f_inc + 0.008 or b1["g"] < 5:
                continue
            for b2 in catalog:
                if b2["f"] > f_inc + 0.008 or b2["g"] < 5:
                    continue
                if b1["g"] + b2["g"] != r:
                    continue
                if b1["w"] + b2["w"] > m or b1["h"] + b2["h"] > m:
                    continue
                f_hi = max(b1["f"], b2["f"])
                sig = (b1["w"], b1["h"], b1["g"], b2["w"], b2["h"], b2["g"], round(f_hi, 7))
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                cands.append((f_hi - f_inc, max(b1["f"], b2["f"]) + min(b1["f"], b2["f"]), b1, b2))
        cands.sort(key=lambda t: (t[0], t[1]))
        print(f"\n[TARGET n={n}] ({m},+{r}) f_inc={f_inc:.8f} -> {len(cands)} candidate pairs (running top 2)", flush=True)
        for rank, (gap, _, b1, b2) in enumerate(cands[:2]):
            lbl = f"{b1['name']}+{b2['name']}"
            print(f"  [{rank+1}/2] {lbl} ({b1['w']}x{b1['h']}+{b1['g']} + {b2['w']}x{b2['h']}+{b2['g']}) gap={gap:+.6e}", flush=True)
            res = rs2.run_pair(n, m, b1, b2, inc, f"r3b_n{n}_{rank}", seed=77400 + n * 17 + rank * 41)
            if res is not None:
                report[f"n{n}_{rank}_{lbl}"] = {
                    "L": res["L_dec_str"], "dL": res["dL"], "win": bool(res["is_real_win"]), "label": res["label"],
                }
                if res["is_real_win"]:
                    inc = min(inc, Decimal(res["L_dec_str"]))
                    wins.append((n, res["L_dec_str"], res["dL"], lbl))

    (b7.DOCS_DIR / "scale3b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ROUND 3B COMPLETE] in {time.time()-t0:.1f}s wins={len(wins)}: {wins}", flush=True)


if __name__ == "__main__":
    main()
