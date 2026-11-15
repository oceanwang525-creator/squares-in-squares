#!/usr/bin/env python3
"""Hypothesis H77 Round 3: Walk new 0e74cfc pocketpeel/merged bricks + targeted zero-overlap pinwheels."""
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


NEW_POCKETPEEL_BRICKS = [
    ("pp_n975_9x22_g7", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_n975_9x22_g7.json"),
    ("pp_n975_22x9_g7", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_n975_22x9_g7.json"),
    ("pp_n855_20x11_g8", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_p_n855_20x11_g8.json"),
    ("pp_n854_19x11_g7", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_p_n854_19x11_g7.json"),
    ("pp_n914_13x20_g8", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_p_n914_13x20_g8.json"),
    ("pp_n915_21x12_g9", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_p_n915_21x12_g9.json"),
    ("pp_n915_7x18_g6", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_p_n915_7x18_g6.json"),
    ("pp_n916_8x19_g7", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_p_n916_8x19_g7.json"),
    ("pp_n977_20x9_g7", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_p_n977_20x9_g7.json"),
    ("pp_n974_22x10_g7", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_n974_22x10_g7.json"),
    ("pp_n974_9x21_g6", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_n974_9x21_g6.json"),
    ("pp_n917_8x18_g7", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_p_n917_8x18_g7.json"),
    ("pp_n857_21x11_g9", REPO_ROOT / "poc/swarm_pocketpeel/data/bricks/brick_p_n857_21x11_g9.json"),
]


def _walk_job(args: tuple[dict, str]) -> tuple[str, float, float]:
    b, name = args
    res = gf.quick_walk_b8(b, name)
    return name, float(b["f"]), float(res["f"])


def load_json_brick(path: Path, name: str) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    x = np.array(d["x"], dtype=np.float64)
    y = np.array(d["y"], dtype=np.float64)
    th = np.array(d["th"], dtype=np.float64)
    w, h, g = int(d["w"]), int(d["h"]), int(d["g"])
    bx, by, Wf, Hf = bl.rect_bbox(x, y, th)
    fx, fy = float(Wf - w), float(Hf - h)
    f = max(fx, fy)
    return {
        "name": name, "tag": name, "w": w, "h": h, "g": g,
        "src_n": int(d.get("src_n", 0)), "f0": float(d.get("f0", f)),
        "fx": fx, "fy": fy, "f": f, "df": float(d.get("df", 0.0)),
        "x": bx, "y": by, "th": th,
    }


def main() -> None:
    t0 = time.time()
    target_ns = [975, 855, 976, 915, 916, 856, 977, 798, 743]
    port = b7.load_portfolio_targets(target_ns + [854, 914])

    # 1. Queue un-walked pocketpeel bricks
    to_walk: list[tuple[dict, str]] = []
    for name, path in NEW_POCKETPEEL_BRICKS:
        out_p = b7.DATA_DIR / f"brick_{name}.json"
        if not out_p.exists() and path.exists():
            to_walk.append((load_json_brick(path, name), name))

    # Also harvest tight clean corner cuts from merged.csv for 914 (m=30, f=0.729754) and 975 (m=31, f=0.715878) and 856/977 (0.744188)
    for src_n, specs in [
        (914, [(12, 20, 8), (11, 20, 7), (10, 20, 6), (13, 20, 8)]),
        (975, [(9, 22, 7), (10, 22, 8), (11, 21, 8), (11, 20, 7)]),
        (977, [(11, 22, 9), (12, 21, 9), (9, 20, 7), (10, 21, 8)]),
        (743, [(9, 18, 7), (10, 18, 7), (8, 17, 6), (11, 18, 8)]),
    ]:
        px, py, pth, pL = port[src_n]["x"], port[src_n]["y"], port[src_n]["th"], port[src_n]["L"]
        for w, h, g in specs:
            for corner in ("TL", "TR", "BL", "BR"):
                nm = f"cut_n{src_n}_{w}x{h}_g{g}_{corner}"
                if (b7.DATA_DIR / f"brick_{nm}.json").exists():
                    continue
                b = b7.extract_corner_cut(px, py, pth, pL, corner, w, h, g, src_n, nm)
                if b is not None and b["f"] < 0.76:
                    to_walk.append((b, nm))
                    break

    print(f"[STAGE 1] Walking {len(to_walk)} new g=+6..+9 bricks with 2 workers...", flush=True)
    if to_walk:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
            for name, f0, f1 in ex.map(_walk_job, to_walk):
                print(f"  [WALKED] {name}: {f0:.9f} -> {f1:.9f} (df={f1-f0:+.6e})", flush=True)

    # 2. Build full brick catalog (original + transpose) from poc/swarm_b7xplant/data/brick_*.json
    catalog: list[dict] = []
    for p in sorted(b7.DATA_DIR.glob("brick_*.json")):
        b = load_json_brick(p, p.stem[len("brick_"):])
        catalog.append(b)
        if b["w"] != b["h"]:
            bt = bl.transform_brick(b, "T")
            bt["name"] = b["name"] + "_T"
            catalog.append(bt)

    # 3. For each target n, pick top zero-overlap pairs (b1, b2) within f_gap <= 0.012
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
        for i, b1 in enumerate(catalog):
            if b1["f"] > f_inc + 0.012 or b1["g"] < 5:
                continue
            for j, b2 in enumerate(catalog):
                if b2["f"] > f_inc + 0.012 or b2["g"] < 5:
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
        print(f"\n[TARGET n={n}] ({m},+{r}) f_inc={f_inc:.8f} -> {len(cands)} candidate pairs (running top 3)", flush=True)
        for rank, (gap, _, b1, b2) in enumerate(cands[:3]):
            lbl = f"{b1['name']}+{b2['name']}"
            print(f"  [{rank+1}/3] {lbl} ({b1['w']}x{b1['h']}+{b1['g']} + {b2['w']}x{b2['h']}+{b2['g']}) gap={gap:+.6e}", flush=True)
            res = rs2.run_pair(n, m, b1, b2, inc, f"r3_n{n}_{rank}", seed=77300 + n * 13 + rank * 37)
            if res is not None:
                report[f"n{n}_{rank}_{lbl}"] = {
                    "L": res["L_dec_str"], "dL": res["dL"], "win": bool(res["is_real_win"]), "label": res["label"],
                }
                if res["is_real_win"]:
                    inc = min(inc, Decimal(res["L_dec_str"]))
                    wins.append((n, res["L_dec_str"], res["dL"], lbl))

    (b7.DOCS_DIR / "scale3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[ROUND 3 COMPLETE] in {time.time()-t0:.1f}s wins={len(wins)}: {wins}", flush=True)


if __name__ == "__main__":
    main()
