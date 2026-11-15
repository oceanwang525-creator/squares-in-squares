#!/usr/bin/env python3
"""Sanitize and merge all scanned pockets into both Pareto-pruned (for seam DP) and unpruned best-(w,h,g) (for overhang DP)."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "swarm_softseamdp" / "poc"))
import dp_soft  # noqa: E402
import merge_and_sanitize as ms  # noqa: E402

DATA_DIR = REPO / "poc" / "swarm_highdpocket" / "data"


def main() -> None:
    inputs = [
        DATA_DIR / "base_pockets_sane.json",
        DATA_DIR / "bricks_pockets.json",
        DATA_DIR / "scan_b1_valid.json",
        DATA_DIR / "scan_b2_a.json",
        DATA_DIR / "scan_b2_b.json",
        DATA_DIR / "scan_005_a.json",
        DATA_DIR / "scan_005_b.json",
        DATA_DIR / "scan_b3_a.json",
        DATA_DIR / "scan_b3_b.json",
    ]
    all_recs = []
    for p in inputs:
        if p.exists():
            recs = json.loads(p.read_text())
            print(f"  loaded {len(recs)} from {p.name}", flush=True)
            all_recs.extend(recs)

    sane = ms.sanitize_records(all_recs, str(DATA_DIR / "donors_all.csv"))

    best: dict[tuple[int, int, int], dict] = {}
    for g in sane:
        gw, gh, gg, gf = int(g["w"]), int(g["h"]), int(g["g"]), float(g["f"])
        for w, h in ((gw, gh), (gh, gw)):
            k = (w, h, gg)
            if k not in best or gf < best[k]["f"] - 1e-12:
                rec = dict(g)
                rec["w"], rec["h"] = w, h
                rec["nw"], rec["nh"] = gw, gh
                best[k] = rec

    # 1. Unpruned best-(w,h,g)
    surv_np = {}
    for k, v in best.items():
        nk = (int(v["src"]), float(v["x0"]), float(v["x1"]), float(v["y0"]), float(v["y1"]), int(v["g"]))
        if nk not in surv_np:
            rec = {kk: v[kk] for kk in ("src", "w", "h", "g", "f", "cnt", "x0", "x1", "y0", "y1")}
            rec["w"], rec["h"] = v["nw"], v["nh"]
            surv_np[nk] = rec
    np_list = sorted(surv_np.values(), key=lambda v: (v["g"], v["f"]))
    (DATA_DIR / "pockets_all_nopareto.json").write_text(json.dumps(np_list, indent=0))
    print(f"Wrote {len(np_list)} unpruned best-(w,h,g) pockets -> pockets_all_nopareto.json", flush=True)

    # 2. Pareto-pruned
    items = dp_soft.pareto([dict(w=v["w"], h=v["h"], g=v["g"], f=v["f"]) for v in best.values()])
    keep = {(it["w"], it["h"], it["g"]) for it in items}
    surv_p = {}
    for k, v in best.items():
        if k not in keep:
            continue
        nk = (int(v["src"]), float(v["x0"]), float(v["x1"]), float(v["y0"]), float(v["y1"]), int(v["g"]))
        if nk not in surv_p:
            rec = {kk: v[kk] for kk in ("src", "w", "h", "g", "f", "cnt", "x0", "x1", "y0", "y1")}
            rec["w"], rec["h"] = v["nw"], v["nh"]
            surv_p[nk] = rec
    p_list = sorted(surv_p.values(), key=lambda v: (v["g"], v["f"]))
    (DATA_DIR / "pockets_all_sane.json").write_text(json.dumps(p_list, indent=0))
    print(f"Wrote {len(p_list)} Pareto-pruned pockets -> pockets_all_sane.json", flush=True)


if __name__ == "__main__":
    main()
