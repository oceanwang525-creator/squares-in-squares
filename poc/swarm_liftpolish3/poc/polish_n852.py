#!/usr/bin/env python3
"""Polish built_partb/n852.json with lpslack + 4-round mechwalk_compress."""
from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

import json
import sys
from decimal import Decimal
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "swarm_lpslack" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_homocasc2" / "poc"))
import lpslack  # noqa: E402
import homocasc2_lib as hc  # noqa: E402


def main() -> None:
    p = REPO / "poc/swarm_liftpolish3/data/built_partb/n852.json"
    rows = json.loads(p.read_text(encoding="utf-8"))
    old_res = lpslack.certify_pair(852, rows)
    old_L = max(old_res[0], old_res[1])
    print(f"n=852 build_cert L={old_L}", flush=True)

    res = lpslack.optimize_group(
        852,
        rows,
        str(old_L),
        outer_iters=10,
        fixed_rounds=10,
        tr_th_schedule=(1e-3, 5e-4, 1e-4, 5e-5),
    )
    print("lpslack n=852:", res["status"], "old=", res.get("old_L"), "new=", res.get("new_L"), "dL=", res.get("dL"), flush=True)
    cur_rows = res["rows"] if res.get("status") in ("WIN", "TINY_WIN") and "rows" in res else rows
    cur_L = Decimal(res["new_L"]) if res.get("status") in ("WIN", "TINY_WIN") and "new_L" in res else old_L

    x, y, th, _ = hc._rows_to_arrays(cur_rows)
    rng = np.random.default_rng(8522026)
    cert, _ = hc.mechwalk_compress(852, x, y, th, cur_L, rng, rounds=4, threads=1)
    if cert is not None and cert.get("is_real_win"):
        cur_rows = cert["rows"]
        cur_L = Decimal(max(cert["L_lit"], cert["L_pd"]))
        print(f"mechwalk n=852 WIN new_L={cur_L} dL={cert['dL']}", flush=True)

    win_path = REPO / "poc/swarm_liftpolish3/data/wins/win_n852.json"
    win_path.parent.mkdir(parents=True, exist_ok=True)
    c_final = hc.certify_rows_against_frontier(852, cur_rows, old_L + Decimal("1.0"))
    win_path.write_text(json.dumps({
        "n": 852,
        "label": "partb_dp_lpslack_mechwalk",
        "L_lit": c_final["L_lit"],
        "L_pd": c_final["L_pd"],
        "L_kr": c_final["L_kr"],
        "L_float": c_final["L_float"],
        "dL": str(cur_L - old_L),
        "rows": cur_rows,
    }, indent=2), encoding="utf-8")
    print("Saved", win_path, flush=True)


if __name__ == "__main__":
    main()
