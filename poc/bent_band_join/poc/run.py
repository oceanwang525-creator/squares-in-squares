"""Build, certify and report the bent-band / monolithic-band target results.

Run from the repository root::

    python3 poc/bent_band_join/poc/run.py --out-dir /tmp/bent_band_out

Every emitted packing is certified by the unchanged official metric on the
literal CSV and on a plain ``pandas.read_csv(...).to_csv(index=False)``
round-trip (pandas 3.1.0).  Nothing is written inside the repository unless
``--out-dir`` points there.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))

from squares.core import certify, read_submission  # noqa: E402

import bent_band as bb  # noqa: E402
import search  # noqa: E402

TARGETS = {
    645: "25.87031812251975364740446552983030849778887781162520",
    751: "27.86424008616910855961199900976410503972147125750780",
    925: "30.86321837316950898175795253708386667312879581004381",
    987: "31.86630526508366147711534673847566523363639134913683",
}

RATIONAL_ANGLES = {
    "45": math.pi / 4,
    "arctan(1/2)": math.atan(1 / 2),
    "arctan(2/3)": math.atan(2 / 3),
    "arctan(3/2)": math.atan(3 / 2),
    "30": math.pi / 6,
}


def bent_capacity_at(L: float, limit: int = 2400):
    """Best bent two-block capacity at side L over a small discrete grid."""
    best = (0, None)
    tried = 0
    for t1n, t1 in RATIONAL_ANGLES.items():
        for t2n, t2 in RATIONAL_ANGLES.items():
            for w1 in (2, 3, 4, 5, 6):
                for w2 in (2, 3, 4, 5, 6):
                    for l1 in range(10, 40, 3):
                        for l2 in range(10, 40, 3):
                            tried += 1
                            if tried > limit:
                                return best, tried
                            pts, stats = bb.bent_two_block(L, t1, t2, w1, w2, l1, l2)
                            cap = len(pts)
                            if cap > best[0]:
                                best = (cap, dict(t1=t1n, t2=t2n, w1=w1, w2=w2,
                                                  l1=l1, l2=l2, **stats))
    return best, tried


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE.parent / "data"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    total_delta = 0.0
    for n, inc_str in TARGETS.items():
        inc = float(inc_str)
        t0 = time.time()
        best, tried = search.best_exact(n)
        cap_inc, cap_params = search.max_capacity_at(inc)
        if best is None:
            results.append(dict(n=n, status="no_exact_configuration"))
            continue
        L, w, l, a1, a2, b1, b2 = best
        q, Lb, drop = bb.monolithic_45(w, l, a1, a2, b1, b2)
        rows = bb.to_rows(q, dilate=1.00000002, decimals=17)

        lit_csv = out / f"n{n}-literal.csv"
        lit_csv.write_text(bb.csv_text(n, rows), encoding="utf-8")
        L_lit = certify(n, read_submission(lit_csv)[n])

        pd_csv = out / f"n{n}-pandas.csv"
        pd.read_csv(lit_csv).to_csv(pd_csv, index=False)
        L_pd = certify(n, read_submission(pd_csv)[n])

        delta = float(L_pd) - inc
        contrib = delta / math.sqrt(n)
        total_delta += contrib
        (bent_cap, bent_best), bent_tried = bent_capacity_at(inc)
        results.append(dict(
            n=n, status="certified", params=dict(w=w, l=l, a1=a1, a2=a2, b1=b1, b2=b2),
            search_tried=tried, predicted_L=L, L_literal=L_lit, L_pandas=L_pd,
            incumbent_L=inc_str, delta_L=delta, score_delta=contrib,
            max_capacity_at_incumbent=cap_inc,
            shortfall_at_incumbent=n - cap_inc,
            max_capacity_params=cap_params,
            bent_capacity_at_incumbent=bent_cap,
            bent_best_params=bent_best,
            bent_configs_tried=bent_tried,
            seconds=round(time.time() - t0, 1),
        ))
        print(json.dumps(results[-1], indent=2))

    summary = dict(
        score_delta_total=total_delta,
        reachable=total_delta < 0,
        results=results,
    )
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TOTAL score delta:", total_delta)
    print("reachable:", total_delta < 0)


if __name__ == "__main__":
    main()
