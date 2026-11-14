"""Two cheap diagnostics for the minus-E / shear family.

(A) Seam probe.  Split the (K+1)x(K+1) minus-L assembly along its anti-diagonal
    staircase and measure the largest perpendicular slide d for which the two
    halves stay overlap-free, i.e. how much the box can be closed by pushing the
    top-right block by (-d,-d) (H5's stated move).  Also the LP side of the
    deepest feasible slide.

(B) Protrusion bound.  Any square that adjoins a straight K-edge protrudes by
    (|cos t| + |sin t|)/2 >= 1/sqrt 2, so a packing that keeps an intact KxK
    upright block can never have side < K + 1/sqrt 2.  Printed next to the
    incumbent and frontier sides for the three targets.

Run:  python poc/minusE_shear/poc/seam_probe.py --n-list 492,908,328
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_core as LP  # noqa: E402
import minusE as M  # noqa: E402

SQ2 = math.sqrt(0.5)


def halves(n):
    """Return [(pattern_name, K, D, G, cells, c0, a0, cuts)] for a few patterns."""
    K = math.isqrt(n)
    D = n - K * K
    G = K + 1
    inner = [(i, j) for i in range(K) for j in range(K)]
    pats = {
        "topL": [(x, G - 1) for x in range(min(D, G))],
        "corner4_4": ([(x, G - 1) for x in range(G - D // 2, G)] if D else []),
        "outer": ([(x, G - 1) for x in range(D // 2)]
                  + [(G - 1, y) for y in range(D - D // 2)]),
    }
    out = []
    for pname, keep in pats.items():
        if not keep:
            continue
        cells = inner + list(keep)
        c0, a0 = M.state_from_cells(cells)
        ii = np.array([c[0] for c in cells])
        jj = np.array([c[1] for c in cells])
        cuts = {
            "halfTR(i+j>=G-1)": np.flatnonzero(ii + jj >= G - 1),
            "stripUp(i+j>=G)": np.flatnonzero(ii + jj >= G),
            "rightStrip(i>=G-2)": np.flatnonzero(ii >= G - 2),
            "topStrip(j>=G-2)": np.flatnonzero(jj >= G - 2),
        }
        out.append((pname, K, D, G, cells, c0, a0, cuts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", default="492,908,328")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "docs" / "seam_probe.json"))
    args = ap.parse_args()

    gap = {int(r["n"]): r for r in csv.DictReader(
        open(REPO / "poc/koboshchan_coordinator/data/gap_vs_frontier.csv"))}
    report = {}
    for n in (int(x) for x in args.n_list.split(",") if x.strip()):
        ours = Decimal(gap[n]["ours"])
        front = Decimal(gap[n]["frontier"])
        entry = dict(n=n, ours=str(ours), frontier=str(front),
                     bound_K_plus_inv_sqrt2=math.isqrt(n) + SQ2,
                     bound_minus_frontier=math.isqrt(n) + SQ2 - float(front),
                     bound_minus_ours=math.isqrt(n) + SQ2 - float(ours),
                     patterns={})
        for pname, K, D, G, cells, c0, a0, cuts in halves(n):
            entry.update(n=n, K=K, D=D, G=G)
            base_side = LP.exact_side(c0, a0)
            pat = dict(base_side=base_side, cuts={})
            for name, idx in cuts.items():
                if len(idx) == 0 or len(idx) == len(cells):
                    continue
                best_d = 0.0
                best_lp = None
                for k in range(1, args.steps + 1):
                    d = k / args.steps
                    for sign in (-1.0, +1.0):
                        cc = M.translate_block(c0, idx, (sign * d, sign * d))
                        if M.overlaps_pairs(cc, a0):
                            continue
                        side = LP.exact_side(cc, a0)
                        if side < base_side - 1e-12 and d > best_d:
                            best_d = d
                            out = LP.fixed_lp_refine(a0, cc, rounds=8)
                            best_lp = out[0] if out else None
                # slides along the seam direction (d, -d) never change the box
                pat["cuts"][name] = dict(block=len(idx),
                                         deepest_bbox_reducing_slide=best_d,
                                         lp_side_at_that_slide=best_lp)
            entry["patterns"][pname] = pat
            print(f"n={n} [{pname}] K={K} D={D} base={base_side:.6f}: "
                  f"K+1/sqrt2={K+SQ2:.9f} ours={float(ours):.9f} "
                  f"frontier={float(front):.9f}", flush=True)
            for name, val in pat["cuts"].items():
                print(f"   {name:22s} block={val['block']:4d} "
                      f"deepest_bbox_reducing_slide="
                      f"{val['deepest_bbox_reducing_slide']:.2f} "
                      f"lp={val['lp_side_at_that_slide']}", flush=True)
        report[n] = entry
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1, default=float)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
