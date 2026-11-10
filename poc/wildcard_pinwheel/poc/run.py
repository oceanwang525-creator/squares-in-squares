"""Bounded search, packing emission, and dual certification for the pinwheel.

Run from anywhere:

    python3 poc/wildcard_pinwheel/poc/run.py

Outputs:
    data/pinwheel_best_n<n>.csv         best-overall member (min L, capacity>=n)
    data/pinwheel_tilted_n<n>.csv       best member with q != 0 (true tilt)
    data/results.json                   measured L strings, both certifications
    data/search.json                    the scanned family's frontier rows
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))

import pandas as pd  # noqa: E402

from squares.core import read_submission, certify, write_submission  # noqa: E402

sys.path.insert(0, str(HERE))
import pinwheel  # noqa: E402

TARGETS = {
    420: "20.96129301997122212179114584340315500626086004934217",
    650: "25.93296800914926226564497079296600645648140925914049",
    930: "30.91746912203587886923132169163779317923158816099717",
}
SLOPES = [(1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (1, 2),
          (1, 3), (2, 3), (4, 1), (1, 4), (5, 2), (2, 5)]
DILATION = 1.0 + 2e-6  # opens every unit-square contact so decimal transport is safe


def evaluate(a, b, p, q, d, family):
    """Return (L, total_capacity, blocks, cells) or None if the config is invalid."""
    theta = math.atan2(q, p)
    dx, dy = (d, d) if family == "corner" else (d, 0.0)
    blocks = pinwheel.make_blocks(a, b, theta, dx, dy)
    if not pinwheel.blocks_pairwise_ok(blocks):
        return None
    lo, hi = pinwheel.bbox(blocks)
    L = float(max(hi - lo))
    blocks = [(c - lo, t, aa, bb) for (c, t, aa, bb) in blocks]
    cells = pinwheel.fill_cells(blocks, L)
    return L, 4 * a * b + len(cells), blocks, cells


def search():
    rows = []
    for a in range(2, 13):
        for b in range(1, a):
            for (p, q) in SLOPES:
                for family in ("corner", "side"):
                    for d in np.linspace(0.3, (a + b) * 0.95, 30):
                        r = evaluate(a, b, p, q, float(d), family)
                        if r is None:
                            continue
                        L, total, _blocks, _cells = r
                        rows.append(dict(L=L, total=total, a=a, b=b,
                                         p=p, q=q, d=float(d), family=family))
    return rows


def choose(rows, n, tilted_only=False):
    pool = [r for r in rows if r["total"] >= n and 4 * r["a"] * r["b"] <= n
            and (r["q"] != 0 or not tilted_only)]
    if not pool:
        return None
    return min(pool, key=lambda r: r["L"])


def emit(a, b, p, q, d, family, n, path):
    built = pinwheel.build(a, b, p, q, d, family, n)
    if built is None:
        raise RuntimeError(f"build failed for {a},{b},{p},{q},{d},{family},n={n}")
    rows_xyz, L_float, info = built
    rows = []
    for i, (x, y, ang) in enumerate(rows_xyz):
        rows.append(dict(
            square_id=str(i),
            x=format(x * DILATION, ".50f"),
            y=format(y * DILATION, ".50f"),
            angle=format(ang, ".50f") if ang else "0",
        ))
    write_submission(path, {n: rows}, validate=True)
    return info


def certify_both(path, n):
    literal = read_submission(path)[n]
    l_literal = certify(n, literal)
    rt_text = pd.read_csv(path).to_csv(index=False)
    rt_path = path.with_name(path.stem + ".roundtrip.csv")
    rt_path.write_text(rt_text, encoding="utf-8")
    transported = read_submission(rt_path)[n]
    l_round = certify(n, transported)
    rt_path.unlink(missing_ok=True)
    return l_literal, l_round


def main():
    data = HERE.parent / "data"
    data.mkdir(parents=True, exist_ok=True)
    rows = search()
    summary = {"scanned_configs": len(rows), "targets": {}}

    results = {}
    for n in TARGETS:
        results[str(n)] = {}
        for tag, tilted in (("best", False), ("tilted", True)):
            cfg = choose(rows, n, tilted_only=tilted)
            if cfg is None:
                results[str(n)][tag] = dict(status="no-capacity-member")
                continue
            path = data / f"pinwheel_{tag}_n{n}.csv"
            info = emit(cfg["a"], cfg["b"], cfg["p"], cfg["q"],
                        cfg["d"], cfg["family"], n, path)
            l_lit, l_rt = certify_both(path, n)
            inc = float(TARGETS[n])
            contrib = float(l_lit) / math.sqrt(n) - 1.0
            inc_contrib = inc / math.sqrt(n) - 1.0
            results[str(n)][tag] = dict(
                config=dict(a=info["a"], b=info["b"], p=info["p"], q=info["q"],
                            d=info["d"], family=info["family"]),
                capacity=info["capacity"], L_measured=l_lit,
                L_roundtrip=l_rt, incumbent_L=TARGETS[n],
                L_over_incumbent=float(l_lit) - inc,
                score_contribution=contrib,
                incumbent_contribution=inc_contrib,
                delta=contrib - inc_contrib,
                csv=str(path.relative_to(ROOT)),
            )
            print(f"n={n} [{tag}] L={l_lit[:24]}... roundtrip={l_rt[:24]}... "
                  f"delta={contrib - inc_contrib:+.6f}")

    for n in TARGETS:
        capok = [r for r in rows if r["total"] >= n and 4 * r["a"] * r["b"] <= n]
        tilt = [r for r in capok if r["q"] != 0]
        summary["targets"][str(n)] = {
            "capacity_matched_configs": len(capok),
            "min_L_overall": min((r["L"] for r in capok), default=None),
            "min_L_tilted": min((r["L"] for r in tilt), default=None),
            "grid_L": float(math.ceil(math.sqrt(n))),
            "incumbent_L": float(TARGETS[n]),
        }

    (data / "search.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (data / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", data / "results.json")


if __name__ == "__main__":
    main()
