#!/usr/bin/env python3
"""Reachability audit for the remaining squares-in-squares score gap.

Reads the current per-n table (merged-literal-official.json) and the public23
merge (baseline-official.json), then writes the tables under ../docs/.

Conventions
-----------
k = floor(sqrt(n)), D = n - k*k  (the convention used by ARGON_BRIEF.md and
high_n_gap_ranking.md; the brief's "ceil" would make D <= 0 for every
non-square, so the named regimes would be empty).

r(n)      = L(n)/sqrt(n) - 1
baseline  = median of r(m) over m in [n-5, n+5], excluding n and squares
local_gap = r(n) - baseline
public_gap= max(0, r(n) - r_public(n))

All arithmetic that feeds a reported total uses Decimal.
"""

from __future__ import annotations

import csv
import json
import math
from decimal import Decimal, getcontext
from pathlib import Path
from statistics import median

getcontext().prec = 60

HERE = Path(__file__).resolve()
BASE = HERE.parents[1]
DOCS = BASE / "docs"
REPO = HERE.parents[3]
SCAN = REPO / "poc/pell_corrugation/data/border-scan"

CURRENT = SCAN / "merged-literal-official.json"
PUBLIC = SCAN / "baseline-official.json"
GALLERY = SCAN / "gallery-repair-provenance.json"

EXPECTED_SCORE = Decimal("22.216763458415837")
TARGET_SCORE = Decimal("22.19")
CORRIDORS = (739, 794, 851, 910)
PLATEAU_MIN_RUN = 7


def dsqrt(n: int) -> Decimal:
    return Decimal(n).sqrt()


def is_square(n: int) -> bool:
    return math.isqrt(n) ** 2 == n


def load_rows(path: Path) -> list[dict]:
    doc = json.loads(path.read_text())
    return doc["results"]


def plateau_runs(L: dict[int, Decimal], tol: Decimal) -> dict[int, int]:
    """n -> length of the constant-L run containing n (if >= PLATEAU_MIN_RUN)."""
    run_len: dict[int, int] = {}
    start = 1
    for n in range(2, 1002):
        if n == 1001 or abs(L[n] - L[start]) > tol:
            length = n - start
            if length >= PLATEAU_MIN_RUN:
                for m in range(start, n):
                    run_len[m] = length
            start = n
    return run_len


def regime_of(n: int, j: int, k: int, run_len: dict[int, int]) -> str:
    if is_square(n):
        return "perfect square"
    if any(abs(n - c) <= 3 for c in CORRIDORS):
        return "corrugation window"
    if j == 1:
        return "k^2+1 boundary"
    if k - 5 <= j <= k:
        return "D near k"
    if n in run_len:
        return "heptagonal plateau"
    return "general interior"


def mechanism_for(regime: str, n: int, j: int, k: int) -> str:
    if regime == "k^2+1 boundary":
        return "rational-slope (atan 1/2) or split-angle corner pocket across the D=1 step"
    if regime == "corrugation window":
        return "tooth deletion / vacancy propagation on the Pell-17 flank"
    if regime == "D near k":
        return "vacancy deletion plus multi-angle cluster; relax the overfilled top of the interval"
    if regime == "heptagonal plateau":
        return "break the grid ceiling with an off-center thick band"
    if regime == "perfect square":
        return "none (r = 0 by definition)"
    if j <= 8:
        return "off-center thick band with split angles and a corner gadget"
    if j <= k:
        return "multi-angle cluster with independent sub-bands, then LP polish"
    return "multi-block composite / defect-vacancy relaxation"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)

    cur = {x["n"]: x for x in load_rows(CURRENT)}
    pub = {x["n"]: x for x in load_rows(PUBLIC)}

    L = {n: Decimal(str(cur[n]["L"])) for n in cur}
    Lpub = {n: Decimal(str(pub[n]["L"])) for n in pub}
    r = {n: L[n] / dsqrt(n) - 1 for n in L}
    rpub = {n: Lpub[n] / dsqrt(n) - 1 for n in Lpub}

    total = sum(r[n] for n in range(1, 1001))
    total_json = sum(Decimal(str(cur[n]["score"])) for n in range(1, 1001))
    gap_to_target = total - TARGET_SCORE

    baseline: dict[int, Decimal] = {}
    for n in range(1, 1001):
        vals = sorted(
            r[m]
            for m in range(max(1, n - 5), min(1000, n + 5) + 1)
            if m != n and not is_square(m)
        )
        if vals:
            half = len(vals) // 2
            baseline[n] = vals[half] if len(vals) % 2 else (vals[half - 1] + vals[half]) / 2

    local_gap = {n: r[n] - baseline[n] for n in baseline}
    pos_gap = {n: max(Decimal(0), local_gap[n]) for n in local_gap}
    public_gap = {n: max(Decimal(0), r[n] - rpub[n]) for n in range(1, 1001)}

    runs = plateau_runs(L, Decimal("1e-9"))

    rows = []
    for n in range(1, 1001):
        k = math.isqrt(n)
        j = n - k * k
        k_ceil = math.ceil(math.sqrt(n))
        regime = regime_of(n, j, k, runs)
        if L[n] < Lpub[n]:
            status = "public worse"
        elif L[n] == Lpub[n]:
            status = "tied"
        else:
            status = "public better"
        rows.append(
            {
                "n": n,
                "k": k,
                "D": j,
                "k_ceil": k_ceil,
                "D_ceil": n - k_ceil * k_ceil,
                "is_square": int(is_square(n)),
                "regime": regime,
                "plateau_run": runs.get(n, 0),
                "L": L[n],
                "r": r[n],
                "local_baseline": baseline.get(n),
                "local_gap": local_gap.get(n),
                "abs_local_gap": abs(local_gap[n]) if n in local_gap else None,
                "positive_local_gap": pos_gap.get(n, Decimal(0)),
                "L_public": Lpub[n],
                "r_public": rpub[n],
                "public_gap": public_gap[n],
                "public_status": status,
            }
        )

    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, Decimal):
            return format(x, "f")
        return x

    fieldnames = list(rows[0].keys())
    with (DOCS / "per_n_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(v) for k, v in row.items()})

    def write_ranked(name, key, items, limit=100):
        with (DOCS / name).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "rank",
                    "n",
                    "k",
                    "D",
                    "regime",
                    "L",
                    "r",
                    "local_baseline",
                    "local_gap",
                    "abs_local_gap",
                    "public_gap",
                    "public_status",
                    "suggested_mechanism",
                ]
            )
            for rank, (n, val) in enumerate(items[:limit], 1):
                row = rows[n - 1]
                w.writerow(
                    [
                        rank,
                        n,
                        row["k"],
                        row["D"],
                        row["regime"],
                        fmt(row["L"]),
                        fmt(row["r"]),
                        fmt(row["local_baseline"]),
                        fmt(row["local_gap"]),
                        fmt(row["abs_local_gap"]),
                        fmt(row["public_gap"]),
                        row["public_status"],
                        mechanism_for(row["regime"], n, row["D"], row["k"]),
                    ]
                )

    abs_rank = sorted(local_gap.items(), key=lambda kv: -abs(kv[1]))
    pos_rank = sorted(pos_gap.items(), key=lambda kv: -kv[1])
    pub_rank = sorted(public_gap.items(), key=lambda kv: -kv[1])
    write_ranked("local_gap_abs_top.csv", None, abs_rank)
    write_ranked("local_gap_positive_top.csv", None, pos_rank)
    write_ranked("public_gap_top.csv", None, pub_rank)

    # Actionable list: n >= 100, ranked by positive local gap.
    actionable = sorted(
        ((n, pos_gap[n]) for n in range(100, 1001) if not is_square(n)),
        key=lambda kv: -kv[1],
    )
    with (DOCS / "targets_top30.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rank",
                "n",
                "k",
                "D",
                "regime",
                "L",
                "r",
                "local_baseline",
                "local_gap",
                "public_gap",
                "public_status",
                "suggested_mechanism",
            ]
        )
        for rank, (n, g) in enumerate(actionable[:30], 1):
            row = rows[n - 1]
            w.writerow(
                [
                    rank,
                    n,
                    row["k"],
                    row["D"],
                    row["regime"],
                    fmt(row["L"]),
                    fmt(row["r"]),
                    fmt(row["local_baseline"]),
                    fmt(row["local_gap"]),
                    fmt(row["public_gap"]),
                    row["public_status"],
                    mechanism_for(row["regime"], n, row["D"], row["k"]),
                ]
            )

    # Regime aggregation.
    regimes = {}
    for row in rows:
        agg = regimes.setdefault(
            row["regime"],
            {"count": 0, "signed": Decimal(0), "positive": Decimal(0), "public": Decimal(0), "gaps": []},
        )
        agg["count"] += 1
        agg["signed"] += row["local_gap"] if row["local_gap"] is not None else Decimal(0)
        agg["positive"] += row["positive_local_gap"]
        agg["public"] += row["public_gap"]
        if row["local_gap"] is not None:
            agg["gaps"].append(row["positive_local_gap"])
    with (DOCS / "regime_breakdown.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "regime",
                "count",
                "signed_local_gap_sum",
                "positive_local_gap_sum",
                "public_gap_sum",
                "median_positive_local_gap",
                "max_positive_local_gap",
                "top10_positive_local_gap_sum",
            ]
        )
        for name in sorted(regimes, key=lambda k: -regimes[k]["positive"]):
            agg = regimes[name]
            gaps = sorted(agg["gaps"], reverse=True)
            w.writerow(
                [
                    name,
                    agg["count"],
                    fmt(agg["signed"]),
                    fmt(agg["positive"]),
                    fmt(agg["public"]),
                    fmt(median(gaps)),
                    fmt(gaps[0]),
                    fmt(sum(gaps[:10])),
                ]
            )

    all_pos = sorted(pos_gap.values(), reverse=True)
    by_threshold = {}
    for threshold in (1, 100, 500, 600):
        vals = sorted((pos_gap[n] for n in range(threshold, 1001)), reverse=True)
        by_threshold[str(threshold)] = {
            "count": len(vals),
            "sum": str(sum(vals)),
            "top": {str(N): str(sum(vals[:N])) for N in (10, 20, 30, 50, 100)},
        }

    gallery = json.loads(GALLERY.read_text())
    wins = {int(k): v for k, v in gallery["wins"].items()}
    gallery_current = sum(r[n] for n in wins)
    gallery_old = sum(Decimal(str(w["old_L"])) / dsqrt(n) - 1 for n, w in wins.items())
    gallery_box = sum(Decimal(str(w["box_growth"])) / dsqrt(n) for n, w in wins.items())
    gallery_gain = sum(Decimal(str(w["gain_score"])) for w in wins.values())

    public_better = sum(1 for row in rows if row["public_status"] == "public better")
    tied = sum(1 for row in rows if row["public_status"] == "tied")
    public_worse = sum(1 for row in rows if row["public_status"] == "public worse")

    summary = {
        "current_file": str(CURRENT.relative_to(REPO)),
        "public_file": str(PUBLIC.relative_to(REPO)),
        "total_from_L_strings": str(total),
        "total_from_per_n_scores": str(total_json),
        "expected_total": str(EXPECTED_SCORE),
        "delta_from_expected": str(total - EXPECTED_SCORE),
        "target": str(TARGET_SCORE),
        "gap_to_target": str(gap_to_target),
        "local_gap_signed_sum": str(sum(local_gap.values())),
        "local_gap_positive_sum": str(sum(pos_gap.values())),
        "local_gap_positive_count": sum(1 for g in pos_gap.values() if g > 0),
        "public_gap_sum": str(sum(public_gap.values())),
        "public_positive_count": sum(1 for g in public_gap.values() if g > 0),
        "public_status_counts": {
            "public_worse": public_worse,
            "tied": tied,
            "public_better": public_better,
        },
        "top_positive_local_gap_sums": {
            "all": {str(N): str(sum(all_pos[:N])) for N in (10, 20, 50, 100)},
            "n_ge_100": by_threshold["100"]["top"],
            "n_ge_600": by_threshold["600"]["top"],
        },
        "by_n_threshold": by_threshold,
        "regimes": {
            name: {
                "count": agg["count"],
                "signed": str(agg["signed"]),
                "positive": str(agg["positive"]),
                "public": str(agg["public"]),
            }
            for name, agg in regimes.items()
        },
        "gallery_repair": {
            "wins": len(wins),
            "current_contribution": str(gallery_current),
            "contribution_at_old_L": str(gallery_old),
            "banked_gain_score": str(gallery_gain),
            "box_growth_over_sqrt_sum": str(gallery_box),
        },
    }
    (DOCS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Human-readable full tables.
    def md_rows(items, limit, include_sign=True):
        out = [
            "| rank | n | k | D | regime | L | r | baseline | local gap | public gap | public status | mechanism |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for rank, (n, _) in enumerate(items[:limit], 1):
            row = rows[n - 1]
            out.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    rank,
                    n,
                    row["k"],
                    row["D"],
                    row["regime"],
                    fmt(row["L"]),
                    fmt(row["r"]),
                    fmt(row["local_baseline"]),
                    fmt(row["local_gap"]),
                    fmt(row["public_gap"]),
                    row["public_status"],
                    mechanism_for(row["regime"], n, row["D"], row["k"]),
                )
            )
        return "\n".join(out)

    lines = [
        "# Gap reachability: full tables",
        "",
        "Generated by `poc/gap_reachability/poc/analyze_gap.py`. All totals are Decimal sums of the L strings.",
        "",
        "## Headline",
        "",
        f"- Current total from L strings: `{total}`",
        f"- Expected: `{EXPECTED_SCORE}` (delta `{total - EXPECTED_SCORE}`)",
        f"- Gap to `{TARGET_SCORE}`: `{gap_to_target}`",
        f"- Local-gap signed sum: `{sum(local_gap.values())}`",
        f"- Positive local-gap sum: `{sum(pos_gap.values())}`",
        f"- Public-gap sum: `{sum(public_gap.values())}` (positive n: {sum(1 for g in public_gap.values() if g > 0)})",
        f"- Public status: {public_worse} n where public is worse, {tied} tied, {public_better} public better",
        "",
        "## Top positive local gaps, all n",
        "",
        md_rows(pos_rank, 40),
        "",
        "## Top absolute local gaps, all n",
        "",
        md_rows(abs_rank, 40),
        "",
        "## Top 30 actionable targets (n >= 100, non-square)",
        "",
        md_rows(actionable, 30),
        "",
        "## Per-regime breakdown",
        "",
        "| regime | count | signed sum | positive sum | public sum | median positive | max positive | top-10 positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(regimes, key=lambda k: -regimes[k]["positive"]):
        agg = regimes[name]
        gaps = sorted(agg["gaps"], reverse=True)
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                agg["count"],
                fmt(agg["signed"]),
                fmt(agg["positive"]),
                fmt(agg["public"]),
                fmt(median(gaps)),
                fmt(gaps[0]),
                fmt(sum(gaps[:10])),
            )
        )
    lines += [
        "",
        "## Top-k closure totals",
        "",
        "| cohort | top 10 | top 20 | top 50 | top 100 | all positive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, th in (("all n", 1), ("n >= 100", 100), ("n >= 600", 600)):
        vals = sorted((pos_gap[n] for n in range(th, 1001)), reverse=True)
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                label,
                fmt(sum(vals[:10])),
                fmt(sum(vals[:20])),
                fmt(sum(vals[:50])),
                fmt(sum(vals[:100])),
                fmt(sum(vals)),
            )
        )
    lines += [
        "",
        "## Gallery repair residual",
        "",
        f"- {len(wins)} repaired n; current contribution `{gallery_current}`",
        f"- Sum of `box_growth/sqrt(n)`: `{gallery_box}`",
        f"- Banked `gain_score`: `{gallery_gain}`",
        "",
    ]
    (DOCS / "full_tables.md").write_text("\n".join(lines))

    print(f"total={total}")
    print(f"expected={EXPECTED_SCORE} delta={total - EXPECTED_SCORE}")
    print(f"gap_to_target={gap_to_target}")
    print(f"positive_local_gap={sum(pos_gap.values())}")
    print(f"public_gap={sum(public_gap.values())}")
    print(f"gallery_box_growth_over_sqrt={gallery_box}")
    print(f"wrote {DOCS}")


if __name__ == "__main__":
    main()
