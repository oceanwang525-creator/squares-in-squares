"""Pell-17 Flank Perimeter Tooth Deletion POC (POC 1).

100% synthetic CPU construction for large-n square packings via parametric
tooth deletion on flank staircases of Pell-17 cores (D=17).
Evaluates exact-count matching at discrete count deficits (e.g. n = 908, 909, 543)
and executes the fast falsification gate defined in research/docs/q5_beyond_atlas_and_plan.md.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "data"
DOCS = Path(__file__).resolve().parents[1] / "docs"
REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/pell_corrugation/poc"))
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))

from squares import _official_metric as metric
from squares.core import certify
import pell_corrugation as pc

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2

# Targeted incumbents and portfolio baselines
PUBLIC_TARGET_INCUMBENTS = pc.PUBLIC_TARGET_INCUMBENTS
PORTFOLIO_BASELINE = pc.PORTFOLIO_BASELINE

# Falsification gate threshold from q5_beyond_atlas_and_plan.md
FALSIFICATION_INCUMBENT_909 = Decimal("30.65685424949258200000000000000000000000000000000000")


def normalize_tooth_spec(
    spec: Iterable[int | tuple[int, int]] | None,
    a: int,
) -> set[tuple[int, int]]:
    """Convert perimeter tooth indices k or coordinate tuples (i, j) into a coordinate set.

    In flank staircase a, perimeter teeth satisfy i + j = a - 1.
    Integer index k maps to (k, a - 1 - k) for 0 <= k < a.
    Specifically:
      k = a - 1 maps to (a - 1, 0) [outer SW perimeter tooth]
      k = 0 maps to (0, a - 1) [outer NE perimeter tooth]
    """
    if spec is None:
        return set()
    coords: set[tuple[int, int]] = set()
    for item in spec:
        if isinstance(item, (int, np.integer)):
            k = int(item)
            if not (0 <= k < a):
                raise ValueError(f"Perimeter tooth index {k} out of range [0, {a - 1}]")
            coords.add((k, a - 1 - k))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            i, j = int(item[0]), int(item[1])
            coords.add((i, j))
        else:
            raise TypeError(f"Invalid tooth specification: {item}")
    return coords


def build_staircase_band_tooth_deleted(
    w: int,
    l: int,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    deleted_teeth_a1: Iterable[int | tuple[int, int]] | None = None,
    deleted_teeth_a2: Iterable[int | tuple[int, int]] | None = None,
    drop_band: int = 0,
) -> tuple[np.ndarray, float]:
    """Construct an exact uncorrugated 45-degree staircase band with perimeter teeth deleted."""
    L0 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    del_a1 = normalize_tooth_spec(deleted_teeth_a1, a1)
    del_a2 = normalize_tooth_spec(deleted_teeth_a2, a2)

    q: list[list[float]] = []

    # SE flank staircase (a1)
    for i in range(a1):
        for j in range(a1 - i):
            if (i, j) in del_a1:
                continue
            if i == a1 - 1 and j == 0 and a1 + b1 > L0 + 1e-9:
                continue
            if i == 0 and j == a1 - 1 and a1 + b2 > L0 + 1e-9:
                continue
            q.append([L0 - 0.5 - i, 0.5 + j, 0.0])

    # NW flank staircase (a2)
    for i in range(a2):
        for j in range(a2 - i):
            if (i, j) in del_a2:
                continue
            if i == 0 and j == a2 - 1 and a2 + b1 > L0 + 1e-9:
                continue
            if i == a2 - 1 and j == 0 and a2 + b2 > L0 + 1e-9:
                continue
            q.append([0.5 + i, L0 - 0.5 - j, 0.0])

    # SW corner staircase (b1)
    for i in range(b1):
        for j in range(b1 - i):
            q.append([0.5 + i, 0.5 + j, 0.0])

    # NE corner staircase (b2)
    for i in range(b2):
        for j in range(b2 - i):
            q.append([L0 - 0.5 - i, L0 - 0.5 - j, 0.0])

    # Tilted 45-degree central band
    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    u_wall = 0.5 * (w - 1.0) + 0.5 * abs(a1 - a2) * INV_S2 + INV_S2
    u_lo = max((b1 + 1.0) * INV_S2 + 0.5, u_wall)
    u_hi = S2 * L0 - max((b2 + 1.0) * INV_S2 + 0.5, u_wall)
    u_cols = np.linspace(u_lo, u_hi, l) if l > 1 else np.array([0.5 * (u_lo + u_hi)])

    band: list[list[float]] = []
    for c in range(l):
        uc = u_cols[c]
        for s in range(w):
            v = v_start + s * 1.0
            x = (uc + v) * INV_S2
            y = (uc - v) * INV_S2
            band.append([x, y, math.pi / 4.0])
    if drop_band > 0:
        band = band[:-drop_band]
    q.extend(band)
    return np.array(q, dtype=float), L0


def build_corrugated_band_tooth_deleted(
    w: int,
    l: int,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    delta: float = 0.002935,
    deleted_teeth_a1: Iterable[int | tuple[int, int]] | None = None,
    deleted_teeth_a2: Iterable[int | tuple[int, int]] | None = None,
) -> tuple[np.ndarray, float]:
    """Construct an out-of-phase corrugated staircase band with flank teeth deleted."""
    L0 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    del_a1 = normalize_tooth_spec(deleted_teeth_a1, a1)
    del_a2 = normalize_tooth_spec(deleted_teeth_a2, a2)

    q: list[list[float]] = []
    gap_se_ne = max(0.0, (L0 - b2) - a1 - 0.05)
    gap_se_sw = max(0.0, (L0 - b1) - a1 - 0.05)

    for i in range(a1):
        for j in range(a1 - i):
            if (i, j) in del_a1:
                continue
            x, y = L0 - 0.5 - i, 0.5 + j * (1.0 + 1e-6)
            if i == a1 - 1 and j == 0 and gap_se_sw > 0:
                x -= min(0.19, 0.45 * gap_se_sw)
            if i == 0 and j == a1 - 1 and gap_se_ne > 0:
                x -= min(0.20, 0.45 * gap_se_ne)
            q.append([x, y, 0.0])

    for i in range(a2):
        for j in range(a2 - i):
            if (i, j) in del_a2:
                continue
            x, y = 0.5 + i * (1.0 + 1e-6), L0 - 0.5 - j
            q.append([x, y, 0.0])

    for i in range(b1):
        for j in range(b1 - i):
            q.append([0.5 + i * (1.0 + 1e-6), 0.5 + j, 0.0])

    for i in range(b2):
        for j in range(b2 - i):
            q.append([L0 - 0.5 - i * (1.0 + 1e-6), L0 - 0.5 - j, 0.0])

    u_teeth_se = np.array([(L0 + a1 - 1.0 - 2.0 * i) * INV_S2 for i in range(a1)])
    u_teeth_nw = np.array([(L0 - a2 + 1.0 + 2.0 * j) * INV_S2 for j in range(a2)])

    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    u_lo = (b1 + 1.0) * INV_S2 + 0.5
    u_hi = S2 * L0 - (b2 + 1.0) * INV_S2 - 0.5
    u_cols = np.linspace(u_lo, u_hi, l)

    du_se_arr = np.array([uc - u_teeth_se[np.argmin(np.abs(uc - u_teeth_se))] for uc in u_cols])
    du_nw_arr = np.array([uc - u_teeth_nw[np.argmin(np.abs(uc - u_teeth_nw))] for uc in u_cols])

    pinch_c: list[int] = []
    pinch_dth: list[float] = []
    for c in range(l):
        if max(abs(du_se_arr[c]), abs(du_nw_arr[c])) < 0.50:
            pinch_c.append(c)
            pinch_dth.append(-abs(delta) if du_se_arr[c] > 0 else +abs(delta))
    dth_cols = np.interp(np.arange(l), pinch_c, pinch_dth)

    for c in range(l):
        uc = u_cols[c]
        du_se = du_se_arr[c]
        du_nw = du_nw_arr[c]
        dth = dth_cols[c]

        v_shift = np.zeros(w)
        if c == 0 and a1 > a2:
            v_shift[:] = -0.15
        elif c == l - 1 and a1 > a2:
            v_shift[:] = -0.17
        elif abs(dth) < 0.5 * abs(delta):
            if abs(du_se) >= 0.50:
                v_shift[:] = +0.85 * (abs(du_se) - 0.50)
            elif abs(du_nw) >= 0.50:
                v_shift[:] = -0.85 * (abs(du_nw) - 0.50)
        else:
            if abs(du_nw) >= 0.50:
                v_shift[0] = -0.05 * (abs(du_nw) - 0.50)
            elif abs(du_se) >= 0.50:
                v_shift[-1] = +0.05 * (abs(du_se) - 0.50)

        for s in range(w):
            v = v_start + s * 1.0 + v_shift[s]
            u = uc + (s - 0.5 * (w - 1)) * math.sin(dth)
            x = (u + v) * INV_S2
            y = (u - v) * INV_S2
            q.append([x, y, math.pi / 4.0 + dth])

    return np.array(q, dtype=float), L0


def build_pell17_tooth_deleted_core(
    w: int,
    deleted_teeth_a1: Iterable[int | tuple[int, int]] | None = None,
    deleted_teeth_a2: Iterable[int | tuple[int, int]] | None = None,
    corrugated: bool = True,
    iterations: int = 35,
    clearance: float = 5e-9,
) -> tuple[pc.PolishResult, float, dict]:
    """Build and polish a Pell-17 core with flank perimeter teeth deleted."""
    b = int(math.floor(w * INV_S2))
    a1, a2 = b + 9, b + 8
    l = w + 12

    if corrugated:
        q_init, L0 = build_corrugated_band_tooth_deleted(
            w, l, a1, a2, b, b, delta=0.002935,
            deleted_teeth_a1=deleted_teeth_a1,
            deleted_teeth_a2=deleted_teeth_a2,
        )
    else:
        q_init, L0 = build_staircase_band_tooth_deleted(
            w, l, a1, a2, b, b,
            deleted_teeth_a1=deleted_teeth_a1,
            deleted_teeth_a2=deleted_teeth_a2,
        )

    res = pc.polish_band(q_init, L0, iterations=iterations, clearance=clearance)
    if res is None:
        raise RuntimeError(f"Tooth-deleted Pell-17 core polish failed for w={w}")

    del_a1 = normalize_tooth_spec(deleted_teeth_a1, a1)
    del_a2 = normalize_tooth_spec(deleted_teeth_a2, a2)
    meta = dict(
        family="pell17_tooth_deleted_core",
        w=w,
        l=l,
        a=[a1, a2],
        b=[b, b],
        D=17,
        corrugated=corrugated,
        deleted_teeth_a1=sorted(list(del_a1)),
        deleted_teeth_a2=sorted(list(del_a2)),
        teeth_deleted_count=len(del_a1) + len(del_a2),
        k0=int(math.floor(L0)),
        n_core=len(res.q),
        L0=L0,
        L_polished=res.L,
        gain_vs_L0=res.L - L0,
    )
    return res, L0, meta


def run_falsification_test(
    uncorrugated: bool = True,
    iterations: int = 10,
    clearance: float = 5e-9,
) -> dict:
    """Execute the exact Fast Falsification Test from q5_beyond_atlas_and_plan.md.

    1. Load uncorrugated (w=13, l=25, a1=18, a2=17, b=9) seed.
    2. Remove single outermost tooth (17, 0) from a1, yielding n_core = 738.
    3. Wrap with t=3 upright border (171 squares) to form exactly n = 909.
    4. Run 10 iterations of polish_band.
    5. Falsification Gate:
       - Certified L(909) >= 30.656854 (public incumbent)?
       - Does removing the tooth cause LP dilation (L > L0)?
    """
    w, l, a1, a2, b = 13, 25, 18, 17, 9
    deleted_teeth = [(17, 0)]  # Outermost SW tooth

    if uncorrugated:
        q_core_init, L0_core = build_staircase_band_tooth_deleted(
            w, l, a1, a2, b, b, deleted_teeth_a1=deleted_teeth
        )
    else:
        q_core_init, L0_core = build_corrugated_band_tooth_deleted(
            w, l, a1, a2, b, b, delta=0.002935, deleted_teeth_a1=deleted_teeth
        )

    # Core count must be exactly 738 (739 - 1)
    assert len(q_core_init) == 738, f"Expected core count 738, got {len(q_core_init)}"

    # Polish core
    res_core = pc.polish_band(q_core_init, L0_core, iterations=iterations, clearance=clearance)
    if res_core is None:
        raise RuntimeError("Core polish failed in falsification test")

    # Wrap with t=3 border (171 squares, k_total=30) -> total count exactly 909
    t = 3
    k_total = 30
    q_wrapped = pc.add_l_border(res_core.q, t=t, k_total=k_total, margin=1e-7)
    assert len(q_wrapped) == 909, f"Expected wrapped count 909, got {len(q_wrapped)}"

    L0_wrapped = L0_core + t
    res_wrapped = pc.polish_band(
        q_wrapped,
        pc.extent(q_wrapped),
        iterations=iterations,
        position_trust=0.05,
        angle_trust=0.005,
        clearance=clearance,
    )
    if res_wrapped is None:
        raise RuntimeError("Wrapped polish failed in falsification test")

    sat_ok, sat_pen = pc.sat_check(res_wrapped.q)
    cert_L = certify(909, pc.rows(res_wrapped.q, dilate=1.0))
    cert_L_dec = Decimal(str(cert_L))

    incumbent_dec = FALSIFICATION_INCUMBENT_909
    portfolio_baseline_dec = Decimal(PORTFOLIO_BASELINE[909])

    box_dilated = float(cert_L) > (L0_wrapped + 1e-6)
    beats_public = cert_L_dec < incumbent_dec
    beats_portfolio = cert_L_dec < portfolio_baseline_dec

    return dict(
        target_n=909,
        uncorrugated=uncorrugated,
        core_squares=len(res_core.q),
        border_squares=len(q_wrapped) - len(res_core.q),
        total_squares=len(q_wrapped),
        L0_wrapped=L0_wrapped,
        certified_L=str(cert_L),
        sat_ok=sat_ok,
        sat_penalty=sat_pen,
        box_dilated=box_dilated,
        falsification_threshold=str(incumbent_dec),
        beats_public_incumbent=beats_public,
        beats_portfolio_baseline=beats_portfolio,
        gap_to_public=str(cert_L_dec - incumbent_dec),
        gap_to_portfolio=str(cert_L_dec - portfolio_baseline_dec),
        falsified_against_public=not beats_public,
        compactness_preserved=not box_dilated,
    )


def generate_vacancy_suite(iterations: int = 20) -> tuple[dict[int, list[dict[str, str]]], dict]:
    """Generate tooth-deleted configurations across Pell-17 target neighbors."""
    puzzles: dict[int, list[dict[str, str]]] = {}
    records: dict[int, dict] = {}

    # Target 1: n = 909 (Pell-17 w=13, t=3, m=1 tooth deleted from a1)
    res_core_909, L0_core_909, _ = build_pell17_tooth_deleted_core(
        w=13, deleted_teeth_a1=[(17, 0)], corrugated=True, iterations=iterations
    )
    q_909 = pc.add_l_border(res_core_909.q, t=3, k_total=30, margin=1e-7)
    res_909 = pc.polish_band(q_909, pc.extent(q_909), iterations=iterations)
    assert res_909 is not None and len(res_909.q) == 909
    puzzles[909] = pc.rows(res_909.q, dilate=1.0)
    records[909] = dict(
        mechanism="tooth_deletion_m1",
        w=13, t=3, deleted_teeth=[(17, 0)],
        L_polished=res_909.L,
        count=len(res_909.q),
    )

    # Target 2: n = 908 (Pell-17 w=13, t=3, m=2 teeth deleted from a1: SW and NE)
    res_core_908, L0_core_908, _ = build_pell17_tooth_deleted_core(
        w=13, deleted_teeth_a1=[(17, 0), (0, 17)], corrugated=True, iterations=iterations
    )
    q_908 = pc.add_l_border(res_core_908.q, t=3, k_total=30, margin=1e-7)
    res_908 = pc.polish_band(q_908, pc.extent(q_908), iterations=iterations)
    assert res_908 is not None and len(res_908.q) == 908
    puzzles[908] = pc.rows(res_908.q, dilate=1.0)
    records[908] = dict(
        mechanism="tooth_deletion_m2",
        w=13, t=3, deleted_teeth=[(17, 0), (0, 17)],
        L_polished=res_908.L,
        count=len(res_908.q),
    )

    # Target 3: n = 543 (Pell-17 w=9, b=6, a1=15, t=2, m=1 tooth deleted: SW tooth (14, 0))
    res_core_543, L0_core_543, _ = build_pell17_tooth_deleted_core(
        w=9, deleted_teeth_a1=[(14, 0)], corrugated=True, iterations=iterations
    )
    q_543 = pc.add_l_border(res_core_543.q, t=2, k_total=23, margin=1e-7)
    res_543 = pc.polish_band(q_543, pc.extent(q_543), iterations=iterations)
    assert res_543 is not None and len(res_543.q) == 543
    puzzles[543] = pc.rows(res_543.q, dilate=1.0)
    records[543] = dict(
        mechanism="tooth_deletion_m1",
        w=9, t=2, deleted_teeth=[(14, 0)],
        L_polished=res_543.L,
        count=len(res_543.q),
    )

    # Target 4: n = 970 (Pell-17 w=13, t=4, m=1 tooth deleted: SW tooth (17, 0))
    res_core_970, L0_core_970, _ = build_pell17_tooth_deleted_core(
        w=13, deleted_teeth_a1=[(17, 0)], corrugated=True, iterations=iterations
    )
    q_970 = pc.add_l_border(res_core_970.q, t=4, k_total=31, margin=1e-7)
    res_970 = pc.polish_band(q_970, pc.extent(q_970), iterations=iterations)
    assert res_970 is not None and len(res_970.q) == 970
    puzzles[970] = pc.rows(res_970.q, dilate=1.0)
    records[970] = dict(
        mechanism="tooth_deletion_m1",
        w=13, t=4, deleted_teeth=[(17, 0)],
        L_polished=res_970.L,
        count=len(res_970.q),
    )

    return puzzles, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-falsification", action="store_true", help="Run the fast falsification gate")
    parser.add_argument("--generate", action="store_true", help="Generate and certify target suite")
    parser.add_argument("--iterations", type=int, default=20, help="LP polisher iterations")
    parser.add_argument("--out-dir", type=Path, default=HERE, help="Output directory for data")
    args = parser.parse_args()

    if args.test_falsification or not args.generate:
        print("=== FAST FALSIFICATION TEST (Pell-17 n=909 Tooth Deletion) ===")
        res_uncorr = run_falsification_test(uncorrugated=True, iterations=min(15, args.iterations))
        print("Uncorrugated Seed:")
        print(json.dumps(res_uncorr, indent=2))
        res_corr = run_falsification_test(uncorrugated=False, iterations=min(15, args.iterations))
        print("Corrugated Seed:")
        print(json.dumps(res_corr, indent=2))

    if args.generate:
        print(f"Generating vacancy deletion suite (iterations={args.iterations})...")
        puzzles, records = generate_vacancy_suite(iterations=args.iterations)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.out_dir / "output.csv"
        text = pc.csv_text(puzzles)
        csv_path.write_text(text)
        print(f"Saved {len(puzzles)} puzzles to {csv_path}")

        # Official transport certification
        checks = pc.certify_transport(text, args.out_dir / "output-pandas.csv")
        summary = dict(records=records, certification=checks)
        res_path = args.out_dir / "results.json"
        res_path.write_text(json.dumps(summary, indent=2))
        print(f"Certification summary saved to {res_path}")


if __name__ == "__main__":
    main()
