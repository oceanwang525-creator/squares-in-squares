"""Vacancy Deletion POC 2: Band-column end deletion and vacancy collapse.

Investigates whether deleting 1-2 squares directly from diagonal band columns
(outermost columns c=0 or c=l-1, or pinch columns where tooth offset is smallest)
on the certified n0=739 Pell-17 core allows adjacent columns and flank staircases
to collapse inward under 4-sign-piece epigraph LP polish, shrinking L below
incumbent side lengths for n=909 and n=971.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

# Set up module paths relative to repository root
REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
sys.path.insert(0, str(REPO / "poc/pell_corrugation/poc"))

import pell_corrugation as pc

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2

# Incumbents from poc/pell_corrugation/data/border-scan/merged.csv
INCUMBENTS = {
    908: "30.63702955961203500000000000000000000000000000000000",
    909: "30.65685424949258200000000000000000000000000000000000",
    910: "30.69092107418828100723564591975988495430653455035650",
    970: "31.65685424949258200000000000000000000000000000000000",
    971: "31.67877987534328000000000000000000000000000000000000",
    972: "31.69238815542838555961566082107349804168366869287818",
}


@dataclass(frozen=True)
class BandDeletionSpec:
    """Specification of squares to delete from diagonal band columns."""
    column: int
    square: int  # 0: negative v end (SE), w-1: positive v end (NW)
    label: str


def find_pinch_columns(
    w: int = 13,
    l: int = 25,
    a1: int = 18,
    a2: int = 17,
    b1: int = 9,
    b2: int = 9,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Identify pinch columns where flank tooth offset is minimal."""
    L0 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    u_teeth_se = np.array([(L0 + a1 - 1.0 - 2.0 * i) * INV_S2 for i in range(a1)])
    u_teeth_nw = np.array([(L0 - a2 + 1.0 + 2.0 * j) * INV_S2 for j in range(a2)])

    u_lo = (b1 + 1.0) * INV_S2 + 0.5
    u_hi = S2 * L0 - (b2 + 1.0) * INV_S2 - 0.5
    u_cols = np.linspace(u_lo, u_hi, l)

    du_se = np.array([uc - u_teeth_se[np.argmin(np.abs(uc - u_teeth_se))] for uc in u_cols])
    du_nw = np.array([uc - u_teeth_nw[np.argmin(np.abs(uc - u_teeth_nw))] for uc in u_cols])

    pinch_c = [c for c in range(l) if max(abs(du_se[c]), abs(du_nw[c])) < 0.50]
    return pinch_c, du_se, du_nw


def build_band_vacancy_core(
    w: int = 13,
    l: int = 25,
    a1: int = 18,
    a2: int = 17,
    b1: int = 9,
    b2: int = 9,
    deletions: list[BandDeletionSpec] | None = None,
    delta: float = 0.002935,
    collapse_shift: float = 0.25,
) -> tuple[np.ndarray, float, dict]:
    """Construct Pell-17 core with specified band-column squares deleted and collapsed.

    Geometry:
    - a1: SE flank staircase (171 squares)
    - a2: NW flank staircase (153 squares)
    - b1: SW end staircase (45 squares)
    - b2: NE end staircase (45 squares)
    - Band: l columns of w squares (325 squares), indices 414..738
    - Intact total = 739 squares.
    """
    if deletions is None:
        deletions = []

    L0 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    q: list[list[float]] = []
    gap_se_ne = max(0.0, (L0 - b2) - a1 - 0.05)
    gap_se_sw = max(0.0, (L0 - b1) - a1 - 0.05)

    # 1. SE Flank Staircase a1
    for i in range(a1):
        for j in range(a1 - i):
            x, y = L0 - 0.5 - i, 0.5 + j * (1.0 + 1e-6)
            if i == a1 - 1 and j == 0 and gap_se_sw > 0:
                x -= min(0.19, 0.45 * gap_se_sw)
            if i == 0 and j == a1 - 1 and gap_se_ne > 0:
                x -= min(0.20, 0.45 * gap_se_ne)
            q.append([x, y, 0.0])

    # 2. NW Flank Staircase a2
    for i in range(a2):
        for j in range(a2 - i):
            x, y = 0.5 + i * (1.0 + 1e-6), L0 - 0.5 - j
            q.append([x, y, 0.0])

    # 3. SW End Staircase b1
    for i in range(b1):
        for j in range(b1 - i):
            q.append([0.5 + i * (1.0 + 1e-6), 0.5 + j, 0.0])

    # 4. NE End Staircase b2
    for i in range(b2):
        for j in range(b2 - i):
            q.append([L0 - 0.5 - i * (1.0 + 1e-6), L0 - 0.5 - j, 0.0])

    # 5. Band Columns
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

    # Set of deleted (column, square) pairs
    del_set = {(d.column, d.square) for d in deletions}
    del_cols = {d.column for d in deletions}

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

        # Apply vacancy collapse initial shift along column c or adjacent columns
        if c in del_cols and collapse_shift > 0.0:
            deleted_s = [d.square for d in deletions if d.column == c]
            if 0 in deleted_s and (w - 1) not in deleted_s:
                # Square 0 deleted (SE cavity): shift remaining squares negatively in v
                v_shift -= collapse_shift
            elif (w - 1) in deleted_s and 0 not in deleted_s:
                # Square w-1 deleted (NW cavity): shift remaining squares positively in v
                v_shift += collapse_shift

        for s in range(w):
            if (c, s) in del_set:
                # Omit deleted square to create internal vacancy
                continue
            v = v_start + s * 1.0 + v_shift[s]
            u = uc + (s - 0.5 * (w - 1)) * math.sin(dth)
            x = (u + v) * INV_S2
            y = (u - v) * INV_S2
            q.append([x, y, math.pi / 4.0 + dth])

    meta = {
        "family": "band_vacancy_pell17",
        "w": w,
        "l": l,
        "n_core": len(q),
        "deletions": [d.label for d in deletions],
        "deleted_count": len(deletions),
        "L0": L0,
    }
    return np.array(q, dtype=float), L0, meta


def wrap_vacancy_core(
    q_core: np.ndarray,
    t: int,
    k_total: int,
    extra_squares: int = 0,
    margin: float = 1e-7,
) -> np.ndarray:
    """Wrap core with t-layer upright L-border on N and E edges, with optional extra squares."""
    q_norm = pc.normalize_box(q_core, margin=0.0)
    L_core = pc.extent(q_norm)
    L_total = L_core + t + 2.0 * margin
    k_core = k_total - t
    border: list[list[float]] = []

    # East border columns
    for i in range(t):
        for j in range(k_total):
            border.append([L_total - 0.5 - i * (1.0 + 1e-8), 0.5 + j * (1.0 + 1e-8), 0.0])

    # North border rows
    for i in range(k_core):
        for j in range(t):
            border.append([0.5 + i * (1.0 + 1e-8), L_total - 0.5 - j * (1.0 + 1e-8), 0.0])

    # Optional extra squares at unoccupied border positions
    if extra_squares > 0:
        for ex in range(extra_squares):
            # Place at top-left corner continuation
            border.append([0.5 + (k_core + ex) * (1.0 + 1e-8), L_total - 0.5, 0.0])

    return np.vstack([q_norm, np.array(border, dtype=float)])


def build_candidate_configurations() -> list[dict]:
    """Define vacancy deletion configurations across outermost and pinch columns."""
    pinch_c, _, _ = find_pinch_columns()
    p_min = pinch_c[len(pinch_c) // 2] if pinch_c else 12

    configs = [
        {
            "name": "outer_c0_s0_del1",
            "deletions": [BandDeletionSpec(column=0, square=0, label="c0_s0_SE_end")],
            "desc": "1 square deleted from outermost column c=0 at SE flank end (s=0)",
        },
        {
            "name": "outer_c0_s12_del1",
            "deletions": [BandDeletionSpec(column=0, square=12, label="c0_s12_NW_end")],
            "desc": "1 square deleted from outermost column c=0 at NW flank end (s=12)",
        },
        {
            "name": "outer_c24_s0_del1",
            "deletions": [BandDeletionSpec(column=24, square=0, label="c24_s0_SE_end")],
            "desc": "1 square deleted from outermost column c=24 at SE flank end (s=0)",
        },
        {
            "name": "outer_c24_s12_del1",
            "deletions": [BandDeletionSpec(column=24, square=12, label="c24_s12_NW_end")],
            "desc": "1 square deleted from outermost column c=24 at NW flank end (s=12)",
        },
        {
            "name": "outer_c0_both_ends_del2",
            "deletions": [
                BandDeletionSpec(column=0, square=0, label="c0_s0_SE_end"),
                BandDeletionSpec(column=0, square=12, label="c0_s12_NW_end"),
            ],
            "desc": "2 squares deleted from outermost column c=0 at both ends (s=0 and s=12)",
        },
        {
            "name": "pinch_min_s0_del1",
            "deletions": [BandDeletionSpec(column=p_min, square=0, label=f"c{p_min}_s0_pinch_SE")],
            "desc": f"1 square deleted from central pinch column c={p_min} at SE end (s=0)",
        },
        {
            "name": "pinch_min_s12_del1",
            "deletions": [BandDeletionSpec(column=p_min, square=12, label=f"c{p_min}_s12_pinch_NW")],
            "desc": f"1 square deleted from central pinch column c={p_min} at NW end (s=12)",
        },
    ]
    return configs


def evaluate_vacancy_poc(
    iterations: int = 25,
    clearance: float = 5e-9,
) -> dict:
    """Run full evaluation of POC 2 candidate configurations against incumbents."""
    configs = build_candidate_configurations()
    results = {}

    for cfg in configs:
        name = cfg["name"]
        deletions = cfg["deletions"]
        del_count = len(deletions)

        # 1. Build vacancy core
        q_core, L0, meta = build_band_vacancy_core(
            deletions=deletions,
            collapse_shift=0.25,
        )
        core_n = len(q_core)

        # Polish core
        res_core = pc.polish_band(q_core, L0, iterations=iterations, clearance=clearance)
        core_polished = res_core.q if res_core is not None else q_core
        L_core = pc.extent(core_polished)

        # Target evaluation for t=3 (n=909 for del=1, n=908 for del=2)
        target_t3 = 910 - del_count
        q_t3 = wrap_vacancy_core(core_polished, t=3, k_total=30)
        res_t3 = pc.polish_band(
            q_t3,
            pc.extent(q_t3),
            iterations=iterations,
            position_trust=0.05,
            angle_trust=0.005,
            clearance=clearance,
        )
        q_t3_final = res_t3.q if res_t3 is not None else q_t3
        L_t3 = pc.extent(q_t3_final)
        sat_t3, gap_t3 = pc.sat_check(q_t3_final)

        # Target evaluation for t=4 (n=971 for del=1 with extra border square, or n=970)
        q_t4 = wrap_vacancy_core(core_polished, t=4, k_total=31, extra_squares=(1 if del_count == 1 else 0))
        res_t4 = pc.polish_band(
            q_t4,
            pc.extent(q_t4),
            iterations=iterations,
            position_trust=0.05,
            angle_trust=0.005,
            clearance=clearance,
        )
        q_t4_final = res_t4.q if res_t4 is not None else q_t4
        L_t4 = pc.extent(q_t4_final)
        sat_t4, gap_t4 = pc.sat_check(q_t4_final)

        inc_t3 = float(INCUMBENTS.get(target_t3, 30.70710678))
        inc_971 = float(INCUMBENTS[971])

        results[name] = {
            "desc": cfg["desc"],
            "del_count": del_count,
            "core_n": core_n,
            "L_core": L_core,
            "t3": {
                "n": len(q_t3_final),
                "L": L_t3,
                "incumbent_L": inc_t3,
                "delta_L": L_t3 - inc_t3,
                "sat_ok": sat_t3,
                "min_gap": gap_t3,
                "beat_incumbent": (L_t3 < inc_t3),
            },
            "t4": {
                "n": len(q_t4_final),
                "L": L_t4,
                "incumbent_L": inc_971,
                "delta_L": L_t4 - inc_971,
                "sat_ok": sat_t4,
                "min_gap": gap_t4,
                "beat_incumbent": (L_t4 < inc_971),
            },
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Pell-17 Band Column Vacancy Deletion POC 2")
    parser.add_argument("--iterations", type=int, default=25, help="LP polish iterations")
    parser.add_argument("--summary", action="store_true", help="Print summary of vacancy POC results")
    args = parser.parse_args()

    print("Evaluating Vacancy Deletion POC 2 (Band-Column End Deletion)...", flush=True)
    results = evaluate_vacancy_poc(iterations=args.iterations)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
