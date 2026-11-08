#!/usr/bin/env python3
"""Export and certify n=907 and n=968 packings against merged.csv incumbents.

Generates candidate packings:
  - n=907: Symmetric odd-D corrugated core (D=51, w=3, iterations=40) polished with 4-sign-piece LP.
  - n=968: Dual-pocket composite packing from slope_band suite.
Exports standalone CSVs to poc/pell_corrugation/data/candidates/, certifies them on literal text
and pandas 3.1.0 round-trip, computes minimum SAT clearances, compares against merged.csv incumbents,
and emits certification_report.json.
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "poc/baseline/poc/src"))
sys.path.insert(0, str(REPO_ROOT / "poc/multi_pell_cores/1/poc"))
sys.path.insert(0, str(REPO_ROOT / "poc/slope_band/3/poc"))

from multi_pell_cores import build_symmetric_odd_d_core
from slope_band_3 import generate_slope_band_suite
from squares._official_metric import _parse_submission_csv
from squares.core import certify

MERGED_CSV = REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
CANDIDATES_DIR = REPO_ROOT / "poc/pell_corrugation/data/candidates"


def compute_min_sat_clearance(q: np.ndarray) -> float:
    """Rigorous separating axis theorem minimum clearance across all square pairs."""
    q_arr = np.asarray(q, dtype=np.longdouble)
    c, s = np.cos(q_arr[:, 2]), np.sin(q_arr[:, 2])
    u = np.column_stack((c, s))
    v = np.column_stack((-s, c))
    min_clearance = np.longdouble(np.inf)
    for i in range(len(q_arr) - 1):
        d = q_arr[i + 1 :, :2] - q_arr[i, :2]
        best = np.full(len(d), -np.inf, dtype=np.longdouble)
        for owner_axes in (False, True):
            for basis in (u, v):
                a = basis[i + 1 :] if owner_axes else np.broadcast_to(basis[i], d.shape)
                hi = 0.5 * (np.abs(a @ u[i]) + np.abs(a @ v[i]))
                hj = 0.5 * (np.abs(np.sum(a * u[i + 1 :], axis=1)) + np.abs(np.sum(a * v[i + 1 :], axis=1)))
                g = np.abs(np.sum(d * a, axis=1)) - hi - hj
                best = np.maximum(best, g)
        min_clearance = min(min_clearance, np.min(best))
    return float(min_clearance)


def get_incumbents(merged_path: Path, target_ns: set[int]) -> dict[int, str]:
    """Parse merged.csv and certify incumbents to get 50-decimal L values."""
    rows_by_n: dict[int, list[dict[str, str]]] = {n: [] for n in target_ns}
    with open(merged_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_val = int(row["n"])
            if n_val in target_ns:
                rows_by_n[n_val].append(row)

    incumbents = {}
    for n_val in target_ns:
        if not rows_by_n[n_val]:
            raise ValueError(f"Target n={n_val} not found in {merged_path}")
        l_cert = certify(n_val, rows_by_n[n_val])
        incumbents[n_val] = l_cert
    return incumbents


def export_candidate_csv(csv_path: Path, n: int, q: np.ndarray) -> None:
    """Export packing array to standard CSV with header id,n,square_id,x,y,angle."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for idx, (x, y, theta) in enumerate(q):
            writer.writerow([idx, n, idx, format(x, ".17g"), format(y, ".17g"), format(theta, ".17g")])


def certify_candidate(csv_path: Path, n: int) -> tuple[str, str, str, str]:
    """Certify CSV on literal text and on pandas read_csv/to_csv round-trip."""
    literal_text = csv_path.read_text(encoding="utf-8")
    lit_parsed = _parse_submission_csv(literal_text)
    try:
        lit_L = certify(n, lit_parsed[n])
        lit_status = "ok"
    except Exception as e:
        lit_L = ""
        lit_status = f"error: {e}"

    df = pd.read_csv(csv_path)
    pandas_text = df.to_csv(index=False)
    pan_parsed = _parse_submission_csv(pandas_text)
    try:
        pan_L = certify(n, pan_parsed[n])
        pan_status = "ok"
    except Exception as e:
        pan_L = ""
        pan_status = f"error: {e}"

    return lit_status, lit_L, pan_status, pan_L


def run() -> dict:
    started = time.time()
    assert pd.__version__ == "3.1.0", f"Expected pandas 3.1.0, got {pd.__version__}"

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching incumbents from merged.csv...")
    incumbents = get_incumbents(MERGED_CSV, {907, 968})
    for n, l_val in incumbents.items():
        print(f"  Incumbent n={n}: L={l_val}")

    # Generate n=907 candidate
    print("\nGenerating candidate n=907 via build_symmetric_odd_d_core(51, 3, iterations=40)...")
    res_907, L0_907, meta_907 = build_symmetric_odd_d_core(51, 3, iterations=40)
    q_907 = res_907.q
    assert len(q_907) == 907, f"Expected 907 squares, got {len(q_907)}"

    csv_907 = CANDIDATES_DIR / "candidate_907.csv"
    export_candidate_csv(csv_907, 907, q_907)
    print(f"  Exported candidate_907.csv ({len(q_907)} rows)")

    lit_status_907, lit_L_907, pan_status_907, pan_L_907 = certify_candidate(csv_907, 907)
    sat_gap_907 = compute_min_sat_clearance(q_907)
    with localcontext() as ctx:
        ctx.prec = 60
        delta_907 = str(Decimal(lit_L_907) - Decimal(incumbents[907]))
        beats_907 = Decimal(lit_L_907) < Decimal(incumbents[907])
    print(f"  Candidate n=907: L_lit={lit_L_907}, L_pan={pan_L_907}")
    print(f"  SAT clearance: {sat_gap_907:.10e}")
    print(f"  Delta vs incumbent: {delta_907} (beats={beats_907})")

    # Generate n=968 candidate
    print("\nGenerating candidate n=968 via generate_slope_band_suite()...")
    packings_sb, records_sb = generate_slope_band_suite()
    q_968 = packings_sb[968]
    assert len(q_968) == 968, f"Expected 968 squares, got {len(q_968)}"

    csv_968 = CANDIDATES_DIR / "candidate_968.csv"
    export_candidate_csv(csv_968, 968, q_968)
    print(f"  Exported candidate_968.csv ({len(q_968)} rows)")

    lit_status_968, lit_L_968, pan_status_968, pan_L_968 = certify_candidate(csv_968, 968)
    sat_gap_968 = compute_min_sat_clearance(q_968)
    with localcontext() as ctx:
        ctx.prec = 60
        delta_968 = str(Decimal(lit_L_968) - Decimal(incumbents[968]))
        beats_968 = Decimal(lit_L_968) < Decimal(incumbents[968])
    print(f"  Candidate n=968: L_lit={lit_L_968}, L_pan={pan_L_968}")
    print(f"  SAT clearance: {sat_gap_968:.10e}")
    print(f"  Delta vs incumbent: {delta_968} (beats={beats_968})")

    report = {
        "summary": {
            "pandas_version": pd.__version__,
            "elapsed_seconds": time.time() - started,
            "target_evaluations": [907, 968],
        },
        "907": {
            "n": 907,
            "candidate_L": lit_L_907,
            "incumbent_L": incumbents[907],
            "delta_L": delta_907,
            "beats_incumbent": beats_907,
            "sat_clearance": sat_gap_907,
            "literal_certification_status": lit_status_907,
            "pandas_3_1_0_roundtrip_certification_status": pan_status_907,
            "candidate_L_literal": lit_L_907,
            "candidate_L_pandas": pan_L_907,
            "initial_L0": L0_907,
            "lp_accepted_steps": res_907.accepted,
        },
        "968": {
            "n": 968,
            "candidate_L": lit_L_968,
            "incumbent_L": incumbents[968],
            "delta_L": delta_968,
            "beats_incumbent": beats_968,
            "sat_clearance": sat_gap_968,
            "literal_certification_status": lit_status_968,
            "pandas_3_1_0_roundtrip_certification_status": pan_status_968,
            "candidate_L_literal": lit_L_968,
            "candidate_L_pandas": pan_L_968,
            "slope_band_meta": records_sb.get(968, {}),
        },
    }

    report_path = CANDIDATES_DIR / "certification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWritten certification report to {report_path}")

    return report


if __name__ == "__main__":
    run()
