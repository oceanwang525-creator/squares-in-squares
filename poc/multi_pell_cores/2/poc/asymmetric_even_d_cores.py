"""Asymmetric Even-D Corrugated Band Cores (D=10, 34, 58) + Upright L-Borders.

Implements POC 2 of the Multi-Pell Corrugated Core architecture:
Asymmetric even-D out-of-phase pinch-column corrugated cores where:
  - b1 = b = floor(w / sqrt(2)), b2 = b + 1  => b1 + b2 = 2b + 1 (odd)
  - a1 = b + D//2 + 1, a2 = b + D//2          => a1 + a2 = D + 2b + 1 (odd)
  - Out-of-phase tooth parity: (a1 + a2) = 1 (mod 2)
  - Longitudinal span l = w + floor(D / sqrt(2))
  - Longitudinal slack: slack(D) = D/sqrt(2) - floor(D/sqrt(2))
  - Container wall clearance condition: {w/sqrt(2)} > 0.50 (in (0.55, 0.95))
    guarantees positive right-wall gap: L45 - b2 - a1 = {w/sqrt(2)} - 0.5 > 0.

Polished with the 4-sign-piece epigraph LP polisher and certified on literal CSV
and pandas 3.1.0 transport against official and portfolio incumbents.
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
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DOCS = HERE.parent / "docs"
REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO / "poc/pell_corrugation/poc"))
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))

import pell_corrugation as pc
from squares import _official_metric as metric
from squares.core import certify

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2

# Even D configurations under study
EVEN_D_CONFIGS = {
    10: {
        "D": 10,
        "floor_d_s2": 7,
        "slack": 10.0 * INV_S2 - 7.0,  # ~0.071068
        "default_delta": 0.0030,
        "name": "Diophantine-10",
        "description": "10^2 - 2*7^2 = +2",
    },
    34: {
        "D": 34,
        "floor_d_s2": 24,
        "slack": 34.0 * INV_S2 - 24.0,  # ~0.041631
        "default_delta": 0.0028,
        "name": "Double-Pell-34",
        "description": "2 x 17 Pell order",
    },
    58: {
        "D": 58,
        "floor_d_s2": 41,
        "slack": 58.0 * INV_S2 - 41.0,  # ~0.012193
        "default_delta": 0.0010,
        "name": "Diophantine-58",
        "description": "58^2 - 2*41^2 = +2",
    },
}

# Targeted cores and parameters from research brief
TARGET_SPECS = [
    {"D": 10, "w": 4, "t": 0, "desc": "D=10, w=4, core n0=117"},
    {"D": 10, "w": 11, "t": 0, "desc": "D=10, w=11, core n0=431"},
    {"D": 10, "w": 11, "t": 8, "desc": "D=10, w=11 wrapped with t=8 (n=831)"},
    {"D": 34, "w": 4, "t": 0, "desc": "D=34, w=4, core n0=521"},
    {"D": 34, "w": 8, "t": 0, "desc": "D=34, w=8, core n0=821"},
    {"D": 34, "w": 8, "t": 1, "desc": "D=34, w=8 wrapped with t=1 (n=880)"},
    {"D": 58, "w": 1, "t": 0, "desc": "D=58, w=1, core n0=943"},
]


def frac_part_s2(w: int) -> float:
    """Return {w / sqrt(2)} in [0, 1)."""
    val = w * INV_S2
    return float(val - math.floor(val))


def check_asymmetric_clearance(w: int) -> float:
    """Evaluate right wall clearance gap: {w/sqrt(2)} - 0.5.

    Returns the gap in container units. Must be strictly positive (> 0) to avoid
    collision between the NE end staircase (b2 = b+1) and SE flank staircase (a1).
    """
    frac = frac_part_s2(w)
    return frac - 0.5


def get_asymmetric_even_d_parameters(D: int, w: int) -> dict[str, Any]:
    """Compute exact discrete parameters for an asymmetric even-D corrugated core."""
    if D not in EVEN_D_CONFIGS:
        raise ValueError(f"D={D} is not supported. Supported even D: {list(EVEN_D_CONFIGS.keys())}")
    if D % 2 != 0:
        raise ValueError(f"D must be even for asymmetric cores, got D={D}")

    frac = frac_part_s2(w)
    right_wall_gap = frac - 0.5
    if right_wall_gap <= 0:
        raise ValueError(
            f"Width w={w} violates asymmetric right-wall clearance: "
            f"{{w/sqrt(2)}} = {frac:.5f} <= 0.50 (gap={right_wall_gap:.5f} <= 0). "
            f"Requires {{w/sqrt(2)}} in (0.55, 0.95)."
        )

    b = int(math.floor(w * INV_S2))
    b1 = b
    b2 = b + 1
    a1 = b + (D // 2) + 1
    a2 = b + (D // 2)

    cfg = EVEN_D_CONFIGS[D]
    l = w + cfg["floor_d_s2"]
    slack = cfg["slack"]

    # Verify Diophantine invariants
    assert (a1 + a2) - (b1 + b2) == D
    assert (a1 + a2) % 2 == 1, "Out-of-phase condition (a1+a2 odd) violated"
    assert (b1 + b2) % 2 == 1, "Asymmetric end parity (b1+b2 odd) violated"

    L45 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    k0 = int(math.floor(L45))

    # Core square count: T(a1) + T(a2) + T(b1) + T(b2) + w*l
    n_a1 = a1 * (a1 + 1) // 2
    n_a2 = a2 * (a2 + 1) // 2
    n_b1 = b1 * (b1 + 1) // 2
    n_b2 = b2 * (b2 + 1) // 2
    n_band = w * l
    n0 = n_a1 + n_a2 + n_b1 + n_b2 + n_band

    return dict(
        D=D,
        w=w,
        l=l,
        b1=b1,
        b2=b2,
        b=[b1, b2],
        a1=a1,
        a2=a2,
        a=[a1, a2],
        L45=L45,
        k0=k0,
        n0=n0,
        n_staircases=n_a1 + n_a2 + n_b1 + n_b2,
        n_band=n_band,
        frac_w_s2=frac,
        right_wall_gap=right_wall_gap,
        slack=slack,
        default_delta=cfg["default_delta"],
    )


def build_asymmetric_corrugated_band(
    D: int,
    w: int,
    delta: float | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Construct an asymmetric out-of-phase corrugated staircase band packing."""
    params = get_asymmetric_even_d_parameters(D, w)
    l = params["l"]
    a1, a2 = params["a1"], params["a2"]
    b1, b2 = params["b1"], params["b2"]
    L0 = params["L45"]

    if delta is None:
        delta = params["default_delta"]

    q: list[list[float]] = []

    # Container wall clearances for outermost tooth adjustment
    gap_se_ne = max(0.0, (L0 - b2) - a1 - 0.05)
    gap_se_sw = max(0.0, (L0 - b1) - a1 - 0.05)

    # 1. SE flank staircase (a1 steps) at container (L0, 0)
    for i in range(a1):
        for j in range(a1 - i):
            x, y = L0 - 0.5 - i, 0.5 + j * (1.0 + 1e-6)
            if i == a1 - 1 and j == 0 and gap_se_sw > 0:
                x -= min(0.19, 0.45 * gap_se_sw)
            if i == 0 and j == a1 - 1 and gap_se_ne > 0:
                x -= min(0.20, 0.45 * gap_se_ne)
            q.append([x, y, 0.0])

    # 2. NW flank staircase (a2 steps) at container (0, L0)
    for i in range(a2):
        for j in range(a2 - i):
            x, y = 0.5 + i * (1.0 + 1e-6), L0 - 0.5 - j
            q.append([x, y, 0.0])

    # 3. SW end staircase (b1 steps) at (0, 0)
    for i in range(b1):
        for j in range(b1 - i):
            q.append([0.5 + i * (1.0 + 1e-6), 0.5 + j, 0.0])

    # 4. NE end staircase (b2 steps) at (L0, L0)
    for i in range(b2):
        for j in range(b2 - i):
            q.append([L0 - 0.5 - i * (1.0 + 1e-6), L0 - 0.5 - j, 0.0])

    # 5. Inward flank tooth positions along diagonal coordinate u = (x+y)/sqrt(2)
    u_teeth_se = np.array([(L0 + a1 - 1.0 - 2.0 * i) * INV_S2 for i in range(a1)])
    u_teeth_nw = np.array([(L0 - a2 + 1.0 + 2.0 * j) * INV_S2 for j in range(a2)])

    # 6. Diagonal band longitudinal and transverse ranges
    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    u_lo = (b1 + 1.0) * INV_S2 + 0.5
    u_hi = S2 * L0 - (b2 + 1.0) * INV_S2 - 0.5

    if l > 1:
        u_cols = np.linspace(u_lo, u_hi, l)
    else:
        u_cols = np.array([0.5 * (u_lo + u_hi)])

    du_se_arr = np.array([uc - u_teeth_se[np.argmin(np.abs(uc - u_teeth_se))] for uc in u_cols])
    du_nw_arr = np.array([uc - u_teeth_nw[np.argmin(np.abs(uc - u_teeth_nw))] for uc in u_cols])

    # 7. Identify pinch columns and assign alternating corrugation angle delta_c
    pinch_c: list[int] = []
    pinch_dth: list[float] = []
    for c in range(l):
        if max(abs(du_se_arr[c]), abs(du_nw_arr[c])) < 0.50:
            pinch_c.append(c)
            pinch_dth.append(-abs(delta) if du_se_arr[c] > 0 else +abs(delta))

    if len(pinch_c) >= 2:
        dth_cols = np.interp(np.arange(l), pinch_c, pinch_dth)
    elif len(pinch_c) == 1:
        dth_cols = np.full(l, pinch_dth[0])
    else:
        dth_cols = np.zeros(l)

    # 8. Column transverse positioning and square placement
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

    q_arr = np.array(q, dtype=float)
    assert len(q_arr) == params["n0"], f"Expected {params['n0']} squares, got {len(q_arr)}"
    params["num_pinch_columns"] = len(pinch_c)
    params["applied_delta"] = delta
    return q_arr, L0, params


def build_asymmetric_even_d_core(
    D: int,
    w: int,
    iterations: int = 35,
    clearance: float = 5e-9,
    delta: float | None = None,
) -> tuple[pc.PolishResult, float, dict[str, Any]]:
    """Build and polish an asymmetric even-D corrugated core."""
    q_init, L0, params = build_asymmetric_corrugated_band(D, w, delta=delta)
    res = pc.polish_band(q_init, L0, iterations=iterations, clearance=clearance)
    if res is None:
        raise RuntimeError(f"Asymmetric core polish failed for D={D}, w={w}")

    gain = res.L - L0
    meta = dict(
        family=f"asymmetric_even_d_corrugated_band",
        D=D,
        w=w,
        l=params["l"],
        a=params["a"],
        b=params["b"],
        k0=params["k0"],
        n0=len(res.q),
        L45=L0,
        L_corrugated=res.L,
        corrugation_gain=gain,
        right_wall_gap=params["right_wall_gap"],
        slack=params["slack"],
        num_pinch_columns=params["num_pinch_columns"],
        accepted_iterations=res.accepted,
    )
    return res, L0, meta


def load_incumbents() -> dict[int, str]:
    """Load reference incumbent side lengths from border-scan official results."""
    official_file = REPO / "poc/pell_corrugation/data/border-scan/merged-literal-official.json"
    if official_file.exists():
        data = json.loads(official_file.read_text())
        return {r["n"]: r["L"] for r in data["results"]}
    return {}


def evaluate_candidate(
    q: np.ndarray,
    n: int,
    meta: dict[str, Any],
    incumbents: dict[int, str],
) -> dict[str, Any]:
    """Certify candidate on literal and pandas 3.1.0 and evaluate vs incumbent."""
    q_norm = pc.normalize_box(q, margin=0.0)
    puzzles = {n: pc.rows(q_norm, dilate=1.0)}
    text = pc.csv_text(puzzles)
    checks = pc.certify_transport(text)
    chk = checks[str(n)]

    entry: dict[str, Any] = {
        "n": n,
        "meta": meta,
        "literal_ok": chk["literal"]["ok"],
        "pandas_ok": chk["actual_pandas"]["ok"],
    }
    if chk["literal"]["ok"]:
        entry["L_literal"] = chk["literal"]["L"]
        entry["L_pandas"] = chk["actual_pandas"]["L"]
        L_lit = Decimal(chk["literal"]["L"])

        inc_str = incumbents.get(n)
        if inc_str is not None:
            inc_dec = Decimal(inc_str)
            delta = L_lit - inc_dec
            entry["incumbent_L"] = inc_str
            entry["delta_vs_incumbent"] = str(delta)
            entry["improves_incumbent"] = bool(delta < Decimal("-1e-8"))
        else:
            entry["incumbent_L"] = None
            entry["delta_vs_incumbent"] = None
            entry["improves_incumbent"] = False

    return entry


def run_all_assigned_evaluations(
    iterations: int = 30,
    clearance: float = 5e-9,
) -> dict[str, Any]:
    """Run all assigned configurations (D=10, 34, 58) and document gate results."""
    incumbents = load_incumbents()
    results: list[dict[str, Any]] = []

    # 1. Core sweeps and evaluations
    cores_cache: dict[tuple[int, int], tuple[pc.PolishResult, float, dict[str, Any]]] = {}

    unique_cores = [
        (10, 4),
        (10, 11),
        (34, 4),
        (34, 8),
        (58, 1),
    ]

    for D, w in unique_cores:
        res, L0, meta = build_asymmetric_even_d_core(
            D, w, iterations=iterations, clearance=clearance
        )
        cores_cache[(D, w)] = (res, L0, meta)
        ev = evaluate_candidate(res.q, meta["n0"], meta, incumbents)
        results.append(ev)

    # 2. Upright L-Border wrapped evaluations
    wrapped_targets = [
        {"D": 10, "w": 11, "t": 8, "k_total": 29},
        {"D": 34, "w": 8, "t": 1, "k_total": 30},
    ]

    for spec in wrapped_targets:
        D, w, t, k_total = spec["D"], spec["w"], spec["t"], spec["k_total"]
        res_core, L0_core, meta_core = cores_cache[(D, w)]
        q_wrapped = pc.add_l_border(res_core.q, t=t, k_total=k_total, margin=1e-7)
        n_wrapped = len(q_wrapped)

        res_w = pc.polish_band(
            q_wrapped,
            pc.extent(q_wrapped),
            iterations=15,
            position_trust=0.05,
            angle_trust=0.005,
            clearance=clearance,
        )
        if res_w is None:
            raise RuntimeError(f"Polish of wrapped core failed for D={D}, w={w}, t={t}")

        meta_wrap = dict(
            family="asymmetric_even_d_l_border",
            D=D,
            w=w,
            t=t,
            k_total=k_total,
            k_core=meta_core["k0"],
            n=n_wrapped,
            L45=meta_core["L45"] + t,
            L_polished=res_w.L,
            corrugation_gain=res_w.L - (meta_core["L45"] + t),
            core_meta=meta_core,
        )
        ev = evaluate_candidate(res_w.q, n_wrapped, meta_wrap, incumbents)
        results.append(ev)

    return {
        "timestamp": time.time(),
        "num_evaluations": len(results),
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Asymmetric Even-D Corrugated Cores Generator (D=10, 34, 58)"
    )
    ap.add_argument("--d", type=int, choices=[10, 34, 58], default=None, help="Specific D to run")
    ap.add_argument("--w", type=int, default=None, help="Specific width w to run")
    ap.add_argument("--iterations", type=int, default=30, help="LP polish iterations")
    ap.add_argument("--all", action="store_true", help="Run all assigned test targets")
    ap.add_argument("--out", type=Path, default=DATA, help="Output directory")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    if args.all or (args.d is None and args.w is None):
        print("Running all assigned asymmetric even-D core evaluations...")
        res = run_all_assigned_evaluations(iterations=args.iterations)
        out_json = args.out / "asymmetric_even_d_results.json"
        out_json.write_text(json.dumps(res, indent=2))
        print(f"Results written to {out_json}")
        for r in res["results"]:
            m = r["meta"]
            gain = m.get("corrugation_gain", 0.0)
            print(
                f"n={r['n']:<4} family={m['family']} D={m.get('D')} w={m.get('w')} "
                f"L={r.get('L_literal')} (gain={gain:+.6f}) "
                f"cert_literal={r['literal_ok']} cert_pandas={r['pandas_ok']} "
                f"improves_incumbent={r.get('improves_incumbent')}"
            )
    else:
        D = args.d or 10
        w = args.w or 4
        print(f"Running asymmetric even-D core D={D}, w={w}...")
        res, L0, meta = build_asymmetric_even_d_core(D, w, iterations=args.iterations)
        print(f"Done: n0={meta['n0']}, L45={L0:.6f}, L_corr={res.L:.6f}, gain={meta['corrugation_gain']:+.6f}")


if __name__ == "__main__":
    main()
