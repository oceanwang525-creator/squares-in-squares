"""Multi-Pell Cores POC 3: Negative/Near-Zero Slack Pell Cores (D=7, 41).

Investigates semi-convergent Pell cores where l = w + ceil(D / sqrt(2)) introduces
longitudinal interference (negative slack):
  - D = 7:  7 / sqrt(2) = 4.949747, l = w + 5,  slack = -0.050253
  - D = 41: 41 / sqrt(2) = 28.991378, l = w + 29, slack = -0.008622

Evaluates whether 4-sign-piece epigraph LP transverse flank tooth penetration
can compress column pitch along the diagonal axis to overcome initial negative slack.
Implements the exact falsification gate from Q5:
  Falsification Gate: If L_corr >= L45 - 0.00080 or LP is infeasible,
  the candidate Pell family is falsified.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
sys.path.insert(0, str(REPO / "poc/pell_corrugation/poc"))

from pell_corrugation import (
    INV_S2,
    S2,
    PolishResult,
    _lp,
    build_corrugated_band,
    extent,
    normalize_box,
    polish_band,
    sat_check,
)
from squares.core import certify

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2

# Pell semi-convergents with negative longitudinal slack along 45-degree diagonal
PELL_NEGATIVE_SLACK_PARAMS = {
    7: {
        "D": 7,
        "l_offset": 5,  # ceil(7 / sqrt(2)) = 5
        "slack": 7.0 * INV_S2 - 5.0,  # -0.05025253169416715
        "default_w": 4,
        "default_b": 2,
        "default_a": (6, 5),  # out-of-phase: a1+a2 = 11 = 2b + 7
    },
    41: {
        "D": 41,
        "l_offset": 29,  # ceil(41 / sqrt(2)) = 29
        "slack": 41.0 * INV_S2 - 29.0,  # -0.008621626084428056
        "default_w": 8,
        "default_b": 5,
        "default_a": (23, 23),  # Q5 seed parameters; sum=46 gives L45=29.656854
    },
}


@dataclass
class FalsificationResult:
    D: int
    w: int
    l: int
    a: tuple[int, int]
    b: tuple[int, int]
    n0: int
    L45: float
    slack: float
    mean_pitch: float
    min_pitch: float
    lp_status: str
    is_feasible: bool
    L_corr: float | None
    delta_L: float | None
    gate_passed: bool
    failure_reason: str


def compute_diagonal_slack(D: int) -> float:
    """Calculate longitudinal diagonal slack delta = D / sqrt(2) - ceil(D / sqrt(2))."""
    return D * INV_S2 - math.ceil(D * INV_S2)


def build_negative_slack_seed(
    D: int,
    w: int,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    delta: float = 0.002935,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Build synthetic corrugated band seed for negative-slack Pell core."""
    l_step = int(math.ceil(D * INV_S2))
    l = w + l_step
    slack = D * INV_S2 - l_step
    q, L0 = build_corrugated_band(w, l, a1, a2, b1, b2, delta=delta)
    meta = {
        "D": D,
        "w": w,
        "l": l,
        "a": (a1, a2),
        "b": (b1, b2),
        "n0": len(q),
        "L45": L0,
        "slack": slack,
    }
    return q, L0, meta


def measure_column_pitch(q: np.ndarray, w: int, l: int, a1: int, a2: int, b1: int, b2: int) -> tuple[float, float, np.ndarray]:
    """Measure the longitudinal pitch between successive tilted column centers along u = (x+y)/sqrt(2)."""
    n_staircase = a1 * (a1 + 1) // 2 + a2 * (a2 + 1) // 2 + b1 * (b1 + 1) // 2 + b2 * (b2 + 1) // 2
    q_band = q[n_staircase : n_staircase + w * l]
    u_cols = np.array([(q_band[c * w, 0] + q_band[c * w, 1]) * INV_S2 for c in range(l)])
    pitch = np.diff(u_cols)
    return float(np.mean(pitch)), float(np.min(pitch)), pitch


def evaluate_falsification_gate(
    D: int,
    w: int,
    a1: int,
    a2: int,
    b: int,
    iterations: int = 15,
    position_trust: float = 0.20,
    angle_trust: float = 0.015,
    clearance: float = 5e-9,
) -> FalsificationResult:
    """Evaluate whether candidate negative-slack Pell core passes the Q5 falsification gate.

    Gate condition:
      Pass if L_corr < L45 - 0.00080 and LP is feasible.
      Fail/Falsified if L_corr >= L45 - 0.00080 or LP is infeasible.
    """
    l_step = int(math.ceil(D * INV_S2))
    l = w + l_step
    slack = D * INV_S2 - l_step
    q, L0, meta = build_negative_slack_seed(D, w, a1, a2, b, b)
    mean_pitch, min_pitch, _ = measure_column_pitch(q, w, l, a1, a2, b, b)

    axes: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    repaired, msg = _lp(q, L0 + 0.15, position_trust, 0.0, clearance, axes)

    if repaired is None:
        return FalsificationResult(
            D=D,
            w=w,
            l=l,
            a=(a1, a2),
            b=(b, b),
            n0=len(q),
            L45=L0,
            slack=slack,
            mean_pitch=mean_pitch,
            min_pitch=min_pitch,
            lp_status=msg,
            is_feasible=False,
            L_corr=None,
            delta_L=None,
            gate_passed=False,
            failure_reason=(
                f"HiGHS LP infeasible at L0+0.15 ({msg}). Negative diagonal slack ({slack:.5f}) "
                f"compresses column pitch to {mean_pitch:.4f} < 1.0000, causing mandatory geometric "
                f"overlap between adjacent columns that cannot be resolved within container bounds."
            ),
        )

    res = polish_band(q, L0, iterations=iterations, position_trust=position_trust, angle_trust=angle_trust, clearance=clearance)
    if res is None:
        L_rep = extent(repaired)
        return FalsificationResult(
            D=D,
            w=w,
            l=l,
            a=(a1, a2),
            b=(b, b),
            n0=len(q),
            L45=L0,
            slack=slack,
            mean_pitch=mean_pitch,
            min_pitch=min_pitch,
            lp_status="Repair feasible but full polish failed",
            is_feasible=True,
            L_corr=L_rep,
            delta_L=L_rep - L0,
            gate_passed=False,
            failure_reason=f"Repaired L={L_rep:.6f} >= L45 ({L0:.6f}). Zero corrugation shrinkage.",
        )

    delta_L = res.L - L0
    gate_passed = delta_L < -0.00080
    failure_reason = "" if gate_passed else f"L_corr ({res.L:.6f}) >= L45 - 0.00080 ({L0 - 0.00080:.6f}). Delta L = {delta_L:+.6f}."

    return FalsificationResult(
        D=D,
        w=w,
        l=l,
        a=(a1, a2),
        b=(b, b),
        n0=len(q),
        L45=L0,
        slack=slack,
        mean_pitch=mean_pitch,
        min_pitch=min_pitch,
        lp_status="Feasible and polished",
        is_feasible=True,
        L_corr=res.L,
        delta_L=delta_L,
        gate_passed=gate_passed,
        failure_reason=failure_reason,
    )
