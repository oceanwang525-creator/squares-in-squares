"""Pytest suite for Multi-Pell Cores POC 3 (Negative/Near-Zero Slack Pell Cores).

Verifies the mathematical properties, column pitch compression, and falsification gate
outcomes for D=7 and D=41 semi-convergent Pell cores.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "baseline/poc/src"))
sys.path.insert(0, str(REPO / "pell_corrugation/poc"))

from multi_pell_cores_3 import (
    compute_diagonal_slack,
    build_negative_slack_seed,
    measure_column_pitch,
    evaluate_falsification_gate,
)
from pell_corrugation import _lp, extent, sat_check


def test_negative_slack_algebra():
    """Verify Diophantine negative slack algebra for Pell semi-convergents."""
    slack_7 = compute_diagonal_slack(7)
    slack_41 = compute_diagonal_slack(41)

    assert slack_7 < 0.0
    assert abs(slack_7 - (-0.050252531694)) < 1e-6

    assert slack_41 < 0.0
    assert abs(slack_41 - (-0.008621626084)) < 1e-6


def test_column_pitch_compression_d7():
    """Verify that D=7 core forces column pitch below 1.0, creating mandatory overlap."""
    q, L0, meta = build_negative_slack_seed(D=7, w=4, a1=6, a2=5, b1=2, b2=2)
    mean_pitch, min_pitch, pitch = measure_column_pitch(q, w=4, l=9, a1=6, a2=5, b1=2, b2=2)

    # Unit squares tilted in parallel columns require longitudinal spacing >= 1.0
    assert mean_pitch < 1.0
    assert min_pitch < 1.0
    assert min_pitch < 0.995

    # SAT check on initial unrelaxed seed must fail due to longitudinal pitch compression
    is_valid, min_gap = sat_check(q)
    assert not is_valid
    assert min_gap < -0.05


def test_column_pitch_compression_d41():
    """Verify that D=41 core forces column pitch below 1.0, creating mandatory overlap."""
    q, L0, meta = build_negative_slack_seed(D=41, w=8, a1=23, a2=23, b1=5, b2=5)
    mean_pitch, min_pitch, pitch = measure_column_pitch(q, w=8, l=37, a1=23, a2=23, b1=5, b2=5)

    assert mean_pitch < 1.0
    assert min_pitch < 1.0

    is_valid, min_gap = sat_check(q)
    assert not is_valid
    assert min_gap < -0.05


def test_falsification_gate_d7():
    """Verify that D=7 core fails the Q5 falsification gate (HiGHS LP infeasible)."""
    res = evaluate_falsification_gate(D=7, w=4, a1=6, a2=5, b=2, iterations=15)
    assert not res.gate_passed
    assert not res.is_feasible
    assert "infeasible" in res.failure_reason.lower()


def test_falsification_gate_d41():
    """Verify that D=41 core fails the Q5 falsification gate (HiGHS LP infeasible)."""
    res = evaluate_falsification_gate(D=41, w=8, a1=23, a2=23, b=5, iterations=15)
    assert not res.gate_passed
    assert not res.is_feasible
    assert "infeasible" in res.failure_reason.lower()


def test_dilation_proves_no_shrinkage_d7():
    """Verify that allowing container dilation for D=7 yields L > L45 (zero shrinkage)."""
    q, L0, _ = build_negative_slack_seed(D=7, w=4, a1=6, a2=5, b1=2, b2=2)
    # Allow container to expand significantly
    repaired, msg = _lp(q, L0 + 0.50, 0.40, 0.0, 5e-9, {})
    assert repaired is not None
    dilated_L = extent(repaired)
    # Dilation is strictly positive: container expands rather than shrinks
    assert dilated_L > L0
    assert dilated_L - L0 > 0.10
