#!/usr/bin/env python3
"""Unit tests for poc/swarm_brickslp."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
POC_DIR = REPO / "poc" / "swarm_brickslp" / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import brick_slp as bs  # noqa: E402
import frontier_io as fio  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def test_rect_bbox_and_assemble_pinwheel():
    # Simple 2x2,+1 brick (5 squares in 2.707 x 2.707)
    s2 = 1.0 / np.sqrt(2.0)
    f = s2
    x = np.array([0.5, 0.5, 2.2071067811865475, 2.2071067811865475, 0.5 + s2 + 0.5 * s2])
    # Use actual n=5 from true frontier to verify assemble_pinwheel cleanly
    packs, _ = fio.load_true_frontier(check_finals=False)
    x5, y5, th5, L5 = packs[5]
    nx, ny, W, H = bs.rect_bbox(x5, y5, th5)
    b = {"w": 2, "h": 2, "g": 1, "f": max(W - 2, H - 2), "fx": W - 2, "fy": H - 2, "x": nx, "y": ny, "th": th5}
    asm = bs.assemble_pinwheel(6, b, b, align_mode="center")
    assert asm is not None
    ax, ay, at, L = asm
    assert len(ax) == 6 * 6 + 2
    pen, bb = cpp_eval_sat_pen_bbox(ax, ay, at)
    assert pen < 1e-10
    assert abs(bb - L) < 1e-6


def test_joint_xytheta_slp_rect_feasibility():
    packs, _ = fio.load_true_frontier(check_finals=False)
    x10, y10, th10, L10 = packs[10]
    nx, ny, W, H = bs.rect_bbox(x10, y10, th10)
    # 10 = 3*3 + 1
    xs, ys, ts, fs, ok = bs.joint_xytheta_slp_rect(nx * 1.005, ny * 1.005, th10, 3, 3, rounds=6)
    assert ok
    pen, _ = cpp_eval_sat_pen_bbox(xs, ys, ts)
    assert pen <= 1e-9
    assert fs <= (max(W - 3, H - 3)) + 1e-3


def test_final_csv_valid():
    final_csv = REPO / "poc" / "swarm_brickslp" / "final.csv"
    if final_csv.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("check_final_csv", REPO / ".github/scripts/check_final_csv.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.validate(final_csv.read_text())
