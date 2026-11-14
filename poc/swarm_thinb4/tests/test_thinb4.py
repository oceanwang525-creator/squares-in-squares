#!/usr/bin/env python3
"""Unit tests for poc/swarm_thinb4/poc/thinb4_lib.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1] / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import thinb4_lib as tl  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def test_extract_corner_bricks_and_zero_overlap_pinwheel():
    packs, _ = tl.fast_load_frontier(target_ns=[684, 737, 848, 849, 908, 969])
    bank = tl.build_h52_bank(packs, compress=False)

    expected = [
        ("B4_9x16_n684", 9, 16, 4),
        ("B4_17x9_n684", 17, 9, 4),
        ("B4_10x16_n737", 10, 16, 4),
        ("B4_11x16_n849", 11, 16, 4),
        ("B4_13x16_n908", 13, 16, 4),
        ("B4_17x13_n908", 17, 13, 4),
        ("B4_16x14_n969", 16, 14, 4),
        ("B3_9x15_n848", 9, 15, 3),
    ]
    for name, w, h, g in expected:
        assert name in bank, f"Missing {name}"
        b = bank[name]
        assert (b["w"], b["h"], b["g"]) == (w, h, g)
        assert len(b["x"]) == w * h + g
        pen, _ = cpp_eval_sat_pen_bbox(b["x"], b["y"], b["th"])
        assert pen <= 1e-7
        assert b["f"] < 0.70

    # Zero-overlap pinwheel checks:
    # n=684 (9x16 + 16x10: 9+16<=26, 16+10=26), n=849 (11x16 + 17x13: 11+17<=29, 16+13=29), n=737 (10x16 + 16x11: 10+16<=27, 16+11=27)
    for n, m, b1_name, b2_name in [
        (684, 26, "B4_9x16_n684", "B4_10x16_n737_T"),
        (849, 29, "B4_11x16_n849", "B4_17x13_n908"),
        (737, 27, "B4_10x16_n737", "B4_11x16_n849_T"),
    ]:
        asms = tl.assemble_pair_best(m, 8, bank[b1_name], bank[b2_name], shift_fracs=(0.0, 0.5))
        assert len(asms) >= 1
        best = min(asms, key=lambda c: c["pen0"])
        assert len(best["x"]) == n
        assert best["pen0"] <= 1e-6, f"n={n} pen0={best['pen0']}"
