"""Unit tests for H40 B5_10x19 / B4_17x9 harvest and pinwheel/interlock assembly."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1] / "poc"
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import b5_lib as bl  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def test_harvest_n850_bricks():
    packs, _ = bl.fast_load_frontier(target_ns=[850])
    bank = bl.load_b5_bank(packs, compress_b5_b4=False)
    tags = {b["tag"]: b for b in bank}
    assert "B5_10x19_n850" in tags
    b5 = tags["B5_10x19_n850"]
    assert (b5["w"], b5["h"], b5["g"]) == (10, 19, 5)
    assert len(b5["x"]) == 195
    assert b5["fx"] < 0.6426
    assert b5["fy"] < 0.8937

    b5_20 = tags["B5_10x20_n850"]
    assert (b5_20["w"], b5_20["h"], b5_20["g"]) == (10, 20, 5)
    assert len(b5_20["x"]) == 205
    assert b5_20["f"] < 0.6426

    b4_18 = tags["B4_18x9_n850"]
    assert (b4_18["w"], b4_18["h"], b4_18["g"]) == (18, 9, 4)
    assert len(b4_18["x"]) == 166
    assert b4_18["f"] < 0.6426

    b5t = tags["B5_10x19_n850_T"]
    assert (b5t["w"], b5t["h"], b5t["g"]) == (19, 10, 5)
    assert abs(b5t["fx"] - b5["fy"]) < 1e-9
    assert abs(b5t["fy"] - b5["fx"]) < 1e-9

    b4 = tags["B4_17x9_n850"]
    assert (b4["w"], b4["h"], b4["g"]) == (17, 9, 4)
    assert len(b4["x"]) == 157
    assert b4["fy"] < 0.6426

    b4_15 = tags["B4_15x9_n850"]
    assert (b4_15["w"], b4_15["h"], b4_15["g"]) == (15, 9, 4)
    assert len(b4_15["x"]) == 139


def test_sym_and_assemble_counts():
    packs, _ = bl.fast_load_frontier(target_ns=[793, 910, 911])
    bank = bl.load_b5_bank(packs, compress_b5_b4=False)
    tags = {b["tag"]: b for b in bank}
    b5 = tags["B5_10x19_n850"]
    b5t = tags["B5_10x19_n850_T"]
    b4 = tags["B4_17x9_n850"]

    for s in range(4):
        sb = bl.sym_brick(b5, s)
        pen, _ = cpp_eval_sat_pen_bbox(sb["x"], sb["y"], sb["th"])
        assert pen <= 1e-7
        assert len(sb["x"]) == 195

    # n=793: m=28, r=+9 (B5_10x19 + B4_17x9)
    asm793 = bl.assemble_b5_pair(28, 9, b5, b4, 0, 0, f_mode="tight")
    assert asm793 is not None
    assert len(asm793[0]) == 793

    # n=910: m=30, r=+10 (B5_10x19 + B5_19x10)
    asm910 = bl.assemble_b5_pair(30, 10, b5, b5t, 0, 0, f_mode="tight")
    assert asm910 is not None
    assert len(asm910[0]) == 910

    # n=911: m=30, r=+11 (B5_10x19 + B6_21x10_w910_T)
    if "B6_10x21_w910_T" in tags:
        b6t = tags["B6_10x21_w910_T"]
        asm911 = bl.assemble_b5_pair(30, 11, b5, b6t, 0, 0, f_mode="tight")
        assert asm911 is not None
        assert len(asm911[0]) == 911
