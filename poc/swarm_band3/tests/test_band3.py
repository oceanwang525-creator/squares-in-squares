#!/usr/bin/env python3
"""Tests for H94 (band3) pinwheel construction and certified final.csv."""
from __future__ import annotations

import io
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

BAND3_DIR = Path(__file__).resolve().parents[1]
POC_DIR = BAND3_DIR / "poc"
REPO_ROOT = BAND3_DIR.parents[1]

sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(REPO_ROOT / "poc" / "auto_union" / "poc"))

import band3_lib as b3  # noqa: E402
import union as auto_union  # noqa: E402


def test_build_pinwheel_count() -> None:
    b_p = BAND3_DIR / "data" / "brick_B7_11x19_n975_TL.json"
    if not b_p.exists():
        return
    b1 = b3.load_brick_json(b_p)
    b2 = b3.bl.transform_brick(b1, "T")
    built = b3.build_pinwheel(30, b1, b2, sf=0.0, max_ov=1)
    assert built is not None
    ax, ay, at, _, _ = built
    assert len(ax) == 30 * 30 + 14
    assert len(ay) == 914
    assert len(at) == 914


def test_final_csv_certified() -> None:
    final_p = BAND3_DIR / "final.csv"
    assert final_p.exists()
    txt = final_p.read_text(encoding="utf-8")
    lines = txt.splitlines()
    assert lines[0] == "id,x,y,deg"
    if len(lines) <= 1:
        return

    lit = auto_union.parse_text(txt)
    pan_txt = pd.read_csv(io.StringIO(txt)).to_csv(index=False)
    pan = auto_union.parse_text(pan_txt)
    assert set(lit.keys()) == set(pan.keys())
    for n in sorted(lit):
        res = auto_union.check((n, lit[n], pan[n]))
        assert res is not None, f"Certification failed for n={n}"
        assert Decimal(res[0]) > 0 and Decimal(res[1]) > 0
