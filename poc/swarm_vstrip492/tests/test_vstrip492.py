#!/usr/bin/env python3
"""Tests for H106 (vstrip492) brick splicing/removal, pinwheel builder, and certified final.csv."""
from __future__ import annotations

import io
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

VSTRIP_DIR = Path(__file__).resolve().parents[1]
POC_DIR = VSTRIP_DIR / "poc"
REPO_ROOT = VSTRIP_DIR.parents[1]

sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(REPO_ROOT / "poc" / "auto_union" / "poc"))

import union as auto_union  # noqa: E402
import vstrip492_lib as vl  # noqa: E402


def test_splice_and_remove_vertical_strip() -> None:
    src_p = REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data" / "brick_B6_10x21_mw972_walk.json"
    if not src_p.exists():
        return
    b = vl.load_brick_json(src_p, "B6_10x21")
    v_plus = vl.gl.splice_vertical_strip(b)
    assert len(v_plus) >= 1
    _, bp = v_plus[0]
    assert (bp["w"], bp["h"], bp["g"]) == (10, 22, 7)
    assert len(bp["x"]) == 10 * 22 + 7

    v_minus = vl.remove_vertical_strip(b)
    assert len(v_minus) >= 1
    _, bm = v_minus[0]
    assert (bm["w"], bm["h"], bm["g"]) == (10, 20, 5)
    assert len(bm["x"]) == 10 * 20 + 5


def test_build_pinwheel_count() -> None:
    bricks = vl.load_all_bricks(include_external=True)
    b6 = [b for b in bricks if b["g"] == 6 and b["w"] <= 12 and b["h"] <= 22]
    b5 = [b for b in bricks if b["g"] == 5 and b["w"] <= 20 and b["h"] <= 12]
    assert b6 and b5
    built = vl.build_pinwheel(30, b6[0], b5[0], sf=0.0, max_ov=1)
    assert built is not None
    ax, ay, at, _, _ = built
    assert len(ax) == 30 * 30 + 11
    assert len(ay) == 911
    assert len(at) == 911


def test_final_csv_certified() -> None:
    final_p = VSTRIP_DIR / "final.csv"
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
