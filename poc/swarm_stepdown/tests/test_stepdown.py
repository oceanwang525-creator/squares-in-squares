#!/usr/bin/env python3
"""Tests for H97 (stepdown) brick utilities, pinwheel assembly, and final.csv."""
from __future__ import annotations

import io
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

STEPDOWN_DIR = Path(__file__).resolve().parents[1]
POC_DIR = STEPDOWN_DIR / "poc"
REPO_ROOT = STEPDOWN_DIR.parents[1]

sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(REPO_ROOT / "poc" / "auto_union" / "poc"))

import stepdown_lib as sl  # noqa: E402
import union as auto_union  # noqa: E402


def test_stepdown_brick_loading_and_sym() -> None:
    b_p = STEPDOWN_DIR / "data" / "brick_B7_17x8_g6.json"
    assert b_p.exists()
    d = json.loads(b_p.read_text(encoding="utf-8"))
    b = {
        "w": int(d["w"]),
        "h": int(d["h"]),
        "g": int(d["g"]),
        "x": np.asarray(d["x"], dtype=np.float64),
        "y": np.asarray(d["y"], dtype=np.float64),
        "th": np.asarray(d["th"], dtype=np.float64),
        "f": float(d["f"]),
        "tag": "17x8_g6",
    }
    assert len(b["x"]) == b["w"] * b["h"] + b["g"]
    bt = sl.make_sym_brick(b, "T")
    assert bt["w"] == b["h"] and bt["h"] == b["w"]
    assert abs(bt["f"] - b["f"]) < 1e-9


def test_assemble_pinwheel_count() -> None:
    b_p = STEPDOWN_DIR / "data" / "brick_B7_17x8_g6.json"
    d = json.loads(b_p.read_text(encoding="utf-8"))
    b1 = {
        "w": int(d["w"]),
        "h": int(d["h"]),
        "g": int(d["g"]),
        "x": np.asarray(d["x"], dtype=np.float64),
        "y": np.asarray(d["y"], dtype=np.float64),
        "th": np.asarray(d["th"], dtype=np.float64),
        "f": float(d["f"]),
        "tag": "17x8_g6",
    }
    b2 = sl.make_sym_brick(b1, "T")
    cands = sl.assemble_zero_overlap_pinwheel(27, 12, b1, b2)
    assert len(cands) > 0
    ax, ay, at, _, _ = cands[0]
    assert len(ax) == 27 * 27 + 12
    assert len(ay) == 741
    assert len(at) == 741


def test_final_csv_dual_certified() -> None:
    final_p = STEPDOWN_DIR / "final.csv"
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
