"""Tests for poc/swarm_vstrip852."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
VSTRIP_DIR = REPO / "poc" / "swarm_vstrip852"
sys.path.insert(0, str(VSTRIP_DIR / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_gstep" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))

import gstep_lib as gl  # noqa: E402
import vstrip_lib as vl  # noqa: E402
import union as auto_union  # noqa: E402


def test_orient_and_vstrip_count():
    base = vl.extract_champion_bricks()
    assert "n852_9x17" in base
    assert "n852_11x20" in base
    b9 = base["n852_9x17"]
    b11 = base["n852_11x20"]
    assert (b9["w"], b9["h"], b9["g"]) == (9, 17, 5)
    assert (b11["w"], b11["h"], b11["g"]) == (11, 20, 6)
    c9 = gl.splice_vertical_strip(b9)
    c11 = gl.splice_vertical_strip(b11)
    assert len(c9) > 0 and len(c11) > 0
    for _, cand in c9:
        assert (cand["w"], cand["h"], cand["g"]) == (9, 18, 6)
        assert len(cand["x"]) == 9 * 18 + 6
    for _, cand in c11:
        assert (cand["w"], cand["h"], cand["g"]) == (11, 21, 7)
        assert len(cand["x"]) == 11 * 21 + 7


def test_final_csv_valid_and_dual_cert():
    final_csv = VSTRIP_DIR / "final.csv"
    if not final_csv.exists():
        return
    text = final_csv.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("check_final_csv", REPO / ".github/scripts/check_final_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.validate(text)

    lit = auto_union.parse_text(text)
    pd_text = pd.read_csv(io.StringIO(text)).to_csv(index=False)
    pan = auto_union.parse_text(pd_text)
    assert set(lit) == set(pan)
    for n in sorted(lit):
        res = auto_union.check((n, lit[n], pan[n]))
        assert res is not None, f"n={n} failed official literal/pandas certification"
