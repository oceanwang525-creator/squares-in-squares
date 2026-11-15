#!/usr/bin/env python3
"""Tests for H115 (swarm_highdpocket) final.csv and DP helpers."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HIGH_DIR = REPO / "poc" / "swarm_highdpocket"
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(HIGH_DIR / "poc"))

import union as auto_union  # noqa: E402
import overhang_dp  # noqa: E402


def test_final_csv_structure_and_dual_cert() -> None:
    final_csv = HIGH_DIR / "final.csv"
    text = final_csv.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("chk", REPO / ".github" / "scripts" / "check_final_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.validate(text)

    lit, pdt = auto_union.load(final_csv)
    assert sorted(lit.keys()) == [857, 915, 978]
    for n in sorted(lit):
        res = auto_union.check((n, lit[n], pdt[n]))
        assert res is not None
        assert res[0] == res[1]


def test_overhang_dp_formula() -> None:
    # Synthetic 1-axis overhang pair for K=29, D=16: (22,11,+10) + (8,17,+7)
    items = [
        {"w": 22, "h": 11, "g": 10, "f": 0.755, "src": 917, "tr": False, "pk": {}},
        {"w": 8, "h": 17, "g": 7, "f": 0.761, "src": 541, "tr": False, "pk": {}},
    ]
    L_inc = {29 * 29 + 16: 29.7615364}
    cells = overhang_dp.search_overhang(items, L_inc, k_min=29, k_max=29, d_min=16, d_max=16, slack_max=0.006)
    assert len(cells) == 1
    assert cells[0]["n"] == 857
    assert cells[0]["pairs"][0]["kx"] == 1
    assert cells[0]["pairs"][0]["ky"] == 0
