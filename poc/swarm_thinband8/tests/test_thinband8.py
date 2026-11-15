"""Tests for H112 (thinband8) final.csv certification and pinwheel geometry helpers."""
from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
THIN_DIR = Path(__file__).resolve().parents[1]
FINAL_CSV = THIN_DIR / "final.csv"

sys.path.insert(0, str(REPO_ROOT / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(THIN_DIR / "poc"))
import thinband8_lib as tb  # noqa: E402
import union as auto_union  # noqa: E402


def _load_check_final_csv():
    spec = importlib.util.spec_from_file_location(
        "check_final_csv", REPO_ROOT / ".github" / "scripts" / "check_final_csv.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_build_pinwheel_kxky_counts():
    m = 28
    b1 = tb.make_brick(9, 20, 6, np.zeros(186) + 0.5, np.zeros(186) + 0.5, np.zeros(186), tag="b1")
    b2 = tb.make_brick(20, 8, 6, np.zeros(166) + 0.5, np.zeros(166) + 0.5, np.zeros(166), tag="b2")
    # kx = 9 + 20 - 28 = +1, ky = 20 + 8 - 28 = 0 -> valid
    built = tb.build_pinwheel_kxky(m, b1, b2)
    assert built is not None
    ax, ay, at, _, _ = built
    assert len(ax) == m * m + 12

    # kx = +1, ky = +1 -> invalid
    b3 = tb.make_brick(20, 9, 6, np.zeros(186) + 0.5, np.zeros(186) + 0.5, np.zeros(186), tag="b3")
    assert tb.build_pinwheel_kxky(m, b1, b3) is None


def test_final_csv_structure_and_certify():
    assert FINAL_CSV.exists(), "final.csv must exist"
    txt = FINAL_CSV.read_text(encoding="utf-8")
    _load_check_final_csv().validate(txt)

    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) <= 1:
        # Header-only final.csv (valid on 0-win retire)
        return

    lit = auto_union.parse_text(txt)
    pd_text = pd.read_csv(FINAL_CSV).to_csv(index=False)
    pd_groups = auto_union.parse_text(pd_text)
    assert sorted(lit.keys()) == sorted(pd_groups.keys())

    frontier = tb.fast_load_true_frontier(list(lit.keys()), certify=True)
    for n in sorted(lit.keys()):
        res = auto_union.check((n, lit[n], pd_groups[n]))
        assert res is not None, f"n={n} failed auto_union certification"
        assert res[0] == res[1], f"n={n}: literal and pandas bounding boxes must match"
        assert Decimal(res[0]) < frontier[n]["L_dec"], f"n={n}: not a strict win over frontier"
