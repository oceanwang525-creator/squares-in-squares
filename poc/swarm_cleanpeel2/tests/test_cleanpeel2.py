"""Tests for H74 (cleanpeel2) final.csv geometry and certification."""
from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FINAL_CSV = Path(__file__).resolve().parents[1] / "final.csv"

sys.path.insert(0, str(REPO_ROOT / "poc" / "auto_union" / "poc"))
import union as auto_union  # noqa: E402


def _load_check_final_csv():
    spec = importlib.util.spec_from_file_location(
        "check_final_csv", REPO_ROOT / ".github" / "scripts" / "check_final_csv.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_final_csv_structure_and_certify():
    txt = FINAL_CSV.read_text(encoding="utf-8")
    _load_check_final_csv().validate(txt)

    lit = auto_union.parse_text(txt)
    pd_text = pd.read_csv(FINAL_CSV).to_csv(index=False)
    pd_groups = auto_union.parse_text(pd_text)

    assert sorted(lit.keys()) == [972]
    assert sorted(pd_groups.keys()) == [972]

    res = auto_union.check((972, lit[972], pd_groups[972]))
    assert res is not None, "n=972 failed auto_union certification"
    assert res[0] == res[1], "literal and pandas bounding boxes must match"
    assert Decimal(res[0]) < Decimal("31.65651217859088684474960132018240325435968880086005")
