"""Tests for H113 (pocketpeel) sub-pocket peeling, pinwheel assembly, and final.csv certification."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PP_DIR = Path(__file__).resolve().parents[1]
PP_POC = PP_DIR / "poc"
FINAL_CSV = PP_DIR / "final.csv"

for _p in (PP_POC, REPO_ROOT / "poc" / "auto_union" / "poc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pocketpeel_lib as pp  # noqa: E402
import union as auto_union  # noqa: E402


def _load_check_final_csv():
    spec = importlib.util.spec_from_file_location(
        "check_final_csv", REPO_ROOT / ".github" / "scripts" / "check_final_csv.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_peel_n852_donors():
    fr = pp.load_merged_frontier([852], certify=False)
    b6 = pp.cp2.extract_corner_brick(fr[852]["x"], fr[852]["y"], fr[852]["th"], fr[852]["L"], "BR", 20, 11, 6, "t_20x11")
    b5 = pp.cp2.extract_corner_brick(fr[852]["x"], fr[852]["y"], fr[852]["th"], fr[852]["L"], "TL", 9, 17, 5, "t_9x17")
    assert b6 is not None and b5 is not None
    p_top = pp.peel_outer_unit_layer(b6, "top", "p_top")
    p_right = pp.peel_outer_unit_layer(b5, "right", "p_right")
    assert p_top is not None and (p_top["w"], p_top["h"], p_top["g"]) == (20, 10, 6)
    assert p_right is not None and (p_right["w"], p_right["h"], p_right["g"]) == (8, 17, 5)


def test_final_csv_structure_and_certify():
    if not FINAL_CSV.exists():
        return
    txt = FINAL_CSV.read_text(encoding="utf-8")
    _load_check_final_csv().validate(txt)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return
    lit = auto_union.parse_text(txt)
    pd_text = pd.read_csv(FINAL_CSV).to_csv(index=False)
    pd_groups = auto_union.parse_text(pd_text)
    assert sorted(lit.keys()) == sorted(pd_groups.keys())
    for n in sorted(lit.keys()):
        res = auto_union.check((n, lit[n], pd_groups[n]))
        assert res is not None, f"n={n} failed auto_union certification"
        assert res[0] == res[1], f"n={n} literal != pandas"
