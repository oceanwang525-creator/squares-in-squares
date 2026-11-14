"""Tests for poc/swarm_softseamdp."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SOFT_DIR = REPO / "poc" / "swarm_softseamdp"
sys.path.insert(0, str(SOFT_DIR / "poc"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))

import dp_soft  # noqa: E402
import union as auto_union  # noqa: E402


def test_dp_soft_constraints_and_top3():
    items = [
        dict(w=5, h=6, g=3, f=0.60, src=1, tag="a", transposed=False),
        dict(w=7, h=8, g=4, f=0.61, src=2, tag="b", transposed=False),
        dict(w=4, h=5, g=2, f=0.62, src=3, tag="c", transposed=False),
        dict(w=8, h=8, g=5, f=0.63, src=4, tag="d", transposed=False),
        dict(w=10, h=10, g=6, f=0.65, src=5, tag="e", transposed=False),
    ]
    K = 16
    L_inc = {K * K + D: K + 0.62 for D in range(1, 2 * K + 2)}
    res = dp_soft.solve_for_K(K, items, L_inc, slack_max=6.0e-3, top_k=3)
    assert res, "expected at least one qualifying cell"
    for cell in res:
        assert cell["K"] == K
        assert cell["best_dL_rigid"] <= 6.0e-3 + 1e-12
        seen_keys = set()
        for seam in cell["seams"]:
            m = seam["m"]
            assert m in (2, 3)
            comp = seam["comp"]
            assert len(comp) == m
            assert sum(c["w"] for c in comp) <= K
            assert sum(c["h"] for c in comp) <= K
            assert sum(c["w"] + c["h"] for c in comp) + m - 1 <= 2 * K
            assert sum(c["g"] for c in comp) >= cell["D"]
            key = tuple(sorted((c["w"], c["h"], c["g"], c["src"]) for c in comp))
            assert key not in seen_keys
            seen_keys.add(key)


def test_final_csv_structure_and_dual_cert():
    final_csv = SOFT_DIR / "final.csv"
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
    if not lit:
        return

    report_path = SOFT_DIR / "data" / "union_report.json"
    rep = {int(r["n"]): Decimal(r["L"]) for r in json.loads(report_path.read_text())["results"]} if report_path.exists() else {}
    for n in sorted(lit):
        res = auto_union.check((n, lit[n], pan[n]))
        assert res is not None, f"n={n} failed official literal/pandas certification"
        if n in rep:
            assert max(Decimal(res[0]), Decimal(res[1])) <= rep[n] - Decimal("1e-9"), f"n={n} did not beat incumbent"
