"""Tests for poc/swarm_b7xplant (H77)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
B7_POC = Path(__file__).resolve().parents[1] / "poc"
sys.path.insert(0, str(B7_POC))
import b7_lib as b7  # noqa: E402
import b8_lib as bl  # noqa: E402


def test_make_brick_and_transform() -> None:
    b8 = bl.ensure_b8_brick_json()
    bk = b7.make_brick(b8["w"], b8["h"], b8["g"], b8["x"], b8["y"], b8["th"], src_n=639, tag="b8_test")
    assert bk is not None
    assert bk["w"] == 10 and bk["h"] == 18 and bk["g"] == 8
    bt = bl.transform_brick(bk, "T")
    assert bt["w"] == 18 and bt["h"] == 10 and bt["g"] == 8
    assert abs(bt["f"] - bk["f"]) < 1e-10


def test_final_csv_structure_and_dual_cert() -> None:
    final_path = Path(__file__).resolve().parents[1] / "final.csv"
    if not final_path.exists():
        return
    text = final_path.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "check_final_csv", REPO_ROOT / ".github" / "scripts" / "check_final_csv.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.validate(text)

    lit, pd_t = b7.auto_union.load(final_path)
    assert set(lit) == set(pd_t)
    for n in sorted(lit):
        res = b7.auto_union.check((n, lit[n], pd_t[n]))
        assert res is not None, f"dual certification failed for n={n}"
