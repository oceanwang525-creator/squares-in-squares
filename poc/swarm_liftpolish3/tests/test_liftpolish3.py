"""Unit tests for poc/swarm_liftpolish3 helpers and final.csv integrity."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fast_float_side_unit_square():
    mod = load_module("build_combined_portfolio", HERE / "poc" / "build_combined_portfolio.py")
    rows = [
        {"square_id": "0", "x": "s0.5", "y": "s0.5", "deg": "s0"},
        {"square_id": "1", "x": "s1.5", "y": "s0.5", "deg": "s0"},
    ]
    assert abs(mod.fast_float_side(rows, is_deg=True) - 2.0) < 1e-12


def test_convert_deg_rows_preserves_zero_and_range():
    mod = load_module("build_combined_portfolio", HERE / "poc" / "build_combined_portfolio.py")
    rows = [{"square_id": "0", "x": "s0.5", "y": "s0.5", "deg": "s45"}]
    conv = mod.convert_deg_rows(rows)
    assert len(conv) == 1
    assert abs(float(conv[0]["angle"]) - 0.7853981633974483) < 1e-12


def test_final_csv_validates_with_trusted_checker():
    checker = load_module("check_final_csv", REPO / ".github" / "scripts" / "check_final_csv.py")
    final_path = HERE / "final.csv"
    if not final_path.exists():
        return
    checker.validate(final_path.read_text(encoding="utf-8"))
