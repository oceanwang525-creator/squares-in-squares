"""Tests for poc/swarm_b8xplant (H25 + H28)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
B8_POC = Path(__file__).resolve().parents[1] / "poc"
sys.path.insert(0, str(B8_POC))
import b8_lib as bl  # noqa: E402


def test_b8_brick_extracted_and_feasible() -> None:
    b8 = bl.ensure_b8_brick_json()
    assert b8["w"] == 10 and b8["h"] == 18 and b8["g"] == 8
    assert len(b8["x"]) == 188
    pen, _ = bl.cpp_eval_sat_pen_bbox(b8["x"], b8["y"], b8["th"])
    assert pen <= 1e-10
    assert 0.774 < b8["f"] < 0.7746


def test_localized_contact_jacobian_restricts_to_free_mask() -> None:
    b8 = bl.ensure_b8_brick_json()
    x, y, th, _ = bl.normalize(b8["x"], b8["y"], b8["th"])
    free_mask = np.zeros(len(x), dtype=bool)
    free_mask[:30] = True
    jac, fidx, info = bl.localized_contact_jacobian(x, y, th, free_mask, contact_tol=1e-4)
    assert len(fidx) == 30
    assert jac.shape[1] == 90
    assert info["constraints"] > 0
    singular, modes = bl.localized_soft_modes(jac, k=4)
    assert modes.shape == (4, 90)
    assert np.all(np.isfinite(singular))


def test_final_csv_structure_and_dual_cert() -> None:
    final_path = Path(__file__).resolve().parents[1] / "final.csv"
    assert final_path.exists()
    text = final_path.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "check_final_csv", REPO_ROOT / ".github" / "scripts" / "check_final_csv.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.validate(text)

    lit, pd_t = bl.auto_union.load(final_path)
    assert set(lit) == set(pd_t)
    for n in sorted(lit):
        res = bl.auto_union.check((n, lit[n], pd_t[n]))
        assert res is not None, f"dual certification failed for n={n}"
