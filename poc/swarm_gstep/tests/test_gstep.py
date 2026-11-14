"""Unit and certification tests for H79 (gstep) synthesized bricks and final.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_DIR.parents[1]
sys.path.insert(0, str(PKG_DIR / "poc"))
sys.path.insert(0, str(REPO_ROOT / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(REPO_ROOT / "poc" / "swarm_lpslack" / "poc"))

import gstep_lib as gl  # noqa: E402
import union as au  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def test_synthesized_bricks_feasible_and_excess():
    brick_files = sorted((PKG_DIR / "data").glob("brick_*.json"))
    assert brick_files, "expected synthesized brick JSON files in poc/swarm_gstep/data/"
    for bf in brick_files:
        b = gl.load_brick_json(bf)
        n_sq = len(b["x"])
        assert n_sq == b["w"] * b["h"] + b["g"], f"{bf.name}: {n_sq} != {b['w']}*{b['h']}+{b['g']}"
        pen, _ = cpp_eval_sat_pen_bbox(b["x"], b["y"], b["th"])
        assert pen <= 1e-10, f"{bf.name}: nonzero overlap penalty {pen}"
        _, _, lx, ly = gl.bl.rect_bbox(b["x"], b["y"], b["th"])
        assert lx <= b["w"] + b["f"] + 1e-9
        assert ly <= b["h"] + b["f"] + 1e-9
        assert 0.0 < b["f"] < 1.0


def test_final_csv_dual_certified():
    final_path = PKG_DIR / "final.csv"
    assert final_path.exists(), "poc/swarm_gstep/final.csv must exist"
    text = final_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0] == "id,x,y,deg"
    if len(lines) == 1:
        return

    lit_groups, pan_groups = au.load(final_path)
    assert len(lit_groups) > 0
    assert set(lit_groups) == set(pan_groups)
    for n in sorted(lit_groups):
        rows = lit_groups[n]
        assert len(rows) == n
        res = au.check((n, rows, pan_groups[n]))
        assert res is not None, f"n={n}: certification failed"
        lit, pan = res
        assert lit == pan, f"n={n}: literal {lit} != pandas {pan}"
        # Verify exact normalized coordinates are finite
        xs = np.array([float(r["x"][1:]) for r in rows])
        ys = np.array([float(r["y"][1:]) for r in rows])
        assert np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))
