"""Unit tests for poc/swarm_armslack."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
ARMSLACK_POC = Path(__file__).resolve().parents[1] / "poc"
if str(ARMSLACK_POC) not in sys.path:
    sys.path.insert(0, str(ARMSLACK_POC))

import strip_slide as ss  # noqa: E402

MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
FINAL_CSV = Path(__file__).resolve().parents[1] / "final.csv"


def _load_norm(n: int):
    df = pd.read_csv(MERGED_CSV)
    sub = df[df["n"] == n].sort_values("square_id")
    x = sub["x"].to_numpy(np.float64, copy=True)
    y = sub["y"].to_numpy(np.float64, copy=True)
    th = sub["angle"].to_numpy(np.float64, copy=True)
    hx = ss._half_extents(th)
    x -= float((x - hx).min())
    y -= float((y - hx).min())
    L = float(max((x + hx).max(), (y + hx).max()))
    return x, y, th, L


def test_detect_and_slide_n915():
    x, y, th, L = _load_norm(915)
    info = ss.detect_endcap_strips(x, y, th, L)
    assert info["r_y0"] > 1.0
    assert len(info["r_idx"]) >= 8
    nx, ny, nth, moved = ss.slide_endcap_strip(x, y, th, L, info["r_idx"], "right", inward_dx=0.16)
    assert len(moved) == len(info["r_idx"])
    # First square slid to y=0.5, last square shifted inward
    assert abs(ny[moved[0]] - 0.5) < 1e-9
    assert abs(nx[moved[-1]] - (L - 0.66)) < 1e-9


def test_detect_asym_n914():
    x, y, th, L = _load_norm(914)
    info = ss.detect_endcap_strips(x, y, th, L)
    assert info["dA"] < -0.020
    assert abs(info["dB"]) < 1e-5


def test_build_seeds_shapes():
    x, y, th, L = _load_norm(917)
    rng = np.random.default_rng(42)
    q_list, m_list, ls_list, labels = ss.build_armslack_seeds(917, x, y, th, L, rng)
    assert len(q_list) == 24
    assert all(len(q) == 3 * 917 for q in q_list)
    assert all(len(m) == 3 * 917 for m in m_list)
    assert all(len(ls) == 4 for ls in ls_list)
    assert len(labels) == 24


def test_canonicalize_tr_rotated():
    x, y, th, L = _load_norm(577)
    xc, yc, tc, best_q = ss.canonicalize_tr(x, y, th, L)
    assert best_q in (0, 1, 2, 3)
    _, _, _, q_after = ss.canonicalize_tr(xc, yc, tc, L)
    assert q_after == 0
    info = ss.detect_endcap_strips(xc, yc, tc, L)
    assert len(info["r_idx"]) >= 3 or len(info["t_idx"]) >= 3


def test_final_csv_schema_if_present():
    if not FINAL_CSV.exists():
        return
    spec = importlib.util.spec_from_file_location("check_final_csv", REPO_ROOT / ".github/scripts/check_final_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.validate(FINAL_CSV.read_text(encoding="utf-8"))
