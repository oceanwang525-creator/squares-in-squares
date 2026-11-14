"""Unit tests for the H131 third-generation masked melt."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
NEW159_POC = Path(__file__).resolve().parents[1] / "poc"
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
for p in (NEW159_POC, HIGHD_POC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pocket_seeds as ps  # noqa: E402
import idea1_highd_pilot as i1  # noqa: E402

MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
FINAL_CSV = Path(__file__).resolve().parents[1] / "final.csv"


def _load_norm(n: int):
    df = pd.read_csv(MERGED_CSV)
    sub = df[df["n"] == n].sort_values("square_id")
    return i1._norm(sub)


def test_tightest_contact_clusters():
    x, y, th, L = _load_norm(634)
    pts = ps.tightest_contact_clusters(x, y, th, L, n_pts=6, min_sep=2.6)
    assert len(pts) == 6
    for px, py, stress in pts:
        assert 0.0 < px < L
        assert 0.0 < py < L
        assert stress > 0.0


def test_build_h131_seeds_shapes():
    x, y, th, L = _load_norm(459)
    rng = np.random.default_rng(131100 + 459)
    q_list, m_list, ls_list, labels = ps.build_h131_seeds(
        459, x, y, th, L, rng, L_expand=0.042
    )
    assert len(q_list) == 30
    assert len(labels) == 30
    assert sum(lbl.startswith("mpk_void_") for lbl in labels) == 12
    assert sum(lbl.startswith("mpk_hot_") for lbl in labels) == 10
    assert sum(lbl.startswith("mpk_seam_") for lbl in labels) == 8
    assert all(len(q) == 3 * 459 for q in q_list)
    assert all(len(m) == 3 * 459 for m in m_list)
    assert all(abs(ls[0] - (L + 0.042)) < 1e-12 for ls in ls_list)
    assert all(len(ls) == 5 for ls in ls_list)


def test_final_csv_schema_if_present():
    if not FINAL_CSV.exists():
        return
    spec = importlib.util.spec_from_file_location("check_final_csv", REPO_ROOT / ".github/scripts/check_final_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.validate(FINAL_CSV.read_text(encoding="utf-8"))
