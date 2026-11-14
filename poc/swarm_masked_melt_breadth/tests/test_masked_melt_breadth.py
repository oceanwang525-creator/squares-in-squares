"""Unit tests for the H132 breadth-first masked-melt gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
BREADTH_POC = Path(__file__).resolve().parents[1] / "poc"
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
for p in (BREADTH_POC, HIGHD_POC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import idea1_highd_pilot as i1  # noqa: E402

_POCKET_SPEC = importlib.util.spec_from_file_location(
    "h132_masked_melt_breadth_pocket_seeds",
    BREADTH_POC / "pocket_seeds.py",
)
if _POCKET_SPEC is None or _POCKET_SPEC.loader is None:
    raise RuntimeError("cannot load H132 pocket_seeds.py")
ps = importlib.util.module_from_spec(_POCKET_SPEC)
sys.modules[_POCKET_SPEC.name] = ps
_POCKET_SPEC.loader.exec_module(ps)

MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
FINAL_CSV = Path(__file__).resolve().parents[1] / "final.csv"


def _load_norm(n: int):
    df = pd.read_csv(MERGED_CSV)
    sub = df[df["n"] == n].sort_values("square_id")
    return i1._norm(sub)


def test_tightest_contact_clusters():
    x, y, th, L = _load_norm(367)
    pts = ps.tightest_contact_clusters(x, y, th, L, n_pts=6, min_sep=2.6)
    assert len(pts) == 6
    for px, py, stress in pts:
        assert 0.0 < px < L
        assert 0.0 < py < L
        assert stress > 0.0


def test_build_h13_seeds_shapes():
    x, y, th, L = _load_norm(367)
    preset = ps.preset_for(367)
    rng = np.random.default_rng(preset.seed)
    q_list, m_list, ls_list, labels = ps.build_h13_seeds(367, x, y, th, L, rng)
    assert len(q_list) == 24
    assert len(labels) == 24
    assert sum(lbl.startswith("mpk_void_") for lbl in labels) == 10
    assert sum(lbl.startswith("mpk_hot_") for lbl in labels) == 8
    assert sum(lbl.startswith("mpk_seam_") for lbl in labels) == 6
    assert all(len(q) == 3 * 367 for q in q_list)
    assert all(len(m) == 3 * 367 for m in m_list)
    assert all(abs(ls[0] - (L + preset.l_expand)) < 1e-12 for ls in ls_list)
    assert all(len(ls) == 4 for ls in ls_list)


def test_presets_are_target_specific():
    family = (442, 485, 528, 529, 530, 531, 532, 533, 576, 577, 578, 785, 786, 842, 901, 962)
    seeds = {ps.preset_for(n).seed for n in family}
    assert len(seeds) == len(family)
    assert ps.preset_for(530).radius_min == ps.preset_for(531).radius_min
    assert ps.preset_for(842).l_expand == ps.preset_for(962).l_expand


def test_expansion_sweep_builds_48_starts():
    x, y, th, L = _load_norm(842)
    preset = ps.preset_for(842)
    rng = np.random.default_rng(preset.seed + 84200)
    q_list, m_list, ls_list, labels = ps.build_h13_seeds(
        842,
        x,
        y,
        th,
        L,
        rng,
        waves=2,
        l_expand_values=(0.044, 0.050),
    )
    assert len(q_list) == 48
    assert len(m_list) == 48
    assert any(abs(ls[0] - (L + 0.044)) < 1e-12 for ls in ls_list)
    assert any(abs(ls[0] - (L + 0.050)) < 1e-12 for ls in ls_list)
    assert any("w0" in label for label in labels)
    assert any("w1" in label for label in labels)


def test_final_csv_schema_if_present():
    if not FINAL_CSV.exists():
        return
    spec = importlib.util.spec_from_file_location("check_final_csv", REPO_ROOT / ".github/scripts/check_final_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.validate(FINAL_CSV.read_text(encoding="utf-8"))
