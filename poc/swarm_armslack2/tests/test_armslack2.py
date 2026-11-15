"""Unit tests for poc/swarm_armslack2."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1] / "poc"
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(POC_DIR))

import arm_model as am  # noqa: E402
import armslack2_lib as a2  # noqa: E402
import pslp  # noqa: E402


def _synthetic_pinwheel() -> tuple[np.ndarray, float, int]:
    """Build a synthetic 12x12-ish packing with rigid 0-deg corners and a tilted band."""
    K = 12
    L = float(K) + 0.55
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append((i + 0.5, j + 0.5, 0.0))
            pts.append((L - 0.5 - i, L - 0.5 - j, 0.0))
    for k in range(-3, 4):
        pts.append((6.0 + 0.55 * k, 6.0 - 0.55 * k, 0.42))
    return np.array(pts, dtype=np.float64), L, K


def test_identify_interface_layers():
    a, L, _ = _synthetic_pinwheel()
    info = am.identify_interface_layers(a, layer1_r=2.2, layer2_r=3.5)
    assert int(info["tilted"].sum()) == 7
    assert int(info["layer1"].sum()) > 0
    assert int(info["layer2"].sum()) >= int(info["layer1"].sum())


def test_build_model_liberates_boundary():
    a, L, _ = _synthetic_pinwheel()
    m1 = am.build_model(a, L, act_reach=1.8, layer=1.2, min_comp=4)
    m2 = am.build_model(a, L, act_reach=3.5, layer=1.2, min_comp=4)
    assert m1["reduced"] and m2["reduced"]
    assert m2["block_counts"]["active"] >= m1["block_counts"]["active"]


def test_build_interface_melt_seeds():
    a, L, _ = _synthetic_pinwheel()
    rng = np.random.default_rng(7)
    q_list, m_list, ls_list, labels = a2.build_interface_melt_seeds(a[:, 0], a[:, 1], a[:, 2], L, rng)
    assert len(q_list) > 0
    n = len(a)
    assert all(len(q) == 3 * n for q in q_list)
    assert all(len(m) == 3 * n for m in m_list)
    assert all(len(ls) == 4 for ls in ls_list)
    assert len(labels) == len(q_list)


def test_pslp_squeeze_basic():
    a = np.array([[0.5, 3.5, 0.0], [3.5, 0.5, 0.0], [2.0, 2.0, 0.35]], dtype=np.float64)
    f, Xn, Yn, Tn, v, _ = pslp.squeeze(a[:, 0], a[:, 1], a[:, 2], 3, 3, 1.0, trust0=0.05, max_it=30)
    assert v <= 1e-12
    assert f < 0.9


def test_final_csv_schema_if_present():
    fc = Path(__file__).resolve().parents[1] / "final.csv"
    if not fc.exists():
        return
    spec = importlib.util.spec_from_file_location("check_final_csv", REPO_ROOT / ".github/scripts/check_final_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.validate(fc.read_text(encoding="utf-8"))
