"""Unit tests for Idea 20 surplus peels, family-stratified polish selection, and exact repair."""

import math
from pathlib import Path
import sys

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

from idea20_partner_shift import (  # noqa: E402
    _polish_and_repair_worker,
    _select_polish_indices,
    collect_surplus_peels,
    load_verified_disk_wins,
)


def test_collect_surplus_peels_ranks_post_drop_overhang() -> None:
    """Verify collect_surplus_peels sorts by max_oh_rem ascending and maintains len == n + surplus."""
    # Synthetic donor packing for n2=20 inside L=4.90 (4x5 grid centers spaced at 0.98..3.92)
    xs, ys = np.meshgrid(np.linspace(0.5, 4.4, 4), np.linspace(0.5, 4.4, 5), indexing="ij")
    x20 = xs.ravel()
    y20 = ys.ravel()
    th20 = np.zeros(20)
    mats = {20: (x20, y20, th20, 4.90)}
    # Target n=16, L_inc=4.00 -> c=1 peel gives L_box=3.90 < 4.00
    peels = collect_surplus_peels(16, 4.00, mats)
    assert all(len(p["x"]) == 16 + p["surplus"] for p in peels)
    keys = [(p["max_oh_rem"], p["surplus"], p["L"]) for p in peels]
    assert keys == sorted(keys)


def test_select_polish_indices_stratifies_families() -> None:
    """Verify peel and deficit families retain reserved slots even when inc_melt leads feasible list."""
    labels = [f"inc_melt_{i}" for i in range(10)] + ["s377_10_10_s0", "s377_10_10_s1", "x375_2_2_s0", "m339_d1_0"]
    lf = np.array([18.92 + 0.001 * i for i in range(10)] + [1e10, 1e10, 1e10, 1e10])
    po = np.array([0.0] * 10 + [0.002, 0.003, 0.004, 0.005])
    chosen = _select_polish_indices(labels, lf, po, k_keep=10)
    assert len(chosen) == 10
    picked_labels = [labels[i] for i in chosen]
    assert any(lb.startswith("s") for lb in picked_labels)
    assert any(lb.startswith("x") for lb in picked_labels)
    assert any(lb.startswith("m") for lb in picked_labels)
    assert sum(lb.startswith("inc_") for lb in picked_labels) >= 4


def test_polish_and_repair_certifies_exact_packing() -> None:
    """Verify _polish_and_repair_worker repairs slight grid perturbation into exact certified packing."""
    # n=16 unit squares perturbed slightly around an exact 4x4 grid of side 4.0001
    gx, gy = np.meshgrid(np.arange(4) + 0.50002, np.arange(4) + 0.50002, indexing="ij")
    x = gx.ravel().astype(np.float64)
    y = gy.ravel().astype(np.float64)
    th = np.full(16, 1e-4, dtype=np.float64)
    res = _polish_and_repair_worker((0, "test_grid", "inc_melt", 16, x, y, th, 0.0, 0.0, 4.01))
    assert res["verify"]["ok"] is True
    assert res["pen_fin"] == 0.0
    assert float(res["verify"]["L_float"]) < 4.02


def test_load_verified_disk_wins_handles_empty(tmp_path: Path) -> None:
    """Verify load_verified_disk_wins returns empty dict when no win_n*.csv files exist."""
    assert load_verified_disk_wins(tmp_path) == {}
