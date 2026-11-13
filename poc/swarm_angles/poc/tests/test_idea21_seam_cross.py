"""Unit tests for Idea 21 vectorized fine interior cross-cuts and gentle seam relaxation."""

from pathlib import Path
import sys

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

from idea21_seam_cross import _cross_cut_seeds, find_fine_cross_cuts  # noqa: E402


def test_find_fine_cross_cuts_exact_count_and_low_pen() -> None:
    """Verify vectorized cross-cut search locates exact-count cut on synthetic 5x5 donor."""
    xs, ys = np.meshgrid(np.arange(5) + 0.5, np.arange(5) + 0.5, indexing="ij")
    mats = {25: (xs.ravel().astype(np.float64), ys.ravel().astype(np.float64), np.zeros(25), 5.0)}
    cuts = find_fine_cross_cuts(16, 4.05, mats, step=0.25, max_cuts=4)
    assert len(cuts) >= 1
    assert cuts[0]["L"] == 4.0
    assert len(cuts[0]["x"]) == 16
    assert cuts[0]["pen0"] < 1e-9


def test_cross_cut_seeds_shape_and_schedule() -> None:
    """Verify _cross_cut_seeds produces 3*n dimensional free vectors and monotonic L_sched."""
    xs, ys = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
    cut = {
        "src": "xc25_c1_20_20", "n2": 25, "c": 1, "xc": 2.0, "yc": 2.0,
        "L": 4.0, "surplus": 0, "tilt_cut": 0, "pen0": 0.0,
        "x": xs.ravel().astype(np.float64), "y": ys.ravel().astype(np.float64), "th": np.zeros(16),
    }
    seeds, lsc = _cross_cut_seeds(cut, 4.02, 2, np.random.default_rng(21))
    assert len(seeds) == 2
    assert seeds[0].shape == (48,)
    assert lsc[0] >= lsc[-1]
