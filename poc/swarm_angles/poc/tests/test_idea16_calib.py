"""Unit tests for bell-slipped n=50 calibration and Grid4+local HiGHS solver."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[4]
POC_DIR = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(POC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from poc.swarm_angles.poc.idea14_clusters_highs import make_bell_slipped_cluster  # noqa: E402
from poc.swarm_angles.poc.idea16_grid4_local_highs import solve_grid4_local  # noqa: E402


def test_bell_slipped_n50_calibration_exact():
    L50 = 53.0 / 7.0
    tilted = make_bell_slipped_cluster(
        L50, 3, 5, math.degrees(math.atan2(3, 4)), 0.20, 4.0 / 21.0, True, 0.5 * L50, 0.5 * L50
    )
    assert len(tilted) == 16
    tag, fam, tot, nt, nup, st, up_pts = solve_grid4_local(("c50", "bell", L50, tilted, 0.0, 0.0, 3.0))
    assert st == 0
    assert nt == 16
    assert nup == 34
    assert tot == 50
    assert len(up_pts) == 34
