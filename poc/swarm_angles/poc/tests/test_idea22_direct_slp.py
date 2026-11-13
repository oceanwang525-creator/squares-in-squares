"""Unit tests for Idea 22 direct multi-margin joint_xytheta_slp."""

from pathlib import Path
import sys

import numpy as np

POC_DIR = Path(__file__).resolve().parents[1]
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

from idea22_direct_slp import direct_slp_multimargin  # noqa: E402


def test_direct_slp_multimargin_tightens_n5_exact() -> None:
    """Verify direct_slp_multimargin recovers tight n=5 optimal packing and passes official verify."""
    s2 = 1.0 / np.sqrt(2.0)
    L_opt = 2.0 + s2
    x = np.array([0.5, L_opt - 0.5 + 0.015, 0.5, L_opt - 0.5 + 0.015, 0.5 * L_opt + 0.005])
    y = np.array([0.5, 0.5, L_opt - 0.5 + 0.015, L_opt - 0.5 + 0.015, 0.5 * L_opt + 0.005])
    th = np.array([0.0, 0.0, 0.0, 0.0, 0.25 * np.pi + 0.01])
    ver, L_cert = direct_slp_multimargin(5, x, y, th, rounds=10)
    assert ver["ok"] is True
    assert abs(L_cert - L_opt) < 1e-5
