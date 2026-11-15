#!/usr/bin/env python3
"""Tests for H126 (band3ext) pinwheel construction and dual-certified final.csv."""
from __future__ import annotations

import io
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

EXT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXT_ROOT.parents[1]
AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
BAND3_POC = REPO_ROOT / "poc" / "swarm_band3" / "poc"
B8_POC = REPO_ROOT / "poc" / "swarm_b8xplant" / "poc"
GSTEP_POC = REPO_ROOT / "poc" / "swarm_gstep" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"

for _p in (AUTO_UNION_POC, BAND3_POC, B8_POC, GSTEP_POC, LPSLACK_POC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import band3_lib as b3  # noqa: E402
import union as auto_union  # noqa: E402


def test_build_pinwheel_count() -> None:
    """Verify 2-brick pinwheel assembly square count for a synthetic 10x20(g=5) + 20x10(g=5) pair at m=30."""
    # Construct minimal valid brick dicts matching 10x20(g=5) and 20x10(g=5)
    rng = np.random.default_rng(0)
    b1 = {
        "w": 10,
        "h": 20,
        "g": 5,
        "x": rng.uniform(0.5, 9.5, 205),
        "y": rng.uniform(0.5, 19.5, 205),
        "th": np.zeros(205),
        "fx": 0.65,
        "fy": 0.65,
        "f": 0.65,
        "tag": "syn1",
    }
    b2 = {
        "w": 20,
        "h": 10,
        "g": 5,
        "x": rng.uniform(0.5, 19.5, 205),
        "y": rng.uniform(0.5, 9.5, 205),
        "th": np.zeros(205),
        "fx": 0.65,
        "fy": 0.65,
        "f": 0.65,
        "tag": "syn2",
    }
    built = b3.build_pinwheel(30, b1, b2, sf=0.0, max_ov=1)
    assert built is not None
    ax, ay, at, _, _ = built
    assert len(ax) == 910
    assert len(ay) == 910
    assert len(at) == 910


def test_final_csv_dual_certified() -> None:
    """Verify poc/swarm_band3ext/final.csv exists, passes literal==pandas dual certification, and wins match JSON."""
    final_p = EXT_ROOT / "final.csv"
    assert final_p.exists()
    txt = final_p.read_text(encoding="utf-8")
    lines = txt.splitlines()
    assert lines[0] == "id,x,y,deg"
    if len(lines) <= 1:
        return

    lit = auto_union.parse_text(txt)
    pan_txt = pd.read_csv(io.StringIO(txt)).to_csv(index=False)
    pan = auto_union.parse_text(pan_txt)
    assert set(lit.keys()) == set(pan.keys())
    for n in sorted(lit):
        res = auto_union.check((n, lit[n], pan[n]))
        assert res is not None, f"Certification failed for n={n}"
        assert res[0] == res[1], f"Literal vs pandas mismatch for n={n}: {res}"
        assert Decimal(res[0]) > 0
        wp = EXT_ROOT / "data" / f"win_n{n}.json"
        if wp.exists():
            wd = json.loads(wp.read_text(encoding="utf-8"))
            assert Decimal(res[0]) <= max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])) + Decimal("1e-20")
