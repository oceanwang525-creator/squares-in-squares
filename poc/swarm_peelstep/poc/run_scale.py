#!/usr/bin/env python3
"""H109 (peelstep) scale-up sweep across r = +10..+15, m = 25..31."""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_gate as rg  # noqa: E402

# Ordered by highest headroom (f_inc - f_hi) among priority + m=25..31, r=+10..+15 cells
PRIORITY_TARGETS = [
    973, 854, 795, 797, 914, 911, 910, 640, 639, 686,
    799, 798, 638, 687, 637, 690, 971, 855, 851, 976,
    853, 796, 740, 741, 743, 742,
]
FULL_GRID_TARGETS = [
    m * m + r
    for m in range(31, 24, -1)
    for r in range(10, 16)
    if m * m + r <= 1000 and (m * m + r) not in PRIORITY_TARGETS and (m * m + r) != 913
]


def main() -> None:
    if len(sys.argv) <= 1:
        all_targets = PRIORITY_TARGETS + FULL_GRID_TARGETS
        sys.argv = [
            sys.argv[0],
            "--targets", ",".join(str(x) for x in all_targets),
            "--top-k", "4",
            "--max-excess", "0.002",
            "--time-limit", "5400",
            "--out", str(rg.pl.DOCS_DIR / "scale_summary.json"),
        ]
    rg.main()


if __name__ == "__main__":
    main()
