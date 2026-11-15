#!/usr/bin/env python3
"""H122 (watchxplant) scale-up sweep across m=21..31, r=+6..+18."""
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


def main() -> None:
    if len(sys.argv) <= 1:
        all_targets = [
            m * m + r
            for m in range(31, 20, -1)
            for r in range(6, 19)
            if m * m + r <= 1000
        ]
        sys.argv = [
            sys.argv[0],
            "--targets", ",".join(str(x) for x in all_targets),
            "--top-k", "4",
            "--max-excess", "0.002",
            "--time-limit", "5400",
            "--fast",
            "--resume",
            "--out", str(rg.wx.DOCS_DIR / "scale_summary.json"),
        ]
    rg.main()


if __name__ == "__main__":
    main()
