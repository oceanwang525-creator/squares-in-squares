"""Re-score and verify the full 1000-puzzle merged.csv via official literal check + pandas roundtrip."""
from __future__ import annotations

import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "poc" / "baseline" / "poc" / "src"))
import squares.core as sc  # noqa: E402


def main() -> None:
    t0 = time.time()
    csv_path = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    puzzles = sc.read_submission(csv_path)
    lit = sc.score(puzzles, 1000)
    print(
        f"literal: score={lit['score']:.15f}, accepted={lit['accepted']}, "
        f"missing={lit['missing']}, rejected={lit['rejected']} ({time.time() - t0:.1f}s)",
        flush=True,
    )
    t1 = time.time()
    rt = sc.kaggle_roundtrip(puzzles)
    pd_res = sc.score(rt, 1000)
    print(
        f"pandas:  score={pd_res['score']:.15f}, accepted={pd_res['accepted']}, "
        f"missing={pd_res['missing']}, rejected={pd_res['rejected']} ({time.time() - t1:.1f}s)",
        flush=True,
    )
    out = {
        "literal": {k: lit[k] for k in ("score", "accepted", "missing", "rejected")},
        "pandas": {k: pd_res[k] for k in ("score", "accepted", "missing", "rejected")},
        "elapsed_s": round(time.time() - t0, 2),
    }
    out_path = Path(__file__).resolve().parents[1] / "docs" / "incumbent_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
