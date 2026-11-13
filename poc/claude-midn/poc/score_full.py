#!/usr/bin/env python3
"""Official literal + pandas-roundtrip score of a full submission CSV.

usage: score_full.py SUBMISSION.csv OUT_PREFIX [JOBS]

Every puzzle goes through the unchanged official per-puzzle evaluator
(`squares.core.evaluate` -> `_official_metric._evaluate_puzzle_worker`), exactly
as `core.score` does; the only difference is the pool size (the bundled metric
hard-codes 4 workers, ~10 min per transport on this machine).  The total is
summed in sorted-n order like the official `_evaluate_puzzles_parallel`.
The pandas transport is an actual `pd.read_csv(path).to_csv(index=False)`.

Writes OUT_PREFIX-literal.json and OUT_PREFIX-pandas.json.
"""
import json
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
from squares import core  # noqa: E402
from squares import _official_metric as metric  # noqa: E402


def _eval(item):
    return metric._evaluate_puzzle_worker(item)


def score(puzzles, jobs):
    work = sorted(puzzles.items(), key=lambda kv: -kv[0])
    with mp.Pool(jobs) as pool:
        results = pool.map(_eval, work, chunksize=1)
    seen = {r[0] for r in results}
    for n in range(1, 1001):
        if n not in seen:
            results.append((n, metric.infeasible_score(n), None, "missing from submission"))
    results.sort(key=lambda r: r[0])
    return dict(
        score=sum(r[1] for r in results), max_n=1000,
        accepted=sum(r[3] == "ok" for r in results),
        missing=sum(r[3] == "missing from submission" for r in results),
        rejected=sum(r[3] not in ("ok", "missing from submission") for r in results),
        results=[dict(n=n, score=v, L=s, reason=why) for n, v, s, why in results],
    )


def score_both(src, jobs):
    import pandas as pd
    lit = score(core.read_submission(src), jobs)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as h:
        tmp = Path(h.name)
    try:
        pd.read_csv(src).to_csv(tmp, index=False)
        pan = score(core.read_submission(tmp), jobs)
    finally:
        tmp.unlink(missing_ok=True)
    return lit, pan


def main():
    src, prefix = Path(sys.argv[1]), sys.argv[2]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    lit, pan = score_both(src, jobs)
    for tag, rep in (("literal", lit), ("pandas", pan)):
        Path(f"{prefix}-{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")
        print(f"{tag}: score={rep['score']!r} accepted={rep['accepted']} "
              f"missing={rep['missing']} rejected={rep['rejected']}", flush=True)


if __name__ == "__main__":
    main()
