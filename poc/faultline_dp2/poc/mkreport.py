"""Build the official per-n side table (report.json) for the incumbent.

usage: mkreport.py merged.csv out.json
Writes {"results": [{"n":.., "L": "<50-digit string>", "score":.., "reason":..}]}
"""
import json
import sys
import time

sys.path.insert(0, 'poc/baseline/poc/src')
from squares.core import read_submission, score


def main():
    src, dst = sys.argv[1], sys.argv[2]
    t0 = time.time()
    puzzles = read_submission(src)
    rep = score(puzzles)
    print(f'read {len(puzzles)} groups in {time.time() - t0:.1f}s; '
          f'score {rep["score"]!r} accepted {rep["accepted"]} missing {rep["missing"]} '
          f'rejected {rep["rejected"]}')
    json.dump(rep, open(dst, 'w'))
    print(f'wrote {dst} ({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
