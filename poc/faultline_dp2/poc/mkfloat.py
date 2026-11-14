"""Fast float64 per-n side table (same JSON shape as report.json) from a submission CSV.

Used for the DP sweep while the official decimal report is being computed; the
official report wins for every claim that matters.
"""
import json
import sys

sys.path.insert(0, 'poc/faultline_dp2/poc')
from gadlib import load_all


def main():
    src, dst = sys.argv[1], sys.argv[2]
    D = load_all(src)
    res = [dict(n=int(n), L=float(D[n][2])) for n in sorted(D)]
    json.dump(dict(results=res), open(dst, 'w'))
    print(f'{len(res)} groups -> {dst}')


if __name__ == '__main__':
    main()
