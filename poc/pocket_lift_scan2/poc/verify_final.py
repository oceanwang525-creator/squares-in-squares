"""Re-certify final.csv from its own bytes.

`final.csv` stores degrees and `s`-prefixed decimals.  This reads the file,
converts deg -> rad at 80 digits, and runs the official metric on the literal
strings *and* through a real pandas read_csv/to_csv transport, comparing against
a fresh official side table (`data/report.json`, built by
`poc/faultline_dp2/poc/mkreport.py` from `poc/pell_corrugation/data/border-scan/merged.csv`).

usage: verify_final.py [final.csv] [report.json]
"""
import collections
import csv
import json
import sys

import mpmath as mp

sys.path.insert(0, 'poc/baseline/poc/src')
from squares.core import certify, kaggle_roundtrip


def main():
    final = sys.argv[1] if len(sys.argv) > 1 else 'poc/pocket_lift_scan2/final.csv'
    report = sys.argv[2] if len(sys.argv) > 2 else 'poc/pocket_lift_scan2/data/report.json'
    L = {r['n']: mp.mpf(str(r['L'])) for r in json.load(open(report))['results']}
    rows = collections.defaultdict(list)
    with open(final) as f:
        for r in csv.DictReader(f):
            n = int(r['id'].split('_')[0])
            sid = r['id'].split('_')[1]
            rows[n].append((sid, r['x'].lstrip('s'), r['y'].lstrip('s'), r['deg'].lstrip('s')))
    print(f'{final}: {len(rows)} groups, {sum(len(v) for v in rows.values())} rows')
    tot = mp.mpf(0)
    ok_all = True
    for n in sorted(rows):
        if len(rows[n]) != n:
            print(f'  n={n}: WRONG COUNT {len(rows[n])}')
            ok_all = False
            continue
        with mp.workdps(80):
            r = [dict(square_id=sid, x=x, y=y, angle=mp.nstr(mp.mpf(d) * mp.pi / 180, 60))
                 for sid, x, y, d in sorted(rows[n], key=lambda t: int(t[0]))]
        L1 = certify(n, r)
        L2 = certify(n, kaggle_roundtrip({n: r})[n])
        d1, d2 = mp.mpf(L1) - L[n], mp.mpf(L2) - L[n]
        win = d1 < mp.mpf('-1e-9') and d2 < mp.mpf('-1e-9')
        ok_all &= win
        gain = min(d1, d2) / mp.sqrt(n)
        tot += gain
        print(f'  n={n}: incumbent {mp.nstr(L[n], 20)}  literal {mp.nstr(mp.mpf(L1), 20)} / '
              f'pandas {mp.nstr(mp.mpf(L2), 20)}  dL {mp.nstr(d1, 8)} / {mp.nstr(d2, 8)}  '
              f'{"STRICT WIN" if win else "FAIL"}  gain {mp.nstr(gain, 8)}')
    print(f'total score gain from the final.csv bytes: {mp.nstr(tot, 8)}')
    print('ALL STRICT WINS' if ok_all else 'NOT ALL STRICT')


if __name__ == '__main__':
    main()
