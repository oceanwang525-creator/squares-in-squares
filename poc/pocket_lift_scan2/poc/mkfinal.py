"""Assemble + re-certify final.csv from every group in data/built/n*.json.

Every group is certified against the fresh-main 50-digit side table
(`data/report.json`) in BOTH transports -- literal decimal strings and a real
pandas read_csv/to_csv round trip -- and only strict wins (dL <= -1e-9 in both)
are written.

usage: mkfinal.py [--report R] [--built DIR] [--final final.csv]
"""
import argparse
import glob
import json
import os
import sys

import mpmath as mp

sys.path.insert(0, 'poc/baseline/poc/src')
from squares.core import certify, kaggle_roundtrip

sys.path.insert(0, 'poc/pocket_lift_scan2/poc')
from build_cert import write_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='poc/pocket_lift_scan2/data/report.json')
    ap.add_argument('--built', nargs='+',
                    default=['poc/pocket_lift_scan2/data/built',
                             'poc/pocket_lift_scan2/data/built_prop'])
    ap.add_argument('--final', default='poc/pocket_lift_scan2/final.csv')
    ap.add_argument('--only', type=int, nargs='+')
    a = ap.parse_args()
    L = {r['n']: mp.mpf(str(r['L'])) for r in json.load(open(a.report))['results']}
    groups, tot, bestdL = {}, mp.mpf(0), {}
    files = []
    for d in a.built:
        files += glob.glob(os.path.join(d, 'n*.json'))
    files.sort(key=lambda f: int(os.path.basename(f)[1:-5]))
    for f in files:
        n = int(os.path.basename(f)[1:-5])
        if a.only and n not in a.only:
            continue
        rows = json.load(open(f))
        if len(rows) != n:
            print(f'n={n}: {len(rows)} rows, skip'); continue
        try:
            s1 = certify(n, rows)
            s2 = certify(n, kaggle_roundtrip({n: rows})[n])
        except Exception as ex:
            print(f'n={n}: certify raised {ex}'); continue
        inc = L[n]
        d1, d2 = mp.mpf(s1) - inc, mp.mpf(s2) - inc
        win = d1 < mp.mpf('-1e-9') and d2 < mp.mpf('-1e-9')
        print(f'  n={n}: inc {mp.nstr(inc, 22)} lit {mp.nstr(mp.mpf(s1), 22)} '
              f'pan {mp.nstr(mp.mpf(s2), 22)}  dL {mp.nstr(d1, 8)} / {mp.nstr(d2, 8)}  '
              f'{"STRICT WIN" if win else "REJECT"}  gain {mp.nstr(min(d1, d2) / mp.sqrt(n), 6)}')
        if win:
            # several builders may hit the same n: keep the smallest side
            if n in bestdL and min(d1, d2) >= bestdL[n]:
                continue
            bestdL[n] = min(d1, d2)
            groups[n] = rows
    for n in groups:
        tot += bestdL[n] / mp.sqrt(n)
    write_final(a.final, groups)
    print(f'wrote {a.final}: {len(groups)} groups {sorted(groups)}, '
          f'{sum(len(v) for v in groups.values())} rows, total gain {mp.nstr(tot, 8)}')


if __name__ == '__main__':
    main()
