"""Verify final.csv from its own bytes: degrees -> radians at 60+ digits, then the official
metric in BOTH transports, compared with the incumbent L for that n.

usage: verify_final.py [../final.csv]
"""
import csv, sys
from decimal import Decimal
import mpmath as mp

import common


def main(path='../final.csv'):
    groups = {}
    for r in csv.DictReader(open(path)):
        n = int(r['id'].split('_')[0])
        sid = r['id'].split('_')[1]
        groups.setdefault(n, []).append(dict(
            n=str(n), square_id=sid,
            x=r['x'].lstrip('s'), y=r['y'].lstrip('s'),
            angle=mp.nstr(mp.mpf(r['deg'].lstrip('s')) * mp.pi / 180, 80)))
    if not groups:
        print('final.csv is header-only: nothing to verify')
        return
    ours = common.ours_L()
    tot = 0.0
    for n, rows in sorted(groups.items()):
        rows.sort(key=lambda r: int(r['square_id']))
        n2, L1, L2, why = common.certify(n, rows)
        if L1 is None or L2 is None:
            print(f'  n={n}: FAILED ({why})')
            continue
        d1 = ours[n] - float(L1); d2 = ours[n] - float(L2)
        import math
        tot += (d1 + d2) / 2 / math.sqrt(n)
        print(f'  n={n}: {len(rows)} squares, ours {ours[n]:.12f} -> literal {L1} / pandas {L2}'
              f'  dL {d1:+.3e} / {d2:+.3e}  {"STRICT WIN" if min(d1, d2) > 1e-9 else "NO WIN"}')
    print(f'total score gain from final.csv bytes: {tot:.4e}')


if __name__ == '__main__':
    main(*sys.argv[1:])
