"""Shared helpers: read the incumbent, float L per n, official two-transport certification,
final.csv writer (the format the coordinator mandates).

Official metric = poc/baseline/poc/src/squares (installed editable package).
"""
import csv, io, json, math, os, sys
from decimal import Decimal

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
INCUMBENT = os.path.join(ROOT, 'poc/pell_corrugation/data/border-scan/merged.csv')
sys.path.insert(0, os.path.join(ROOT, 'poc/baseline/poc/src'))
from squares._official_metric import evaluate_single_puzzle  # noqa: E402


def load_groups(path=INCUMBENT, ns=None):
    by = {}
    for r in csv.DictReader(open(path)):
        n = int(r['n'])
        if ns is not None and n not in ns:
            continue
        by.setdefault(n, []).append(r)
    for n in by:
        by[n].sort(key=lambda r: int(r['square_id']))
    return by


def float_L(rows):
    x = np.array([float(r['x']) for r in rows]); y = np.array([float(r['y']) for r in rows])
    a = np.array([float(r['angle']) for r in rows])
    e = 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))
    return float(max((x + e).max() - (x - e).min(), (y + e).max() - (y - e).min()))


def ours_L(path=INCUMBENT, cache='../data/oursL.json'):
    if os.path.exists(cache):
        return {int(k): float(v) for k, v in json.load(open(cache)).items()}
    g = load_groups(path)
    L = {n: float_L(rows) for n, rows in g.items()}
    json.dump({str(k): v for k, v in L.items()}, open(cache, 'w'))
    return L


def pandas_transport(rows):
    import pandas as pd
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=['id', 'n', 'square_id', 'x', 'y', 'angle'])
    w.writeheader()
    for i, r in enumerate(rows):
        w.writerow({'id': i, **{k: r[k] for k in ('n', 'square_id', 'x', 'y', 'angle')}})
    df = pd.read_csv(io.StringIO(buf.getvalue()))
    out = df.to_csv(index=False)
    return [{k: r[k] for k in ('n', 'square_id', 'x', 'y', 'angle')}
            for r in csv.DictReader(io.StringIO(out))]


def certify(n, rows):
    """Official metric on the literal strings and after a real pandas read_csv/to_csv round trip."""
    if len(rows) != n:
        return n, None, None, f'count {len(rows)}'
    _, _, L1, why1 = evaluate_single_puzzle(n, rows)
    if L1 is None:
        return n, None, None, 'literal:' + str(why1)
    t = pandas_transport(rows)
    _, _, L2, why2 = evaluate_single_puzzle(n, t)
    if L2 is None:
        return n, None, None, 'pandas:' + str(why2)
    return n, L1, L2, 'ok'


def rows_from_P(n, P, eps=2e-12, s_prefix=False):
    """Unit-square centres/angles -> submission rows. `eps` dilates so touching squares certify."""
    P = np.asarray(P, float).copy()
    P[:, 0] *= (1 + eps); P[:, 1] *= (1 + eps)
    out = []
    for i, (x, y, t) in enumerate(P):
        out.append({'n': str(n), 'square_id': str(i),
                    'x': ('s' if s_prefix else '') + repr(float(x)),
                    'y': ('s' if s_prefix else '') + repr(float(y)),
                    'angle': ('s' if s_prefix else '') + repr(float(t))})
    return out


def write_final(path, groups):
    """groups: {n: [dict(square_id, x, y, angle)]}; header id,x,y,deg; decimals s-prefixed."""
    import mpmath as mp
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['id', 'x', 'y', 'deg'])
        with mp.workdps(95):
            for n, rows in sorted(groups.items()):
                for r in sorted(rows, key=lambda r: int(r['square_id'])):
                    deg = mp.nstr(mp.mpf(str(r['angle'])) * 180 / mp.pi, 75)
                    w.writerow([f'{n:03}_{r["square_id"]}', 's' + str(r['x']), 's' + str(r['y']), 's' + deg])


if __name__ == '__main__':
    L = ours_L()
    print('ours L for', len(L), 'n; e.g. 791 =', L[791], ' 901 =', L[901])
