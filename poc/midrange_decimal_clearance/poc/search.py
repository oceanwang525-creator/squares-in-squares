"""Sweep decimal-scale shrink and centering variants for near-tight midrange layouts."""
import argparse
import csv
import io
import json
import math
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, localcontext
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'poc/baseline/poc/src'))
from squares._official_metric import evaluate_single_puzzle

FIELDS = ['id', 'n', 'square_id', 'x', 'y', 'angle']


def serialize(rows):
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return f.getvalue()


def vertices(rows):
    p = np.array([[float(r[k]) for k in FIELDS[-3:]] for r in rows])
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    return p, c, s


def screen(rows):
    p, c, s = vertices(rows)
    i, j = np.triu_indices(len(p), 1)
    dx, dy = (p[j, :2] - p[i, :2]).T
    near = (abs(dx) < 1.42) & (abs(dy) < 1.42)
    i, j, dx, dy = i[near], j[near], dx[near], dy[near]
    proj = np.maximum.reduce([abs(dx*c[i]+dy*s[i]), abs(dy*c[i]-dx*s[i]),
                              abs(dx*c[j]+dy*s[j]), abs(dy*c[j]-dx*s[j])])
    d = p[j, 2] - p[i, 2]
    h = (abs(np.cos(d)) + abs(np.sin(d)) + 1)/2
    return (proj-h).min(initial=0)


def shifted(rows, cx, cy, s):
    def value(raw, ctr):
        return format((Decimal(raw)-ctr)*s, '.32f')
    with localcontext() as context:
        context.prec = 65
        return [dict(r, x=value(r['x'], cx), y=value(r['y'], cy)) for r in rows]


def check(job):
    n, rows, old_l, old_p = job
    p, c, s = vertices(rows)
    h = (abs(c)+abs(s))[:, None]/2
    middle = tuple(map(lambda x: Decimal(str(x)), np.round(((p[:, :2]+h).max(0)+(p[:, :2]-h).min(0))/2, 3)))
    # The incumbent coordinates normally include tiny multiplicative safety
    # padding. Recenter at several fractional offsets to change float rounding.
    centers = [(Decimal(0), Decimal(0)), middle, *[
        (middle[0]+Decimal(x), middle[1]+Decimal(y))
        for x, y in [('.01','.03'), ('.23','-.17'), ('-.13','.31'), ('-.34','-.11'), ('.111','.333')]
    ]]
    certs, screens = 0, 0
    for delta in ('8e-15', '4e-15', '2e-15', '1.4e-15', '1e-15', '6e-16', '3e-16', '1e-16'):
        scale = 1-Decimal(delta)
        literal = None
        for cx, cy in centers:
            trial = shifted(rows, cx, cy, scale)
            if literal is None:
                literal = evaluate_single_puzzle(n, trial)
                certs += 1
                if literal[3] != 'ok' or Decimal(literal[2]) >= old_l:
                    break
            raw = pd.read_csv(io.StringIO(serialize(trial))).to_csv(index=False)
            transported = list(csv.DictReader(io.StringIO(raw)))
            screens += 1
            if screen(transported) < -2.4e-15:
                continue
            after = evaluate_single_puzzle(n, transported)
            certs += 1
            if after[3] != 'ok' or Decimal(after[2]) >= old_p:
                continue
            # Repeat the literal check for the actual serialized translation.
            final = evaluate_single_puzzle(n, trial)
            certs += 1
            if final[3] == 'ok' and Decimal(final[2]) < old_l:
                return n, trial, dict(delta=delta, L_literal=final[2], L_pandas=after[2],
                                      old_literal=str(old_l), old_pandas=str(old_p), certs=certs, screens=screens)
    return n, [], dict(certs=certs, screens=screens)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('incumbent', type=Path)
    ap.add_argument('reports', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    if pd.__version__ != '3.1.0':
        raise ValueError('pandas 3.1.0 required')
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [{r['n']: Decimal(r['L']) for r in json.loads((args.reports/f'final-{mode}-official.json').read_text())['results']}
               for mode in ('literal', 'pandas')]
    rows = {}
    with args.incumbent.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            n = int(r['n'])
            if n > 200:
                break
            if 100 <= n:
                rows.setdefault(n, []).append(r)
    # Exact integer square sides and zero clearances cannot benefit from this sweep.
    jobs = [(n, r, reports[0][n], reports[1][n]) for n, r in rows.items()
            if not reports[0][n] == math.ceil(math.sqrt(n)) and screen(r) >= -1e-14]
    out, report = [], {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for n, winner, record in pool.map(check, jobs, chunksize=1):
            report[n] = record
            if winner:
                print(n, record, flush=True)
                out.extend(winner)
            if len(report)%5 == 0 or winner:
                (args.output/'scan.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
                if out:
                    (args.output/'clearance_candidates.csv').write_text(serialize(out), encoding='utf-8')
    (args.output/'scan.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if out:
        (args.output/'clearance_candidates.csv').write_text(serialize(out), encoding='utf-8')
    print('done', len(jobs), 'sizes', sorted({int(r['n']) for r in out}), flush=True)


if __name__ == '__main__':
    main()
