"""Validate midrange rows with the official interval metric and pandas transport."""
import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'poc/baseline/poc/src'))
from squares._official_metric import evaluate_single_puzzle

FIELDS = ['id', 'n', 'square_id', 'x', 'y', 'angle']
MODES = ('literal', 'pandas')


def read(handle):
    reader = csv.DictReader(handle)
    if reader.fieldnames != FIELDS:
        raise ValueError(f'Unexpected columns: {reader.fieldnames}')
    groups = defaultdict(list)
    for row in reader:
        n = int(row['n'])
        groups[n].append(row)
    for n, rows in groups.items():
        if len(rows) != n or sorted(int(r['square_id']) for r in rows) != list(range(n)):
            raise ValueError(f'Invalid count or square IDs at n={n}')
    return dict(groups)


def serialize(groups):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    for n, rows in sorted(groups.items()):
        writer.writerows(dict(row, id=n*(n-1)//2+i, square_id=i) for i, row in enumerate(rows))
    return out.getvalue()


def transport(text):
    return read(io.StringIO(pd.read_csv(io.StringIO(text)).to_csv(index=False)))


def check(job):
    n, l, p = job
    literal = evaluate_single_puzzle(n, l)
    pandas = literal if l == p else evaluate_single_puzzle(n, p)
    return n, {tag: dict(score=r[1], L=r[2], reason=r[3]) for tag, r in zip(MODES, (literal, pandas))}


def verify(pool, l, p, ns):
    return dict(pool.map(check, ((n, l[n], p[n]) for n in ns), chunksize=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('incumbent', type=Path)
    ap.add_argument('official_reports', type=Path, help='Upstream directory with COMPLETE.json and both final official reports')
    ap.add_argument('output', type=Path)
    ap.add_argument('candidates', nargs='*', type=Path)
    ap.add_argument('--jobs', default=4, type=int)
    args = ap.parse_args()
    if pd.__version__ != '3.1.0':
        raise ValueError(f'Team validation requires pandas 3.1.0; found {pd.__version__}')
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.incumbent.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    upstream = json.loads((args.official_reports/'COMPLETE.json').read_text())
    if digest != upstream['sha256']:
        raise ValueError(f'Incumbent hash does not match the upstream proof: {digest}')
    text = raw.decode('utf-8-sig')
    base = read(io.StringIO(text))
    bp = transport(text)
    expected = set(range(1, 1001))
    if set(base) != expected or set(bp) != expected:
        raise ValueError('The incumbent is not a full 1,000-puzzle portfolio')
    reference = {}
    for mode in MODES:
        rep = json.loads((args.official_reports/f'final-{mode}-official.json').read_text())
        reference[mode] = {r['n']: r for r in rep['results']}
        if set(reference[mode]) != expected or any(r['reason'] != 'ok' for r in rep['results']):
            raise ValueError(f'Incomplete upstream {mode} proof')
    improved, evidence, rejected = {}, {}, []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        checks = verify(pool, base, bp, range(100, 201))
        for n, record in checks.items():
            for mode, rec in record.items():
                ref = reference[mode][n]
                if rec['reason'] != 'ok' or Decimal(rec['L']) != Decimal(ref['L']):
                    raise ValueError(f'Official result disagrees at {n=} {mode=}: {rec} vs {ref}')
        for path in args.candidates:
            with path.open(encoding='utf-8-sig', newline='') as f:
                trial = read(f)
            if not set(trial).issubset(checks):
                raise ValueError(f'Candidate sizes outside 100–200: {path}')
            trial_p = transport(serialize(trial))
            for n, record in verify(pool, trial, trial_p, sorted(trial)).items():
                old = evidence[n]['candidate'] if n in evidence else checks[n]
                if all(record[mode]['reason'] == 'ok' and Decimal(record[mode]['L']) < Decimal(old[mode]['L'])
                       for mode in MODES):
                    improved[n] = trial[n]
                    evidence[n] = dict(source=path.name, incumbent=checks[n], candidate=record)
                else:
                    rejected.append(dict(source=path.name, n=n, results=record))
        scores = {}
        if improved:
            merged = serialize({**base, **improved})
            ml, mp = read(io.StringIO(merged)), transport(merged)
            # Prove unchanged groups have identical metric inputs, both before
            # and after the actual *full-file* pandas roundtrip.
            affected = set(improved)
            for n in expected - affected:
                for old, new in [(base, ml), (bp, mp)]:
                    if old[n] != new[n]:
                        raise ValueError(f'Unchanged group drift at n={n}')
            full_checks = verify(pool, ml, mp, sorted(affected))
            for n, record in full_checks.items():
                for mode, rec in record.items():
                    if rec['reason'] != 'ok' or Decimal(rec['L']) >= Decimal(checks[n][mode]['L']):
                        raise ValueError(f'Merged candidate fails strict improvement at n={n} {mode}: {rec}')
                evidence[n]['candidate'] = record
            (args.output/'improved_rows.csv').write_text(serialize(improved), encoding='utf-8')
            (args.output/'candidate_full.csv').write_text(merged, encoding='utf-8')
        for mode in MODES:
            scores[mode] = dict(
                baseline=math.fsum(v['score'] for v in reference[mode].values()),
                result=math.fsum(evidence[n]['candidate'][mode]['score'] if n in evidence else v['score']
                                 for n, v in reference[mode].items()),
                accepted=1000,
                regression_count=0,
                recomputed_baseline_sizes=len(checks),
            )
    report = dict(baseline_sha256=digest, pandas_version=pd.__version__,
                  verification=scores, wins=evidence, rejected_candidates=rejected)
    (args.output/'certification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
