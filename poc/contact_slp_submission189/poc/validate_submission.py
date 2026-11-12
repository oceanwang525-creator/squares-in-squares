"""Certify the complete submission and compare measured L to pinned upstream receipts."""
import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'baseline' / 'poc' / 'src'))
from squares import core
from squares import _official_metric as metric


def evaluate_job(job):
    mode, n, rows = job
    _, score, length, reason = core.evaluate(n, rows)
    return mode, {'n': n, 'score': score, 'L': length, 'reason': reason}


def validate_both(literal, transported):
    counts = {'literal': 0, 'pandas': 0}
    reports = {'literal': [], 'pandas': []}
    tasks = chain.from_iterable(
        (('literal', n, literal[n]), ('pandas', n, transported[n]))
        for n in range(1, 1001)
    )
    with ProcessPoolExecutor(max_workers=6) as pool:
        for mode, result in pool.map(evaluate_job, tasks, chunksize=4):
            reports[mode].append(result)
            counts[mode] += 1
            if counts[mode] % 200 == 0:
                print(f'{mode}: {counts[mode]}/1000 evaluated', flush=True)
    summary = {}
    for mode, results in reports.items():
        accepted = sum(row['reason'] == 'ok' for row in results)
        value = sum(row['score'] for row in results)
        summary[mode] = {
            'score': value, 'max_n': 1000, 'accepted': accepted, 'missing': 0,
            'rejected': 1000 - accepted, 'results': results,
        }
        print(f'{mode}: {accepted}/1000 accepted, {1000 - accepted} rejected, score {value:.15f}', flush=True)
        if accepted != 1000:
            print([row for row in results if row['reason'] != 'ok'], flush=True)
    return summary['literal'], summary['pandas']


def compare(literal, transported, reference, reference_transport, output):
    columns = ('n', 'upstream_L', 'candidate_L', 'literal_decrease',
               'upstream_pandas_L', 'candidate_pandas_L', 'pandas_decrease')
    wins = 0
    with output.open('w', newline='', encoding='utf-8') as handle, localcontext() as ctx:
        ctx.prec = 80
        writer = csv.writer(handle)
        writer.writerow(columns)
        for old, new, old_pd, new_pd in zip(reference['results'], literal['results'],
                                             reference_transport['results'], transported['results'], strict=True):
            if not old['n'] == new['n'] == old_pd['n'] == new_pd['n']:
                raise ValueError('Official reports disagree on puzzle order')
            gain = Decimal(old['L']) - Decimal(new['L'])
            pd_gain = Decimal(old_pd['L']) - Decimal(new_pd['L'])
            if gain > 0 or pd_gain > 0:
                writer.writerow([new['n'], old['L'], new['L'], str(gain),
                                 old_pd['L'], new_pd['L'], str(pd_gain)])
                wins += 1
    return wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submission', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.submission)
    if len(frame) != 500_500 or set(frame['n']) != set(range(1, 1001)):
        parser.error('Expected 500,500 rows covering all 1,000 puzzle sizes')
    if not frame['id'].equals(pd.Series(range(500_500))):
        parser.error('Expected consecutive integer IDs')
    transported = metric._parse_submission_csv(frame.to_csv(index=False))
    literal, transported = validate_both(core.read_submission(args.submission), transported)
    for name, report in [('literal', literal), ('pandas', transported)]:
        (args.out / f'{name}.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if any(report['accepted'] != 1000 for report in (literal, transported)):
        raise SystemExit('At least one layout did not pass official certification')
    reference = PROJECT / 'pell_corrugation' / 'data' / 'border-scan'
    old = json.loads((reference / 'merged-literal-official.json').read_text())
    old_pd = json.loads((reference / 'merged-pandas-official.json').read_text())
    wins = compare(literal, transported, old, old_pd, args.out / 'lower_sides.csv')
    summary = {
        'input_sha256': hashlib.sha256(args.submission.read_bytes()).hexdigest(),
        'pandas': pd.__version__,
        'csv_rows': len(frame),
        'lower_side_rows': wins,
        'literal': {k: literal[k] for k in ('accepted', 'missing', 'rejected', 'score')},
        'transport': {k: transported[k] for k in ('accepted', 'missing', 'rejected', 'score')},
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()

