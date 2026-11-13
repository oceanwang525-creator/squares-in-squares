"""Export and certify complete winning groups in the team's PR CSV format."""
import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import mpmath as mp
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'poc/baseline/poc/src'))
from squares._official_metric import evaluate_single_puzzle


def encode(rows):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(('id', 'x', 'y', 'deg'))
    writer.writerows(rows)
    return stream.getvalue()


def convert_angle(value, to_degrees):
    with mp.workdps(95):
        angle = mp.mpf(value)
        converted = angle * 180 / mp.pi if to_degrees else angle * mp.pi / 180
        return mp.nstr(converted, 75)


def decode(text):
    groups = defaultdict(list)
    for row in csv.DictReader(io.StringIO(text)):
        n, sid = map(int, row['id'].split('_'))
        groups[n].append(dict(square_id=sid, x=row['x'], y=row['y'],
                              angle=convert_angle(row['deg'].removeprefix('s'), False)))
    return groups


def transport(text):
    return pd.read_csv(io.StringIO(text)).to_csv(index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('incumbent', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if pd.__version__ != '3.1.0':
        raise ValueError('pandas 3.1.0 required')
    data = Path(__file__).resolve().parents[1] / 'data'
    prior = json.loads((data / 'certification.json').read_text())
    digest = hashlib.sha256(args.incumbent.read_bytes()).hexdigest()
    if digest != prior['baseline_sha256']:
        raise ValueError('Incumbent changed. Re-certify against its current rows before exporting.')
    with (data / 'improved_rows.csv').open(newline='') as f:
        rows = [(f"{int(r['n']):03}_{int(r['square_id'])}", r['x'], r['y'],
                 's' + convert_angle(r['angle'], True)) for r in csv.DictReader(f)]
    candidate = encode(rows)
    literal, pandas = decode(candidate), decode(transport(candidate))
    checks, wins = {}, set()
    for n, group in literal.items():
        by_mode = {}
        for mode, inputs in [('literal', group), ('pandas', pandas[n])]:
            _, score, side, reason = evaluate_single_puzzle(n, inputs)
            by_mode[mode] = dict(score=score, L=side, reason=reason)
        checks[n] = by_mode
        incumbent = prior['wins'][str(n)]['incumbent']
        if all(r['reason'] == 'ok' and Decimal(r['L']) < Decimal(incumbent[m]['L'])
               for m, r in by_mode.items()):
            wins.add(n)
    if not wins:
        raise ValueError('No strict improvements survived the final.csv format')
    final = encode(r for r in rows if int(r[0].split('_')[0]) in wins)
    roundtrip = decode(transport(final))
    if dict(roundtrip) != {n: pandas[n] for n in wins}:
        raise ValueError('Filtered CSV changed the pandas inputs')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'final.csv').write_text(final, encoding='utf-8', newline='')
    report = dict(pandas_version=pd.__version__, baseline_sha256=digest,
                  final_csv_sha256=hashlib.sha256(final.encode()).hexdigest(),
                  verified=sorted(wins), results=checks)
    (args.output / 'final-transport-check.json').write_text(json.dumps(report, indent=2) + '\n',
                                                           encoding='utf-8')
    print(f'Accepted and strictly better in both modes: {len(wins)}/{len(checks)}')


if __name__ == '__main__':
    main()
