import argparse
import csv
import io
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from squares.core import evaluate

FIELDS = ['id', 'n', 'square_id', 'x', 'y', 'angle']
OUTPUT = Path('candidates')
N_MIN, N_MAX = 100, 133


def arguments(description, extra_seeds=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('incumbent', type=Path, help='Full incumbent CSV')
    if extra_seeds:
        parser.add_argument('extra_seeds', nargs='*', type=Path, help='Additional seed CSVs')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Candidate output directory')
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args()


def groups(path):
    result = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source):
            n = int(row['n'])
            if N_MIN <= n <= N_MAX:
                result[n].append(row)
    return result


def array(rows):
    return np.array([[float(row[key]) for key in FIELDS[3:]] for row in rows])


def convert(base, points):
    return [dict(row, x=repr(float(x)), y=repr(float(y)), angle=repr(float(a)))
            for row, (x, y, a) in zip(base, points, strict=True)]


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def transported(rows):
    data = pd.read_csv(io.StringIO(csv_text(rows)))
    return list(csv.DictReader(io.StringIO(data.to_csv(index=False))))


def certificate(n, rows):
    return tuple(evaluate(n, version) for version in (rows, transported(rows)))


def write_candidate(n, rows, baseline, output, label):
    original = certificate(n, baseline)
    candidate = certificate(n, rows)
    if not all(old[-1] == new[-1] == 'ok' and Decimal(new[2]) < Decimal(old[2])
               for old, new in zip(original, candidate)):
        return None
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{label}-{n}.csv').write_text(csv_text(rows), encoding='utf-8', newline='')
    return candidate
