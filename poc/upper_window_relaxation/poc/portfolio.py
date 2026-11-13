import csv
import itertools
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(os.environ.get('SQUARE_PACKING_REPOSITORY', Path(__file__).resolve().parents[3]))
OUT = Path(os.environ.get('SQUARE_PACKING_OUTPUT',Path(__file__).resolve().parent))
OUT.mkdir(parents=True,exist_ok=True)
if 'SQUARE_PACKING_SQK' in os.environ:
    sys.path.insert(0, os.environ['SQUARE_PACKING_SQK'])
sys.path.insert(0,str(ROOT/'poc/baseline/poc/src'))

import numpy as np
from squares._official_metric import evaluate_single_puzzle

START, END = 167, 200
TEAM = ROOT / 'poc/pell_corrugation/data/border-scan/merged.csv'
FIELDS = ['id', 'n', 'square_id', 'x', 'y', 'angle']


def load(path, lo=START, hi=END, full=True):
    groups = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as inp:
        reader = csv.DictReader(inp)
        if reader.fieldnames != FIELDS:
            return {}
        source = itertools.islice(reader, lo * (lo - 1) // 2, hi * (hi + 1) // 2) if full else reader
        for row in source:
            n = int(row['n'])
            if lo <= n <= hi:
                groups[n].append(row)
    return {n: rows for n, rows in groups.items() if len(rows) == n}


def points(rows):
    return np.array([[float(row[k]) for k in ('x', 'y', 'angle')] for row in rows])


def to_rows(n, p):
    return [dict(id=n * (n - 1) // 2 + sid, n=n, square_id=sid,
                 x=repr(x), y=repr(y), angle=repr(a)) for sid, (x, y, a) in enumerate(p.tolist())]


def write(path, groups):
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for n in sorted(groups):
            writer.writerows(groups[n])
