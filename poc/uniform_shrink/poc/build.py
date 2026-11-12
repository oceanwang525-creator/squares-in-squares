"""Build full CSV: merged.csv with rows replaced for every n in state/best.json.
usage: build.py merged.csv best.json out.csv
"""
import csv, json, sys

merged, bestp, out = sys.argv[1:4]
best = {int(k): v for k, v in json.load(open(bestp)).items()}
by = {}
with open(merged, newline='') as f:
    for r in csv.DictReader(f):
        by.setdefault(int(r['n']), []).append(r)
i = 0
with open(out, 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
    for n in range(1, 1001):
        rows = best[n]['rows'] if n in best else by[n]
        rows = sorted(rows, key=lambda r: int(r['square_id']))
        assert [int(r['square_id']) for r in rows] == list(range(n)), n
        for r in rows:
            w.writerow([i, n, r['square_id'], r['x'].strip(), r['y'].strip(), r['angle'].strip()])
            i += 1
print('rows', i, 'replaced n', len(best))
