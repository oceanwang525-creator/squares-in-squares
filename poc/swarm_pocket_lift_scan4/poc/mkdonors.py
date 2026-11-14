"""Donor CSV for the iteration rounds: fresh main + every built group.

Built groups are appended under fresh group numbers n = 2000 + i so they never
collide with an incumbent n; `scan_pockets.py --rows 2000,2001,...` then treats
them as ordinary donors.

usage: mkdonors.py [--merged M] [--built DIR] [--out data/donors.csv]
"""
import argparse
import csv
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged', default='poc/swarm_pocket_lift_scan4/data/upstream-merged.csv')
    ap.add_argument('--built', default='poc/swarm_pocket_lift_scan4/data/built')
    ap.add_argument('--out', default='poc/swarm_pocket_lift_scan4/data/donors.csv')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.built, 'n*.json')),
                   key=lambda f: int(os.path.basename(f)[1:-5]))
    nid = 0
    with open(a.out, 'w', newline='') as o:
        w = csv.writer(o, lineterminator='\n')
        w.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
        with open(a.merged) as f:
            for r in csv.DictReader(f):
                w.writerow([nid, r['n'], r['square_id'], r['x'], r['y'], r['angle']])
                nid += 1
        rows2 = []
        for i, f in enumerate(files):
            rows = json.load(open(f))
            n2 = 2000 + i
            for r in rows:
                w.writerow([nid, n2, r['square_id'], r['x'], r['y'], r['angle']])
                nid += 1
            rows2.append((n2, int(os.path.basename(f)[1:-5])))
        print('extra donor groups:', rows2)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
