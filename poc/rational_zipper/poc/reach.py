"""Per-target reachability of the existing (donor-window) pocket library.

Uses the shipped pocket_lift_scan2 pipeline modules (faultline/pocket DP is closed
against the current incumbent) to report, for every top-gap target n, the best
composable side  K + f  (staircase composition rule) and the deficit vs the
incumbent and vs the frontier.

usage: reach.py [--pockets ...] [--lib ...] [--report ...]
"""
import argparse
import json
import math
import sys

sys.path.insert(0, 'poc/pocket_lift_scan2/poc')
import dp as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', default='poc/faultline_dp2/data/gadlib.json')
    ap.add_argument('--pockets', default='poc/pocket_lift_scan2/data/pockets_pareto2.json')
    ap.add_argument('--report', default='poc/rational_zipper/data/report.json')
    ap.add_argument('--gap', default='poc/koboshchan_coordinator/data/gap_vs_frontier.csv')
    ap.add_argument('--out', default='poc/rational_zipper/data/reach.json')
    a = ap.parse_args()

    rep = {r['n']: r for r in json.load(open(a.report))['results']}
    base = D.pareto(D.items_from_gadlib(a.lib))
    pk = D.items_from_pockets(a.pockets)
    items = D.pareto(base + pk)
    print(f'items: gadlib-pareto {len(base)}, pockets {len(pk)}, merged-pareto {len(items)}')
    pred = D.dp_levels(items, 31)
    print(f'pred cells: {len(pred)}')

    # frontier / gap table
    rows = []
    with open(a.gap) as f:
        hdr = f.readline()
        for line in f:
            p = line.strip().split(',')
            rows.append(dict(n=int(p[0]), K=int(p[1]), D=int(p[2]), ours=float(p[3]),
                             frontier=float(p[4]), gap_score=float(p[8])))
    rows.sort(key=lambda r: -r['gap_score'])

    out = {}
    print(f"{'n':>4} {'K':>3} {'D':>3} {'L_inc':>14} {'c_inc':>9} {'L_front':>14} "
          f"{'c_fr':>9} {'K+f_best':>12} {'deficit':>10} {'f_best':>9} {'win':>4}")
    for r in rows[:40]:
        n = r['n']
        L = float(rep[n]['L'])
        K = math.isqrt(n)
        got = pred.get(n)
        if got is None:
            s = float('nan'); f = float('nan')
        else:
            Kp, f = got
            s = Kp + f
        win = (s < L - 1e-9) if got else False
        out[n] = dict(K=K, D=n - K * K, L_inc=L, c_inc=L - K, L_front=r['frontier'],
                      c_front=r['frontier'] - K, best_side=(None if got is None else s),
                      best_f=(None if got is None else f), deficit=(None if got is None else s - L),
                      gap_score=r['gap_score'], dp_win=bool(win))
        print(f"{n:4d} {K:3d} {n-K*K:3d} {L:14.9f} {L-K:9.6f} {r['frontier']:14.9f} "
              f"{r['frontier']-K:9.6f} {s:12.9f} {s-L:10.2e} {f:9.6f} {str(win):>4}")
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
