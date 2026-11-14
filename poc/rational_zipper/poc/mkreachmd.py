"""Render data/reach.json as docs/reach.md."""
import json

r = json.load(open('poc/rational_zipper/data/reach.json'))


def key(n):
    return -r[n]['gap_score']


head = [
    '# Composition reachability of the shipped pocket library',
    '',
    '`poc/reach.py`: the staircase composition DP (`pocket_lift_scan2/poc/dp.py`)',
    'over `gadlib.json` + `pockets_pareto2.json` (591 + 1539 items -> 1699 after',
    'the Pareto merge), measured against fresh main (`data/report.json`,',
    'score 22.15258085488673).  `K+f_best` is the smallest side the composition',
    'rule can reach for that n.',
    '',
    '| n | K | D | L_inc | K+f_best | deficit | dp_win |',
    '|---:|---:|---:|---:|---:|---:|:---|',
]
lines = list(head)
for n in sorted(r, key=key):
    d = r[n]
    bs = d['best_side']
    lines.append('| {n} | {K} | {D} | {L:.9f} | {b} | {df} | {w} |'.format(
        n=n, K=d['K'], D=d['D'], L=d['L_inc'],
        b='-' if bs is None else '%.9f' % bs,
        df='-' if d['deficit'] is None else '%.2e' % d['deficit'],
        w=d['dp_win']))
lines += [
    '',
    'Reading: at the top-gap rows the DP best equals the incumbent to ~1e-13',
    '(those rows *are* compositions of the library) or is worse (492, 974, 975,',
    '852: upstream uses geometry outside this family).  No strict win survives a',
    '1e-9 threshold; the only negative deficit printed (n=854, -1.01e-9) is',
    'floating-point noise in the DP level, not a packing.',
    '',
]
open('poc/rational_zipper/docs/reach.md', 'w').write('\n'.join(lines))
print('\n'.join(lines[:14]))
