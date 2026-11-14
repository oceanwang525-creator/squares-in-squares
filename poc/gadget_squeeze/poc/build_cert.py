"""Build every DP-predicted cell, check it with the SAT module, then certify it with the
official metric in BOTH transports (literal strings and a real pandas read_csv/to_csv trip).

usage: build_cert.py pred.json [--min-gain 1e-9] [--out ../data/wins.json]
Writes ../final.csv (header-only when nothing certifies) and ../data/wins.json.
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                              # noqa: E402
import compose                             # noqa: E402
import squeeze as S                        # noqa: E402

EPS = 2e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pred')
    ap.add_argument('--lib', default='../data/gadlib.json')
    ap.add_argument('--min-gain', type=float, default=1e-9)
    ap.add_argument('--out', default='../data/wins.json')
    ap.add_argument('--final', default='../final.csv')
    a = ap.parse_args()
    lib = compose.load_lib(a.lib)
    pred = {int(k): v for k, v in json.load(open(a.pred)).items()}
    ours = common.ours_L()
    wins = {}
    for n, p in sorted(pred.items()):
        K, f, parts = p['K'], p['f'], [tuple(x) for x in p['parts']]
        P = compose.realise(lib, K, f, parts, n)
        desc = ' + '.join(f'{lib[i]["w"]}x{lib[i]["h"]}+{lib[i]["g"]}{"T" if tr else ""}'
                          f'[src n={lib[i].get("src")} f={lib[i]["f"]:.6f}]' for i, tr in parts)
        if P is None or len(P) != n:
            print(f'  n={n}: build failed ({desc})')
            continue
        # float gate with the repo SAT
        v = S.violation(P, K + f, K + f)
        x0, x1, y0, y1 = S.bbox(P)
        side = (x1 - x0, y1 - y0)
        if v > 1e-9 or max(side) > K + f + 1e-9:
            print(f'  n={n}: float gate failed (viol {v:.2e}, box {side[0]:.6f}x{side[1]:.6f} vs {K+f:.6f})')
            continue
        rows = common.rows_from_P(n, P, eps=EPS)
        n2, L1, L2, why = common.certify(n, rows)
        if L1 is None or L2 is None:
            print(f'  n={n}: certification failed ({why})')
            continue
        d1 = ours[n] - float(L1); d2 = ours[n] - float(L2)
        ok = min(d1, d2) > a.min_gain
        print(f'  n={n} K={K}: {ours[n]:.9f} -> literal {float(L1):.12f} (dL {d1:+.3e}), '
              f'pandas {float(L2):.12f} (dL {d2:+.3e}) {"WIN" if ok else "no gain"}   {desc}', flush=True)
        if ok:
            wins[n] = dict(L_lit=str(L1), L_pd=str(L2), dL_lit=d1, dL_pd=d2,
                           rows=[{k: str(r[k]) for k in ('square_id', 'x', 'y', 'angle')} for r in rows],
                           desc=desc, K=K, f=f)
    json.dump({str(k): v for k, v in wins.items()}, open(a.out, 'w'))
    groups = {n: [{'square_id': r['square_id'], 'x': r['x'], 'y': r['y'], 'angle': r['angle']}
                  for r in w['rows']] for n, w in wins.items()}
    common.write_final(a.final, groups)
    tot = sum((w['dL_lit'] / math.sqrt(int(n)) + w['dL_pd'] / math.sqrt(int(n))) / 2
              for n, w in wins.items())
    print(f'certified wins: {len(wins)}; total score gain {tot:.3e}; wrote {a.final}')


if __name__ == '__main__':
    main()
