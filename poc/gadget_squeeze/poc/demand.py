"""What shape would pay: leftover seam budget at the level just below the binding f.

For the biggest-gap n (from poc/koboshchan_coordinator/data/gap_vs_frontier.csv) this prints
  - the DP composition at the highest level strictly below the frontier's f (so: what we can
    already do at the frontier's own f), and
  - the remaining seam budget (K - sum w, K - sum h, 2K+1 - sum(w+h+1)),
i.e. the (w, h) envelope in which ONE more gadget of gain 1..4 would lift the row to the
frontier's side at the frontier's f. This is the concrete shopping list for a new pocket.

usage: demand.py lib.json [n ...]
"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp                                # noqa: E402
import common                            # noqa: E402

GAP = os.path.join(common.ROOT, 'poc/koboshchan_coordinator/data/gap_vs_frontier.csv')


def gap_rows(k=16):
    import csv
    r = list(csv.DictReader(open(GAP)))
    r.sort(key=lambda x: -float(x['gap_score']))
    return r[:k]


def analyse(lib, n, frontier_L, K, ours):
    d = dp.items_of(lib)
    levels = sorted({v[0] for v in d.values()})
    target = frontier_L - K                      # the f the frontier achieves
    below = [f for f in levels if f < target - 1e-12]
    f_lo = below[-1] if below else None
    out = dict(n=n, K=K, target=target, ours=ours - K, f_lo=f_lo, need=None, j=n - K * K)
    if f_lo is None:
        return out
    gm, parts = dp.best_at(lib, K, f_lo)
    sw = sum(lib[i]['w'] if not tr else lib[i]['h'] for i, tr in parts)
    sh = sum(lib[i]['h'] if not tr else lib[i]['w'] for i, tr in parts)
    ssum = sum((lib[i]['w'] if not tr else lib[i]['h']) + (lib[i]['h'] if not tr else lib[i]['w']) + 1
               for i, tr in parts)
    j = n - K * K
    out.update(gain=gm, parts=parts, sw=sw, sh=sh, ssum=ssum, j=j)
    if gm >= j:
        out['need'] = f'gain {gm} >= j={j}: already achievable at f={f_lo:.6f}'
    else:
        need_g = j - gm
        out['need'] = (f'need +{need_g} more gain at f<={target:.6f}: budget '
                       f'dw={K-sw}, dh={K-sh}, dsum={2*K+1-ssum} '
                       f'(current gain {gm} at f<={f_lo:.6f})')
    return out


if __name__ == '__main__':
    lib = dp.load_lib(sys.argv[1])
    ours = common.ours_L()
    rows = gap_rows(int(sys.argv[2]) if len(sys.argv) > 2 else 14)
    for r in rows:
        n = int(r['n']); K = int(r['K']); FL = float(r['frontier'])
        a = analyse(lib, n, FL, K, ours[n])
        desc = ' + '.join(f'{lib[i]["w"]}x{lib[i]["h"]}+{lib[i]["g"]}{"T" if tr else ""}' for i, tr in a.get('parts', [])) or '(none)'
        print(f"n={n} K={K} j={a['j']} ours f={a['ours']:.6f} frontier f={a['target']:.6f} "
              f"gap_score={r['gap_score']}")
        print(f"   {a['need']}")
        print(f"   composition at f<={a['f_lo']:.6f}: {desc}" if a['f_lo'] is not None else '   no composition below the frontier level')
