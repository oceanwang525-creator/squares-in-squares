"""Staircase composition DP over (gadlib.json  U  scanned pockets).

Same rule as `poc/faultline_dp2/poc/dpx.py`:

    sum w <= K,  sum h <= K,  sum (w + h + 1) <= 2K + 1   =>   n = K^2 + sum g
    side = K + max_i f_i            (f_i = max(W_i - w_i, H_i - h_i) of a pocket)

Difference: this copy keeps parent pointers, so the winning composition can be
reconstructed (each predicted cell is reported as a list of items, which
`build_cert.py` then realises and certifies).

usage:
  dp.py base   --lib lib.json --report report.json [--out pred.json]
  dp.py extra  --lib lib.json --pockets pockets.json --report report.json [--out pred.json]
  dp.py pack   --lib lib.json --n 442            # print the composition of one n
"""
import argparse
import json
import math

import numpy as np

KMAX = 31
NEG = -10 ** 6


# ------------------------------------------------------------------ item sets
def items_from_gadlib(path):
    lib = json.load(open(path))
    best = {}
    for g in lib:
        fx, fy = g.get('fx', g.get('f', 0.0)), g.get('fy', g.get('f', 0.0))
        f = max(fx, fy)
        for (w, h) in ((g['w'], g['h']), (g['h'], g['w'])):
            k = (w, h, g['g'])
            if k not in best or f < best[k][0] - 1e-12:
                best[k] = (f, 'n=%d' % g['src'], g['src'])
    return [dict(w=w, h=h, g=g, f=v[0], src=v[2], tag=v[1]) for (w, h, g), v in best.items()]


def items_from_pockets(path):
    lib = json.load(open(path))
    best = {}
    for g in lib:
        for (w, h) in ((g['w'], g['h']), (g['h'], g['w'])):
            k = (w, h, g['g'])
            if k not in best or g['f'] < best[k]['f'] - 1e-12:
                best[k] = dict(w=w, h=h, g=g['g'], f=g['f'], src=g['src'],
                               pocket=g, tag='n=%d(%s)' % (g['src'], 'T' if (w, h) != (g['w'], g['h']) else ''),
                               transposed=(w, h) != (g['w'], g['h']))
    return list(best.values())


def pareto(items):
    """drop items dominated by w'<=w, h'<=h, g'>=g, f'<=f."""
    it = sorted(items, key=lambda r: (r['f'], -r['g'], r['w'] + r['h']))
    keep = []
    for a in it:
        dom = False
        for b in keep:
            if b['w'] <= a['w'] and b['h'] <= a['h'] and b['g'] >= a['g']:
                dom = True
                break
        if not dom:
            keep.append(a)
    return keep


# ------------------------------------------------------------------------ DP
def knap(items, K, best=None):
    """in-place unbounded knapsack, ascending sweeps."""
    S = 2 * K + 1
    if best is None:
        best = np.full((K + 1, K + 1, S + 1), NEG, np.int32)
        best[0, 0, 0] = 0
    for it in items:
        w, h, gg = it['w'], it['h'], it['g']
        d = w + h + 1
        if w > K or h > K or d > S:
            continue
        for x in range(w, K + 1):
            src = best[x - w, :K + 1 - h, :S + 1 - d]
            dst = best[x, h:, d:]
            cand = np.where(src >= 0, src + gg, -1)
            np.maximum(dst, cand, out=dst)
    best[best < 0] = -1
    return best


def knap_parent(items, K):
    """unbounded knapsack with full parent tracking (small K, so plain loops)."""
    S = 2 * K + 1
    best = np.full((K + 1, K + 1, S + 1), NEG, np.int32)
    best[0, 0, 0] = 0
    pi = -np.ones((K + 1, K + 1, S + 1), np.int32)
    for ii, it in enumerate(items):
        w, h, gg = it['w'], it['h'], it['g']
        d = w + h + 1
        if w > K or h > K or d > S:
            continue
        for x in range(w, K + 1):
            for y in range(h, K + 1):
                row = best[x, y]
                prev = best[x - w, y - h]
                for s in range(d, S + 1):
                    v = prev[s - d]
                    if v >= 0 and v + gg > row[s]:
                        row[s] = v + gg
                        pi[x, y, s] = ii
    return best, pi


def dp_levels(items, Kmax=KMAX, nmax=1000, levels=None):
    """for every K, the smallest f level at which n = K^2+j becomes reachable."""
    fs = sorted({round(it['f'], 12) for it in items}) if levels is None else levels
    by_f = {}
    for it in items:
        by_f.setdefault(round(it['f'], 12), []).append(it)
    pred = {}
    for K in range(2, Kmax + 1):
        todo = {j for j in range(1, 2 * K + 1) if K * K + j <= nmax}
        if not todo:
            continue
        S = 2 * K + 1
        best = np.full((K + 1, K + 1, S + 1), NEG, np.int32)
        best[0, 0, 0] = 0
        for f in fs:
            sel = [it for it in by_f.get(f, []) if it['w'] <= K and it['h'] <= K
                   and it['w'] + it['h'] + 1 <= S]
            if sel:
                knap(sel, K, best=best)
            gmax = int(best.max())
            if gmax <= 0 or not todo:
                continue
            for j in sorted(todo):
                if j <= gmax:
                    pred[K * K + j] = (K, f)
            todo = {j for j in todo if j > gmax}
            if not todo:
                break
    return pred


def reconstruct(K, j, fmax, items):
    """items (all f <= fmax) whose sum g >= j; returns the item list."""
    S = 2 * K + 1
    sel = [it for it in items if it['f'] <= fmax + 1e-12 and it['w'] <= K and it['h'] <= K
           and it['w'] + it['h'] + 1 <= S]
    best, pi = knap_parent(sel, K)
    # best state with value >= j, preferring larger g (== best overall)
    st = np.unravel_index(np.argmax(best), best.shape)
    if best[st] < j:
        return None
    x, y, s = st
    out = []
    while (x, y, s) != (0, 0, 0):
        ii = int(pi[x, y, s])
        if ii < 0:
            return None
        it = sel[ii]
        out.append(it)
        x, y, s = x - it['w'], y - it['h'], s - it['w'] - it['h'] - 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['base', 'extra', 'pack'])
    ap.add_argument('--lib', required=True, help='gadlib.json')
    ap.add_argument('--pockets')
    ap.add_argument('--report', required=True)
    ap.add_argument('--out')
    ap.add_argument('--n', type=int)
    ap.add_argument('--kmax', type=int, default=KMAX)
    ap.add_argument('--min-pred-d', type=float, default=1e-10,
                    help='prediction gate; final certification still requires dL < -1e-9')
    a = ap.parse_args()
    rep = json.load(open(a.report))['results']
    L = {r['n']: float(r['L']) for r in rep}
    base = pareto(items_from_gadlib(a.lib))
    print(f'gadlib items: {len(base)} after pareto')
    items = base
    if a.pockets:
        pk = items_from_pockets(a.pockets)
        print(f'scanned pockets: {len(pk)} distinct oriented items '
              f'({len({(p["w"], p["h"], p["g"]) for p in pk})} oriented types)')
        cur = {(p['w'], p['h'], p['g']): p['f'] for p in base}
        new = [p for p in pk if (p['w'], p['h'], p['g']) not in cur]
        better = [p for p in pk if (p['w'], p['h'], p['g']) in cur
                  and p['f'] < cur[(p['w'], p['h'], p['g'])] - 1e-9]
        print(f'  of which {len(new)} new oriented shapes, {len(better)} improve an existing one')
        items = pareto(base + pk)
        print(f'merged items after pareto: {len(items)}')
    pred = dp_levels(items, a.kmax)
    rows = []
    for n, (K, f) in pred.items():
        d = L[n] - (K + f)
        if d > a.min_pred_d:
            rows.append((d / math.sqrt(n), n, L[n], K + f, K, f))
    rows.sort(key=lambda r: -r[0])
    print(f'predicted cells {len(pred)}; DP strictly better at {len(rows)} n; '
          f'score gain {sum(r[0] for r in rows):.4e}')
    for sg, n, ours, Lp, K, f in rows:
        comp = reconstruct(K, n - K * K, f, items)
        desc = ' + '.join(f"({c['w']},{c['h']},+{c['g']})@{c['f']:.6f}<{c['tag']}>"
                          for c in comp) if comp else '??'
        print(f'  n={n} K={K} j={n - K * K}: ours {ours:.9f} -> {Lp:.9f} (gain {sg:.4e})  {desc}')
    if a.out:
        out = {}
        for sg, n, ours, Lp, K, f in rows:
            comp = reconstruct(K, n - K * K, f, items)
            out[str(n)] = dict(K=K, j=n - K * K, f=f, gain=sg, ours=ours,
                               comp=[{k: c[k] for k in ('w', 'h', 'g', 'f', 'src', 'tag')}
                                     | {'transposed': bool(c.get('transposed', False))} for c in (comp or [])])
        json.dump(out, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
