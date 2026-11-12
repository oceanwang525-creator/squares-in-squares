"""Tiny certified wins: (1) deletion from n+k packings with smaller certified L;
(2) grid-border wrap of n0 by t rows/columns (side L0 + t + eps) then deletion to target n.
Border squares are placed eps beyond the old packing's high-precision extremes.
usage: tiny_pass.py full.csv lit.json pandas.json out.json
"""
import csv, json, math, sys
from decimal import Decimal
from multiprocessing import Pool
import mpmath
import merge_public as mp

mpmath.mp.dps = 80
EPS = mpmath.mpf('5e-14')


def load(path):
    by = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            by.setdefault(int(r['n']), []).append({k: r[k] for k in ('n', 'square_id', 'x', 'y', 'angle')})
    for n in by:
        by[n].sort(key=lambda r: int(r['square_id']))
    return by


def extremes(rows):
    xs = []; ys = []
    for r in rows:
        x = mpmath.mpf(r['x']); y = mpmath.mpf(r['y']); a = mpmath.mpf(r['angle'])
        c, s = mpmath.cos(a) / 2, mpmath.sin(a) / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                xs.append(x + sx * c - sy * s); ys.append(y + sx * s + sy * c)
    return min(xs), max(xs), min(ys), max(ys)


from decimal import ROUND_CEILING, getcontext
getcontext().prec = 80
Q = Decimal('1e-25')


def up(v):
    """mpmath value -> Decimal rounded UP to 25 decimals."""
    return Decimal(mpmath.nstr(v, 70, strip_zeros=False, min_fixed=-100, max_fixed=100)).quantize(Decimal('1e-25'), rounding=ROUND_CEILING)


STEP = Decimal(1) + Decimal('2e-14')   # border squares are spaced, never exactly touching


def fmt(d):
    return format(d, 'f')


def wrap(rows, t, target):
    xmin, xmax, ymin, ymax = extremes(rows)
    L0 = max(xmax - xmin, ymax - ymin)
    X0 = up(xmin + L0 + EPS)        # east strip starts here (>= xmax)
    Y0 = up(ymin + L0 + EPS)        # north strip starts here (>= ymax)
    XS = up(xmin + EPS); YS = up(ymin + EPS)
    H = Decimal('0.5')
    out = [dict(r) for r in rows]
    m_e = int(mpmath.floor(L0 + t)); m_n = int(mpmath.floor(L0))
    for i in range(t):                       # east strip: t columns, full new height
        for j in range(m_e):
            out.append({'x': fmt(X0 + i * STEP + H), 'y': fmt(YS + j * STEP + H), 'angle': '0.0'})
    for i in range(m_n):                     # north strip: t rows over the old width
        for j in range(t):
            out.append({'x': fmt(XS + i * STEP + H), 'y': fmt(Y0 + j * STEP + H), 'angle': '0.0'})
    out = out[:target]
    for i, r in enumerate(out):
        r['n'] = str(target); r['square_id'] = str(i)
    return out


def main():
    full, litp, pdp, outp = sys.argv[1:5]
    lit = {r['n']: Decimal(r['L']) for r in json.load(open(litp))['results']}
    pdl = {r['n']: Decimal(r['L']) for r in json.load(open(pdp))['results']}
    L = {n: max(lit[n], pdl[n]) for n in lit}
    by = load(full)
    cands = {}
    for n in range(1, 1000):                 # deletion
        for m in range(n + 1, min(1000, n + 6) + 1):
            if L[m] < L[n] and (n not in cands or L[m] < cands[n][0]):
                rows = [dict(r) for r in by[m][:n]]
                for i, r in enumerate(rows):
                    r['n'] = str(n); r['square_id'] = str(i)
                cands[n] = (L[m], f'delete-from-{m}', rows)
    dels = dict(cands); cands = {}
    for n0 in range(1, 1001):                # wrap
        m0 = int(L[n0])
        for t in range(1, 35):
            nfull = n0 + 2 * t * m0 + t * t
            for nn in range(min(nfull, 1000), 0, -1):
                est = L[n0] + t + Decimal('1e-13')
                if est < L[nn]:
                    if nn not in cands or est < cands[nn][0]:
                        cands[nn] = (est, f'wrap-{n0}-t{t}', (n0, t))
                else:
                    break
            if nfull > 1000:
                break
    jobs = [(n, rows, src) for n, (est, src, rows) in dels.items()]
    for n, (est, src, payload) in cands.items():
        rows = wrap(by[payload[0]], payload[1], n) if isinstance(payload, tuple) else payload
        jobs.append((n, rows, src))
    print('candidates', len(jobs), flush=True)
    wins = {}
    with Pool(8) as pool:
        for idx, (n, res, why) in enumerate(pool.imap(mp.certify, [(n, rows) for n, rows, _ in jobs])):
            src = jobs[idx][2]
            if res is None:
                print('reject', n, src, why[:80], flush=True); continue
            Lnew = max(Decimal(res[0]), Decimal(res[1]))
            if Decimal(res[0]) < lit[n] and Decimal(res[1]) < pdl[n]:
                rows = jobs[idx][1]
                if n in wins and Decimal(wins[n]['L']) <= Lnew:
                    continue
                wins[n] = {'L': str(Lnew), 'rows': rows, 'src': src}
                print(f'win n={n} {src}: {L[n]} -> {Lnew}  dL={L[n]-Lnew:.3e}', flush=True)
            else:
                print('no gain', n, src, f'{L[n]-Lnew:.3e}', flush=True)
    json.dump(wins, open(outp, 'w'))
    print('wins', len(wins), 'score gain', sum(float(L[n] - Decimal(w['L'])) / math.sqrt(n) for n, w in wins.items()))


if __name__ == '__main__':
    main()
