"""Library of pockets -> composition DP -> build every improved n -> official certification.

usage: apply_lib.py base_lib.json baseL.json out_wins.json extra1.json|dir [extra2 ...] [--no-cert] [--min-gain 1e-9]
Extra entries: JSON files (or directories of them) with keys w, h, g, f, sq.
"""
import glob, json, math, os, sys, time
import numpy as np
from multiprocessing import Pool
from compose import load_lib, predict, realise
from pslp import feas

EPS = 2e-12        # uniform dilation so touching squares certify with a positive gap


def rows_of(n, P):
    P = P.copy(); P[:, 0] *= (1 + EPS); P[:, 1] *= (1 + EPS)
    return [{'n': str(n), 'square_id': str(i), 'x': repr(float(x)), 'y': repr(float(y)), 'angle': repr(float(t))}
            for i, (x, y, t) in enumerate(P)]


def cert(args):
    import merge_public as mp
    n, rows = args
    return mp.certify((n, rows))


def load_extra(paths):
    out = []
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, '*.json'))) if os.path.isdir(p) else [p]
        for f in files:
            d = json.load(open(f))
            items = d if isinstance(d, list) else ([d] if 'sq' in d else list(d.values()))
            for o in items:
                if not isinstance(o, dict) or 'sq' not in o:
                    continue
                if o.get('v', 0.0) > 1e-12:
                    continue
                out.append(dict(w=o['w'], h=o['h'], g=o['g'], f=o['f'], sq=o['sq'], src=o.get('key', o.get('src', os.path.basename(f))),
                                whole=False, tag=os.path.basename(f)))
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    no_cert = '--no-cert' in sys.argv
    min_gain = 1e-9
    if '--min-gain' in sys.argv:
        min_gain = float(sys.argv[sys.argv.index('--min-gain') + 1]); args = [a for a in args if a != str(min_gain) and a != sys.argv[sys.argv.index('--min-gain') + 1]]
    base_lib, baseL, out_p = args[:3]
    lib = load_lib(base_lib)
    nbase = len(lib)
    extra = load_extra(args[3:])
    lib += extra
    L = {int(k): v for k, v in json.load(open(baseL)).items()}
    t = time.time(); pred = predict(lib)
    cand = []
    for n, (K, f, parts) in sorted(pred.items()):
        if L[n] - (K + f) * (1 + EPS) > min_gain:
            cand.append((n, K, f, parts))
    print(f'{len(extra)} extra pockets; DP {time.time()-t:.0f}s; predicted improved n: {len(cand)}; predicted gain {sum((L[n]-(K+f))/math.sqrt(n) for n,K,f,_ in cand):.3e}', flush=True)
    jobs = []; meta = {}
    for n, K, f, parts in cand:
        P = realise(lib, K, f + 1e-13, parts, n)
        desc = ' + '.join(f'{lib[i]["w"]}x{lib[i]["h"]}+{lib[i]["g"]}{"T" if tr else ""}[{lib[i].get("src")}]' for i, tr in parts)
        if P is None:
            print(f'  n={n}: build failed ({desc})'); continue
        mp_, mw_ = feas(P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy(), K + f + 1e-13, K + f + 1e-13)
        if min(mp_, mw_) < -1e-12:
            print(f'  n={n}: float check failed (pair {mp_:.1e}, wall {mw_:.1e}) ({desc})'); continue
        meta[n] = dict(K=K, f=f, desc=desc, new=any(i >= nbase for i, _ in parts))
        jobs.append((n, rows_of(n, P)))
        print(f'  n={n} K={K}: {L[n]:.9f} -> {K+f:.9f} ({L[n]-K-f:+.2e})  {desc}', flush=True)
    wins = {}
    if not no_cert and jobs:
        with Pool(6) as pool:
            for (n, rows), (n2, res, why) in zip(jobs, pool.imap(cert, jobs)):
                if res is None:
                    print(f'  CERT FAIL n={n}: {why}', flush=True); continue
                Lnew = max(float(res[0]), float(res[1]))
                if Lnew < L[n]:
                    wins[n] = dict(L_lit=res[0], L_pd=res[1], rows=rows, **meta[n])
                    print(f'  CERTIFIED n={n}: {L[n]:.12f} -> {Lnew:.12f}  dL={L[n]-Lnew:.3e}', flush=True)
                else:
                    print(f'  no gain after certification n={n}: {Lnew:.12f} vs {L[n]:.12f}', flush=True)
        json.dump({str(k): v for k, v in wins.items()}, open(out_p, 'w'))
        print(f'certified wins: {len(wins)}; score gain {sum((L[n]-max(float(w["L_lit"]),float(w["L_pd"])))/math.sqrt(n) for n,w in wins.items()):.6e}')
