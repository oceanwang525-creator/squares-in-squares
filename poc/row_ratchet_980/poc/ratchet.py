"""Ratchet one row: team mechanism walk -> my full-row SLP squeeze -> repeat while it improves.
usage: ratchet.py n K start.json max_cycles   (env REPO_ROOT, CPP_LIB, THREADS)"""
import json, os, subprocess, sys, time
import numpy as np
from pslp import squeeze
n, K, start, cycles = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
os.makedirs(f'wins{n}', exist_ok=True)
cur = start; L = json.load(open(start)); L = (L if 'L' in L else next(iter(L.values()))).get('L')
for c in range(cycles):
    t = time.time(); out = f'mechwalk/r{n}_{c}'
    subprocess.run([sys.executable, '-u', 'mechwalk_run.py', out, str(n), cur], stdout=open(f'{out}.log', 'w'), stderr=subprocess.STDOUT)
    wp = f'{out}/win_{n}.json'
    if not os.path.exists(wp):
        print(f'cycle {c}: mechanism walk found no win from L={L} [{time.time()-t:.0f}s]', flush=True); break
    w = json.load(open(wp)); rows = sorted(w['rows'], key=lambda r: int(r['square_id']))
    a = np.array([(float(r['x']), float(r['y']), float(r['angle'])) for r in rows])
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2]))); a[:, 0] -= (a[:, 0] - e).min(); a[:, 1] -= (a[:, 1] - e).min()
    Lw = max((a[:, 0] + e).max(), (a[:, 1] + e).max())
    f, X, Y, T, v, it = squeeze(a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), K, K, Lw - K + 1e-12)
    if v <= 1e-12 and K + f < Lw:
        a = np.column_stack([X, Y, T]); Ln = K + f
    else:
        Ln = Lw
    cur = f'wins{n}/ratchet_{c}.json'; json.dump(dict(n=n, L=Ln, sq=a.tolist()), open(cur, 'w'))
    print(f'cycle {c}: walk {Lw:.9f} -> squeeze {Ln:.9f} (was {L:.9f}, gain {L-Ln:.3e}) [{time.time()-t:.0f}s]', flush=True)
    L = Ln
print('final', cur, L)
