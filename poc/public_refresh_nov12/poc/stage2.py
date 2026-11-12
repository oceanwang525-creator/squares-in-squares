"""Stage 2: build each gaining block seed with slack, then shrink-relax to its tightest c.
usage: stage2.py blocks.json c_seed secs workers
"""
import json, math, os, sys, time
import numpy as np
from multiprocessing import Pool
from blockgen import build_block
from opt_n import work  # shrink-and-relax loop (writes state/opt/{n}.json)

os.makedirs('state/stage2', exist_ok=True)


def run(args):
    (g, cfg), cseed, secs = args
    r, s, thdeg, fx, fy, k = cfg; r, s, k = int(r), int(s), int(k)
    L = k + cseed
    P = build_block(L, math.floor(L / 2) + fx, math.floor(L / 2) + fy, math.radians(thdeg), r, s, -1e-12)
    n = len(P)
    X = np.empty(3 * n); X[0::3] = P[:, 0]; X[1::3] = P[:, 1]; X[2::3] = P[:, 2]
    n_, L0, Lc, tries, acc = work((n, X, secs, int(1e6 * fx + 1e3 * fy + r * 31 + s)))
    os.replace(f'state/opt/{n}.json', f'state/stage2/{r}x{s}_{thdeg}_{n}.json') if os.path.exists(f'state/opt/{n}.json') else None
    return g, cfg, n, k, L0 - k, Lc - k, tries, acc


if __name__ == '__main__':
    blocks = json.load(open(sys.argv[1])); cseed = float(sys.argv[2]); secs = float(sys.argv[3]); W = int(sys.argv[4])
    # one config per (r, s, gain): keep the first (best) occurrence
    seen = set(); todo = []
    for g, cfg in blocks:
        key = (g, int(cfg[0]), int(cfg[1]))
        if key not in seen:
            seen.add(key); todo.append((g, cfg))
    with Pool(W) as pool:
        for g, cfg, n, k, c0, c1, tries, acc in pool.imap_unordered(run, [(t, cseed, secs) for t in todo]):
            print(time.strftime('%H:%M:%S'), f'gain={g} block={int(cfg[0])}x{int(cfg[1])}@{cfg[2]} n={n} k={k} c_seed={c0:.4f} -> c*={c1:.6f} tries={tries} acc={acc}', flush=True)
