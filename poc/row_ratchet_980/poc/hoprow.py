"""Basin hopping on one row with the reduced model (arms active, grid blocks rigid).
usage: hoprow.py n K cx cy minutes workers [start.json]
"""
import json, math, os, sys, time
import numpy as np
from multiprocessing import Pool
from row980 import load_row, build_model, run_squeeze, full_viol


def worker(args):
    n, K, corner, secs, seed, start = args
    rng = np.random.default_rng(seed)
    if start:
        d = json.load(open(start)); a = np.array(d['sq']); L = d['L']
    else:
        a, L = load_row(n)
    os.makedirs('state/hoprow', exist_ok=True)
    best_a, best_L = a.copy(), L + 1e-12
    t_end = time.time() + secs; rounds = 0; hits = 0; log = []; allr = []
    while time.time() < t_end:
        rounds += 1
        a0 = best_a.copy(); L0 = best_L
        idx, act, stat, mov, mov_full = build_model(a0, L0, K, tr_corner=corner)
        sub = a0[idx].copy(); m = len(idx)
        actI = np.nonzero(act)[0]
        delta = float(rng.choice([0.002, 0.006, 0.02, 0.05]))
        X = sub[:, 0].copy(); Y = sub[:, 1].copy(); T = sub[:, 2].copy()
        X[mov] += delta; Y[mov] += delta                      # loosen: far block moves out with the wall
        mode = rng.random()
        if mode < 0.3:                                        # global jitter of the arms
            sig = float(rng.choice([0.002, 0.005, 0.01]))
            X[actI] += rng.normal(0, sig, len(actI)); Y[actI] += rng.normal(0, sig, len(actI)); T[actI] += rng.normal(0, sig, len(actI))
            kind = 'global'
        elif mode < 0.65:                                     # local shake
            c = actI[rng.integers(len(actI))]; rad = float(rng.choice([1.5, 2.5, 4.0])); sig = float(rng.choice([0.01, 0.03, 0.06]))
            sel = actI[np.hypot(X[actI] - X[c], Y[actI] - Y[c]) < rad]
            X[sel] += rng.normal(0, sig, len(sel)); Y[sel] += rng.normal(0, sig, len(sel)); T[sel] += rng.normal(0, sig, len(sel))
            kind = 'local'
        else:                                                 # slide one tilted row along itself
            td = np.degrees(np.mod(T, np.pi / 2)); til = actI[np.minimum(td[actI], 90 - td[actI]) > 2.0]
            c = til[rng.integers(len(til))]; th = T[c]
            if rng.random() < 0.5: th += math.pi / 2
            ux, uy = math.cos(th), math.sin(th)
            v = -(X[til] - X[c]) * uy + (Y[til] - Y[c]) * ux
            dang = np.abs(np.mod(T[til] - T[c] + math.pi / 4, math.pi / 2) - math.pi / 4)
            sel = til[(np.abs(v) < 0.35) & (dang < 0.02)]
            s = float(rng.choice([-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]))
            X[sel] += s * ux; Y[sel] += s * uy
            kind = 'rowslide'
        try:
            full, Ln, v_, it = run_squeeze(a0, L0 + delta, K, idx, stat, mov, mov_full, X=X, Y=Y, T=T)
        except Exception:
            continue
        # run_squeeze shifts the unlisted moving squares by (f - f0) relative to L0 + delta: add the loosening back
        rest = np.ones(len(a0), bool); rest[idx] = False
        full[rest & mov_full, 0] += delta; full[rest & mov_full, 1] += delta
        allr.append((kind, delta, Ln - best_L, v_))
        if Ln < best_L - 1e-10 and v_ <= 1e-12 and full_viol(full, Ln) <= 1e-12:
            hits += 1; log.append((rounds, kind, best_L - Ln))
            best_a, best_L = full, Ln
            json.dump(dict(n=n, L=best_L, sq=best_a.tolist()), open(f'state/hoprow/{n}_{seed}.json', 'w'))
    json.dump(allr, open(f'state/hoprow/log_{n}_{seed}.json', 'w'))
    return seed, rounds, hits, best_L, log


if __name__ == '__main__':
    n, K = int(sys.argv[1]), int(sys.argv[2]); corner = (float(sys.argv[3]), float(sys.argv[4]))
    mins = float(sys.argv[5]); W_ = int(sys.argv[6]); start = sys.argv[7] if len(sys.argv) > 7 else None
    a, L = load_row(n)
    print(f'n={n} start L={L:.12f}', flush=True)
    with Pool(W_) as pool:
        for seed, rounds, hits, bl, log in pool.imap_unordered(worker, [(n, K, corner, mins * 60, 9000 + s, start) for s in range(W_)]):
            print(time.strftime('%H:%M:%S'), f'seed {seed}: {rounds} rounds, {hits} improvements, L={bl:.12f} (dL {L-bl:.3e})', [(r, k, f'{d:.1e}') for r, k, d in log][:12], flush=True)
