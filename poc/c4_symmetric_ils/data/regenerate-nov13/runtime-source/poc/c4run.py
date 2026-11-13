import json, os, sys, time
from multiprocessing import Pool
from c4ils import ils_worker
if __name__ == '__main__':
    n, L, secs, W = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    os.makedirs('poc/c4_symmetric_ils/data/regenerate-nov13/c4', exist_ok=True)
    best = (1e9, None)
    with Pool(W) as pool:
        for n_, L_, E, p, evals in pool.imap_unordered(ils_worker, [(n, L, secs, 104729 * n + s) for s in range(W)]):
            print(time.strftime('%H:%M:%S'), f'n={n_} L={L_} worker min overlap E={E:.3e} relaxations={evals}', flush=True)
            if E < best[0]: best = (E, p)
    json.dump(dict(n=n, L=L, E=best[0], p=best[1]), open(f'poc/c4_symmetric_ils/data/regenerate-nov13/c4/{n}_{L}.json', 'w'))
    print('BEST', n, L, best[0], 'FEASIBLE' if best[0] < 1e-26 else '')
