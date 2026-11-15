"""Delete d squares from row n+d (greedy, one at a time) and squeeze: candidates for row n.
usage: delrow.py n_src K workers out.json [max_candidates]
"""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from row980 import load_row, build_model, run_squeeze, full_viol


def work(args):
    a, L, K, i = args
    b = np.delete(a, i, axis=0)
    try:
        idx, act, stat, mov, mov_full = build_model(b, L, K)
        full, Ln, v, it = run_squeeze(b, L, K, idx, stat, mov, mov_full)
    except Exception as e:
        return i, None, None
    if v > 1e-12 or full_viol(full, Ln) > 1e-12:
        return i, None, None
    return i, Ln, full.tolist()


if __name__ == '__main__':
    n_src, K, W_, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    a, L = load_row(n_src)
    idx, act, stat, mov, mov_full = build_model(a, L, K)
    cand = idx[act]
    if len(sys.argv) > 5: cand = cand[:int(sys.argv[5])]
    print(f'n_src={n_src} L={L:.9f}; model kept {len(idx)} active {act.sum()} static {stat.sum()} moving {mov.sum()}; deleting each of {len(cand)} active squares', flush=True)
    res = []; best = (9e9, None, None); t = time.time()
    with Pool(W_) as pool:
        for i, Ln, full in pool.imap_unordered(work, [(a, L + 1e-12, K, int(i)) for i in cand], chunksize=2):
            if Ln is None: continue
            res.append((Ln, i))
            if Ln < best[0]:
                best = (Ln, i, full)
                json.dump(dict(n=n_src - 1, L=Ln, deleted=int(i), sq=full), open(out, 'w'))
    res.sort()
    print(f'{len(res)} deletions squeezed in {time.time()-t:.0f}s; best L={res[0][0]:.9f} (square {res[0][1]}); top 8:', [f'{l:.6f}' for l, _ in res[:8]], flush=True)
    print('L drops:', 'median %.5f' % (L - res[len(res)//2][0]), 'best %.5f' % (L - res[0][0]))
