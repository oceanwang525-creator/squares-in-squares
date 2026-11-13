"""Squeeze every extracted proper sub-gadget on its own (hidden slack inside multi-pocket rows).
usage: slack_scan.py gadlib.json out.json [shake_rounds]
"""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from pslp import squeeze, shake_squeeze, viol

LIB = None


def work(args):
    i, rounds, G = args
    a = np.array(G['sq']); w, h = G['w'], G['h']
    f0 = max(G['fx'], G['fy'])
    X, Y, T = a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()
    v0 = viol(X, Y, T, w, h, f0 + 1e-9)
    f, X2, Y2, T2, v, it = squeeze(X, Y, T, w, h, f0 + 1e-9)
    if rounds:
        rng = np.random.default_rng(1000 + i)
        f, X2, Y2, T2 = shake_squeeze(X2, Y2, T2, w, h, f, rounds, rng)
        v = viol(X2, Y2, T2, w, h, f)
    return i, f0, f, v, v0, np.column_stack([X2, Y2, T2]).tolist()


if __name__ == '__main__':
    LIB = json.load(open(sys.argv[1]))
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    idx = [i for i, g in enumerate(LIB) if not g['whole']]
    out = {}; t = time.time()
    with Pool(8) as pool:
        for i, f0, f, v, v0, sq in pool.imap_unordered(work, [(i, rounds, LIB[i]) for i in idx], chunksize=2):
            G = LIB[i]
            out[i] = dict(src=G['src'], w=G['w'], h=G['h'], g=G['g'], f0=f0, f=f, v=v, sq=sq)
            if f0 - f > 1e-7:
                print(f'  gadget {i} from n={G["src"]} (w,h,g)=({G["w"]},{G["h"]},{G["g"]}) N={len(sq)}: f {f0:.8f} -> {f:.8f} (-{f0-f:.2e}) viol {v:.0e} start-viol {v0:.0e}', flush=True)
    json.dump(out, open(sys.argv[2], 'w'))
    print(f'{len(out)} sub-gadgets squeezed in {time.time()-t:.0f}s; improved by >1e-7: {sum(1 for o in out.values() if o["f0"]-o["f"]>1e-7)}')
