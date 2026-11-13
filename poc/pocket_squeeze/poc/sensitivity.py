"""Which pocket types matter: for each proper sub-gadget type, lower its f by delta and count improved cells."""
import json, math, sys, copy
from multiprocessing import Pool
from compose import load_lib, predict

DELTA = 1e-3


def run(args):
    key, idxs, lib_path, baseL = args
    lib = load_lib(lib_path)
    for i in idxs:
        lib[i]['f'] -= DELTA
    pred = predict(lib, Kmin=max(2, min(key[0], key[1])))
    L = {int(k): v for k, v in json.load(open(baseL)).items()}
    ns = []; gain = 0.0
    for n, (K, f, parts) in pred.items():
        d = L[n] - (K + f)
        if d > 1e-6:
            ns.append(n); gain += d / math.sqrt(n)
    return key, len(ns), gain, sorted(ns)


if __name__ == '__main__':
    lib = load_lib('gadlib.json')
    types = {}
    for i, g in enumerate(lib):
        if g['whole']:
            continue
        k = (min(g['w'], g['h']), max(g['w'], g['h']), g['g'])
        types.setdefault(k, []).append(i)
    best_f = {k: min(lib[i]['f'] for i in v) for k, v in types.items()}
    jobs = []
    for k, v in types.items():
        # only the instances at the best f of the type are lowered
        idxs = [i for i in v if lib[i]['f'] <= best_f[k] + 1e-9]
        jobs.append((k, idxs, 'gadlib.json', 'baseL.json'))
    res = []
    with Pool(7) as pool:
        for r in pool.imap_unordered(run, jobs):
            res.append(r)
    res.sort(key=lambda r: -r[2])
    json.dump([[list(r[0]), r[1], r[2], r[3]] for r in res], open('sensitivity.json', 'w'))
    print(f'pocket type (w,h,+g) at f | cells improved if f drops by {DELTA} | score gain per {DELTA} | example n')
    for k, cnt, gain, ns in res[:30]:
        print(f'  ({k[0]},{k[1]},+{k[2]}) f={best_f[k]:.6f} | {cnt:3d} | {gain:.3e} | {ns[:10]}')
    print('types with any effect:', sum(1 for r in res if r[1] > 0), 'of', len(res))
