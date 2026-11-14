"""Pick the best-f pocket instance for each requested (w, h, g) type from data/gadlib.json.

usage: pick.py W,H,G [W,H,G ...]   (prints f, source n, and writes data/targets.json)
"""
import json, sys
import numpy as np

LIB = '../data/gadlib.json'


def key(w, h, g):
    return (min(w, h), max(w, h), g)


if __name__ == '__main__':
    wants = [tuple(int(x) for x in a.split(',')) for a in sys.argv[1:]]
    lib = json.load(open(LIB))
    best = {}
    for g in lib:
        k = key(g['w'], g['h'], g['g'])
        f = max(g['fx'], g['fy'])
        if k not in best or f < max(best[k]['fx'], best[k]['fy']) - 1e-12:
            best[k] = g
    out = {}
    for w in wants:
        k = key(*w)
        if k not in best:
            print(f'{w}: NOT IN LIBRARY')
            continue
        g = best[k]
        # orient so that w <= h is the stored orientation used in the squashed key
        f = max(g['fx'], g['fy'])
        print(f'({g["w"]},{g["h"]},+{g["g"]}) src n={g["src"]} f={f:.6f} (fx={g["fx"]:.6f} fy={g["fy"]:.6f}) count={g["cnt"]} need={g["w"]*g["h"]+g["g"]}')
        out[f'{g["w"]}x{g["h"]}g{g["g"]}'] = dict(w=g['w'], h=g['h'], g=g['g'], f=f, src=g['src'], sq=g['sq'], cnt=g['cnt'])
    json.dump(out, open('../data/targets.json', 'w'))
    # count occurrences of each type across the library (how binding a shape is)
    for w in wants:
        k = key(*w)
        occ = [(g['src'], max(g['fx'], g['fy'])) for g in lib if key(g['w'], g['h'], g['g']) == k]
        print(f'  {w}: {len(occ)} instances, srcs {sorted(set(s for s, _ in occ))}')
