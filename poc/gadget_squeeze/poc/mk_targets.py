"""Dump one seed pocket per distinct (w, h, +g) type (the best-f instance) to data/targets_all.json."""
import json

LIB = '../data/gadlib.json'
OUT = '../data/targets_all.json'

lib = json.load(open(LIB))
best = {}
for g in lib:
    k = (g['w'], g['h'], g['g'])
    f = max(g['fx'], g['fy'])
    if k not in best or f < best[k]['f'] - 1e-12:
        best[k] = dict(w=g['w'], h=g['h'], g=g['g'], f=f, src=g['src'], sq=g['sq'], cnt=g['cnt'])
json.dump({f'{v["w"]}x{v["h"]}g{v["g"]}': v for v in best.values()}, open(OUT, 'w'), indent=0)
print(len(best), 'types ->', OUT)
for k in sorted(best, key=lambda k: best[k]['f']):
    v = best[k]
    print(f'  ({v["w"]},{v["h"]},+{v["g"]}) f={v["f"]:.6f} N={v["w"]*v["h"]+v["g"]} src n={v["src"]}')
