"""Merge the scanned pocket files, keep the Pareto-front items (with geometry).

For the DP only the Pareto set matters (item (w,h,g,f) is useless if another item
has w'<=w, h'<=h, g'>=g, f'<=f).  This also shrinks the ~180 MB of raw scan
output to a small file that `dp.py` / `build_cert.py` consume.

usage: merge_pockets.py --gadlib gadlib.json --out pockets_pareto.json f1.json f2.json ...
"""
import argparse
import json
import sys

sys.path.insert(0, 'poc/swarm_pocket_lift_scan4/poc')
import dp as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gadlib', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', help='also write every best-per-type pocket (polish seeds)')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    base = D.items_from_gadlib(a.gadlib)
    cur = {}
    for it in base:
        k = (it['w'], it['h'], it['g'])
        if k not in cur or it['f'] < cur[k] - 1e-12:
            cur[k] = it['f']
    print(f'gadlib: {len(base)} oriented items, {len(cur)} oriented types')

    best = {}          # (w,h,g) -> record with geometry
    raw = 0
    for path in a.files:
        lib = json.load(open(path))
        raw += len(lib)
        for g in lib:
            for (w, h) in ((g['w'], g['h']), (g['h'], g['w'])):
                k = (w, h, g['g'])
                if k not in best or g['f'] < best[k]['f'] - 1e-12:
                    rec = dict(g)
                    rec['w'], rec['h'] = w, h          # oriented type (for the DP)
                    rec['nw'], rec['nh'] = g['w'], g['h']   # native dims of the geometry
                    best[k] = rec
    print(f'{raw} raw gadgets -> {len(best)} distinct oriented (w,h,g)')

    new = [k for k in best if k not in cur]
    better = [k for k in best if k in cur and best[k]['f'] < cur[k] - 1e-9]
    print(f'  new oriented shapes {len(new)}; improve an existing shape {len(better)}')
    for k in sorted(better, key=lambda k: cur[k] - best[k]['f'])[:15]:
        print(f'   improves ({k[0]},{k[1]},+{k[2]}): {cur[k]:.9f} -> {best[k]["f"]:.9f} '
              f'(n={best[k]["src"]})')

    items = D.pareto(base + [{k: v[k] for k in ('w', 'h', 'g', 'f')} for v in best.values()])
    keep = {(it['w'], it['h'], it['g']) for it in items}
    # keep ONE native record per pocket geometry (the swapped orientation is the
    # same pocket, rotated; build_cert rotates it when needed)
    surv = {}
    for k, v in best.items():
        if k not in keep:
            continue
        nk = (v['src'], v['x0'], v['x1'], v['y0'], v['y1'], v['g'])
        if nk not in surv:
            rec = {kk: v[kk] for kk in ('src', 'w', 'h', 'g', 'f', 'cnt', 'x0', 'x1', 'y0', 'y1')}
            rec['w'], rec['h'] = v['nw'], v['nh']    # geometry-native orientation
            surv[nk] = rec
    surv = sorted(surv.values(), key=lambda v: (v['g'], v['f']))
    print(f'merged Pareto set: {len(items)} items; scanned pockets surviving: {len(surv)}')
    json.dump(surv, open(a.out, 'w'), indent=0)
    print('wrote', a.out, f'({len(surv)} gadgets with geometry)')
    if a.seeds:
        allb = {}
        for k, v in best.items():
            nk = (v['src'], v['x0'], v['x1'], v['y0'], v['y1'], v['g'])
            if nk not in allb:
                allb[nk] = {kk: v[kk] for kk in ('src', 'g', 'f', 'cnt',
                                                 'x0', 'x1', 'y0', 'y1')}
                allb[nk]['w'], allb[nk]['h'] = v['nw'], v['nh']
        json.dump(list(allb.values()), open(a.seeds, 'w'), indent=0)
        print('wrote', a.seeds, f'({len(allb)} polish seeds)')


if __name__ == '__main__':
    main()
