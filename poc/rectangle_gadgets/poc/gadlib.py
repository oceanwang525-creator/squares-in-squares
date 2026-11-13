"""Sub-gadget library: minimal clean-cut rectangles around tilted clusters in every packing.
gadget = (w, h, g, fx, fy): a (w+fx) x (h+fy) rectangle holding w*h + g squares, with its squares.
usage: gadlib.py merged.csv out.json
"""
import csv, json, math, sys, collections
import numpy as np


def load_all(path):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[int(r['n'])].append((float(r['x']), float(r['y']), float(r['angle'])))
    out = {}
    for n, rows in by.items():
        a = np.array(rows); e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
        a[:, 0] -= (a[:, 0] - e).min(); a[:, 1] -= (a[:, 1] - e).min()
        out[n] = (a, e, max((a[:, 0] + e).max(), (a[:, 1] + e).max()))
    return out


def clusters(a, mask, link=1.5):
    idx = np.nonzero(mask)[0]; lab = {}; c = 0
    for s in idx:
        if s in lab: continue
        st = [s]; lab[s] = c
        while st:
            i = st.pop()
            d = np.hypot(a[idx, 0] - a[i, 0], a[idx, 1] - a[i, 1])
            for j in idx[d < link]:
                if j not in lab: lab[j] = c; st.append(j)
        c += 1
    return [np.array([i for i in idx if lab[i] == q]) for q in range(c)]


def clean_rect(a, e, x0, x1, y0, y1, tol=1e-6):
    lo_x = a[:, 0] - e; hi_x = a[:, 0] + e; lo_y = a[:, 1] - e; hi_y = a[:, 1] + e
    for _ in range(400):
        inside = (hi_x > x0 + tol) & (lo_x < x1 - tol) & (hi_y > y0 + tol) & (lo_y < y1 - tol)
        nx0 = min(x0, lo_x[inside].min()); nx1 = max(x1, hi_x[inside].max())
        ny0 = min(y0, lo_y[inside].min()); ny1 = max(y1, hi_y[inside].max())
        if (nx0, nx1, ny0, ny1) == (x0, x1, y0, y1): break
        x0, x1, y0, y1 = nx0, nx1, ny0, ny1
    return x0, x1, y0, y1, np.nonzero(inside)[0]


def gadgets_of(n, a, e, L, min_tilt_deg=0.5):
    t = np.degrees(np.mod(a[:, 2], np.pi / 2)); tilt = np.minimum(t, 90 - t) > min_tilt_deg
    out = []
    if tilt.sum() == 0:
        return out
    cls = clusters(a, tilt)
    seeds = [c for c in cls]
    # also unions of consecutive cluster pairs (for interlocked pairs)
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            seeds.append(np.concatenate([cls[i], cls[j]]))
    seen = set()
    for cl in seeds:
        x0 = (a[cl, 0] - e[cl]).min(); x1 = (a[cl, 0] + e[cl]).max(); y0 = (a[cl, 1] - e[cl]).min(); y1 = (a[cl, 1] + e[cl]).max()
        X0, X1, Y0, Y1, idx = clean_rect(a, e, x0, x1, y0, y1)
        W, H = X1 - X0, Y1 - Y0
        w, h = int(W + 1e-7), int(H + 1e-7)
        key = (round(X0, 4), round(Y0, 4), round(W, 4), round(H, 4))
        if key in seen: continue
        seen.add(key)
        g = len(idx) - w * h
        if g <= 0: continue
        sq = a[idx].copy(); sq[:, 0] -= X0; sq[:, 1] -= Y0
        out.append(dict(src=n, w=w, h=h, g=int(g), fx=float(W - w), fy=float(H - h), cnt=int(len(idx)),
                        whole=bool(W > L - 1e-6 and H > L - 1e-6), sq=sq.tolist()))
    return out


if __name__ == '__main__':
    D = load_all(sys.argv[1])
    lib = []
    for n in sorted(D):
        a, e, L = D[n]
        K = int(L + 1e-9)
        gs = gadgets_of(n, a, e, L)
        lib += gs
        # the whole packing as a gadget (covers pure constructions and anything not isolated)
        if n > K * K and L - K < 1 - 1e-9 and not any(g['whole'] for g in gs):
            lib.append(dict(src=n, w=K, h=K, g=n - K * K, fx=L - K, fy=L - K, cnt=n, whole=True, sq=a.tolist()))
    json.dump(lib, open(sys.argv[2], 'w'))
    sub = [g for g in lib if not g['whole']]
    print(len(lib), 'gadgets;', len(sub), 'proper sub-gadgets')
    best = {}
    for g in sub:
        for (w, h) in ((g['w'], g['h']), (g['h'], g['w'])):
            k = (min(w, h), max(w, h), g['g']); f = max(g['fx'], g['fy'])
            if k not in best or f < best[k][0]: best[k] = (f, g['src'])
    print('distinct proper (w<=h, g):', len(best))
    for k in sorted(best, key=lambda k: (k[2], k[0] * k[1])):
        print(f'  (w,h,g)={k} f={best[k][0]:.6f} from n={best[k][1]}')
